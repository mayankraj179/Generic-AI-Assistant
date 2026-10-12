"""How the backend authenticates to a connector's system.

Today: a service credential the backend holds (an API key, or an OAuth
client-credentials token, cached and refreshed). The user's identity travels
separately, as a server-injected parameter (see runtime.py).

Extension point for on-behalf-of / token exchange: implement ConnectorAuth.
``apply`` already receives the Principal, so an implementation can exchange
the caller's token (RFC 8693) for one scoped to that user and cache it per
principal. Two things it would need that don't exist yet: the inbound access
token itself (PrincipalContext carries validated claims, not the raw token),
and an ``auth.method`` in app/config/connectors.py (``on_behalf_of`` is
reserved there and currently rejected).
"""

from __future__ import annotations

import asyncio
import os
import time
from functools import lru_cache
from typing import Protocol

import httpx
from dotenv import dotenv_values

from app.config.connectors import (
    ApiKeyAuthConfig,
    ClientCredentialsAuthConfig,
    ConnectorConfig,
)
from app.config.knowledge_sources import env_var_name
from app.core.principal import PrincipalContext
from app.observability.redaction import register_secret

# Refresh this long before the issuer's expiry, so a token never expires
# between being read from the cache and reaching the API.
_EXPIRY_SKEW_SECONDS = 30.0


class ConnectorCallError(Exception):
    """A connector call that failed in a known way. ``kind`` is a fixed
    category (it is what the audit row records); ``message`` is fixed text
    safe to show the model. Neither ever contains upstream response content
    or credentials."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str | None]:
    # Same file Settings reads, so a value set there works for connectors too.
    return dotenv_values(".env")


def resolve_secret(ref: str) -> str:
    name = env_var_name(ref)
    value = os.environ.get(name) or _dotenv().get(name)
    if not value:
        raise ConnectorCallError("not_configured", "the connector's credentials are not configured")
    register_secret(value)
    return value


class ConnectorAuth(Protocol):
    async def apply(
        self, headers: dict[str, str], *, client: httpx.AsyncClient, principal: PrincipalContext
    ) -> None:
        """Adds credentials to an outgoing request's headers."""

    def invalidate(self) -> None:
        """Drops any cached credential (called once after a 401)."""


class ApiKeyAuth:
    def __init__(self, config: ApiKeyAuthConfig) -> None:
        self._config = config

    async def apply(
        self, headers: dict[str, str], *, client: httpx.AsyncClient, principal: PrincipalContext
    ) -> None:
        headers[self._config.header] = resolve_secret(self._config.key)

    def invalidate(self) -> None:
        return None


class ClientCredentialsAuth:
    """Fetches a token with the client-credentials grant and reuses it until
    shortly before expiry. One instance per distinct credential set, shared
    across turns (see auth_for), so turns don't each fetch a new token."""

    def __init__(self, config: ClientCredentialsAuthConfig) -> None:
        self._config = config
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()
        self.tokens_issued = 0

    async def apply(
        self, headers: dict[str, str], *, client: httpx.AsyncClient, principal: PrincipalContext
    ) -> None:
        headers["Authorization"] = f"Bearer {await self._access_token(client)}"

    def invalidate(self) -> None:
        self._token, self._expires_at = None, 0.0

    async def _access_token(self, client: httpx.AsyncClient) -> str:
        async with self._lock:
            if self._token is not None and time.monotonic() < self._expires_at:
                return self._token
            client_id = resolve_secret(self._config.client_id)
            client_secret = resolve_secret(self._config.client_secret)
            data = {"grant_type": "client_credentials"}
            if self._config.scope:
                data["scope"] = self._config.scope
            try:
                response = await client.post(
                    self._config.token_url, data=data, auth=(client_id, client_secret)
                )
            except httpx.TimeoutException as exc:
                raise ConnectorCallError("timeout", "the token endpoint timed out") from exc
            except httpx.HTTPError as exc:
                raise ConnectorCallError(
                    "unavailable", "the token endpoint could not be reached"
                ) from exc
            if response.status_code in (400, 401, 403):
                raise ConnectorCallError(
                    "auth_failed", "the system rejected the service credential"
                )
            if response.status_code != 200:
                raise ConnectorCallError("unavailable", "the token endpoint returned an error")
            try:
                body = response.json()
                token = str(body["access_token"])
                expires_in = float(body.get("expires_in", 300))
            except (ValueError, KeyError, TypeError) as exc:
                raise ConnectorCallError(
                    "bad_response", "the token endpoint returned an unexpected response"
                ) from exc
            register_secret(token)
            self._token = token
            self._expires_at = time.monotonic() + max(0.0, expires_in - _EXPIRY_SKEW_SECONDS)
            self.tokens_issued += 1
            return token


# Process-wide, keyed by the credential set (not the assistant), because
# configs are re-read on every request and a cached token must outlive that.
_auth_cache: dict[tuple, ConnectorAuth] = {}


def auth_for(connector: ConnectorConfig) -> ConnectorAuth:
    config = connector.auth
    if isinstance(config, ApiKeyAuthConfig):
        return ApiKeyAuth(config)
    key = (connector.name, config.token_url, config.client_id, config.client_secret, config.scope)
    auth = _auth_cache.get(key)
    if auth is None:
        auth = _auth_cache[key] = ClientCredentialsAuth(config)
    return auth


def reset_auth_cache() -> None:
    """Tests only: forget cached tokens."""
    _auth_cache.clear()
