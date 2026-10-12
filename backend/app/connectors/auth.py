"""How the backend authenticates to a connector's system.

Every strategy is a service credential the backend holds: an API key, HTTP
Basic, a static bearer token, or an OAuth token from the client-credentials
or JWT-bearer grant (cached until shortly before expiry, refreshed on demand,
dropped after a 401). The user's identity travels separately, as a
server-injected parameter (see runtime.py).

Extension point for on-behalf-of / token exchange: implement ConnectorAuth.
``apply`` already receives the Principal, so an implementation can exchange
the caller's token (RFC 8693, or Entra's OBO flow) for one scoped to that user
and cache it per principal. Two things it would need that don't exist yet:
the inbound access token itself (PrincipalContext carries validated claims,
not the raw token), and an ``auth.method`` in app/config/connectors.py
(``on_behalf_of`` is reserved there and currently rejected).
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import httpx
import jwt
from dotenv import dotenv_values

from app.config.connectors import (
    ApiKeyAuthConfig,
    BasicAuthConfig,
    BearerAuthConfig,
    ClientCredentialsAuthConfig,
    ConnectorConfig,
    JwtBearerAuthConfig,
)
from app.config.knowledge_sources import env_var_name
from app.core.principal import PrincipalContext
from app.observability.redaction import register_secret

# Refresh this long before the issuer's expiry, so a token never expires
# between being read from the cache and reaching the API.
_EXPIRY_SKEW_SECONDS = 30.0


class ConnectorCallError(Exception):
    """A connector call that failed in a known way. ``kind`` is a fixed
    category (it is what the audit row records); ``message`` is fixed text,
    never upstream response content or credentials."""

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


@dataclass(frozen=True)
class AuthContext:
    client: httpx.AsyncClient
    principal: PrincipalContext
    # Network check (SSRF) for any URL the strategy itself calls.
    guard: Callable[[str], Awaitable[None]]


class ConnectorAuth(Protocol):
    async def apply(self, headers: dict[str, str], ctx: AuthContext) -> None:
        """Adds credentials to an outgoing request's headers."""

    def invalidate(self) -> None:
        """Drops any cached credential (called once after a 401)."""


class ApiKeyAuth:
    def __init__(self, config: ApiKeyAuthConfig) -> None:
        self._config = config

    async def apply(self, headers: dict[str, str], ctx: AuthContext) -> None:
        headers[self._config.header] = f"{self._config.prefix}{resolve_secret(self._config.key)}"

    def invalidate(self) -> None:
        return None


class BasicAuth:
    def __init__(self, config: BasicAuthConfig) -> None:
        self._config = config

    async def apply(self, headers: dict[str, str], ctx: AuthContext) -> None:
        user = resolve_secret(self._config.username)
        password = resolve_secret(self._config.password)
        encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
        register_secret(encoded)
        headers["Authorization"] = f"Basic {encoded}"

    def invalidate(self) -> None:
        return None


class BearerAuth:
    def __init__(self, config: BearerAuthConfig) -> None:
        self._config = config

    async def apply(self, headers: dict[str, str], ctx: AuthContext) -> None:
        headers["Authorization"] = f"Bearer {resolve_secret(self._config.token)}"

    def invalidate(self) -> None:
        return None


class _CachedTokenAuth:
    """Shared caching for OAuth grants. One instance per distinct credential
    set, shared across turns (see auth_for)."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()
        self.tokens_issued = 0

    async def apply(self, headers: dict[str, str], ctx: AuthContext) -> None:
        headers["Authorization"] = f"Bearer {await self._access_token(ctx)}"

    def invalidate(self) -> None:
        self._token, self._expires_at = None, 0.0

    async def _access_token(self, ctx: AuthContext) -> str:
        async with self._lock:
            if self._token is not None and time.monotonic() < self._expires_at:
                return self._token
            token_url, data, auth = self._token_request()
            await ctx.guard(token_url)
            try:
                response = await ctx.client.post(token_url, data=data, auth=auth)
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
                expires_in = float(body.get("expires_in") or self._default_ttl())
            except (ValueError, KeyError, TypeError) as exc:
                raise ConnectorCallError(
                    "bad_response", "the token endpoint returned an unexpected response"
                ) from exc
            register_secret(token)
            self._token = token
            self._expires_at = time.monotonic() + max(0.0, expires_in - _EXPIRY_SKEW_SECONDS)
            self.tokens_issued += 1
            return token

    def _default_ttl(self) -> float:
        return 300.0

    def _token_request(self) -> tuple[str, dict[str, str], Any]:
        raise NotImplementedError


class ClientCredentialsAuth(_CachedTokenAuth):
    def __init__(self, config: ClientCredentialsAuthConfig) -> None:
        super().__init__()
        self._config = config

    def _token_request(self) -> tuple[str, dict[str, str], Any]:
        config = self._config
        client_id = resolve_secret(config.client_id)
        client_secret = resolve_secret(config.client_secret)
        data = {**config.extra_params, "grant_type": "client_credentials"}
        if config.scope:
            data["scope"] = config.scope
        if config.client_auth == "post":
            data.update(client_id=client_id, client_secret=client_secret)
            return config.token_url, data, None
        return config.token_url, data, (client_id, client_secret)


class JwtBearerAuth(_CachedTokenAuth):
    GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"

    def __init__(self, config: JwtBearerAuthConfig) -> None:
        super().__init__()
        self._config = config

    def _default_ttl(self) -> float:
        return float(self._config.default_token_ttl_seconds)

    def _token_request(self) -> tuple[str, dict[str, str], Any]:
        config = self._config
        now = int(time.time())
        claims = {
            "iss": resolve_secret(config.client_id),
            "sub": resolve_secret(config.subject),
            "aud": config.audience,
            "iat": now,
            "exp": now + config.assertion_ttl_seconds,
            "jti": uuid.uuid4().hex,
        }
        try:
            assertion = jwt.encode(claims, resolve_secret(config.private_key), config.algorithm)
        except (ValueError, TypeError, jwt.PyJWTError) as exc:
            raise ConnectorCallError(
                "not_configured", "the connector's signing key could not be used"
            ) from exc
        register_secret(assertion)
        data = {**config.extra_params, "grant_type": self.GRANT_TYPE, "assertion": assertion}
        if config.scope:
            data["scope"] = config.scope
        return config.token_url, data, None


# Process-wide for token grants, keyed by the full auth config (not the
# assistant), because configs are re-read on every request and a cached token
# must outlive that.
_auth_cache: dict[tuple[str, str], ConnectorAuth] = {}


def auth_for(connector: ConnectorConfig) -> ConnectorAuth:
    config = connector.auth
    if isinstance(config, ApiKeyAuthConfig):
        return ApiKeyAuth(config)
    if isinstance(config, BasicAuthConfig):
        return BasicAuth(config)
    if isinstance(config, BearerAuthConfig):
        return BearerAuth(config)
    key = (connector.name, config.model_dump_json())
    auth = _auth_cache.get(key)
    if auth is None:
        if isinstance(config, ClientCredentialsAuthConfig):
            auth = ClientCredentialsAuth(config)
        else:
            auth = JwtBearerAuth(config)
        _auth_cache[key] = auth
    return auth


def reset_auth_cache() -> None:
    """Tests only: forget cached tokens."""
    _auth_cache.clear()
