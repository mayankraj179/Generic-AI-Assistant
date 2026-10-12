"""connectors: the per-assistant config block that enables live, read-only
lookups against external systems at question time (see app/connectors/).

A connector is either ``type: rest``, with its operations written inline
(app/connectors/spec.py: method, path, parameters, response fields,
pagination, timeout, payload cap, labels), or a preset type
(app/connectors/presets/*.yaml) whose reviewed operations it enables by name.
An unknown type or operation, or a method other than GET / read_only_post
POST, fails config loading.

Secrets follow the knowledge_sources rule: credential fields take only an env
var reference, ``${NAME}``, resolved at call time. A literal value there, an
unknown key (e.g. a stray ``api_key: abc``), or credentials in a URL fail
config loading.

Network rules (SSRF): every request goes to ``base_url``'s host. Plain http,
or a base_url/token_url that is a private, loopback or link-local address, is
refused unless the connector sets ``allow_private_network: true`` (for
on-prem systems). The address check is repeated on every request, after DNS.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config.knowledge_sources import EnvRef, _no_url_credentials
from app.connectors.catalog import CONNECTOR_PRESETS, REST_TYPE
from app.connectors.spec import (
    MAX_TOOL_NAME_LENGTH,
    OperationSpec,
    check_labels,
    tool_name,
)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9-]+$")

AUTH_METHODS = ("api_key", "basic", "bearer", "oauth_client_credentials", "oauth_jwt_bearer")
# Named so a config asking for them gets a clear error rather than "unknown".
# See ConnectorAuth in app/connectors/auth.py for where they would plug in.
_PLANNED_AUTH_METHODS = ("on_behalf_of", "token_exchange")
# Token-request parameters that are credentials, so may not be literal extras.
_SECRET_PARAM_RE = re.compile(r"(?i)secret|password|assertion|token|key")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)


def _http_url(value: str, field: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{field} must be an http(s) URL")
    if parts.query or parts.fragment:
        raise ValueError(f"{field} must not have a query string or fragment")
    _no_url_credentials(value)
    return value.rstrip("/")


def _is_private_literal(host: str) -> bool:
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return host.lower() in ("localhost", "localhost.localdomain")


def _check_network(url: str, field: str, allow_private_network: bool) -> None:
    if allow_private_network:
        return
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ValueError(
            f"{field} must use https (plain http is only allowed with allow_private_network: true)"
        )
    if _is_private_literal(parts.hostname or ""):
        raise ValueError(
            f"{field} points at a private or local address; set allow_private_network: true "
            "only for an on-prem system you mean to reach"
        )


# --- auth strategies ----------------------------------------------------------


class ApiKeyAuthConfig(_Strict):
    method: Literal["api_key"]
    header: str = "X-API-Key"
    key: EnvRef
    # e.g. "SSWS " for systems that want a scheme before the key.
    prefix: str = ""

    @field_validator("header")
    @classmethod
    def _valid_header(cls, v: str) -> str:
        if not _HEADER_NAME_RE.fullmatch(v):
            raise ValueError("header must be a plain HTTP header name")
        return v


class BasicAuthConfig(_Strict):
    method: Literal["basic"]
    username: EnvRef
    password: EnvRef


class BearerAuthConfig(_Strict):
    """A static token (e.g. a personal access token)."""

    method: Literal["bearer"]
    token: EnvRef


class ClientCredentialsAuthConfig(_Strict):
    """OAuth 2.0 client credentials (RFC 6749 section 4.4). The token is
    cached until shortly before it expires and refreshed on demand."""

    method: Literal["oauth_client_credentials"]
    token_url: str
    client_id: EnvRef
    client_secret: EnvRef
    scope: str | None = None
    # How the client authenticates to the token endpoint: HTTP Basic
    # (client_secret_basic) or form fields (client_secret_post).
    client_auth: Literal["basic", "post"] = "basic"
    # Non-secret extra form fields some issuers need (e.g. resource, audience).
    extra_params: dict[str, str] = Field(default_factory=dict)

    @field_validator("token_url")
    @classmethod
    def _valid_token_url(cls, v: str) -> str:
        return _http_url(v, "token_url")

    @field_validator("extra_params")
    @classmethod
    def _no_secret_extras(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_extra_params(v)


class JwtBearerAuthConfig(_Strict):
    """OAuth 2.0 JWT bearer grant (RFC 7523), e.g. Salesforce server-to-server:
    the backend signs a short-lived assertion with its private key and swaps
    it for an access token."""

    method: Literal["oauth_jwt_bearer"]
    token_url: str
    client_id: EnvRef  # the assertion's iss
    subject: EnvRef  # the assertion's sub (the integration user)
    audience: str  # the assertion's aud
    private_key: EnvRef  # PEM
    algorithm: Literal["RS256", "RS384", "RS512", "ES256", "ES384"] = "RS256"
    scope: str | None = None
    assertion_ttl_seconds: int = Field(default=180, gt=0, le=600)
    # Used when the token response has no expires_in (Salesforce omits it).
    default_token_ttl_seconds: int = Field(default=600, gt=0, le=7200)
    extra_params: dict[str, str] = Field(default_factory=dict)

    @field_validator("token_url")
    @classmethod
    def _valid_token_url(cls, v: str) -> str:
        return _http_url(v, "token_url")

    @field_validator("extra_params")
    @classmethod
    def _no_secret_extras(cls, v: dict[str, str]) -> dict[str, str]:
        return _check_extra_params(v)


def _check_extra_params(v: dict[str, str]) -> dict[str, str]:
    bad = sorted(k for k in v if _SECRET_PARAM_RE.search(k) or k in ("grant_type", "client_id"))
    if bad:
        raise ValueError(f"extra_params may not set {bad}; credentials go in env var references")
    return v


ConnectorAuthConfig = Annotated[
    ApiKeyAuthConfig
    | BasicAuthConfig
    | BearerAuthConfig
    | ClientCredentialsAuthConfig
    | JwtBearerAuthConfig,
    Field(discriminator="method"),
]


# --- operations ---------------------------------------------------------------


class OperationRef(_Strict):
    """Enables one of a preset's operations, with this assistant's labels and
    optional tighter limits."""

    name: str
    required_labels: frozenset[str]
    timeout_seconds: float | None = Field(default=None, gt=0, le=30)
    max_response_bytes: int | None = Field(default=None, gt=0, le=2_000_000)

    @field_validator("required_labels")
    @classmethod
    def _valid_labels(cls, v: frozenset[str]) -> frozenset[str]:
        return check_labels(v, field="required_labels", required=True)


DEFAULT_IDENTITY_CLAIM = "email"


class ConnectorIdentityConfig(_Strict):
    """Which validated JWT claim holds the user's id in the external system.
    Its value is injected into identity parameters server-side.

    Defaults to ``email``, a development default: production should map an
    immutable employee-id claim from the IdP (emails get reassigned and
    renamed). When the claim used is ``email``, the token must also carry
    ``email_verified: true`` or nothing is injected (see runtime.py)."""

    claim: str = Field(default=DEFAULT_IDENTITY_CLAIM, min_length=1)


class ConnectorConfig(_Strict):
    name: str
    type: str
    # Shown to the user in the source line, e.g. "Source: HR system, live".
    display_name: str = Field(min_length=1)
    base_url: str
    allow_private_network: bool = False
    auth: ConnectorAuthConfig
    identity: ConnectorIdentityConfig = Field(default_factory=ConnectorIdentityConfig)
    # type: rest -> full OperationSpec entries; a preset -> OperationRef entries.
    operations: list[OperationSpec | OperationRef] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _parse_operations(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not isinstance(data.get("operations"), list):
            return data
        model = OperationSpec if data.get("type") == REST_TYPE else OperationRef
        parsed = []
        for index, op in enumerate(data["operations"]):
            if isinstance(op, OperationSpec | OperationRef):
                parsed.append(op)
                continue
            if model is OperationRef and isinstance(op, dict) and "path" in op:
                raise ValueError(
                    f"operations[{index}]: a preset's operations can't be redefined here; "
                    "use type: rest to write your own"
                )
            parsed.append(model.model_validate(op))
        return {**data, "operations": parsed}

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v):
            raise ValueError(
                f"connector name '{v}' must start with a letter and use only "
                "lowercase letters, digits or '_'"
            )
        return v

    @field_validator("base_url")
    @classmethod
    def _valid_base_url(cls, v: str) -> str:
        return _http_url(v, "base_url")

    @field_validator("auth", mode="before")
    @classmethod
    def _known_auth_method(cls, v: Any) -> Any:
        if isinstance(v, dict):
            method = v.get("method")
            if method in _PLANNED_AUTH_METHODS:
                raise ValueError(
                    f"auth method '{method}' is not implemented yet "
                    f"(supported: {', '.join(AUTH_METHODS)})"
                )
            if method not in AUTH_METHODS:
                raise ValueError(
                    f"unknown auth method '{method}' (expected one of: {', '.join(AUTH_METHODS)})"
                )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> ConnectorConfig:
        _check_network(self.base_url, "base_url", self.allow_private_network)
        token_url = getattr(self.auth, "token_url", None)
        if token_url:
            _check_network(token_url, "auth.token_url", self.allow_private_network)

        if self.type == REST_TYPE:
            for op in self.operations:
                if not isinstance(op, OperationSpec):  # pragma: no cover - parsed above
                    raise ValueError("type: rest operations must be full definitions")
                check_labels(op.required_labels, field=f"{op.name}.required_labels", required=True)
        else:
            preset = CONNECTOR_PRESETS.get(self.type)
            if preset is None:
                known = ", ".join(sorted([REST_TYPE, *CONNECTOR_PRESETS]))
                raise ValueError(
                    f"connector '{self.name}' has unknown type '{self.type}' (known types: {known})"
                )
            if self.auth.method not in preset.auth_methods:
                raise ValueError(
                    f"connector '{self.name}': type '{self.type}' supports auth methods "
                    f"{preset.auth_methods}, not '{self.auth.method}'"
                )
            unknown = sorted(op.name for op in self.operations if preset.operation(op.name) is None)
            if unknown:
                raise ValueError(
                    f"connector '{self.name}' (type '{self.type}') has unknown operation(s): "
                    f"{', '.join(unknown)} (known: "
                    f"{', '.join(op.name for op in preset.operations)})"
                )

        names = [op.name for op in self.operations]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"connector '{self.name}' lists duplicate operations: {duplicates}")
        for op in self.resolved_operations():
            if len(tool_name(self.name, op.name)) > MAX_TOOL_NAME_LENGTH:
                raise ValueError(
                    f"connector '{self.name}' operation '{op.name}': the combined tool "
                    f"name is longer than {MAX_TOOL_NAME_LENGTH} characters"
                )
        return self

    def resolved_operations(self) -> list[OperationSpec]:
        """Every enabled operation as a full spec: inline ones as written,
        preset ones with this config's labels added and limits tightened."""
        if self.type == REST_TYPE:
            return [op for op in self.operations if isinstance(op, OperationSpec)]
        preset = CONNECTOR_PRESETS[self.type]
        resolved = []
        for ref in self.operations:
            assert isinstance(ref, OperationRef)
            base = preset.operation(ref.name)
            assert base is not None
            update: dict[str, Any] = {"required_labels": base.required_labels | ref.required_labels}
            if ref.timeout_seconds is not None:
                update["timeout_seconds"] = min(base.timeout_seconds, ref.timeout_seconds)
            if ref.max_response_bytes is not None:
                update["max_response_bytes"] = min(base.max_response_bytes, ref.max_response_bytes)
            resolved.append(base.model_copy(update=update))
        return resolved

    def tool_names(self) -> list[str]:
        return [tool_name(self.name, op.name) for op in self.operations]


def check_unique_connector_names(connectors: list[ConnectorConfig]) -> list[ConnectorConfig]:
    names = [c.name for c in connectors]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate connector name(s): {', '.join(duplicates)}")
    return connectors
