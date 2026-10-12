"""connectors: the per-assistant config block that enables live, read-only
lookups against an external system at question time (see app/connectors/).

Each connector names a known ``type`` (app/connectors/catalog.py), a
``base_url``, how the backend authenticates, which JWT claim identifies the
user to that system, and the ``operations`` this assistant may call, each
with the labels a caller must hold. An unknown type or operation fails config
loading.

Secrets follow the knowledge_sources rule: credential fields take only an env
var reference, ``${NAME}``, resolved at call time. A literal value there, an
unknown key (e.g. a stray ``api_key: abc``), or credentials in a URL fail
config loading.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config.knowledge_sources import EnvRef, _no_url_credentials
from app.connectors.base import MAX_TOOL_NAME_LENGTH, tool_name
from app.connectors.catalog import CONNECTOR_TYPES

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+:[A-Za-z0-9_.-]+$")
_HTTP_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)

AUTH_METHODS = ("api_key", "oauth_client_credentials")
# Named here so a config asking for it gets a clear error rather than a
# generic "unknown method". See ConnectorAuth in app/connectors/auth.py for
# where it would plug in.
_PLANNED_AUTH_METHODS = ("on_behalf_of", "token_exchange")


def _http_url(value: str, field: str) -> str:
    if not _HTTP_URL_RE.fullmatch(value):
        raise ValueError(f"{field} must be an http(s) URL")
    _no_url_credentials(value)
    return value.rstrip("/")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)


class ApiKeyAuthConfig(_Strict):
    method: Literal["api_key"]
    header: str = "X-API-Key"
    key: EnvRef


class ClientCredentialsAuthConfig(_Strict):
    """OAuth 2.0 client credentials (RFC 6749 section 4.4). The token is
    cached until shortly before it expires and refreshed on demand."""

    method: Literal["oauth_client_credentials"]
    token_url: str
    client_id: EnvRef
    client_secret: EnvRef
    scope: str | None = None

    @field_validator("token_url")
    @classmethod
    def _valid_token_url(cls, v: str) -> str:
        return _http_url(v, "token_url")


ConnectorAuthConfig = Annotated[
    ApiKeyAuthConfig | ClientCredentialsAuthConfig, Field(discriminator="method")
]


class ConnectorIdentityConfig(_Strict):
    """Which validated JWT claim holds the user's id in the external system.
    Its value is injected into the operation's identity params server-side."""

    claim: str = Field(min_length=1)


class ConnectorOperationConfig(_Strict):
    name: str
    # The caller must hold ALL of these (plus any the operation itself
    # declares). Required and non-empty: there is no "everyone" default.
    required_labels: frozenset[str]
    # Optional tightening of the operation's own defaults.
    timeout_seconds: float | None = Field(default=None, gt=0, le=30)
    max_response_bytes: int | None = Field(default=None, gt=0, le=1_000_000)

    @field_validator("required_labels")
    @classmethod
    def _valid_labels(cls, v: frozenset[str]) -> frozenset[str]:
        if not v:
            raise ValueError("required_labels must list at least one label")
        bad = sorted(label for label in v if not _LABEL_RE.fullmatch(label))
        if bad:
            raise ValueError(f"required_labels must look like 'namespace:value': {bad}")
        return v


class ConnectorConfig(_Strict):
    name: str
    type: str
    # Shown to the user in the source line, e.g. "Source: HR system, live".
    display_name: str = Field(min_length=1)
    base_url: str
    auth: ConnectorAuthConfig
    identity: ConnectorIdentityConfig | None = None
    operations: list[ConnectorOperationConfig] = Field(min_length=1)

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
    def _known_type_and_operations(self) -> ConnectorConfig:
        connector_type = CONNECTOR_TYPES.get(self.type)
        if connector_type is None:
            raise ValueError(
                f"connector '{self.name}' has unknown type '{self.type}' "
                f"(known types: {', '.join(sorted(CONNECTOR_TYPES))})"
            )
        names = [op.name for op in self.operations]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"connector '{self.name}' lists duplicate operations: {duplicates}")
        unknown = sorted(n for n in names if n not in connector_type.operations)
        if unknown:
            raise ValueError(
                f"connector '{self.name}' (type '{self.type}') has unknown operation(s): "
                f"{', '.join(unknown)} (known: {', '.join(sorted(connector_type.operations))})"
            )
        for op_name in names:
            if len(tool_name(self.name, op_name)) > MAX_TOOL_NAME_LENGTH:
                raise ValueError(
                    f"connector '{self.name}' operation '{op_name}': the combined tool "
                    f"name is longer than {MAX_TOOL_NAME_LENGTH} characters"
                )
            if connector_type.operations[op_name].identity_params and self.identity is None:
                raise ValueError(
                    f"connector '{self.name}' operation '{op_name}' needs the caller's "
                    "identity; set identity.claim to the JWT claim holding their id"
                )
        return self

    def tool_names(self) -> list[str]:
        return [tool_name(self.name, op.name) for op in self.operations]


def check_unique_connector_names(connectors: list[ConnectorConfig]) -> list[ConnectorConfig]:
    names = [c.name for c in connectors]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate connector name(s): {', '.join(duplicates)}")
    return connectors
