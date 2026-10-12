"""Declarative connector operations.

An operation is data, not code: method, path template, parameters, which
response fields to keep, pagination, timeout, payload cap and labels. The same
schema describes an operation written inline in an assistant config
(``type: rest``) and one shipped in a preset (app/connectors/presets/*.yaml).

Read-only is enforced here, at load time: the method must be GET, or POST on
an operation that declares ``read_only_post: true`` (search/query endpoints
that take their query in a body). Anything else fails config loading.

What the model controls is exactly the non-identity ``parameters``. It never
supplies a URL, host, path or header: path parameters are percent-encoded into
the configured template, query and body parameters are values only, and
identity parameters are filled server-side from the validated Principal.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TOOL_NAME_SEPARATOR = "__"
MAX_TOOL_NAME_LENGTH = 64
ALLOWED_METHODS = ("GET", "POST")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PARAM_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_.$-]*$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+:[A-Za-z0-9_.-]+$")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")
# Static headers are for things like Accept or ConsistencyLevel. Anything that
# looks like a credential belongs in ``auth`` (env var references only).
_FORBIDDEN_HEADER_RE = re.compile(
    r"(?i)^(host|cookie|proxy-.*)$|auth|token|secret|key|session|password|signature"
)

Scalar = str | int | float | bool


def tool_name(connector_name: str, operation_name: str) -> str:
    return f"{connector_name}{TOOL_NAME_SEPARATOR}{operation_name}"


def check_labels(labels: frozenset[str], *, field: str, required: bool) -> frozenset[str]:
    if required and not labels:
        raise ValueError(f"{field} must list at least one label")
    bad = sorted(label for label in labels if not _LABEL_RE.fullmatch(label))
    if bad:
        raise ValueError(f"{field} must look like 'namespace:value': {bad}")
    return labels


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", hide_input_in_errors=True, frozen=True, populate_by_name=True
    )


class ParamSpec(_Strict):
    """One input. ``identity: true`` means the value is the caller's id in the
    external system, taken from a validated JWT claim (the connector's
    ``identity.claim``, or ``claim`` here). Identity params are never shown to
    the model and anything the model sends for them is ignored."""

    type: Literal["string", "integer", "number", "boolean"] = "string"
    location: Literal["path", "query", "body"] = Field(alias="in")
    description: str = ""
    enum: list[str | int | float] | None = None
    identity: bool = False
    claim: str | None = None
    max_length: int = Field(default=200, gt=0, le=2000)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ParamSpec:
        if self.claim is not None and not self.identity:
            raise ValueError("'claim' is only valid on an identity parameter")
        if self.identity and self.type != "string":
            raise ValueError("identity parameters must be strings")
        if self.identity and self.enum is not None:
            raise ValueError("identity parameters cannot have an enum")
        return self

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.type == "string":
            schema["maxLength"] = self.max_length
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema


class ResponseSpec(_Strict):
    """Which part of the response reaches the model. ``root`` is a dotted
    path to the records (a list, or a single object); ``fields`` are dotted
    paths kept from each record (empty keeps everything). Data minimisation:
    list only what an answer needs."""

    root: str | None = None
    fields: list[str] = Field(default_factory=list)
    max_items: int = Field(default=25, gt=0, le=500)


class PaginationSpec(_Strict):
    """How to fetch further pages, bounded by ``max_pages`` and
    ``response.max_items``. Next-page links (``link_header``,
    ``odata_next_link``) are followed only when they point at the connector's
    own base URL; offset and cursor values go in the query for GET and in the
    body for a read_only_post operation."""

    strategy: Literal["none", "offset", "cursor", "link_header", "odata_next_link"] = "none"
    max_pages: int = Field(default=3, ge=1, le=20)
    page_size: int | None = Field(default=None, gt=0, le=1000)
    page_size_param: str | None = None
    offset_param: str = "offset"
    cursor_param: str = "cursor"
    cursor_path: str | None = None
    # OData v4: "@odata.nextLink"; OData v2 (e.g. SAP Gateway): "d.__next".
    next_link_path: str = "@odata.nextLink"

    @model_validator(mode="after")
    def _consistent(self) -> PaginationSpec:
        if self.strategy == "cursor" and not self.cursor_path:
            raise ValueError("cursor pagination needs cursor_path (where the next cursor is)")
        if self.page_size_param and self.page_size is None:
            raise ValueError("page_size_param needs page_size")
        return self

    def control_params(self) -> set[str]:
        names = {self.page_size_param} if self.page_size_param else set()
        if self.strategy == "offset":
            names.add(self.offset_param)
        if self.strategy == "cursor":
            names.add(self.cursor_param)
        return names


class OperationSpec(_Strict):
    name: str
    description: str = Field(min_length=1)
    method: str = "GET"
    read_only_post: bool = False
    path: str
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    fixed_query: dict[str, Scalar] = Field(default_factory=dict)
    fixed_body: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    response: ResponseSpec = Field(default_factory=ResponseSpec)
    pagination: PaginationSpec = Field(default_factory=PaginationSpec)
    timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    max_response_bytes: int = Field(default=256_000, gt=0, le=2_000_000)
    # In a preset these are a baseline the assistant config adds to; inline
    # (type: rest) they are the operation's labels and must not be empty.
    required_labels: frozenset[str] = Field(default_factory=frozenset)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v):
            raise ValueError(
                f"operation name '{v}' must start with a letter and use only "
                "lowercase letters, digits or '_'"
            )
        return v

    @field_validator("method")
    @classmethod
    def _read_only_method(cls, v: str) -> str:
        method = v.upper()
        if method not in ALLOWED_METHODS:
            raise ValueError(
                f"method '{v}' is not allowed: connectors are read-only (GET, or POST "
                "on an operation that declares read_only_post: true)"
            )
        return method

    @field_validator("required_labels")
    @classmethod
    def _valid_labels(cls, v: frozenset[str]) -> frozenset[str]:
        return check_labels(v, field="required_labels", required=False)

    @field_validator("headers")
    @classmethod
    def _no_credential_headers(cls, v: dict[str, str]) -> dict[str, str]:
        bad = sorted(name for name in v if _FORBIDDEN_HEADER_RE.search(name))
        if bad:
            raise ValueError(
                f"headers {bad} are not allowed here; credentials go in auth (env var "
                "references only)"
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> OperationSpec:
        if self.method == "POST" and not self.read_only_post:
            raise ValueError(
                f"operation '{self.name}': POST is only allowed with read_only_post: true "
                "(for search/query endpoints that do not change anything)"
            )
        if self.method == "GET" and self.read_only_post:
            raise ValueError(f"operation '{self.name}': read_only_post is only valid with POST")
        self._check_path()
        for pname, param in self.parameters.items():
            if not _PARAM_RE.fullmatch(pname):
                raise ValueError(f"operation '{self.name}': invalid parameter name '{pname}'")
            if param.location == "body" and self.method != "POST":
                raise ValueError(
                    f"operation '{self.name}': body parameter '{pname}' needs a POST operation"
                )
        if self.fixed_body and self.method != "POST":
            raise ValueError(f"operation '{self.name}': fixed_body needs a POST operation")
        unknown_required = sorted(set(self.required) - self.parameters.keys())
        if unknown_required:
            raise ValueError(
                f"operation '{self.name}': required names unknown parameters: {unknown_required}"
            )
        reserved = set(self.fixed_query) | set(self.fixed_body) | self.pagination.control_params()
        clash = sorted(reserved & self.parameters.keys())
        if clash:
            raise ValueError(
                f"operation '{self.name}': parameters {clash} clash with fixed values or "
                "pagination parameters"
            )
        if self.pagination.strategy != "none" and not self.response.root:
            raise ValueError(
                f"operation '{self.name}': pagination needs response.root (the record list)"
            )
        if self.pagination.strategy in ("link_header", "odata_next_link") and self.method != "GET":
            raise ValueError(
                f"operation '{self.name}': {self.pagination.strategy} pagination is GET-only"
            )
        return self

    def _check_path(self) -> None:
        path = self.path
        if (
            not path.startswith("/")
            or "//" in path
            or "://" in path
            or "?" in path
            or "#" in path
            or "\\" in path
        ):
            raise ValueError(
                f"operation '{self.name}': path must be a relative path starting with '/', "
                "with no scheme, host, query or fragment"
            )
        if any(segment in ("..", ".") for segment in path.split("/")):
            raise ValueError(f"operation '{self.name}': path must not contain '.' or '..'")
        placeholders = set(_PLACEHOLDER_RE.findall(path))
        path_params = {n for n, p in self.parameters.items() if p.location == "path"}
        if placeholders != path_params:
            raise ValueError(
                f"operation '{self.name}': path placeholders {sorted(placeholders)} must match "
                f"the 'in: path' parameters {sorted(path_params)}"
            )
        for name in path_params:
            if name not in self.required and not self.parameters[name].identity:
                raise ValueError(
                    f"operation '{self.name}': path parameter '{name}' must be required"
                )

    def identity_params(self) -> dict[str, ParamSpec]:
        return {n: p for n, p in self.parameters.items() if p.identity}

    def model_input_schema(self) -> dict[str, Any]:
        """The JSON Schema the model sees: identity parameters removed."""
        visible = {n: p for n, p in self.parameters.items() if not p.identity}
        return {
            "type": "object",
            "properties": {n: p.json_schema() for n, p in visible.items()},
            "required": [n for n in self.required if n in visible],
            "additionalProperties": False,
        }


class PresetSpec(_Strict):
    """A bundled connector type: reviewed operations plus the auth methods
    the system supports. An assistant enables a preset's operations by name."""

    type: str
    description: str
    auth_methods: list[str] = Field(min_length=1)
    operations: list[OperationSpec] = Field(min_length=1)

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v) or v == "rest":
            raise ValueError(f"preset type '{v}' is invalid or reserved")
        return v

    def operation(self, name: str) -> OperationSpec | None:
        return next((op for op in self.operations if op.name == name), None)
