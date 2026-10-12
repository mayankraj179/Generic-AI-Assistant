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
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
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
    # "template": the value is only ever substituted, escaped, into one of the
    # operation's reviewed query templates (see TemplateSpec).
    location: Literal["path", "query", "body", "template"] = Field(alias="in")
    description: str = ""
    enum: list[str | int | float] | None = None
    identity: bool = False
    claim: str | None = None
    # The value comes from the connector's ``settings`` (fixed by an admin in
    # config, e.g. a SharePoint site or list id). Hidden from the model, like
    # identity; anything the model sends for it is ignored.
    setting: str | None = None
    # Only with ``setting`` in a template: inserted unquoted, as a field name
    # (e.g. an OData column). The setting value must be a plain identifier.
    raw: bool = False
    max_length: int = Field(default=200, gt=0, le=2000)
    # A regex the whole value must match (e.g. an issue key, a case number).
    pattern: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    @field_validator("pattern")
    @classmethod
    def _valid_pattern(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v)
            except re.error as exc:
                raise ValueError(f"pattern is not a valid regex: {exc}") from None
            if not (v.startswith("^") and v.endswith("$")):
                raise ValueError("pattern must be anchored (start with ^ and end with $)")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> ParamSpec:
        if self.claim is not None and not self.identity:
            raise ValueError("'claim' is only valid on an identity parameter")
        if self.identity and self.type != "string":
            raise ValueError("identity parameters must be strings")
        if self.identity and self.enum is not None:
            raise ValueError("identity parameters cannot have an enum")
        if self.setting is not None and self.identity:
            raise ValueError("a parameter is either identity or setting, not both")
        if self.setting is not None and not IDENTIFIER_RE.fullmatch(self.setting):
            raise ValueError(f"setting name '{self.setting}' must be an identifier")
        if self.raw and (self.setting is None or self.location != "template"):
            raise ValueError("'raw' is only valid on a template parameter taken from a setting")
        return self

    @property
    def hidden(self) -> bool:
        """Filled server-side; never shown to or taken from the model."""
        return self.identity or self.setting is not None

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.type == "string":
            schema["maxLength"] = self.max_length
        if self.pattern is not None:
            schema["pattern"] = self.pattern
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema


TemplateLanguage = Literal["jql", "soql", "servicenow", "odata"]


class TemplateSpec(_Strict):
    """A reviewed query (JQL, SOQL, a ServiceNow encoded query, an OData
    $filter) sent as one query or body field. The text is fixed in config;
    ``{param}`` placeholders take ``in: template`` parameters, which are
    escaped and quoted as literals of ``language`` (see templates.py), so a
    value can never change the query's structure. The model never writes
    query text.

    ``parts`` are joined with ``separator``. A part is kept only when every
    placeholder in it has a value, so optional filters go in their own part.
    Put the user-scoping clause (with the identity placeholder) in a part of
    its own: identity always has a value, so that part is always present."""

    location: Literal["query", "body"] = Field(default="query", alias="in")
    language: TemplateLanguage
    parts: list[str] = Field(min_length=1)
    separator: str = " "

    def placeholders(self) -> set[str]:
        return {name for part in self.parts for name in _PLACEHOLDER_RE.findall(part)}


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

    # next_link: a next-page URL in the body at next_link_path (e.g.
    # Salesforce's nextRecordsUrl); odata_next_link is the same with OData's
    # default path.
    strategy: Literal["none", "offset", "cursor", "link_header", "odata_next_link", "next_link"] = (
        "none"
    )
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
        if self.strategy == "next_link" and "next_link_path" not in self.model_fields_set:
            raise ValueError("next_link pagination needs next_link_path")
        return self

    @property
    def follows_links(self) -> bool:
        return self.strategy in ("link_header", "odata_next_link", "next_link")

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
    templates: dict[str, TemplateSpec] = Field(default_factory=dict)
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
        self._check_templates()
        unknown_required = sorted(set(self.required) - self.parameters.keys())
        if unknown_required:
            raise ValueError(
                f"operation '{self.name}': required names unknown parameters: {unknown_required}"
            )
        reserved = (
            set(self.fixed_query)
            | set(self.fixed_body)
            | set(self.templates)
            | self.pagination.control_params()
        )
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
        page_size = self.pagination.page_size
        if (
            self.pagination.strategy != "none"
            and page_size
            and page_size >= self.response.max_items
        ):
            raise ValueError(
                f"operation '{self.name}': pagination.page_size ({page_size}) must be smaller "
                f"than response.max_items ({self.response.max_items}), or no second page is "
                "ever fetched"
            )
        if self.pagination.follows_links and self.method != "GET":
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
            if name not in self.required and not self.parameters[name].hidden:
                raise ValueError(
                    f"operation '{self.name}': path parameter '{name}' must be required"
                )

    def _check_templates(self) -> None:
        template_params = {n for n, p in self.parameters.items() if p.location == "template"}
        used: set[str] = set()
        for tname, template in self.templates.items():
            if template.location == "body" and self.method != "POST":
                raise ValueError(
                    f"operation '{self.name}': body template '{tname}' needs a POST operation"
                )
            unknown = sorted(template.placeholders() - template_params)
            if unknown:
                raise ValueError(
                    f"operation '{self.name}': template '{tname}' uses {unknown}, which are "
                    "not 'in: template' parameters"
                )
            if template.language == "soql":
                text = template.separator.join(template.parts)
                if not text.lstrip().upper().startswith("SELECT ") or ";" in text:
                    raise ValueError(
                        f"operation '{self.name}': SOQL template '{tname}' must be a single "
                        "SELECT statement"
                    )
            used |= template.placeholders()
        unused = sorted(template_params - used)
        if unused:
            raise ValueError(
                f"operation '{self.name}': 'in: template' parameters {unused} are not used by "
                "any template"
            )

    def identity_params(self) -> dict[str, ParamSpec]:
        return {n: p for n, p in self.parameters.items() if p.identity}

    def setting_params(self) -> dict[str, ParamSpec]:
        return {n: p for n, p in self.parameters.items() if p.setting is not None}

    def model_input_schema(self) -> dict[str, Any]:
        """The JSON Schema the model sees: identity and setting parameters
        removed."""
        visible = {n: p for n, p in self.parameters.items() if not p.hidden}
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
    # True when the system keys users by something the email default can't
    # match (e.g. an SAP user name): the config must set identity.claim, or
    # loading fails instead of every lookup silently finding nothing.
    requires_identity_claim: bool = False
    operations: list[OperationSpec] = Field(min_length=1)

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v) or v == "rest":
            raise ValueError(f"preset type '{v}' is invalid or reserved")
        return v

    def operation(self, name: str) -> OperationSpec | None:
        return next((op for op in self.operations if op.name == name), None)
