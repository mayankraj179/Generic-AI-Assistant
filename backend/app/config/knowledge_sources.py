"""knowledge_sources: the per-assistant config block that says where an
assistant's knowledge comes from, so onboarding a data source is YAML only.

Three source types, all read-only: ``filesystem`` (local/uploaded
documents), ``object_storage`` (S3-compatible: bucket + prefix) and
``database`` (reviewed, parameterised SELECTs; one document per row).

Secrets never appear in YAML. Every credential field takes only an env var
reference, ``${NAME}``, resolved when a sync runs (so a config still loads
on a machine that hasn't set them). A literal value in a credential field,
or a password embedded in a URL, fails config loading.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import BeforeValidator

ENV_REF_RE = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+:[A-Za-z0-9_.-]+$")
_URL_USERINFO_RE = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@]*@", re.IGNORECASE)
_SQL_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

SOURCE_TYPES = ("database", "filesystem", "object_storage")


def _env_ref(value: Any) -> str:
    """A credential field: only ``${NAME}`` is accepted, never the value."""
    if not isinstance(value, str) or not ENV_REF_RE.fullmatch(value.strip()):
        raise ValueError(
            "literal secrets are not allowed here; use an env var reference "
            "such as ${MY_SECRET} and set the value in the environment"
        )
    return value.strip()


EnvRef = Annotated[str, BeforeValidator(_env_ref)]


def env_var_name(ref: str) -> str:
    match = ENV_REF_RE.fullmatch(ref)
    assert match is not None  # guaranteed by EnvRef validation
    return match.group(1)


def _no_url_credentials(value: str | None) -> str | None:
    if value is not None and _URL_USERINFO_RE.match(value):
        raise ValueError(
            "credentials must not be embedded in a URL; "
            "put them in env vars and reference them with ${NAME}"
        )
    return value


class _SourceBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Unique within the assistant. Part of every document's source_uri
    # (<type>://<name>/...), which is also what scopes a re-sync's retiring
    # of vanished content to this one source.
    name: str
    access_labels: frozenset[str]

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v):
            raise ValueError(f"source name '{v}' must be lowercase letters, digits, '-' or '_'")
        return v

    @field_validator("access_labels")
    @classmethod
    def _valid_labels(cls, v: frozenset[str]) -> frozenset[str]:
        # Explicit and non-empty: a chunk with no labels is readable by no one
        # (search requires a shared label), so an empty set is always a mistake.
        if not v:
            raise ValueError("access_labels must list at least one label")
        bad = sorted(label for label in v if not _LABEL_RE.fullmatch(label))
        if bad:
            raise ValueError(f"access_labels must look like 'namespace:value': {bad}")
        return v


class FilesystemSourceConfig(_SourceBase):
    type: Literal["filesystem"]
    # A directory (searched recursively for supported files) or one file.
    # Relative paths resolve against the backend directory.
    path: str


class ObjectStorageSourceConfig(_SourceBase):
    type: Literal["object_storage"]
    bucket: str
    prefix: str = ""
    # None means AWS's own endpoint; set it for MinIO or another S3-compatible store.
    endpoint_url: str | None = None
    region: str = "us-east-1"
    access_key_id: EnvRef
    secret_access_key: EnvRef
    # More objects than this fails the sync instead of silently ingesting a
    # partial listing (which would also retire everything past the cutoff).
    max_objects: int = Field(default=1000, gt=0, le=100_000)

    @field_validator("endpoint_url")
    @classmethod
    def _endpoint_has_no_credentials(cls, v: str | None) -> str | None:
        return _no_url_credentials(v)


class DatabaseQueryConfig(BaseModel):
    """One reviewed query. The SQL is written and reviewed here, in YAML,
    and bound parameters are the only way values get into it; nothing is
    ever generated or string-formatted at runtime."""

    model_config = ConfigDict(extra="forbid")

    name: str
    sql: str
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    # Column whose value identifies the row: it becomes part of the row's
    # source_uri and title, so it must be unique and non-null.
    key_column: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _NAME_RE.fullmatch(v):
            raise ValueError(f"query name '{v}' must be lowercase letters, digits, '-' or '_'")
        return v

    @field_validator("sql")
    @classmethod
    def _single_select(cls, v: str) -> str:
        sql = v.strip().rstrip(";").strip()
        if not _SQL_START_RE.match(sql):
            raise ValueError("query sql must be a single SELECT (or WITH ... SELECT) statement")
        if ";" in sql:
            raise ValueError("query sql must be a single statement (no ';' separators)")
        return sql


class DatabaseSourceConfig(_SourceBase):
    type: Literal["database"]
    # The full connection URL (it carries the password), e.g.
    # postgresql+asyncpg://kb_reader:...@host/db. Use a read-only role;
    # each sync also runs in a READ ONLY transaction.
    url: EnvRef
    queries: list[DatabaseQueryConfig] = Field(min_length=1)
    # Per query: more rows than this fails the sync rather than truncating.
    max_rows: int = Field(default=5000, gt=0, le=100_000)
    statement_timeout_ms: int = Field(default=15_000, gt=0, le=300_000)

    @field_validator("queries")
    @classmethod
    def _unique_query_names(cls, v: list[DatabaseQueryConfig]) -> list[DatabaseQueryConfig]:
        names = [q.name for q in v]
        if len(names) != len(set(names)):
            raise ValueError("query names must be unique within a database source")
        return v


KnowledgeSourceConfig = Annotated[
    FilesystemSourceConfig | ObjectStorageSourceConfig | DatabaseSourceConfig,
    Field(discriminator="type"),
]


def check_source_types(value: Any) -> Any:
    """Before-validator for AssistantConfig.knowledge_sources: pydantic's own
    discriminator errors are terse, so name a missing or unknown type."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            source_type = item.get("type")
            if source_type is None:
                raise ValueError(
                    f"knowledge_sources[{index}] has no 'type' "
                    f"(expected one of: {', '.join(SOURCE_TYPES)})"
                )
            if source_type not in SOURCE_TYPES:
                raise ValueError(
                    f"knowledge_sources[{index}] has unknown type '{source_type}' "
                    f"(expected one of: {', '.join(SOURCE_TYPES)})"
                )
    return value


def check_unique_source_names(sources: list[Any]) -> list[Any]:
    names = [source.name for source in sources]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate knowledge source name(s): {', '.join(duplicates)}")
    return sources
