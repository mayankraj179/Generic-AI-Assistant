from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache

from dotenv import dotenv_values

from app.config.knowledge_sources import env_var_name
from app.ingestion.pipeline import DiscoveredSource, ParsedDocument, SourceWithAccess


class SourceError(Exception):
    """A whole knowledge source can't be synced: an unset env var, an
    unreachable endpoint, a failing query, a listing over its cap. The sync
    reports it against that source and retires nothing from it."""


class KnowledgeSource(ABC):
    """The DocumentSource shape: discover what's there, resolve each item's
    access labels before anything is parsed, then fetch it as text.

    Every adapter is read-only against its system. Each document's
    source_uri is ``<type>://<source name>/<locator>``, so the prefix
    ``<type>://<source name>/`` covers exactly this source's documents.
    """

    source_type: str

    def __init__(self, name: str, access_labels: frozenset[str]) -> None:
        self.name = name
        self.access_labels = access_labels

    @property
    def uri_prefix(self) -> str:
        return f"{self.source_type}://{self.name}/"

    def uri_for(self, locator: str) -> str:
        return f"{self.uri_prefix}{locator}"

    def locator_of(self, uri: str) -> str:
        if not uri.startswith(self.uri_prefix):
            raise FileNotFoundError(f"not a document of source '{self.name}': {uri}")
        return uri[len(self.uri_prefix) :]

    @abstractmethod
    async def discover(self) -> list[DiscoveredSource]:
        """The complete current listing. Raises SourceError rather than
        returning a partial one: anything missing from it gets retired."""

    def resolve_access(self, source: DiscoveredSource) -> SourceWithAccess:
        """Labels come from the source's reviewed config only, never from
        document content or path segments."""
        return SourceWithAccess(source=source, access_labels=self.access_labels)

    @abstractmethod
    async def fetch(self, source: SourceWithAccess) -> ParsedDocument:
        """Read one discovered item as text. Raises FileNotFoundError or
        ParseError for that item alone."""


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str | None]:
    # Same file Settings reads (env_file=".env", relative to the working
    # directory), so a value set there works for sources too.
    return dotenv_values(".env")


def resolve_env_ref(ref: str, *, field: str, source: str) -> str:
    """Resolves a validated ``${NAME}`` reference at sync time."""
    name = env_var_name(ref)
    value = os.environ.get(name) or _dotenv().get(name)
    if not value:
        raise SourceError(f"source '{source}': env var {name} (for {field}) is not set")
    return value
