"""Knowledge-source adapters: one per AssistantConfig.knowledge_sources type."""

from __future__ import annotations

from pathlib import Path

from app.config.knowledge_sources import (
    DatabaseSourceConfig,
    FilesystemSourceConfig,
    ObjectStorageSourceConfig,
)
from app.ingestion.sources.base import KnowledgeSource, SourceError
from app.ingestion.sources.database import DatabaseSource
from app.ingestion.sources.filesystem import FilesystemSource
from app.ingestion.sources.object_storage import ObjectStorageSource

__all__ = [
    "DatabaseSource",
    "FilesystemSource",
    "KnowledgeSource",
    "ObjectStorageSource",
    "SourceError",
    "build_source",
]


def build_source(
    config: FilesystemSourceConfig | ObjectStorageSourceConfig | DatabaseSourceConfig,
    *,
    base_dir: Path,
) -> KnowledgeSource:
    if isinstance(config, FilesystemSourceConfig):
        return FilesystemSource(config, base_dir=base_dir)
    if isinstance(config, ObjectStorageSourceConfig):
        return ObjectStorageSource(config)
    return DatabaseSource(config)
