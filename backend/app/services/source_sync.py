from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config.assistant_config import AssistantConfig
from app.ingestion.pipeline import ParseError
from app.ingestion.sources import KnowledgeSource, SourceError, build_source
from app.services.embedding import EmbeddingProvider
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


@dataclass
class SourceSyncResult:
    name: str
    type: str
    status: str = "ok"  # "ok" or "failed" (discovery failed: nothing ingested or retired)
    error: str | None = None
    discovered: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    empty: int = 0
    failed: int = 0
    retired: int = 0
    chunks_embedded: int = 0
    failed_items: list[str] = field(default_factory=list)
    retired_items: list[str] = field(default_factory=list)


async def sync_source(
    source: KnowledgeSource,
    *,
    source_type: str,
    ingestion: IngestionService,
    tenant_id: str,
    assistant_id: str,
    embedder: EmbeddingProvider,
) -> SourceSyncResult:
    """discover -> resolve_access -> fetch per item, then the shared
    pipeline (chunk, embed, idempotent atomic replace), then retire whatever
    this source no longer has.

    A discovery failure stops this source before anything is written. One
    unreadable item is skipped and counted, and is not retired (it's still
    at the source). An embedding failure propagates: the sync stops, and
    nothing is retired for a source whose items weren't all processed.
    """
    result = SourceSyncResult(name=source.name, type=source_type)
    try:
        discovered = await source.discover()
    except SourceError as exc:
        logger.error("knowledge source sync failed during discovery: %s", exc)
        result.status, result.error = "failed", str(exc)
        return result

    result.discovered = len(discovered)
    for item in discovered:
        with_access = source.resolve_access(item)
        try:
            parsed = await source.fetch(with_access)
        except (ParseError, FileNotFoundError):
            logger.exception("skipping knowledge source item that failed to fetch: %s", item.uri)
            result.failed += 1
            result.failed_items.append(item.uri)
            continue
        outcome = await ingestion.ingest_parsed(
            parsed, tenant_id=tenant_id, assistant_id=assistant_id, embedder=embedder
        )
        if outcome.action == "created":
            result.created += 1
        elif outcome.action == "replaced":
            result.updated += 1
        elif outcome.action == "unchanged":
            result.unchanged += 1
        else:
            result.empty += 1
        result.chunks_embedded += len(outcome.chunks)

    retired = await ingestion.vector_store.retire_missing_documents(
        tenant_id=tenant_id,
        assistant_id=assistant_id,
        uri_prefix=source.uri_prefix,
        keep_uris={item.uri for item in discovered},
    )
    for uri in retired:
        logger.info("retired document no longer at its source: %s", uri)
    result.retired, result.retired_items = len(retired), retired
    return result


async def sync_knowledge_sources(
    config: AssistantConfig,
    *,
    tenant_id: str,
    ingestion: IngestionService,
    embedder: EmbeddingProvider,
    base_dir: Path,
    only: str | None = None,
) -> list[SourceSyncResult]:
    """Syncs an assistant's configured sources (or just the one named
    ``only``) into that assistant's index for ``tenant_id``."""
    results: list[SourceSyncResult] = []
    for source_config in config.knowledge_sources:
        if only is not None and source_config.name != only:
            continue
        source = build_source(source_config, base_dir=base_dir)
        results.append(
            await sync_source(
                source,
                source_type=source_config.type,
                ingestion=ingestion,
                tenant_id=tenant_id,
                assistant_id=config.assistant_id,
                embedder=embedder,
            )
        )
    return results
