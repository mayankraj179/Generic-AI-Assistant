from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from app.config.settings import Settings
from app.db.database import create_session_factory
from app.ingestion.pipeline import (
    Chunk,
    DiscoveredSource,
    ParsedDocument,
    ParseError,
    SourceWithAccess,
    capture_access,
    chunk,
    discover,
    parse,
    run_pipeline,
)
from app.services.embedding import EmbeddingProvider
from app.services.gemini_embedding import GeminiEmbeddingProvider
from app.services.vector_store import PgVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestOutcome:
    """What ingest_parsed() did with one document: ``created`` and
    ``replaced`` embedded ``chunks``; ``unchanged`` (content, labels and
    embedding model all match what's stored) and ``empty`` embedded nothing."""

    action: Literal["created", "replaced", "unchanged", "empty"]
    chunks: list[Chunk]


class IngestionService:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider | None = None,
        vector_store: PgVectorStore | None = None,
    ) -> None:
        self.embedder = embedder or GeminiEmbeddingProvider(api_key=Settings().gemini_api_key)
        self.vector_store = vector_store or PgVectorStore(session_factory=create_session_factory())

    async def ingest_file(
        self,
        *,
        source_uri: str,
        tenant_id: str,
        assistant_id: str,
        access_labels: frozenset[str] | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> list[Chunk]:
        """Ingests one file. Raises FileNotFoundError/ParseError directly —
        a single explicit ingest_file() call should fail loudly, unlike
        ingest_directory()'s per-file tolerance below.

        Three-way behavior against what's already stored at this
        (tenant_id, assistant_id, source_uri):
          - nothing stored yet -> insert a new document/generation
          - stored with the same content_hash AND access_labels -> skip
            entirely, no re-parsing cost already paid, no embedding call
            made, and the return is `[]` (feedback is via the log line,
            not the return value — see the module's PR summary for why
            this wasn't turned into a new result type)
          - stored but content_hash or access_labels differ -> atomic
            two-phase replace (PgVectorStore.store_document)
        """
        discovered = DiscoveredSource(uri=source_uri, source_type="filesystem")
        return await self._ingest_source(
            discovered,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            access_labels=access_labels,
            embedder=embedder,
        )

    async def ingest_directory(
        self,
        *,
        source_root: str,
        tenant_id: str,
        assistant_id: str,
        embedder: EmbeddingProvider | None = None,
    ) -> list[Chunk]:
        """Ingests every supported file under source_root. Unlike
        ingest_file(), a single bad file here is logged and skipped rather
        than aborting the whole batch — one corrupt PDF in a directory of
        fifty good ones shouldn't block the other forty-nine.
        """
        all_chunks: list[Chunk] = []
        for discovered in discover(source_root):
            try:
                chunks = await self._ingest_source(
                    discovered, tenant_id=tenant_id, assistant_id=assistant_id, embedder=embedder
                )
            except (ParseError, FileNotFoundError):
                logger.exception("skipping file that failed to ingest: %s", discovered.uri)
                continue
            all_chunks.extend(chunks)
        return all_chunks

    def pipeline(self, source_root: str) -> list[Chunk]:
        return run_pipeline(source_root)

    async def _ingest_source(
        self,
        discovered: DiscoveredSource,
        *,
        tenant_id: str,
        assistant_id: str,
        access_labels: frozenset[str] | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> list[Chunk]:
        with_access = (
            capture_access(discovered)
            if access_labels is None
            else SourceWithAccess(source=discovered, access_labels=frozenset(access_labels))
        )

        parsed = parse(with_access)
        outcome = await self.ingest_parsed(
            parsed, tenant_id=tenant_id, assistant_id=assistant_id, embedder=embedder
        )
        return outcome.chunks

    async def ingest_parsed(
        self,
        parsed: ParsedDocument,
        *,
        tenant_id: str,
        assistant_id: str,
        embedder: EmbeddingProvider | None = None,
    ) -> IngestOutcome:
        """Everything after parsing, shared by every source: the
        skip-if-unchanged fingerprint check, chunking, embedding and the
        atomic generation replace. Keyed by parsed.source.source.uri."""
        with_access = parsed.source
        discovered = with_access.source
        active_embedder = embedder or self.embedder
        existing = await self.vector_store.get_document_fingerprint(
            tenant_id=tenant_id, assistant_id=assistant_id, source_uri=discovered.uri
        )
        # The embedding model is part of the fingerprint: stored chunks from a
        # different model are invisible to search (see PgVectorStore.search),
        # so an otherwise-unchanged document must be re-embedded, not skipped.
        if existing is not None and existing == (
            parsed.content_hash,
            with_access.access_labels,
            active_embedder.model_id,
        ):
            logger.info(
                "ingestion skipped (content, access labels and embedding model unchanged): %s",
                discovered.uri,
            )
            return IngestOutcome("unchanged", [])

        chunked = chunk(parsed)
        if not chunked:
            logger.info("ingestion produced no chunks (empty document): %s", discovered.uri)
            return IngestOutcome("empty", [])

        embeddings = await active_embedder.embed_documents([item.embedded_text for item in chunked])
        await self.vector_store.store_document(
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            source_uri=discovered.uri,
            title=parsed.title,
            access_labels=with_access.access_labels,
            content_hash=parsed.content_hash,
            chunks=chunked,
            embeddings=embeddings,
            embedding_model=active_embedder.model_id,
        )
        action = "created new document" if existing is None else "replaced generation for"
        logger.info("ingestion %s: %s (%d chunks)", action, discovered.uri, len(chunked))
        return IngestOutcome("created" if existing is None else "replaced", chunked)
