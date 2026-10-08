from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.database import create_session_factory
from app.db.models import ChunkRecord, DocumentRecord
from app.ingestion.pipeline import Chunk

logger = logging.getLogger(__name__)


def _authorized_chunk_predicates(
    *, tenant_id: str, assistant_id: str, principal_labels: frozenset[str]
) -> list[Any]:
    """The single definition of which chunk rows a principal may read: same
    tenant and assistant, the current generation only, and at least one
    shared access label. Used by both search() and fetch_cited_chunks() so
    the access rule can't drift between them."""
    return [
        ChunkRecord.tenant_id == tenant_id,
        ChunkRecord.assistant_id == assistant_id,
        ChunkRecord.is_current.is_(True),
        ChunkRecord.access_labels.op("&&")(list(sorted(principal_labels))),
    ]


class PgVectorStore:
    """Concrete PostgreSQL + pgvector store for document chunks."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._session_factory = session_factory or create_session_factory()

    async def get_document_fingerprint(
        self, *, tenant_id: str, assistant_id: str, source_uri: str
    ) -> tuple[str, frozenset[str], str] | None:
        """Lightweight read-only lookup used by IngestionService to decide
        whether re-ingesting a source is a no-op, without doing any
        embedding work first. Returns ``(content_hash, access_labels,
        embedding_model)`` for the existing DocumentRecord at this
        source_uri, or ``None`` if nothing has ever been ingested from it.
        Never mutates anything.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    DocumentRecord.content_hash,
                    DocumentRecord.access_labels,
                    DocumentRecord.embedding_model,
                ).where(
                    DocumentRecord.tenant_id == tenant_id,
                    DocumentRecord.assistant_id == assistant_id,
                    DocumentRecord.source_uri == source_uri,
                )
            )
            row = result.first()
            if row is None:
                return None
            content_hash, access_labels, embedding_model = row
            return content_hash, frozenset(access_labels or []), embedding_model

    async def store_document(
        self,
        *,
        tenant_id: str,
        assistant_id: str,
        source_uri: str,
        title: str,
        access_labels: frozenset[str],
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
        embedding_model: str,
    ) -> DocumentRecord:
        """Insert-or-atomically-replace: always the caller's job to have
        already decided this call is actually needed (IngestionService skips
        calling this entirely when content_hash and access_labels are both
        unchanged from what's already stored).

        If a DocumentRecord already exists at this (tenant_id, assistant_id,
        source_uri), this performs the LLD §15.5 two-phase generation
        replace in ONE transaction: the new chunk rows are inserted with a
        fresh generation_id and is_current=true, the OLD generation's chunks
        are flipped to is_current=false, and the DocumentRecord's metadata is
        updated — all committed together. If anything raises before
        session.commit(), the `async with` session context rolls back
        everything in this method, leaving the old (current) generation's
        rows completely untouched — no separate/explicit transaction
        boundary was needed beyond what AsyncSession already gives a single
        `async with` block, since there's no concurrent-writer scenario to
        guard against in this framework's current scope.

        Callers must have already parsed/chunked/embedded the NEW content in
        full before calling this — that expensive, fallible work happens
        outside this method (and outside any transaction), exactly so a
        failure there never touches anything currently visible.
        """
        async with self._session_factory() as session:
            existing = await session.execute(
                select(DocumentRecord).where(
                    DocumentRecord.tenant_id == tenant_id,
                    DocumentRecord.assistant_id == assistant_id,
                    DocumentRecord.source_uri == source_uri,
                )
            )
            document = existing.scalar_one_or_none()
            sorted_labels = list(sorted(access_labels))

            if document is None:
                document = DocumentRecord(
                    tenant_id=tenant_id,
                    assistant_id=assistant_id,
                    source_uri=source_uri,
                    title=title,
                    access_labels=sorted_labels,
                    content_hash=content_hash,
                    embedding_model=embedding_model,
                    acl_version=1,
                )
                session.add(document)
                await session.flush()
            else:
                # Retire the old generation. This never deletes anything —
                # is_current=false is what makes the old rows invisible to
                # search while keeping them physically intact until this
                # same transaction commits both changes atomically.
                await session.execute(
                    update(ChunkRecord)
                    .where(
                        ChunkRecord.document_id == document.id,
                        ChunkRecord.is_current.is_(True),
                    )
                    .values(is_current=False)
                )
                document.title = title
                document.access_labels = sorted_labels
                document.content_hash = content_hash
                document.embedding_model = embedding_model
                document.acl_version = (document.acl_version or 1) + 1

            generation_id = uuid.uuid4()
            for index, chunk_item in enumerate(chunks):
                record = ChunkRecord(
                    document_id=document.id,
                    tenant_id=tenant_id,
                    assistant_id=assistant_id,
                    chunk_index=index,
                    display_text=chunk_item.display_text,
                    embedded_text=chunk_item.embedded_text,
                    embedding=embeddings[index],
                    embedding_model=embedding_model,
                    access_labels=list(sorted(chunk_item.access_labels)),
                    acl_version=document.acl_version,
                    generation_id=generation_id,
                    is_current=True,
                )
                session.add(record)

            await session.commit()
            await session.refresh(document)
            return document

    async def retire_missing_documents(
        self,
        *,
        tenant_id: str,
        assistant_id: str,
        uri_prefix: str,
        keep_uris: set[str],
    ) -> list[str]:
        """Retires every document under ``uri_prefix`` (one knowledge
        source's namespace) whose URI isn't in ``keep_uris``: its current
        chunks become is_current=false, the same soft retire a generation
        replace uses, so nothing is deleted. Its content_hash is reset to ''
        (never a real sha256), so if it reappears at the source it is
        re-ingested rather than skipped as unchanged. Returns the URIs
        retired by this call; ones already retired aren't counted again.
        """
        async with self._session_factory() as session:
            rows = await session.execute(
                select(DocumentRecord.id, DocumentRecord.source_uri).where(
                    DocumentRecord.tenant_id == tenant_id,
                    DocumentRecord.assistant_id == assistant_id,
                    DocumentRecord.source_uri.startswith(uri_prefix, autoescape=True),
                    DocumentRecord.content_hash != "",
                )
            )
            gone = [(doc_id, uri) for doc_id, uri in rows if uri not in keep_uris]
            if not gone:
                return []
            ids = [doc_id for doc_id, _ in gone]
            await session.execute(
                update(ChunkRecord)
                .where(ChunkRecord.document_id.in_(ids), ChunkRecord.is_current.is_(True))
                .values(is_current=False)
            )
            await session.execute(
                update(DocumentRecord).where(DocumentRecord.id.in_(ids)).values(content_hash="")
            )
            await session.commit()
        return sorted(uri for _, uri in gone)

    async def count_chunks(self, *, tenant_id: str, assistant_id: str) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count()).select_from(ChunkRecord).where(
                    ChunkRecord.tenant_id == tenant_id,
                    ChunkRecord.assistant_id == assistant_id,
                    ChunkRecord.is_current.is_(True),
                )
            )
            value = result.scalar_one()
            return int(value or 0)

    async def search(
        self,
        *,
        query_embedding: list[float],
        embedding_model: str,
        tenant_id: str,
        assistant_id: str,
        principal_labels: frozenset[str],
        top_k: int = 5,
    ) -> list[Chunk]:
        if not principal_labels:
            return []

        async with self._session_factory() as session:
            # Referenced in both the SELECT list and ORDER BY so the raw
            # cosine distance is available on the result row, not just used
            # to order it and then discarded — that's what lets Chunk carry
            # a real similarity_score instead of RetrievalService only ever
            # knowing "some chunk came back."
            distance = ChunkRecord.embedding.cosine_distance(query_embedding)
            stmt = (
                select(
                    ChunkRecord,
                    DocumentRecord.title.label("document_title"),
                    distance.label("distance"),
                )
                .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
                .where(
                    *_authorized_chunk_predicates(
                        tenant_id=tenant_id,
                        assistant_id=assistant_id,
                        principal_labels=principal_labels,
                    ),
                    # Only chunks in the query's own vector space — see
                    # ChunkRecord.embedding. Chunks left over from an
                    # assistant's previous embedding model are invisible
                    # until re-ingested, rather than scored meaninglessly.
                    ChunkRecord.embedding_model == embedding_model,
                )
                .order_by(distance)
                .limit(top_k)
            )

            rows = await session.execute(stmt)
            matches: list[Chunk] = []
            for row, document_title, chunk_distance in rows:
                matches.append(
                    Chunk(
                        document_title=document_title,
                        chunk_index=row.chunk_index,
                        display_text=row.display_text,
                        embedded_text=row.embedded_text,
                        access_labels=frozenset(row.access_labels or []),
                        similarity_score=1.0 - float(chunk_distance),
                    )
                )
            return matches

    async def fetch_cited_chunks(
        self,
        *,
        tenant_id: str,
        assistant_id: str,
        principal_labels: frozenset[str],
        citations: Sequence[tuple[str, int]],
    ) -> list[Chunk]:
        """Re-resolves persisted (document_title, chunk_index) citations to
        chunks the principal may read right now, under the same access rule
        as search(). A citation is dropped when it no longer resolves (access
        revoked, document gone) and also when its title matches more than one
        readable document, since the original turn's document can't then be
        told apart. Returned in citation order, without similarity scores."""
        wanted = list(dict.fromkeys(citations))
        if not principal_labels or not wanted:
            return []

        async with self._session_factory() as session:
            stmt = (
                select(ChunkRecord, DocumentRecord.title.label("document_title"))
                .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
                .where(
                    *_authorized_chunk_predicates(
                        tenant_id=tenant_id,
                        assistant_id=assistant_id,
                        principal_labels=principal_labels,
                    ),
                    or_(
                        *(
                            and_(DocumentRecord.title == title, ChunkRecord.chunk_index == index)
                            for title, index in wanted
                        )
                    ),
                )
            )
            rows = await session.execute(stmt)

        found: dict[tuple[str, int], list[Chunk]] = {}
        for row, document_title in rows:
            found.setdefault((document_title, row.chunk_index), []).append(
                Chunk(
                    document_title=document_title,
                    chunk_index=row.chunk_index,
                    display_text=row.display_text,
                    embedded_text=row.embedded_text,
                    access_labels=frozenset(row.access_labels or []),
                )
            )

        resolved: list[Chunk] = []
        for key in wanted:
            matches = found.get(key, [])
            if len(matches) == 1:
                resolved.append(matches[0])
            elif len(matches) > 1:
                logger.warning(
                    "dropping ambiguous cited chunk %s: its title matches %d documents",
                    key,
                    len(matches),
                )
        return resolved
