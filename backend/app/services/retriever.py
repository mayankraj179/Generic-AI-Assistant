from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from app.core.principal import PrincipalContext
from app.ingestion.pipeline import Chunk
from app.observability.audit import record_retrieval
from app.observability.tracing import set_attributes, start_span
from app.services.embedding import EmbeddingProvider
from app.services.vector_store import PgVectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, *, embedder: EmbeddingProvider, vector_store: PgVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    async def search(
        self,
        *,
        query: str,
        principal: PrincipalContext,
        assistant_id: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        embedder: EmbeddingProvider | None = None,
    ) -> list[Chunk]:
        if principal.is_zero_label:
            return []

        # Per-call override lets ChatOrchestrator resolve the calling
        # assistant's own configured embedding provider (see
        # app/services/embedding_provider_factory.py) instead of always
        # using this service's construction-time default — the query must
        # be embedded with the same provider that embedded that assistant's
        # documents, since different providers/models are not guaranteed to
        # share a vector space.
        active_embedder = embedder or self.embedder
        with start_span(
            "retrieval.search",
            assistant_id=assistant_id,
            embedding_model=active_embedder.model_id,
            top_k=top_k,
            min_similarity=min_similarity,
        ) as span:
            query_embedding = await active_embedder.embed_query(query)
            chunks = await self.vector_store.search(
                query_embedding=query_embedding,
                embedding_model=active_embedder.model_id,
                tenant_id=principal.tenant_id,
                assistant_id=assistant_id,
                principal_labels=principal.labels,
                top_k=top_k,
            )
            # A relevance/business-rule filter, not an access-control predicate:
            # applied here, after the ACL-filtered query, rather than in
            # PgVectorStore's WHERE clause. A chunk with no similarity_score
            # (e.g. from a fake store in tests) is never filtered.
            passed = [
                chunk
                for chunk in chunks
                if chunk.similarity_score is None or chunk.similarity_score >= min_similarity
            ]
            _report_search(span, active_embedder.model_id, chunks, passed, min_similarity, top_k)
            return passed

    async def fetch_cited_chunks(
        self,
        *,
        citations: Sequence[tuple[str, int]],
        principal: PrincipalContext,
        assistant_id: str,
    ) -> list[Chunk]:
        """Re-fetches chunks an earlier turn cited, re-checked against the
        principal's current access (see PgVectorStore.fetch_cited_chunks).
        No similarity filter applies: these are not a relevance search."""
        if principal.is_zero_label:
            return []
        with start_span("retrieval.refetch_cited", cited=len(citations)) as span:
            chunks = await self.vector_store.fetch_cited_chunks(
                tenant_id=principal.tenant_id,
                assistant_id=assistant_id,
                principal_labels=principal.labels,
                citations=citations,
            )
            span.set_attribute("authorized", len(chunks))
            return chunks


def _report_search(
    span: Any,
    embedding_model: str,
    candidates: list[Chunk],
    passed: list[Chunk],
    min_similarity: float,
    top_k: int,
) -> None:
    """One INFO line, span attributes and the turn audit's retrieval stats.
    The best rejected score is the number that explains a near miss."""
    scores = [c.similarity_score for c in candidates if c.similarity_score is not None]
    passed_ids = {id(c) for c in passed}
    rejected = [
        c.similarity_score
        for c in candidates
        if id(c) not in passed_ids and c.similarity_score is not None
    ]
    stats = {
        "embedding_model": embedding_model,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "candidates": len(candidates),
        "passed": len(passed),
        "top_score": round(max(scores), 4) if scores else None,
        "best_rejected": round(max(rejected), 4) if rejected else None,
    }
    set_attributes(span, {f"retrieval.{k}": v for k, v in stats.items()})
    span.set_attribute("retrieval.scores", [round(x, 4) for x in scores])
    record_retrieval(**stats)
    logger.info(
        "retrieval: %d of %d candidates >= %.3f (top %s, best rejected %s)",
        stats["passed"],
        stats["candidates"],
        min_similarity,
        stats["top_score"],
        stats["best_rejected"],
        extra={"event": "retrieval", "retrieval": stats},
    )
