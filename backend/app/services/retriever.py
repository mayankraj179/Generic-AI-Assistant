from __future__ import annotations

from app.core.principal import PrincipalContext
from app.ingestion.pipeline import Chunk
from app.services.embedding import EmbeddingProvider
from app.services.vector_store import PgVectorStore


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
        query_embedding = await active_embedder.embed_query(query)
        chunks = await self.vector_store.search(
            query_embedding=query_embedding,
            embedding_model=active_embedder.model_id,
            tenant_id=principal.tenant_id,
            assistant_id=assistant_id,
            principal_labels=principal.labels,
            top_k=top_k,
        )

        # A relevance/business-rule filter, not an access-control predicate —
        # deliberately applied here in Python, after the ACL-filtered query,
        # rather than folded into PgVectorStore's WHERE clause. PgVectorStore
        # stays focused on its one job (authorized, distance-ordered rows);
        # "is this actually relevant enough to use" is retrieval business
        # logic, the same layer that already short-circuits on a zero-label
        # principal above. A chunk with no similarity_score (e.g. from a
        # fake vector store in tests) is never filtered — only a chunk that
        # was actually scored and scored below the bar is dropped.
        return [
            chunk
            for chunk in chunks
            if chunk.similarity_score is None or chunk.similarity_score >= min_similarity
        ]
