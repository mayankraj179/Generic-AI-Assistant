"""Chunks from different embedding models (and dimensions) sharing one table.

Since alembic 0005, chunks.embedding is dimensionless and every row carries
embedding_model; PgVectorStore.search only compares within the query's own
model. The DB-backed tests use HashEmbeddingProvider at two widths as stand-ins
for two real models (e.g. 3072-dim Gemini vs 2048-dim Nemotron).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.config.assistant_config import AssistantConfig
from app.config.settings import Settings
from app.core.principal import PrincipalContext
from app.db.models import ChunkRecord, DocumentRecord
from app.services.embedding import EmbeddingProviderError, HashEmbeddingProvider
from app.services.embedding_provider_factory import get_embedding_provider
from app.services.ingestion_service import IngestionService
from app.services.openrouter_embedding import OpenRouterEmbeddingProvider
from app.services.retriever import RetrievalService


async def _chunk_rows(vector_store, *, tenant_id: str, source_uri: str) -> list[ChunkRecord]:
    async with vector_store._session_factory() as session:
        result = await session.execute(
            select(ChunkRecord)
            .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
            .where(DocumentRecord.tenant_id == tenant_id, DocumentRecord.source_uri == source_uri)
        )
        return list(result.scalars().all())


def _principal(tenant_id: str) -> PrincipalContext:
    return PrincipalContext(
        tenant_id=tenant_id, principal_id="p1", labels=frozenset({"role:employee"})
    )


@pytest.mark.asyncio
async def test_different_dimension_chunks_coexist_and_search_stays_within_one_model(tmp_path):
    wide, narrow = HashEmbeddingProvider(dim=3072), HashEmbeddingProvider(dim=2048)
    tenant_id, assistant_id = f"tenant-mixed-{uuid.uuid4()}", "hr_assistant"
    wide_file, narrow_file = tmp_path / "wide.txt", tmp_path / "narrow.txt"
    wide_file.write_text("Fictional stipend policy embedded by the wide model.", encoding="utf-8")
    narrow_file.write_text(
        "Fictional stipend policy embedded by the narrow model.", encoding="utf-8"
    )

    service = IngestionService(embedder=wide)
    for path, embedder in ((wide_file, wide), (narrow_file, narrow)):
        await service.ingest_file(
            source_uri=f"file://{path.as_posix()}?labels=role:employee",
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            embedder=embedder,
        )

    # Without the embedding_model filter this query would hit rows of both
    # widths and Postgres would raise "different vector dimensions".
    for embedder, expected in ((wide, "wide model"), (narrow, "narrow model")):
        retriever = RetrievalService(embedder=embedder, vector_store=service.vector_store)
        results = await retriever.search(
            query="fictional stipend policy",
            principal=_principal(tenant_id),
            assistant_id=assistant_id,
            top_k=10,
        )
        assert results
        assert all(expected in result.display_text for result in results)


@pytest.mark.asyncio
async def test_embedding_model_change_reembeds_instead_of_skipping(tmp_path):
    old_model, new_model = HashEmbeddingProvider(dim=3072), HashEmbeddingProvider(dim=2048)
    tenant_id, assistant_id = f"tenant-model-switch-{uuid.uuid4()}", "hr_assistant"
    policy_file = tmp_path / "unchanged.txt"
    policy_file.write_text("Identical fictional text across both ingests.", encoding="utf-8")
    source_uri = f"file://{policy_file.as_posix()}?labels=role:employee"

    service = IngestionService(embedder=old_model)
    await service.ingest_file(source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id)

    # Same content and labels — only the model differs. Skipping here would
    # leave the document with no chunks visible to the new model's queries.
    reingested = await service.ingest_file(
        source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id, embedder=new_model
    )
    assert reingested

    rows = await _chunk_rows(service.vector_store, tenant_id=tenant_id, source_uri=source_uri)
    assert {row.embedding_model for row in rows if row.is_current} == {new_model.model_id}
    assert {row.embedding_model for row in rows if not row.is_current} == {old_model.model_id}

    # And a second ingest with the new model is a no-op again.
    assert (
        await service.ingest_file(
            source_uri=source_uri,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            embedder=new_model,
        )
        == []
    )


def _config(**retrieval) -> AssistantConfig:
    return AssistantConfig(
        assistant_id="mixed_test",
        display_name="Mixed",
        description="test",
        tenant_id="tenant-a",
        model={"provider": "openrouter", "model_name": "x"},
        retrieval={"collection_name": "c", **retrieval},
        system_prompt="test",
    )


def test_factory_passes_embedding_model_through():
    embedder = get_embedding_provider(
        config=_config(
            embedding_provider="openrouter", embedding_model="nvidia/nemotron-3-embed-1b:free"
        ),
        settings=Settings(openrouter_api_key="k"),
    )
    assert embedder.model_id == "openrouter:nvidia/nemotron-3-embed-1b:free"
    assert embedder.dim == 2048


def test_factory_default_models_are_unchanged():
    settings = Settings(gemini_api_key="k", openrouter_api_key="k")
    gemini = get_embedding_provider(config=_config(), settings=settings)
    openrouter = get_embedding_provider(
        config=_config(embedding_provider="openrouter"), settings=settings
    )
    assert (gemini.model_id, gemini.dim) == ("gemini:gemini-embedding-001", 3072)
    assert (openrouter.model_id, openrouter.dim) == ("openrouter:google/gemini-embedding-001", 3072)


def test_unknown_openrouter_model_dimension_fails_loudly():
    with pytest.raises(EmbeddingProviderError, match="unknown output dimension"):
        OpenRouterEmbeddingProvider(api_key="k", model_name="some/unverified-model")
