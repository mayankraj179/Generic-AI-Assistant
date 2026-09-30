"""DB-backed tests for PgVectorStore.fetch_cited_chunks: the re-check that
decides whether an earlier turn's cited chunks may be reused for a chart
follow-up. Each test uses its own random tenant id, so rows left behind by
earlier runs in the dev database can never collide with these."""

from __future__ import annotations

import uuid

import pytest

from app.core.principal import PrincipalContext
from app.db.models import EMBEDDING_DIM
from app.services.embedding import HashEmbeddingProvider
from app.services.ingestion_service import IngestionService
from app.services.retriever import RetrievalService


def _principal(tenant_id: str, *labels: str) -> PrincipalContext:
    return PrincipalContext(tenant_id=tenant_id, principal_id="person-1", labels=frozenset(labels))


async def _ingest(service: IngestionService, path, *, tenant_id: str, labels: set[str]):
    await service.ingest_file(
        source_uri=str(path),
        tenant_id=tenant_id,
        assistant_id="finance_assistant",
        access_labels=frozenset(labels),
    )


@pytest.fixture
def service() -> IngestionService:
    return IngestionService(embedder=HashEmbeddingProvider(dim=EMBEDDING_DIM))


@pytest.mark.asyncio
async def test_authorized_citation_is_refetched_with_its_text(tmp_path, service):
    tenant = f"refetch-{uuid.uuid4()}"
    report = tmp_path / "revenue_report.txt"
    report.write_text("FY2021 42.3, FY2025 96.5.", encoding="utf-8")
    await _ingest(service, report, tenant_id=tenant, labels={"dept:finance"})

    chunks = await service.vector_store.fetch_cited_chunks(
        tenant_id=tenant,
        assistant_id="finance_assistant",
        principal_labels=frozenset({"dept:finance"}),
        citations=[("revenue_report", 0)],
    )

    assert [(c.document_title, c.chunk_index) for c in chunks] == [("revenue_report", 0)]
    assert "96.5" in chunks[0].display_text


@pytest.mark.asyncio
async def test_citation_is_dropped_for_a_principal_without_a_matching_label(tmp_path, service):
    tenant = f"refetch-{uuid.uuid4()}"
    report = tmp_path / "revenue_report.txt"
    report.write_text("FY2021 42.3, FY2025 96.5.", encoding="utf-8")
    await _ingest(service, report, tenant_id=tenant, labels={"dept:finance"})

    chunks = await service.vector_store.fetch_cited_chunks(
        tenant_id=tenant,
        assistant_id="finance_assistant",
        principal_labels=frozenset({"dept:sales"}),
        citations=[("revenue_report", 0)],
    )

    assert chunks == []


@pytest.mark.asyncio
async def test_access_revoked_between_turns_is_honoured(tmp_path, service):
    # Turn 1 could read it; the document's labels then change (re-ingest with
    # a new ACL), so the same principal must no longer get it back.
    tenant = f"refetch-{uuid.uuid4()}"
    report = tmp_path / "revenue_report.txt"
    report.write_text("FY2021 42.3, FY2025 96.5.", encoding="utf-8")
    await _ingest(service, report, tenant_id=tenant, labels={"dept:finance"})
    await _ingest(service, report, tenant_id=tenant, labels={"role:executive"})

    chunks = await service.vector_store.fetch_cited_chunks(
        tenant_id=tenant,
        assistant_id="finance_assistant",
        principal_labels=frozenset({"dept:finance"}),
        citations=[("revenue_report", 0)],
    )

    assert chunks == []


@pytest.mark.asyncio
async def test_other_tenant_and_other_assistant_are_never_returned(tmp_path, service):
    tenant = f"refetch-{uuid.uuid4()}"
    report = tmp_path / "revenue_report.txt"
    report.write_text("FY2021 42.3, FY2025 96.5.", encoding="utf-8")
    await _ingest(service, report, tenant_id=tenant, labels={"dept:finance"})

    other_tenant = await service.vector_store.fetch_cited_chunks(
        tenant_id=f"refetch-{uuid.uuid4()}",
        assistant_id="finance_assistant",
        principal_labels=frozenset({"dept:finance"}),
        citations=[("revenue_report", 0)],
    )
    other_assistant = await service.vector_store.fetch_cited_chunks(
        tenant_id=tenant,
        assistant_id="hr_assistant",
        principal_labels=frozenset({"dept:finance"}),
        citations=[("revenue_report", 0)],
    )

    assert other_tenant == []
    assert other_assistant == []


@pytest.mark.asyncio
async def test_ambiguous_title_and_unknown_citations_are_dropped(tmp_path, service):
    tenant = f"refetch-{uuid.uuid4()}"
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        same_name = tmp_path / folder / "revenue_report.txt"
        same_name.write_text(f"Figures from folder {folder}.", encoding="utf-8")
        await _ingest(service, same_name, tenant_id=tenant, labels={"dept:finance"})

    chunks = await service.vector_store.fetch_cited_chunks(
        tenant_id=tenant,
        assistant_id="finance_assistant",
        principal_labels=frozenset({"dept:finance"}),
        citations=[("revenue_report", 0), ("no_such_document", 0), ("revenue_report", 7)],
    )

    assert chunks == []


@pytest.mark.asyncio
async def test_retrieval_service_refuses_a_zero_label_principal(tmp_path, service):
    tenant = f"refetch-{uuid.uuid4()}"
    report = tmp_path / "revenue_report.txt"
    report.write_text("FY2021 42.3, FY2025 96.5.", encoding="utf-8")
    await _ingest(service, report, tenant_id=tenant, labels={"dept:finance"})
    retrieval = RetrievalService(
        embedder=HashEmbeddingProvider(dim=EMBEDDING_DIM), vector_store=service.vector_store
    )

    chunks = await retrieval.fetch_cited_chunks(
        citations=[("revenue_report", 0)],
        principal=_principal(tenant),
        assistant_id="finance_assistant",
    )

    assert chunks == []
