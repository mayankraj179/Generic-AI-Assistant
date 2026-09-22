from __future__ import annotations

import pytest

from app.core.principal import PrincipalContext
from app.db.models import EMBEDDING_DIM
from app.services.embedding import HashEmbeddingProvider
from app.services.ingestion_service import IngestionService
from app.services.retriever import RetrievalService


@pytest.mark.asyncio
async def test_document_ingestion_and_vector_search_round_trip(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    leave_policy = docs_root / "leave_policy.txt"
    leave_policy.write_text(
        "Leave policy explains employee vacation entitlement and the approval process.",
        encoding="utf-8",
    )

    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=provider)
    source_uri = f"file://{leave_policy.as_posix()}?labels=role:employee&labels=dept:hr"
    chunks = await service.ingest_file(
        source_uri=source_uri,
        tenant_id="tenant-a",
        assistant_id="hr_assistant",
    )

    assert len(chunks) >= 1
    vector_store = service.vector_store
    assert await vector_store.count_chunks(tenant_id="tenant-a", assistant_id="hr_assistant") >= 1

    retriever = RetrievalService(embedder=provider, vector_store=vector_store)
    results = await retriever.search(
        query="What is the leave policy?",
        principal=PrincipalContext(
            tenant_id="tenant-a",
            principal_id="person-1",
            labels=frozenset({"role:employee", "dept:hr"}),
        ),
        assistant_id="hr_assistant",
        top_k=5,
    )

    assert results
    assert any("leave" in result.display_text.lower() for result in results)


@pytest.mark.asyncio
async def test_acl_filtering_excludes_unauthorized_content(tmp_path):
    docs_root = tmp_path / "documents"
    docs_root.mkdir()
    employee_policy = docs_root / "employee_policy.txt"
    employee_policy.write_text(
        "Employee policy says basic vacation is fifteen days per year.",
        encoding="utf-8",
    )
    payroll_policy = docs_root / "payroll_policy.txt"
    payroll_policy.write_text(
        "Payroll policy reveals executive compensation and confidential salary data.",
        encoding="utf-8",
    )

    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=provider)
    await service.ingest_file(
        source_uri=f"file://{employee_policy.as_posix()}?labels=role:employee",
        tenant_id="tenant-a",
        assistant_id="hr_assistant",
    )
    await service.ingest_file(
        source_uri=f"file://{payroll_policy.as_posix()}?labels=role:hr_admin",
        tenant_id="tenant-a",
        assistant_id="hr_assistant",
    )

    retriever = RetrievalService(embedder=provider, vector_store=service.vector_store)
    results = await retriever.search(
        query="What is the employee policy?",
        principal=PrincipalContext(
            tenant_id="tenant-a",
            principal_id="person-2",
            labels=frozenset({"role:employee"}),
        ),
        assistant_id="hr_assistant",
        top_k=10,
    )

    assert all("hr_admin" not in label for result in results for label in result.access_labels)
    assert any("employee" in label for result in results for label in result.access_labels)
