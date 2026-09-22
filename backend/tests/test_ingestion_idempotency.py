from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.principal import PrincipalContext
from app.db.models import EMBEDDING_DIM, ChunkRecord, DocumentRecord
from app.services.embedding import HashEmbeddingProvider
from app.services.ingestion_service import IngestionService
from app.services.retriever import RetrievalService


async def _fetch_chunk_rows(
    vector_store, *, tenant_id: str, assistant_id: str, source_uri: str
) -> list[ChunkRecord]:
    """White-box helper: reads every chunk row (current and superseded) for
    one document directly, so tests can assert on is_current/generation_id
    without going through the ACL-filtered, is_current-only search() path.
    """
    async with vector_store._session_factory() as session:
        result = await session.execute(
            select(ChunkRecord)
            .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
            .where(
                DocumentRecord.tenant_id == tenant_id,
                DocumentRecord.assistant_id == assistant_id,
                DocumentRecord.source_uri == source_uri,
            )
        )
        return list(result.scalars().all())


@pytest.mark.asyncio
async def test_unchanged_reingest_is_a_noop(tmp_path, caplog):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    policy_file = docs_root / "noop_policy.txt"
    policy_file.write_text("Stable policy content that never changes.", encoding="utf-8")

    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=provider)
    tenant_id, assistant_id = "tenant-idempotency-noop", "hr_assistant"
    source_uri = f"file://{policy_file.as_posix()}"

    first_chunks = await service.ingest_file(
        source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id
    )
    assert first_chunks

    count_after_first = await service.vector_store.count_chunks(
        tenant_id=tenant_id, assistant_id=assistant_id
    )

    with caplog.at_level("INFO"):
        second_chunks = await service.ingest_file(
            source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id
        )

    assert second_chunks == []
    count_after_second = await service.vector_store.count_chunks(
        tenant_id=tenant_id, assistant_id=assistant_id
    )
    assert count_after_second == count_after_first

    rows = await _fetch_chunk_rows(
        service.vector_store, tenant_id=tenant_id, assistant_id=assistant_id, source_uri=source_uri
    )
    assert len(rows) == len(first_chunks)  # no duplicate rows inserted

    assert any(
        "skipped" in record.message and source_uri in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_changed_content_creates_new_generation_and_retires_old(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    policy_file = docs_root / "changing_policy.txt"
    policy_file.write_text("Original fictional policy text version one.", encoding="utf-8")

    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=provider)
    tenant_id, assistant_id = "tenant-idempotency-replace", "hr_assistant"
    source_uri = f"file://{policy_file.as_posix()}?labels=role:employee"

    await service.ingest_file(
        source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id
    )

    policy_file.write_text(
        "Completely different fictional policy text version two.", encoding="utf-8"
    )
    new_chunks = await service.ingest_file(
        source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id
    )
    assert new_chunks

    rows = await _fetch_chunk_rows(
        service.vector_store, tenant_id=tenant_id, assistant_id=assistant_id, source_uri=source_uri
    )
    old_rows = [row for row in rows if not row.is_current]
    current_rows = [row for row in rows if row.is_current]

    assert old_rows  # old generation still physically present
    assert all("version one" in row.display_text for row in old_rows)
    assert current_rows
    assert all("version two" in row.display_text for row in current_rows)
    assert len({row.generation_id for row in old_rows}) == 1
    assert len({row.generation_id for row in current_rows}) == 1
    assert old_rows[0].generation_id != current_rows[0].generation_id

    retriever = RetrievalService(embedder=provider, vector_store=service.vector_store)
    principal = PrincipalContext(
        tenant_id=tenant_id, principal_id="p1", labels=frozenset({"role:employee"})
    )
    results = await retriever.search(
        query="fictional policy text", principal=principal, assistant_id=assistant_id, top_k=10
    )
    assert results
    assert all("version one" not in result.display_text for result in results)
    assert any("version two" in result.display_text for result in results)


@pytest.mark.asyncio
async def test_access_labels_only_change_forces_new_generation(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    policy_file = docs_root / "label_only_change.txt"
    policy_file.write_text("Identical text that never changes across re-ingests.", encoding="utf-8")

    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=provider)
    tenant_id, assistant_id = "tenant-idempotency-labels", "hr_assistant"
    source_uri = f"file://{policy_file.as_posix()}"

    await service.ingest_file(
        source_uri=source_uri,
        tenant_id=tenant_id,
        assistant_id=assistant_id,
        access_labels=frozenset({"role:employee"}),
    )

    # Same file, same text (same content_hash) -> only access_labels differ.
    new_chunks = await service.ingest_file(
        source_uri=source_uri,
        tenant_id=tenant_id,
        assistant_id=assistant_id,
        access_labels=frozenset({"role:hr_admin"}),
    )
    assert new_chunks  # not treated as a no-op skip, despite identical content

    rows = await _fetch_chunk_rows(
        service.vector_store, tenant_id=tenant_id, assistant_id=assistant_id, source_uri=source_uri
    )
    old_rows = [row for row in rows if not row.is_current]
    current_rows = [row for row in rows if row.is_current]

    assert old_rows
    assert current_rows
    assert old_rows[0].generation_id != current_rows[0].generation_id
    assert all(row.access_labels == ["role:employee"] for row in old_rows)
    assert all(row.access_labels == ["role:hr_admin"] for row in current_rows)


@pytest.mark.asyncio
async def test_batch_ingest_directory_skips_one_corrupt_file_without_aborting(tmp_path):
    docs_root = tmp_path / "batch"
    docs_root.mkdir()
    (docs_root / "good_one.txt").write_text("Good document one content.", encoding="utf-8")
    (docs_root / "good_two.txt").write_text("Good document two content.", encoding="utf-8")
    (docs_root / "corrupt.pdf").write_bytes(b"not a real pdf, just garbage bytes")

    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=provider)
    tenant_id, assistant_id = "tenant-idempotency-batch", "hr_assistant"

    chunks = await service.ingest_directory(
        source_root=str(docs_root), tenant_id=tenant_id, assistant_id=assistant_id
    )

    titles = {c.document_title for c in chunks}
    assert titles == {"good_one", "good_two"}


@pytest.mark.asyncio
async def test_atomic_replace_leaves_old_generation_intact_on_commit_failure(tmp_path, monkeypatch):
    """The atomicity guarantee (LLD §15.5): a failure between embedding and
    the commit of the replace transaction must leave the OLD generation
    fully intact and queryable — never a partial or missing document.

    Simulated by patching AsyncSession.commit to raise, scoped tightly (via
    monkeypatch, auto-reverted at test teardown) around only the second,
    replacing ingest_file() call. get_document_fingerprint()/count_chunks()/
    search() never call commit() themselves (read-only), so this doesn't
    affect anything else in this test.
    """
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    policy_file = docs_root / "atomicity_policy.txt"
    policy_file.write_text("Old generation content that must survive a crash.", encoding="utf-8")

    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=provider)
    # Unique per run (unlike the other tests here): this is the one test that
    # asserts an exact count_chunks() total, which is scoped to
    # (tenant_id, assistant_id) only, not to source_uri — a fixed tenant_id
    # would make that assertion brittle against leftover rows from a prior
    # run in this dev database (no test-cleanup convention exists in this
    # suite; see test_vector_storage_and_retrieval.py).
    tenant_id = f"tenant-idempotency-atomicity-{uuid.uuid4().hex[:8]}"
    assistant_id = "hr_assistant"
    source_uri = f"file://{policy_file.as_posix()}"

    await service.ingest_file(
        source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id
    )
    rows_before = await _fetch_chunk_rows(
        service.vector_store, tenant_id=tenant_id, assistant_id=assistant_id, source_uri=source_uri
    )
    assert rows_before
    assert all(row.is_current for row in rows_before)

    policy_file.write_text("New generation content that must never appear.", encoding="utf-8")

    class SimulatedCommitFailure(Exception):
        pass

    async def failing_commit(self) -> None:
        raise SimulatedCommitFailure("simulated failure between embedding and commit")

    monkeypatch.setattr(AsyncSession, "commit", failing_commit)
    with pytest.raises(SimulatedCommitFailure):
        await service.ingest_file(
            source_uri=source_uri, tenant_id=tenant_id, assistant_id=assistant_id
        )
    monkeypatch.undo()

    rows_after = await _fetch_chunk_rows(
        service.vector_store, tenant_id=tenant_id, assistant_id=assistant_id, source_uri=source_uri
    )
    assert len(rows_after) == len(rows_before)
    assert all(row.is_current for row in rows_after)
    assert {row.id for row in rows_after} == {row.id for row in rows_before}
    assert all("Old generation content" in row.display_text for row in rows_after)
    assert all("New generation content" not in row.display_text for row in rows_after)

    count = await service.vector_store.count_chunks(tenant_id=tenant_id, assistant_id=assistant_id)
    assert count == len(rows_before)
