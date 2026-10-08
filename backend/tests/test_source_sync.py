"""Sync behaviour through the real pipeline and Postgres store, with an
in-memory fake source standing in for a filesystem/bucket/database: no-op
re-sync, updates, retiring vanished items, and failure handling. Plus the
/admin/ingest/sync endpoint's auth and request checks."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app.auth.provider import JwtAuthProvider
from app.config.settings import Settings
from app.core.principal import PrincipalContext
from app.db.models import EMBEDDING_DIM
from app.ingestion.pipeline import (
    DiscoveredSource,
    ParsedDocument,
    ParseError,
    SourceWithAccess,
    parsed_document,
)
from app.ingestion.sources import KnowledgeSource, SourceError
from app.main import app
from app.services.embedding import HashEmbeddingProvider
from app.services.ingestion_service import IngestionService
from app.services.retriever import RetrievalService
from app.services.source_sync import SourceSyncResult, sync_source


class _CountingEmbedder(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dim=EMBEDDING_DIM)
        self.embedded: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return await super().embed_documents(texts)


class _FakeSource(KnowledgeSource):
    source_type = "fake"

    def __init__(self, name: str = "src", labels: frozenset[str] | None = None) -> None:
        super().__init__(name, labels or frozenset({"role:authenticated"}))
        self.items: dict[str, str] = {}
        self.unreadable: set[str] = set()
        self.discovery_error: str | None = None

    async def discover(self) -> list[DiscoveredSource]:
        if self.discovery_error:
            raise SourceError(self.discovery_error)
        return [DiscoveredSource(self.uri_for(k), "fake") for k in sorted(self.items)]

    async def fetch(self, source: SourceWithAccess) -> ParsedDocument:
        locator = self.locator_of(source.source.uri)
        if locator in self.unreadable:
            raise ParseError(f"corrupt: {locator}")
        return parsed_document(source, title=locator, raw_text=self.items[locator])


@pytest.fixture
def ctx():
    embedder = _CountingEmbedder()
    tenant = f"tenant-sync-{uuid.uuid4().hex[:10]}"  # fresh per test: no leftovers
    service = IngestionService(embedder=embedder)

    async def sync(source: _FakeSource) -> SourceSyncResult:
        return await sync_source(
            source,
            source_type="fake",
            ingestion=service,
            tenant_id=tenant,
            assistant_id="kb_test",
            embedder=embedder,
        )

    async def current_chunks() -> int:
        return await service.vector_store.count_chunks(tenant_id=tenant, assistant_id="kb_test")

    async def search(text: str, labels: frozenset[str]) -> list[str]:
        retriever = RetrievalService(embedder=embedder, vector_store=service.vector_store)
        principal = PrincipalContext(tenant_id=tenant, principal_id="u", labels=labels, clearance=1)
        chunks = await retriever.search(
            # Visibility, not relevance: hash-embedding scores can be negative.
            query=text,
            assistant_id="kb_test",
            principal=principal,
            top_k=10,
            min_similarity=-1.0,
        )
        return sorted({c.document_title for c in chunks})

    class Ctx:
        pass

    c = Ctx()
    c.embedder, c.sync, c.current_chunks, c.search = embedder, sync, current_chunks, search
    return c


def _counts(result: SourceSyncResult) -> tuple[int, ...]:
    return (result.created, result.updated, result.unchanged, result.failed, result.retired)


async def test_first_sync_ingests_and_unchanged_resync_is_a_noop(ctx):
    source = _FakeSource()
    source.items = {"leave": "Annual leave is 15 days.", "travel": "Claims within 30 days."}

    first = await ctx.sync(source)
    assert (first.status, first.discovered) == ("ok", 2)
    assert _counts(first) == (2, 0, 0, 0, 0)
    chunks, embedded = await ctx.current_chunks(), len(ctx.embedder.embedded)

    second = await ctx.sync(source)
    assert _counts(second) == (0, 0, 2, 0, 0)
    assert second.chunks_embedded == 0
    assert len(ctx.embedder.embedded) == embedded  # no embedding call at all
    assert await ctx.current_chunks() == chunks


async def test_changed_item_is_replaced_and_others_skipped(ctx):
    source = _FakeSource()
    source.items = {"leave": "Annual leave is 15 days.", "travel": "Claims within 30 days."}
    await ctx.sync(source)
    ctx.embedder.embedded.clear()

    source.items["leave"] = "Annual leave is 18 days."
    result = await ctx.sync(source)
    assert _counts(result) == (0, 1, 1, 0, 0)
    assert ctx.embedder.embedded == ["leave: Annual leave is 18 days."]


async def test_removed_item_is_retired_and_returns_if_it_reappears(ctx):
    source = _FakeSource()
    source.items = {"leave": "Annual leave is 15 days.", "travel": "Claims within 30 days."}
    await ctx.sync(source)
    labels = frozenset({"role:authenticated"})
    assert await ctx.search("leave", labels) == ["leave", "travel"]

    del source.items["travel"]
    result = await ctx.sync(source)
    assert (result.retired, result.retired_items) == (1, ["fake://src/travel"])
    assert await ctx.search("travel", labels) == ["leave"]

    again = await ctx.sync(source)  # already retired: not counted twice
    assert again.retired == 0

    source.items["travel"] = "Claims within 30 days."  # same content as before
    back = await ctx.sync(source)
    assert back.updated == 1  # re-embedded, not skipped as "unchanged"
    assert await ctx.search("travel", labels) == ["leave", "travel"]


async def test_retiring_is_scoped_to_its_own_source(ctx):
    a, b = _FakeSource("a"), _FakeSource("b")
    a.items, b.items = {"one": "Alpha content."}, {"two": "Beta content."}
    await ctx.sync(a)
    await ctx.sync(b)

    a.items = {}
    result = await ctx.sync(a)
    assert result.retired_items == ["fake://a/one"]
    assert await ctx.search("content", frozenset({"role:authenticated"})) == ["two"]


async def test_discovery_failure_reports_and_retires_nothing(ctx):
    source = _FakeSource()
    source.items = {"leave": "Annual leave is 15 days."}
    await ctx.sync(source)
    chunks = await ctx.current_chunks()

    source.discovery_error = "source 'src': could not list bucket"
    result = await ctx.sync(source)
    assert (result.status, result.error) == ("failed", "source 'src': could not list bucket")
    assert _counts(result) == (0, 0, 0, 0, 0)
    assert await ctx.current_chunks() == chunks  # still searchable


async def test_unreadable_item_is_skipped_but_not_retired(ctx):
    source = _FakeSource()
    source.items = {"leave": "Annual leave is 15 days.", "travel": "Claims within 30 days."}
    await ctx.sync(source)
    chunks = await ctx.current_chunks()

    source.unreadable = {"travel"}
    source.items["leave"] = "Annual leave is 18 days."
    result = await ctx.sync(source)
    assert _counts(result) == (0, 1, 0, 1, 0)
    assert result.failed_items == ["fake://src/travel"]
    assert await ctx.current_chunks() == chunks  # travel's old generation kept


async def test_config_labels_gate_retrieval(ctx):
    source = _FakeSource(labels=frozenset({"dept:finance"}))
    source.items = {"budget": "The travel budget is 2 million."}
    await ctx.sync(source)
    assert await ctx.search("budget", frozenset({"role:authenticated"})) == []
    assert await ctx.search("budget", frozenset({"dept:finance"})) == ["budget"]


# ---------- POST /admin/ingest/sync ----------


@pytest.fixture
def client(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    settings = Settings(
        auth_enabled=True,
        auth_issuer="http://issuer",
        auth_audience="generic-ai-api",
    )
    public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    monkeypatch.setattr(
        app.state, "auth_provider", JwtAuthProvider(settings=settings, signing_key=public)
    )

    import time

    import jwt

    token = jwt.encode(
        {
            "iss": "http://issuer",
            "aud": "generic-ai-api",
            "sub": "admin-1",
            "exp": int(time.time()) + 600,
        },
        private_key,
        algorithm="RS256",
    )
    calls: list[dict] = []

    async def fake_sync(config, *, tenant_id, ingestion, embedder, base_dir, only=None):
        calls.append({"assistant": config.assistant_id, "tenant": tenant_id, "only": only})
        return [SourceSyncResult(name="it_handbook", type="filesystem", discovered=1, created=1)]

    monkeypatch.setattr("app.main.sync_knowledge_sources", fake_sync)
    monkeypatch.setattr("app.main.get_embedding_provider", lambda **_: HashEmbeddingProvider(dim=8))
    test_client = TestClient(app)
    test_client.headers["Authorization"] = f"Bearer {token}"
    test_client.calls = calls
    return test_client


def test_sync_endpoint_requires_authentication(client):
    response = client.post(
        "/admin/ingest/sync",
        json={"assistant_id": "kb_demo_assistant", "tenant_id": "t1"},
        headers={"Authorization": ""},
    )
    assert response.status_code == 401
    assert client.calls == []


def test_sync_endpoint_requires_the_admin_ingest_permission(client, monkeypatch):
    from app.auth.policy import DevelopmentAuthorizationPolicy

    monkeypatch.setattr(
        DevelopmentAuthorizationPolicy,
        "default_permissions",
        DevelopmentAuthorizationPolicy.default_permissions - {"admin:ingest"},
    )
    response = client.post(
        "/admin/ingest/sync", json={"assistant_id": "kb_demo_assistant", "tenant_id": "t1"}
    )
    assert response.status_code == 403
    assert client.calls == []


def test_sync_endpoint_syncs_the_configured_sources(client):
    response = client.post(
        "/admin/ingest/sync",
        json={"assistant_id": "kb_demo_assistant", "tenant_id": "t1", "source": "it_handbook"},
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["assistant_id"], body["tenant_id"]) == ("kb_demo_assistant", "t1")
    assert body["sources"][0]["name"] == "it_handbook"
    assert client.calls == [
        {"assistant": "kb_demo_assistant", "tenant": "t1", "only": "it_handbook"}
    ]


@pytest.mark.parametrize(
    ("payload", "status", "detail"),
    [
        ({"assistant_id": "nope", "tenant_id": "t1"}, 404, "unknown assistant_id 'nope'"),
        (
            {"assistant_id": "hr_assistant", "tenant_id": "t1"},
            422,
            "no knowledge_sources configured",
        ),
        (
            {"assistant_id": "kb_demo_assistant", "tenant_id": "t1", "source": "x"},
            422,
            "no knowledge source 'x'",
        ),
    ],
)
def test_sync_endpoint_rejects_bad_requests(client, payload, status, detail):
    response = client.post("/admin/ingest/sync", json=payload)
    assert response.status_code == status
    assert detail in response.json()["detail"]
    assert client.calls == []
