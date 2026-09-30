"""Turn audit, request ids and quota windows. Orchestrator tests use a
capturing audit store, so they stay DB-free like the rest of the unit tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config.assistant_config import (
    AssistantConfig,
    GuardrailsConfig,
    ModelCapabilities,
    RetrievalConfig,
)
from app.core.principal import PrincipalContext
from app.ingestion.pipeline import Chunk
from app.observability.audit import AuditStore, TurnRecorder, record_provider_call
from app.observability.context import principal_ref
from app.observability.provider_errors import ErrorKind, classify_http
from app.observability.usage import window_start
from app.orchestration.chat_service import (
    NO_AUTHORIZED_CONTEXT_REPLY,
    ChatOrchestrator,
    ChatStreamDone,
    ChatStreamError,
)
from app.orchestration.model_provider import GroundedPrompt, ModelProviderError, ModelReply
from app.tools.builtin import execute_tool


class CapturingAuditStore:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def write(self, recorder: TurnRecorder) -> None:
        self.rows.append(recorder.as_row())


class BrokenAuditStore(AuditStore):
    def __init__(self) -> None:
        super().__init__(session_factory=self._explode)

    def _explode(self):
        raise RuntimeError("database is down")


class FakeRetrieval:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or []

    async def search(
        self, *, query, principal, assistant_id, top_k, min_similarity=0.0, embedder=None
    ):
        return self.chunks

    async def fetch_cited_chunks(self, *, citations, principal, assistant_id):
        return []


class FakeConversations:
    def __init__(self) -> None:
        self.messages: dict[uuid.UUID, list] = {}

    async def create_conversation(self, *, principal, assistant_id):
        conversation_id = uuid.uuid4()
        self.messages[conversation_id] = []
        return SimpleNamespace(id=conversation_id, assistant_id=assistant_id)

    async def get_conversation(self, *, conversation_id, principal):
        return SimpleNamespace(id=conversation_id)

    async def list_messages(self, *, conversation_id, principal):
        return list(self.messages[conversation_id])

    async def append_message(self, *, conversation_id, principal, role, content, citations=()):
        self.messages[conversation_id].append(
            SimpleNamespace(role=role, content=content, citations=tuple(citations))
        )


class ScriptedProvider:
    """Counts one model call like a real provider, optionally runs a tool,
    then answers or raises."""

    def __init__(
        self,
        *,
        text: str = "Revenue was $96.5M.",
        error: Exception | None = None,
        tool: str | None = None,
    ):
        self.text, self.error, self.tool = text, error, tool

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:
        record_provider_call(
            "model", "openrouter", "test-model", prompt_tokens=10, completion_tokens=5
        )
        if self.tool:
            await execute_tool(self.tool, {"expression": "2 + 2"})
        if self.error:
            raise self.error
        return ModelReply(text=self.text, grounded=bool(prompt.retrieved_chunks))

    async def generate_stream(self, prompt: GroundedPrompt):
        record_provider_call("model", "openrouter", "test-model")
        if self.error:
            raise self.error
        from app.orchestration.model_provider import ModelStreamEvent

        yield ModelStreamEvent(delta=self.text)
        yield ModelStreamEvent(
            is_final=True, text=self.text, grounded=bool(prompt.retrieved_chunks)
        )


def _config(**overrides) -> AssistantConfig:
    values = dict(
        assistant_id="finance_assistant",
        display_name="Finance",
        description="test",
        tenant_id="bitwise-global",
        model=ModelCapabilities(provider="openrouter", model_name="test-model"),
        retrieval=RetrievalConfig(enabled=True, collection_name="c", top_k=4, min_similarity=0.5),
        guardrails=GuardrailsConfig(),
        system_prompt="Answer from context.",
        min_clearance=0,
    )
    values.update(overrides)
    return AssistantConfig(**values)


def _chunk() -> Chunk:
    return Chunk(
        document_title="nova_horizon_fy2025",
        chunk_index=0,
        display_text="Revenue $96.5M",
        embedded_text="nova: Revenue $96.5M",
        access_labels=frozenset({"role:authenticated"}),
    )


def _principal() -> PrincipalContext:
    return PrincipalContext(
        tenant_id="local-development",
        principal_id="real-user-42",
        labels=frozenset({"role:authenticated"}),
    )


def _orchestrator(provider, *, chunks=None, audit_store=None) -> ChatOrchestrator:
    return ChatOrchestrator(
        retrieval_service=FakeRetrieval(chunks),
        provider_factory=lambda config: provider,
        conversation_store=FakeConversations(),
        audit_store=audit_store,
    )


@pytest.mark.asyncio
async def test_grounded_turn_writes_an_audit_row_with_hashed_principal_and_call_counts():
    store = CapturingAuditStore()
    orchestrator = _orchestrator(ScriptedProvider(), chunks=[_chunk()], audit_store=store)

    result = await orchestrator.handle(principal=_principal(), config=_config(), message="revenue?")

    (row,) = store.rows
    assert row["operation"] == "chat"
    assert row["outcome"] == "answered"
    assert row["grounded"] is True
    assert row["citations_count"] == 1
    assert row["provider"] == "openrouter" and row["model_name"] == "test-model"
    assert row["conversation_id"] == result.conversation_id
    assert row["principal_ref"] == principal_ref("local-development", "real-user-42")
    assert "real-user-42" not in str(row)
    assert row["provider_calls"] == [
        {
            "kind": "model",
            "provider": "openrouter",
            "model": "test-model",
            "count": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
    ]
    assert row["error_kind"] is None and row["error"] is None


@pytest.mark.asyncio
async def test_no_context_turn_is_audited_as_no_context():
    store = CapturingAuditStore()
    orchestrator = _orchestrator(ScriptedProvider(), chunks=[], audit_store=store)

    result = await orchestrator.handle(principal=_principal(), config=_config(), message="weather?")

    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY
    assert store.rows[0]["outcome"] == "no_context"
    assert store.rows[0]["provider_calls"] == []


@pytest.mark.asyncio
async def test_provider_failure_is_audited_with_its_classification_and_logged_once(caplog):
    failure = classify_http(
        provider="openrouter",
        status=429,
        operation="chat",
        model="test-model",
        body={"error": {"message": "Rate limit exceeded: free-models-per-day.", "code": 429}},
    )
    store = CapturingAuditStore()
    orchestrator = _orchestrator(
        ScriptedProvider(error=ModelProviderError("OpenRouter returned an error", failure=failure)),
        chunks=[_chunk()],
        audit_store=store,
    )

    with caplog.at_level("INFO"), pytest.raises(ModelProviderError):
        await orchestrator.handle(principal=_principal(), config=_config(), message="revenue?")

    row = store.rows[0]
    assert row["outcome"] == "error"
    assert row["error_kind"] == ErrorKind.QUOTA_EXCEEDED
    assert row["error"]["quota"] == "free-models-per-day"
    failure_lines = [r for r in caplog.records if getattr(r, "event", None) == "provider_failure"]
    assert len(failure_lines) == 1
    assert failure_lines[0].levelname == "WARNING"
    assert failure_lines[0].exc_info is None  # no traceback at WARNING


@pytest.mark.asyncio
async def test_stream_provider_failure_becomes_an_error_event_and_is_audited():
    failure = classify_http(
        provider="openrouter",
        status=503,
        operation="chat",
        model="test-model",
        body={"error": {"message": "overloaded", "code": 503}},
    )
    store = CapturingAuditStore()
    orchestrator = _orchestrator(
        ScriptedProvider(error=ModelProviderError("down", failure=failure)),
        chunks=[_chunk()],
        audit_store=store,
    )

    events = [
        e
        async for e in orchestrator.handle_stream(
            principal=_principal(), config=_config(), message="revenue?"
        )
    ]

    assert isinstance(events[-1], ChatStreamError)
    assert store.rows[0]["operation"] == "chat_stream"
    assert store.rows[0]["error_kind"] == ErrorKind.OVERLOADED


@pytest.mark.asyncio
async def test_successful_stream_is_audited_as_answered():
    store = CapturingAuditStore()
    orchestrator = _orchestrator(ScriptedProvider(), chunks=[_chunk()], audit_store=store)

    events = [
        e
        async for e in orchestrator.handle_stream(
            principal=_principal(), config=_config(), message="revenue?"
        )
    ]

    assert isinstance(events[-1], ChatStreamDone)
    assert store.rows[0]["outcome"] == "answered"
    assert store.rows[0]["citations_count"] == 1


@pytest.mark.asyncio
async def test_tool_calls_and_guardrail_downgrades_are_recorded():
    store = CapturingAuditStore()
    orchestrator = _orchestrator(
        ScriptedProvider(text="I cannot answer that from the documents.", tool="calculate"),
        chunks=[_chunk()],
        audit_store=store,
    )

    await orchestrator.handle(
        principal=_principal(), config=_config(enabled_tools=["calculate"]), message="sum?"
    )

    row = store.rows[0]
    assert row["tool_calls"] == [{"name": "calculate", "ok": True}]
    assert row["guardrail_actions"] == [
        {"action": "citation_downgrade", "reason": "reply_appears_to_refuse"}
    ]
    assert row["grounded"] is False


@pytest.mark.asyncio
async def test_a_failing_audit_store_never_fails_the_turn():
    orchestrator = _orchestrator(
        ScriptedProvider(), chunks=[_chunk()], audit_store=BrokenAuditStore()
    )

    result = await orchestrator.handle(principal=_principal(), config=_config(), message="revenue?")

    assert result.text == "Revenue was $96.5M."


def test_request_id_is_generated_and_a_well_formed_one_is_kept():
    from app.main import app

    client = TestClient(app)
    generated = client.get("/health").headers["X-Request-ID"]
    kept = client.get("/health", headers={"X-Request-ID": "trace-abc_123"}).headers["X-Request-ID"]
    replaced = client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})

    assert len(generated) == 32
    assert kept == "trace-abc_123"
    assert replaced.headers["X-Request-ID"] != "bad id\nwith newline"


def test_quota_windows_follow_each_providers_reset_clock():
    now = datetime(2026, 9, 30, 3, 0, tzinfo=UTC)  # 2026-09-29 20:00 in Los Angeles (PDT)
    assert window_start("openrouter", now) == datetime(2026, 9, 30, 0, 0, tzinfo=UTC)
    assert window_start("gemini", now) == datetime(2026, 9, 29, 7, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "logger_name, message",
    [
        ("google_adk.google.adk.workflow._node_runner", "Node execution failed with exception"),
        ("opentelemetry.context", "Failed to detach context"),
    ],
)
def test_library_traceback_noise_is_demoted_to_debug(logger_name, message):
    import logging

    from app.observability.logging_setup import DemoteToDebugFilter

    record = logging.getLogger(logger_name).makeRecord(
        logger_name, logging.ERROR, __file__, 1, message, None, None
    )
    root = logging.getLogger()
    previous = root.level
    try:
        root.setLevel(logging.INFO)
        assert DemoteToDebugFilter().filter(record) is False  # dropped at INFO
        assert record.levelname == "DEBUG"
        root.setLevel(logging.DEBUG)
        assert DemoteToDebugFilter().filter(record) is True  # still visible at DEBUG
    finally:
        root.setLevel(previous)
