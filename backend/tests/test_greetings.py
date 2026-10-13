"""Plain greetings get a fixed reply with no retrieval, embedding or model
call, on every assistant (app/orchestration/greetings.py and
ChatOrchestrator._answer_greeting)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_principal
from app.config.assistant_config import AssistantConfig, ModelCapabilities, RetrievalConfig
from app.config.loader import load_all_assistant_configs, load_assistant_config
from app.core.principal import PrincipalContext
from app.ingestion.pipeline import Chunk
from app.main import app
from app.orchestration.chat_service import ChatOrchestrator, ChatStreamDelta, ChatStreamDone
from app.orchestration.greetings import greeting_reply, is_greeting, normalise
from app.orchestration.model_provider import ModelReply, ModelStreamEvent
from app.services.conversation_store import ConversationNotFoundError

BACKEND = Path(__file__).resolve().parent.parent

GREETINGS = [
    "hi",
    "Hi!",
    "HELLO",
    "hello.",
    "hey",
    "hey there",
    "Hey there!!",
    "hiya",
    "howdy",
    "namaste",
    "Namaste 🙏",
    "greetings",
    "good morning",
    "Good Morning!",
    "good afternoon",
    "good evening everyone",
    "hiii",
    "heyyy",
    "hellooo",
    "👋 hi",
    "hi 😊",
    "hello team",
    "hi all",
    "hi everyone",
    "hi, how are you?",
    "hello there, how are you doing?",
    "good morning team how r u",
    "Hello HR Policy Assistant",
    "  hi  ",
]
NOT_GREETINGS = [
    "hi, how many casual leaves do I get?",
    "hello, plot revenue",
    "hi there what is the leave policy",
    "high",
    "history",
    "hi can you help me",
    "hello world",
    "good",
    "morning",
    "his leave balance",
    "hey, what's today's date?",
    "hi there team",
    "hello there hello",
    "how are you",
    "Hi " + "x" * 200,
]


@pytest.mark.parametrize("message", GREETINGS)
def test_greetings_match(message):
    assert is_greeting(message, display_name="HR Policy Assistant")


@pytest.mark.parametrize("message", NOT_GREETINGS)
def test_real_requests_and_lookalikes_do_not_match(message):
    assert not is_greeting(message, display_name="HR Policy Assistant")


def test_normalise():
    assert normalise("  Hiii!!! 👋  ") == "hi"
    assert normalise("Good   MORNING, team") == "god morning team"


def test_reply_default_and_override():
    assert greeting_reply("HR Policy Assistant") == (
        "Hi! I'm HR Policy Assistant. How can I help you today?"
    )
    assert greeting_reply("Ops", "Hello from {display_name} {x}") == "Hello from Ops {x}"


# ---------------------------------------------------------------------------
# Orchestrator: no retrieval, no model, persisted, audited
# ---------------------------------------------------------------------------


class ExplodingRetrieval:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, **kwargs):
        self.calls += 1
        return [
            Chunk(
                document_title="leave_policy",
                chunk_index=0,
                display_text="Casual leave: 6 days per year.",
                embedded_text="x",
                access_labels=frozenset({"role:authenticated"}),
            )
        ]


class RecordingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt):
        self.calls += 1
        return ModelReply(text="You get 6 casual leave days per year.", grounded=True)

    async def generate_stream(self, prompt):
        self.calls += 1
        yield ModelStreamEvent(delta="You get 6 casual leave days per year.")
        yield ModelStreamEvent(
            is_final=True, text="You get 6 casual leave days per year.", grounded=True
        )


class MemoryStore:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, tuple]] = []

    async def create_conversation(self, *, principal, assistant_id):
        return SimpleNamespace(id=uuid.uuid4(), assistant_id=assistant_id)

    async def get_conversation(self, *, conversation_id, principal):
        raise ConversationNotFoundError("not found")

    async def list_messages(self, *, conversation_id, principal):
        return []

    async def append_message(self, *, conversation_id, principal, role, content, citations=()):
        self.messages.append((role, content, tuple(citations)))


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def write(self, recorder):
        self.rows.append(recorder.as_row())


class EmbedderRequests:
    """Records each embedder lookup (a greeting must make none); returning
    None leaves retrieval on its default embedder, as in production."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, config):
        self.calls += 1
        return None


def _setup():
    retrieval, provider, store, audit_store = (
        ExplodingRetrieval(),
        RecordingProvider(),
        MemoryStore(),
        Audit(),
    )
    retrieval.embedder_requests = EmbedderRequests()
    orchestrator = ChatOrchestrator(
        retrieval_service=retrieval,
        provider_factory=lambda config: provider,
        conversation_store=store,
        audit_store=audit_store,
        embedding_provider_factory=retrieval.embedder_requests,
    )
    return orchestrator, retrieval, provider, store, audit_store


def _principal() -> PrincipalContext:
    return PrincipalContext(
        tenant_id="t1", principal_id="u1", labels=frozenset({"role:authenticated"}), clearance=1
    )


def _config(**overrides) -> AssistantConfig:
    data = dict(
        assistant_id="ops_assistant",
        display_name="Ops Assistant",
        description="d",
        tenant_id="t1",
        model=ModelCapabilities(provider="openrouter", model_name="m"),
        retrieval=RetrievalConfig(collection_name="c"),
        system_prompt="s",
    )
    data.update(overrides)
    return AssistantConfig(**data)


def _assert_greeting_turn(retrieval, provider, store, audit_store, text):
    assert text == "Hi! I'm Ops Assistant. How can I help you today?"
    assert retrieval.calls == 0 and provider.calls == 0
    assert retrieval.embedder_requests.calls == 0
    assert store.messages == [("user", "Hi there!", ()), ("assistant", text, ())]
    (row,) = audit_store.rows
    assert row["outcome"] == "greeting"
    assert row["provider_calls"] == [] and row["citations_count"] == 0
    assert row["grounded"] is False and row["tool_calls"] == []


@pytest.mark.asyncio
async def test_greeting_non_streaming():
    orchestrator, retrieval, provider, store, audit_store = _setup()
    result = await orchestrator.handle(
        principal=_principal(), config=_config(), message="Hi there!"
    )
    assert result.greeting is True and result.citations == () and result.chart is None
    _assert_greeting_turn(retrieval, provider, store, audit_store, result.text)


@pytest.mark.asyncio
async def test_greeting_streaming_is_delta_then_done():
    orchestrator, retrieval, provider, store, audit_store = _setup()
    events = [
        e
        async for e in orchestrator.handle_stream(
            principal=_principal(), config=_config(), message="Hi there!"
        )
    ]
    assert [type(e) for e in events] == [ChatStreamDelta, ChatStreamDone]
    assert events[1].greeting is True and events[1].grounded is False
    _assert_greeting_turn(retrieval, provider, store, audit_store, events[0].text)


@pytest.mark.asyncio
async def test_greeting_with_a_real_request_goes_through_retrieval_and_the_model():
    orchestrator, retrieval, provider, _store, audit_store = _setup()
    result = await orchestrator.handle(
        principal=_principal(), config=_config(), message="hi, how many casual leaves do I get?"
    )
    assert result.greeting is False and result.grounded is True
    assert retrieval.calls == 1 and provider.calls == 1
    assert retrieval.embedder_requests.calls == 1
    assert audit_store.rows[0]["outcome"] == "answered"


@pytest.mark.asyncio
async def test_enabled_false_turns_it_off():
    orchestrator, retrieval, provider, _store, audit_store = _setup()
    result = await orchestrator.handle(
        principal=_principal(), config=_config(greeting={"enabled": False}), message="hi"
    )
    assert result.greeting is False
    assert retrieval.calls == 1 and provider.calls == 1


@pytest.mark.asyncio
async def test_custom_message():
    orchestrator, *_ = _setup()
    config = _config(greeting={"message": "Hello! Ask {display_name} about runbooks."})
    result = await orchestrator.handle(principal=_principal(), config=config, message="hey")
    assert result.text == "Hello! Ask Ops Assistant about runbooks."


@pytest.mark.asyncio
async def test_new_yaml_assistant_without_greeting_block_replies(tmp_path):
    path = tmp_path / "new_assistant.yaml"
    path.write_text(
        "assistant_id: brand_new\n"
        "display_name: Brand New Helper\n"
        "description: d\n"
        "tenant_id: t1\n"
        "model: {provider: gemini, model_name: gemini-2.5-flash}\n"
        "retrieval: {collection_name: c}\n"
        "system_prompt: s\n",
        encoding="utf-8",
    )
    config = load_assistant_config(path)
    assert config.greeting.enabled is True and config.greeting.message is None
    orchestrator, retrieval, provider, *_ = _setup()
    result = await orchestrator.handle(
        principal=_principal(), config=config, message="Good morning!"
    )
    assert result.text == "Hi! I'm Brand New Helper. How can I help you today?"
    assert retrieval.calls == provider.calls == 0


def test_greeting_block_rejects_unknown_keys():
    with pytest.raises(ValueError):
        _config(greeting={"enabled": True, "mesage": "typo"})


def _every_assistant_config() -> list[AssistantConfig]:
    configs = list(load_all_assistant_configs(BACKEND / "configs").values())
    configs += list(load_all_assistant_configs(BACKEND / "configs/examples").values())
    return configs


@pytest.mark.asyncio
@pytest.mark.parametrize("config", _every_assistant_config(), ids=lambda c: c.assistant_id)
async def test_every_existing_assistant_answers_a_greeting_without_a_model(config):
    assert config.greeting.enabled is True  # existing configs load unchanged
    orchestrator, retrieval, provider, _store, audit_store = _setup()
    result = await orchestrator.handle(principal=_principal(), config=config, message="hello")
    assert result.text == f"Hi! I'm {config.display_name}. How can I help you today?"
    assert retrieval.calls == 0 and provider.calls == 0
    assert retrieval.embedder_requests.calls == 0
    assert audit_store.rows[0]["outcome"] == "greeting"
    assert audit_store.rows[0]["provider_calls"] == []


# ---------------------------------------------------------------------------
# HTTP: /chat and /chat/stream carry the greeting flag
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    orchestrator, *_ = _setup()
    app.state.chat_orchestrator = orchestrator
    app.dependency_overrides[get_current_principal] = _principal
    yield TestClient(app)
    app.dependency_overrides.clear()
    del app.state.chat_orchestrator


def test_chat_route_returns_greeting_flag(client):
    body = client.post("/chat", json={"assistant_id": "finance_assistant", "message": "hi"}).json()
    assert body["greeting"] is True and body["grounded"] is False and body["citations"] == []
    assert body["reply"] == "Hi! I'm Financial Report Analysis Assistant. How can I help you today?"


def test_stream_route_follows_the_sse_contract(client):
    with client.stream(
        "POST", "/chat/stream", json={"assistant_id": "hr_assistant", "message": "Hello!"}
    ) as response:
        raw = "".join(response.iter_text())
    events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["delta", "done"]
    assert events[0]["text"] == "Hi! I'm HR Policy Assistant. How can I help you today?"
    assert events[1]["greeting"] is True and events[1]["grounded"] is False


def test_non_greeting_responses_say_greeting_false(client):
    body = client.post(
        "/chat", json={"assistant_id": "hr_assistant", "message": "how many casual leaves?"}
    ).json()
    assert body["greeting"] is False
