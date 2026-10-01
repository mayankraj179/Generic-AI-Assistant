"""Regression tests for model-invented citation markup leaking into replies.

Reported live on finance_assistant (OpenRouter, nvidia/nemotron-3-super-
120b-a12b) for "how many days left till 2nd October and how is that a
floating holiday?": the reply contained 【{"cursor": 0, "loc": 0}】 and
【{"cursor": 0, "loc": 123}】 inline. The raw capture showed the model wrote
these itself (nothing like them was in its context); they must never reach
the user-facing text, on either path."""

from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace

import httpx
import pytest

from app.config.assistant_config import (
    AssistantConfig,
    GuardrailsConfig,
    ModelCapabilities,
    RetrievalConfig,
)
from app.core.principal import PrincipalContext
from app.ingestion.pipeline import Chunk
from app.observability.audit import TurnRecorder
from app.orchestration.chat_service import (
    ChatOrchestrator,
    ChatStreamDelta,
    ChatStreamDone,
    strip_citation_markup,
)
from app.orchestration.model_provider import GroundedPrompt, ModelStreamEvent
from app.orchestration.openrouter_provider import OpenRouterProvider

# The reply exactly as reported (transcribed from the widget).
REPORTED_REPLY = (
    'There are 2 days left until October 2, 2026.\n【{"cursor": 0, "loc": 0}】\n\n'
    "No, October 2, 2026 is not a floating holiday. In the India holiday list for 2026, "
    'October 2 is shown as "Gandhi Jayanti" and marked **Official Holiday**, not as a '
    'Floating Holiday 【{"cursor": 0, "loc": 123}】. The accompanying note states that "Rest '
    'all days are NOT Floating Holidays," confirming that any day not explicitly listed as a '
    'floating holiday is not one 【{"cursor": 0, "loc": 123}】.'
)
# The variant the same question produced in the live reproduction.
REPRODUCED_REPLY = (
    "No, Gandhi Jayanti is not a floating holiday. There are 2 days left until 2 October 2026 "
    '(Gandhi Jayanti). The note states "Rest all days are NOT Floating Holidays."'
    "【source: holidays_2026, chunk 3】【source: current_datetime】【source: calculate】"
)

_LEAK_PATTERNS = [
    re.compile(r"\{\s*\"\w+\"\s*:"),  # any JSON-object-looking fragment
    re.compile(r"\"(cursor|loc)\""),
    re.compile("【"),
]


def _assert_clean(text: str) -> None:
    for pattern in _LEAK_PATTERNS:
        assert not pattern.search(text), f"leaked markup {pattern.pattern!r} in: {text!r}"


def _holiday_chunk() -> Chunk:
    text = (
        "Gandhi Jayanti  2026-10-02 (Friday)  India  Official Holiday\n"
        "Rest all days are NOT Floating Holidays."
    )
    return Chunk(
        document_title="holidays_2026",
        chunk_index=3,
        display_text=text,
        embedded_text=f"holidays_2026: {text}",
        access_labels=frozenset({"role:authenticated"}),
    )


class _Retrieval:
    async def search(
        self, *, query, principal, assistant_id, top_k, min_similarity=0.0, embedder=None
    ):
        return [_holiday_chunk()]


class _Conversations:
    def __init__(self) -> None:
        self.messages: dict[uuid.UUID, list] = {}

    async def create_conversation(self, *, principal, assistant_id):
        cid = uuid.uuid4()
        self.messages[cid] = []
        return SimpleNamespace(id=cid)

    async def append_message(self, *, conversation_id, principal, role, content, citations=()):
        self.messages[conversation_id].append(SimpleNamespace(role=role, content=content))


class _Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def write(self, recorder: TurnRecorder) -> None:
        self.rows.append(recorder.as_row())


def _config() -> AssistantConfig:
    return AssistantConfig(
        assistant_id="finance_assistant",
        display_name="Finance",
        description="test",
        tenant_id="bitwise-global",
        model=ModelCapabilities(provider="openrouter", model_name="nemotron-test"),
        retrieval=RetrievalConfig(enabled=True, collection_name="c", top_k=4, min_similarity=0.2),
        guardrails=GuardrailsConfig(),
        system_prompt="Answer from context.",
        min_clearance=0,
        enabled_tools=["current_datetime", "calculate"],
    )


def _principal() -> PrincipalContext:
    return PrincipalContext(
        tenant_id="local-development", principal_id="u", labels=frozenset({"role:authenticated"})
    )


_RealAsyncClient = httpx.AsyncClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_reply", [REPORTED_REPLY, REPRODUCED_REPLY], ids=["reported", "reproduced"]
)
async def test_tool_call_plus_retrieval_reply_never_contains_citation_markup(
    monkeypatch: pytest.MonkeyPatch, model_reply
):
    # Real OpenRouterProvider tool loop over a mocked HTTP API: first a
    # current_datetime tool call, then the final answer with the markup.
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "current_datetime", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"role": "assistant", "content": model_reply}}]},
    ]
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(sent) - 1])

    import app.orchestration.openrouter_provider as module

    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )
    audit = _Audit()
    orchestrator = ChatOrchestrator(
        retrieval_service=_Retrieval(),
        provider_factory=lambda config: OpenRouterProvider(model_name="nemotron-test", api_key="k"),
        conversation_store=_Conversations(),
        audit_store=audit,
    )

    result = await orchestrator.handle(
        principal=_principal(),
        config=_config(),
        message="how many days left till 2nd October and how is that a floating holiday?",
    )

    assert len(sent) == 2  # the tool round-trip really happened
    assert sent[1]["messages"][-1]["role"] == "tool"
    _assert_clean(result.text)
    assert "Official Holiday" in result.text or "floating holiday" in result.text
    # Sources still arrive, structurally, for the widget's "Source: X (chunk Y)" list.
    assert [(c.document_title, c.chunk_index) for c in result.citations] == [("holidays_2026", 3)]
    actions = audit.rows[0]["guardrail_actions"]
    assert any(a["action"] == "citation_markup_stripped" and a["count"] >= 1 for a in actions)


class _StreamingProvider:
    """Streams the reported reply with each marker split across deltas."""

    async def generate(self, prompt: GroundedPrompt):  # pragma: no cover - stream only
        raise NotImplementedError

    async def generate_stream(self, prompt: GroundedPrompt):
        text = REPORTED_REPLY
        cuts = [text.index("【") + 3, text.index('"loc"'), text.rindex("【") + 1, len(text) - 4]
        start = 0
        for cut in [*cuts, len(text)]:
            yield ModelStreamEvent(delta=text[start:cut])
            start = cut
        yield ModelStreamEvent(is_final=True, text=text, grounded=True)


@pytest.mark.asyncio
async def test_streamed_deltas_and_persisted_text_are_clean_and_consistent():
    conversations = _Conversations()
    orchestrator = ChatOrchestrator(
        retrieval_service=_Retrieval(),
        provider_factory=lambda config: _StreamingProvider(),
        conversation_store=conversations,
    )

    events = [
        e
        async for e in orchestrator.handle_stream(
            principal=_principal(), config=_config(), message="days till 2nd October?"
        )
    ]

    streamed = "".join(e.text for e in events if isinstance(e, ChatStreamDelta))
    _assert_clean(streamed)
    assert isinstance(events[-1], ChatStreamDone)
    (conversation,) = conversations.messages.values()
    persisted = conversation[-1].content
    _assert_clean(persisted)
    assert persisted == strip_citation_markup(REPORTED_REPLY)[0]
    # Equal up to whitespace already sent before a marker began (see
    # _StreamingMarkupStripper): no delta boundaries are altered for that.
    squash = lambda s: re.sub(r"\s+", " ", s).replace(" .", ".")  # noqa: E731
    assert squash(streamed) == squash(persisted)


def test_ordinary_bracketed_text_is_left_alone():
    text = "See the 【Note】 at the end."
    assert strip_citation_markup(text) == (text, 0)
