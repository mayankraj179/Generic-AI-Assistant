from __future__ import annotations

import pytest

from app.config.assistant_config import (
    AssistantConfig,
    GuardrailsConfig,
    ModelCapabilities,
    RetrievalConfig,
)
from app.core.principal import PrincipalContext
from app.ingestion.pipeline import Chunk
from app.observability.audit import record_tool_call
from app.orchestration.chat_service import (
    CHART_WITHOUT_CONTEXT_REPLY,
    NO_AUTHORIZED_CONTEXT_REPLY,
    ChatOrchestrator,
    ChatStreamDelta,
    ChatStreamDone,
)
from app.orchestration.model_provider import GroundedPrompt, ModelReply, ModelStreamEvent
from app.services.conversation_store import ConversationNotFoundError

# ---------------------------------------------------------------------------
# Shared fixtures / fakes (self-contained — mirrors test_chart_generation.py's
# convention of not importing from test_chat_orchestration.py, matching this
# codebase's existing per-file convention of no shared conftest fixtures).
#
# What's tested here is deliberately scoped to the orchestration LAYER:
# that AssistantConfig.enabled_tools/max_tool_calls correctly become
# GroundedPrompt.enabled_tools/max_tool_calls, and that existing output
# guardrails still apply to whatever text a provider returns regardless of
# whether tools were involved in producing it. The actual tool-call LOOP
# mechanics (propose -> execute -> feed back -> repeat, bounded by
# max_tool_calls, graceful degradation) live inside each provider — see
# tests/test_openrouter_provider.py's "Tool-calling loop" section for that;
# ChatOrchestrator never drives the loop itself (see GroundedPrompt.
# enabled_tools's docstring in app/orchestration/model_provider.py for why).
# ---------------------------------------------------------------------------


def _make_config(*, enabled_tools: list[str] | None = None, max_tool_calls: int = 4, **overrides):
    defaults = dict(
        assistant_id="finance_assistant",
        display_name="Financial Report Analysis Assistant",
        description="Test assistant",
        tenant_id="bitwise-global",
        model=ModelCapabilities(provider="google_adk", model_name="gemini-2.5-flash"),
        retrieval=RetrievalConfig(
            enabled=True, collection_name="finance_reports", top_k=8, min_similarity=0.6
        ),
        enabled_tools=enabled_tools or [],
        max_tool_calls=max_tool_calls,
        guardrails=GuardrailsConfig(),
        system_prompt="You are a financial report analysis assistant.",
        min_clearance=0,
    )
    defaults.update(overrides)
    return AssistantConfig(**defaults)


_NOW = {"iso": "2026-09-24T10:00:00+00:00", "timezone": "UTC", "day_of_week": "Thursday"}
_SUM = {"result": 4.0, "expression": "2+2", "kind": "arithmetic"}


def _make_chunk() -> Chunk:
    return Chunk(
        document_title="nova_horizon_fy2025",
        chunk_index=0,
        display_text="Revenue was $96.5M, up 22% YoY. Net profit was $18.2M.",
        embedded_text="nova_horizon_fy2025: Revenue was $96.5M. Net profit was $18.2M.",
        access_labels=frozenset({"role:employee"}),
    )


class FakeRetrievalService:
    def __init__(self, chunks_to_return: list[Chunk] | None = None):
        self.chunks_to_return = chunks_to_return or []

    async def search(
        self, *, query, principal, assistant_id, top_k, min_similarity=0.0, embedder=None
    ):
        return self.chunks_to_return


class FakeToolAwareModelProvider:
    """Captures every GroundedPrompt it receives (so the test can assert on
    enabled_tools/max_tool_calls) and returns a caller-supplied reply text —
    standing in for a provider that may or may not have used a tool
    internally to produce that text; ChatOrchestrator can't tell the
    difference either way, by design. ``tool_calls`` are (name, result)
    pairs, result None meaning the call failed, reported to the turn recorder
    the way every real provider reports them; that is what
    ChatOrchestrator._require_tool_result checks."""

    def __init__(
        self,
        *,
        reply_text: str = "The net profit was $18.2M.",
        tool_calls: list[tuple[str, dict | None]] | None = None,
        deltas: list[str] | None = None,
    ):
        self.reply_text = reply_text
        self.tool_calls = tool_calls or []
        self.deltas = deltas or [reply_text]
        self.calls: list[GroundedPrompt] = []

    def _run_tools(self) -> None:
        for name, result in self.tool_calls:
            if result is None:
                record_tool_call(name, ok=False, error="simulated failure")
            else:
                record_tool_call(name, ok=True, result=result)

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:
        self.calls.append(prompt)
        self._run_tools()
        return ModelReply(text=self.reply_text, grounded=bool(prompt.retrieved_chunks))

    async def generate_stream(self, prompt: GroundedPrompt):
        self.calls.append(prompt)
        self._run_tools()
        for delta in self.deltas:
            yield ModelStreamEvent(delta=delta)
        yield ModelStreamEvent(
            is_final=True, text="".join(self.deltas), grounded=bool(prompt.retrieved_chunks)
        )


class FakeConversationStore:
    def __init__(self) -> None:
        self._conversations: dict = {}
        self._messages: dict = {}

    async def create_conversation(self, *, principal: PrincipalContext, assistant_id: str):
        import uuid
        from types import SimpleNamespace

        conversation_id = uuid.uuid4()
        self._conversations[conversation_id] = {
            "tenant_id": principal.tenant_id,
            "principal_id": principal.principal_id,
            "assistant_id": assistant_id,
        }
        self._messages[conversation_id] = []
        return SimpleNamespace(id=conversation_id, assistant_id=assistant_id)

    async def get_conversation(self, *, conversation_id, principal: PrincipalContext):
        from types import SimpleNamespace

        self._assert_owned(conversation_id, principal)
        conv = self._conversations[conversation_id]
        return SimpleNamespace(id=conversation_id, assistant_id=conv["assistant_id"])

    async def append_message(self, *, conversation_id, principal, role, content, citations=()):
        import uuid
        from types import SimpleNamespace

        self._assert_owned(conversation_id, principal)
        sequence_no = len(self._messages[conversation_id])
        message = SimpleNamespace(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=tuple(citations),
            sequence_no=sequence_no,
        )
        self._messages[conversation_id].append(message)
        return message

    async def list_messages(self, *, conversation_id, principal):
        self._assert_owned(conversation_id, principal)
        return list(self._messages[conversation_id])

    def _assert_owned(self, conversation_id, principal) -> None:
        conv = self._conversations.get(conversation_id)
        if (
            conv is None
            or conv["tenant_id"] != principal.tenant_id
            or conv["principal_id"] != principal.principal_id
        ):
            raise ConversationNotFoundError("conversation not found")


def _principal(**overrides) -> PrincipalContext:
    defaults = dict(
        tenant_id="dev-tenant",
        principal_id="dev-user-123",
        labels=frozenset({"role:authenticated", "scope:development"}),
        clearance=1,
    )
    defaults.update(overrides)
    return PrincipalContext(**defaults)


def _make_orchestrator(*, retrieval, provider) -> ChatOrchestrator:
    return ChatOrchestrator(
        retrieval_service=retrieval,
        provider_factory=lambda config: provider,
        conversation_store=FakeConversationStore(),
    )


# ---------------------------------------------------------------------------
# Config -> AssistantConfig validation (unknown tool name, duplicates)
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_tool_name():
    with pytest.raises(Exception, match="unknown tool name"):
        _make_config(enabled_tools=["current_datetime", "not_a_real_tool"])


def test_config_accepts_known_tool_names():
    config = _make_config(enabled_tools=["current_datetime", "calculate"])
    assert config.enabled_tools == ["current_datetime", "calculate"]


def test_config_max_tool_calls_defaults_to_four():
    config = _make_config()
    assert config.max_tool_calls == 4


def test_config_rejects_non_positive_max_tool_calls():
    with pytest.raises(ValueError):
        _make_config(max_tool_calls=0)


# ---------------------------------------------------------------------------
# AssistantConfig -> GroundedPrompt wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enabled_tools_and_max_tool_calls_flow_into_the_prompt():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeToolAwareModelProvider()
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)
    config = _make_config(enabled_tools=["calculate"], max_tool_calls=2)

    await orchestrator.handle(
        principal=_principal(), config=config, message="What was the net profit?"
    )

    assert provider.calls[0].enabled_tools == ("calculate",)
    assert provider.calls[0].max_tool_calls == 2


@pytest.mark.asyncio
async def test_enabled_tools_defaults_to_empty_tuple_when_config_has_none():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeToolAwareModelProvider()
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)
    config = _make_config()  # enabled_tools=[] by default

    await orchestrator.handle(
        principal=_principal(), config=config, message="What was the net profit?"
    )

    assert provider.calls[0].enabled_tools == ()
    assert provider.calls[0].max_tool_calls == 4


@pytest.mark.asyncio
async def test_multiple_enabled_tools_all_flow_into_the_prompt():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeToolAwareModelProvider()
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)
    config = _make_config(enabled_tools=["current_datetime", "calculate"])

    await orchestrator.handle(
        principal=_principal(), config=config, message="What day is it and what's the margin?"
    )

    assert provider.calls[0].enabled_tools == ("current_datetime", "calculate")


# ---------------------------------------------------------------------------
# Output guardrails still apply regardless of tool involvement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_guardrails_still_apply_to_a_tool_assisted_reply():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeToolAwareModelProvider(
        reply_text="Here is some confidential internal-only content."
    )
    config = _make_config(
        enabled_tools=["calculate"],
        guardrails=GuardrailsConfig(unsafe_output_patterns=[r"confidential internal-only"]),
    )
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="Compute the margin for me."
    )

    # The unsafe-output guardrail fires exactly as it would for a non-tool
    # reply — enabling tools does not create a bypass.
    assert result.grounded is False
    from app.orchestration.chat_service import OUTPUT_GUARDRAIL_FALLBACK_REPLY

    assert result.text == OUTPUT_GUARDRAIL_FALLBACK_REPLY


@pytest.mark.asyncio
async def test_citations_still_reflect_this_turns_retrieval_with_tools_enabled():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    provider = FakeToolAwareModelProvider(reply_text="Net profit was $18.2M, per the report.")
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="What was net profit?"
    )

    assert result.grounded is True
    assert len(result.citations) == 1
    assert result.citations[0].document_title == "nova_horizon_fy2025"


# ---------------------------------------------------------------------------
# The gate change: zero retrieved chunks no longer means an automatic
# refusal when the assistant has tools configured — see
# ChatOrchestrator._prepare_turn / NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_chunks_with_tools_enabled_calls_the_model_instead_of_refusing():
    """The live bug this session fixes: finance_assistant (enabled_tools=
    [current_datetime, calculate]) asked a pure-calculation question with
    zero relevant retrieved chunks must reach the model — not the hard
    safe-refusal — because calculate() can answer it with no document
    involvement at all."""
    retrieval = FakeRetrievalService([])  # nothing relevant/authorized retrieved
    provider = FakeToolAwareModelProvider(
        reply_text="There are 462 days left until 2026-12-31.",
        tool_calls=[
            ("current_datetime", _NOW),
            ("calculate", {"result": 462, "expression": "days between", "kind": "date_difference"}),
        ],
    )
    config = _make_config(enabled_tools=["current_datetime", "calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(),
        config=config,
        message="how many days left till 31st december 2026",
    )

    assert len(provider.calls) == 1  # the model WAS called, unlike the hard-gate path
    assert result.text == "There are 462 days left until 2026-12-31."
    assert result.text != NO_AUTHORIZED_CONTEXT_REPLY


# ---------------------------------------------------------------------------
# _require_tool_result: on a zero-chunk turn the model's reply stands only if
# a tool call succeeded. Live 2026-09-30, gpt-6-luna and gemini-2.5-flash both
# answered "what is the capital of France" from general knowledge here, and
# gpt-6-luna wrote a full Bali itinerary, despite the system-prompt guidance.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply_text",
    [
        "Paris.",
        "Here's a relaxed 4-day Bali itinerary: Day 1 — Ubud ...",
        # Even a well-phrased decline is replaced: the fixed reply is the
        # same on every provider, and nothing the model wrote reaches the user.
        "I don't have enough information to answer that.",
    ],
)
async def test_zero_chunks_reply_without_any_tool_call_is_replaced(reply_text):
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(reply_text=reply_text, tool_calls=[])
    config = _make_config(enabled_tools=["current_datetime", "calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="what is the capital of France"
    )

    assert len(provider.calls) == 1
    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY
    assert result.grounded is False
    assert result.citations == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply_text",
    ["Paris.", "Here's a relaxed 4-day Bali itinerary: start in Ubud, end in Seminyak."],
)
async def test_zero_chunks_reply_after_an_unrelated_tool_call_is_replaced(reply_text):
    """Live 2026-09-30: gpt-6-luna called current_datetime on nearly every
    out-of-scope question. A successful call whose result the reply never
    states unlocks nothing."""
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(
        reply_text=reply_text, tool_calls=[("current_datetime", _NOW)]
    )
    config = _make_config(enabled_tools=["current_datetime", "calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="what is the capital of France"
    )

    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY


@pytest.mark.asyncio
async def test_zero_chunks_reply_after_only_failed_tool_calls_is_replaced():
    """A failed tool call backs nothing, so it doesn't count."""
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(
        reply_text="It is 14:00.", tool_calls=[("current_datetime", None)]
    )
    config = _make_config(enabled_tools=["current_datetime"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="what time is it"
    )

    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY


@pytest.mark.asyncio
async def test_replaced_reply_is_what_gets_persisted():
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(reply_text="Paris.")
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)
    principal = _principal()

    result = await orchestrator.handle(
        principal=principal, config=config, message="what is the capital of France"
    )

    messages = await orchestrator._conversation_store.list_messages(
        conversation_id=result.conversation_id, principal=principal
    )
    assert [m.content for m in messages] == [
        "what is the capital of France",
        NO_AUTHORIZED_CONTEXT_REPLY,
    ]


@pytest.mark.asyncio
async def test_replacement_is_recorded_as_a_guardrail_action():
    from app.observability import audit

    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(reply_text="Paris.")
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    await orchestrator.handle(
        principal=_principal(), config=config, message="what is the capital of France"
    )

    recorder = audit.current()
    assert recorder is not None
    assert recorder.outcome == "no_context"
    assert {
        "action": "ungrounded_reply_replaced",
        "tool_calls": 0,
        "successful_tool_calls": 0,
    } in recorder.guardrail_actions


@pytest.mark.asyncio
async def test_zero_chunks_chart_request_without_tool_call_gets_the_fixed_chart_reply():
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(reply_text="I cannot plot graphs.")
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="plot me a graph"
    )

    assert result.text == CHART_WITHOUT_CONTEXT_REPLY
    assert result.chart is None


@pytest.mark.asyncio
async def test_chunks_found_reply_stands_without_any_tool_call():
    """The check applies only to zero-chunk turns: a grounded answer never
    needs a tool."""
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeToolAwareModelProvider(reply_text="Net profit was $18.2M.", tool_calls=[])
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="What was the net profit?"
    )

    assert result.text == "Net profit was $18.2M."
    assert result.grounded is True


async def _collect(stream) -> list:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_stream_zero_chunks_without_tool_call_never_sends_the_model_text():
    """The model's deltas are held back on these turns, so a replaced reply
    never reaches the client, even partially."""
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(deltas=["Here's a 4-day ", "Bali itinerary."])
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    events = await _collect(
        orchestrator.handle_stream(
            principal=_principal(), config=config, message="plan me a trip to Bali"
        )
    )

    deltas = [e.text for e in events if isinstance(e, ChatStreamDelta)]
    assert deltas == [NO_AUTHORIZED_CONTEXT_REPLY]
    assert isinstance(events[-1], ChatStreamDone)
    assert events[-1].grounded is False


@pytest.mark.asyncio
async def test_stream_zero_chunks_with_tool_result_sends_the_reply_as_one_delta():
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(
        deltas=["2 + 2 ", "= 4."], tool_calls=[("calculate", _SUM)]
    )
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    events = await _collect(
        orchestrator.handle_stream(principal=_principal(), config=config, message="what's 2+2")
    )

    deltas = [e.text for e in events if isinstance(e, ChatStreamDelta)]
    assert deltas == ["2 + 2 = 4."]
    assert isinstance(events[-1], ChatStreamDone)


@pytest.mark.asyncio
async def test_stream_with_chunks_still_streams_deltas_unchanged():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeToolAwareModelProvider(deltas=["Net profit ", "was $18.2M."])
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    events = await _collect(
        orchestrator.handle_stream(
            principal=_principal(), config=config, message="What was the net profit?"
        )
    )

    deltas = [e.text for e in events if isinstance(e, ChatStreamDelta)]
    assert deltas == ["Net profit ", "was $18.2M."]


@pytest.mark.asyncio
async def test_zero_chunks_with_no_tools_configured_is_unchanged_regression_guard():
    """REGRESSION GUARD: this is the exact pre-existing behavior that must
    stay completely unchanged by this session's gate change. An assistant
    with retrieval enabled but enabled_tools empty (e.g. hr_assistant) must
    still hit the hard safe-refusal gate on zero chunks and never call the
    model — identical to before tools existed at all."""
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider()
    config = _make_config(enabled_tools=[])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    from app.orchestration.chat_service import NO_AUTHORIZED_CONTEXT_REPLY

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="What is executive comp?"
    )

    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY
    assert result.grounded is False
    assert result.citations == ()
    assert provider.calls == []  # the model must never be called — unchanged


@pytest.mark.asyncio
async def test_zero_chunks_with_tools_prompt_includes_no_document_context_guidance():
    from app.orchestration.chat_service import NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE

    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider()
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    await orchestrator.handle(
        principal=_principal(), config=config, message="how many days left till 2026-12-31"
    )

    sent_prompt = provider.calls[0]
    assert NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE in sent_prompt.system_prompt
    assert sent_prompt.retrieved_chunks == ()
    # No confusing empty "RETRIEVED CONTEXT:" section — the plain question only.
    assert "RETRIEVED CONTEXT:" not in sent_prompt.user_message
    assert sent_prompt.user_message == "how many days left till 2026-12-31"


@pytest.mark.asyncio
async def test_chunks_found_with_tools_does_not_get_the_no_document_context_guidance():
    """The guidance is only relevant/added for a genuinely no-document-
    context turn — a turn with real retrieved chunks (tools or not) must
    not get an instruction that contradicts the actual state of this turn."""
    from app.orchestration.chat_service import NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE

    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeToolAwareModelProvider()
    config = _make_config(enabled_tools=["calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    await orchestrator.handle(
        principal=_principal(), config=config, message="What was the net profit?"
    )

    sent_prompt = provider.calls[0]
    assert NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE not in sent_prompt.system_prompt


@pytest.mark.asyncio
async def test_tool_only_answer_is_never_grounded_and_carries_no_citations():
    """Key correctness property from this session: a tool-only reply (zero
    document involvement) must never claim/imply document backing —
    grounded stays strictly "backed by retrieved document content" (see
    ModelReply/GroundedPrompt), so it must be False here, and citations
    must be empty, even though the model successfully answered via a tool."""
    retrieval = FakeRetrievalService([])
    provider = FakeToolAwareModelProvider(
        reply_text="It is currently Thursday, 2026-09-24.",
        tool_calls=[("current_datetime", _NOW)],
    )
    config = _make_config(enabled_tools=["current_datetime"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="what day is it today"
    )

    assert result.grounded is False
    assert result.citations == ()
    assert "according to" not in result.text.lower()
    assert "report" not in result.text.lower()
