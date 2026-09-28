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
from app.orchestration.chat_service import ChatOrchestrator
from app.orchestration.model_provider import GroundedPrompt, ModelReply
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
    difference either way, by design."""

    def __init__(self, *, reply_text: str = "The net profit was $18.2M."):
        self.reply_text = reply_text
        self.calls: list[GroundedPrompt] = []

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:
        self.calls.append(prompt)
        return ModelReply(text=self.reply_text, grounded=bool(prompt.retrieved_chunks))


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
    provider = FakeToolAwareModelProvider(reply_text="There are 462 days left until 2026-12-31.")
    config = _make_config(enabled_tools=["current_datetime", "calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(),
        config=config,
        message="how many days left till 31st december 2026",
    )

    assert len(provider.calls) == 1  # the model WAS called, unlike the hard-gate path
    assert result.text == "There are 462 days left until 2026-12-31."
    from app.orchestration.chat_service import NO_AUTHORIZED_CONTEXT_REPLY

    assert result.text != NO_AUTHORIZED_CONTEXT_REPLY


@pytest.mark.asyncio
async def test_zero_chunks_with_tools_enabled_still_prompt_null_no_such_tool_declines():
    """Having tools enabled does not mean every zero-retrieval question gets
    a confident answer — a question no available tool can actually answer
    must still end in a decline. ChatOrchestrator itself makes no such
    decision (see GroundedPrompt.enabled_tools's docstring: the tool-call
    loop lives inside the provider); this only asserts the orchestrator
    faithfully passes through whatever the model decides and never
    substitutes/fabricates its own answer."""
    retrieval = FakeRetrievalService([])
    decline_text = (
        "I don't have a way to answer that with the tools available to me. "
        "I don't have enough information to answer."
    )
    provider = FakeToolAwareModelProvider(reply_text=decline_text)
    config = _make_config(enabled_tools=["current_datetime", "calculate"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="what's our marketing budget"
    )

    assert len(provider.calls) == 1
    assert result.text == decline_text
    assert result.grounded is False
    assert result.citations == ()


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
    provider = FakeToolAwareModelProvider(reply_text="It is currently Thursday, 2026-09-24.")
    config = _make_config(enabled_tools=["current_datetime"])
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="what day is it today"
    )

    assert result.grounded is False
    assert result.citations == ()
    assert "according to" not in result.text.lower()
    assert "report" not in result.text.lower()
