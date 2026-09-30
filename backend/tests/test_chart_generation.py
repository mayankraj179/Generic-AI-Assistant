from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from app.config.assistant_config import (
    AssistantConfig,
    GuardrailsConfig,
    ModelCapabilities,
    RetrievalConfig,
)
from app.core.principal import PrincipalContext
from app.ingestion.pipeline import Chunk
from app.orchestration.chat_service import (
    CHART_TURN_GUIDANCE,
    CHART_WITHOUT_CONTEXT_GUIDANCE,
    NO_AUTHORIZED_CONTEXT_REPLY,
    ChatOrchestrator,
    ChatStreamChart,
    ChatStreamDelta,
    ChatStreamDone,
    ChatStreamError,
)
from app.orchestration.model_provider import (
    ChartSeries,
    ChartSourceChunk,
    ChartSpec,
    GroundedPrompt,
    ModelReply,
    ModelStreamEvent,
)
from app.services.conversation_store import ConversationNotFoundError, MessageCitation

# ---------------------------------------------------------------------------
# Shared fixtures / fakes (self-contained — mirrors test_chat_orchestration.py's
# conventions rather than importing from it, matching this codebase's
# existing per-file convention of no shared conftest fixtures).
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> AssistantConfig:
    defaults = dict(
        assistant_id="hr_assistant",
        display_name="HR Policy Assistant",
        description="Test assistant",
        tenant_id="bitwise-global",
        model=ModelCapabilities(provider="google_adk", model_name="gemini-2.5-flash"),
        retrieval=RetrievalConfig(
            enabled=True, collection_name="hr_policy_docs", top_k=8, min_similarity=0.6
        ),
        guardrails=GuardrailsConfig(),
        system_prompt="You are an HR policy assistant. Answer only from retrieved content.",
        min_clearance=0,
    )
    defaults.update(overrides)
    return AssistantConfig(**defaults)


def _make_chunk(
    text: str = "Q1 headcount was 40, Q2 was 45, Q3 was 50.",
    *,
    document_title: str = "headcount_report",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        document_title=document_title,
        chunk_index=chunk_index,
        display_text=text,
        embedded_text=f"{document_title}: {text}",
        access_labels=frozenset({"role:employee"}),
    )


def _make_chart(
    *,
    chart_type: str = "bar",
    source_chunks: Sequence[ChartSourceChunk] | None = None,
) -> ChartSpec:
    return ChartSpec(
        chart_type=chart_type,
        title="Headcount by Quarter",
        labels=["Q1", "Q2", "Q3"],
        series=[ChartSeries(name="Headcount", values=[40, 45, 50])],
        source_chunks=list(source_chunks)
        if source_chunks is not None
        else [ChartSourceChunk(document_title="headcount_report", chunk_index=0)],
    )


class FakeRetrievalService:
    def __init__(self, chunks_to_return: list[Chunk] | None = None):
        self.chunks_to_return = chunks_to_return or []
        self.calls: list[dict] = []

    async def search(
        self, *, query, principal, assistant_id, top_k, min_similarity=0.0, embedder=None
    ):
        self.calls.append({"query": query, "principal": principal})
        return self.chunks_to_return


class FakeChartModelProvider:
    """Non-streaming fake that returns a caller-supplied ChartSpec (or None)
    alongside the text reply, exactly the shape ModelReply.chart carries."""

    def __init__(self, *, reply_text: str = "Here is the data.", chart: ChartSpec | None = None):
        self.reply_text = reply_text
        self.chart = chart
        self.calls: list[GroundedPrompt] = []

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:
        self.calls.append(prompt)
        return ModelReply(
            text=self.reply_text, grounded=bool(prompt.retrieved_chunks), chart=self.chart
        )

    async def generate_stream(self, prompt: GroundedPrompt):  # pragma: no cover - unused here
        raise NotImplementedError("this fake is for the non-streaming path only")


class FakeChartStreamingModelProvider:
    """Streaming counterpart: yields the given deltas, then a final event
    carrying the caller-supplied ChartSpec (or None)."""

    def __init__(
        self,
        *,
        deltas: list[str],
        chart: ChartSpec | None = None,
        grounded: bool | None = None,
    ):
        self.deltas = deltas
        self.chart = chart
        self.grounded_override = grounded
        self.calls: list[GroundedPrompt] = []

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:  # pragma: no cover - unused
        raise NotImplementedError("this fake is for the streaming path only")

    async def generate_stream(self, prompt: GroundedPrompt):
        self.calls.append(prompt)
        for delta in self.deltas:
            yield ModelStreamEvent(delta=delta)
        grounded = (
            self.grounded_override
            if self.grounded_override is not None
            else bool(prompt.retrieved_chunks)
        )
        yield ModelStreamEvent(
            is_final=True, text="".join(self.deltas), grounded=grounded, chart=self.chart
        )


class FakeConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[uuid.UUID, dict] = {}
        self._messages: dict[uuid.UUID, list] = {}

    async def create_conversation(self, *, principal: PrincipalContext, assistant_id: str):
        from types import SimpleNamespace

        conversation_id = uuid.uuid4()
        self._conversations[conversation_id] = {
            "tenant_id": principal.tenant_id,
            "principal_id": principal.principal_id,
            "assistant_id": assistant_id,
        }
        self._messages[conversation_id] = []
        return SimpleNamespace(id=conversation_id, assistant_id=assistant_id)

    async def get_conversation(self, *, conversation_id: uuid.UUID, principal: PrincipalContext):
        from types import SimpleNamespace

        self._assert_owned(conversation_id, principal)
        conv = self._conversations[conversation_id]
        return SimpleNamespace(id=conversation_id, assistant_id=conv["assistant_id"])

    async def append_message(
        self, *, conversation_id, principal, role, content, citations=()
    ):
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


def _make_orchestrator(*, retrieval, provider, conversation_store=None) -> ChatOrchestrator:
    return ChatOrchestrator(
        retrieval_service=retrieval,
        provider_factory=lambda config: provider,
        conversation_store=conversation_store or FakeConversationStore(),
    )


# ---------------------------------------------------------------------------
# ChartSpec's own structural validation (Part 0)
# ---------------------------------------------------------------------------


def test_pie_chart_with_more_than_one_series_is_rejected():
    with pytest.raises(ValueError):
        ChartSpec(
            chart_type="pie",
            title="Bad pie",
            labels=["A", "B"],
            series=[ChartSeries(name="s1", values=[1, 2]), ChartSeries(name="s2", values=[3, 4])],
            source_chunks=[ChartSourceChunk(document_title="doc", chunk_index=0)],
        )


def test_pie_chart_with_exactly_one_series_is_valid():
    chart = ChartSpec(
        chart_type="pie",
        title="Good pie",
        labels=["A", "B"],
        series=[ChartSeries(name="s1", values=[1, 2])],
        source_chunks=[ChartSourceChunk(document_title="doc", chunk_index=0)],
    )
    assert chart.chart_type == "pie"


# ---------------------------------------------------------------------------
# chart_requested wiring (Part 1 + Part 2, non-streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chart_requested_is_true_when_keyword_matches_and_chunks_present():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeChartModelProvider()
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    await orchestrator.handle(
        principal=_principal(),
        config=_make_config(),
        message="Can you show me a chart of headcount by quarter?",
    )

    assert provider.calls[0].chart_requested is True


@pytest.mark.asyncio
async def test_chart_requested_is_false_without_keyword():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeChartModelProvider()
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    await orchestrator.handle(
        principal=_principal(), config=_make_config(), message="What was the headcount in Q2?"
    )

    assert provider.calls[0].chart_requested is False


# ---------------------------------------------------------------------------
# Server-side chart validation (Part 2/5): grounded, valid source_chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_chart_with_authorized_source_chunks_passes_through():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    chart = _make_chart(
        source_chunks=[
            ChartSourceChunk(document_title=chunk.document_title, chunk_index=chunk.chunk_index)
        ]
    )
    provider = FakeChartModelProvider(chart=chart)
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=_make_config(), message="Show me a chart of headcount."
    )

    assert result.chart is not None
    assert result.chart.chart_type == "bar"
    assert result.chart.source_chunks == chart.source_chunks


@pytest.mark.asyncio
async def test_chart_citing_a_chunk_not_in_this_turns_retrieval_is_discarded():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    # This document/chunk pair was never actually retrieved this turn.
    chart = _make_chart(
        source_chunks=[ChartSourceChunk(document_title="unrelated_doc", chunk_index=7)]
    )
    provider = FakeChartModelProvider(chart=chart)
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=_make_config(), message="Show me a chart of headcount."
    )

    assert result.chart is None
    # Text-only fallback — the rest of the turn is unaffected.
    assert result.text == "Here is the data."
    assert result.grounded is True


@pytest.mark.asyncio
async def test_chart_with_no_source_chunks_at_all_is_discarded():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    chart = _make_chart(source_chunks=[])
    provider = FakeChartModelProvider(chart=chart)
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=_make_config(), message="Show me a chart of headcount."
    )

    assert result.chart is None


@pytest.mark.asyncio
async def test_no_chart_worthy_data_means_no_chart_and_turn_is_unaffected():
    retrieval = FakeRetrievalService([_make_chunk("Leave policy text with no numbers.")])
    provider = FakeChartModelProvider(reply_text="Here is the policy.", chart=None)
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(),
        config=_make_config(),
        message="Can you chart this for me?",
    )

    assert result.chart is None
    assert result.text == "Here is the policy."
    assert result.grounded is True
    assert len(result.citations) == 1


@pytest.mark.asyncio
async def test_chart_suppressed_when_text_reply_is_a_refusal_downgrading_grounded():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    chart = _make_chart(
        source_chunks=[
            ChartSourceChunk(document_title=chunk.document_title, chunk_index=chunk.chunk_index)
        ]
    )
    provider = FakeChartModelProvider(
        reply_text="I don't have enough information to answer this question.", chart=chart
    )
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=_make_config(), message="Chart this for me."
    )

    assert result.grounded is False
    assert result.chart is None


@pytest.mark.asyncio
async def test_chart_suppressed_when_output_guardrail_rejects_the_reply():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    chart = _make_chart(
        source_chunks=[
            ChartSourceChunk(document_title=chunk.document_title, chunk_index=chunk.chunk_index)
        ]
    )
    provider = FakeChartModelProvider(
        reply_text="Here is some confidential internal-only content.", chart=chart
    )
    config = _make_config(
        guardrails=GuardrailsConfig(unsafe_output_patterns=[r"confidential internal-only"])
    )
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(), config=config, message="Chart this for me."
    )

    assert result.grounded is False
    assert result.chart is None


# ---------------------------------------------------------------------------
# Streaming: chart event ordering and absence (Part 3/5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_stream_chart_event_arrives_after_deltas_and_before_done():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    chart = _make_chart(
        source_chunks=[
            ChartSourceChunk(document_title=chunk.document_title, chunk_index=chunk.chunk_index)
        ]
    )
    provider = FakeChartStreamingModelProvider(deltas=["Here ", "is the data."], chart=chart)
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    events = [
        event
        async for event in orchestrator.handle_stream(
            principal=_principal(), config=_make_config(), message="Chart this for me."
        )
    ]

    event_types = [type(e) for e in events]
    assert event_types == [
        ChatStreamDelta,
        ChatStreamDelta,
        ChatStreamChart,
        ChatStreamDone,
    ]
    chart_event = next(e for e in events if isinstance(e, ChatStreamChart))
    assert chart_event.chart.source_chunks == chart.source_chunks


@pytest.mark.asyncio
async def test_handle_stream_no_chart_event_when_provider_returns_no_chart():
    retrieval = FakeRetrievalService([_make_chunk()])
    provider = FakeChartStreamingModelProvider(deltas=["Hello ", "world"], chart=None)
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    events = [
        event
        async for event in orchestrator.handle_stream(
            principal=_principal(), config=_make_config(), message="hi"
        )
    ]

    assert not any(isinstance(e, ChatStreamChart) for e in events)
    assert isinstance(events[-1], ChatStreamDone)


@pytest.mark.asyncio
async def test_handle_stream_no_chart_event_when_source_chunks_unauthorized():
    chunk = _make_chunk()
    retrieval = FakeRetrievalService([chunk])
    chart = _make_chart(
        source_chunks=[ChartSourceChunk(document_title="unrelated_doc", chunk_index=99)]
    )
    provider = FakeChartStreamingModelProvider(deltas=["Here ", "is the data."], chart=chart)
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    events = [
        event
        async for event in orchestrator.handle_stream(
            principal=_principal(), config=_make_config(), message="Chart this for me."
        )
    ]

    assert not any(isinstance(e, ChatStreamChart) for e in events)
    done_events = [e for e in events if isinstance(e, ChatStreamDone)]
    assert len(done_events) == 1
    assert not any(isinstance(e, ChatStreamError) for e in events)


# ---------------------------------------------------------------------------
# Chart follow-ups that reuse the previous answer's cited chunks
# ---------------------------------------------------------------------------


_REVENUE = "FY2021 42.3 | FY2022 51.8 | FY2023 63.4 | FY2024 79.1 | FY2025 96.5"
_PRIOR_CITATION = (MessageCitation(document_title="nova_horizon_fy2025", chunk_index=0),)


def _finance_chunk(*, labels: frozenset[str] = frozenset({"role:authenticated"})) -> Chunk:
    return Chunk(
        document_title="nova_horizon_fy2025",
        chunk_index=0,
        display_text=_REVENUE,
        embedded_text=f"nova_horizon_fy2025: {_REVENUE}",
        access_labels=labels,
    )


def _revenue_chart(source: tuple[str, int] = ("nova_horizon_fy2025", 0)) -> ChartSpec:
    return ChartSpec(
        chart_type="line",
        title="Revenue",
        labels=["FY2021", "FY2022", "FY2023", "FY2024", "FY2025"],
        series=[ChartSeries(name="Revenue ($M)", values=[42.3, 51.8, 63.4, 79.1, 96.5])],
        source_chunks=[ChartSourceChunk(document_title=source[0], chunk_index=source[1])],
    )


class FakeFollowUpRetrievalService:
    """search() returns ``fresh`` (nothing, for a vague follow-up).
    fetch_cited_chunks() resolves citations from ``stored`` and applies the
    same label-overlap rule the real store does, so revoked access is
    actually simulated rather than stubbed as an empty list."""

    def __init__(self, *, stored: list[Chunk], fresh: list[Chunk] | None = None):
        self.stored = {(c.document_title, c.chunk_index): c for c in stored}
        self.fresh = fresh or []
        self.search_calls: list[str] = []
        self.fetch_calls: list[dict] = []

    async def search(
        self, *, query, principal, assistant_id, top_k, min_similarity=0.0, embedder=None
    ):
        self.search_calls.append(query)
        return self.fresh

    async def fetch_cited_chunks(self, *, citations, principal, assistant_id):
        self.fetch_calls.append(
            {"citations": list(citations), "principal": principal, "assistant_id": assistant_id}
        )
        return [
            self.stored[key]
            for key in citations
            if key in self.stored and self.stored[key].access_labels & principal.labels
        ]


async def _conversation_after_grounded_answer(store, principal, *, citations=_PRIOR_CITATION):
    conversation = await store.create_conversation(principal=principal, assistant_id="hr_assistant")
    await store.append_message(
        conversation_id=conversation.id,
        principal=principal,
        role="user",
        content="what are profit and revenue in the last 5 years",
    )
    await store.append_message(
        conversation_id=conversation.id,
        principal=principal,
        role="assistant",
        content="Revenue was 42.3, 51.8, 63.4, 79.1 and 96.5.",
        citations=citations,
    )
    return conversation.id


def _cited(items) -> list[tuple[str, int]]:
    return [(c.document_title, c.chunk_index) for c in items]


@pytest.mark.asyncio
async def test_chart_follow_up_reuses_previous_answers_chunks_and_produces_a_chart():
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    retrieval = FakeFollowUpRetrievalService(stored=[_finance_chunk()])
    provider = FakeChartModelProvider(chart=_revenue_chart())
    orchestrator = _make_orchestrator(
        retrieval=retrieval, provider=provider, conversation_store=store
    )

    result = await orchestrator.handle(
        principal=principal,
        config=_make_config(),
        message="Plot a graph of these for me",
        conversation_id=conversation_id,
    )

    assert retrieval.search_calls == ["Plot a graph of these for me"]
    assert retrieval.fetch_calls[0]["citations"] == [("nova_horizon_fy2025", 0)]
    assert retrieval.fetch_calls[0]["principal"] is principal
    prompt = provider.calls[0]
    assert prompt.chart_requested is True
    assert _cited(prompt.retrieved_chunks) == [("nova_horizon_fy2025", 0)]
    assert CHART_WITHOUT_CONTEXT_GUIDANCE not in prompt.system_prompt
    assert CHART_TURN_GUIDANCE in prompt.system_prompt
    assert result.grounded is True
    assert _cited(result.citations) == [("nova_horizon_fy2025", 0)]
    assert result.chart is not None
    assert result.chart.series[0].values == [42.3, 51.8, 63.4, 79.1, 96.5]


@pytest.mark.asyncio
async def test_reused_context_still_enforces_chart_source_validation():
    # A chart citing anything outside the re-fetched set is discarded, exactly
    # as for freshly retrieved chunks.
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    orchestrator = _make_orchestrator(
        retrieval=FakeFollowUpRetrievalService(stored=[_finance_chunk()]),
        provider=FakeChartModelProvider(chart=_revenue_chart(source=("other_report", 3))),
        conversation_store=store,
    )

    result = await orchestrator.handle(
        principal=principal,
        config=_make_config(),
        message="Plot a graph of these for me",
        conversation_id=conversation_id,
    )

    assert result.grounded is True
    assert result.chart is None


@pytest.mark.asyncio
async def test_chart_follow_up_with_revoked_access_gets_no_chart_and_honest_guidance():
    # The previous answer's chunk now needs a label this principal lacks.
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    retrieval = FakeFollowUpRetrievalService(
        stored=[_finance_chunk(labels=frozenset({"dept:finance"}))]
    )
    provider = FakeChartModelProvider(
        reply_text="I need the specific figures to chart.", chart=_revenue_chart()
    )
    orchestrator = _make_orchestrator(
        retrieval=retrieval, provider=provider, conversation_store=store
    )

    result = await orchestrator.handle(
        principal=principal,
        config=_make_config(enabled_tools=["calculate"]),
        message="Plot a graph of these for me",
        conversation_id=conversation_id,
    )

    assert len(retrieval.fetch_calls) == 1
    prompt = provider.calls[0]
    assert prompt.retrieved_chunks == ()
    assert prompt.chart_requested is False
    assert CHART_WITHOUT_CONTEXT_GUIDANCE in prompt.system_prompt
    assert CHART_TURN_GUIDANCE not in prompt.system_prompt
    assert result.chart is None
    assert result.citations == ()


@pytest.mark.asyncio
async def test_chart_turn_guidance_is_added_only_when_a_chart_is_attempted():
    # Provider-neutral: fresh chart turns get it, ordinary turns never do.
    provider = FakeChartModelProvider(chart=_make_chart())
    orchestrator = _make_orchestrator(
        retrieval=FakeRetrievalService([_make_chunk()]), provider=provider
    )

    await orchestrator.handle(principal=_principal(), config=_make_config(), message="chart it")
    await orchestrator.handle(
        principal=_principal(), config=_make_config(), message="what was Q2 headcount?"
    )

    assert CHART_TURN_GUIDANCE in provider.calls[0].system_prompt
    assert CHART_TURN_GUIDANCE not in provider.calls[1].system_prompt


@pytest.mark.asyncio
async def test_revoked_access_without_tools_still_gets_the_fixed_safe_refusal():
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    provider = FakeChartModelProvider(chart=_revenue_chart())
    orchestrator = _make_orchestrator(
        retrieval=FakeFollowUpRetrievalService(
            stored=[_finance_chunk(labels=frozenset({"dept:finance"}))]
        ),
        provider=provider,
        conversation_store=store,
    )

    result = await orchestrator.handle(
        principal=principal,
        config=_make_config(),
        message="Plot a graph of these for me",
        conversation_id=conversation_id,
    )

    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY
    assert result.chart is None
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled_tools", [[], ["calculate"]])
async def test_standalone_chart_request_with_no_history_behaves_as_before(enabled_tools):
    retrieval = FakeFollowUpRetrievalService(stored=[_finance_chunk()])
    provider = FakeChartModelProvider(chart=_revenue_chart())
    orchestrator = _make_orchestrator(retrieval=retrieval, provider=provider)

    result = await orchestrator.handle(
        principal=_principal(),
        config=_make_config(enabled_tools=enabled_tools),
        message="Plot a graph of these for me",
    )

    # A brand-new conversation has no previous answer, so nothing is re-fetched.
    assert retrieval.fetch_calls == []
    assert result.chart is None
    if enabled_tools:
        assert CHART_WITHOUT_CONTEXT_GUIDANCE in provider.calls[0].system_prompt
    else:
        assert result.text == NO_AUTHORIZED_CONTEXT_REPLY
        assert provider.calls == []


@pytest.mark.asyncio
async def test_non_chart_follow_up_never_reuses_previous_context():
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    retrieval = FakeFollowUpRetrievalService(stored=[_finance_chunk()])
    orchestrator = _make_orchestrator(
        retrieval=retrieval, provider=FakeChartModelProvider(), conversation_store=store
    )

    result = await orchestrator.handle(
        principal=principal,
        config=_make_config(),
        message="and what about these?",
        conversation_id=conversation_id,
    )

    assert retrieval.fetch_calls == []
    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY


@pytest.mark.asyncio
async def test_fresh_chunks_take_precedence_over_previous_context():
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    fresh = _make_chunk()
    retrieval = FakeFollowUpRetrievalService(stored=[_finance_chunk()], fresh=[fresh])
    provider = FakeChartModelProvider(chart=_make_chart())
    orchestrator = _make_orchestrator(
        retrieval=retrieval, provider=provider, conversation_store=store
    )

    result = await orchestrator.handle(
        principal=principal,
        config=_make_config(),
        message="chart the headcount",
        conversation_id=conversation_id,
    )

    assert retrieval.fetch_calls == []
    assert provider.calls[0].retrieved_chunks == (fresh,)
    assert result.chart is not None


@pytest.mark.asyncio
async def test_only_the_most_recent_answer_is_reused():
    # The latest answer cited nothing (e.g. it was a refusal); an older
    # grounded answer must not be reached back to.
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    for role, content in [("user", "what is the weather?"), ("assistant", "Not in the documents.")]:
        await store.append_message(
            conversation_id=conversation_id, principal=principal, role=role, content=content
        )
    retrieval = FakeFollowUpRetrievalService(stored=[_finance_chunk()])
    provider = FakeChartModelProvider(chart=_revenue_chart())
    orchestrator = _make_orchestrator(
        retrieval=retrieval, provider=provider, conversation_store=store
    )

    result = await orchestrator.handle(
        principal=principal,
        config=_make_config(),
        message="Plot a graph of these for me",
        conversation_id=conversation_id,
    )

    assert retrieval.fetch_calls == []
    assert result.chart is None
    assert result.text == NO_AUTHORIZED_CONTEXT_REPLY


@pytest.mark.asyncio
async def test_handle_stream_chart_follow_up_reuses_previous_context():
    store = FakeConversationStore()
    principal = _principal()
    conversation_id = await _conversation_after_grounded_answer(store, principal)
    orchestrator = _make_orchestrator(
        retrieval=FakeFollowUpRetrievalService(stored=[_finance_chunk()]),
        provider=FakeChartStreamingModelProvider(
            deltas=["Here ", "it is."], chart=_revenue_chart()
        ),
        conversation_store=store,
    )

    events = [
        e
        async for e in orchestrator.handle_stream(
            principal=principal,
            config=_make_config(),
            message="Plot a graph of these for me",
            conversation_id=conversation_id,
        )
    ]

    charts = [e for e in events if isinstance(e, ChatStreamChart)]
    assert len(charts) == 1
    assert charts[0].chart.labels[-1] == "FY2025"
    assert isinstance(events[-1], ChatStreamDone)
