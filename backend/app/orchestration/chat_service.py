from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

from app.config.assistant_config import AssistantConfig
from app.config.settings import Settings
from app.core.principal import PrincipalContext
from app.guardrails import input as input_guardrails
from app.guardrails import output as output_guardrails
from app.guardrails import pii
from app.ingestion.pipeline import Chunk
from app.orchestration.errors import (
    AssistantAccessDeniedError,
    InputGuardrailError,
    RetrievalUnavailableError,
)
from app.orchestration.model_provider import (
    ChartSpec,
    ConversationTurn,
    GroundedPrompt,
    ModelProvider,
    ModelProviderError,
)
from app.orchestration.provider_factory import get_model_provider
from app.services.conversation_store import Conversation, ConversationStore, MessageCitation
from app.services.gemini_embedding import GeminiEmbeddingProvider
from app.services.retriever import RetrievalService
from app.services.vector_store import PgVectorStore

logger = logging.getLogger(__name__)

NO_AUTHORIZED_CONTEXT_REPLY = (
    "I don't have enough information in the documents I'm authorized to "
    "access to answer that question. Please rephrase, or check with someone "
    "who has access to the relevant material."
)

OUTPUT_GUARDRAIL_FALLBACK_REPLY = (
    "I can't share that response as generated. Please rephrase your "
    "question, or contact support if you believe this is an error."
)


@dataclass(frozen=True)
class ChunkCitation:
    document_title: str
    chunk_index: int


@dataclass(frozen=True)
class ChatResult:
    text: str
    conversation_id: uuid.UUID
    citations: tuple[ChunkCitation, ...] = ()
    grounded: bool = False
    chart: ChartSpec | None = None


@dataclass(frozen=True)
class ChatStreamDelta:
    text: str


@dataclass(frozen=True)
class ChatStreamChart:
    """Sent at most once per turn, only when a chart survived
    ChatOrchestrator._validate_chart — never an empty/null chart event.
    Ordering: after the last ChatStreamDelta, before ChatStreamDone (see
    handle_stream) — both are derived from the same final provider stream
    event, so no extra per-turn delay is introduced beyond whatever the
    provider itself took to (optionally) generate the chart before yielding
    that final event."""

    chart: ChartSpec


@dataclass(frozen=True)
class ChatStreamDone:
    conversation_id: uuid.UUID
    citations: tuple[ChunkCitation, ...] = ()
    grounded: bool = False


@dataclass(frozen=True)
class ChatStreamError:
    detail: str


ChatStreamEvent = ChatStreamDelta | ChatStreamChart | ChatStreamDone | ChatStreamError


def _format_retrieved_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(
        f"[source: {chunk.document_title}, chunk {chunk.chunk_index}]\n{chunk.display_text}"
        for chunk in chunks
    )


def build_grounded_user_turn(*, user_question: str, retrieved_chunks: Sequence[Chunk]) -> str:
    """Compose the user turn with retrieved context clearly separated from the
    question, per the framework's grounded-prompt convention:

        RETRIEVED CONTEXT:
        [source metadata]
        <authorized retrieved content>

        USER:
        <user question>

    The configured system prompt is passed separately (as the model's system
    instruction) — never folded into this text.
    """
    return (
        "RETRIEVED CONTEXT:\n"
        f"{_format_retrieved_context(retrieved_chunks)}\n\n"
        "USER:\n"
        f"{user_question}"
    )


ProviderFactory = Callable[[AssistantConfig], ModelProvider]


def _message_requests_chart(message: str, patterns: Sequence[str]) -> bool:
    """Deterministic, cheap chart-intent gate — a small keyword/pattern set,
    not a classifier or a second model call. False means chart generation
    is skipped entirely for this turn: zero extra provider cost/latency on
    ordinary turns. ``patterns`` is per-assistant tunable
    (AssistantConfig.chart_trigger_patterns) rather than hardcoded here.
    """
    return any(re.search(pattern, message, re.IGNORECASE) for pattern in patterns)


@dataclass(frozen=True)
class _PreparedTurn:
    """Shared setup result for both handle() and handle_stream(): the
    resolved/created conversation, and either a fully-assembled prompt ready
    for the model, or ``prompt=None`` meaning "no authorized/relevant/safe
    context was found — send the safe reply, never call the model." That
    ``None`` case now covers three causes uniformly: retrieval found nothing,
    everything it found scored below the relevance threshold, or everything
    it found was stripped by prompt-injection screening — all three are the
    same outcome from the caller's point of view, so they share one path
    rather than three subtly different messages.
    """

    conversation: Conversation
    prompt: GroundedPrompt | None
    chunks: tuple[Chunk, ...] = ()


@dataclass(frozen=True)
class _GuardedOutput:
    """Result of applying output guardrails to a fully-assembled model reply.

    ``rejected=True`` means the reply was replaced outright (PII/unsafe
    content) — callers must never persist or present the original text, and
    on the streaming path must not send a normal "done" event. A citation
    downgrade (the model refused despite grounded=True) is NOT a rejection:
    ``text`` is untouched, only ``grounded``/``citations`` change.
    """

    text: str
    grounded: bool
    citations: tuple[Chunk, ...]
    rejected: bool = False


class ChatOrchestrator:
    """Router/orchestration layer: wires an authenticated principal, an
    assistant configuration, a persisted conversation, the existing
    ACL-aware retriever, input/output guardrails, and a provider-neutral
    model call into one conversational turn — either as a single blocking
    result (``handle``) or as a stream of deltas (``handle_stream``). Both
    share the same clearance/conversation/retrieval/input-guardrail setup
    via ``_prepare_turn``, the same output-guardrail pass via
    ``_apply_output_guardrails``, and the same persistence via
    ``_persist_turn``, so none of these rules can drift between the two
    paths.

    Owns no vendor-specific code — that stays behind ``ModelProvider``.
    """

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        provider_factory: ProviderFactory,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._provider_factory = provider_factory
        self._conversation_store = conversation_store or ConversationStore()

    async def handle(
        self,
        *,
        principal: PrincipalContext,
        config: AssistantConfig,
        message: str,
        conversation_id: uuid.UUID | None = None,
    ) -> ChatResult:
        setup = await self._prepare_turn(
            principal=principal,
            config=config,
            message=message,
            conversation_id=conversation_id,
        )

        if setup.prompt is None:
            # No authorized/relevant/safe content was found — never fall
            # back to an ungrounded model call, and never widen the search.
            # Still a real turn in this conversation, so it's still persisted.
            await self._persist_turn(
                conversation_id=setup.conversation.id,
                principal=principal,
                user_message=message,
                reply_text=NO_AUTHORIZED_CONTEXT_REPLY,
                citations=(),
            )
            return ChatResult(
                text=NO_AUTHORIZED_CONTEXT_REPLY,
                conversation_id=setup.conversation.id,
                citations=(),
                grounded=False,
            )

        provider = self._provider_factory(config)
        reply = await provider.generate(setup.prompt)

        guarded = self._apply_output_guardrails(
            config=config,
            reply_text=reply.text,
            model_grounded=reply.grounded,
            chunks=setup.chunks,
        )

        chart = self._validate_chart(
            chart=reply.chart, grounded=guarded.grounded, chunks=setup.chunks
        )

        citations = tuple(
            ChunkCitation(document_title=chunk.document_title, chunk_index=chunk.chunk_index)
            for chunk in guarded.citations
        )

        await self._persist_turn(
            conversation_id=setup.conversation.id,
            principal=principal,
            user_message=message,
            reply_text=guarded.text,
            citations=citations,
        )

        return ChatResult(
            text=guarded.text,
            conversation_id=setup.conversation.id,
            citations=citations,
            grounded=guarded.grounded,
            chart=chart,
        )

    async def handle_stream(
        self,
        *,
        principal: PrincipalContext,
        config: AssistantConfig,
        message: str,
        conversation_id: uuid.UUID | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Streaming counterpart to ``handle``.

        Output guardrails need the FULL assembled reply, so they only run
        once the provider's final stream event arrives — deltas render to
        the client in real time exactly as they're produced, unaffected.
        If the fully-assembled reply then gets rejected outright (PII/unsafe
        content — see _apply_output_guardrails/_GuardedOutput.rejected), the
        raw text has already been streamed as deltas and cannot be un-sent;
        the best this can do is refuse to call it a success: a final
        {"type":"error"} event is sent instead of {"type":"done"}, and
        nothing is persisted. A citation downgrade (the model refused
        despite grounded=True) is not a rejection and still ends in a normal
        "done" event with corrected grounded/citations.

        Persistence policy on early termination (client disconnect, a
        provider error before the final event arrives, or an output
        guardrail rejection): persist NOTHING for this turn. The complete
        user message and assistant reply are only written once they've
        fully passed every check — never a partial or rejected one — so a
        cut-off/rejected stream simply leaves no trace of that turn rather
        than risking a reply that looks complete but isn't.
        """
        setup = await self._prepare_turn(
            principal=principal,
            config=config,
            message=message,
            conversation_id=conversation_id,
        )

        if setup.prompt is None:
            # The safe reply is fully known upfront (never partial), so
            # persisting it before yielding is safe — this isn't the
            # "don't persist partial replies" case at all.
            await self._persist_turn(
                conversation_id=setup.conversation.id,
                principal=principal,
                user_message=message,
                reply_text=NO_AUTHORIZED_CONTEXT_REPLY,
                citations=(),
            )
            yield ChatStreamDelta(text=NO_AUTHORIZED_CONTEXT_REPLY)
            yield ChatStreamDone(
                conversation_id=setup.conversation.id, citations=(), grounded=False
            )
            return

        provider = self._provider_factory(config)

        final_text: str | None = None
        final_grounded = False
        final_chart: ChartSpec | None = None

        try:
            async for stream_event in provider.generate_stream(setup.prompt):
                if stream_event.is_final:
                    final_text = stream_event.text
                    final_grounded = stream_event.grounded
                    final_chart = stream_event.chart
                elif stream_event.delta:
                    yield ChatStreamDelta(text=stream_event.delta)
        except ModelProviderError:
            # Nothing persisted — see the persistence policy above.
            yield ChatStreamError(detail="the assistant model is temporarily unavailable")
            return

        if final_text is None:
            yield ChatStreamError(detail="the assistant model is temporarily unavailable")
            return

        guarded = self._apply_output_guardrails(
            config=config,
            reply_text=final_text,
            model_grounded=final_grounded,
            chunks=setup.chunks,
        )

        if guarded.rejected:
            yield ChatStreamError(detail="the assistant's response could not be delivered")
            return

        chart = self._validate_chart(
            chart=final_chart, grounded=guarded.grounded, chunks=setup.chunks
        )

        citations = tuple(
            ChunkCitation(document_title=chunk.document_title, chunk_index=chunk.chunk_index)
            for chunk in guarded.citations
        )

        await self._persist_turn(
            conversation_id=setup.conversation.id,
            principal=principal,
            user_message=message,
            reply_text=guarded.text,
            citations=citations,
        )

        if chart is not None:
            yield ChatStreamChart(chart=chart)

        yield ChatStreamDone(
            conversation_id=setup.conversation.id,
            citations=citations,
            grounded=guarded.grounded,
        )

    async def _prepare_turn(
        self,
        *,
        principal: PrincipalContext,
        config: AssistantConfig,
        message: str,
        conversation_id: uuid.UUID | None,
    ) -> _PreparedTurn:
        """Input guardrails, clearance check, conversation resolution, and
        guarded retrieval — the setup shared by handle() and handle_stream().

        Raises InputGuardrailError / AssistantAccessDeniedError /
        ConversationNotFoundError / RetrievalUnavailableError exactly as
        handle() did before guardrails/streaming were factored out; callers
        on the streaming path must let these propagate before yielding
        anything, so the route can still map them to a real HTTP status code
        instead of an in-band error event.
        """
        # Input guardrails run first and cheaply, before any DB work
        # (conversation creation) or retrieval — fail fast on malformed
        # input regardless of what else this turn would have done.
        input_guardrails.validate_input_message(
            message, max_input_chars=config.guardrails.max_input_chars
        )

        input_pii_hits = pii.detect_pii(message)
        if input_pii_hits:
            logger.warning(
                "input message for assistant '%s' matched PII pattern(s): %s",
                config.assistant_id,
                input_pii_hits,
            )
            if config.guardrails.input_pii_block:
                raise InputGuardrailError(
                    "message appears to contain personal information and cannot be processed"
                )

        if principal.clearance < config.min_clearance:
            raise AssistantAccessDeniedError(
                f"principal clearance {principal.clearance} is below assistant "
                f"minimum clearance {config.min_clearance}"
            )

        if conversation_id is not None:
            # Raises ConversationNotFoundError (never a generic 500) if this
            # conversation doesn't exist or belongs to someone else — the
            # same ownership discipline as retrieval's label checks, applied
            # to conversations instead of documents.
            conversation = await self._conversation_store.get_conversation(
                conversation_id=conversation_id, principal=principal
            )
            prior_messages = await self._conversation_store.list_messages(
                conversation_id=conversation.id, principal=principal
            )
        else:
            conversation = await self._conversation_store.create_conversation(
                principal=principal, assistant_id=config.assistant_id
            )
            prior_messages = []

        prior_turns = tuple(
            ConversationTurn(role=m.role, content=m.content) for m in prior_messages
        )

        retrieval_enabled = config.retrieval is not None and config.retrieval.enabled
        chunks: list[Chunk] = []

        if retrieval_enabled:
            assert config.retrieval is not None
            try:
                raw_chunks = await self._retrieval_service.search(
                    query=message,
                    principal=principal,
                    assistant_id=config.assistant_id,
                    top_k=config.retrieval.top_k,
                    min_similarity=config.retrieval.min_similarity,
                )
            except Exception as exc:  # never leak DB/connection details upward
                raise RetrievalUnavailableError("retrieval backend is unavailable") from exc

            chunks = input_guardrails.screen_retrieved_chunks(raw_chunks)

            if not chunks:
                return _PreparedTurn(conversation=conversation, prompt=None, chunks=())

            user_turn = build_grounded_user_turn(user_question=message, retrieved_chunks=chunks)
            chart_requested = _message_requests_chart(message, config.chart_trigger_patterns)
        else:
            user_turn = message
            chart_requested = False

        prompt = GroundedPrompt(
            system_prompt=config.system_prompt,
            user_message=user_turn,
            retrieved_chunks=tuple(chunks),
            prior_turns=prior_turns,
            chart_requested=chart_requested,
        )
        return _PreparedTurn(conversation=conversation, prompt=prompt, chunks=tuple(chunks))

    def _apply_output_guardrails(
        self,
        *,
        config: AssistantConfig,
        reply_text: str,
        model_grounded: bool,
        chunks: tuple[Chunk, ...],
    ) -> _GuardedOutput:
        """Applied to the FULL assembled reply, after the model call,
        before persistence or the response reaches the caller.

        ``model_grounded`` is the input-side signal from the provider
        (relevant, safe context was retrieved and included in the prompt —
        see ModelProvider/GeminiProvider). This method's citation-downgrade
        step is the output-side backstop the LLD calls for: the model can
        still legitimately refuse to answer even from relevant context, and
        a refusal must never carry citations/grounded=True just because
        *something* relevant was retrieved.
        """
        text = reply_text
        grounded = model_grounded
        cite_chunks = chunks if grounded else ()

        # Citation-required-for-knowledge-answers backstop: a downgrade of
        # metadata only, never a rejection — the model's own text is exactly
        # what the caller should see either way.
        if grounded and output_guardrails.reply_appears_to_refuse(text):
            grounded = False
            cite_chunks = ()

        output_pii_hits = pii.detect_pii(text)
        if output_pii_hits:
            logger.warning(
                "model output for assistant '%s' matched PII pattern(s): %s",
                config.assistant_id,
                output_pii_hits,
            )
            if config.guardrails.output_pii_block:
                return _GuardedOutput(
                    text=OUTPUT_GUARDRAIL_FALLBACK_REPLY,
                    grounded=False,
                    citations=(),
                    rejected=True,
                )

        if output_guardrails.check_unsafe_output(
            text, patterns=config.guardrails.unsafe_output_patterns
        ):
            logger.warning(
                "model output for assistant '%s' matched a configured unsafe-content pattern",
                config.assistant_id,
            )
            return _GuardedOutput(
                text=OUTPUT_GUARDRAIL_FALLBACK_REPLY, grounded=False, citations=(), rejected=True
            )

        return _GuardedOutput(text=text, grounded=grounded, citations=cite_chunks, rejected=False)

    def _validate_chart(
        self,
        *,
        chart: ChartSpec | None,
        grounded: bool,
        chunks: tuple[Chunk, ...],
    ) -> ChartSpec | None:
        """Server-side backstop for a provider-produced chart — never trust
        the provider's own citation claim, the same discipline
        _apply_output_guardrails already applies to text citations, plus one
        rule stronger: a chart reads as more authoritative than prose, so a
        fabricated chart is worse than no chart at all.

        Two independent reasons a chart is discarded (falls back to
        text-only, never a turn failure):
          - the turn itself isn't grounded (retrieval found nothing relevant/
            authorized, or the output guardrails downgraded/rejected the
            text reply) — a chart is never shown on an ungrounded turn.
          - any of the chart's source_chunks isn't a member of THIS turn's
            actual authorized, retrieved chunk set — catches a provider that
            hallucinated a plausible-looking citation or recalled one from
            unrelated context, not just a malformed response.
        """
        if chart is None:
            return None
        if not grounded:
            logger.warning("discarding a generated chart: turn is not grounded")
            return None

        authorized = {(c.document_title, c.chunk_index) for c in chunks}
        chart_sources = {(sc.document_title, sc.chunk_index) for sc in chart.source_chunks}
        if not chart_sources or not chart_sources.issubset(authorized):
            logger.warning(
                "discarding a generated chart: source_chunks were not a non-empty "
                "subset of this turn's authorized retrieved chunks"
            )
            return None

        return chart

    async def _persist_turn(
        self,
        *,
        conversation_id: uuid.UUID,
        principal: PrincipalContext,
        user_message: str,
        reply_text: str,
        citations: tuple[ChunkCitation, ...],
    ) -> None:
        await self._conversation_store.append_message(
            conversation_id=conversation_id,
            principal=principal,
            role="user",
            content=user_message,
        )
        await self._conversation_store.append_message(
            conversation_id=conversation_id,
            principal=principal,
            role="assistant",
            content=reply_text,
            citations=[
                MessageCitation(document_title=c.document_title, chunk_index=c.chunk_index)
                for c in citations
            ],
        )


def build_default_chat_orchestrator(*, settings: Settings) -> ChatOrchestrator:
    """Construct the orchestrator using the framework's real embedding
    provider and vector-store implementation, resolving each assistant's
    model provider from its own configuration at call time.
    """
    embedder = GeminiEmbeddingProvider(api_key=settings.gemini_api_key)
    vector_store = PgVectorStore()
    retrieval_service = RetrievalService(embedder=embedder, vector_store=vector_store)

    def provider_factory(config: AssistantConfig) -> ModelProvider:
        return get_model_provider(model=config.model, settings=settings)

    return ChatOrchestrator(retrieval_service=retrieval_service, provider_factory=provider_factory)
