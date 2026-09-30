from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace

from app.config.assistant_config import AssistantConfig
from app.config.settings import Settings
from app.core.principal import PrincipalContext
from app.guardrails import input as input_guardrails
from app.guardrails import output as output_guardrails
from app.guardrails import pii
from app.ingestion.pipeline import Chunk
from app.observability import audit
from app.observability.audit import AuditStore, TurnRecorder, record_guardrail
from app.observability.context import (
    assistant_id_var,
    conversation_id_var,
    principal_ref,
    principal_ref_var,
)
from app.observability.provider_errors import find_failure, log_provider_failure
from app.observability.tracing import set_attributes, start_span, tracer
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
from app.services.conversation_store import (
    Conversation,
    ConversationNotFoundError,
    ConversationStore,
    Message,
    MessageCitation,
)
from app.services.embedding import EmbeddingProvider
from app.services.embedding_provider_factory import get_embedding_provider
from app.services.gemini_embedding import GeminiEmbeddingProvider
from app.services.retriever import RetrievalService
from app.services.vector_store import PgVectorStore

logger = logging.getLogger(__name__)

NO_AUTHORIZED_CONTEXT_REPLY = (
    "I don't have enough information in the documents I'm authorized to "
    "access to answer that question. Please rephrase, or check with someone "
    "who has access to the relevant material."
)

# Appended to system_prompt (never substituted for it) only for a turn with
# no document context at all but at least one tool configured — see
# ChatOrchestrator._prepare_turn. The hard gate lets this turn reach the
# model instead of refusing outright, so this instruction is what actually
# prevents a fabricated/document-flavored answer: the model must still
# decline, on its own, when no available tool genuinely answers the
# question, exactly the same "state clearly when you can't answer rather
# than fabricate" principle the rest of this project already applies.
NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE = (
    "No document context was retrieved for this turn — you have no "
    "authorized document content to draw on right now. Do not answer as "
    "though you do, and never state or imply that an answer is backed by "
    "the documents or a report. Only answer if one of your available tools "
    "can directly answer the user's question on its own. If no available "
    "tool applies, say clearly that you don't have enough information to "
    "answer rather than guessing, estimating, or fabricating a response."
)

# Appended, for every provider, on a turn where a chart will be attempted
# (chart_requested with document context). Without it a model tends to reply
# "I cannot plot graphs" right above the chart the application draws from the
# provider's separate extraction call. Worded so the text still reads
# correctly if that extraction ends up producing no chart.
CHART_TURN_GUIDANCE = (
    "The application renders any chart itself from the figures in the "
    "retrieved content. Do not say you cannot create, display, or generate "
    "charts or graphs; answer with the relevant figures from the retrieved "
    "content."
)

# Appended, for every provider, when the user asked for a chart but this turn
# has no document content to build one from (nothing retrieved, and no
# still-authorized content from the previous answer to reuse). Without it the
# model tends to claim it "cannot plot graphs", which is false: the
# application draws charts itself when it has the figures.
CHART_WITHOUT_CONTEXT_GUIDANCE = (
    "The user asked for a chart or graph, but no document content is "
    "available for this request, so no chart can be drawn this time. Do not "
    "say that you are unable to create charts or graphs in general. Say "
    "instead that you need the specific figures or topic named in the "
    "request (for example, which metric and which years) to build a chart "
    "from the documents."
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
EmbeddingProviderFactory = Callable[[AssistantConfig], EmbeddingProvider]


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
    context was found, AND no tool could possibly help either — send the
    safe reply, never call the model." That ``None`` case covers three
    zero-chunk causes uniformly (retrieval found nothing, everything it
    found scored below the relevance threshold, or everything it found was
    stripped by prompt-injection screening — all three are the same outcome
    from the caller's point of view) AND requires config.enabled_tools to
    also be empty. When zero chunks coincide with a non-empty
    enabled_tools, ``prompt`` is still assembled (with an empty
    retrieved_chunks and an added system-prompt instruction — see
    NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE) so the model gets a chance at a
    tool-only answer instead of a hard refusal.
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
        embedding_provider_factory: EmbeddingProviderFactory | None = None,
        audit_store: AuditStore | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._provider_factory = provider_factory
        self._conversation_store = conversation_store or ConversationStore()
        # None disables the per-turn audit row (unit tests stay DB-free);
        # build_default_chat_orchestrator passes the real store.
        self._audit_store = audit_store
        # None (the default — every existing caller/test that doesn't pass
        # this) preserves the exact prior behavior: RetrievalService.search()
        # falls back to its own construction-time embedder untouched. Only
        # build_default_chat_orchestrator wires a real factory, so a config
        # can opt into a non-Gemini embedding_provider (see
        # app/services/embedding_provider_factory.py).
        self._embedding_provider_factory = embedding_provider_factory

    # --- per-turn observability ------------------------------------------

    def _begin_turn(
        self,
        *,
        operation: str,
        principal: PrincipalContext,
        config: AssistantConfig,
        conversation_id: uuid.UUID | None,
    ) -> TurnRecorder:
        """Binds this turn's identifiers and audit recorder to the current
        context. Deliberately never reset: on the streaming path the rest of
        the turn runs in a different task, where resetting a token raises."""
        ref = principal_ref(principal.tenant_id, principal.principal_id)
        assistant_id_var.set(config.assistant_id)
        principal_ref_var.set(ref)
        conversation_id_var.set(str(conversation_id) if conversation_id else None)
        recorder = TurnRecorder(
            operation=operation,
            tenant_id=principal.tenant_id,
            assistant_id=config.assistant_id,
            principal_ref=ref,
            provider=config.model.provider,
            model_name=config.model.model_name,
            conversation_id=conversation_id,
        )
        audit.bind(recorder)
        return recorder

    def _note_failure(self, recorder: TurnRecorder, exc: BaseException) -> None:
        """The single place a failed turn is logged: one classified line for
        provider failures (traceback at DEBUG), one INFO line for expected
        rejections, and a full traceback only for genuinely unexpected
        errors."""
        if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
            recorder.outcome = "cancelled"
            logger.info("turn cancelled (client disconnected)", extra={"event": "cancelled"})
            return
        failure = find_failure(exc)
        if failure is not None:
            recorder.failure, recorder.outcome = failure, "error"
            log_provider_failure(logger, failure, exc)
            return
        expected = {
            InputGuardrailError: "input_rejected",
            AssistantAccessDeniedError: "access_denied",
            ConversationNotFoundError: "conversation_not_found",
        }
        for error_type, outcome in expected.items():
            if isinstance(exc, error_type):
                recorder.outcome, recorder.error_class = outcome, error_type.__name__
                logger.info("turn not processed: %s (%s)", outcome, exc, extra={"event": outcome})
                return
        recorder.outcome, recorder.error_class = "error", type(exc).__name__
        if isinstance(exc, RetrievalUnavailableError):
            cause = exc.__cause__
            logger.error(
                "retrieval unavailable: %s: %s", type(cause).__name__, cause,
                extra={"event": "retrieval_unavailable"},
            )
            logger.debug("retrieval failure traceback", exc_info=exc)
            return
        logger.exception("unexpected error in chat turn", exc_info=exc)

    async def _end_turn(self, recorder: TurnRecorder) -> None:
        if self._audit_store is not None:
            await self._audit_store.write(recorder)

    # --- turns ------------------------------------------------------------

    async def handle(
        self,
        *,
        principal: PrincipalContext,
        config: AssistantConfig,
        message: str,
        conversation_id: uuid.UUID | None = None,
    ) -> ChatResult:
        recorder = self._begin_turn(
            operation="chat", principal=principal, config=config, conversation_id=conversation_id
        )
        try:
            with start_span(
                "chat.turn",
                assistant_id=config.assistant_id,
                provider=config.model.provider,
                model=config.model.model_name,
                streaming=False,
            ) as span:
                result = await self._handle_turn(
                    principal=principal,
                    config=config,
                    message=message,
                    conversation_id=conversation_id,
                )
                recorder.grounded = result.grounded
                recorder.citations_count = len(result.citations)
                recorder.chart_returned = result.chart is not None
                recorder.conversation_id = result.conversation_id
                set_attributes(
                    span,
                    {"outcome": recorder.outcome, "grounded": result.grounded,
                     "citations": len(result.citations)},
                )
                return result
        except BaseException as exc:
            self._note_failure(recorder, exc)
            raise
        finally:
            await self._end_turn(recorder)

    async def _handle_turn(
        self,
        *,
        principal: PrincipalContext,
        config: AssistantConfig,
        message: str,
        conversation_id: uuid.UUID | None,
    ) -> ChatResult:
        recorder = audit.current()
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
            if recorder is not None:
                recorder.outcome = "no_context"
            return ChatResult(
                text=NO_AUTHORIZED_CONTEXT_REPLY,
                conversation_id=setup.conversation.id,
                citations=(),
                grounded=False,
            )

        provider = self._provider_factory(config)
        with start_span(
            "model.generate",
            provider=config.model.provider,
            model=config.model.model_name,
            chart_requested=setup.prompt.chart_requested,
            tools=list(setup.prompt.enabled_tools),
        ):
            reply = await provider.generate(setup.prompt)

        guarded = self._apply_output_guardrails(
            config=config,
            reply_text=reply.text,
            model_grounded=reply.grounded,
            chunks=setup.chunks,
        )
        if recorder is not None:
            recorder.outcome = "output_rejected" if guarded.rejected else "answered"

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

        Observability: the route pulls the first event in the request task
        and Starlette iterates the rest in another task, so a span entered
        here and exited after a ``yield`` would detach in the wrong context.
        The turn span is therefore created unentered and activated with
        use_span only around awaited sections that contain no ``yield``.
        """
        recorder = self._begin_turn(
            operation="chat_stream",
            principal=principal,
            config=config,
            conversation_id=conversation_id,
        )
        turn_span = tracer.start_span(
            "chat.turn",
            attributes={
                "assistant_id": config.assistant_id,
                "provider": config.model.provider,
                "model": config.model.model_name,
                "streaming": True,
            },
        )
        try:
            async for event in self._handle_stream_turn(
                principal=principal,
                config=config,
                message=message,
                conversation_id=conversation_id,
                recorder=recorder,
                turn_span=turn_span,
            ):
                if isinstance(event, ChatStreamDone):
                    recorder.grounded = event.grounded
                    recorder.citations_count = len(event.citations)
                    recorder.conversation_id = event.conversation_id
                elif isinstance(event, ChatStreamChart):
                    recorder.chart_returned = True
                yield event
        except BaseException as exc:
            self._note_failure(recorder, exc)
            turn_span.record_exception(exc)
            raise
        finally:
            set_attributes(turn_span, {"outcome": recorder.outcome, "grounded": recorder.grounded})
            turn_span.end()
            await self._end_turn(recorder)

    async def _handle_stream_turn(
        self,
        *,
        principal: PrincipalContext,
        config: AssistantConfig,
        message: str,
        conversation_id: uuid.UUID | None,
        recorder: TurnRecorder,
        turn_span: Any,
    ) -> AsyncIterator[ChatStreamEvent]:
        with trace.use_span(turn_span, end_on_exit=False):
            setup = await self._prepare_turn(
                principal=principal,
                config=config,
                message=message,
                conversation_id=conversation_id,
            )

        if setup.prompt is None:
            recorder.outcome = "no_context"
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

        model_span = tracer.start_span(
            "model.generate",
            context=trace.set_span_in_context(turn_span),
            attributes={
                "provider": config.model.provider,
                "model": config.model.model_name,
                "chart_requested": setup.prompt.chart_requested,
                "streaming": True,
            },
        )
        stream = provider.generate_stream(setup.prompt)
        try:
            while True:
                with trace.use_span(model_span, end_on_exit=False):
                    try:
                        stream_event = await stream.__anext__()
                    except StopAsyncIteration:
                        break
                if stream_event.is_final:
                    final_text = stream_event.text
                    final_grounded = stream_event.grounded
                    final_chart = stream_event.chart
                elif stream_event.delta:
                    yield ChatStreamDelta(text=stream_event.delta)
        except ModelProviderError as exc:
            # Nothing persisted — see the persistence policy above. The
            # failure is logged and audited here because it never reaches
            # the route: it becomes an in-band error event instead.
            self._note_failure(recorder, exc)
            model_span.record_exception(exc)
            yield ChatStreamError(detail="the assistant model is temporarily unavailable")
            return
        finally:
            model_span.end()

        if final_text is None:
            recorder.outcome, recorder.error_class = "error", "EmptyStream"
            yield ChatStreamError(detail="the assistant model is temporarily unavailable")
            return

        guarded = self._apply_output_guardrails(
            config=config,
            reply_text=final_text,
            model_grounded=final_grounded,
            chunks=setup.chunks,
        )

        if guarded.rejected:
            recorder.outcome = "output_rejected"
            yield ChatStreamError(detail="the assistant's response could not be delivered")
            return
        recorder.outcome = "answered"

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
            record_guardrail(
                "input_pii",
                patterns=list(input_pii_hits),
                blocked=config.guardrails.input_pii_block,
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

        conversation_id_var.set(str(conversation.id))
        recorder = audit.current()
        if recorder is not None:
            recorder.conversation_id = conversation.id

        prior_turns = tuple(
            ConversationTurn(role=m.role, content=m.content) for m in prior_messages
        )

        retrieval_enabled = config.retrieval is not None and config.retrieval.enabled
        chunks: list[Chunk] = []
        chart_trigger_matched = _message_requests_chart(message, config.chart_trigger_patterns)

        if retrieval_enabled:
            assert config.retrieval is not None
            embedder = (
                self._embedding_provider_factory(config)
                if self._embedding_provider_factory is not None
                else None
            )
            try:
                raw_chunks = await self._retrieval_service.search(
                    query=message,
                    principal=principal,
                    assistant_id=config.assistant_id,
                    top_k=config.retrieval.top_k,
                    min_similarity=config.retrieval.min_similarity,
                    embedder=embedder,
                )
            except Exception as exc:  # never leak DB/connection details upward
                raise RetrievalUnavailableError("retrieval backend is unavailable") from exc

            chunks = input_guardrails.screen_retrieved_chunks(raw_chunks)
            if len(chunks) < len(raw_chunks):
                record_guardrail("chunk_injection_dropped", dropped=len(raw_chunks) - len(chunks))

            if not chunks and chart_trigger_matched:
                # A follow-up like "plot a graph of these" names no topic, so
                # its own retrieval finds nothing. Reuse what the previous
                # answer was grounded on, re-checked against current access.
                chunks = await self._reuse_previous_answer_context(
                    prior_messages=prior_messages, principal=principal, config=config
                )

            if not chunks and not config.enabled_tools:
                # No relevant/authorized document content AND no tools that
                # could possibly help instead — this is the ONLY remaining
                # case that hits the hard safe-refusal gate. When
                # enabled_tools is non-empty we fall through instead of
                # returning here, so the model gets a chance at a tool-only
                # answer (see NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE below for
                # what stops it from fabricating a document-backed one).
                # Regression guard: an assistant with retrieval on but no
                # tools configured (e.g. hr_assistant) always takes this
                # branch on zero chunks, exactly as before this change.
                return _PreparedTurn(conversation=conversation, prompt=None, chunks=())

        if chunks:
            user_turn = build_grounded_user_turn(user_question=message, retrieved_chunks=chunks)
            chart_requested = chart_trigger_matched
        else:
            # Either retrieval is off, or it ran and found nothing relevant/
            # authorized but this assistant has tools configured (see the
            # gate above) — either way there is no document context to
            # present, so send the plain question with no "RETRIEVED
            # CONTEXT:" section at all rather than an empty/confusing one.
            user_turn = message
            chart_requested = False

        system_prompt = config.system_prompt
        if not chunks and config.enabled_tools:
            # Tools are this turn's only possible route to a real answer —
            # the hard gate no longer protects this case, so the model must
            # be told explicitly not to fabricate a document-backed answer,
            # and to decline on its own if no available tool actually
            # applies to this question either.
            system_prompt = f"{system_prompt}\n\n{NO_DOCUMENT_CONTEXT_TOOL_GUIDANCE}"
            if chart_trigger_matched:
                system_prompt = f"{system_prompt}\n\n{CHART_WITHOUT_CONTEXT_GUIDANCE}"
        if chart_requested:
            system_prompt = f"{system_prompt}\n\n{CHART_TURN_GUIDANCE}"

        prompt = GroundedPrompt(
            system_prompt=system_prompt,
            user_message=user_turn,
            retrieved_chunks=tuple(chunks),
            prior_turns=prior_turns,
            chart_requested=chart_requested,
            enabled_tools=tuple(config.enabled_tools),
            max_tool_calls=config.max_tool_calls,
        )
        return _PreparedTurn(conversation=conversation, prompt=prompt, chunks=tuple(chunks))

    async def _reuse_previous_answer_context(
        self,
        *,
        prior_messages: Sequence[Message],
        principal: PrincipalContext,
        config: AssistantConfig,
    ) -> list[Chunk]:
        """Chunks the most recent assistant turn cited, re-fetched and
        re-authorized for this principal now, then screened like fresh
        retrieval. Only that one turn is considered: an older answer is not a
        safe guess for what "these" refers to. Returns [] when there is no
        such turn, it cited nothing, or nothing it cited is still readable."""
        previous_answer = next((m for m in reversed(prior_messages) if m.role == "assistant"), None)
        if previous_answer is None or not previous_answer.citations:
            return []

        citations = [(c.document_title, c.chunk_index) for c in previous_answer.citations]
        try:
            refetched = await self._retrieval_service.fetch_cited_chunks(
                citations=citations, principal=principal, assistant_id=config.assistant_id
            )
        except Exception as exc:  # never leak DB/connection details upward
            raise RetrievalUnavailableError("retrieval backend is unavailable") from exc

        chunks = input_guardrails.screen_retrieved_chunks(refetched)
        if len(chunks) < len(refetched):
            record_guardrail("chunk_injection_dropped", dropped=len(refetched) - len(chunks))
        recorder = audit.current()
        if recorder is not None:
            recorder.retrieval["reused_previous_answer"] = {
                "cited": len(citations),
                "reused": len(chunks),
            }
        logger.info(
            "chart follow-up for '%s': reused %d of %d chunks cited by the previous answer",
            config.assistant_id,
            len(chunks),
            len(citations),
        )
        return chunks

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
            record_guardrail("citation_downgrade", reason="reply_appears_to_refuse")
            logger.info(
                "output guardrail: reply reads as a refusal; grounded=false, citations dropped",
                extra={"event": "citation_downgrade"},
            )

        output_pii_hits = pii.detect_pii(text)
        if output_pii_hits:
            logger.warning(
                "model output for assistant '%s' matched PII pattern(s): %s",
                config.assistant_id,
                output_pii_hits,
            )
            record_guardrail(
                "output_pii",
                patterns=list(output_pii_hits),
                blocked=config.guardrails.output_pii_block,
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
            record_guardrail("unsafe_output", blocked=True)
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
    model provider — and, independently, each assistant's embedding
    provider (see RetrievalConfig.embedding_provider) — from its own
    configuration at call time.
    """
    # Gemini stays RetrievalService's own construction-time default — every
    # assistant whose config doesn't set retrieval.embedding_provider (i.e.
    # every assistant that predates this field) resolves to this exact same
    # GeminiEmbeddingProvider via embedding_provider_factory below anyway,
    # so behavior is unchanged; this is just also RetrievalService's
    # fallback for any caller that bypasses the factory entirely.
    embedder = GeminiEmbeddingProvider(api_key=settings.gemini_api_key)
    vector_store = PgVectorStore()
    retrieval_service = RetrievalService(embedder=embedder, vector_store=vector_store)

    def provider_factory(config: AssistantConfig) -> ModelProvider:
        return get_model_provider(model=config.model, settings=settings)

    def embedding_provider_factory(config: AssistantConfig) -> EmbeddingProvider:
        return get_embedding_provider(config=config, settings=settings)

    return ChatOrchestrator(
        audit_store=AuditStore(),
        retrieval_service=retrieval_service,
        provider_factory=provider_factory,
        embedding_provider_factory=embedding_provider_factory,
    )
