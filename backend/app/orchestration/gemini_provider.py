from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

from app.observability.audit import record_provider_call, record_tokens, record_tool_call
from app.observability.provider_errors import (
    classify_exception,
    classify_google,
    log_chart_extraction_failure,
)
from app.orchestration.model_provider import (
    ChartSeries,
    ChartSourceChunk,
    ChartSpec,
    GroundedPrompt,
    ModelProviderError,
    ModelReply,
    ModelStreamEvent,
)
from app.tools.builtin import TOOL_REGISTRY, calculate, current_datetime

logger = logging.getLogger(__name__)

# Returned instead of raising when the tool-call loop hits its bound
# (RunConfig.max_llm_calls, set from GroundedPrompt.max_tool_calls) without
# the model reaching a final text answer — an honest "I couldn't finish"
# reply, never a truncated or fabricated one.
TOOL_LOOP_EXHAUSTED_REPLY = (
    "I wasn't able to finish answering using the available tools within the "
    "allowed number of steps. Please try rephrasing your question or "
    "breaking it into smaller parts."
)

# Maps GroundedPrompt.enabled_tools names to the actual Python function
# google-adk's FunctionTool wraps. FunctionTool derives the tool name the
# model sees from the function's own __name__ (verified against the
# installed SDK — FunctionTool.__init__ takes no name-override parameter),
# so these functions are named exactly "current_datetime"/"calculate" in
# app/tools/builtin.py for that reason.
_TOOL_FUNCTIONS_BY_NAME = {
    "current_datetime": current_datetime,
    "calculate": calculate,
}


def _build_function_tools(enabled_tools: tuple[str, ...], FunctionTool: Any) -> list[Any]:
    """Resolves GroundedPrompt.enabled_tools into google-adk FunctionTool
    instances. Silently skips any name not in TOOL_REGISTRY/
    _TOOL_FUNCTIONS_BY_NAME rather than raising — AssistantConfig's own
    field_validator (app/config/assistant_config.py) already rejects an
    unknown tool name at config-load time, long before a request reaches
    here, so this is defense in depth, not the primary validation point.
    """
    tools = []
    for name in enabled_tools:
        if name not in TOOL_REGISTRY:
            continue
        func = _TOOL_FUNCTIONS_BY_NAME.get(name)
        if func is not None:
            tools.append(FunctionTool(func))
    return tools


# Structured-output schema for the chart-extraction call — a Pydantic model
# passed directly as google-genai's GenerateContentConfig.response_schema,
# verified live against the installed SDK (google-genai 2.24.0): the
# response comes back on GenerateContentResponse.parsed as an actual
# instance of this class, not text this code has to hand-parse as JSON.
# Deliberately a *separate* schema from ChartSpec/ChartSeries/
# ChartSourceChunk (app/orchestration/model_provider.py) rather than reusing
# those frozen dataclasses directly: response_schema needs a Pydantic
# model, and keeping the vendor-facing wire schema separate from the
# framework's own vendor-neutral dataclass keeps this file the only place
# that shape is allowed to be Gemini-specific.
class _ChartSeriesSchema(BaseModel):
    name: str
    values: list[float]


class _ChartSourceChunkSchema(BaseModel):
    document_title: str
    chunk_index: int


class _ChartPayloadSchema(BaseModel):
    chart_type: Literal["bar", "line", "pie"]
    title: str
    labels: list[str]
    series: list[_ChartSeriesSchema]
    source_chunks: list[_ChartSourceChunkSchema]


class _ChartExtractionSchema(BaseModel):
    """``chart`` is optional/nullable on purpose — the model is instructed
    to omit it rather than invent a chart when the retrieved context has no
    genuinely chart-worthy structured numeric data."""

    chart: _ChartPayloadSchema | None = None


_CHART_EXTRACTION_INSTRUCTION = (
    "You extract chart-ready structured numeric data from the CONTEXT below, "
    "if and only if the context genuinely contains tabular or numeric data "
    "suited to a bar, line, or pie chart that answers the user's question.\n"
    "Rules, all mandatory:\n"
    "1. Use ONLY numbers that are explicitly present in the CONTEXT below — "
    "never invent, estimate, infer, or round a number that isn't stated.\n"
    "2. Every entry in source_chunks must exactly copy a 'document_title' and "
    "'chunk_index' from one of the [source: ...] headers actually shown in "
    "the CONTEXT — never a document/chunk you did not see in this context.\n"
    "3. A pie chart must have exactly one series.\n"
    "4. If the CONTEXT does not contain genuinely chart-worthy structured "
    "numeric data, return chart: null. Returning null is strongly preferred "
    "over guessing or approximating — a wrong chart is worse than no chart."
)


def _is_grounded(prompt: GroundedPrompt) -> bool:
    """The input-side definition of "grounded": relevant, safe context was
    retrieved and genuinely included in this prompt — not "the model's
    answer appears to rely on it" (output-side, and much harder to detect
    reliably). ``prompt.retrieved_chunks`` is guaranteed by
    ChatOrchestrator._prepare_turn to already be the post-similarity-
    threshold, post-injection-screening set that was actually folded into
    ``prompt.user_message`` — never the raw, unfiltered retrieval result —
    so this is a real relevance check, not just "some chunk existed."

    A model can still legitimately refuse to answer even from context that
    passes this check; that's handled separately, as an output-side
    guardrail (ChatOrchestrator._apply_output_guardrails), not folded into
    this flag.
    """
    return bool(prompt.retrieved_chunks)


class GeminiProvider:
    """The only Gemini/Google-ADK-specific code in the framework.

    Everything above this class (routes, orchestration, retrieval,
    authorization) talks to the ``ModelProvider`` Protocol only. This is the
    single place ADK's ``Agent``/``Runner`` and google-genai's ``Client`` are
    imported and constructed.
    """

    AGENT_NAME = "assistant"

    def __init__(self, *, model_name: str, api_key: str | None) -> None:
        if not api_key:
            raise ModelProviderError(
                "Gemini API key is not configured — set GEMINI_API_KEY"
            )
        self._model_name = model_name
        self._api_key = api_key

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:
        try:
            from google.adk.agents import Agent
            from google.adk.agents.invocation_context import LlmCallsLimitExceededError
            from google.adk.agents.run_config import RunConfig
            from google.adk.events.event import Event
            from google.adk.models.google_llm import Gemini
            from google.adk.runners import InMemoryRunner
            from google.adk.tools import FunctionTool
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - depends on optional SDK
            raise ModelProviderError(
                "Google ADK / google-genai SDK is not installed"
            ) from exc

        tools = _build_function_tools(prompt.enabled_tools, FunctionTool)
        # Tool calls happen as extra LLM round-trips *inside* this same
        # run_async() call (google-adk's Runner executes each proposed
        # FunctionTool call and feeds the result back internally — no
        # separate loop needed here, unlike OpenRouterProvider's manual one)
        # — so the run-wide cap is what bounds them. +1 allows one final
        # answer-producing call after up to max_tool_calls tool round-trips.
        run_config = RunConfig(max_llm_calls=prompt.max_tool_calls + 1)

        try:
            runner, session, user_id = await self._new_seeded_session(
                prompt, Agent=Agent, Event=Event, Gemini=Gemini,
                InMemoryRunner=InMemoryRunner, genai_types=genai_types, tools=tools,
            )

            final_text: str | None = None
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=genai_types.UserContent(
                    parts=[genai_types.Part(text=prompt.user_message)]
                ),
                run_config=run_config,
            ):
                if event.error_message:
                    raise ModelProviderError(
                        f"Gemini returned an error (code={event.error_code})",
                        failure=classify_google(
                            code=None,
                            status=event.error_code,
                            message=event.error_message,
                            operation="chat",
                            model=self._model_name,
                        ),
                    )
                self._observe_event(event)
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(
                        part.text or "" for part in event.content.parts
                    ).strip()
        except LlmCallsLimitExceededError:
            return ModelReply(
                text=TOOL_LOOP_EXHAUSTED_REPLY, grounded=_is_grounded(prompt), chart=None
            )
        except ModelProviderError:
            raise
        except Exception as exc:  # network/auth/SDK failures from the ADK/genai stack
            raise ModelProviderError(
                "Gemini request failed",
                failure=classify_exception(
                    exc, provider="gemini", operation="chat", model=self._model_name
                ),
            ) from exc

        if not final_text:
            raise ModelProviderError("Gemini returned an empty response")

        chart = await self._maybe_generate_chart(prompt)
        return ModelReply(text=final_text, grounded=_is_grounded(prompt), chart=chart)

    async def generate_stream(self, prompt: GroundedPrompt) -> AsyncIterator[ModelStreamEvent]:
        try:
            from google.adk.agents import Agent
            from google.adk.agents.invocation_context import LlmCallsLimitExceededError
            from google.adk.agents.run_config import RunConfig, StreamingMode
            from google.adk.events.event import Event
            from google.adk.models.google_llm import Gemini
            from google.adk.runners import InMemoryRunner
            from google.adk.tools import FunctionTool
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - depends on optional SDK
            raise ModelProviderError(
                "Google ADK / google-genai SDK is not installed"
            ) from exc

        tools = _build_function_tools(prompt.enabled_tools, FunctionTool)

        # StreamingMode.SSE is what actually makes run_async yield incremental
        # partial-text events instead of one blocking final event — verified
        # against the installed SDK (google-adk 2.9.1): RunConfig.streaming_mode
        # flows into base_llm_flow's `stream=run_config.streaming_mode ==
        # StreamingMode.SSE`, which in turn calls Gemini.generate_content_async
        # with stream=True. Each event's `partial` flag distinguishes a delta
        # (partial=True, content.parts[].text is the incremental new text —
        # confirmed via google.adk.utils.streaming_utils.StreamingResponseAggregator,
        # which forwards each raw streamed chunk's own text unmodified) from
        # the single final aggregated event produced by that same aggregator's
        # close() (partial=False, content is the full merged text) — exactly
        # mirroring what event.is_final_response() already detects in
        # generate() above. Not guessed.
        run_config = RunConfig(
            streaming_mode=StreamingMode.SSE, max_llm_calls=prompt.max_tool_calls + 1
        )

        final_text: str | None = None
        loop_exhausted = False
        try:
            runner, session, user_id = await self._new_seeded_session(
                prompt, Agent=Agent, Event=Event, Gemini=Gemini,
                InMemoryRunner=InMemoryRunner, genai_types=genai_types, tools=tools,
            )

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=genai_types.UserContent(
                    parts=[genai_types.Part(text=prompt.user_message)]
                ),
                run_config=run_config,
            ):
                if event.error_message:
                    raise ModelProviderError(
                        f"Gemini returned an error (code={event.error_code})",
                        failure=classify_google(
                            code=None,
                            status=event.error_code,
                            message=event.error_message,
                            operation="chat",
                            model=self._model_name,
                        ),
                    )
                self._observe_event(event)
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(
                        part.text or "" for part in event.content.parts
                    ).strip()
                    continue
                if event.partial and event.content and event.content.parts:
                    delta_text = "".join(part.text or "" for part in event.content.parts)
                    if delta_text:
                        yield ModelStreamEvent(delta=delta_text)
        except LlmCallsLimitExceededError:
            loop_exhausted = True
        except ModelProviderError:
            raise
        except Exception as exc:  # network/auth/SDK failures from the ADK/genai stack
            raise ModelProviderError(
                "Gemini request failed",
                failure=classify_exception(
                    exc, provider="gemini", operation="chat", model=self._model_name
                ),
            ) from exc

        if loop_exhausted:
            yield ModelStreamEvent(
                is_final=True,
                text=TOOL_LOOP_EXHAUSTED_REPLY,
                grounded=_is_grounded(prompt),
                chart=None,
            )
            return

        if not final_text:
            raise ModelProviderError("Gemini returned an empty response")

        # The chart-extraction call (when it fires at all — see
        # _maybe_generate_chart) happens here, after the last text delta has
        # already been yielded and before this final event — this is the
        # source of the "short pause after text finishes streaming" the
        # chart feature can introduce, entirely gated on chart_requested so
        # ordinary turns never pay it.
        chart = await self._maybe_generate_chart(prompt)
        yield ModelStreamEvent(
            is_final=True, text=final_text, grounded=_is_grounded(prompt), chart=chart
        )

    async def _maybe_generate_chart(self, prompt: GroundedPrompt) -> ChartSpec | None:
        """Second, separate model call made ONLY when chart_requested and
        retrieved_chunks is non-empty — see GroundedPrompt.chart_requested
        for why this is never a hidden cost on ordinary turns.

        Uses google-genai's ``Client`` directly (the same pattern
        app/services/gemini_embedding.py already establishes for a one-shot,
        non-conversational SDK call) rather than routing through ADK's
        Agent/Runner used for the text reply above: this is intentionally a
        second call, not one combined call, because it's a one-shot
        structured-extraction task with no conversational state, and ADK's
        Agent/Runner has no clean per-turn ``response_schema`` knob to graft
        onto the same call that also has to keep streaming free-text deltas.
        Any failure here (SDK error, malformed/unparseable response, or a
        response that fails ChartSpec's own pie-chart-series validation) is
        swallowed and logged — a chart is a bonus on top of the text answer,
        never something that should fail or degrade the text answer itself.
        """
        if not prompt.chart_requested or not prompt.retrieved_chunks:
            return None

        try:
            from google.genai import Client
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - depends on optional SDK
            raise ModelProviderError(
                "Google ADK / google-genai SDK is not installed"
            ) from exc

        context_text = "\n\n".join(
            f"[source: {chunk.document_title}, chunk {chunk.chunk_index}]\n{chunk.display_text}"
            for chunk in prompt.retrieved_chunks
        )
        contents = (
            f"{_CHART_EXTRACTION_INSTRUCTION}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION:\n{prompt.user_message}"
        )

        record_provider_call("model", "gemini", self._model_name)
        try:
            client = Client(api_key=self._api_key)
            response = await client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_ChartExtractionSchema,
                ),
            )
        except Exception as exc:
            log_chart_extraction_failure(
                logger,
                exc,
                provider="gemini",
                model=self._model_name,
                failure=classify_exception(
                    exc, provider="gemini", operation="chart", model=self._model_name
                ),
            )
            return None

        parsed = response.parsed
        if not isinstance(parsed, _ChartExtractionSchema) or parsed.chart is None:
            return None

        payload = parsed.chart
        try:
            return ChartSpec(
                chart_type=payload.chart_type,
                title=payload.title,
                labels=list(payload.labels),
                series=[
                    ChartSeries(name=series.name, values=list(series.values))
                    for series in payload.series
                ],
                source_chunks=[
                    ChartSourceChunk(
                        document_title=source.document_title, chunk_index=source.chunk_index
                    )
                    for source in payload.source_chunks
                ],
            )
        except ValueError:
            # ChartSpec.__post_init__ rejected a malformed response (e.g. a
            # pie chart with more than one series) — treat exactly like "no
            # chart" rather than propagating a validation error upward.
            logger.warning("discarding a malformed chart extraction response")
            return None

    def _count_model_call(self, callback_context: Any, llm_request: Any) -> None:
        """ADK before_model_callback: runs once per model request ADK makes,
        so tool-call rounds are counted too. Returns None: never alters the
        request."""
        record_provider_call("model", "gemini", self._model_name)
        return None

    def _observe_event(self, event: Any) -> None:
        """Token usage and tool outcomes from ADK events, for the turn audit.
        Partial streaming events are skipped so usage isn't counted twice."""
        usage = getattr(event, "usage_metadata", None)
        if usage is not None and not getattr(event, "partial", False):
            # Gemini 2.5 bills thinking tokens as output, and they can dwarf
            # the visible answer (live: 77 visible vs 1,658 thinking), so
            # completion tokens are candidates + thoughts.
            visible = getattr(usage, "candidates_token_count", None)
            thoughts = getattr(usage, "thoughts_token_count", None)
            completion = None if visible is None and thoughts is None else (
                (visible or 0) + (thoughts or 0)
            )
            record_tokens(
                "model",
                "gemini",
                self._model_name,
                prompt_tokens=getattr(usage, "prompt_token_count", None),
                completion_tokens=completion,
            )
        for response in event.get_function_responses() or []:
            result = response.response if isinstance(response.response, dict) else {}
            error = result.get("error")
            record_tool_call(
                response.name or "",
                ok=error is None,
                error=str(error) if error else None,
                result=result,
            )

    async def _new_seeded_session(
        self,
        prompt: GroundedPrompt,
        *,
        Agent: Any,
        Event: Any,
        Gemini: Any,
        InMemoryRunner: Any,
        genai_types: Any,
        tools: list[Any] | None = None,
    ) -> tuple[Any, Any, str]:
        """Builds a fresh Agent/Runner/session and replays prior conversation
        turns into it — the shared setup behind both generate() and
        generate_stream(), so the two don't duplicate ADK wiring. Takes the
        already-imported ADK/genai symbols rather than re-importing, so the
        single ImportError -> ModelProviderError check stays in each public
        method's own try block.
        """
        agent = Agent(
            name=self.AGENT_NAME,
            model=Gemini(model=self._model_name, client_kwargs={"api_key": self._api_key}),
            instruction=prompt.system_prompt,
            tools=tools or [],
            before_model_callback=self._count_model_call,
        )
        runner = InMemoryRunner(agent=agent, app_name="generic-ai-assistant-framework")

        user_id = "chat-turn"
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=uuid.uuid4().hex,
        )

        # Seed prior conversation turns into this fresh session before
        # running the new one, so the model sees full history. This is
        # ADK's own mechanism for session-based conversation memory
        # (Runner.run_async appends each new turn to the session the
        # same way) — verified against the installed SDK, not guessed.
        for turn in prompt.prior_turns:
            if turn.role == "user":
                content = genai_types.UserContent(parts=[genai_types.Part(text=turn.content)])
                author = "user"
            else:
                content = genai_types.ModelContent(parts=[genai_types.Part(text=turn.content)])
                author = self.AGENT_NAME
            await runner.session_service.append_event(
                session=session,
                event=Event(invocation_id=uuid.uuid4().hex, author=author, content=content),
            )

        return runner, session, user_id
