from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from app.observability.audit import record_provider_call
from app.observability.provider_errors import (
    ErrorKind,
    ProviderFailure,
    classify_exception,
    classify_http,
    log_chart_extraction_failure,
)
from app.observability.tracing import set_attributes, start_span
from app.orchestration.model_provider import (
    ChartSeries,
    ChartSourceChunk,
    ChartSpec,
    GroundedPrompt,
    ModelProviderError,
    ModelReply,
    ModelStreamEvent,
)
from app.tools.builtin import TOOL_REGISTRY, ToolExecutionError, execute_tool

logger = logging.getLogger(__name__)

# Returned instead of raising when the tool-call loop uses up
# max_tool_calls round-trips without the model reaching a final answer —
# an honest "I couldn't finish" reply, never a truncated or fabricated
# one. Shared with GeminiProvider's identical constant/behavior.
TOOL_LOOP_EXHAUSTED_REPLY = (
    "I wasn't able to finish answering using the available tools within the "
    "allowed number of steps. Please try rephrasing your question or "
    "breaking it into smaller parts."
)

# Verified live against OpenRouter's current docs (openrouter.ai/docs, fetched
# 2026-09-22 — raw markdown, not the OpenAI-compatible shape assumed
# up front): endpoint, request/response JSON, error shape, and SSE framing
# below all come from openrouter.ai/docs/quickstart.md,
# openrouter.ai/docs/api_reference/streaming.md and
# openrouter.ai/docs/api_reference/errors-and-debugging.md, plus a live call
# to GET /api/v1/models to confirm model slugs and structured-output support.
_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_REQUEST_TIMEOUT_SECONDS = 60.0
_APP_TITLE = "Generic AI Assistant Framework"
# Omitting max_tokens lets OpenRouter default to the routed model's own max
# output (e.g. 65536 for anthropic/claude-sonnet-5) — confirmed live: that
# default alone triggered a 402 "insufficient credits" against a real,
# funded key affording only 4000 tokens for that request. A grounded RAG
# answer never needs anywhere near that much output, so a modest explicit
# cap avoids over-requesting against the account's actual headroom.
_DEFAULT_MAX_TOKENS = 2048
_SSE_DATA_PREFIX = "data: "
_SSE_DONE = "[DONE]"


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
    """Same shape as GeminiProvider's/GrokProvider's schema, kept per-provider
    so no provider module imports another vendor's module."""

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
    """Same input-side definition GeminiProvider uses — see that module for
    the full rationale. Kept duplicated rather than shared: it's a one-line
    check on ``GroundedPrompt`` alone, not vendor-specific behavior, so a
    shared helper would just be indirection for indirection's sake.
    """
    return bool(prompt.retrieved_chunks)


def _build_tools_payload(enabled_tools: tuple[str, ...]) -> list[dict[str, Any]]:
    """Maps GroundedPrompt.enabled_tools to OpenAI-shape tool declarations —
    verified live against openrouter.ai/docs/guides/features/tool-calling:
    {"type": "function", "function": {"name", "description", "parameters"}},
    with "parameters" being a plain JSON Schema object (exactly what
    ToolDefinition.input_schema already is). Unknown names are silently
    skipped, same defense-in-depth rationale as GeminiProvider's
    _build_function_tools — AssistantConfig already rejects them at
    config-load time.
    """
    tools = []
    for name in enabled_tools:
        definition = TOOL_REGISTRY.get(name)
        if definition is None:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            }
        )
    return tools


def _build_messages(prompt: GroundedPrompt) -> list[dict[str, str]]:
    """OpenRouter's chat-completions endpoint uses OpenAI-shaped messages
    with role values ("system"/"user"/"assistant") that already match
    ConversationTurn.role and GroundedPrompt's own vocabulary exactly — unlike
    Gemini/ADK (see GeminiProvider._new_seeded_session), no role translation
    is needed here.
    """
    # Chart-turn wording guidance is added by ChatOrchestrator for every
    # provider (see chat_service.CHART_TURN_GUIDANCE), not here.
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt.system_prompt}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in prompt.prior_turns)
    messages.append({"role": "user", "content": prompt.user_message})
    return messages


def _chart_will_be_attempted(prompt: GroundedPrompt) -> bool:
    return prompt.chart_requested and bool(prompt.retrieved_chunks)


def _parse_json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _raise_for_error(
    status_code: int, body: Any, *, model: str | None = None, operation: str = "chat"
) -> None:
    """Wraps an OpenRouter error into ModelProviderError, classified.

    OpenRouter's documented error shape is ``{"error": {"code", "message",
    "metadata"}}`` (openrouter.ai/docs/api_reference/errors-and-debugging).
    It can also arrive under HTTP 200 when the upstream model fails after
    routing — seen live as ``{"id": ..., "error": {"message": "Upstream error
    from Nvidia: Service temporarily overloaded", "code": 503, ...}}`` — so an
    error body is treated as an error whatever the status.
    """
    has_error_body = isinstance(body, dict) and body.get("error") is not None
    if status_code < 400 and not has_error_body:
        return
    detail: str | None = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                detail = message
    suffix = f": {detail}" if detail else ""
    raise ModelProviderError(
        f"OpenRouter returned an error (status={status_code}){suffix}",
        failure=classify_http(
            provider="openrouter", status=status_code, body=body, operation=operation, model=model
        ),
    )


class OpenRouterProvider:
    """A second ``ModelProvider`` implementation, alongside GeminiProvider.

    Talks to OpenRouter's HTTP API directly via httpx (OpenRouter has no
    first-party Python SDK equivalent to google-genai) rather than an SDK,
    unlike GeminiProvider. All HTTP/auth/rate-limit failures are wrapped into
    ModelProviderError — never a raw httpx exception escapes this class.

    Chart generation uses a second structured-output call, made only when
    the orchestrator set chart_requested and retrieval produced chunks.
    Structured-output support is per endpoint on OpenRouter, not universal
    (openrouter.ai/docs/guides/features/structured-outputs). Without
    ``provider.require_parameters``, an endpoint lacking support silently
    ignores ``response_format`` and may return free text; with it set, the
    request is never routed to such an endpoint and fails instead. So the
    chart call always sets it, and any failure or non-conforming output
    becomes "no chart" rather than a guessed one. The parsed result is
    validated locally, and ChatOrchestrator._validate_chart re-checks its
    sources. Checked live 2026-09-29: nvidia/nemotron-3-super-120b-a12b:free
    declares structured_outputs and response_format on its only endpoint.
    """

    def __init__(self, *, model_name: str, api_key: str | None) -> None:
        if not api_key:
            raise ModelProviderError(
                "OpenRouter API key is not configured — set OPENROUTER_API_KEY"
            )
        self._model_name = model_name
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Optional app-attribution header (openrouter.ai/docs/quickstart) —
            # OpenRouter-specific headers are optional and only affect
            # leaderboard attribution, never request correctness, so a fixed
            # value identifying this framework is sufficient; no HTTP-Referer
            # is sent since this framework has no single public app URL to
            # report and that header is likewise optional.
            "X-OpenRouter-Title": _APP_TITLE,
        }

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:
        messages = _build_messages(prompt)
        tools_payload = _build_tools_payload(prompt.enabled_tools)
        text = await self._resolve_final_text(messages, tools_payload, prompt.max_tool_calls)
        chart = await self._maybe_generate_chart(prompt)
        return ModelReply(text=text, grounded=_is_grounded(prompt), chart=chart)

    async def _resolve_final_text(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        max_tool_calls: int,
    ) -> str:
        """Shared by generate() and (for the tool-negotiation phase only —
        see generate_stream()) the streaming path: runs the bounded
        tool-call loop when tools_payload is non-empty, otherwise makes a
        single plain call — either way returning the final answer text.
        """
        if not tools_payload:
            message = await self._chat_completions_once(messages, None)
            return self._extract_text(message)
        return await self._negotiate_tool_calls(messages, tools_payload, max_tool_calls)

    async def _negotiate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        max_tool_calls: int,
    ) -> str:
        """OpenRouter's manual tool-call loop — OpenRouter has no built-in
        agent runner the way google-adk's Runner executes FunctionTool calls
        internally for GeminiProvider (see that module), so this provider
        drives the propose -> execute -> feed-result-back -> repeat cycle
        itself, bounded by max_tool_calls round-trips (never unbounded, even
        for these read-only tools).

        Verified live against openrouter.ai/docs/guides/features/tool-calling:
        the full `tools` array must be resent on every request in the loop;
        `message["tool_calls"][i]["function"]["arguments"]` is a
        JSON-*encoded string*, not an object, so it must be json.loads()'d;
        and a tool's result is fed back as a
        {"role": "tool", "tool_call_id": ..., "content": <json string>}
        message.

        Graceful degradation: OpenRouter proxies many different underlying
        models/providers, and — the same per-model inconsistency already
        found for structured-output/chart support — not all of them accept
        the `tools` parameter. If the very first call in this loop comes
        back as a 400 Bad Request (OpenRouter's documented code for
        "invalid or missing params"), this assumes the routed model/provider
        rejected `tools` specifically and retries once with a plain,
        tool-less call rather than failing the whole turn.
        """
        remaining_tools_payload: list[dict[str, Any]] | None = tools_payload
        for call_index in range(max_tool_calls):
            try:
                message = await self._chat_completions_once(messages, remaining_tools_payload)
            except ModelProviderError as exc:
                if call_index == 0 and remaining_tools_payload and "status=400" in str(exc):
                    logger.warning(
                        "OpenRouter rejected tools for model '%s' (400 Bad Request) — "
                        "retrying once without tools; per-model tool-calling support is "
                        "not universal on OpenRouter.",
                        self._model_name,
                    )
                    remaining_tools_payload = None
                    message = await self._chat_completions_once(messages, None)
                else:
                    raise

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return self._extract_text(message)

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                messages.append(await self._execute_one_tool_call(call))

        return TOOL_LOOP_EXHAUSTED_REPLY

    async def _execute_one_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            result: dict[str, Any] = {"error": "tool call arguments were not valid JSON"}
        else:
            if not isinstance(arguments, dict):
                result = {"error": "tool call arguments must be a JSON object"}
            else:
                try:
                    result = await execute_tool(name, arguments)
                except ToolExecutionError as exc:
                    result = {"error": str(exc)}
        return {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "content": json.dumps(result),
        }

    def _extract_text(self, message: dict[str, Any]) -> str:
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            raise ModelProviderError("OpenRouter returned an empty response")
        return text

    async def _chat_completions_once(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        operation = "chart" if response_format is not None else "chat"
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"
        if response_format is not None:
            payload["response_format"] = response_format
            # Never route to an endpoint that would silently ignore it.
            payload["provider"] = {"require_parameters": True}

        with start_span(
            "openrouter.chat_completions", provider="openrouter", model=self._model_name,
            operation=operation, tools=bool(tools_payload),
        ) as span:
            try:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        _CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
                    )
            except ModelProviderError:
                raise
            except Exception as exc:  # network/timeout/TLS failures from httpx
                record_provider_call("model", "openrouter", self._model_name)
                raise ModelProviderError(
                    "OpenRouter request failed",
                    failure=classify_exception(
                        exc, provider="openrouter", operation=operation, model=self._model_name
                    ),
                ) from exc

            body = _parse_json_body(response.content)
            usage = body.get("usage") if isinstance(body, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            record_provider_call(
                "model", "openrouter", self._model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
            set_attributes(span, {
                "http.status_code": response.status_code,
                "usage.prompt_tokens": usage.get("prompt_tokens"),
                "usage.completion_tokens": usage.get("completion_tokens"),
            })
            _raise_for_error(
                response.status_code, body, model=self._model_name, operation=operation
            )

            try:
                return body["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                keys = sorted(body) if isinstance(body, dict) else type(body).__name__
                raise ModelProviderError(
                    "OpenRouter returned an unexpected response shape",
                    failure=ProviderFailure(
                        kind=ErrorKind.MALFORMED_RESPONSE, provider="openrouter",
                        operation=operation, model=self._model_name,
                        status=response.status_code, message=f"no choices; keys={keys}",
                    ),
                ) from exc

    async def generate_stream(self, prompt: GroundedPrompt) -> AsyncIterator[ModelStreamEvent]:
        tools_payload = _build_tools_payload(prompt.enabled_tools)
        if tools_payload:
            # Tool-call negotiation needs the complete (non-streamed)
            # response at each round-trip to know whether it's a tool call
            # or the final answer — there is no meaningful way to stream
            # partial tokens *during* that back-and-forth. So when tools are
            # enabled, the whole bounded loop runs first via
            # _negotiate_tool_calls (identical to generate()'s path), and
            # only the finished answer is delivered — as a single delta
            # followed by the final event, not live token-by-token. An
            # ordinary tool-less turn (the vast majority) is completely
            # unaffected and keeps real-time streaming exactly as before.
            text = await self._negotiate_tool_calls(
                _build_messages(prompt), tools_payload, prompt.max_tool_calls
            )
            yield ModelStreamEvent(delta=text)
            chart = await self._maybe_generate_chart(prompt)
            yield ModelStreamEvent(
                is_final=True, text=text, grounded=_is_grounded(prompt), chart=chart
            )
            return

        payload = {
            "model": self._model_name,
            "messages": _build_messages(prompt),
            "stream": True,
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        final_text_parts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                async with client.stream(
                    "POST", _CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
                ) as response:
                    record_provider_call("model", "openrouter", self._model_name)
                    if response.status_code >= 400:
                        raw = await response.aread()
                        _raise_for_error(
                            response.status_code, _parse_json_body(raw), model=self._model_name
                        )

                    async for line in response.aiter_lines():
                        # Per openrouter.ai/docs/api_reference/streaming: lines
                        # starting with ":" are keep-alive comments (e.g.
                        # ": OPENROUTER PROCESSING"), never JSON — ignore them.
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith(_SSE_DATA_PREFIX):
                            continue
                        data = line[len(_SSE_DATA_PREFIX):]
                        if data == _SSE_DONE:
                            break

                        event = _parse_json_body(data.encode("utf-8"))
                        if not isinstance(event, dict):
                            continue
                        # An upstream failure after the stream has started
                        # arrives as an SSE event carrying "error"; it must
                        # surface as an error, not be skipped as a frame.
                        _raise_for_error(response.status_code, event, model=self._model_name)
                        choices = event.get("choices") or []
                        if not choices:
                            # The documented final usage-accounting chunk has
                            # a content-free delta repeating finish_reason
                            # alongside a top-level "usage" object — treat any
                            # choice-less chunk the same way: an accounting
                            # frame, not a text delta.
                            continue
                        delta = (choices[0] or {}).get("delta") or {}
                        content = delta.get("content")
                        if content:
                            final_text_parts.append(content)
                            yield ModelStreamEvent(delta=content)
        except ModelProviderError:
            raise
        except Exception as exc:  # network/timeout/TLS failures from httpx
            raise ModelProviderError(
                "OpenRouter request failed",
                failure=classify_exception(
                    exc, provider="openrouter", operation="chat", model=self._model_name
                ),
            ) from exc

        final_text = "".join(final_text_parts).strip()
        if not final_text:
            raise ModelProviderError("OpenRouter returned an empty response")

        # As in GeminiProvider: the chart call runs after the last text delta
        # and before the final event, and only on chart turns.
        chart = await self._maybe_generate_chart(prompt)
        yield ModelStreamEvent(
            is_final=True, text=final_text, grounded=_is_grounded(prompt), chart=chart
        )

    async def _maybe_generate_chart(self, prompt: GroundedPrompt) -> ChartSpec | None:
        """Any failure (error status, no supporting endpoint, a 200 body
        without choices, unparseable or schema-invalid JSON, an invalid pie
        chart) is logged and becomes "no chart"; it never affects the text
        answer."""
        if not _chart_will_be_attempted(prompt):
            return None

        context = "\n\n".join(
            f"[source: {chunk.document_title}, chunk {chunk.chunk_index}]\n{chunk.display_text}"
            for chunk in prompt.retrieved_chunks
        )
        messages = [
            {"role": "system", "content": _CHART_EXTRACTION_INSTRUCTION},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{prompt.user_message}",
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "chart_extraction",
                "strict": True,
                "schema": _ChartExtractionSchema.model_json_schema(),
            },
        }
        try:
            message = await self._chat_completions_once(
                messages, None, response_format=response_format
            )
            parsed = _ChartExtractionSchema.model_validate_json(message.get("content") or "")
        except (ModelProviderError, ValidationError, ValueError) as exc:
            log_chart_extraction_failure(
                logger, exc, provider="openrouter", model=self._model_name
            )
            return None

        payload = parsed.chart
        if payload is None:
            return None
        try:
            return ChartSpec(
                chart_type=payload.chart_type,
                title=payload.title,
                labels=list(payload.labels),
                series=[ChartSeries(name=s.name, values=list(s.values)) for s in payload.series],
                source_chunks=[
                    ChartSourceChunk(document_title=s.document_title, chunk_index=s.chunk_index)
                    for s in payload.source_chunks
                ],
            )
        except ValueError:
            logger.warning("discarding a malformed chart extraction response")
            return None
