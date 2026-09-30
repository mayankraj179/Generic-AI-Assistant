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

# Same wording and behavior as GeminiProvider's/OpenRouterProvider's constant
# of the same name — kept per-provider rather than imported, so no provider
# module depends on another vendor's module.
TOOL_LOOP_EXHAUSTED_REPLY = (
    "I wasn't able to finish answering using the available tools within the "
    "allowed number of steps. Please try rephrasing your question or "
    "breaking it into smaller parts."
)

# Verified against xAI's own raw docs (docs.x.ai, fetched 2026-09-29):
# developers/rest-api-reference/inference/chat-completions.md,
# developers/debugging.md, developers/model-capabilities/text/streaming.md and
# .../structured-outputs.md, developers/tools/function-calling.md.
#
# Chat Completions (not the newer Responses API) on purpose: xAI documents it
# as the stateless, supported predecessor of Responses — it is not on the
# Legacy & Deprecated list. This framework already resends conversation
# history from its own store each turn, so Responses' server-side state would
# add nothing except xAI retaining conversations.
_CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"
# Reasoning models can take well over a minute; xAI's streaming docs explicitly
# warn to raise the client timeout for them.
_REQUEST_TIMEOUT_SECONDS = 180.0
# max_tokens is deprecated on this endpoint in favour of max_completion_tokens,
# which bounds visible output only (reasoning tokens are not counted), so a
# modest cap never starves a reasoning model's answer.
_MAX_COMPLETION_TOKENS = 2048


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
    """Same shape as GeminiProvider's schema. Its generated JSON Schema uses
    only features xAI's structured outputs enforce ($defs/$ref, anyOf with
    null, enum, arrays, objects)."""

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
    """Same input-side definition the other providers use."""
    return bool(prompt.retrieved_chunks)


def _build_tools_payload(enabled_tools: tuple[str, ...]) -> list[dict[str, Any]]:
    """Chat Completions function declarations: {"type": "function",
    "function": {"name", "description", "parameters"}}. Unknown names are
    skipped; AssistantConfig already rejects them at config-load time."""
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


def _build_messages(prompt: GroundedPrompt) -> list[dict[str, Any]]:
    """xAI accepts system/user/assistant roles in any order (developers/
    models.md), which already match ConversationTurn.role — no translation."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompt.system_prompt}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in prompt.prior_turns)
    messages.append({"role": "user", "content": prompt.user_message})
    return messages


def _parse_json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _error_detail(body: Any) -> str | None:
    """Pulls a human-readable message out of an error body. xAI's docs list
    status codes but do not document the error body's JSON shape, so this
    accepts the plausible shapes rather than assuming one: {"error": "..."},
    {"error": {"message": "..."}}, and {"code": ..., "error": "..."}."""
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, str) and error:
        return error
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return None


def _raise_for_error(
    status_code: int, body: Any, *, model: str | None = None, operation: str = "chat"
) -> None:
    """Raises ModelProviderError for an error status OR for any body that
    carries an "error" key, whatever the status. The second case is the
    failure mode OpenRouter hit live (HTTP 200 wrapping an upstream error);
    xAI doesn't document it, so it is checked rather than assumed away."""
    has_error_body = isinstance(body, dict) and body.get("error") is not None
    if status_code < 400 and not has_error_body:
        return
    detail = _error_detail(body)
    suffix = f": {detail}" if detail else ""
    raise ModelProviderError(
        f"xAI returned an error (status={status_code}){suffix}",
        failure=classify_http(
            provider="xai", status=status_code, body=body, operation=operation, model=model
        ),
    )


def _format_chart_context(prompt: GroundedPrompt) -> str:
    return "\n\n".join(
        f"[source: {chunk.document_title}, chunk {chunk.chunk_index}]\n{chunk.display_text}"
        for chunk in prompt.retrieved_chunks
    )


class GrokProvider:
    """A third ``ModelProvider`` implementation, alongside GeminiProvider and
    OpenRouterProvider, talking to xAI's API directly over httpx (no xAI SDK
    dependency). Every HTTP/auth/rate-limit/malformed-response failure is
    wrapped into ModelProviderError.

    Tool calling is a client-side loop like OpenRouterProvider's: xAI returns
    ``tool_calls`` for functions we define and expects results fed back as
    ``role: "tool"`` messages (developers/tools/function-calling.md); there is
    no server-side runner for custom functions.

    Chart generation IS implemented here (unlike OpenRouterProvider): xAI
    documents ``response_format: {"type": "json_schema", ...}`` output as
    guaranteed to match the schema for the supported feature subset, which
    the chart schema stays within. The result is still validated locally, and
    ChatOrchestrator._validate_chart re-checks its sources.
    """

    def __init__(self, *, model_name: str, api_key: str | None) -> None:
        if not api_key:
            raise ModelProviderError("xAI API key is not configured — set XAI_API_KEY")
        self._model_name = model_name
        self._api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: GroundedPrompt) -> ModelReply:
        messages = _build_messages(prompt)
        tools_payload = _build_tools_payload(prompt.enabled_tools)
        if tools_payload:
            text = await self._negotiate_tool_calls(messages, tools_payload, prompt.max_tool_calls)
        else:
            text = self._extract_text(await self._chat_completions_once(messages))
        chart = await self._maybe_generate_chart(prompt)
        return ModelReply(text=text, grounded=_is_grounded(prompt), chart=chart)

    async def generate_stream(self, prompt: GroundedPrompt) -> AsyncIterator[ModelStreamEvent]:
        # Not built yet: generate() is verified live first, then streaming is
        # implemented against the real SSE behavior rather than written blind.
        raise ModelProviderError("GrokProvider streaming is not implemented yet")
        yield  # pragma: no cover - makes this an async generator

    async def _negotiate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        max_tool_calls: int,
    ) -> str:
        """Bounded propose -> execute -> feed-back loop. The tools array is
        resent on every call; each call's ``function.arguments`` is a
        JSON-encoded string."""
        for _ in range(max_tool_calls):
            message = await self._chat_completions_once(messages, tools_payload)
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return self._extract_text(message)

            messages.append(
                {"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls}
            )
            for call in tool_calls:
                messages.append(await self._execute_one_tool_call(call))

        return TOOL_LOOP_EXHAUSTED_REPLY

    async def _execute_one_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
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
        return {"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result)}

    def _extract_text(self, message: dict[str, Any]) -> str:
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if text:
            return text
        # xAI reports a model-side refusal in its own field with empty content.
        # Returning it (rather than raising) lets the orchestrator's refusal
        # detection treat it like any other refusal.
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()
        raise ModelProviderError("xAI returned an empty response")

    async def _chat_completions_once(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None = None,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": _MAX_COMPLETION_TOKENS,
        }
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"
        if response_format is not None:
            payload["response_format"] = response_format

        operation = "chart" if response_format is not None else "chat"
        with start_span(
            "xai.chat_completions",
            provider="xai",
            model=self._model_name,
            operation=operation,
            tools=bool(tools_payload),
        ) as span:
            try:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        _CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
                    )
            except Exception as exc:  # network/timeout/TLS failures from httpx
                record_provider_call("model", "xai", self._model_name)
                raise ModelProviderError(
                    "xAI request failed",
                    failure=classify_exception(
                        exc, provider="xai", operation=operation, model=self._model_name
                    ),
                ) from exc

            body = _parse_json_body(response.content)
            usage = body.get("usage") if isinstance(body, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            record_provider_call(
                "model",
                "xai",
                self._model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
            set_attributes(
                span,
                {
                    "http.status_code": response.status_code,
                    "usage.prompt_tokens": usage.get("prompt_tokens"),
                    "usage.completion_tokens": usage.get("completion_tokens"),
                    "usage.sources_used": usage.get("num_sources_used"),
                },
            )
            _raise_for_error(
                response.status_code, body, model=self._model_name, operation=operation
            )

            try:
                return body["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                keys = sorted(body) if isinstance(body, dict) else type(body).__name__
                raise ModelProviderError(
                    f"xAI returned an unexpected response shape (status={response.status_code}, "
                    f"keys={keys})",
                    failure=ProviderFailure(
                        kind=ErrorKind.MALFORMED_RESPONSE,
                        provider="xai",
                        operation=operation,
                        model=self._model_name,
                        status=response.status_code,
                        message=f"no choices; keys={keys}",
                    ),
                ) from exc

    async def _maybe_generate_chart(self, prompt: GroundedPrompt) -> ChartSpec | None:
        """Separate structured-output call, made only when the orchestrator
        flagged chart_requested and retrieval produced chunks. Any failure is
        logged and becomes "no chart"; it never affects the text answer."""
        if not prompt.chart_requested or not prompt.retrieved_chunks:
            return None

        messages = [
            {"role": "system", "content": _CHART_EXTRACTION_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{_format_chart_context(prompt)}\n\n"
                    f"USER QUESTION:\n{prompt.user_message}"
                ),
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "chart_extraction",
                "schema": _ChartExtractionSchema.model_json_schema(),
                "strict": True,
            },
        }
        try:
            message = await self._chat_completions_once(messages, response_format=response_format)
            parsed = _ChartExtractionSchema.model_validate_json(message.get("content") or "")
        except (ModelProviderError, ValidationError, ValueError) as exc:
            log_chart_extraction_failure(logger, exc, provider="xai", model=self._model_name)
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
