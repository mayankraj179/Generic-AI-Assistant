"""TEMP (testing, 2026-09-30): Azure AI Foundry via its OpenAI-compatible v1 API.

Confirmed against Microsoft's REST reference (learn.microsoft.com/en-us/azure/
foundry/openai/latest, "Create chat completion") and live against the
hrinitiatives resource:
  - POST {endpoint}/openai/v1/chat/completions, no api-version needed.
  - Auth: the reference lists three accepted methods, any one of them: the
    key in an ``api-key`` header, the key in an ``authorization`` header, or
    an Entra ID bearer token. This provider sends ``api-key``, the form the
    REST samples use for key auth. Live, a bogus key under either header gets
    the same 401, so both reach the same check.
  - Error body, live: {"error": {"code": "401", "message": "Access denied due
    to invalid subscription key or wrong API endpoint. ..."}}, with ``code``
    as a string, plus an ``apim-request-id`` response header for support.
  - ``max_completion_tokens`` counts visible output AND reasoning tokens
    (unlike xAI's, which excludes reasoning); ``max_tokens`` is deprecated and
    rejected by reasoning models.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.observability.audit import record_provider_call, record_tokens
from app.observability.provider_errors import (
    ErrorKind,
    ProviderFailure,
    classify_exception,
    classify_http,
    log_chart_extraction_failure,
)
from app.observability.tracing import set_attributes, start_span, tracer
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

# Same wording and behavior as the other providers' constant of the same name,
# kept per-provider so no provider module imports another vendor's module.
TOOL_LOOP_EXHAUSTED_REPLY = (
    "I wasn't able to finish answering using the available tools within the "
    "allowed number of steps. Please try rephrasing your question or "
    "breaking it into smaller parts."
)

_REQUEST_TIMEOUT_SECONDS = 180.0
# Includes reasoning tokens on this endpoint, so it is larger than the other
# providers' caps: a reasoning model must not spend the whole budget thinking.
_MAX_COMPLETION_TOKENS = 4096
_SSE_DATA_PREFIX = "data: "
_SSE_DONE = "[DONE]"


# Azure's strict structured outputs reject a schema unless every object sets
# "additionalProperties": false and lists every property as required (live 400:
# "'additionalProperties' is required to be supplied and to be false"), so these
# models forbid extra fields and `chart` is required but nullable.
class _ChartSeriesSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    values: list[float]


class _ChartSourceChunkSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_title: str
    chunk_index: int


class _ChartPayloadSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_type: Literal["bar", "line", "pie"]
    title: str
    labels: list[str]
    series: list[_ChartSeriesSchema]
    source_chunks: list[_ChartSourceChunkSchema]


class _ChartExtractionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart: _ChartPayloadSchema | None


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
    return bool(prompt.retrieved_chunks)


def _build_tools_payload(enabled_tools: tuple[str, ...]) -> list[dict[str, Any]]:
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
    messages: list[dict[str, Any]] = [{"role": "system", "content": prompt.system_prompt}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in prompt.prior_turns)
    messages.append({"role": "user", "content": prompt.user_message})
    return messages


def _parse_json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _raise_for_error(
    status_code: int,
    body: Any,
    *,
    model: str,
    operation: str,
    request_id: str | None,
) -> None:
    """An error status, or an error body under any status (checked rather
    than assumed away, as OpenRouter returned one under HTTP 200)."""
    has_error_body = isinstance(body, dict) and body.get("error") is not None
    if status_code < 400 and not has_error_body:
        return
    error = body.get("error") if isinstance(body, dict) else None
    detail = error.get("message") if isinstance(error, dict) else None
    suffix = f": {detail}" if detail else ""
    support = f" (apim-request-id={request_id})" if request_id else ""
    raise ModelProviderError(
        f"Azure AI returned an error (status={status_code}){suffix}{support}",
        failure=classify_http(
            provider="azure_ai", status=status_code, body=body, operation=operation, model=model
        ),
    )


class AzureAIProvider:
    """TEMP testing provider: Azure AI Foundry's OpenAI-compatible v1 chat
    completions over httpx. Tool calling is a client-side loop, as for
    OpenRouter and xAI. Charts use ``response_format: json_schema``, which the
    reference documents as ensuring the output matches the schema; the result
    is still validated locally, and ChatOrchestrator re-checks its sources."""

    def __init__(self, *, model_name: str, api_key: str | None, endpoint: str) -> None:
        if not api_key:
            raise ModelProviderError("Azure AI API key is not configured — set AZURE_AI_API_KEY")
        self._model_name = model_name
        self._api_key = api_key
        self._url = f"{endpoint.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"api-key": self._api_key, "Content-Type": "application/json"}

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
        """Streams a plain text turn. Built against a live capture of this
        endpoint's SSE (2026-09-30), which differs from OpenAI's and
        OpenRouter's: ``data:`` frames only (no ``:`` keep-alive comments),
        terminated by ``data: [DONE]``, but with two choice-less Azure frames
        (a leading ``prompt_filter_results`` frame and a trailing usage frame
        carrying ``latency_checkpoint``/``routing``), an empty-content role
        frame first, and ``content_filter_results``/``obfuscation`` on every
        chunk.

        A turn with tools runs the non-streamed tool loop first (which sends
        reasoning_effort "none", required for tools on gpt-6-luna) and then
        delivers the answer as one delta, as the OpenRouter and xAI providers
        do. A chart is a separate structured-output call after the text."""
        tools_payload = _build_tools_payload(prompt.enabled_tools)
        if tools_payload:
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
            "stream_options": {"include_usage": True},
            "max_completion_tokens": _MAX_COMPLETION_TOKENS,
        }
        parts: list[str] = []
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        # Created unentered and ended in finally: this generator yields across
        # the orchestrator's span boundaries, where entering a span context
        # here would be detached in the wrong context.
        span = tracer.start_span(
            "azure_ai.chat_completions",
            attributes={"provider": "azure_ai", "model": self._model_name, "streaming": True},
        )
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                async with client.stream(
                    "POST", self._url, headers=self._headers(), json=payload
                ) as response:
                    request_id = response.headers.get("apim-request-id")
                    set_attributes(
                        span,
                        {
                            "http.status_code": response.status_code,
                            "azure.apim_request_id": request_id,
                        },
                    )
                    record_provider_call("model", "azure_ai", self._model_name)
                    if response.status_code >= 400:
                        raw = await response.aread()
                        _raise_for_error(
                            response.status_code,
                            _parse_json_body(raw),
                            model=self._model_name,
                            operation="chat",
                            request_id=request_id,
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith(_SSE_DATA_PREFIX):
                            continue  # blank separators (and any comment lines)
                        data = line[len(_SSE_DATA_PREFIX) :]
                        if data == _SSE_DONE:
                            break
                        event = _parse_json_body(data.encode("utf-8"))
                        if not isinstance(event, dict):
                            continue
                        _raise_for_error(
                            response.status_code,
                            event,
                            model=self._model_name,
                            operation="chat",
                            request_id=request_id,
                        )
                        if isinstance(event.get("usage"), dict):
                            usage = event["usage"]
                        choices = event.get("choices") or []
                        if not choices:
                            continue  # prompt_filter_results frame or usage frame
                        choice = choices[0] or {}
                        finish_reason = choice.get("finish_reason") or finish_reason
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            parts.append(content)
                            yield ModelStreamEvent(delta=content)
        except ModelProviderError:
            raise
        except Exception as exc:  # network/timeout/TLS failures from httpx
            raise ModelProviderError(
                "Azure AI request failed",
                failure=classify_exception(
                    exc, provider="azure_ai", operation="chat", model=self._model_name
                ),
            ) from exc
        finally:
            set_attributes(
                span,
                {
                    "finish_reason": finish_reason,
                    "usage.prompt_tokens": usage.get("prompt_tokens"),
                    "usage.completion_tokens": usage.get("completion_tokens"),
                },
            )
            span.end()

        record_tokens(
            "model",
            "azure_ai",
            self._model_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        text = "".join(parts).strip()
        if finish_reason == "content_filter":
            # Text already streamed can't be unsent; the orchestrator turns
            # this into an error event and persists nothing.
            self._extract_text({"content": None, "_finish_reason": finish_reason})
        if not text:
            self._extract_text({"content": None, "_finish_reason": finish_reason})

        chart = await self._maybe_generate_chart(prompt)
        yield ModelStreamEvent(is_final=True, text=text, grounded=_is_grounded(prompt), chart=chart)

    async def _negotiate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        max_tool_calls: int,
    ) -> str:
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
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()
        finish = message.get("_finish_reason")
        if finish == "length":
            raise ModelProviderError(
                "Azure AI returned no text: the token budget ran out (reasoning counts toward "
                "max_completion_tokens)",
                failure=ProviderFailure(
                    kind=ErrorKind.MALFORMED_RESPONSE,
                    provider="azure_ai",
                    operation="chat",
                    model=self._model_name,
                    message="finish_reason=length with empty content",
                ),
            )
        if finish == "content_filter":
            raise ModelProviderError(
                "Azure AI withheld the response (content filter)",
                failure=ProviderFailure(
                    kind=ErrorKind.BAD_REQUEST,
                    provider="azure_ai",
                    operation="chat",
                    model=self._model_name,
                    message="content filter: finish_reason=content_filter",
                ),
            )
        raise ModelProviderError("Azure AI returned an empty response")

    async def _chat_completions_once(
        self,
        messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None = None,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        operation = "chart" if response_format is not None else "chat"
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": _MAX_COMPLETION_TOKENS,
        }
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"
            # Live 400 for gpt-6-luna: "Function tools with reasoning_effort
            # are not supported ... in /v1/chat/completions. To use function
            # tools, use /v1/responses or set reasoning_effort to 'none'."
            # Only tool-carrying requests set it; others keep the default.
            payload["reasoning_effort"] = "none"
        if response_format is not None:
            payload["response_format"] = response_format

        with start_span(
            "azure_ai.chat_completions",
            provider="azure_ai",
            model=self._model_name,
            operation=operation,
            tools=bool(tools_payload),
        ) as span:
            try:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                    response = await client.post(self._url, headers=self._headers(), json=payload)
            except Exception as exc:  # network/timeout/TLS failures from httpx
                record_provider_call("model", "azure_ai", self._model_name)
                raise ModelProviderError(
                    "Azure AI request failed",
                    failure=classify_exception(
                        exc, provider="azure_ai", operation=operation, model=self._model_name
                    ),
                ) from exc

            request_id = response.headers.get("apim-request-id")
            body = _parse_json_body(response.content)
            usage = body.get("usage") if isinstance(body, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            record_provider_call(
                "model",
                "azure_ai",
                self._model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
            set_attributes(
                span,
                {
                    "http.status_code": response.status_code,
                    "azure.apim_request_id": request_id,
                    "usage.prompt_tokens": usage.get("prompt_tokens"),
                    "usage.completion_tokens": usage.get("completion_tokens"),
                },
            )
            _raise_for_error(
                response.status_code,
                body,
                model=self._model_name,
                operation=operation,
                request_id=request_id,
            )

            try:
                choice = body["choices"][0]
                message = dict(choice["message"])
            except (KeyError, IndexError, TypeError) as exc:
                keys = sorted(body) if isinstance(body, dict) else type(body).__name__
                raise ModelProviderError(
                    f"Azure AI returned an unexpected response shape "
                    f"(status={response.status_code}, keys={keys})",
                    failure=ProviderFailure(
                        kind=ErrorKind.MALFORMED_RESPONSE,
                        provider="azure_ai",
                        operation=operation,
                        model=self._model_name,
                        status=response.status_code,
                        message=f"no choices; keys={keys}",
                    ),
                ) from exc
            message["_finish_reason"] = choice.get("finish_reason")
            span.set_attribute("finish_reason", str(choice.get("finish_reason")))
            return message

    async def _maybe_generate_chart(self, prompt: GroundedPrompt) -> ChartSpec | None:
        if not prompt.chart_requested or not prompt.retrieved_chunks:
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
            message = await self._chat_completions_once(messages, response_format=response_format)
            parsed = _ChartExtractionSchema.model_validate_json(message.get("content") or "")
        except (ModelProviderError, ValidationError, ValueError) as exc:
            log_chart_extraction_failure(logger, exc, provider="azure_ai", model=self._model_name)
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
