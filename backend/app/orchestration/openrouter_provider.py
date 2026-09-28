from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.orchestration.model_provider import (
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
    messages: list[dict[str, str]] = [{"role": "system", "content": prompt.system_prompt}]
    messages.extend({"role": turn.role, "content": turn.content} for turn in prompt.prior_turns)
    messages.append({"role": "user", "content": prompt.user_message})
    return messages


def _parse_json_body(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _raise_for_error(status_code: int, body: Any) -> None:
    """Wraps a non-2xx OpenRouter response into ModelProviderError.

    OpenRouter's documented error shape is ``{"error": {"code", "message",
    "metadata"}}`` (openrouter.ai/docs/api_reference/errors-and-debugging) —
    distinct from Gemini's ADK event.error_message/error_code, but the same
    "never let vendor detail leak past this provider" discipline applies.
    """
    if status_code < 400:
        return
    detail: str | None = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                detail = message
    suffix = f": {detail}" if detail else ""
    raise ModelProviderError(f"OpenRouter returned an error (status={status_code}){suffix}")


class OpenRouterProvider:
    """A second ``ModelProvider`` implementation, alongside GeminiProvider.

    Talks to OpenRouter's HTTP API directly via httpx (OpenRouter has no
    first-party Python SDK equivalent to google-genai) rather than an SDK,
    unlike GeminiProvider. All HTTP/auth/rate-limit failures are wrapped into
    ModelProviderError — never a raw httpx exception escapes this class.

    Chart generation is intentionally NOT implemented here (generate()/
    generate_stream() always return ``chart=None``): OpenRouter proxies many
    different underlying models, and structured-output support
    (``response_format``/``structured_outputs``) is per-model-endpoint, not
    universal — confirmed live via GET /api/v1/models's
    ``supported_parameters``, where plenty of catalog entries omit
    "structured_outputs" entirely. Silently attempting chart extraction
    against an arbitrary OpenRouter-routed model would mean it either fails
    unpredictably or (worse) a model without real structured-output support
    "successfully" returns free-text dressed up as JSON. Rather than guess
    per-model or maintain a hand-curated allowlist, this provider simply
    never attempts it — a real capability gap for OpenRouter-backed
    assistants today, not an oversight.
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
        return ModelReply(text=text, grounded=_is_grounded(prompt), chart=None)

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
        self, messages: list[dict[str, Any]], tools_payload: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        if tools_payload:
            payload["tools"] = tools_payload
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    _CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
                )
        except ModelProviderError:
            raise
        except Exception as exc:  # network/timeout/TLS failures from httpx
            raise ModelProviderError("OpenRouter request failed") from exc

        body = _parse_json_body(response.content)
        _raise_for_error(response.status_code, body)

        try:
            return body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelProviderError("OpenRouter returned an unexpected response shape") from exc

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
            yield ModelStreamEvent(
                is_final=True, text=text, grounded=_is_grounded(prompt), chart=None
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
                    if response.status_code >= 400:
                        raw = await response.aread()
                        _raise_for_error(response.status_code, _parse_json_body(raw))

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
            raise ModelProviderError("OpenRouter request failed") from exc

        final_text = "".join(final_text_parts).strip()
        if not final_text:
            raise ModelProviderError("OpenRouter returned an empty response")

        yield ModelStreamEvent(
            is_final=True, text=final_text, grounded=_is_grounded(prompt), chart=None
        )
