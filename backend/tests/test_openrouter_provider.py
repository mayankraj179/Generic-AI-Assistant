from __future__ import annotations

import json

import httpx
import pytest

from app.orchestration.model_provider import ConversationTurn, GroundedPrompt, ModelProviderError
from app.orchestration.openrouter_provider import OpenRouterProvider

# ---------------------------------------------------------------------------
# Test helpers — a real httpx.AsyncClient backed by httpx.MockTransport
# (built into httpx itself, no extra dependency) so request shape and error
# handling are exercised through the real httpx request/response machinery,
# not a hand-rolled fake.
# ---------------------------------------------------------------------------


_RealAsyncClient = httpx.AsyncClient


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    captured: list[httpx.Request] = []

    def _capturing_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    def _fake_async_client(**kwargs) -> httpx.AsyncClient:
        return _RealAsyncClient(transport=httpx.MockTransport(_capturing_handler))

    import app.orchestration.openrouter_provider as module

    monkeypatch.setattr(module.httpx, "AsyncClient", _fake_async_client)
    return captured


def _json_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _sse_response(status_code: int, lines: list[str]) -> httpx.Response:
    body = "\n".join(lines).encode("utf-8")
    return httpx.Response(status_code, content=body, headers={"content-type": "text/event-stream"})


def _prompt(**overrides) -> GroundedPrompt:
    defaults = dict(
        system_prompt="You are a helpful assistant.",
        user_message="What is our Q2 revenue?",
        retrieved_chunks=(),
        prior_turns=(),
    )
    defaults.update(overrides)
    return GroundedPrompt(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_model_provider_error():
    with pytest.raises(ModelProviderError):
        OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key=None)


def test_empty_api_key_raises_model_provider_error():
    with pytest.raises(ModelProviderError):
        OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="")


# ---------------------------------------------------------------------------
# generate() — request shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_sends_expected_request_shape(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {"choices": [{"index": 0, "message": {"role": "assistant", "content": "42."}}]},
        )

    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    prompt = _prompt(
        prior_turns=(ConversationTurn(role="user", content="Hi"),
                     ConversationTurn(role="assistant", content="Hello!")),
    )
    reply = await provider.generate(prompt)

    assert reply.text == "42."
    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"

    body = json.loads(request.content)
    assert body["model"] == "anthropic/claude-sonnet-5"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is our Q2 revenue?"},
    ]


@pytest.mark.asyncio
async def test_generate_is_grounded_only_when_chunks_present(monkeypatch: pytest.MonkeyPatch):
    from app.ingestion.pipeline import Chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200, {"choices": [{"message": {"content": "Answer."}}]}
        )

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    chunk = Chunk(
        document_title="doc",
        chunk_index=0,
        display_text="text",
        embedded_text="doc: text",
        access_labels=frozenset({"role:employee"}),
    )
    reply = await provider.generate(_prompt(retrieved_chunks=(chunk,)))
    assert reply.grounded is True

    reply = await provider.generate(_prompt(retrieved_chunks=()))
    assert reply.grounded is False


@pytest.mark.asyncio
async def test_generate_never_returns_a_chart(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"choices": [{"message": {"content": "Answer."}}]})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    reply = await provider.generate(_prompt(chart_requested=True))
    assert reply.chart is None


# ---------------------------------------------------------------------------
# generate() — error wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_wraps_http_error_status_with_openrouter_error_detail(
    monkeypatch: pytest.MonkeyPatch,
):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            401,
            {"error": {"code": 401, "message": "Invalid credentials"}},
        )

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="bad-key")

    with pytest.raises(ModelProviderError, match="Invalid credentials"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_rate_limit_error(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(429, {"error": {"code": 429, "message": "Rate limited"}})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    with pytest.raises(ModelProviderError, match="status=429"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_malformed_response_shape(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"unexpected": "shape"})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    with pytest.raises(ModelProviderError, match="unexpected response shape"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_empty_text_response(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"choices": [{"message": {"content": "   "}}]})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    with pytest.raises(ModelProviderError, match="empty response"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_transport_level_failure(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    with pytest.raises(ModelProviderError, match="OpenRouter request failed"):
        await provider.generate(_prompt())


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stream_yields_deltas_then_final_event(monkeypatch: pytest.MonkeyPatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        ": OPENROUTER PROCESSING",
        'data: {"choices":[{"delta":{"content":"lo!"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":12}}',
        "data: [DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(200, lines)

    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    events = [event async for event in provider.generate_stream(_prompt())]

    deltas = [e.delta for e in events if not e.is_final]
    assert deltas == ["Hel", "lo!"]
    assert events[-1].is_final is True
    assert events[-1].text == "Hello!"
    assert events[-1].chart is None

    body = json.loads(captured[0].content)
    assert body["stream"] is True


@pytest.mark.asyncio
async def test_generate_stream_wraps_empty_stream_as_error(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(200, ["data: [DONE]"])

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    with pytest.raises(ModelProviderError, match="empty response"):
        async for _ in provider.generate_stream(_prompt()):
            pass


@pytest.mark.asyncio
async def test_generate_stream_wraps_http_error_status(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(402, {"error": {"code": 402, "message": "Insufficient credits"}})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="anthropic/claude-sonnet-5", api_key="test-key")

    with pytest.raises(ModelProviderError, match="Insufficient credits"):
        async for _ in provider.generate_stream(_prompt()):
            pass


# ---------------------------------------------------------------------------
# Tool-calling loop
# ---------------------------------------------------------------------------


def _assistant_message_with_tool_call(name: str, arguments: dict) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
    }


def _assistant_final_message(text: str) -> dict:
    return {"message": {"role": "assistant", "content": text}}


def _sequenced_json_handler(responses: list[dict]):
    """Returns a handler that replies with each response body in order, one
    per request — simulates a multi-round tool-call negotiation."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = calls["count"]
        calls["count"] += 1
        body = {"choices": [responses[index]]}
        return _json_response(200, body)

    return handler, calls


@pytest.mark.asyncio
async def test_generate_with_no_tool_call_makes_a_single_request(monkeypatch: pytest.MonkeyPatch):
    handler, calls = _sequenced_json_handler([_assistant_final_message("42.")])
    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",)))

    assert reply.text == "42."
    assert calls["count"] == 1
    body = json.loads(captured[0].content)
    assert body["tools"][0]["function"]["name"] == "calculate"
    assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_generate_executes_a_real_tool_call_and_feeds_result_back(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_json_handler(
        [
            _assistant_message_with_tool_call("calculate", {"expression": "2 + 2"}),
            _assistant_final_message("The answer is 4."),
        ]
    )
    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",)))

    assert reply.text == "The answer is 4."
    assert calls["count"] == 2

    # Second request must carry the assistant's tool_calls message plus a
    # "tool" role message with the REAL calculate() result, not a canned one.
    second_body = json.loads(captured[1].content)
    messages = second_body["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "calculate"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_1"
    tool_result = json.loads(messages[-1]["content"])
    assert tool_result == {"result": 4.0, "expression": "2 + 2", "kind": "arithmetic"}


@pytest.mark.asyncio
async def test_generate_tool_loop_rejects_injection_shaped_expression_live_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """Same injection-rejection proof as test_tool_builtin.py, but through
    the actual OpenRouter tool-call plumbing (JSON-string arguments ->
    execute_tool -> calculate()) rather than calling calculate() directly."""
    handler, calls = _sequenced_json_handler(
        [
            _assistant_message_with_tool_call(
                "calculate", {"expression": "__import__('os').system('ls')"}
            ),
            _assistant_final_message("I can't compute that."),
        ]
    )
    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",)))

    assert reply.text == "I can't compute that."
    second_body = json.loads(captured[1].content)
    tool_result = json.loads(second_body["messages"][-1]["content"])
    assert "error" in tool_result
    assert "result" not in tool_result


@pytest.mark.asyncio
async def test_generate_tool_loop_stops_at_max_tool_calls(monkeypatch: pytest.MonkeyPatch):
    # Every response proposes another tool call — the model never "finishes"
    # — so the loop must stop at the configured bound, not run forever.
    handler, calls = _sequenced_json_handler(
        [_assistant_message_with_tool_call("calculate", {"expression": "1 + 1"})] * 10
    )
    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",), max_tool_calls=3))

    assert calls["count"] == 3
    assert "wasn't able to finish" in reply.text


@pytest.mark.asyncio
async def test_generate_tool_unknown_name_returns_structured_error_to_model(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_json_handler(
        [
            _assistant_message_with_tool_call("not_a_real_tool", {}),
            _assistant_final_message("Sorry, I couldn't do that."),
        ]
    )
    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",), max_tool_calls=3))

    assert reply.text == "Sorry, I couldn't do that."
    second_body = json.loads(captured[1].content)
    tool_result = json.loads(second_body["messages"][-1]["content"])
    assert "error" in tool_result


@pytest.mark.asyncio
async def test_generate_degrades_gracefully_when_model_rejects_tools(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls["count"] += 1
        if "tools" in body:
            return _json_response(
                400, {"error": {"code": 400, "message": "this model does not support tools"}}
            )
        return _json_response(200, {"choices": [_assistant_final_message("A plain answer.")]})

    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",)))

    assert reply.text == "A plain answer."
    assert calls["count"] == 2
    assert "tools" not in json.loads(captured[1].content)


@pytest.mark.asyncio
async def test_generate_stream_with_tools_negotiates_then_streams_final_text_as_one_delta(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_json_handler(
        [
            _assistant_message_with_tool_call("calculate", {"expression": "3 * 3"}),
            _assistant_final_message("Nine."),
        ]
    )
    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    events = [
        event
        async for event in provider.generate_stream(_prompt(enabled_tools=("calculate",)))
    ]

    assert calls["count"] == 2  # negotiation ran non-streamed
    deltas = [e.delta for e in events if not e.is_final]
    assert deltas == ["Nine."]
    assert events[-1].is_final is True
    assert events[-1].text == "Nine."


@pytest.mark.asyncio
async def test_generate_without_enabled_tools_never_sends_tools_field(
    monkeypatch: pytest.MonkeyPatch,
):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" not in body
        return _json_response(200, {"choices": [_assistant_final_message("no tools here")]})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt())  # enabled_tools defaults to ()
    assert reply.text == "no tools here"
