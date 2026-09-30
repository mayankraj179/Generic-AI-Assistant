from __future__ import annotations

import json

import httpx
import pytest

from app.ingestion.pipeline import Chunk
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
async def test_generate_returns_no_chart_when_requested_but_nothing_retrieved(
    monkeypatch: pytest.MonkeyPatch,
):
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


# ---------------------------------------------------------------------------
# Chart generation via structured outputs
# ---------------------------------------------------------------------------


def _finance_chunk() -> Chunk:
    text = "FY2021 42.3 | FY2022 51.8 | FY2023 63.4 | FY2024 79.1 | FY2025 96.5"
    return Chunk(
        document_title="nova_horizon_fy2025",
        chunk_index=0,
        display_text=text,
        embedded_text=f"nova_horizon_fy2025: {text}",
        access_labels=frozenset({"role:authenticated"}),
    )


_REVENUE_CHART = {
    "chart": {
        "chart_type": "line",
        "title": "Revenue FY2021-FY2025",
        "labels": ["FY2021", "FY2022", "FY2023", "FY2024", "FY2025"],
        "series": [{"name": "Revenue ($M)", "values": [42.3, 51.8, 63.4, 79.1, 96.5]}],
        "source_chunks": [{"document_title": "nova_horizon_fy2025", "chunk_index": 0}],
    }
}


def _sequenced_raw_handler(responses: list[tuple[int, dict]]):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = responses[calls["count"]]
        calls["count"] += 1
        return _json_response(status, body)

    return handler, calls


def _answer(text: str) -> tuple[int, dict]:
    return 200, {"choices": [{"message": {"role": "assistant", "content": text}}]}


@pytest.mark.asyncio
async def test_no_chart_call_and_no_chart_guidance_when_chart_not_requested(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_raw_handler([_answer("Revenue was $96.5M.")])
    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(_prompt(retrieved_chunks=(_finance_chunk(),)))

    assert reply.chart is None
    assert calls["count"] == 1
    body = json.loads(captured[0].content)
    assert "renders any chart itself" not in body["messages"][0]["content"]
    assert "response_format" not in body and "provider" not in body


@pytest.mark.asyncio
async def test_chart_turn_makes_strict_schema_call_that_requires_supporting_endpoints(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_raw_handler(
        [_answer("Revenue grew from 42.3 to 96.5."), _answer(json.dumps(_REVENUE_CHART))]
    )
    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(
        _prompt(retrieved_chunks=(_finance_chunk(),), chart_requested=True)
    )

    assert calls["count"] == 2
    answer_body = json.loads(captured[0].content)
    # Chart-turn wording guidance is the orchestrator's job for every
    # provider; this provider passes the system prompt through unchanged.
    assert answer_body["messages"][0]["content"] == "You are a helpful assistant."
    assert "response_format" not in answer_body

    chart_body = json.loads(captured[1].content)
    assert chart_body["provider"] == {"require_parameters": True}
    assert chart_body["response_format"]["type"] == "json_schema"
    assert chart_body["response_format"]["json_schema"]["strict"] is True
    assert "[source: nova_horizon_fy2025, chunk 0]" in chart_body["messages"][-1]["content"]
    assert "tools" not in chart_body

    assert reply.text == "Revenue grew from 42.3 to 96.5."
    assert reply.chart is not None
    assert reply.chart.chart_type == "line"
    assert reply.chart.series[0].values == [42.3, 51.8, 63.4, 79.1, 96.5]
    assert reply.chart.source_chunks[0].document_title == "nova_horizon_fy2025"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chart_response",
    [
        (404, {"error": {"message": "No endpoints found that support the requested parameters"}}),
        (200, {"id": "gen-1", "error": {"message": "Upstream error: overloaded", "code": 503}}),
        _answer("Sure! Here's your chart: revenue went up."),
        _answer(json.dumps({"chart": {"chart_type": "radar"}})),
        _answer(json.dumps({"chart": None})),
        _answer(
            json.dumps(
                {
                    "chart": {
                        **_REVENUE_CHART["chart"],
                        "chart_type": "pie",
                        "series": [{"name": "a", "values": [1]}, {"name": "b", "values": [2]}],
                    }
                }
            )
        ),
    ],
    ids=["no-supporting-endpoint", "200-embedded-error", "free-text", "schema-invalid",
         "null", "bad-pie"],
)
async def test_any_chart_failure_yields_no_chart_and_keeps_the_text_answer(
    monkeypatch: pytest.MonkeyPatch, chart_response
):
    handler, _ = _sequenced_raw_handler([_answer("Revenue was $96.5M."), chart_response])
    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    reply = await provider.generate(
        _prompt(retrieved_chunks=(_finance_chunk(),), chart_requested=True)
    )

    assert reply.text == "Revenue was $96.5M."
    assert reply.chart is None


@pytest.mark.asyncio
async def test_generate_stream_final_event_carries_the_chart(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            return _sse_response(
                200,
                [
                    'data: {"choices":[{"delta":{"content":"Revenue "}}]}',
                    'data: {"choices":[{"delta":{"content":"rose."}}]}',
                    "data: [DONE]",
                ],
            )
        chart_message = {"message": {"content": json.dumps(_REVENUE_CHART)}}
        return _json_response(200, {"choices": [chart_message]})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterProvider(model_name="m", api_key="test-key")

    events = [
        e
        async for e in provider.generate_stream(
            _prompt(retrieved_chunks=(_finance_chunk(),), chart_requested=True)
        )
    ]

    assert [e.delta for e in events if not e.is_final] == ["Revenue ", "rose."]
    assert events[-1].is_final
    assert events[-1].text == "Revenue rose."
    assert events[-1].chart is not None
    assert events[-1].chart.labels[-1] == "FY2025"
