from __future__ import annotations

import json

import httpx
import pytest

from app.config.assistant_config import ModelCapabilities
from app.config.settings import Settings
from app.ingestion.pipeline import Chunk
from app.orchestration.grok_provider import TOOL_LOOP_EXHAUSTED_REPLY, GrokProvider
from app.orchestration.model_provider import ConversationTurn, GroundedPrompt, ModelProviderError
from app.orchestration.provider_factory import get_model_provider

# ---------------------------------------------------------------------------
# Test helpers — a real httpx.AsyncClient backed by httpx.MockTransport, the
# same approach as tests/test_openrouter_provider.py.
# ---------------------------------------------------------------------------


_RealAsyncClient = httpx.AsyncClient


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    captured: list[httpx.Request] = []

    def _capturing_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    def _fake_async_client(**kwargs) -> httpx.AsyncClient:
        return _RealAsyncClient(transport=httpx.MockTransport(_capturing_handler))

    import app.orchestration.grok_provider as module

    monkeypatch.setattr(module.httpx, "AsyncClient", _fake_async_client)
    return captured


def _json_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _completion(content: str | None, **message_fields) -> dict:
    return {
        "id": "resp-1",
        "object": "chat.completion",
        "model": "grok-4.3",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **message_fields},
                "finish_reason": "stop",
            }
        ],
    }


def _chunk(text: str = "Revenue was $96.5M.", index: int = 0) -> Chunk:
    return Chunk(
        document_title="nova_horizon_fy2025",
        chunk_index=index,
        display_text=text,
        embedded_text=f"nova_horizon_fy2025: {text}",
        access_labels=frozenset({"role:authenticated"}),
    )


def _prompt(**overrides) -> GroundedPrompt:
    defaults = dict(
        system_prompt="You are a helpful assistant.",
        user_message="What is our Q2 revenue?",
        retrieved_chunks=(),
        prior_turns=(),
    )
    defaults.update(overrides)
    return GroundedPrompt(**defaults)


def _sequenced_handler(bodies: list[tuple[int, dict]]):
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = bodies[calls["count"]]
        calls["count"] += 1
        return _json_response(status, body)

    return handler, calls


def _tool_call_completion(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return _completion(
        None,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


# ---------------------------------------------------------------------------
# Construction and factory wiring
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_model_provider_error():
    with pytest.raises(ModelProviderError):
        GrokProvider(model_name="grok-4.3", api_key=None)


def test_empty_api_key_raises_model_provider_error():
    with pytest.raises(ModelProviderError):
        GrokProvider(model_name="grok-4.3", api_key="")


def test_factory_resolves_xai_to_grok_provider():
    provider = get_model_provider(
        model=ModelCapabilities(provider="xai", model_name="grok-4.3"),
        settings=Settings(xai_api_key="test-key"),
    )
    assert isinstance(provider, GrokProvider)


# ---------------------------------------------------------------------------
# generate() — request shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_sends_expected_request_shape(monkeypatch: pytest.MonkeyPatch):
    captured = _install_transport(monkeypatch, lambda r: _json_response(200, _completion("Hi.")))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(
        _prompt(prior_turns=(ConversationTurn(role="user", content="earlier"),
                             ConversationTurn(role="assistant", content="reply")))
    )

    assert reply.text == "Hi."
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.x.ai/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "grok-4.3"
    assert body["stream"] is False
    # max_tokens is deprecated on xAI's endpoint; the replacement is used.
    assert body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
    assert body["messages"][-1]["content"] == "What is our Q2 revenue?"
    assert "tools" not in body
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_generate_is_grounded_only_when_chunks_present(monkeypatch: pytest.MonkeyPatch):
    _install_transport(monkeypatch, lambda r: _json_response(200, _completion("ok")))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    assert (await provider.generate(_prompt())).grounded is False
    assert (await provider.generate(_prompt(retrieved_chunks=(_chunk(),)))).grounded is True


# ---------------------------------------------------------------------------
# generate() — error wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_body",
    [
        {"error": "Incorrect API key provided"},
        {"code": "Client specified an invalid argument", "error": "Incorrect API key provided"},
        {"error": {"message": "Incorrect API key provided"}},
    ],
)
async def test_generate_wraps_http_error_with_detail_in_any_documented_shape(
    monkeypatch: pytest.MonkeyPatch, error_body
):
    _install_transport(monkeypatch, lambda r: _json_response(401, error_body))
    provider = GrokProvider(model_name="grok-4.3", api_key="bad-key")

    with pytest.raises(ModelProviderError, match=r"status=401.*Incorrect API key provided"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_rate_limit_error(monkeypatch: pytest.MonkeyPatch):
    _install_transport(monkeypatch, lambda r: _json_response(429, {"error": "rate limited"}))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    with pytest.raises(ModelProviderError, match="status=429"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_treats_200_with_embedded_error_body_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # The failure mode OpenRouter returned live: HTTP 200 whose body is an
    # error, not a completion. It must surface the real message.
    body = {"id": "gen-1", "error": {"message": "Service temporarily overloaded", "code": 503}}
    _install_transport(monkeypatch, lambda r: _json_response(200, body))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    with pytest.raises(ModelProviderError, match=r"status=200.*Service temporarily overloaded"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_malformed_response_shape_and_names_its_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_transport(monkeypatch, lambda r: _json_response(200, {"id": "x", "object": "?"}))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    with pytest.raises(
        ModelProviderError, match=r"unexpected response shape.*keys=\['id', 'object'\]"
    ):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_non_json_body(monkeypatch: pytest.MonkeyPatch):
    _install_transport(monkeypatch, lambda r: httpx.Response(502, content=b"<html>bad gateway"))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    with pytest.raises(ModelProviderError, match="status=502"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_wraps_empty_text_response(monkeypatch: pytest.MonkeyPatch):
    _install_transport(monkeypatch, lambda r: _json_response(200, _completion("   ")))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    with pytest.raises(ModelProviderError, match="empty response"):
        await provider.generate(_prompt())


@pytest.mark.asyncio
async def test_generate_returns_model_refusal_text_when_content_is_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    body = _completion(None, refusal="I can't help with that.")
    _install_transport(monkeypatch, lambda r: _json_response(200, body))
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    assert (await provider.generate(_prompt())).text == "I can't help with that."


@pytest.mark.asyncio
async def test_generate_wraps_transport_level_failure(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    with pytest.raises(ModelProviderError, match="request failed"):
        await provider.generate(_prompt())


# ---------------------------------------------------------------------------
# generate() — tool calling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_with_tools_but_no_tool_call_makes_one_request(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_handler([(200, _completion("42."))])
    captured = _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",)))

    assert reply.text == "42."
    assert calls["count"] == 1
    body = json.loads(captured[0].content)
    assert body["tools"][0] == {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": body["tools"][0]["function"]["description"],
            "parameters": body["tools"][0]["function"]["parameters"],
        },
    }
    assert body["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_generate_executes_a_real_tool_call_and_feeds_result_back(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_handler(
        [
            (200, _tool_call_completion("calculate", {"expression": "2 + 2"})),
            (200, _completion("The answer is 4.")),
        ]
    )
    captured = _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",)))

    assert reply.text == "The answer is 4."
    assert calls["count"] == 2
    messages = json.loads(captured[1].content)["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "calculate"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_1"
    assert json.loads(messages[-1]["content"]) == {
        "result": 4.0,
        "expression": "2 + 2",
        "kind": "arithmetic",
    }
    # The tools array is resent on every request in the loop.
    assert json.loads(captured[1].content)["tools"]


@pytest.mark.asyncio
async def test_generate_tool_loop_stops_at_max_tool_calls(monkeypatch: pytest.MonkeyPatch):
    handler, calls = _sequenced_handler(
        [(200, _tool_call_completion("calculate", {"expression": "1 + 1"}))] * 3
    )
    _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(_prompt(enabled_tools=("calculate",), max_tool_calls=3))

    assert reply.text == TOOL_LOOP_EXHAUSTED_REPLY
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_generate_unknown_tool_name_returns_structured_error_to_model(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, _ = _sequenced_handler(
        [
            (200, _tool_call_completion("delete_everything", {})),
            (200, _completion("I can't do that.")),
        ]
    )
    captured = _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    await provider.generate(_prompt(enabled_tools=("calculate",)))

    tool_message = json.loads(captured[1].content)["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "error" in json.loads(tool_message["content"])


@pytest.mark.asyncio
async def test_generate_tool_call_with_invalid_json_arguments_reports_error_to_model(
    monkeypatch: pytest.MonkeyPatch,
):
    bad_call = _completion(
        None,
        tool_calls=[
            {"id": "c", "type": "function", "function": {"name": "calculate", "arguments": "{nope"}}
        ],
    )
    handler, _ = _sequenced_handler([(200, bad_call), (200, _completion("Sorry."))])
    captured = _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    await provider.generate(_prompt(enabled_tools=("calculate",)))

    tool_message = json.loads(captured[1].content)["messages"][-1]
    assert json.loads(tool_message["content"]) == {
        "error": "tool call arguments were not valid JSON"
    }


# ---------------------------------------------------------------------------
# generate() — chart generation via structured outputs
# ---------------------------------------------------------------------------


_CHART_JSON = {
    "chart": {
        "chart_type": "bar",
        "title": "FY2025",
        "labels": ["Revenue", "Net Profit"],
        "series": [{"name": "USD M", "values": [96.5, 18.2]}],
        "source_chunks": [{"document_title": "nova_horizon_fy2025", "chunk_index": 0}],
    }
}


@pytest.mark.asyncio
async def test_no_chart_call_when_chart_not_requested(monkeypatch: pytest.MonkeyPatch):
    handler, calls = _sequenced_handler([(200, _completion("Revenue was $96.5M."))])
    _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(_prompt(retrieved_chunks=(_chunk(),), chart_requested=False))

    assert reply.chart is None
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_no_chart_call_when_requested_but_nothing_retrieved(monkeypatch: pytest.MonkeyPatch):
    handler, calls = _sequenced_handler([(200, _completion("No data."))])
    _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(_prompt(chart_requested=True))

    assert reply.chart is None
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_chart_requested_makes_strict_json_schema_call_and_returns_chart(
    monkeypatch: pytest.MonkeyPatch,
):
    handler, calls = _sequenced_handler(
        [(200, _completion("Here are the figures.")), (200, _completion(json.dumps(_CHART_JSON)))]
    )
    captured = _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(_prompt(retrieved_chunks=(_chunk(),), chart_requested=True))

    assert reply.text == "Here are the figures."
    assert calls["count"] == 2
    chart_body = json.loads(captured[1].content)
    fmt = chart_body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "chart_extraction"
    assert "chart" in fmt["json_schema"]["schema"]["properties"]
    assert "[source: nova_horizon_fy2025, chunk 0]" in chart_body["messages"][-1]["content"]
    assert reply.chart is not None
    assert reply.chart.chart_type == "bar"
    assert reply.chart.series[0].values == [96.5, 18.2]
    assert reply.chart.source_chunks[0].document_title == "nova_horizon_fy2025"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_response",
    [
        (200, _completion("not json at all")),
        (200, _completion(json.dumps({"chart": {"chart_type": "radar"}}))),
        (500, {"error": "internal"}),
        (200, {"error": {"message": "overloaded"}}),
        (200, _completion(json.dumps({"chart": None}))),
        (
            200,
            _completion(
                json.dumps(
                    {
                        "chart": {
                            **_CHART_JSON["chart"],
                            "chart_type": "pie",
                            "series": [{"name": "a", "values": [1]}, {"name": "b", "values": [2]}],
                        }
                    }
                )
            ),
        ),
    ],
    ids=["unparseable", "schema-invalid", "http-500", "200-embedded-error", "null", "bad-pie"],
)
async def test_any_chart_failure_yields_no_chart_and_keeps_the_text_answer(
    monkeypatch: pytest.MonkeyPatch, second_response
):
    handler, _ = _sequenced_handler([(200, _completion("Revenue was $96.5M.")), second_response])
    _install_transport(monkeypatch, handler)
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    reply = await provider.generate(_prompt(retrieved_chunks=(_chunk(),), chart_requested=True))

    assert reply.text == "Revenue was $96.5M."
    assert reply.chart is None


# ---------------------------------------------------------------------------
# generate_stream() — not built yet; must fail loudly, not silently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stream_is_not_implemented_and_raises_model_provider_error():
    provider = GrokProvider(model_name="grok-4.3", api_key="test-key")

    with pytest.raises(ModelProviderError, match="not implemented"):
        async for _ in provider.generate_stream(_prompt()):
            pass
