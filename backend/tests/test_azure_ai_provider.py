from __future__ import annotations

import json

import httpx
import pytest

from app.config.assistant_config import ModelCapabilities
from app.config.settings import Settings
from app.ingestion.pipeline import Chunk
from app.observability.provider_errors import ErrorKind
from app.orchestration.azure_ai_provider import TOOL_LOOP_EXHAUSTED_REPLY, AzureAIProvider
from app.orchestration.model_provider import ConversationTurn, GroundedPrompt, ModelProviderError
from app.orchestration.provider_factory import get_model_provider

ENDPOINT = "https://hrinitiatives.services.ai.azure.com/openai/v1"

# Captured live from the hrinitiatives resource with a bogus key (2026-09-30),
# verbatim, including the apim-request-id header.
AZURE_401_BODY = {
    "error": {
        "code": "401",
        "message": "Access denied due to invalid subscription key or wrong API endpoint. Make "
        "sure to provide a valid key for an active subscription and use a correct regional API "
        "endpoint for your resource.",
    }
}
# NOT captured live (needs a working key and real load): Azure OpenAI's
# documented error format for a token-rate 429 and a content-filter 400.
AZURE_429_BODY = {
    "error": {
        "code": "429",
        "message": "Requests to the ChatCompletions_Create Operation have exceeded token rate "
        "limit of your current AIServices S0 pricing tier. Please retry after 22 seconds.",
    }
}
AZURE_CONTENT_FILTER_BODY = {
    "error": {
        "code": "content_filter",
        "message": "The response was filtered due to the prompt triggering Azure OpenAI's "
        "content management policy.",
        "param": "prompt",
    }
}

_RealAsyncClient = httpx.AsyncClient


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    captured: list[httpx.Request] = []

    def _capturing(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    import app.orchestration.azure_ai_provider as module

    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kwargs: _RealAsyncClient(transport=httpx.MockTransport(_capturing)),
    )
    return captured


def _completion(content, *, finish_reason="stop", **message_fields) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-6-luna",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, **message_fields},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }


def _json(status: int, body: dict, *, request_id: str = "req-123") -> httpx.Response:
    return httpx.Response(status, json=body, headers={"apim-request-id": request_id})


def _prompt(**overrides) -> GroundedPrompt:
    values = dict(system_prompt="You are helpful.", user_message="What is the stipend?")
    values.update(overrides)
    return GroundedPrompt(**values)


def _provider() -> AzureAIProvider:
    return AzureAIProvider(model_name="gpt-6-luna", api_key="test-key", endpoint=ENDPOINT)


def _sequenced(bodies: list[tuple[int, dict]]):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = bodies[calls["n"]]
        calls["n"] += 1
        return _json(status, body)

    return handler, calls


# --- construction and wiring -------------------------------------------------


def test_missing_api_key_raises_model_provider_error():
    with pytest.raises(ModelProviderError, match="AZURE_AI_API_KEY"):
        AzureAIProvider(model_name="gpt-6-luna", api_key=None, endpoint=ENDPOINT)


def test_factory_resolves_azure_ai_with_the_configured_endpoint():
    provider = get_model_provider(
        model=ModelCapabilities(provider="azure_ai", model_name="gpt-6-luna"),
        settings=Settings(azure_ai_api_key="k"),
    )
    assert isinstance(provider, AzureAIProvider)
    assert provider._url == f"{ENDPOINT}/chat/completions"


# --- request shape --------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_uses_api_key_header_and_max_completion_tokens(monkeypatch):
    captured = _install_transport(monkeypatch, lambda r: _json(200, _completion("$85 per day.")))

    reply = await _provider().generate(
        _prompt(
            prior_turns=(
                ConversationTurn(role="user", content="hi"),
                ConversationTurn(role="assistant", content="hello"),
            )
        )
    )

    assert reply.text == "$85 per day."
    request = captured[0]
    assert str(request.url) == f"{ENDPOINT}/chat/completions"
    assert "api-version" not in str(request.url)
    assert request.headers["api-key"] == "test-key"
    assert "authorization" not in {k.lower() for k in request.headers}
    body = json.loads(request.content)
    assert body["model"] == "gpt-6-luna"
    assert body["max_completion_tokens"] == 4096
    assert "max_tokens" not in body
    assert body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
    assert "tools" not in body and "response_format" not in body
    assert "reasoning_effort" not in body  # only tool-carrying requests set it


@pytest.mark.asyncio
async def test_grounded_only_when_chunks_present(monkeypatch):
    _install_transport(monkeypatch, lambda r: _json(200, _completion("ok")))
    chunk = Chunk(
        document_title="travel_policy",
        chunk_index=1,
        display_text="$85",
        embedded_text="travel_policy: $85",
        access_labels=frozenset({"role:authenticated"}),
    )
    assert (await _provider().generate(_prompt())).grounded is False
    assert (await _provider().generate(_prompt(retrieved_chunks=(chunk,)))).grounded is True


# --- errors -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_captured_401_is_invalid_key_with_support_id(monkeypatch):
    _install_transport(monkeypatch, lambda r: _json(401, AZURE_401_BODY, request_id="deeb287b"))

    with pytest.raises(ModelProviderError) as info:
        await _provider().generate(_prompt())

    assert "status=401" in str(info.value) and "apim-request-id=deeb287b" in str(info.value)
    failure = info.value.failure
    assert failure.kind is ErrorKind.INVALID_KEY
    assert failure.provider == "azure_ai"
    assert "AZURE_AI_API_KEY" in failure.summary()


@pytest.mark.asyncio
async def test_documented_429_is_rate_limited_with_retry_after(monkeypatch):
    _install_transport(monkeypatch, lambda r: _json(429, AZURE_429_BODY))

    with pytest.raises(ModelProviderError) as info:
        await _provider().generate(_prompt())

    assert info.value.failure.kind is ErrorKind.RATE_LIMITED
    assert info.value.failure.retry_after_s == 22


@pytest.mark.asyncio
async def test_content_filter_rejection_is_bad_request_and_says_so(monkeypatch):
    _install_transport(monkeypatch, lambda r: _json(400, AZURE_CONTENT_FILTER_BODY))

    with pytest.raises(ModelProviderError) as info:
        await _provider().generate(_prompt())

    assert info.value.failure.kind is ErrorKind.BAD_REQUEST
    assert info.value.failure.message.startswith("content filter:")


@pytest.mark.asyncio
async def test_error_body_under_http_200_is_still_an_error(monkeypatch):
    body = {"error": {"code": "503", "message": "Service temporarily overloaded"}}
    _install_transport(monkeypatch, lambda r: _json(200, body))

    with pytest.raises(ModelProviderError) as info:
        await _provider().generate(_prompt())

    assert info.value.failure.kind is ErrorKind.OVERLOADED


@pytest.mark.asyncio
async def test_malformed_response_is_classified(monkeypatch):
    _install_transport(monkeypatch, lambda r: _json(200, {"id": "x", "object": "chat.completion"}))

    with pytest.raises(ModelProviderError, match="unexpected response shape") as info:
        await _provider().generate(_prompt())

    assert info.value.failure.kind is ErrorKind.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_empty_content_with_length_finish_explains_the_reasoning_budget(monkeypatch):
    _install_transport(monkeypatch, lambda r: _json(200, _completion(None, finish_reason="length")))

    with pytest.raises(ModelProviderError, match="token budget ran out") as info:
        await _provider().generate(_prompt())

    assert info.value.failure.kind is ErrorKind.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_content_filtered_completion_is_bad_request(monkeypatch):
    body = _completion(None, finish_reason="content_filter")
    _install_transport(monkeypatch, lambda r: _json(200, body))

    with pytest.raises(ModelProviderError, match="content filter") as info:
        await _provider().generate(_prompt())

    assert info.value.failure.kind is ErrorKind.BAD_REQUEST


@pytest.mark.asyncio
async def test_transport_failure_is_network(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_transport(monkeypatch, handler)

    with pytest.raises(ModelProviderError, match="request failed") as info:
        await _provider().generate(_prompt())

    assert info.value.failure.kind is ErrorKind.NETWORK


# --- tool calling -------------------------------------------------------------


def _tool_call(name: str, arguments: dict) -> dict:
    return _completion(
        None,
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


@pytest.mark.asyncio
async def test_tool_call_is_executed_and_fed_back(monkeypatch):
    handler, calls = _sequenced(
        [(200, _tool_call("calculate", {"expression": "2 + 2"})), (200, _completion("It is 4."))]
    )
    captured = _install_transport(monkeypatch, handler)

    reply = await _provider().generate(_prompt(enabled_tools=("calculate",)))

    assert reply.text == "It is 4."
    assert calls["n"] == 2
    first, second = (json.loads(r.content) for r in captured)
    assert first["tools"][0]["function"]["name"] == "calculate"
    assert first["tool_choice"] == "auto"
    # gpt-6-luna rejects function tools unless reasoning_effort is "none" (live 400).
    assert first["reasoning_effort"] == "none"
    tool_message = second["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["result"] == 4.0
    # The internal finish-reason marker is never sent back to the API.
    assert all("_finish_reason" not in m for m in second["messages"])


@pytest.mark.asyncio
async def test_tool_loop_stops_at_max_tool_calls(monkeypatch):
    handler, calls = _sequenced([(200, _tool_call("calculate", {"expression": "1 + 1"}))] * 2)
    _install_transport(monkeypatch, handler)

    reply = await _provider().generate(_prompt(enabled_tools=("calculate",), max_tool_calls=2))

    assert reply.text == TOOL_LOOP_EXHAUSTED_REPLY
    assert calls["n"] == 2


# --- charts -------------------------------------------------------------------


_CHUNK = Chunk(
    document_title="nova_horizon_fy2025",
    chunk_index=0,
    display_text="FY2021 42.3 FY2025 96.5",
    embedded_text="nova: FY2021 42.3 FY2025 96.5",
    access_labels=frozenset({"role:authenticated"}),
)
_CHART = {
    "chart": {
        "chart_type": "line",
        "title": "Revenue",
        "labels": ["FY2021", "FY2025"],
        "series": [{"name": "Revenue", "values": [42.3, 96.5]}],
        "source_chunks": [{"document_title": "nova_horizon_fy2025", "chunk_index": 0}],
    }
}


@pytest.mark.asyncio
async def test_chart_uses_strict_json_schema_and_returns_a_chart(monkeypatch):
    handler, calls = _sequenced(
        [(200, _completion("Revenue rose.")), (200, _completion(json.dumps(_CHART)))]
    )
    captured = _install_transport(monkeypatch, handler)

    reply = await _provider().generate(_prompt(retrieved_chunks=(_CHUNK,), chart_requested=True))

    assert calls["n"] == 2
    fmt = json.loads(captured[1].content)["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    # Azure strict mode (live 400): every object must set additionalProperties
    # false and require all of its properties.
    schema = fmt["json_schema"]["schema"]
    for obj in [schema, *schema.get("$defs", {}).values()]:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert reply.chart is not None and reply.chart.series[0].values == [42.3, 96.5]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second",
    [
        (200, _completion("not json")),
        (400, AZURE_CONTENT_FILTER_BODY),
        (200, _completion(json.dumps({"chart": None}))),
    ],
    ids=["unparseable", "content-filter", "null"],
)
async def test_chart_failure_keeps_the_text_answer(monkeypatch, second):
    handler, _ = _sequenced([(200, _completion("Revenue rose.")), second])
    _install_transport(monkeypatch, handler)

    reply = await _provider().generate(_prompt(retrieved_chunks=(_CHUNK,), chart_requested=True))

    assert reply.text == "Revenue rose."
    assert reply.chart is None


# --- streaming ---------------------------------------------------------------

# Frame sequence and keys reproduce the live capture from the hrinitiatives
# resource (2026-09-30): a leading choice-less prompt_filter_results frame, an
# empty-content role frame, content chunks carrying content_filter_results and
# an obfuscation field, a finish frame with an empty delta, a choice-less
# usage frame, then data: [DONE]. Blank lines separate frames; no ":" comments.
_FILTER = {"hate": {"filtered": False, "severity": "safe"}}


def _chunk_frame(delta: dict, finish=None) -> dict:
    return {
        "id": "chatcmpl-s",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-6-luna",
        "obfuscation": "Xy1",
        "service_tier": "default",
        "system_fingerprint": None,
        "usage": None,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
                "logprobs": None,
                "content_filter_results": _FILTER,
            }
        ],
    }


def _azure_sse(texts, *, finish="stop", usage=(21, 46), extra_frames=()) -> bytes:
    frames = [
        {
            "id": "",
            "object": "",
            "created": 0,
            "model": "",
            "choices": [],
            "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _FILTER}],
        },
        _chunk_frame({"content": "", "refusal": None, "role": "assistant"}),
        *[_chunk_frame({"content": t}) for t in texts],
        *extra_frames,
        _chunk_frame({}, finish=finish),
        {
            "id": "chatcmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-6-luna",
            "choices": [],
            "latency_checkpoint": {},
            "routing": {},
            "obfuscation": "z",
            "usage": {
                "prompt_tokens": usage[0],
                "completion_tokens": usage[1],
                "total_tokens": sum(usage),
            },
        },
    ]
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
    return body.encode("utf-8")


def _sse(status: int, body: bytes) -> httpx.Response:
    return httpx.Response(
        status,
        content=body,
        headers={"content-type": "text/event-stream; charset=utf-8", "apim-request-id": "r1"},
    )


async def _collect(provider, prompt):
    return [event async for event in provider.generate_stream(prompt)]


@pytest.mark.asyncio
async def test_stream_yields_deltas_in_order_skipping_azure_only_frames(monkeypatch):
    captured = _install_transport(
        monkeypatch, lambda r: _sse(200, _azure_sse(["Red", ",", " yellow", ",", " blue"]))
    )

    events = await _collect(_provider(), _prompt())

    assert [e.delta for e in events if not e.is_final] == ["Red", ",", " yellow", ",", " blue"]
    final = events[-1]
    assert final.is_final and final.text == "Red, yellow, blue" and final.chart is None
    body = json.loads(captured[0].content)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["max_completion_tokens"] == 4096 and "max_tokens" not in body
    assert "reasoning_effort" not in body  # plain streamed turn: model default
    assert captured[0].headers["api-key"] == "test-key"


@pytest.mark.asyncio
async def test_stream_usage_frame_is_recorded_in_the_turn_audit(monkeypatch):
    from app.observability import audit
    from app.observability.audit import TurnRecorder

    _install_transport(monkeypatch, lambda r: _sse(200, _azure_sse(["ok"], usage=(30, 12))))
    recorder = TurnRecorder(
        operation="chat_stream",
        tenant_id="t",
        assistant_id="a",
        principal_ref="p",
        provider="azure_ai",
        model_name="gpt-6-luna",
    )
    audit.bind(recorder)

    await _collect(_provider(), _prompt())

    (call,) = recorder.provider_calls.values()
    assert call == {
        "kind": "model",
        "provider": "azure_ai",
        "model": "gpt-6-luna",
        "count": 1,
        "prompt_tokens": 30,
        "completion_tokens": 12,
    }


@pytest.mark.asyncio
async def test_stream_http_error_before_any_delta_is_classified(monkeypatch):
    _install_transport(monkeypatch, lambda r: _sse(401, json.dumps(AZURE_401_BODY).encode()))

    with pytest.raises(ModelProviderError, match="status=401") as info:
        await _collect(_provider(), _prompt())

    assert info.value.failure.kind is ErrorKind.INVALID_KEY


@pytest.mark.asyncio
async def test_stream_error_event_mid_stream_raises_after_partial_deltas(monkeypatch):
    error_frame = {"error": {"code": "503", "message": "Service temporarily overloaded"}}
    _install_transport(
        monkeypatch, lambda r: _sse(200, _azure_sse(["partial "], extra_frames=[error_frame]))
    )
    seen: list[str] = []

    with pytest.raises(ModelProviderError) as info:
        async for event in _provider().generate_stream(_prompt()):
            seen.append(event.delta)

    assert seen == ["partial "]
    assert info.value.failure.kind is ErrorKind.OVERLOADED


@pytest.mark.asyncio
async def test_stream_content_filter_finish_is_an_error(monkeypatch):
    _install_transport(
        monkeypatch, lambda r: _sse(200, _azure_sse(["Some text"], finish="content_filter"))
    )

    with pytest.raises(ModelProviderError, match="content filter") as info:
        await _collect(_provider(), _prompt())

    assert info.value.failure.kind is ErrorKind.BAD_REQUEST


@pytest.mark.asyncio
async def test_stream_with_tools_uses_the_non_streamed_loop_with_reasoning_effort_none(
    monkeypatch,
):
    handler, calls = _sequenced(
        [(200, _tool_call("calculate", {"expression": "2 + 2"})), (200, _completion("It is 4."))]
    )
    captured = _install_transport(monkeypatch, handler)

    events = await _collect(_provider(), _prompt(enabled_tools=("calculate",)))

    assert [e.delta for e in events if not e.is_final] == ["It is 4."]
    assert events[-1].is_final and events[-1].text == "It is 4."
    assert calls["n"] == 2
    for request in captured:
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "none" and body["stream"] is False


@pytest.mark.asyncio
async def test_stream_chart_is_a_strict_schema_call_after_the_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            return _sse(200, _azure_sse(["Revenue ", "rose."]))
        return _json(200, _completion(json.dumps(_CHART)))

    captured = _install_transport(monkeypatch, handler)

    events = await _collect(_provider(), _prompt(retrieved_chunks=(_CHUNK,), chart_requested=True))

    assert [e.delta for e in events if not e.is_final] == ["Revenue ", "rose."]
    assert events[-1].chart is not None and events[-1].chart.series[0].values == [42.3, 96.5]
    schema = json.loads(captured[1].content)["response_format"]["json_schema"]["schema"]
    for obj in [schema, *schema.get("$defs", {}).values()]:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def _orchestrator_for_stream():
    import uuid
    from types import SimpleNamespace

    from app.config.assistant_config import AssistantConfig, GuardrailsConfig, RetrievalConfig
    from app.core.principal import PrincipalContext
    from app.orchestration.chat_service import ChatOrchestrator

    persisted: list[str] = []

    class _Retrieval:
        async def search(self, **kwargs):
            return [_CHUNK]

    class _Conversations:
        async def create_conversation(self, **kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        async def append_message(self, *, content, **kwargs):
            persisted.append(content)

    orchestrator = ChatOrchestrator(
        retrieval_service=_Retrieval(),
        provider_factory=lambda config: _provider(),
        conversation_store=_Conversations(),
    )
    config = AssistantConfig(
        assistant_id="hr_assistant_azure",
        display_name="x",
        description="x",
        tenant_id="t",
        model=ModelCapabilities(provider="azure_ai", model_name="gpt-6-luna"),
        retrieval=RetrievalConfig(enabled=True, collection_name="c", top_k=4),
        guardrails=GuardrailsConfig(),
        system_prompt="x",
    )
    principal = PrincipalContext(
        tenant_id="t", principal_id="p", labels=frozenset({"role:authenticated"})
    )
    return orchestrator, config, principal, persisted


@pytest.mark.asyncio
async def test_orchestrator_stream_strips_markup_split_across_azure_chunks(monkeypatch):
    from app.orchestration.chat_service import ChatStreamDelta, ChatStreamDone

    pieces = [
        "No, it is an Official Holiday ",
        '【{"cur',
        'sor": 0, "loc": 123}】',
        ".",
        "【sour",
        "ce: holidays_2026, chunk 3】",
    ]
    _install_transport(monkeypatch, lambda r: _sse(200, _azure_sse(pieces)))
    orchestrator, config, principal, persisted = _orchestrator_for_stream()

    events = [
        e
        async for e in orchestrator.handle_stream(
            principal=principal, config=config, message="is 2nd October a floating holiday?"
        )
    ]

    streamed = "".join(e.text for e in events if isinstance(e, ChatStreamDelta))
    assert "【" not in streamed and '"cursor"' not in streamed
    assert isinstance(events[-1], ChatStreamDone)
    assert persisted[-1] == "No, it is an Official Holiday."


@pytest.mark.asyncio
async def test_orchestrator_stream_mid_stream_error_sends_error_and_persists_nothing(
    monkeypatch,
):
    from app.orchestration.chat_service import ChatStreamError

    error_frame = {"error": {"code": "503", "message": "Service temporarily overloaded"}}
    _install_transport(
        monkeypatch, lambda r: _sse(200, _azure_sse(["partial "], extra_frames=[error_frame]))
    )
    orchestrator, config, principal, persisted = _orchestrator_for_stream()

    events = [
        e
        async for e in orchestrator.handle_stream(
            principal=principal, config=config, message="stipend?"
        )
    ]

    assert isinstance(events[-1], ChatStreamError)
    assert persisted == []  # nothing persisted for a failed stream


# --- provider-neutral citation-markup stripping also covers this provider ----


@pytest.mark.asyncio
async def test_citation_markup_from_azure_output_is_stripped_by_the_orchestrator(monkeypatch):
    import uuid
    from types import SimpleNamespace

    from app.config.assistant_config import AssistantConfig, GuardrailsConfig, RetrievalConfig
    from app.core.principal import PrincipalContext
    from app.orchestration.chat_service import ChatOrchestrator

    raw = (
        "No, Gandhi Jayanti is an Official Holiday, not a Floating Holiday "
        '【{"cursor": 0, "loc": 123}】.【source: holidays_2026, chunk 3】'
    )
    _install_transport(monkeypatch, lambda r: _json(200, _completion(raw)))

    class _Retrieval:
        async def search(self, **kwargs):
            return [_CHUNK]

    class _Conversations:
        async def create_conversation(self, **kwargs):
            return SimpleNamespace(id=uuid.uuid4())

        async def append_message(self, **kwargs):
            return None

    orchestrator = ChatOrchestrator(
        retrieval_service=_Retrieval(),
        provider_factory=lambda config: _provider(),
        conversation_store=_Conversations(),
    )
    result = await orchestrator.handle(
        principal=PrincipalContext(
            tenant_id="t", principal_id="p", labels=frozenset({"role:authenticated"})
        ),
        config=AssistantConfig(
            assistant_id="hr_assistant_azure",
            display_name="x",
            description="x",
            tenant_id="t",
            model=ModelCapabilities(provider="azure_ai", model_name="gpt-6-luna"),
            retrieval=RetrievalConfig(enabled=True, collection_name="c", top_k=4),
            guardrails=GuardrailsConfig(),
            system_prompt="x",
        ),
        message="is 2nd October a floating holiday?",
    )

    assert "【" not in result.text and '"cursor"' not in result.text
    assert result.text == "No, Gandhi Jayanti is an Official Holiday, not a Floating Holiday."
