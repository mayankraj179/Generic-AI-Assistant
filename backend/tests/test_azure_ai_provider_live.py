"""Optional LIVE integration tests against the real Azure AI Foundry endpoint.

Skipped by default. Requires both:
  - AZURE_AI_API_KEY set to a real key
  - RUN_LIVE_AZURE_TESTS=1

These make real network calls and cost real money (billed per token) — never
part of the default `pytest` run or CI. Mirrors test_grok_provider_live.py's
gating pattern exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from app.config.loader import load_assistant_config
from app.config.settings import Settings
from app.ingestion.pipeline import Chunk
from app.orchestration.azure_ai_provider import AzureAIProvider
from app.orchestration.model_provider import GroundedPrompt

_RUN_LIVE = os.getenv("RUN_LIVE_AZURE_TESTS") == "1"
_API_KEY = os.getenv("AZURE_AI_API_KEY")
_ENDPOINT = Settings().azure_ai_endpoint

# The model actually configured for the (Azure) HR assistant, not a
# hardcoded literal.
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hr_assistant.yaml"
_MODEL_NAME = (
    load_assistant_config(_CONFIG_PATH).model.model_name if _CONFIG_PATH.exists() else None
)

_live = pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live Azure test requires AZURE_AI_API_KEY and RUN_LIVE_AZURE_TESTS=1",
)

_LEAVE_CONTEXT = (
    "RETRIEVED CONTEXT:\n"
    "[source: leave_policy, chunk 0]\n"
    "Employees receive 15 days of paid annual leave per year.\n\n"
    "USER:\n"
)


def _provider() -> AzureAIProvider:
    return AzureAIProvider(model_name=_MODEL_NAME, api_key=_API_KEY, endpoint=_ENDPOINT)


@_live
@pytest.mark.parametrize(
    "header", ["api-key", "Authorization"], ids=["api-key-header", "authorization-bearer"]
)
def test_live_both_documented_key_headers_are_accepted(header):
    # Settles empirically which of the documented key-auth headers this
    # resource accepts with a real key (a bogus key gets the same 401 either way).
    value = _API_KEY if header == "api-key" else f"Bearer {_API_KEY}"
    response = httpx.post(
        f"{_ENDPOINT}/chat/completions",
        headers={header: value, "Content-Type": "application/json"},
        json={
            "model": _MODEL_NAME,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_completion_tokens": 512,
        },
        timeout=120,
    )
    assert response.status_code == 200, response.text[:300]
    assert response.json()["choices"][0]["message"]


@_live
@pytest.mark.asyncio
async def test_live_azure_answers_a_grounded_question():
    prompt = GroundedPrompt(
        system_prompt="You are an HR policy assistant. Answer only from retrieved content.",
        user_message=_LEAVE_CONTEXT + "How many days of annual leave do employees get?",
    )
    reply = await _provider().generate(prompt)
    assert "15" in reply.text
    assert reply.chart is None


@_live
@pytest.mark.asyncio
async def test_live_azure_uses_the_calculate_tool():
    prompt = GroundedPrompt(
        system_prompt=(
            "You are an HR policy assistant. Answer only from retrieved content. "
            "Use the calculate tool for any arithmetic."
        ),
        user_message=_LEAVE_CONTEXT
        + "If I take 15 days of leave in each of 7 years, how many days is that in total?",
        enabled_tools=("calculate",),
    )
    reply = await _provider().generate(prompt)
    assert "105" in reply.text


@_live
@pytest.mark.asyncio
async def test_live_azure_produces_a_structured_output_chart():
    chunk = Chunk(
        document_title="nova_horizon_fy2025",
        chunk_index=0,
        display_text="FY2025: Revenue $96.5M, Gross Profit $59.3M, Net Profit $18.2M.",
        embedded_text="nova_horizon_fy2025: FY2025 figures",
        access_labels=frozenset({"role:authenticated"}),
    )
    prompt = GroundedPrompt(
        system_prompt="You are a financial report assistant. Answer only from retrieved content.",
        user_message=(
            "RETRIEVED CONTEXT:\n[source: nova_horizon_fy2025, chunk 0]\n"
            f"{chunk.display_text}\n\nUSER:\n"
            "Show a bar chart of revenue, gross profit and net profit for FY2025."
        ),
        retrieved_chunks=(chunk,),
        chart_requested=True,
    )
    reply = await _provider().generate(prompt)
    assert reply.chart is not None
    assert sorted(v for s in reply.chart.series for v in s.values) == [18.2, 59.3, 96.5]


@_live
@pytest.mark.asyncio
async def test_live_azure_streams_a_grounded_answer_incrementally():
    prompt = GroundedPrompt(
        system_prompt="You are an HR policy assistant. Answer only from retrieved content.",
        user_message=_LEAVE_CONTEXT
        + "How many days of annual leave do employees get? Answer in two sentences.",
    )
    deltas, final = [], None
    async for event in _provider().generate_stream(prompt):
        if event.is_final:
            final = event
        else:
            deltas.append(event.delta)

    assert len(deltas) > 1  # genuinely incremental, not one blob
    assert final is not None and "15" in final.text
    assert "".join(deltas).strip() == final.text
