"""Optional LIVE integration test against the real xAI API.

Skipped by default. Requires both:
  - XAI_API_KEY set to a real key
  - RUN_LIVE_GROK_TESTS=1

This makes real network calls and costs real money (xAI API is paid per
token) — it is never run as part of the default `pytest` invocation or CI.
Mirrors test_openrouter_provider_live.py's gating pattern exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config.loader import load_assistant_config
from app.ingestion.pipeline import Chunk
from app.orchestration.grok_provider import GrokProvider
from app.orchestration.model_provider import GroundedPrompt

_RUN_LIVE = os.getenv("RUN_LIVE_GROK_TESTS") == "1"
_API_KEY = os.getenv("XAI_API_KEY")

# The model actually configured for the Grok assistant, not a hardcoded
# literal — keeps this test honest if configs/examples/hr_assistant_grok.yaml's
# model_name ever changes.
_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "examples" / "hr_assistant_grok.yaml"
)
_MODEL_NAME = (
    load_assistant_config(_CONFIG_PATH).model.model_name if _CONFIG_PATH.exists() else None
)

_live = pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live Grok test requires XAI_API_KEY and RUN_LIVE_GROK_TESTS=1",
)

_LEAVE_CONTEXT = (
    "RETRIEVED CONTEXT:\n"
    "[source: leave_policy, chunk 0]\n"
    "Employees receive 15 days of paid annual leave per year.\n\n"
    "USER:\n"
)


@_live
@pytest.mark.asyncio
async def test_live_grok_answers_a_grounded_question():
    provider = GrokProvider(model_name=_MODEL_NAME, api_key=_API_KEY)
    prompt = GroundedPrompt(
        system_prompt="You are an HR policy assistant. Answer only from retrieved content.",
        user_message=_LEAVE_CONTEXT + "How many days of annual leave do employees get?",
        retrieved_chunks=(),
    )
    reply = await provider.generate(prompt)
    assert "15" in reply.text
    assert reply.chart is None


@_live
@pytest.mark.asyncio
async def test_live_grok_uses_the_calculate_tool():
    provider = GrokProvider(model_name=_MODEL_NAME, api_key=_API_KEY)
    prompt = GroundedPrompt(
        system_prompt=(
            "You are an HR policy assistant. Answer only from retrieved content. "
            "Use the calculate tool for any arithmetic."
        ),
        user_message=_LEAVE_CONTEXT
        + "If I take 15 days of leave in each of 7 years, how many days is that in total?",
        retrieved_chunks=(),
        enabled_tools=("calculate",),
    )
    reply = await provider.generate(prompt)
    assert "105" in reply.text


@_live
@pytest.mark.asyncio
async def test_live_grok_produces_a_structured_output_chart():
    chunk = Chunk(
        document_title="nova_horizon_fy2025",
        chunk_index=0,
        display_text="FY2025: Revenue $96.5M, Gross Profit $59.3M, Net Profit $18.2M.",
        embedded_text="nova_horizon_fy2025: FY2025 figures",
        access_labels=frozenset({"role:authenticated"}),
    )
    provider = GrokProvider(model_name=_MODEL_NAME, api_key=_API_KEY)
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
    reply = await provider.generate(prompt)
    assert reply.chart is not None
    assert sorted(v for s in reply.chart.series for v in s.values) == [18.2, 59.3, 96.5]
    assert reply.chart.source_chunks[0].document_title == "nova_horizon_fy2025"
