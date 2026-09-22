"""Optional LIVE integration test against the real Gemini API.

Skipped by default. Requires both:
  - GEMINI_API_KEY set to a real key
  - RUN_LIVE_GEMINI_TESTS=1

This makes a real network call and costs real API quota — it is never run
as part of the default `pytest` invocation or CI.
"""

from __future__ import annotations

import os

import pytest

from app.orchestration.gemini_provider import GeminiProvider
from app.orchestration.model_provider import GroundedPrompt

_RUN_LIVE = os.getenv("RUN_LIVE_GEMINI_TESTS") == "1"
_API_KEY = os.getenv("GEMINI_API_KEY")


@pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live Gemini test requires GEMINI_API_KEY and RUN_LIVE_GEMINI_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_gemini_answers_a_grounded_question():
    provider = GeminiProvider(model_name="gemini-2.5-flash", api_key=_API_KEY)
    prompt = GroundedPrompt(
        system_prompt="You are an HR policy assistant. Answer only from retrieved content.",
        user_message=(
            "RETRIEVED CONTEXT:\n"
            "[source: leave_policy, chunk 0]\n"
            "Employees receive 15 days of paid annual leave per year.\n\n"
            "USER:\n"
            "How many days of annual leave do employees get?"
        ),
        retrieved_chunks=(),
    )
    reply = await provider.generate(prompt)
    assert "15" in reply.text
