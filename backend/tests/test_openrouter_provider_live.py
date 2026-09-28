"""Optional LIVE integration test against the real OpenRouter API.

Skipped by default. Requires both:
  - OPENROUTER_API_KEY set to a real key
  - RUN_LIVE_OPENROUTER_TESTS=1

This makes a real network call and costs real API quota — it is never run
as part of the default `pytest` invocation or CI. Mirrors
test_gemini_provider_live.py's gating pattern exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config.loader import load_assistant_config
from app.orchestration.model_provider import GroundedPrompt
from app.orchestration.openrouter_provider import OpenRouterProvider

_RUN_LIVE = os.getenv("RUN_LIVE_OPENROUTER_TESTS") == "1"
_API_KEY = os.getenv("OPENROUTER_API_KEY")

# The actual model configured for the OpenRouter example assistant, not a
# hardcoded literal — keeps this test honest if configs/
# examples/finance_assistant_openrouter.yaml's model_name ever changes.
_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "examples"
    / "finance_assistant_openrouter.yaml"
)
_MODEL_NAME = (
    load_assistant_config(_CONFIG_PATH).model.model_name if _CONFIG_PATH.exists() else None
)


@pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live OpenRouter test requires OPENROUTER_API_KEY and RUN_LIVE_OPENROUTER_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_openrouter_answers_a_grounded_question():
    provider = OpenRouterProvider(model_name=_MODEL_NAME, api_key=_API_KEY)
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


@pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live OpenRouter test requires OPENROUTER_API_KEY and RUN_LIVE_OPENROUTER_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_openrouter_streams_a_grounded_answer():
    provider = OpenRouterProvider(model_name=_MODEL_NAME, api_key=_API_KEY)
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
    deltas = []
    final_text = None
    async for event in provider.generate_stream(prompt):
        if event.is_final:
            final_text = event.text
        else:
            deltas.append(event.delta)

    assert "".join(deltas)
    assert final_text is not None
    assert "15" in final_text
