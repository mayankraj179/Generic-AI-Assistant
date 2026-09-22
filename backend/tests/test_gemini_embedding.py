"""Tests for the real Gemini embedding provider.

The default (non-live) tests below never call the network — they cover
construction/config-error behavior only, the same way
test_chat_orchestration.py's test_missing_gemini_api_key_raises_model_provider_error
proves GeminiProvider fails cleanly without a key, without ever calling
Gemini for real.

The live similarity test at the bottom follows test_gemini_provider_live.py's
exact gating pattern: skipped unless both GEMINI_API_KEY and
RUN_LIVE_GEMINI_TESTS=1 are set, since it makes real network calls and costs
real API quota.
"""

from __future__ import annotations

import math
import os

import pytest

from app.services.embedding import EmbeddingProviderError
from app.services.gemini_embedding import GeminiEmbeddingProvider

_RUN_LIVE = os.getenv("RUN_LIVE_GEMINI_TESTS") == "1"
_API_KEY = os.getenv("GEMINI_API_KEY")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Default test-suite coverage — no network, no API key required
# ---------------------------------------------------------------------------


def test_provider_can_be_constructed_without_an_api_key():
    # Construction must never fail just because no key is configured yet —
    # this provider is built once at app/ingestion-service startup, and a
    # missing key must not crash startup, only an actual embed call.
    provider = GeminiEmbeddingProvider(api_key=None)
    assert provider.dim == 3072


@pytest.mark.asyncio
async def test_embed_query_without_api_key_raises_embedding_provider_error():
    provider = GeminiEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderError):
        await provider.embed_query("What is the leave policy?")


@pytest.mark.asyncio
async def test_embed_document_without_api_key_raises_embedding_provider_error():
    provider = GeminiEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderError):
        await provider.embed_document("Employees receive 15 days of leave per year.")


@pytest.mark.asyncio
async def test_embed_documents_without_api_key_raises_embedding_provider_error():
    provider = GeminiEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderError):
        await provider.embed_documents(["chunk one", "chunk two"])


@pytest.mark.asyncio
async def test_empty_batch_short_circuits_without_needing_a_key():
    # An empty batch has nothing to embed, so it must not fail just because
    # no key is configured — there's no call to make.
    provider = GeminiEmbeddingProvider(api_key=None)
    assert await provider.embed_documents([]) == []


# ---------------------------------------------------------------------------
# Live integration test — makes real network calls, real API quota
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live embedding test requires GEMINI_API_KEY and RUN_LIVE_GEMINI_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_embedding_similarity_reflects_semantic_meaning():
    """The real proof the integration works: semantically similar text
    embeds closer together than dissimilar text — not just "returns some
    vector of the right shape."""
    provider = GeminiEmbeddingProvider(api_key=_API_KEY)

    similar_a = await provider.embed_document("The cat sat quietly on the warm windowsill.")
    similar_b = await provider.embed_document(
        "A kitten rested peacefully on the sunny window ledge."
    )
    dissimilar = await provider.embed_document(
        "Quarterly tax filings are due by the fifteenth of April."
    )

    assert len(similar_a) == 3072
    assert len(similar_b) == 3072
    assert len(dissimilar) == 3072

    similar_pair_score = _cosine_similarity(similar_a, similar_b)
    dissimilar_pair_score = _cosine_similarity(similar_a, dissimilar)

    assert similar_pair_score > dissimilar_pair_score


@pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live embedding test requires GEMINI_API_KEY and RUN_LIVE_GEMINI_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_batch_embedding_preserves_order_and_count():
    provider = GeminiEmbeddingProvider(api_key=_API_KEY)
    texts = ["first chunk of text", "second chunk of text", "third chunk of text"]

    batch_result = await provider.embed_documents(texts)
    assert len(batch_result) == len(texts)

    individual_first = await provider.embed_document(texts[0])
    # The batched embedding for index 0 should match embedding it alone —
    # proving the batch call actually maps each input to its own output in
    # order, not e.g. averaging or misaligning them.
    assert _cosine_similarity(batch_result[0], individual_first) > 0.99999


@pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live embedding test requires GEMINI_API_KEY and RUN_LIVE_GEMINI_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_document_and_query_task_types_are_genuinely_asymmetric():
    provider = GeminiEmbeddingProvider(api_key=_API_KEY)
    text = "Employees receive 15 days of paid annual leave per year."

    document_embedding = await provider.embed_document(text)
    query_embedding = await provider.embed_query(text)

    # Same text, different task type -> different embedding. If this ever
    # returns ~1.0, the model/SDK stopped distinguishing task types and
    # embed_document/embed_query could be collapsed back into one method.
    similarity = _cosine_similarity(document_embedding, query_embedding)
    assert similarity < 0.99
