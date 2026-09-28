"""Tests for OpenRouterEmbeddingProvider.

The default (non-live) tests below never call the network — mocked-transport
coverage of request shape/error wrapping, mirroring
tests/test_openrouter_provider.py's httpx.MockTransport pattern, plus
construction/config-error behavior mirroring test_gemini_embedding.py's.

The live tests at the bottom follow test_gemini_embedding.py's exact gating
pattern: skipped unless both OPENROUTER_API_KEY and
RUN_LIVE_OPENROUTER_TESTS=1 are set, since they make real network calls and
cost real API quota.
"""

from __future__ import annotations

import json
import math
import os

import httpx
import pytest

from app.services.embedding import EmbeddingProviderError
from app.services.openrouter_embedding import OpenRouterEmbeddingProvider

_RUN_LIVE = os.getenv("RUN_LIVE_OPENROUTER_TESTS") == "1"
_API_KEY = os.getenv("OPENROUTER_API_KEY")

_RealAsyncClient = httpx.AsyncClient


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    captured: list[httpx.Request] = []

    def _capturing_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    def _fake_async_client(**kwargs) -> httpx.AsyncClient:
        return _RealAsyncClient(transport=httpx.MockTransport(_capturing_handler))

    import app.services.openrouter_embedding as module

    monkeypatch.setattr(module.httpx, "AsyncClient", _fake_async_client)
    return captured


def _json_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _embeddings_payload(vectors: list[list[float]]) -> dict:
    return {
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
        "model": "google/gemini-embedding-001",
        "object": "list",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


# ---------------------------------------------------------------------------
# Construction / no-key behavior
# ---------------------------------------------------------------------------


def test_provider_can_be_constructed_without_an_api_key():
    provider = OpenRouterEmbeddingProvider(api_key=None)
    assert provider.dim == 3072


@pytest.mark.asyncio
async def test_embed_query_without_api_key_raises_embedding_provider_error():
    provider = OpenRouterEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderError):
        await provider.embed_query("What is the leave policy?")


@pytest.mark.asyncio
async def test_embed_document_without_api_key_raises_embedding_provider_error():
    provider = OpenRouterEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderError):
        await provider.embed_document("Employees receive 15 days of leave per year.")


@pytest.mark.asyncio
async def test_empty_batch_short_circuits_without_needing_a_key():
    provider = OpenRouterEmbeddingProvider(api_key=None)
    assert await provider.embed_documents([]) == []


# ---------------------------------------------------------------------------
# Request shape / input_type asymmetry (mocked transport)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_document_sends_search_document_input_type(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, _embeddings_payload([[0.1, 0.2, 0.3]]))

    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterEmbeddingProvider(api_key="test-key")

    result = await provider.embed_document("some text")
    assert result == [0.1, 0.2, 0.3]

    body = json.loads(captured[0].content)
    assert body["model"] == "google/gemini-embedding-001"
    assert body["input"] == ["some text"]
    assert body["input_type"] == "search_document"
    assert captured[0].headers["authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_embed_query_sends_search_query_input_type(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, _embeddings_payload([[0.4, 0.5, 0.6]]))

    captured = _install_transport(monkeypatch, handler)
    provider = OpenRouterEmbeddingProvider(api_key="test-key")

    result = await provider.embed_query("some query")
    assert result == [0.4, 0.5, 0.6]

    body = json.loads(captured[0].content)
    assert body["input_type"] == "search_query"


@pytest.mark.asyncio
async def test_embed_documents_preserves_batch_order(monkeypatch: pytest.MonkeyPatch):
    # Response returned out of index order — provider must re-sort by index,
    # never assume the API echoes results in request order.
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            200,
            {
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [2.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0]},
                ],
                "model": "google/gemini-embedding-001",
                "object": "list",
            },
        )

    _install_transport(monkeypatch, handler)
    provider = OpenRouterEmbeddingProvider(api_key="test-key")

    result = await provider.embed_documents(["first", "second"])
    assert result == [[1.0], [2.0]]


# ---------------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_wraps_http_error_status_with_detail(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(401, {"error": {"code": 401, "message": "Invalid credentials"}})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterEmbeddingProvider(api_key="bad-key")

    with pytest.raises(EmbeddingProviderError, match="Invalid credentials"):
        await provider.embed_query("text")


@pytest.mark.asyncio
async def test_embed_wraps_malformed_response_shape(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"unexpected": "shape"})

    _install_transport(monkeypatch, handler)
    provider = OpenRouterEmbeddingProvider(api_key="test-key")

    with pytest.raises(EmbeddingProviderError, match="unexpected"):
        await provider.embed_query("text")


@pytest.mark.asyncio
async def test_embed_wraps_incomplete_batch_response(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, _embeddings_payload([[0.1]]))  # only 1 of 2 requested

    _install_transport(monkeypatch, handler)
    provider = OpenRouterEmbeddingProvider(api_key="test-key")

    with pytest.raises(EmbeddingProviderError, match="incomplete"):
        await provider.embed_documents(["a", "b"])


@pytest.mark.asyncio
async def test_embed_wraps_transport_level_failure(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_transport(monkeypatch, handler)
    provider = OpenRouterEmbeddingProvider(api_key="test-key")

    with pytest.raises(EmbeddingProviderError, match="request failed"):
        await provider.embed_query("text")


# ---------------------------------------------------------------------------
# Live integration tests — makes real network calls, real API quota
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_RUN_LIVE and _API_KEY),
    reason="live embedding test requires OPENROUTER_API_KEY and RUN_LIVE_OPENROUTER_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_embedding_similarity_reflects_semantic_meaning():
    provider = OpenRouterEmbeddingProvider(api_key=_API_KEY)

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
    reason="live embedding test requires OPENROUTER_API_KEY and RUN_LIVE_OPENROUTER_TESTS=1",
)
@pytest.mark.asyncio
async def test_live_document_and_query_input_types_are_genuinely_asymmetric():
    provider = OpenRouterEmbeddingProvider(api_key=_API_KEY)
    text = "Employees receive 15 days of paid annual leave per year."

    document_embedding = await provider.embed_document(text)
    query_embedding = await provider.embed_query(text)

    similarity = _cosine_similarity(document_embedding, query_embedding)
    assert similarity < 0.99
