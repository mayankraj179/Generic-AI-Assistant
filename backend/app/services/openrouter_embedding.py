from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from app.services.embedding import EmbeddingProviderError

# "google/gemini-embedding-001" via OpenRouter's /embeddings endpoint —
# verified LIVE (not assumed): a real call returns 3072-dim vectors,
# identical to GeminiEmbeddingProvider's DEFAULT_DIM (gemini_embedding.py),
# because OpenRouter is routing this specific model straight to Google's own
# embedding backend rather than a different vendor. That match is exactly
# what makes this provider a drop-in for the existing fixed-width
# `chunks.embedding vector(3072)` column (alembic/versions/
# 0003_widen_embedding_to_gemini_dim.py) with zero schema change.
#
# Also verified live: passing input_type="search_document" vs
# "search_query" for the identical input text yields ~0.867 cosine
# similarity between the two — matching GeminiEmbeddingProvider's own
# documented ~0.86 asymmetry for RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY task
# types on the direct google-genai path. OpenRouter's input_type parameter
# is genuinely honored, not a no-op, so the same asymmetric-embedding
# discipline applies here too.
DEFAULT_MODEL_NAME = "google/gemini-embedding-001"
DEFAULT_DIM = 3072

# Native output dimension per supported model, each confirmed live. Selected
# per assistant via RetrievalConfig.embedding_model.
#   nvidia/nemotron-3-embed-1b:free — verified live 2026-09-27: free
#   (pricing.prompt == "0"), 2048-dim, unit-normalized, 32k-token context,
#   input_type genuinely honored (the same text as search_document vs
#   search_query gives ~0.895 cosine, not 1.0).
_KNOWN_DIMS = {
    DEFAULT_MODEL_NAME: DEFAULT_DIM,
    "nvidia/nemotron-3-embed-1b:free": 2048,
}

_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
_REQUEST_TIMEOUT_SECONDS = 60.0
_APP_TITLE = "Generic AI Assistant Framework"

_INPUT_TYPE_DOCUMENT = "search_document"
_INPUT_TYPE_QUERY = "search_query"


def _parse_json_body(raw: bytes) -> Any:
    import json

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _raise_for_error(status_code: int, body: Any) -> None:
    """Same documented OpenRouter error shape as openrouter_provider.py's
    _raise_for_error — kept as a separate copy rather than a shared import:
    this module and openrouter_provider.py are each meant to be the single
    place their own vendor integration lives (mirrors how gemini_embedding.py
    and gemini_provider.py don't share code either), and the function is
    three lines.
    """
    if status_code < 400:
        return
    detail: str | None = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                detail = message
    suffix = f": {detail}" if detail else ""
    raise EmbeddingProviderError(f"OpenRouter returned an error (status={status_code}){suffix}")


class OpenRouterEmbeddingProvider:
    """The embedding-path counterpart to OpenRouterProvider
    (app/orchestration/openrouter_provider.py) — implements the same
    EmbeddingProvider Protocol GeminiEmbeddingProvider implements
    (app/services/embedding.py), so IngestionService/RetrievalService need
    no vendor-specific branching to use either.

    Lets an OpenRouter-only deployment (OPENROUTER_API_KEY set,
    GEMINI_API_KEY unset) run retrieval end-to-end without ever needing a
    direct Google credential — see configs/examples/finance_assistant_openrouter.yaml,
    which sets retrieval.embedding_provider: openrouter for exactly this
    reason.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = DEFAULT_MODEL_NAME,
        dim: int | None = None,
    ) -> None:
        # No eager key validation, deliberately — mirrors GeminiEmbeddingProvider:
        # constructed once at service-startup, a missing key must not crash
        # startup, only an actual embed call should fail.
        if dim is None:
            dim = _KNOWN_DIMS.get(model_name)
            if dim is None:
                raise EmbeddingProviderError(
                    f"unknown output dimension for OpenRouter embedding model "
                    f"'{model_name}' — verify it live and add it to _KNOWN_DIMS"
                )
        self._api_key = api_key
        self._model_name = model_name
        self.dim = dim
        self.model_id = f"openrouter:{model_name}"

    async def embed_document(self, text: str) -> list[float]:
        return (await self._embed([text], input_type=_INPUT_TYPE_DOCUMENT))[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(list(texts), input_type=_INPUT_TYPE_DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], input_type=_INPUT_TYPE_QUERY))[0]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": _APP_TITLE,
        }

    async def _embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        if not texts:
            return []

        if not self._api_key:
            raise EmbeddingProviderError(
                "OpenRouter API key is not configured — set OPENROUTER_API_KEY"
            )

        payload = {
            "model": self._model_name,
            "input": texts,
            "input_type": input_type,
        }
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(_EMBEDDINGS_URL, headers=self._headers(), json=payload)
        except EmbeddingProviderError:
            raise
        except Exception as exc:  # network/timeout/TLS failures from httpx
            raise EmbeddingProviderError("OpenRouter embedding request failed") from exc

        body = _parse_json_body(response.content)
        _raise_for_error(response.status_code, body)

        try:
            items = sorted(body["data"], key=lambda item: item["index"])
            embeddings = [list(item["embedding"]) for item in items]
        except (KeyError, TypeError) as exc:
            raise EmbeddingProviderError(
                "OpenRouter returned an unexpected embeddings response shape"
            ) from exc

        if len(embeddings) != len(texts):
            raise EmbeddingProviderError("OpenRouter embedding response was incomplete")

        return embeddings
