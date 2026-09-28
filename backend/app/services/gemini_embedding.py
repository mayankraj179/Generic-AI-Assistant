from __future__ import annotations

from collections.abc import Sequence

from app.services.embedding import EmbeddingProviderError

# gemini-embedding-001's native output dimension — confirmed live against
# the installed google-genai SDK (2.24.0) via a real embed_content() call,
# not assumed from documentation. text-embedding-004 (an older, commonly
# referenced model name) is no longer listed among this API key's available
# models; gemini-embedding-001 and a newer gemini-embedding-2 both are.
# gemini-embedding-001 was chosen as the well-established, generally
# available option — gemini-embedding-2 is also available and could be
# swapped in later purely via DEFAULT_MODEL_NAME, since nothing else in the
# framework hardcodes a model name.
DEFAULT_MODEL_NAME = "gemini-embedding-001"
DEFAULT_DIM = 3072

_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


class GeminiEmbeddingProvider:
    """The only Gemini-embeddings-specific code in the framework — the
    embedding-path counterpart to app/orchestration/gemini_provider.py.

    Uses the model's native output dimension (3072) rather than requesting
    a truncated ``output_dimensionality``: Gemini's embedding API supports
    truncating to a smaller size (e.g. 1536), but the truncated output is
    not re-normalized by the API itself (confirmed live: a 1536-dim
    truncated vector had norm ~0.69, not 1.0) and loses some retrieval
    quality. Using the full native dimension plus a matching pgvector column
    (alembic/versions/0003_*.py) avoids that tradeoff entirely rather than
    force-fitting a smaller size.

    task_type is genuinely asymmetric for this model — verified live:
    embedding the same sentence with RETRIEVAL_DOCUMENT vs. RETRIEVAL_QUERY
    task types scores ~0.86 cosine similarity against itself, not 1.0. So
    embed_document/embed_query must stay separate calls; nothing here ever
    guesses which one a piece of text is.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str = DEFAULT_MODEL_NAME,
        dim: int = DEFAULT_DIM,
    ) -> None:
        # No eager validation here, deliberately: this provider is
        # constructed once at app/ingestion-service startup (see
        # build_default_chat_orchestrator / IngestionService), and a missing
        # key must not crash startup — only an actual embed call should fail,
        # exactly like GeminiProvider's failures are deferred to request time.
        self._api_key = api_key
        self._model_name = model_name
        self.dim = dim
        self.model_id = f"gemini:{model_name}"

    async def embed_document(self, text: str) -> list[float]:
        return (await self._embed([text], task_type=_TASK_TYPE_DOCUMENT))[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(list(texts), task_type=_TASK_TYPE_DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], task_type=_TASK_TYPE_QUERY))[0]

    async def _embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not texts:
            return []

        if not self._api_key:
            raise EmbeddingProviderError(
                "Gemini API key is not configured — set GEMINI_API_KEY"
            )

        try:
            from google.genai import Client
            from google.genai import types as genai_types
        except ImportError as exc:  # pragma: no cover - depends on optional SDK
            raise EmbeddingProviderError("google-genai SDK is not installed") from exc

        try:
            client = Client(api_key=self._api_key)
            response = await client.aio.models.embed_content(
                model=self._model_name,
                contents=texts,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )
        except Exception as exc:  # network/auth/rate-limit failures from the SDK
            raise EmbeddingProviderError("Gemini embedding request failed") from exc

        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise EmbeddingProviderError("Gemini embedding response was incomplete")

        return [list(embedding.values or []) for embedding in embeddings]
