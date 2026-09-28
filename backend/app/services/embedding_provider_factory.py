from __future__ import annotations

from collections.abc import Callable

from app.config.assistant_config import AssistantConfig
from app.config.settings import Settings
from app.services.embedding import EmbeddingProvider, EmbeddingProviderError
from app.services.gemini_embedding import GeminiEmbeddingProvider
from app.services.openrouter_embedding import OpenRouterEmbeddingProvider

_EmbeddingBuilder = Callable[[Settings, str | None], EmbeddingProvider]


def _build_gemini_embedder(settings: Settings, model_name: str | None) -> EmbeddingProvider:
    if model_name is None:
        return GeminiEmbeddingProvider(api_key=settings.gemini_api_key)
    return GeminiEmbeddingProvider(api_key=settings.gemini_api_key, model_name=model_name)


def _build_openrouter_embedder(settings: Settings, model_name: str | None) -> EmbeddingProvider:
    if model_name is None:
        return OpenRouterEmbeddingProvider(api_key=settings.openrouter_api_key)
    return OpenRouterEmbeddingProvider(api_key=settings.openrouter_api_key, model_name=model_name)


_EMBEDDING_BUILDERS: dict[str, _EmbeddingBuilder] = {
    "gemini": _build_gemini_embedder,
    "openrouter": _build_openrouter_embedder,
}


def get_embedding_provider(*, config: AssistantConfig, settings: Settings) -> EmbeddingProvider:
    """Resolve an assistant's configured embedding provider by name — the
    embedding-path counterpart to app/orchestration/provider_factory.py's
    get_model_provider(). AssistantConfig.retrieval.embedding_provider is the
    sole source of truth; defaults to "gemini" (RetrievalConfig's own
    default), so every assistant that predates this field keeps its exact
    original behavior with zero config changes. retrieval.embedding_model, if
    set, picks a non-default model within that provider.
    """
    provider_name = config.retrieval.embedding_provider if config.retrieval else "gemini"
    model_name = config.retrieval.embedding_model if config.retrieval else None
    builder = _EMBEDDING_BUILDERS.get(provider_name)
    if builder is None:
        raise EmbeddingProviderError(f"unsupported embedding provider '{provider_name}'")
    return builder(settings, model_name)
