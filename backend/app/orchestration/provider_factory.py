from __future__ import annotations

from collections.abc import Callable

from app.config.assistant_config import ModelCapabilities
from app.config.settings import Settings
from app.orchestration.gemini_provider import GeminiProvider
from app.orchestration.grok_provider import GrokProvider
from app.orchestration.model_provider import ModelProvider, ModelProviderError
from app.orchestration.openrouter_provider import OpenRouterProvider

_ProviderBuilder = Callable[[ModelCapabilities, Settings], ModelProvider]


def _build_gemini_provider(model: ModelCapabilities, settings: Settings) -> ModelProvider:
    return GeminiProvider(model_name=model.model_name, api_key=settings.gemini_api_key)


def _build_openrouter_provider(model: ModelCapabilities, settings: Settings) -> ModelProvider:
    return OpenRouterProvider(model_name=model.model_name, api_key=settings.openrouter_api_key)


def _build_xai_provider(model: ModelCapabilities, settings: Settings) -> ModelProvider:
    return GrokProvider(model_name=model.model_name, api_key=settings.xai_api_key)


_PROVIDER_BUILDERS: dict[str, _ProviderBuilder] = {
    "google_adk": _build_gemini_provider,
    "gemini": _build_gemini_provider,
    "openrouter": _build_openrouter_provider,
    "xai": _build_xai_provider,
}


def get_model_provider(*, model: ModelCapabilities, settings: Settings) -> ModelProvider:
    """Resolve an assistant's configured model provider by name.

    The assistant YAML's ``model.provider`` is the sole source of truth for
    which vendor backs a given assistant — never hardcode a vendor in a route
    or in business logic.
    """
    builder = _PROVIDER_BUILDERS.get(model.provider)
    if builder is None:
        raise ModelProviderError(f"unsupported model provider '{model.provider}'")
    return builder(model, settings)
