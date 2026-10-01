from __future__ import annotations

from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven auth and framework configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    auth_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("AUTH_ENABLED", "auth_enabled"),
    )
    auth_issuer: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AUTH_ISSUER", "auth_issuer"),
    )
    auth_audience: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AUTH_AUDIENCE", "auth_audience"),
    )
    auth_jwks_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AUTH_JWKS_URL", "auth_jwks_url"),
    )
    auth_required: bool = Field(
        default=True,
        validation_alias=AliasChoices("AUTH_REQUIRED", "auth_required"),
    )
    development_permission_set: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "assistant:use",
            "conversation:read",
            "conversation:write",
            "retrieval:use",
            "tool:invoke",
        ],
        validation_alias=AliasChoices(
            "DEVELOPMENT_PERMISSION_SET",
            "development_permission_set",
        ),
    )
    development_labels: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["role:authenticated", "scope:development"],
        validation_alias=AliasChoices("DEVELOPMENT_LABELS", "development_labels"),
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "GOOGLE_ADK_API_KEY",
            "gemini_api_key",
        ),
    )
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENROUTER_API_KEY",
            "openrouter_api_key",
        ),
    )
    xai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "XAI_API_KEY",
            "xai_api_key",
        ),
    )
    # TEMP: Azure AI Foundry, added for testing (2026-09-30).
    azure_ai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_AI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "azure_ai_api_key",
        ),
    )
    azure_ai_endpoint: str = Field(
        default="https://hrinitiatives.services.ai.azure.com/openai/v1",
        validation_alias=AliasChoices(
            "AZURE_AI_ENDPOINT",
            "azure_ai_endpoint",
        ),
    )

    @field_validator("development_permission_set", mode="before")
    @classmethod
    def _parse_permission_set(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value)]

    @field_validator("development_labels", mode="before")
    @classmethod
    def _parse_development_labels(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value)]


settings = Settings()
