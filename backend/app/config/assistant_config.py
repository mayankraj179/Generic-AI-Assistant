from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Deterministic, keyword/pattern-based chart-intent gate (see
# ChatOrchestrator._message_requests_chart) — deliberately not a model call
# or classifier. Regex, case-insensitive. Per-assistant tunable via
# AssistantConfig.chart_trigger_patterns; this is just the shipped default.
DEFAULT_CHART_TRIGGER_PATTERNS: list[str] = [
    r"\bchart\b",
    r"\bgraph\b",
    r"\bplot\b",
    r"pie\s+chart",
    r"bar\s+chart",
    r"line\s+chart",
    r"visuali[sz]e",
    r"\btrend\b",
    r"\bbreakdown\b",
    r"compare\b.*\bover\s+time\b",
]


class ModelCapabilities(BaseModel):
    """What the configured model supports, so the core can enable/disable
    features per assistant without hardcoding vendor-specific checks."""

    provider: str  # e.g. "azure_openai", "google_adk", "anthropic"
    model_name: str
    supports_tools: bool = True
    supports_streaming: bool = True
    thinking_level: str | None = None  # None, "low", "medium", "high" — provider-defined
    configurable_thinking: bool = False
    max_context_tokens: int = 128_000


class RetrievalConfig(BaseModel):
    """Controls the RAG lane: what gets embedded and how it's retrieved.

    ``min_similarity`` is a relevance/business-rule cutoff, not an
    access-control predicate — it lives here rather than under
    ``guardrails`` because it's fundamentally a retrieval-quality knob
    (see RetrievalService.search), the same way ``top_k`` is. Its default
    (0.6) was derived empirically against the live Gemini embedding
    provider, not guessed: genuinely unrelated queries against a real
    ingested chunk scored 0.49-0.53 cosine similarity, while genuinely
    relevant paraphrases of the same topic scored 0.70-0.75 — 0.6 sits in
    the gap with margin on both sides.
    """

    enabled: bool = True
    collection_name: str
    exact_search_threshold: int = 500  # below this accessible-set size, use exact search
    top_k: int = 8
    min_similarity: float = 0.6


class NamedQuery(BaseModel):
    """A live/structured data lookup, as opposed to embedded/quotable content.
    Used when the answer is a derived number, not a quote."""

    name: str
    description: str
    sql_template: str
    required_labels: frozenset[str] = Field(default_factory=frozenset)


class GuardrailsConfig(BaseModel):
    """Input/output guardrail toggles and knobs for one assistant.

    PII checks default to log-only (``*_pii_block=False``): over-blocking a
    legitimate question that happens to contain a phone number for context
    is worse than a logged flag for this first pass, so blocking is an
    explicit opt-in per assistant, never a silent default either way.

    ``unsafe_output_patterns`` is an assistant-specific deny-list of regex
    patterns (empty by default, i.e. off) — "unsafe content" is domain
    specific, so there's no sensible universal default to ship built in.
    """

    prompt_injection_screening: bool = True
    input_pii_block: bool = False
    output_pii_block: bool = False
    max_input_chars: int = 4000
    unsafe_output_patterns: list[str] = Field(default_factory=list)


class AssistantConfig(BaseModel):
    """The single source of truth for onboarding a new assistant use case.

    A new assistant must be creatable from this config alone, with zero
    changes to core code — that's the Day-7/Day-8 "second assistant" proof
    this schema exists to support.
    """

    assistant_id: str
    display_name: str
    description: str
    tenant_id: str

    model: ModelCapabilities
    retrieval: RetrievalConfig | None = None
    named_queries: list[NamedQuery] = Field(default_factory=list)
    enabled_tools: list[str] = Field(default_factory=list)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)

    system_prompt: str
    min_clearance: int = 0
    chart_trigger_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CHART_TRIGGER_PATTERNS)
    )

    @field_validator("assistant_id")
    @classmethod
    def _validate_id_format(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"assistant_id '{v}' must be alphanumeric with optional - or _"
            )
        return v

    @field_validator("enabled_tools")
    @classmethod
    def _no_duplicate_tools(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("enabled_tools contains duplicates")
        return v
