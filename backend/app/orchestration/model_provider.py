from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from app.ingestion.pipeline import Chunk


@dataclass(frozen=True)
class ChartSeries:
    """One plotted series. A pie chart's single series' ``values`` are the
    slice sizes, in the same order as ``ChartSpec.labels``."""

    name: str
    values: list[float]


@dataclass(frozen=True)
class ChartSourceChunk:
    """Same identifying shape ChunkCitation/Citation already use elsewhere —
    a chart's numbers must be traceable back to a specific retrieved chunk
    exactly like a text citation is, never a vague "from the documents"."""

    document_title: str
    chunk_index: int


@dataclass(frozen=True)
class ChartSpec:
    """Vendor-neutral, provider-produced chart grounded in retrieved
    context. Deliberately small/fixed-shape — this is a data contract for
    a handful of chart types a simple frontend renderer can draw directly,
    not a general charting-library configuration object.

    Every number here must trace back to ``source_chunks`` — providers must
    build this only from retrieved content (see GeminiProvider), and
    ChatOrchestrator re-validates ``source_chunks`` server-side against the
    turn's actual authorized retrieval set before ever exposing a chart to
    a caller (see ChatOrchestrator._validate_chart) — a provider's own
    citation claim is never trusted outright, the same discipline already
    applied to text citations.
    """

    chart_type: Literal["bar", "line", "pie"]
    title: str
    labels: list[str]
    series: list[ChartSeries]
    source_chunks: list[ChartSourceChunk]

    def __post_init__(self) -> None:
        if self.chart_type == "pie" and len(self.series) != 1:
            raise ValueError("a pie chart must have exactly one series")


@dataclass(frozen=True)
class ConversationTurn:
    """One prior turn of a multi-turn conversation, in provider-neutral form.

    ``role`` is always ``"user"`` or ``"assistant"`` — never a vendor-specific
    role name (e.g. ADK/Gemini's ``"model"``). Translating that is the
    provider's job, not the orchestration layer's.
    """

    role: str
    content: str


@dataclass(frozen=True)
class GroundedPrompt:
    """Framework-assembled model input for one conversational turn.

    ``user_message`` is already fully composed by the orchestration layer
    (question plus any authorized retrieved context, clearly separated).
    ``retrieved_chunks`` is carried alongside only so a provider can report
    whether its answer was grounded — providers must not re-render it.
    ``prior_turns`` is this conversation's history so far (oldest first),
    empty for a brand-new conversation.
    """

    system_prompt: str
    user_message: str
    retrieved_chunks: tuple[Chunk, ...] = ()
    prior_turns: tuple[ConversationTurn, ...] = ()
    chart_requested: bool = False
    """Set by ChatOrchestrator._prepare_turn from a cheap, deterministic
    keyword check on the user's own message (never a second model call just
    to decide this) — True only when the user appears to be asking for a
    chart/graph AND retrieval actually produced authorized chunks. A
    provider must skip chart generation entirely when this is False, so
    ordinary turns pay zero extra latency/cost for this feature."""
    enabled_tools: tuple[str, ...] = ()
    """Tool names (keys into app.tools.builtin.TOOL_REGISTRY) this turn's
    assistant config has opted into via AssistantConfig.enabled_tools —
    empty by default, so an ordinary turn for an assistant with no tools
    configured pays zero extra cost, the same discipline chart_requested
    already established. A provider resolves these names against the
    shared registry itself; this stays a plain tuple of strings rather than
    ToolDefinition objects so GroundedPrompt doesn't have to import the
    tools package.

    The tool-call loop itself (propose -> execute -> feed result back ->
    repeat, bounded by max_tool_calls) lives inside each provider, not in
    ChatOrchestrator: Gemini/ADK and OpenRouter each have a genuinely
    different native mechanism for this (google-adk's Runner executes
    FunctionTool calls internally; OpenRouter's raw HTTP API requires a
    hand-written request/response loop — see GeminiProvider/
    OpenRouterProvider), and ChatOrchestrator only ever sees the single
    final ModelReply either way, exactly like it already does for the
    text/grounded/chart fields. This keeps the Protocol's one-call-per-turn
    shape intact instead of inventing a new cross-provider intermediate
    "tool call" event type."""
    max_tool_calls: int = 4
    """Hard cap on tool-call round-trips a provider may make while
    producing this turn's reply (from AssistantConfig.max_tool_calls) —
    never unbounded, even for read-only tools. A provider that hits this
    cap without reaching a final answer must return a clear, honest reply
    saying so, never a truncated or fabricated one."""


@dataclass(frozen=True)
class ModelReply:
    text: str
    grounded: bool = False
    chart: ChartSpec | None = None


@dataclass(frozen=True)
class ModelStreamEvent:
    """One increment of a streamed model turn.

    Every event before the last one carries a non-empty ``delta`` and
    ``is_final=False``. Exactly one final event closes the stream
    (``is_final=True``), carrying the same information ``ModelReply`` carries
    today — the complete ``text``, ``grounded`` flag, and (per
    ``chart_requested``) ``chart``. Consumers must treat
    ``text``/``grounded``/``chart`` as meaningless on non-final events, and
    must never treat a final event's ``text`` as an additional delta to
    append.
    """

    delta: str = ""
    is_final: bool = False
    text: str = ""
    grounded: bool = False
    chart: ChartSpec | None = None


class ModelProviderError(Exception):
    """Raised when a model provider is misconfigured or a model call fails.

    Callers must treat the message as internal detail — never forward it
    verbatim into an HTTP response.
    """


class ModelProvider(Protocol):
    """Provider-neutral boundary between the framework core and a model vendor SDK.

    The framework core (routes, orchestration, retrieval, authorization) must
    depend only on this Protocol. Vendor-specific code (e.g. Google ADK/Gemini)
    lives entirely behind an implementation of it.
    """

    async def generate(self, prompt: GroundedPrompt) -> ModelReply: ...

    def generate_stream(self, prompt: GroundedPrompt) -> AsyncIterator[ModelStreamEvent]: ...
