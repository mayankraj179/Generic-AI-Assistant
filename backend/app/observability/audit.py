"""Per-turn recorder and the durable turn_audit row it becomes.

A TurnRecorder is bound to the current context for the duration of one chat
turn. Providers, tools and guardrails report into it through the module-level
record_* helpers, which are no-ops outside a turn (e.g. ingestion scripts)."""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.observability.context import request_id_var
from app.observability.provider_errors import ProviderFailure
from app.observability.tracing import current_trace_id

logger = logging.getLogger(__name__)


@dataclass
class TurnRecorder:
    operation: str
    tenant_id: str
    assistant_id: str
    principal_ref: str
    provider: str
    model_name: str
    embedding_model: str | None = None
    conversation_id: uuid.UUID | None = None
    outcome: str = "error"
    grounded: bool = False
    citations_count: int = 0
    retrieval: dict[str, Any] = field(default_factory=dict)
    chart_returned: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Successful tool outputs, for ChatOrchestrator._require_tool_result.
    # Never part of the audit row: a tool result can carry data the audit
    # table has no business keeping.
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    guardrail_actions: list[dict[str, Any]] = field(default_factory=list)
    provider_calls: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    failure: ProviderFailure | None = None
    error_class: str | None = None
    started: float = field(default_factory=time.perf_counter)

    def add_provider_call(
        self,
        kind: str,
        provider: str,
        model: str,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        entry = self.provider_calls.setdefault(
            (kind, provider, model),
            {"kind": kind, "provider": provider, "model": model, "count": 0},
        )
        entry["count"] += 1
        for key, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
        ):
            if value is not None:
                entry[key] = entry.get(key, 0) + int(value)

    def as_row(self) -> dict[str, Any]:
        error = None
        if self.failure is not None:
            error = self.failure.to_dict()
        elif self.error_class is not None:
            error = {"exception": self.error_class}
        return {
            "request_id": request_id_var.get(),
            "trace_id": current_trace_id(),
            "operation": self.operation,
            "tenant_id": self.tenant_id,
            "assistant_id": self.assistant_id,
            "principal_ref": self.principal_ref,
            "conversation_id": self.conversation_id,
            "provider": self.provider,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "outcome": self.outcome,
            "grounded": self.grounded,
            "citations_count": self.citations_count,
            "retrieval": self.retrieval,
            "chart_returned": self.chart_returned,
            "tool_calls": self.tool_calls,
            "guardrail_actions": self.guardrail_actions,
            "provider_calls": list(self.provider_calls.values()),
            "error_kind": str(self.failure.kind) if self.failure else self.error_class,
            "error": error,
            "latency_ms": int((time.perf_counter() - self.started) * 1000),
        }


_current: ContextVar[TurnRecorder | None] = ContextVar("turn_recorder", default=None)


def bind(recorder: TurnRecorder):
    return _current.set(recorder)


def unbind(token) -> None:
    _current.reset(token)


def current() -> TurnRecorder | None:
    return _current.get()


def record_provider_call(
    kind: str,
    provider: str,
    model: str,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    recorder = _current.get()
    if recorder is not None:
        recorder.add_provider_call(
            kind, provider, model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )


def record_tokens(
    kind: str,
    provider: str,
    model: str,
    *,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    """Adds token usage to an already-counted call (providers that report
    usage separately from the request, e.g. ADK events)."""
    recorder = _current.get()
    if recorder is None:
        return
    entry = recorder.provider_calls.setdefault(
        (kind, provider, model), {"kind": kind, "provider": provider, "model": model, "count": 0}
    )
    for key, value in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens)):
        if value is not None:
            entry[key] = entry.get(key, 0) + int(value)


def record_retrieval(**stats: Any) -> None:
    recorder = _current.get()
    if recorder is not None:
        recorder.retrieval.update({k: v for k, v in stats.items() if v is not None})
        recorder.embedding_model = stats.get("embedding_model") or recorder.embedding_model


def record_tool_call(
    name: str, ok: bool, error: str | None = None, result: dict[str, Any] | None = None
) -> None:
    recorder = _current.get()
    if recorder is not None:
        entry: dict[str, Any] = {"name": name, "ok": ok}
        if error:
            entry["error"] = error[:200]
        recorder.tool_calls.append(entry)
        if ok and result is not None:
            recorder.tool_results.append(result)


def record_guardrail(action: str, **detail: Any) -> None:
    recorder = _current.get()
    if recorder is not None:
        recorder.guardrail_actions.append({"action": action, **detail})


class AuditStore:
    """Writes one turn_audit row per turn. A failed write is logged and
    swallowed: auditing must never fail the user's turn."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._session_factory = session_factory

    async def write(self, recorder: TurnRecorder) -> None:
        from app.db.models import TurnAuditRecord

        row = recorder.as_row()
        try:
            if self._session_factory is None:
                from app.db.database import create_session_factory

                self._session_factory = create_session_factory()
            async with self._session_factory() as session:
                session.add(TurnAuditRecord(**row))
                await session.commit()
        except Exception as exc:
            logger.warning(
                "turn audit write failed: %s: %s",
                type(exc).__name__,
                exc,
                extra={"event": "audit_write_failed"},
            )
        logger.info(
            "turn %s: %s/%s outcome=%s grounded=%s citations=%d calls=%s latency_ms=%d",
            row["operation"],
            row["provider"],
            row["model_name"],
            row["outcome"],
            row["grounded"],
            row["citations_count"],
            {f"{c['provider']}:{c['kind']}": c["count"] for c in row["provider_calls"]},
            row["latency_ms"],
            extra={"event": "turn_completed", "error_kind": row["error_kind"]},
        )
