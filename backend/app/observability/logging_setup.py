"""Structured logging: one JSON object per line, stamped with the current
request's identifiers and trace/span ids.

LOG_LEVEL (default INFO) and LOG_FORMAT ("json" default, or "text" for a
human-readable terminal line) are read from the environment."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from app.observability.context import current_context

# Attributes every LogRecord has; anything else was passed via `extra=`.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}
_CONTEXT_FIELDS = (
    "request_id",
    "assistant_id",
    "principal",
    "conversation_id",
    "trace_id",
    "span_id",
)

# Library loggers that log a full traceback for errors this application
# already reports once, classified, at the point it handles them:
#   - google-adk logs "Node execution failed with exception" for every model
#     error before turning it into an error event;
#   - opentelemetry.context logs "Failed to detach context" when ADK's async
#     generators are closed from a different context.
# Their records are demoted to DEBUG (still visible with LOG_LEVEL=DEBUG).
_DEMOTED_LOGGERS = (
    "google_adk.google.adk.workflow._node_runner",
    "opentelemetry.context",
)
# Per-HTTP-request INFO lines from httpx carry no information the provider
# logs and spans don't already have.
_QUIET_LOGGERS = {"httpx": logging.WARNING, "httpcore": logging.WARNING}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in current_context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        try:
            from opentelemetry import trace

            span_context = trace.get_current_span().get_span_context()
            if span_context.is_valid:
                record.trace_id = f"{span_context.trace_id:032x}"
                record.span_id = f"{span_context.span_id:016x}"
        except Exception:  # pragma: no cover - tracing must never break logging
            pass
        return True


class DemoteToDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.levelno, record.levelname = logging.DEBUG, "DEBUG"
        return logging.getLogger().isEnabledFor(logging.DEBUG)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and key not in payload and key not in _CONTEXT_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        rid = getattr(record, "request_id", None)
        line = f"{ts} {record.levelname:<7} {record.name}"
        if rid:
            line += f" [{rid[:8]}]"
        line += f" {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Idempotent: replaces only the handler this function installed."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt_name = (fmt or os.getenv("LOG_FORMAT", "json")).lower()

    root = logging.getLogger()
    root.setLevel(level_name)
    for handler in list(root.handlers):
        if getattr(handler, "_assistant_framework", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler._assistant_framework = True  # type: ignore[attr-defined]
    handler.setFormatter(TextFormatter() if fmt_name == "text" else JsonFormatter())
    handler.addFilter(ContextFilter())
    root.addHandler(handler)

    for name in _DEMOTED_LOGGERS:
        target = logging.getLogger(name)
        if not any(isinstance(f, DemoteToDebugFilter) for f in target.filters):
            target.addFilter(DemoteToDebugFilter())
    for name, quiet_level in _QUIET_LOGGERS.items():
        if level_name != "DEBUG":
            logging.getLogger(name).setLevel(quiet_level)
