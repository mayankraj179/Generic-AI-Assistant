"""Classifies provider failures into a small taxonomy from the status codes and
error bodies the providers actually return, and logs each one as a single
structured line (full traceback at DEBUG only).

Every rule below corresponds to an error shape captured live from these
providers during this project (see tests/test_provider_error_classification.py
for the verbatim bodies)."""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ErrorKind(StrEnum):
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"  # a daily/period allowance is used up
    RATE_LIMITED = "RATE_LIMITED"  # too many requests per short window
    OVERLOADED = "OVERLOADED"  # provider-side capacity problem, transient
    INVALID_KEY = "INVALID_KEY"
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    BAD_REQUEST = "BAD_REQUEST"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"  # other 5xx
    NETWORK = "NETWORK"  # connection/timeout before any response
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"  # a response we can't use
    UNKNOWN = "UNKNOWN"


_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
    "azure_ai": "AZURE_AI_API_KEY",
}


@dataclass(frozen=True)
class ProviderFailure:
    kind: ErrorKind
    provider: str
    operation: str  # "chat" | "embedding" | "chart"
    model: str | None = None
    status: int | str | None = None
    message: str = ""
    quota: str | None = None
    limit: int | None = None
    window: str | None = None  # "per-day" | "per-minute" when the provider says
    retry_after_s: float | None = None
    resets_at: datetime | None = None

    def summary(self) -> str:
        target = f"{self.provider}/{self.model}" if self.model else self.provider
        head = f"[{self.kind}] {target} {self.operation}"
        if self.kind in (ErrorKind.QUOTA_EXCEEDED, ErrorKind.RATE_LIMITED):
            parts = [f"quota {self.quota}" if self.quota else "limit reached"]
            if self.limit is not None:
                parts.append(f"limit {self.limit}" + (f" {self.window}" if self.window else ""))
            if self.resets_at is not None:
                parts.append(f"resets {self.resets_at:%Y-%m-%d %H:%M} UTC")
            elif self.retry_after_s is not None:
                parts.append(f"retry in {self.retry_after_s:.0f}s")
            return f"{head}: " + "; ".join(parts)
        if self.kind is ErrorKind.INVALID_KEY:
            env = _KEY_ENV_VARS.get(self.provider, "the API key")
            return f"{head}: API key rejected ({self.message}); check {env}"
        if self.kind is ErrorKind.OVERLOADED:
            return f"{head}: provider overloaded, transient ({self.message})"
        suffix = f" (status {self.status})" if self.status is not None else ""
        return f"{head}: {self.message}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = str(self.kind)
        data["resets_at"] = self.resets_at.isoformat() if self.resets_at else None
        return {k: v for k, v in data.items() if v not in (None, "")}


def _short(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- Google (Gemini) -------------------------------------------------------

_GOOGLE_QUOTA_RE = re.compile(
    r"Quota exceeded for metric:\s*(?P<metric>\S+?),\s*limit:\s*(?P<limit>\d+)"
    r"(?:,\s*model:\s*(?P<model>[\w.\-]+))?"
)
_GOOGLE_RETRY_RE = re.compile(r"retry in\s*(?P<seconds>[\d.]+)s", re.IGNORECASE)


def _google_error_parts(details: Any) -> tuple[str | None, dict[str, Any]]:
    """Returns (ErrorInfo reason, first QuotaFailure violation + retryDelay)
    from a google.rpc error body, which may or may not be wrapped in "error"."""
    error = details.get("error", details) if isinstance(details, dict) else {}
    reason, extra = None, {}
    for item in error.get("details") or []:
        kind = str(item.get("@type", ""))
        if kind.endswith("ErrorInfo"):
            reason = item.get("reason")
        elif kind.endswith("QuotaFailure"):
            violations = item.get("violations") or []
            if violations:
                extra.update(violations[0])
        elif kind.endswith("RetryInfo"):
            extra["retryDelay"] = item.get("retryDelay")
    return reason, extra


def classify_google(
    *,
    code: int | None,
    status: str | None,
    message: str,
    details: Any = None,
    operation: str,
    model: str | None,
) -> ProviderFailure:
    """Works for both a google.genai APIError (code/status/message/details)
    and ADK's error event, which carries only the status string and text."""
    if status and status.isdigit():
        code, status = int(status), None
    status = (status or "").upper()
    reason, quota_info = _google_error_parts(details)
    text = message or ""

    quota = quota_info.get("quotaMetric")
    limit = quota_info.get("quotaValue")
    window = None
    quota_id = str(quota_info.get("quotaId", ""))
    if "PerDay" in quota_id:
        window = "per-day"
    elif "PerMinute" in quota_id:
        window = "per-minute"
    match = _GOOGLE_QUOTA_RE.search(text)
    if match:
        quota = quota or match["metric"]
        limit = limit or match["limit"]
        model = model or match["model"]
    retry = None
    retry_match = _GOOGLE_RETRY_RE.search(text) or _GOOGLE_RETRY_RE.search(
        f"retry in {quota_info.get('retryDelay', '')}"
    )
    if retry_match:
        retry = float(retry_match["seconds"])

    common = dict(provider="gemini", operation=operation, model=model, status=code or status)
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        kind = (
            ErrorKind.RATE_LIMITED
            if window == "per-minute"
            else (
                ErrorKind.QUOTA_EXCEEDED
                if (quota or "quota" in text.lower())
                else ErrorKind.RATE_LIMITED
            )
        )
        return ProviderFailure(
            kind=kind,
            quota=quota,
            limit=int(limit) if limit is not None else None,
            window=window,
            retry_after_s=retry,
            message=_short(text),
            **common,
        )
    if (
        reason == "API_KEY_INVALID"
        or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or (code in (400, 401, 403) and "api key" in text.lower())
    ):
        return ProviderFailure(kind=ErrorKind.INVALID_KEY, message=_short(text), **common)
    if code == 503 or status == "UNAVAILABLE":
        return ProviderFailure(kind=ErrorKind.OVERLOADED, message=_short(text), **common)
    if (code is not None and code >= 500) or status in ("INTERNAL", "DEADLINE_EXCEEDED"):
        return ProviderFailure(kind=ErrorKind.UPSTREAM_ERROR, message=_short(text), **common)
    if code in (400, 404, 422) or status in (
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "FAILED_PRECONDITION",
    ):
        return ProviderFailure(kind=ErrorKind.BAD_REQUEST, message=_short(text), **common)
    return ProviderFailure(kind=ErrorKind.UNKNOWN, message=_short(text), **common)


# --- OpenAI-shaped HTTP APIs (OpenRouter, xAI) -----------------------------

_RETRY_AFTER_RE = re.compile(r"retry after\s*(?P<seconds>\d+)\s*second", re.IGNORECASE)
_OPENROUTER_LIMIT_RE = re.compile(r"Rate limit exceeded:\s*(?P<name>[\w\-]+)", re.IGNORECASE)


def classify_http(
    *, provider: str, status: int, body: Any, operation: str, model: str | None
) -> ProviderFailure:
    """``body`` is the parsed JSON (or None). Handles an error body under any
    status, including OpenRouter's HTTP 200 wrapping an upstream error."""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message") or "")
        embedded = error.get("code")
        metadata = error.get("metadata") or {}
    else:
        message = str(error or "")
        embedded, metadata = None, {}
    # xAI's shape is {"code": "invalid-argument", "error": "..."}.
    if isinstance(body, dict) and isinstance(body.get("code"), str) and not message:
        message = body["code"]
    # Azure sends numeric codes as strings ({"code": "401"}, seen live).
    if isinstance(embedded, str) and embedded.isdigit():
        embedded = int(embedded)
    effective = embedded if (status < 400 and isinstance(embedded, int)) else status
    common = dict(provider=provider, operation=operation, model=model, status=effective)
    lowered = message.lower()

    # Azure's content filter rejects a prompt with code "content_filter".
    if embedded == "content_filter":
        return ProviderFailure(
            kind=ErrorKind.BAD_REQUEST, message=_short(f"content filter: {message}"), **common
        )

    if metadata.get("error_type") == "provider_overloaded" or effective in (503, 529):
        return ProviderFailure(kind=ErrorKind.OVERLOADED, message=_short(message), **common)
    if effective == 429:
        name_match = _OPENROUTER_LIMIT_RE.search(message)
        name = name_match["name"] if name_match else None
        headers = metadata.get("headers") or {}
        daily = "per-day" in (name or "") or "daily" in str(metadata.get("limit_source", ""))
        limit = headers.get("X-RateLimit-Limit")
        reset_ms = headers.get("X-RateLimit-Reset")
        resets_at = (
            datetime.fromtimestamp(int(reset_ms) / 1000, tz=UTC)
            if reset_ms and str(reset_ms).isdigit()
            else None
        )
        retry = _RETRY_AFTER_RE.search(message)  # Azure: "Please retry after 22 seconds."
        return ProviderFailure(
            kind=ErrorKind.QUOTA_EXCEEDED if daily else ErrorKind.RATE_LIMITED,
            quota=name,
            limit=int(limit) if limit and str(limit).isdigit() else None,
            window="per-day" if daily else None,
            resets_at=resets_at,
            retry_after_s=float(retry["seconds"]) if retry else None,
            message=_short(message),
            **common,
        )
    if effective in (401, 403) or (effective == 400 and "api key" in lowered):
        return ProviderFailure(kind=ErrorKind.INVALID_KEY, message=_short(message), **common)
    if effective == 402:
        return ProviderFailure(
            kind=ErrorKind.INSUFFICIENT_CREDITS, message=_short(message), **common
        )
    if effective in (400, 404, 413, 422):
        return ProviderFailure(kind=ErrorKind.BAD_REQUEST, message=_short(message), **common)
    if isinstance(effective, int) and effective >= 500:
        return ProviderFailure(kind=ErrorKind.UPSTREAM_ERROR, message=_short(message), **common)
    return ProviderFailure(kind=ErrorKind.UNKNOWN, message=_short(message), **common)


# --- Exceptions ------------------------------------------------------------


def classify_exception(
    exc: BaseException, *, provider: str, operation: str, model: str | None
) -> ProviderFailure:
    existing = find_failure(exc)
    if existing is not None:
        return existing
    # Only consult google-genai if it is already loaded: if it isn't, this
    # exception cannot be one of its errors, and classifying another
    # provider's failure must not import the Google SDK.
    genai_errors = sys.modules.get("google.genai.errors")
    for candidate in _chain(exc):
        if genai_errors is not None and isinstance(candidate, genai_errors.APIError):
            return classify_google(
                code=candidate.code,
                status=candidate.status,
                message=candidate.message or str(candidate),
                details=candidate.details,
                operation=operation,
                model=model,
            )
        import httpx

        if isinstance(candidate, httpx.TransportError):
            return ProviderFailure(
                kind=ErrorKind.NETWORK,
                provider=provider,
                operation=operation,
                model=model,
                message=_short(f"{type(candidate).__name__}: {candidate}"),
            )
    return ProviderFailure(
        kind=ErrorKind.UNKNOWN,
        provider=provider,
        operation=operation,
        model=model,
        message=_short(f"{type(exc).__name__}: {exc}"),
    )


def _chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def find_failure(exc: BaseException) -> ProviderFailure | None:
    """The ProviderFailure attached to this exception or any it was raised
    from (e.g. RetrievalUnavailableError <- EmbeddingProviderError)."""
    for candidate in _chain(exc):
        failure = getattr(candidate, "failure", None)
        if isinstance(failure, ProviderFailure):
            return failure
    return None


def log_provider_failure(
    logger: logging.Logger, failure: ProviderFailure, exc: BaseException | None = None
) -> None:
    """One readable WARNING line with the structured fields attached; the
    traceback only at DEBUG."""
    logger.warning(
        failure.summary(), extra={"event": "provider_failure", "failure": failure.to_dict()}
    )
    if exc is not None and logger.isEnabledFor(logging.DEBUG):
        logger.debug("provider failure traceback", exc_info=exc)


def log_chart_extraction_failure(
    logger: logging.Logger,
    exc: BaseException,
    *,
    provider: str,
    model: str | None,
    failure: ProviderFailure | None = None,
) -> None:
    """Chart extraction failures never fail the turn; log them as one
    classified line saying so. Output that is not valid chart JSON is
    MALFORMED_RESPONSE rather than a provider error."""
    failure = (
        failure
        or find_failure(exc)
        or ProviderFailure(
            kind=ErrorKind.MALFORMED_RESPONSE,
            provider=provider,
            operation="chart",
            model=model,
            message=_short(f"output did not match the chart schema ({type(exc).__name__})"),
        )
    )
    logger.warning(
        f"{failure.summary()}; continuing without a chart",
        extra={"event": "chart_extraction_failed", "failure": failure.to_dict()},
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("chart extraction traceback", exc_info=exc)
