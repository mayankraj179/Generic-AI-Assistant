"""Rough per-provider quota usage for the current quota window, from turn_audit.

    python -m app.observability.usage          # from the audit table
    python -m app.observability.usage --live   # plus OpenRouter's own counter

Counts are a lower bound: only provider calls made inside chat turns through
this application are audited. Ingestion and ad-hoc scripts also spend quota
but are not recorded, so the provider's own counter (--live, OpenRouter only)
is authoritative when it disagrees."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

# When each provider's free-tier daily allowance resets.
_WINDOW_TZ = {
    "gemini": ZoneInfo("America/Los_Angeles"),  # Google AI free tier: midnight Pacific
    "openrouter": UTC,  # free-models-per-day: X-RateLimit-Reset is 00:00 UTC
    "xai": UTC,  # paid per token; shown per UTC day for reference
}
# Limits confirmed from a provider's own response, not assumed. OpenRouter's
# free-model cap was read from X-RateLimit-Limit on a real 429 (2026-09-30).
_CONFIRMED_DAILY_LIMITS = {("openrouter", "model"): 50, ("openrouter", "embedding"): 50}


@dataclass
class UsageLine:
    provider: str
    kind: str
    model: str
    calls: int
    window_start: datetime
    prompt_tokens: int = 0
    completion_tokens: int = 0
    limit: int | None = None
    limit_source: str | None = None
    last_quota_error: str | None = None


def window_start(provider: str, now: datetime) -> datetime:
    tz = _WINDOW_TZ.get(provider, UTC)
    local_midnight = datetime.combine(now.astimezone(tz).date(), time(0), tzinfo=tz)
    return local_midnight.astimezone(UTC)


async def usage_summary(session_factory=None, *, now: datetime | None = None) -> list[UsageLine]:
    from app.db.database import create_session_factory
    from app.db.models import TurnAuditRecord

    now = now or datetime.now(UTC)
    earliest = min(window_start(p, now) for p in _WINDOW_TZ) - timedelta(hours=1)
    factory = session_factory or create_session_factory()
    async with factory() as session:
        rows = (
            await session.execute(
                select(
                    TurnAuditRecord.created_at,
                    TurnAuditRecord.provider_calls,
                    TurnAuditRecord.error,
                    TurnAuditRecord.error_kind,
                ).where(TurnAuditRecord.created_at >= earliest)
            )
        ).all()

    lines: dict[tuple[str, str, str], UsageLine] = {}
    quota_errors: dict[str, dict] = {}
    for created_at, provider_calls, error, error_kind in rows:
        for call in provider_calls or []:
            provider = call["provider"]
            start = window_start(provider, now)
            if created_at < start:
                continue
            key = (provider, call["kind"], call["model"])
            line = lines.setdefault(key, UsageLine(provider, call["kind"], call["model"], 0, start))
            line.calls += int(call.get("count", 0))
            line.prompt_tokens += int(call.get("prompt_tokens", 0) or 0)
            line.completion_tokens += int(call.get("completion_tokens", 0) or 0)
        if error_kind == "QUOTA_EXCEEDED" and error:
            provider = error.get("provider", "")
            if created_at >= window_start(provider, now):
                previous = quota_errors.get(provider)
                if previous is None or created_at > previous["_at"]:
                    quota_errors[provider] = {**error, "_at": created_at}

    for line in lines.values():
        seen = quota_errors.get(line.provider)
        if seen is not None:
            line.last_quota_error = (
                f"{seen.get('quota') or 'quota'} exhausted at {seen['_at']:%H:%M} UTC"
                + (f", resets {seen['resets_at']}" if seen.get("resets_at") else "")
            )
            if seen.get("limit"):
                line.limit, line.limit_source = int(seen["limit"]), "provider error"
        if line.limit is None and (line.provider, line.kind) in _CONFIRMED_DAILY_LIMITS:
            line.limit = _CONFIRMED_DAILY_LIMITS[(line.provider, line.kind)]
            line.limit_source = "confirmed header"
    return sorted(lines.values(), key=lambda u: (u.provider, u.kind, u.model))


async def openrouter_live_counter() -> dict | None:
    """OpenRouter's own free-model counter (GET /api/v1/key costs nothing)."""
    import httpx

    from app.config.settings import Settings

    key = Settings().openrouter_api_key
    if not key:
        return None
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}
        )
    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}"}
    return (response.json().get("data") or {}).get("free_model_daily_requests")


def _render(lines: list[UsageLine], live: dict | None) -> str:
    if not lines:
        out = ["No audited provider calls in the current quota windows."]
    else:
        out = [
            f"{'provider':<11}{'kind':<10}{'model':<40}{'calls':>6}{'limit':>7}"
            "  window since (UTC)",
        ]
        for u in lines:
            limit = f"{u.limit}" if u.limit is not None else "?"
            out.append(
                f"{u.provider:<11}{u.kind:<10}{u.model[:39]:<40}{u.calls:>6}{limit:>7}  "
                f"{u.window_start:%Y-%m-%d %H:%M}"
            )
            if u.last_quota_error:
                out.append(f"{'':<21}! {u.last_quota_error}")
    out.append("Counts cover chat turns only (ingestion/scripts are not audited): a lower bound.")
    if live is not None:
        out.append(f"OpenRouter live counter (authoritative): {live}")
    return "\n".join(out)


async def _main(live: bool) -> None:
    lines = await usage_summary()
    print(_render(lines, await openrouter_live_counter() if live else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="also query OpenRouter's counter")
    asyncio.run(_main(parser.parse_args().live))
