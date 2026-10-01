from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

# Heuristic check for the model's own refusal/non-answer language — not true
# semantic entailment checking. Used as an output-side backstop: even when
# retrieval found relevant-enough chunks (input-side grounded=True per
# ChatOrchestrator._prepare_turn's similarity + injection filtering), the
# model can still legitimately say it can't answer from them, and a reply
# like that must not carry citations/grounded=True just because *something*
# relevant was retrieved.
_REFUSAL_PATTERNS = [
    re.compile(r"\bI\s+(cannot|can't|am unable to)\s+answer\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+contain\s+(any\s+)?information\b", re.IGNORECASE),
    re.compile(
        r"\bdo(es)?\s+not\s+(provide|contain|include)\s+(enough|sufficient)?\s*information\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+don'?t\s+have\s+(enough|access\s+to\s+(the|any))\s+information\b", re.IGNORECASE
    ),
    re.compile(
        r"\bnot\s+(mentioned|covered)\s+in\s+the\s+(retrieved|provided|available)\s+"
        r"(content|context|documents?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bI'?m\s+sorry,?\s+(but\s+)?I\s+(cannot|can't|don'?t)\b", re.IGNORECASE),
]


def reply_appears_to_refuse(text: str) -> bool:
    """True if the model's own reply text looks like a refusal/non-answer.

    A small fixed phrase set, not semantic entailment checking — good
    enough to catch the common "the retrieved content doesn't actually
    answer this" phrasing without trying to prove the reply genuinely used
    the retrieved context.
    """
    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


def check_unsafe_output(text: str, *, patterns: Sequence[str]) -> bool:
    """Minimal, config-driven deny-list check on the model's output text —
    NOT a moderation-model call or true unsafe-content classification.

    ``patterns`` are regex strings from AssistantConfig.guardrails
    .unsafe_output_patterns. An empty list (the default) means this check
    is off: "unsafe" is domain-specific per assistant, so there's no
    sensible universal default to ship built in — each assistant opts in by
    configuring its own patterns.
    """
    if not patterns:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


# ---------------------------------------------------------------------------
# Tool-result evidence: on a turn with no document context, a reply stands only
# if it states a value some successful tool call returned. "Some tool call
# succeeded" is not enough: live on 2026-09-30, gpt-6-luna called
# current_datetime on nearly every out-of-scope question.
# ---------------------------------------------------------------------------

_REPLY_NUMBER = re.compile(r"(?<![\d.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_REPLY_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)  # fmt: skip
# Keys whose value only echoes the model's own tool arguments, so a reply that
# repeats them ("2+2", "UTC") shows nothing the tool computed.
_ECHO_KEYS = frozenset({"expression", "timezone", "kind"})


def _reply_numbers(text: str) -> list[tuple[float, int]]:
    """Every number in the reply, with the decimals it was written with."""
    numbers = []
    for match in _REPLY_NUMBER.finditer(text):
        raw = match.group().replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        numbers.append((float(raw), decimals))
    return numbers


def _states_number(value: float, numbers: list[tuple[float, int]]) -> bool:
    # A reply may round the tool's value ("36" for 36.0, "12.3" for 12.34),
    # or drop the sign ("82 days" for a -82 day difference).
    return any(round(abs(value), decimals) == abs(number) for number, decimals in numbers)


def _states_datetime(iso: str, text: str, numbers: list[tuple[float, int]]) -> bool:
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return False
    lowered = text.lower()
    # abs(): an ISO date "2026-09-30" reads as 2026, -9, -30.
    whole = {abs(int(number)) for number, decimals in numbers if decimals == 0}
    month = _MONTHS[moment.month - 1]
    names_month = month in lowered or re.search(rf"\b{month[:3]}\b", lowered) is not None
    if moment.year in whole and moment.day in whole and (names_month or moment.month in whole):
        return True
    hours = {moment.hour, moment.hour % 12 or 12}
    return any(
        int(hour) in hours and int(minute) == moment.minute
        for hour, minute in _REPLY_TIME.findall(text)
    )


def reply_uses_tool_result(text: str, tool_results: Sequence[dict]) -> bool:
    """True if ``text`` states at least one value from ``tool_results``: a
    returned number, the returned date or time, or another returned string
    (a weekday, a formatted date) verbatim.

    A deterministic, fail-closed check, not proof that the whole reply rests on
    tools. A reply that states a tool value and ALSO answers something else
    ("2 + 2 = 4, and the capital of France is Paris") passes.
    """
    numbers = _reply_numbers(text)
    lowered = text.lower()
    for result in tool_results:
        for key, value in result.items():
            if key in _ECHO_KEYS or isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                if _states_number(float(value), numbers):
                    return True
            elif isinstance(value, str) and value.strip():
                if key == "iso" and _states_datetime(value, text, numbers):
                    return True
                if len(value.strip()) >= 3 and value.strip().lower() in lowered:
                    return True
    return False
