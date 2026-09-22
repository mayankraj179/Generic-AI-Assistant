from __future__ import annotations

import re
from collections.abc import Sequence

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
