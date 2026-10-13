"""Deterministic handling of plain greetings ("hi", "good morning", ...).

ChatOrchestrator answers a message with a fixed reply, before retrieval or
any model or embedding call, only when the WHOLE message is a greeting:

    <greeting> [<filler>] [<how are you>]

e.g. "hi", "Hello there!", "good morning team, how are you?". Anything else,
including a greeting followed by a request ("hi, how many casual leaves do I
get?"), goes through the normal flow. The phrase lists below are the single
place to extend.

Matching is done on a normalised form: lowercase, punctuation and emoji
removed, whitespace collapsed, and every run of a repeated letter collapsed
to one letter (on both sides, so "hiii", "heyyy" and "hellooo" match "hi",
"hey" and "hello").
"""

from __future__ import annotations

import re
import unicodedata

GREETINGS: tuple[str, ...] = (
    "hi",
    "hello",
    "hey",
    "hiya",
    "howdy",
    "namaste",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
)
FILLERS: tuple[str, ...] = ("there", "team", "everyone", "all")
HOW_ARE_YOU: tuple[str, ...] = ("how are you", "how are you doing", "how r u")

DEFAULT_REPLY = "Hi! I'm {display_name}. How can I help you today?"

# No greeting is longer than this once normalised; skip the regex work for
# anything that is.
_MAX_GREETING_CHARS = 120
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_REPEATS = re.compile(r"(.)\1+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_WORD.sub(" ", text)  # punctuation, emoji, symbols
    text = _REPEATS.sub(r"\1", text)  # hiii -> hi, hellooo -> helo
    return " ".join(text.split())


def _alternation(phrases: tuple[str, ...] | list[str]) -> str:
    forms = sorted({normalise(p) for p in phrases if normalise(p)}, key=len, reverse=True)
    return "|".join(re.escape(f) for f in forms)


def is_greeting(message: str, *, display_name: str = "") -> bool:
    """True only when the whole message is a greeting, optionally followed by
    a filler word (or the assistant's display name) and "how are you"."""
    text = normalise(message)
    if not text or len(text) > _MAX_GREETING_CHARS:
        return False
    fillers = [*FILLERS, display_name] if display_name.strip() else list(FILLERS)
    pattern = (
        rf"(?:{_alternation(GREETINGS)})"
        rf"(?: (?:{_alternation(fillers)}))?"
        rf"(?: (?:{_alternation(HOW_ARE_YOU)}))?"
    )
    return re.fullmatch(pattern, text) is not None


def greeting_reply(display_name: str, message: str | None = None) -> str:
    """The fixed reply. ``message`` (from the assistant's ``greeting:`` block)
    may use ``{display_name}``; it is substituted literally, not formatted."""
    return (message or DEFAULT_REPLY).replace("{display_name}", display_name).strip()
