"""Masks credentials in log output.

Two layers: known secret values (every connector credential and access token
is registered here the moment it is resolved or issued, so an accidental log
of one is masked wherever it appears), and patterns for credential shapes
this process never saw (bearer headers, JWTs, ``client_secret=...``).
configure_logging applies redact() to every formatted line.
"""

from __future__ import annotations

import re
import threading

MASK = "[REDACTED]"
# Shorter values are too likely to collide with ordinary text.
_MIN_SECRET_LENGTH = 8

_lock = threading.Lock()
_known_secrets: set[str] = set()

_PATTERNS = (
    re.compile(r"(?i)(\b(?:bearer|basic)\s+)[A-Za-z0-9._~+/=-]{6,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"),
    re.compile(
        r"(?i)(\b(?:access_token|refresh_token|id_token|client_secret|api[_-]?key|"
        r"x-api-key|password)\\?[\"']?\s*[:=]\s*\\?[\"']?)"
        r"[^\s\"'\\&,;}]+"
    ),
)


def register_secret(value: str | None) -> None:
    if value and len(value) >= _MIN_SECRET_LENGTH:
        with _lock:
            _known_secrets.add(value)


def redact(text: str) -> str:
    if not text:
        return text
    with _lock:
        secrets = sorted(_known_secrets, key=len, reverse=True)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, MASK)
    for pattern in _PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + MASK, text)
    return text
