from __future__ import annotations

import re

# Deliberately simple pattern-based detection — a first line of defense, not
# a guarantee or a full PII classifier. Shared between input and output
# guardrail checks (app/guardrails/input.py, app/guardrails/output.py) so
# both sides of the conversation apply the same definition of "looks like
# PII".
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
_SSN_PATTERN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)


def detect_pii(text: str) -> list[str]:
    """Returns which PII categories a heuristic pattern matched in ``text``
    — any of ``"email"``, ``"phone"``, ``"ssn"`` — or an empty list if none
    matched. Order-independent; a given category appears at most once.
    """
    categories: list[str] = []
    if _EMAIL_PATTERN.search(text):
        categories.append("email")
    if _SSN_PATTERN.search(text):
        categories.append("ssn")
    if _PHONE_PATTERN.search(text):
        categories.append("phone")
    return categories
