from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.ingestion.pipeline import Chunk
from app.orchestration.errors import InputGuardrailError

logger = logging.getLogger(__name__)


def validate_input_message(message: str, *, max_input_chars: int) -> None:
    """Size/format guardrail on the user's own message. Raises
    InputGuardrailError (never a bare ValueError) for empty/whitespace-only
    input or input beyond the configured length — never silently truncates.
    """
    if not message or not message.strip():
        raise InputGuardrailError("message must not be empty")
    if len(message) > max_input_chars:
        raise InputGuardrailError(
            f"message exceeds the maximum allowed length of {max_input_chars} characters"
        )


# Heuristic, pattern-based first line of defense against prompt injection
# embedded in RETRIEVED (untrusted) document content — not a guarantee and
# not a classifier. Retrieved documents are explicitly untrusted/adversarial
# content per the framework's threat model, so this screens chunk content
# before it ever reaches the prompt, the same way the user's own message
# gets validate_input_message() above.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\b", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+your\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(
        r"forget\s+(all\s+|any\s+)?(previous|prior)\s+(instructions|rules)", re.IGNORECASE
    ),
    re.compile(
        r"act\s+as\s+(if\s+you\s+are\s+)?(a|an)\b.{0,40}\bwithout\s+restrictions", re.IGNORECASE
    ),
]


def _contains_injection_pattern(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def screen_retrieved_chunks(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Drops any retrieved chunk whose content matches an injection
    pattern, logging each one dropped. One poisoned document must not take
    down retrieval for everything else, so this strips and continues rather
    than failing the whole turn — the caller proceeds with whatever chunks
    remain (possibly none, which the caller must treat the same as "no
    authorized context found").
    """
    safe_chunks: list[Chunk] = []
    for chunk in chunks:
        if _contains_injection_pattern(chunk.display_text) or _contains_injection_pattern(
            chunk.embedded_text
        ):
            logger.warning(
                "dropped a retrieved chunk matching a prompt-injection pattern: "
                "document=%r chunk_index=%s",
                chunk.document_title,
                chunk.chunk_index,
            )
            continue
        safe_chunks.append(chunk)
    return safe_chunks
