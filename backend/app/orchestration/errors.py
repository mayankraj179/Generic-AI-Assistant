from __future__ import annotations


class AssistantAccessDeniedError(Exception):
    """Raised when an authenticated principal is not permitted to use a given assistant."""


class RetrievalUnavailableError(Exception):
    """Raised when the retrieval backend cannot be reached or fails unexpectedly.

    Wraps the underlying cause so callers never leak database connection
    details in an API response.
    """


class InputGuardrailError(Exception):
    """Raised when the user's input fails a guardrail check (empty/oversized,
    or a blocked PII pattern) before retrieval or a model call ever runs.

    The message is safe to show the caller as-is — unlike
    RetrievalUnavailableError/ModelProviderError, guardrail messages never
    contain internal detail, only a description of which input rule failed.
    A clean, expected rejection; never let a raw validation/regex exception
    surface as a 500.
    """
