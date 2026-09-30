"""Per-request identifiers carried through contextvars, so every log line and
audit row for one request can be tied together without passing them around."""

from __future__ import annotations

import hashlib
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
assistant_id_var: ContextVar[str | None] = ContextVar("assistant_id", default=None)
principal_ref_var: ContextVar[str | None] = ContextVar("principal_ref", default=None)
conversation_id_var: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def principal_ref(tenant_id: str, principal_id: str) -> str:
    """A stable, non-reversible reference to a principal for logs and audit
    rows: the raw principal id is real user identity and never logged."""
    digest = hashlib.sha256(f"{tenant_id}:{principal_id}".encode()).hexdigest()
    return f"p_{digest[:12]}"


def current_context() -> dict[str, str | None]:
    return {
        "request_id": request_id_var.get(),
        "assistant_id": assistant_id_var.get(),
        "principal": principal_ref_var.get(),
        "conversation_id": conversation_id_var.get(),
    }
