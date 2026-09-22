from __future__ import annotations

from pydantic import BaseModel


class AdminIngestRequest(BaseModel):
    """Deliberately different trust boundary from ChatRequest: this is an
    admin operation, not something a principal does on their own behalf, so
    tenant_id/access_labels are caller-supplied here rather than derived
    from the authenticated principal the way /chat forbids. Gated behind the
    admin:ingest permission (see app/auth/policy.py) precisely because this
    is the one place in the API where that's an intentional, scoped
    exception to the framework's usual "never trust tenant_id from the
    client" rule.
    """

    source_path: str
    assistant_id: str
    tenant_id: str
    access_labels: list[str] | None = None


class AdminIngestResponse(BaseModel):
    source_uri: str
    chunks_ingested: int
