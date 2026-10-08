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


class AdminSyncRequest(BaseModel):
    """Syncs one assistant's configured knowledge_sources. Access labels come
    only from the assistant's config. tenant_id is admin-supplied, the same
    deliberate exception AdminIngestRequest documents: retrieval scopes by
    the principal's tenant, so the admin says which tenant the content is for.
    """

    assistant_id: str
    tenant_id: str
    # Sync only this source (its configured name); all of them if omitted.
    source: str | None = None


class SourceSyncSummary(BaseModel):
    name: str
    type: str
    status: str
    error: str | None = None
    discovered: int
    created: int
    updated: int
    unchanged: int
    empty: int
    failed: int
    retired: int
    chunks_embedded: int
    failed_items: list[str]
    retired_items: list[str]


class AdminSyncResponse(BaseModel):
    assistant_id: str
    tenant_id: str
    sources: list[SourceSyncSummary]
