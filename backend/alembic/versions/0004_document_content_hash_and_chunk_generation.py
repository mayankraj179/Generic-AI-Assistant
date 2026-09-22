"""add content_hash to documents and is_current to chunks.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-23

Non-destructive, unlike 0003. `content_hash` on `documents` gets a
server_default of '' (never a real sha256 hex digest, which is always 64
hex chars) — existing rows simply won't match any freshly-computed hash on
their next re-ingest, which correctly triggers the atomic-replace path
rather than incorrectly being treated as unchanged; no backfill needed.
`is_current` on `chunks` defaults to true — every existing chunk row really
is the (only, so far) current generation, so the default is already
correct with no data migration required.

`generation_id` already existed on `chunks` (added in 0001) but nothing used
it to distinguish current from superseded generations until now; `is_current`
is the piece that actually makes that distinction enforceable in queries.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(length=64), nullable=False, server_default=""),
    )
    op.add_column(
        "chunks",
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    # Mirrors the existing ix_chunks_tenant_assistant pattern (0001) — every
    # query that scopes by tenant/assistant now also scopes by is_current,
    # so the index should cover all three the same way the ACL/relevance
    # query already filters on tenant_id + assistant_id together.
    op.create_index(
        "ix_chunks_tenant_assistant_current",
        "chunks",
        ["tenant_id", "assistant_id", "is_current"],
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_tenant_assistant_current", table_name="chunks")
    op.drop_column("chunks", "is_current")
    op.drop_column("documents", "content_hash")
