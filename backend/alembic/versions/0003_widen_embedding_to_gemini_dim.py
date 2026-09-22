"""widen chunks.embedding to gemini-embedding-001's native 3072 dimensions.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-22

Destructive: deletes all existing rows in `chunks` before changing the
column type. pgvector enforces an exact dimension match per row, so any
existing 1536-dim vectors are incompatible with the new 3072-wide column and
there is no lossless way to reinterpret one as the other. Verified live
against this project's Postgres/pgvector (0.8.6): `ALTER COLUMN ... TYPE
vector(N)` succeeds in-place once the table has no rows of the old width,
and raises `asyncpg.exceptions.DataError: expected N dimensions, not M`
otherwise — no drop/recreate of the column or a vector index is required
(this table has no ivfflat/hnsw index on `embedding` to rebuild).

Acceptable here because nothing in this system has ingested real content
yet — only short-lived test/session data that already gets cleaned up.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_DIM = 1536
NEW_DIM = 3072


def upgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({NEW_DIM})")


def downgrade() -> None:
    op.execute("DELETE FROM chunks")
    op.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({OLD_DIM})")
