"""make chunks.embedding dimensionless and tag every row with its embedding model.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-27

Lets assistants on different embedding models (and dimensions) share the
chunks table: Gemini-backed assistants stay on gemini-embedding-001 (3072-dim)
while finance_assistant_openrouter moves to OpenRouter's free
nvidia/nemotron-3-embed-1b:free (2048-dim). This is pgvector's documented
approach for mixed dimensions: an untyped `vector` column, with every query
restricted to one model (PgVectorStore.search filters on embedding_model).
Adding another model later needs no migration at all.

Non-destructive, unlike 0003:
  - `vector(3072)` -> `vector` only drops the type modifier. Verified live
    against this project's pgvector 0.8.6 on a scratch table that the stored
    values are byte-identical afterwards. No rows are deleted or rewritten.
  - `embedding_model` is added to chunks and documents and backfilled from
    what each existing row was actually embedded with.
    finance_assistant_openrouter was ingested through
    OpenRouterEmbeddingProvider's then-default google/gemini-embedding-001;
    everything else went through GeminiEmbeddingProvider. (Leftover
    pytest-fixture rows, embedded by HashEmbeddingProvider, also get the
    gemini tag. They're test residue that no real query reads.)

Downgrade can't be lossless once non-3072-dim rows exist, so it deletes those
chunk rows before restoring `vector(3072)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

GEMINI_DIRECT = "gemini:gemini-embedding-001"
OPENROUTER_GEMINI = "openrouter:google/gemini-embedding-001"
BACKFILL = f"""
    CASE WHEN assistant_id = 'finance_assistant_openrouter'
         THEN '{OPENROUTER_GEMINI}' ELSE '{GEMINI_DIRECT}' END
"""


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector")
    for table in ("chunks", "documents"):
        op.add_column(table, sa.Column("embedding_model", sa.Text(), nullable=True))
        op.execute(f"UPDATE {table} SET embedding_model = {BACKFILL}")
        op.alter_column(table, "embedding_model", nullable=False)
    op.create_index(
        "ix_chunks_tenant_assistant_model_current",
        "chunks",
        ["tenant_id", "assistant_id", "embedding_model", "is_current"],
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_tenant_assistant_model_current", table_name="chunks")
    op.execute("DELETE FROM chunks WHERE vector_dims(embedding) <> 3072")
    for table in ("chunks", "documents"):
        op.drop_column(table, "embedding_model")
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(3072)")
