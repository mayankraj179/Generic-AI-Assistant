"""turn_audit: per-turn operational metadata (provider, model, grounding, tool
calls, guardrail actions, provider-call counts, classified errors).

Additive only: creates one table, touches nothing existing.

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "turn_audit",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("assistant_id", sa.String(), nullable=False),
        sa.Column("principal_ref", sa.String(), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("grounded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("citations_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retrieval", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("chart_returned", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("guardrail_actions", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("provider_calls", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("error_kind", sa.String(), nullable=True),
        sa.Column("error", postgresql.JSONB(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
    )
    op.create_index("ix_turn_audit_created_at", "turn_audit", ["created_at"])
    op.create_index("ix_turn_audit_assistant_created", "turn_audit", ["assistant_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_turn_audit_assistant_created", table_name="turn_audit")
    op.drop_index("ix_turn_audit_created_at", table_name="turn_audit")
    op.drop_table("turn_audit")
