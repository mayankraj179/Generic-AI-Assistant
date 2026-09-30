from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Boolean, DateTime, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# gemini-embedding-001's native output dimension (verified live against the
# installed google-genai SDK — see app/services/gemini_embedding.py), i.e. the
# default embedder's width. Since alembic 0005 this no longer constrains the
# chunks.embedding column, which is dimensionless; each row's vector space is
# identified by chunks.embedding_model instead.
EMBEDDING_DIM = 3072


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    assistant_id: Mapped[str] = mapped_column(String, nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, server_default="")
    # model_id of the embedder behind this document's current chunk
    # generation. Part of IngestionService's skip-if-unchanged fingerprint,
    # so re-ingesting after an embedding-model change re-embeds instead of
    # skipping.
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    access_labels: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default="{}",
    )
    acl_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    chunks: Mapped[list[ChunkRecord]] = relationship(back_populates="document")


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    assistant_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    display_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedded_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Dimensionless on purpose (pgvector's documented approach for mixed
    # dimensions): embedding_model says which vector space each row is in,
    # and PgVectorStore.search only ever compares within one model. A
    # cross-dimension comparison that slipped through would raise
    # "different vector dimensions", never return a silently wrong score.
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    access_labels: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default="{}",
    )
    acl_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    generation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    assistant_id: Mapped[str] = mapped_column(String, nullable=False)
    principal_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    messages: Mapped[list[MessageRecord]] = relationship(
        back_populates="conversation", order_by="MessageRecord.sequence_no"
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="[]",
    )
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    conversation: Mapped[ConversationRecord] = relationship(back_populates="messages")


class TurnAuditRecord(Base):
    """Operational metadata for one chat turn (alembic 0006): which provider
    and model handled it, whether it was grounded, what tools and guardrails
    did, how many provider calls it made, and how it failed if it did. No
    message content; the conversation itself lives in messages. There is no
    foreign key to conversations, so a row survives a deleted conversation and
    exists for failed turns, which persist no messages."""

    __tablename__ = "turn_audit"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    request_id: Mapped[str | None] = mapped_column(String, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)
    operation: Mapped[str] = mapped_column(String, nullable=False)
    tenant_id: Mapped[str] = mapped_column(String, nullable=False)
    assistant_id: Mapped[str] = mapped_column(String, nullable=False)
    principal_ref: Mapped[str] = mapped_column(String, nullable=False)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    outcome: Mapped[str] = mapped_column(String, nullable=False)
    grounded: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    citations_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    retrieval: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    chart_returned: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    tool_calls: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, server_default="[]")
    guardrail_actions: Mapped[list[dict]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    provider_calls: Mapped[list[dict]] = mapped_column(JSONB, nullable=False, server_default="[]")
    error_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
