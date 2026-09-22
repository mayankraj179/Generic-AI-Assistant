from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.principal import PrincipalContext
from app.db.database import create_session_factory
from app.db.models import ConversationRecord, MessageRecord


class ConversationNotFoundError(Exception):
    """Raised when a conversation doesn't exist, or exists but doesn't belong
    to the requesting principal. Deliberately doesn't distinguish the two
    cases — a caller must never be able to confirm that a conversation ID
    exists under a different tenant/principal by guessing a UUID.
    """


@dataclass(frozen=True)
class Conversation:
    id: uuid.UUID
    tenant_id: str
    assistant_id: str
    principal_id: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MessageCitation:
    document_title: str
    chunk_index: int


@dataclass(frozen=True)
class Message:
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    citations: tuple[MessageCitation, ...] = field(default_factory=tuple)
    sequence_no: int = 0
    created_at: datetime | None = None


class ConversationStore:
    """Owns conversation/message persistence for multi-turn chat.

    Every read AND write is scoped to the requesting ``PrincipalContext``
    (tenant_id + principal_id) — the same fail-closed ownership check
    ``RetrievalService``/``PgVectorStore`` apply to documents applies here to
    conversations: never trust a conversation_id alone, always re-verify who
    it belongs to.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._session_factory = session_factory or create_session_factory()

    async def create_conversation(
        self, *, principal: PrincipalContext, assistant_id: str
    ) -> Conversation:
        async with self._session_factory() as session:
            record = ConversationRecord(
                tenant_id=principal.tenant_id,
                assistant_id=assistant_id,
                principal_id=principal.principal_id,
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return _to_conversation(record)

    async def get_conversation(
        self, *, conversation_id: uuid.UUID, principal: PrincipalContext
    ) -> Conversation:
        async with self._session_factory() as session:
            record = await self._owned_conversation(session, conversation_id, principal)
            return _to_conversation(record)

    async def append_message(
        self,
        *,
        conversation_id: uuid.UUID,
        principal: PrincipalContext,
        role: str,
        content: str,
        citations: Sequence[MessageCitation] = (),
    ) -> Message:
        async with self._session_factory() as session:
            conversation = await self._owned_conversation(session, conversation_id, principal)

            next_sequence = await session.scalar(
                select(func.coalesce(func.max(MessageRecord.sequence_no), -1) + 1).where(
                    MessageRecord.conversation_id == conversation_id
                )
            )
            record = MessageRecord(
                conversation_id=conversation_id,
                role=role,
                content=content,
                citations=[
                    {"document_title": c.document_title, "chunk_index": c.chunk_index}
                    for c in citations
                ],
                sequence_no=next_sequence,
            )
            session.add(record)
            conversation.updated_at = func.now()
            await session.commit()
            await session.refresh(record)
            return _to_message(record)

    async def list_messages(
        self, *, conversation_id: uuid.UUID, principal: PrincipalContext
    ) -> list[Message]:
        async with self._session_factory() as session:
            await self._owned_conversation(session, conversation_id, principal)

            result = await session.execute(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.sequence_no)
            )
            return [_to_message(record) for record in result.scalars().all()]

    async def _owned_conversation(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        principal: PrincipalContext,
    ) -> ConversationRecord:
        record = await session.get(ConversationRecord, conversation_id)
        if (
            record is None
            or record.tenant_id != principal.tenant_id
            or record.principal_id != principal.principal_id
        ):
            raise ConversationNotFoundError("conversation not found")
        return record


def _to_conversation(record: ConversationRecord) -> Conversation:
    return Conversation(
        id=record.id,
        tenant_id=record.tenant_id,
        assistant_id=record.assistant_id,
        principal_id=record.principal_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_message(record: MessageRecord) -> Message:
    return Message(
        id=record.id,
        conversation_id=record.conversation_id,
        role=record.role,
        content=record.content,
        citations=tuple(
            MessageCitation(document_title=c["document_title"], chunk_index=c["chunk_index"])
            for c in (record.citations or [])
        ),
        sequence_no=record.sequence_no,
        created_at=record.created_at,
    )
