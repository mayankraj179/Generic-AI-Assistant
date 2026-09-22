from __future__ import annotations

import uuid

import pytest

from app.core.principal import PrincipalContext
from app.services.conversation_store import (
    ConversationNotFoundError,
    ConversationStore,
    MessageCitation,
)


def _principal(**overrides) -> PrincipalContext:
    defaults = dict(
        tenant_id="tenant-a",
        principal_id="person-1",
        labels=frozenset({"role:employee"}),
        clearance=1,
    )
    defaults.update(overrides)
    return PrincipalContext(**defaults)


@pytest.mark.asyncio
async def test_create_conversation_round_trip():
    store = ConversationStore()
    principal = _principal()

    conversation = await store.create_conversation(principal=principal, assistant_id="hr_assistant")

    assert conversation.tenant_id == principal.tenant_id
    assert conversation.principal_id == principal.principal_id
    assert conversation.assistant_id == "hr_assistant"

    fetched = await store.get_conversation(conversation_id=conversation.id, principal=principal)
    assert fetched.id == conversation.id


@pytest.mark.asyncio
async def test_append_message_assigns_increasing_sequence_numbers():
    store = ConversationStore()
    principal = _principal()
    conversation = await store.create_conversation(principal=principal, assistant_id="hr_assistant")

    first = await store.append_message(
        conversation_id=conversation.id, principal=principal, role="user", content="hi"
    )
    second = await store.append_message(
        conversation_id=conversation.id, principal=principal, role="assistant", content="hello!"
    )
    third = await store.append_message(
        conversation_id=conversation.id, principal=principal, role="user", content="thanks"
    )

    assert (first.sequence_no, second.sequence_no, third.sequence_no) == (0, 1, 2)


@pytest.mark.asyncio
async def test_append_message_stores_citations_in_the_shared_shape():
    store = ConversationStore()
    principal = _principal()
    conversation = await store.create_conversation(principal=principal, assistant_id="hr_assistant")

    message = await store.append_message(
        conversation_id=conversation.id,
        principal=principal,
        role="assistant",
        content="Fifteen days per year.",
        citations=[MessageCitation(document_title="leave_policy", chunk_index=0)],
    )

    assert message.citations == (MessageCitation(document_title="leave_policy", chunk_index=0),)


@pytest.mark.asyncio
async def test_list_messages_returns_them_in_sequence_order():
    store = ConversationStore()
    principal = _principal()
    conversation = await store.create_conversation(principal=principal, assistant_id="hr_assistant")

    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        await store.append_message(
            conversation_id=conversation.id, principal=principal, role=role, content=f"turn {i}"
        )

    messages = await store.list_messages(conversation_id=conversation.id, principal=principal)

    assert [m.content for m in messages] == [f"turn {i}" for i in range(5)]
    assert [m.sequence_no for m in messages] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_get_conversation_denies_a_different_principal_same_tenant():
    store = ConversationStore()
    owner = _principal(principal_id="owner")
    attacker = _principal(principal_id="attacker")
    conversation = await store.create_conversation(principal=owner, assistant_id="hr_assistant")

    with pytest.raises(ConversationNotFoundError):
        await store.get_conversation(conversation_id=conversation.id, principal=attacker)


@pytest.mark.asyncio
async def test_get_conversation_denies_a_different_tenant_same_principal_id():
    store = ConversationStore()
    owner = _principal(tenant_id="tenant-a", principal_id="shared-id")
    attacker = _principal(tenant_id="tenant-b", principal_id="shared-id")
    conversation = await store.create_conversation(principal=owner, assistant_id="hr_assistant")

    with pytest.raises(ConversationNotFoundError):
        await store.get_conversation(conversation_id=conversation.id, principal=attacker)


@pytest.mark.asyncio
async def test_append_message_denies_a_different_principal():
    store = ConversationStore()
    owner = _principal(principal_id="owner")
    attacker = _principal(principal_id="attacker")
    conversation = await store.create_conversation(principal=owner, assistant_id="hr_assistant")

    with pytest.raises(ConversationNotFoundError):
        await store.append_message(
            conversation_id=conversation.id,
            principal=attacker,
            role="user",
            content="an attacker's message",
        )

    # The owner's conversation must remain untouched.
    messages = await store.list_messages(conversation_id=conversation.id, principal=owner)
    assert messages == []


@pytest.mark.asyncio
async def test_list_messages_denies_a_different_principal():
    store = ConversationStore()
    owner = _principal(principal_id="owner")
    attacker = _principal(principal_id="attacker")
    conversation = await store.create_conversation(principal=owner, assistant_id="hr_assistant")
    await store.append_message(
        conversation_id=conversation.id, principal=owner, role="user", content="secret"
    )

    with pytest.raises(ConversationNotFoundError):
        await store.list_messages(conversation_id=conversation.id, principal=attacker)


@pytest.mark.asyncio
async def test_get_conversation_denies_a_nonexistent_id():
    store = ConversationStore()
    principal = _principal()

    with pytest.raises(ConversationNotFoundError):
        await store.get_conversation(conversation_id=uuid.uuid4(), principal=principal)
