import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiAuthError, type AssistantSummary, getAssistants } from "./api";
import { getAccessToken, type SignedInUser } from "./auth";
import { API_BASE } from "./config";
import { type ConversationSummary, loadConversations, removeConversation, touchConversation } from "./history";
import { useChatSession } from "./useChatSession";

export const PREFERRED_ASSISTANT = "hr_assistant";

interface ControllerOptions {
  user: SignedInUser;
  onSignedOut: () => void;
  defaultAssistant?: string;
}

/** Everything the chat layouts render from: assistants, the selected
 * assistant, the sidebar history and the one live chat session. Owned by
 * whichever shell hosts the chat (the full-page app or the launcher), so
 * switching layouts never restarts a conversation. */
export function useChatController({ user, onSignedOut, defaultAssistant = PREFERRED_ASSISTANT }: ControllerOptions) {
  const [assistants, setAssistants] = useState<AssistantSummary[]>([]);
  const [assistantId, setAssistantId] = useState(defaultAssistant);
  const [conversations, setConversations] = useState<ConversationSummary[]>(() => loadConversations(user.subject));
  const [loadError, setLoadError] = useState<string>();

  const onTurnSaved = useCallback(
    (entry: { id: string; assistantId: string; firstMessage: string }) =>
      setConversations((prev) => touchConversation(user.subject, prev, entry)),
    [user.subject],
  );
  const chat = useChatSession({ getToken: getAccessToken, onSessionExpired: onSignedOut, onTurnSaved });

  useEffect(() => {
    void (async () => {
      const token = await getAccessToken();
      if (!token) return onSignedOut();
      try {
        const list = await getAssistants(API_BASE, token);
        setAssistants(list);
        setAssistantId((current) =>
          list.length && !list.some((a) => a.assistantId === current) ? list[0].assistantId : current,
        );
      } catch (err) {
        if (err instanceof ApiAuthError && err.status === 401) return onSignedOut();
        setLoadError(err instanceof Error ? err.message : "Could not reach the assistant service.");
      }
    })();
  }, [onSignedOut]);

  // Chats saved under an assistant the backend no longer offers (e.g. the
  // parked Grok configs) stay in localStorage but aren't listed: opening one
  // would send new messages to an assistant that doesn't exist.
  const visibleConversations = useMemo(
    () => conversations.filter((c) => assistants.some((a) => a.assistantId === c.assistantId)),
    [conversations, assistants],
  );

  const assistantName = assistants.find((a) => a.assistantId === assistantId)?.displayName ?? assistantId;

  const selectAssistant = (id: string) => {
    setAssistantId(id);
    chat.reset();
  };
  const openConversation = (c: ConversationSummary) => {
    setAssistantId(c.assistantId);
    void chat.open(c.id);
  };
  const deleteConversation = (id: string) => {
    setConversations((prev) => removeConversation(user.subject, prev, id));
    if (id === chat.conversationId) chat.reset();
  };

  return {
    user,
    assistants,
    assistantId,
    setAssistantId,
    assistantName,
    conversations: visibleConversations,
    loadError,
    chat,
    selectAssistant,
    openConversation,
    deleteConversation,
  };
}

export type ChatController = ReturnType<typeof useChatController>;
