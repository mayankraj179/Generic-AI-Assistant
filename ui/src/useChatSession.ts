import { useCallback, useRef, useState } from "react";
import { ApiAuthError, getMessages, postChatStream } from "./api";
import { API_BASE } from "./config";
import type { DisplayMessage } from "./Message";

const newId = () => crypto.randomUUID();

interface ChatSessionOptions {
  getToken: () => Promise<string | null>;
  onSessionExpired: () => void;
  onTurnSaved: (entry: { id: string; assistantId: string; firstMessage: string }) => void;
}

/** Messages + streaming for one conversation at a time. The first reply's
 * `done` event is what assigns a conversation id (the backend creates the
 * conversation on the first turn). */
export function useChatSession({ getToken, onSessionExpired, onTurnSaved }: ChatSessionOptions) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [isSending, setIsSending] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  // Bumped on every reset/open so a stream still finishing for a chat the
  // user already navigated away from can't write into the new one.
  const generation = useRef(0);
  // Synchronous in-flight guard: a double-click lands before React commits
  // isSending, and two sends from a new chat would create two conversations.
  const inFlight = useRef(false);

  const patch = (id: string, update: (m: DisplayMessage) => DisplayMessage) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));

  const reset = useCallback(() => {
    generation.current++;
    inFlight.current = false;
    setMessages([]);
    setConversationId(undefined);
    setIsSending(false);
  }, []);

  const open = useCallback(
    async (id: string) => {
      const gen = ++generation.current;
      inFlight.current = false;
      setIsSending(false);
      setIsLoading(true);
      setConversationId(id);
      setMessages([]);
      try {
        const token = await getToken();
        if (!token) {
          setIsLoading(false);
          return onSessionExpired();
        }
        const history = await getMessages(API_BASE, token, id);
        if (gen !== generation.current) return;
        setMessages(
          history.map((m) => ({ id: newId(), role: m.role, content: m.content, citations: m.citations })),
        );
      } catch (err) {
        if (gen !== generation.current) return;
        if (err instanceof ApiAuthError && err.status === 401) {
          setIsLoading(false);
          return onSessionExpired();
        }
        const detail = err instanceof Error ? err.message : "Could not load this chat.";
        setMessages([{ id: newId(), role: "error", content: detail }]);
      } finally {
        if (gen === generation.current) setIsLoading(false);
      }
    },
    [getToken, onSessionExpired],
  );

  const send = useCallback(
    async (assistantId: string, text: string) => {
      if (inFlight.current) return;
      inFlight.current = true;
      const gen = generation.current;
      const replyId = newId();
      setMessages((prev) => [
        ...prev,
        { id: newId(), role: "user", content: text },
        { id: replyId, role: "assistant", content: "", streaming: true },
      ]);
      setIsSending(true);

      const stale = () => gen !== generation.current;
      const finish = () => {
        if (stale()) return;
        inFlight.current = false;
        setIsSending(false);
      };

      const token = await getToken();
      if (!token) {
        finish();
        return onSessionExpired();
      }

      await postChatStream(API_BASE, token, {
        assistantId,
        message: text,
        conversationId,
        onDelta: (delta) => !stale() && patch(replyId, (m) => ({ ...m, content: m.content + delta })),
        onChart: (chart) => !stale() && patch(replyId, (m) => ({ ...m, chart })),
        onDone: (result) => {
          if (stale()) return;
          setConversationId(result.conversationId);
          patch(replyId, (m) => ({ ...m, streaming: false, citations: result.citations, grounded: result.grounded }));
          onTurnSaved({ id: result.conversationId, assistantId, firstMessage: text });
        },
        onError: (detail) => {
          if (stale()) return;
          setMessages((prev) => [
            ...prev.filter((m) => m.id !== replyId),
            { id: newId(), role: "error", content: detail },
          ]);
        },
        onUnauthorized: () => !stale() && onSessionExpired(),
      });
      finish(); // postChatStream never throws — every failure goes through onError
    },
    [conversationId, getToken, onSessionExpired, onTurnSaved],
  );

  return { messages, conversationId, isSending, isLoading, send, open, reset };
}
