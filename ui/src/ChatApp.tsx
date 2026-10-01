import { useCallback, useEffect, useRef, useState } from "react";
import { type AssistantSummary, getAssistants } from "./api";
import { currentUser, getAccessToken, type SignedInUser, signIn, signOut } from "./auth";
import { Composer } from "./Composer";
import { API_BASE, GENERIC_SUGGESTIONS, SUGGESTIONS } from "./config";
import { type ConversationSummary, loadConversations, removeConversation, touchConversation } from "./history";
import { Message, SparkIcon } from "./Message";
import { Sidebar } from "./Sidebar";
import { useChatSession } from "./useChatSession";

const PREFERRED_ASSISTANT = "hr_assistant";
// Same exclusion list as frontend/login.html's FLAT_HIDDEN_IDS: Grok configs
// (xAI rejects the API key) and hr_assistant_azure (duplicates hr_assistant).
const HIDDEN_ASSISTANT_IDS = ["hr_assistant_azure", "hr_assistant_grok", "finance_assistant_grok"];

export function SignInScreen({ error }: { error?: string }) {
  return (
    <main className="signin">
      <div className="signin-card">
        <span className="brand-mark brand-mark-lg"><SparkIcon /></span>
        <h1>Bitwise Assist</h1>
        <p>Answers from your organisation&apos;s policy documents, with sources.</p>
        {error && <p className="signin-error" role="alert">{error}</p>}
        <button type="button" className="primary-btn" onClick={() => void signIn()}>
          Sign in with Keycloak
        </button>
        {import.meta.env.DEV && (
          <p className="signin-hint">Local dev user: <code>dev_user</code> / <code>dev_password123</code></p>
        )}
      </div>
    </main>
  );
}

interface EmptyStateProps {
  name: string;
  assistantId: string;
  disabled: boolean;
  onPick: (s: string) => void;
}

function EmptyState({ name, assistantId, disabled, onPick }: EmptyStateProps) {
  return (
    <div className="empty">
      <span className="brand-mark brand-mark-lg"><SparkIcon /></span>
      <h1 className="empty-title">How can I help today?</h1>
      <p className="empty-sub">You&apos;re chatting with <strong>{name}</strong>.</p>
      <div className="suggestions">
        {(SUGGESTIONS[assistantId] ?? GENERIC_SUGGESTIONS).map((s) => (
          <button type="button" key={s} className="suggestion" disabled={disabled} onClick={() => onPick(s)}>
            {s}
            <span aria-hidden className="suggestion-arrow">→</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function ChatApp({ user, onSignedOut }: { user: SignedInUser; onSignedOut: () => void }) {
  const [assistants, setAssistants] = useState<AssistantSummary[]>([]);
  const [assistantId, setAssistantId] = useState(PREFERRED_ASSISTANT);
  const [conversations, setConversations] = useState<ConversationSummary[]>(() => loadConversations(user.subject));
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loadError, setLoadError] = useState<string>();
  const scrollRef = useRef<HTMLDivElement | null>(null);

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
        const list = (await getAssistants(API_BASE, token)).filter(
          (a) => !HIDDEN_ASSISTANT_IDS.includes(a.assistantId),
        );
        setAssistants(list);
        if (list.length && !list.some((a) => a.assistantId === PREFERRED_ASSISTANT)) {
          setAssistantId(list[0].assistantId);
        }
      } catch (err) {
        setLoadError(err instanceof Error ? err.message : "Could not reach the assistant service.");
      }
    })();
  }, [onSignedOut]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chat.messages]);

  const assistantName = assistants.find((a) => a.assistantId === assistantId)?.displayName ?? assistantId;
  const startNewChat = () => {
    chat.reset();
    setSidebarOpen(false);
  };

  return (
    <div className="app">
      <Sidebar
        open={sidebarOpen}
        user={user}
        assistants={assistants}
        assistantId={assistantId}
        conversations={conversations}
        activeConversationId={chat.conversationId}
        onSelectAssistant={(id) => {
          setAssistantId(id);
          startNewChat();
        }}
        onNewChat={startNewChat}
        onOpenConversation={(c) => {
          setAssistantId(c.assistantId);
          setSidebarOpen(false);
          void chat.open(c.id);
        }}
        onDeleteConversation={(id) => {
          setConversations((prev) => removeConversation(user.subject, prev, id));
          if (id === chat.conversationId) chat.reset();
        }}
        onSignOut={signOut}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="main">
        <header className="topbar">
          <button type="button" className="menu-btn" aria-label="Open chats" onClick={() => setSidebarOpen(true)}>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <div className="topbar-title">
            <span className="status-dot" aria-hidden />
            {assistantName}
          </div>
        </header>

        <div className="scroll" ref={scrollRef}>
          <div className="thread">
            {loadError && <div className="error-card" role="alert">{loadError}</div>}
            {chat.isLoading && <div className="loading">Loading chat…</div>}
            {!chat.isLoading && chat.messages.length === 0 && (
              <EmptyState
                name={assistantName}
                assistantId={assistantId}
                disabled={chat.isSending || assistants.length === 0}
                onPick={(s) => void chat.send(assistantId, s)}
              />
            )}
            {chat.messages.map((m) => (
              <Message key={m.id} message={m} assistantName={assistantName} />
            ))}
          </div>
        </div>

        <div className="composer-wrap">
          <Composer
            disabled={chat.isSending || chat.isLoading || assistants.length === 0}
            placeholder={`Message ${assistantName}…`}
            onSend={(text) => void chat.send(assistantId, text)}
          />
          <p className="disclaimer">Answers come only from documents you&apos;re authorised to see. Check important details with HR.</p>
        </div>
      </main>
    </div>
  );
}

export function Root({ signInError }: { signInError?: string }) {
  const [user, setUser] = useState<SignedInUser | null>(currentUser);
  const handleSignedOut = useCallback(() => setUser(null), []);
  return user ? <ChatApp user={user} onSignedOut={handleSignedOut} /> : <SignInScreen error={signInError} />;
}
