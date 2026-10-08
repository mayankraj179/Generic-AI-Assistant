import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { clearSession, currentUser, type SignedInUser, signIn, signOut } from "./auth";
import { Composer } from "./Composer";
import { GENERIC_SUGGESTIONS, SUGGESTIONS } from "./config";
import { Message, SparkIcon } from "./Message";
import { Sidebar } from "./Sidebar";
import { type ChatController, useChatController } from "./useChatController";

/** `onSignIn` lets the launcher record its own state before the redirect;
 * the standalone app just starts the redirect. */
export function SignInScreen({ error, onSignIn = () => void signIn() }: { error?: string; onSignIn?: () => void }) {
  return (
    <main className="signin">
      <div className="signin-card">
        <span className="brand-mark brand-mark-lg"><SparkIcon /></span>
        <h1>Bitwise Assist</h1>
        <p>Answers from your organisation&apos;s policy documents, with sources.</p>
        {error && <p className="signin-error" role="alert">{error}</p>}
        <button type="button" className="primary-btn" onClick={onSignIn}>
          Sign in with Keycloak
        </button>
      </div>
    </main>
  );
}

interface EmptyStateProps {
  name: string;
  assistantId: string;
  disabled: boolean;
  onPick: (s: string) => void;
  /** Popup variant: two stacked suggestions instead of the 2x2 grid. */
  compact?: boolean;
}

export function EmptyState({ name, assistantId, disabled, onPick, compact = false }: EmptyStateProps) {
  const suggestions = SUGGESTIONS[assistantId] ?? GENERIC_SUGGESTIONS;
  return (
    <div className={`empty ${compact ? "empty-compact" : ""}`}>
      <span className="brand-mark brand-mark-lg"><SparkIcon /></span>
      <h1 className="empty-title">How can I help today?</h1>
      <p className="empty-sub">You&apos;re chatting with <strong>{name}</strong>.</p>
      <div className="suggestions">
        {(compact ? suggestions.slice(0, 2) : suggestions).map((s) => (
          <button type="button" key={s} className="suggestion" disabled={disabled} onClick={() => onPick(s)}>
            {s}
            <span aria-hidden className="suggestion-arrow">→</span>
          </button>
        ))}
      </div>
    </div>
  );
}

interface FullLayoutProps {
  controller: ChatController;
  onSignOut: () => void;
  /** Extra buttons at the right of the top bar (the launcher's Restore/Close). */
  headerActions?: ReactNode;
}

/** The full Bitwise Assist layout: sidebar, top bar, thread, composer. Used
 * by the standalone app and by the launcher's maximized view. */
export function FullLayout({ controller: c, onSignOut, headerActions }: FullLayoutProps) {
  const { chat, assistants, assistantId, assistantName } = c;
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Full-screen chat: hides the sidebar so the conversation gets the whole width.
  const [fullScreen, setFullScreen] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chat.messages]);

  const startNewChat = () => {
    chat.reset();
    setSidebarOpen(false);
  };

  return (
    <div className={`app ${fullScreen ? "is-fullscreen" : ""}`}>
      <Sidebar
        open={sidebarOpen}
        user={c.user}
        assistants={assistants}
        assistantId={assistantId}
        conversations={c.conversations}
        activeConversationId={chat.conversationId}
        onSelectAssistant={(id) => {
          c.selectAssistant(id);
          setSidebarOpen(false);
        }}
        onNewChat={startNewChat}
        onOpenConversation={(conv) => {
          setSidebarOpen(false);
          c.openConversation(conv);
        }}
        onDeleteConversation={c.deleteConversation}
        onSignOut={onSignOut}
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
          <div className="topbar-actions">
            <button
              type="button"
              className="topbar-btn"
              aria-pressed={fullScreen}
              aria-label={fullScreen ? "Exit full screen" : "Expand chat to full screen"}
              title={fullScreen ? "Exit full screen" : "Expand chat to full screen"}
              onClick={() => {
                setFullScreen((v) => !v);
                setSidebarOpen(false);
              }}
            >
              {fullScreen ? <CollapseArrowsIcon /> : <ExpandArrowsIcon />}
            </button>
            {headerActions}
          </div>
        </header>

        <div className="scroll" ref={scrollRef}>
          <div className="thread">
            {c.loadError && <div className="error-card" role="alert">{c.loadError}</div>}
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

const arrowIconProps = {
  viewBox: "0 0 24 24",
  width: 18,
  height: 18,
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

/** Four arrows pointing out to the corners: "expand to full screen". */
function ExpandArrowsIcon() {
  return (
    <svg {...arrowIconProps}>
      <path d="M15 3h6v6M21 3l-7 7M9 21H3v-6M3 21l7-7M21 15v6h-6M21 21l-7-7M3 9V3h6M3 3l7 7" />
    </svg>
  );
}

/** Four arrows pointing in from the corners: "back to normal view". */
function CollapseArrowsIcon() {
  return (
    <svg {...arrowIconProps}>
      <path d="M20 10h-6V4M14 10l7-7M4 14h6v6M10 14l-7 7M14 20v-6h6M14 14l7 7M10 4v6H4M10 10 3 3" />
    </svg>
  );
}

export function ChatApp({ user, onSignedOut }: { user: SignedInUser; onSignedOut: () => void }) {
  const controller = useChatController({ user, onSignedOut });
  return <FullLayout controller={controller} onSignOut={signOut} />;
}

export function Root({ signInError }: { signInError?: string }) {
  const [user, setUser] = useState<SignedInUser | null>(currentUser);
  const [error, setError] = useState(signInError);
  // Reached when the session can't be refreshed or the backend rejects the
  // token: drop it locally so it is never sent again, then ask to sign in.
  const handleSignedOut = useCallback(() => {
    clearSession();
    setError("Your session has ended. Please sign in again.");
    setUser(null);
  }, []);
  return user ? <ChatApp user={user} onSignedOut={handleSignedOut} /> : <SignInScreen error={error} />;
}
