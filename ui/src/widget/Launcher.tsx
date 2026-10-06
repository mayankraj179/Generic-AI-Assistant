import { type ReactNode, type RefObject, useCallback, useEffect, useRef, useState } from "react";
import { clearSession, currentUser, type SignedInUser, signIn, signOut } from "../auth";
import { EmptyState, FullLayout } from "../ChatApp";
import { Composer } from "../Composer";
import { Message, SparkIcon } from "../Message";
import { type ChatController, PREFERRED_ASSISTANT, useChatController } from "../useChatController";
import { clearWidgetState, loadWidgetState, saveWidgetState, type WidgetView } from "./widgetState";

export interface LauncherProps {
  defaultAssistant?: string;
  position?: "right" | "left";
  zIndex?: number;
  signInError?: string;
}

/** Embeddable launcher: bubble → compact popup → maximized overlay.
 * Auth (auth.ts), the chat session and the API layer are the same ones the
 * standalone app uses; this only adds the shell around them. */
export function Launcher({ defaultAssistant = PREFERRED_ASSISTANT, position = "right", zIndex = 2147483000, signInError }: LauncherProps) {
  const [user, setUser] = useState<SignedInUser | null>(currentUser);
  const [view, setView] = useState<WidgetView>(() => loadWidgetState().view);
  const [error, setError] = useState(signInError);

  useEffect(() => saveWidgetState({ view }), [view]);

  // Escape steps down one level: maximized → popup → closed.
  useEffect(() => {
    if (view === "closed") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      setView((v) => (v === "maximized" ? "popup" : "closed"));
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [view]);

  // Refresh failed or the backend rejected the token: drop it, keep the
  // popup open on the sign-in card.
  const handleSessionEnded = useCallback(() => {
    clearSession();
    setError("Your session has ended. Please sign in again.");
    setUser(null);
    setView((v) => (v === "maximized" ? "popup" : v));
  }, []);

  const handleSignOut = useCallback(() => {
    clearWidgetState();
    signOut();
  }, []);

  const startSignIn = () => {
    // Persist "popup" first so it reopens on this page after the redirect.
    saveWidgetState({ view: "popup" });
    void signIn();
  };

  const isOpen = view !== "closed";
  return (
    <div className={`ba-root ba-pos-${position} ${isOpen ? "is-open" : ""}`} style={{ zIndex }}>
      {user ? (
        <SignedInChat
          user={user}
          view={view}
          setView={setView}
          defaultAssistant={defaultAssistant}
          onSessionEnded={handleSessionEnded}
          onSignOut={handleSignOut}
        />
      ) : (
        <Popup open={isOpen} title={<span className="ba-heading-name">Bitwise Assist</span>} onClose={() => setView("closed")}>
          <div className="ba-signin">
            <span className="brand-mark brand-mark-lg"><SparkIcon /></span>
            <h2>Sign in to Bitwise Assist</h2>
            <p>Answers from your organisation&apos;s policy documents, with sources.</p>
            {error && <p className="signin-error" role="alert">{error}</p>}
            <button type="button" className="primary-btn" onClick={startSignIn}>
              Sign in with Keycloak
            </button>
          </div>
        </Popup>
      )}

      {view !== "maximized" && (
        <button
          type="button"
          className="ba-launcher"
          aria-label={isOpen ? "Close Bitwise Assist" : "Open Bitwise Assist"}
          aria-expanded={isOpen}
          title={isOpen ? "Close Bitwise Assist" : "Open Bitwise Assist"}
          onClick={() => setView(isOpen ? "closed" : "popup")}
        >
          {isOpen ? <CloseIcon size={22} /> : <SparkIcon />}
        </button>
      )}
    </div>
  );
}

interface SignedInChatProps {
  user: SignedInUser;
  view: WidgetView;
  setView: (v: WidgetView) => void;
  defaultAssistant: string;
  onSessionEnded: () => void;
  onSignOut: () => void;
}

/** Owns the ONE controller (and so the one useChatSession) for the widget.
 * It stays mounted while the popup is closed or maximized, so switching
 * views never restarts the conversation or aborts a stream. */
function SignedInChat({ user, view, setView, defaultAssistant, onSessionEnded, onSignOut }: SignedInChatProps) {
  const controller = useChatController({ user, onSignedOut: onSessionEnded, defaultAssistant });
  const { chat, assistantId, setAssistantId } = controller;
  const restored = useRef(false);

  // Same login session, new page: reopen the conversation that was active.
  useEffect(() => {
    const saved = loadWidgetState();
    if (saved.conversationId) {
      if (saved.assistantId) setAssistantId(saved.assistantId);
      void chat.open(saved.conversationId);
    }
    restored.current = true;
    // Mount-only: restoring must not re-run when chat callbacks change.
  }, []);

  useEffect(() => {
    if (restored.current) saveWidgetState({ conversationId: chat.conversationId, assistantId });
  }, [chat.conversationId, assistantId]);

  return (
    <>
      <CompactChat controller={controller} open={view === "popup"} setView={setView} />
      {view === "maximized" && (
        <div className="ba-overlay">
          <div className="ba-backdrop" onClick={() => setView("popup")} aria-hidden />
          <div className="ba-max" role="dialog" aria-modal="true" aria-label="Bitwise Assist">
            <FullLayout
              controller={controller}
              onSignOut={onSignOut}
              headerActions={
                <>
                  <IconButton label="Restore to compact view" onClick={() => setView("popup")}><RestoreIcon /></IconButton>
                  <IconButton label="Close Bitwise Assist" onClick={() => setView("closed")}><CloseIcon /></IconButton>
                </>
              }
            />
          </div>
        </div>
      )}
    </>
  );
}

function CompactChat({ controller: c, open, setView }: { controller: ChatController; open: boolean; setView: (v: WidgetView) => void }) {
  const { chat, assistants, assistantId, assistantName } = c;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const panelRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chat.messages, open]);

  // Focus the composer whenever the popup opens (it's not focusable while hidden).
  useEffect(() => {
    if (!open) return;
    const id = window.setTimeout(() => panelRef.current?.querySelector("textarea")?.focus(), 30);
    return () => window.clearTimeout(id);
  }, [open]);

  const disabled = chat.isSending || chat.isLoading || assistants.length === 0;
  return (
    <Popup
      open={open}
      panelRef={panelRef}
      eyebrow="Bitwise Assist"
      title={
        // The assistant name doubles as the switcher when there's a choice.
        assistants.length > 1 ? (
          <select
            className="ba-assistant-select"
            aria-label="Assistant"
            value={assistantId}
            onChange={(e) => c.selectAssistant(e.target.value)}
          >
            {assistants.map((a) => (
              <option key={a.assistantId} value={a.assistantId}>{a.displayName}</option>
            ))}
          </select>
        ) : (
          <span className="ba-heading-name">{assistantName}</span>
        )
      }
      actions={
        <>
          <IconButton label="New chat" onClick={chat.reset}><NewChatIcon /></IconButton>
          <IconButton label="Maximize" onClick={() => setView("maximized")}><MaximizeIcon /></IconButton>
        </>
      }
      onClose={() => setView("closed")}
    >
      <div className="ba-scroll" ref={scrollRef}>
        <div className="ba-thread">
          {c.loadError && <div className="error-card" role="alert">{c.loadError}</div>}
          {chat.isLoading && <div className="loading">Loading chat…</div>}
          {!chat.isLoading && chat.messages.length === 0 && (
            <EmptyState
              compact
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
      <div className="ba-composer">
        <Composer disabled={disabled} placeholder={`Message ${assistantName}…`} onSend={(text) => void chat.send(assistantId, text)} />
        <p className="disclaimer">Answers come only from documents you&apos;re authorised to see.</p>
      </div>
    </Popup>
  );
}

interface PopupProps {
  open: boolean;
  /** Small label above the title. */
  eyebrow?: string;
  title: ReactNode;
  actions?: ReactNode;
  onClose: () => void;
  panelRef?: RefObject<HTMLElement>;
  children: ReactNode;
}

function Popup({ open, eyebrow, title, actions, onClose, panelRef, children }: PopupProps) {
  return (
    <section
      ref={panelRef}
      className={`ba-panel ${open ? "is-open" : ""}`}
      role="dialog"
      aria-label="Bitwise Assist chat"
      aria-hidden={!open}
    >
      <header className="ba-header">
        <span className="brand-mark"><SparkIcon /></span>
        <div className="ba-heading">
          {eyebrow && <span className="ba-eyebrow">{eyebrow}</span>}
          <div className="ba-heading-title">
            <span className="status-dot" aria-hidden />
            {title}
          </div>
        </div>
        <div className="ba-header-actions">
          {actions}
          <IconButton label="Close Bitwise Assist" onClick={onClose}><CloseIcon /></IconButton>
        </div>
      </header>
      {children}
    </section>
  );
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" className="ba-icon-btn" aria-label={label} title={label} onClick={onClick}>
      {children}
    </button>
  );
}

const iconProps = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

function CloseIcon({ size = 18 }: { size?: number }) {
  return <svg {...iconProps} width={size} height={size}><path d="M18 6 6 18M6 6l12 12" /></svg>;
}
function MaximizeIcon() {
  return <svg {...iconProps} width={17} height={17}><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" /></svg>;
}
function RestoreIcon() {
  return <svg {...iconProps} width={17} height={17}><path d="M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7" /></svg>;
}
function NewChatIcon() {
  return <svg {...iconProps} width={17} height={17}><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></svg>;
}
