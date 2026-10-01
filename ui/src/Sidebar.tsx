import type { AssistantSummary } from "./api";
import type { SignedInUser } from "./auth";
import type { ConversationSummary } from "./history";
import { SparkIcon } from "./Message";

interface SidebarProps {
  open: boolean;
  user: SignedInUser;
  assistants: AssistantSummary[];
  assistantId: string;
  conversations: ConversationSummary[];
  activeConversationId?: string;
  onSelectAssistant: (id: string) => void;
  onNewChat: () => void;
  onOpenConversation: (c: ConversationSummary) => void;
  onDeleteConversation: (id: string) => void;
  onSignOut: () => void;
  onClose: () => void;
}

const DAY_MS = 86_400_000;

function groupByRecency(items: ConversationSummary[]): [string, ConversationSummary[]][] {
  const startOfToday = new Date().setHours(0, 0, 0, 0);
  const groups: Record<string, ConversationSummary[]> = { Today: [], "Previous 7 days": [], Older: [] };
  for (const c of items) {
    const key =
      c.updatedAt >= startOfToday ? "Today" : c.updatedAt >= startOfToday - 7 * DAY_MS ? "Previous 7 days" : "Older";
    groups[key].push(c);
  }
  return Object.entries(groups).filter(([, list]) => list.length > 0);
}

export function Sidebar(props: SidebarProps) {
  const { user, assistants, assistantId, conversations, activeConversationId } = props;
  const nameFor = (id: string) => assistants.find((a) => a.assistantId === id)?.displayName ?? id;

  return (
    <>
      <div className={`scrim ${props.open ? "is-open" : ""}`} onClick={props.onClose} aria-hidden />
      <aside className={`sidebar ${props.open ? "is-open" : ""}`} aria-label="Chats">
        <div className="brand">
          <span className="brand-mark"><SparkIcon /></span>
          <span className="brand-name">Bitwise Assist</span>
        </div>

        <button type="button" className="new-chat" onClick={props.onNewChat}>
          <span aria-hidden>+</span> New chat
        </button>

        <label className="field-label" htmlFor="assistant-select">Assistant</label>
        <select
          id="assistant-select"
          className="assistant-select"
          value={assistantId}
          onChange={(e) => props.onSelectAssistant(e.target.value)}
        >
          {assistants.map((a) => (
            <option key={a.assistantId} value={a.assistantId}>
              {a.displayName} ({a.assistantId})
            </option>
          ))}
        </select>

        <nav className="history">
          {conversations.length === 0 && <p className="history-empty">Your chats will appear here.</p>}
          {groupByRecency(conversations).map(([label, list]) => (
            <section key={label}>
              <h2 className="history-heading">{label}</h2>
              {list.map((c) => (
                <div key={c.id} className={`history-item ${c.id === activeConversationId ? "is-active" : ""}`}>
                  <button type="button" className="history-open" onClick={() => props.onOpenConversation(c)}>
                    <span className="history-title">{c.title}</span>
                    <span className="history-meta">{nameFor(c.assistantId)}</span>
                  </button>
                  <button
                    type="button"
                    className="history-delete"
                    aria-label={`Remove "${c.title}" from list`}
                    onClick={() => props.onDeleteConversation(c.id)}
                  >
                    ×
                  </button>
                </div>
              ))}
            </section>
          ))}
        </nav>

        <div className="user-card">
          <span className="user-avatar" aria-hidden>{user.name.slice(0, 1).toUpperCase()}</span>
          <span className="user-name">{user.name}</span>
          <button type="button" className="icon-btn" onClick={props.onSignOut}>Sign out</button>
        </div>
      </aside>
    </>
  );
}
