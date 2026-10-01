/** The backend has no "list my conversations" endpoint, so the sidebar's
 * list is remembered per browser (and per signed-in user). Message content
 * itself is never stored here — opening a chat re-fetches it from
 * GET /sessions/{id}/messages, which enforces ownership server-side. */

export interface ConversationSummary {
  id: string;
  assistantId: string;
  title: string;
  updatedAt: number;
}

const MAX_CONVERSATIONS = 50;
const TITLE_LENGTH = 60;

const storageKey = (userSubject: string) => `gaaf.conversations.${userSubject}`;

export function loadConversations(userSubject: string): ConversationSummary[] {
  try {
    const raw = localStorage.getItem(storageKey(userSubject));
    return raw ? (JSON.parse(raw) as ConversationSummary[]) : [];
  } catch {
    return [];
  }
}

function persist(userSubject: string, conversations: ConversationSummary[]): void {
  try {
    localStorage.setItem(storageKey(userSubject), JSON.stringify(conversations));
  } catch {
    // Storage unavailable (private mode, quota) — the list just won't survive a reload.
  }
}

/** Returns the new list with `id` moved to the top (added if new). */
export function touchConversation(
  userSubject: string,
  conversations: ConversationSummary[],
  entry: { id: string; assistantId: string; firstMessage: string },
): ConversationSummary[] {
  const existing = conversations.find((c) => c.id === entry.id);
  const title =
    existing?.title ??
    (entry.firstMessage.length > TITLE_LENGTH
      ? `${entry.firstMessage.slice(0, TITLE_LENGTH).trimEnd()}…`
      : entry.firstMessage);
  const next = [
    { id: entry.id, assistantId: entry.assistantId, title, updatedAt: Date.now() },
    ...conversations.filter((c) => c.id !== entry.id),
  ].slice(0, MAX_CONVERSATIONS);
  persist(userSubject, next);
  return next;
}

export function removeConversation(
  userSubject: string,
  conversations: ConversationSummary[],
  id: string,
): ConversationSummary[] {
  const next = conversations.filter((c) => c.id !== id);
  persist(userSubject, next);
  return next;
}
