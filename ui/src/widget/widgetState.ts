/** Launcher state that survives page navigation within one login session
 * (same tab). sessionStorage, like the tokens: gone when the tab closes.
 * Holds ids only — message content is re-fetched from
 * GET /sessions/{id}/messages, which enforces ownership server-side. */

export type WidgetView = "closed" | "popup" | "maximized";

export interface WidgetState {
  view: WidgetView;
  conversationId?: string;
  assistantId?: string;
}

const KEY = "gaaf.widget";

export function loadWidgetState(): WidgetState {
  try {
    const raw = sessionStorage.getItem(KEY);
    const parsed = raw ? (JSON.parse(raw) as Partial<WidgetState>) : {};
    const view = parsed.view === "popup" || parsed.view === "maximized" ? parsed.view : "closed";
    return { ...parsed, view };
  } catch {
    return { view: "closed" };
  }
}

export function saveWidgetState(patch: Partial<WidgetState>): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ ...loadWidgetState(), ...patch }));
  } catch {
    // Storage unavailable — the widget just won't survive navigation.
  }
}

/** Fresh sign-in or sign-out: the next chat starts empty. Keeps `view`, so
 * a popup the user opened to sign in reopens after the redirect. */
export function clearWidgetConversation(): void {
  saveWidgetState({ conversationId: undefined, assistantId: undefined });
}

export function clearWidgetState(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    // Nothing to clear.
  }
}
