/** Endpoints come from ui/.env (see .env.example); the defaults are the
 * backend's local-development values (backend/.env.example, infra/keycloak). */
const trimSlash = (url: string) => url.replace(/\/+$/, "");

// `let` so the embeddable launcher can apply its <script data-*> overrides
// (configure() below) before anything reads them; ES module bindings are
// live, so every importer sees the override.
export let API_BASE = trimSlash(import.meta.env.VITE_API_BASE || "http://localhost:8000");

export let KEYCLOAK_ISSUER = trimSlash(
  import.meta.env.VITE_KEYCLOAK_ISSUER || "http://localhost:8080/realms/generic-ai-dev",
);
export let KEYCLOAK_CLIENT_ID = import.meta.env.VITE_KEYCLOAK_CLIENT_ID || "generic-ai-api";

export interface ConfigOverrides {
  apiBase?: string;
  keycloakIssuer?: string;
  keycloakClientId?: string;
}

/** Called once by the embed entry, before rendering. Unset fields keep the defaults above. */
export function configure(overrides: ConfigOverrides): void {
  if (overrides.apiBase) API_BASE = trimSlash(overrides.apiBase);
  if (overrides.keycloakIssuer) KEYCLOAK_ISSUER = trimSlash(overrides.keycloakIssuer);
  if (overrides.keycloakClientId) KEYCLOAK_CLIENT_ID = overrides.keycloakClientId;
}

/** Starter prompts shown on an empty chat, per assistant. Anything not
 * listed falls back to GENERIC_SUGGESTIONS. */
export const SUGGESTIONS: Record<string, string[]> = {
  hr_assistant: [
    "How many casual leaves do I get per year?",
    "Is Diwali a floating holiday in 2026?",
    "Can I carry forward unused earned leave?",
    "How long is paternity leave?",
  ],
  finance_assistant: [
    "What was total revenue in FY2025?",
    "Plot quarterly revenue as a bar chart",
    "What's today's date?",
    "Calculate 18% of 245000",
  ],
};

export const GENERIC_SUGGESTIONS = [
  "What can you help me with?",
  "Summarize the key points of the documents you can see",
];
