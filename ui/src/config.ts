/** Endpoints come from ui/.env (see .env.example); the defaults are the
 * backend's local-development values (backend/.env.example, infra/keycloak). */
const trimSlash = (url: string) => url.replace(/\/+$/, "");

export const API_BASE = trimSlash(import.meta.env.VITE_API_BASE || "http://localhost:8000");

export const KEYCLOAK_ISSUER = trimSlash(
  import.meta.env.VITE_KEYCLOAK_ISSUER || "http://localhost:8080/realms/generic-ai-dev",
);
export const KEYCLOAK_CLIENT_ID = import.meta.env.VITE_KEYCLOAK_CLIENT_ID || "generic-ai-api";

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
