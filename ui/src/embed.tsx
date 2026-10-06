/**
 * Embeddable Bitwise Assist launcher — one script tag on the host page:
 *
 *   <script src=".../bitwise-assist.js"
 *     data-api-base="http://localhost:8000"
 *     data-keycloak-url="http://localhost:8080" data-realm="generic-ai-dev"
 *     data-client-id="generic-ai-api"
 *     data-default-assistant="hr_assistant"
 *     data-position="right" data-z-index="2147483000"></script>
 *
 * Every attribute is optional; defaults come from config.ts. The widget
 * renders inside a Shadow DOM root, so host CSS can't reach it and its CSS
 * can't leak out. No iframe: Keycloak's login page refuses to be framed
 * (X-Frame-Options: SAMEORIGIN), and sign-in is a full-page redirect that
 * returns to the same host page (see auth.ts).
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import appCss from "./app.css?inline";
import { completeSignInIfRedirected } from "./auth";
import { configure } from "./config";
import { Launcher } from "./widget/Launcher";
import widgetCss from "./widget/widget.css?inline";
import { clearWidgetConversation } from "./widget/widgetState";

const FONTS_HREF =
  "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap";

// document.currentScript is only set for classic scripts (the built bundle);
// the dev demo loads src/embed.tsx as a module and marks its tag instead.
const script =
  (document.currentScript as HTMLScriptElement | null) ??
  document.querySelector<HTMLScriptElement>("script[data-bitwise-assist]");
const data: DOMStringMap = script?.dataset ?? {};

function keycloakIssuer(): string | undefined {
  if (!data.keycloakUrl && !data.realm) return undefined;
  const base = (data.keycloakUrl || "http://localhost:8080").replace(/\/+$/, "");
  return `${base}/realms/${data.realm || "generic-ai-dev"}`;
}

configure({ apiBase: data.apiBase, keycloakIssuer: keycloakIssuer(), keycloakClientId: data.clientId });

/** @font-face rules don't register from inside a shadow root, so the web
 * fonts go into the host document's head (once). */
function loadFonts(): void {
  if (document.querySelector(`link[href="${FONTS_HREF}"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = FONTS_HREF;
  document.head.appendChild(link);
}

async function mount(): Promise<void> {
  let signInError: string | undefined;
  try {
    // A fresh sign-in (PKCE code exchanged just now) starts a new chat.
    if (await completeSignInIfRedirected()) clearWidgetConversation();
  } catch (err) {
    signInError = err instanceof Error ? err.message : "Sign-in failed.";
  }

  loadFonts();
  const host = document.createElement("div");
  host.id = "bitwise-assist";
  document.body.appendChild(host);
  const shadow = host.attachShadow({ mode: "open" });

  const style = document.createElement("style");
  // `all: initial` first: nothing inheritable from the host page gets in.
  // app.css's :root tokens become :host tokens inside the shadow tree.
  style.textContent = `:host { all: initial; }\n${appCss.replace(/:root/g, ":host")}\n${widgetCss}`;
  shadow.appendChild(style);

  const mountPoint = document.createElement("div");
  shadow.appendChild(mountPoint);
  const zIndex = Number(data.zIndex);
  createRoot(mountPoint).render(
    <StrictMode>
      <Launcher
        defaultAssistant={data.defaultAssistant || undefined}
        position={data.position === "left" ? "left" : "right"}
        zIndex={Number.isFinite(zIndex) && zIndex > 0 ? zIndex : undefined}
        signInError={signInError}
      />
    </StrictMode>,
  );
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => void mount(), { once: true });
} else {
  void mount();
}
