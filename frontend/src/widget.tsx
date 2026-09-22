import { createRoot, type Root } from "react-dom/client";
import { AssistantWidgetApp } from "./AssistantWidgetApp";

/**
 * <assistant-widget
 *   assistant-id="hr_assistant"
 *   api-base="http://localhost:8000"
 *   auth-token="..."
 * ></assistant-widget>
 *
 * Shadow DOM keeps host-page CSS from leaking in and widget CSS from
 * leaking out. Attributes are read once, at connect time — there is no
 * reactive attribute-watching, so changing an attribute on an already-
 * mounted element has no effect. Re-create the element to pick up new
 * values.
 */
class AssistantWidgetElement extends HTMLElement {
  #root: Root | null = null;

  connectedCallback(): void {
    const shadow = this.attachShadow({ mode: "open" });
    const mountPoint = document.createElement("div");
    shadow.appendChild(mountPoint);

    const assistantId = this.getAttribute("assistant-id") ?? "";
    const apiBase = this.getAttribute("api-base") ?? "";
    const authToken = this.getAttribute("auth-token") ?? "";

    this.#root = createRoot(mountPoint);
    this.#root.render(
      <AssistantWidgetApp assistantId={assistantId} apiBase={apiBase} authToken={authToken} />,
    );
  }

  disconnectedCallback(): void {
    this.#root?.unmount();
    this.#root = null;
  }
}

customElements.define("assistant-widget", AssistantWidgetElement);
