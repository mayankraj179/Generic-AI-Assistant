import { useState } from "react";
import type { Chart, Citation } from "./api";
import { ChartView } from "./ChartView";
import { Markdown } from "./Markdown";

export interface DisplayMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  citations?: Citation[];
  chart?: Chart;
  grounded?: boolean;
  streaming?: boolean;
}

/** One chip per source document, not per chunk — eight chunks of the same
 * policy file read as noise. Document name only; chunk detail is internal. */
function Sources({ citations }: { citations: Citation[] }) {
  return (
    <div className="sources">
      <span className="sources-label">Sources</span>
      {[...new Set(citations.map((c) => c.documentTitle))].map((title) => (
        <span className="source-chip" key={title}>
          <DocIcon />
          {title.replace(/_/g, " ")}
        </span>
      ))}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="icon-btn"
      aria-label="Copy reply"
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export function Message({ message, assistantName }: { message: DisplayMessage; assistantName: string }) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble-user">{message.content}</div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="msg msg-assistant">
        <div className="avatar avatar-error" aria-hidden>!</div>
        <div className="msg-body">
          <div className="error-card" role="alert">{message.content}</div>
        </div>
      </div>
    );
  }

  const isThinking = message.streaming && message.content === "";
  return (
    <div className="msg msg-assistant">
      <div className="avatar" aria-hidden>
        <SparkIcon />
      </div>
      <div className="msg-body">
        <div className="msg-name">{assistantName}</div>
        {isThinking ? (
          <div className="typing" aria-label="Assistant is typing">
            <span /><span /><span />
          </div>
        ) : (
          <div className={`prose ${message.streaming ? "is-streaming" : ""}`}>
            <Markdown text={message.content} />
          </div>
        )}
        {message.chart && (
          <div className="chart-card">
            <ChartView chart={message.chart} />
          </div>
        )}
        {!message.streaming && (
          <div className="msg-footer">
            {message.citations && message.citations.length > 0 && (
              <Sources citations={message.citations} />
            )}
            {/* grounded=false also covers correct tool-only answers (e.g. a
                calculation with nothing retrieved), so don't claim "not found". */}
            {message.grounded === false && (
              <span className="badge-ungrounded">No document sources</span>
            )}
            <CopyButton text={message.content} />
          </div>
        )}
      </div>
    </div>
  );
}

export function SparkIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden>
      <path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9z" />
      <path d="M19 15l.9 2.6 2.6.9-2.6.9L19 22l-.9-2.6-2.6-.9 2.6-.9z" opacity=".6" />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
    </svg>
  );
}
