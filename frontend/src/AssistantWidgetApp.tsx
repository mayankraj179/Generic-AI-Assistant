import { useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { type Chart, type Citation, postChatStream } from "./api";

interface AssistantWidgetAppProps {
  assistantId: string;
  apiBase: string;
  authToken: string;
}

type MessageRole = "user" | "assistant" | "error";

interface DisplayMessage {
  id: string;
  role: MessageRole;
  content: string;
  citations?: Citation[];
  chart?: Chart;
  /** True only in the (rare, chart-only) gap after text streaming has
   * finished but neither the chart nor done event has arrived yet — see
   * armFinalizingIndicator below. Never true on an ordinary, chart-less
   * turn, since onDone normally arrives well inside the debounce window. */
  finalizing?: boolean;
}

function makeId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const CHART_COLORS = [
  "#4f46e5",
  "#22c55e",
  "#f59e0b",
  "#ef4444",
  "#06b6d4",
  "#a855f7",
  "#ec4899",
  "#84cc16",
];

function buildChartRows(chart: Chart): Record<string, string | number>[] {
  return chart.labels.map((label, index) => {
    const row: Record<string, string | number> = { name: label };
    for (const series of chart.series) {
      row[series.name] = series.values[index] ?? 0;
    }
    return row;
  });
}

function ChartView({ chart }: { chart: Chart }) {
  const data = buildChartRows(chart);
  const showLegend = chart.series.length > 1;

  return (
    <div className="aw-chart">
      <div className="aw-chart-title">{chart.title}</div>
      <ResponsiveContainer width="100%" height={220}>
        {chart.chartType === "bar" ? (
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            {showLegend && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {chart.series.map((series, index) => (
              <Bar
                key={series.name}
                dataKey={series.name}
                fill={CHART_COLORS[index % CHART_COLORS.length]}
              />
            ))}
          </BarChart>
        ) : chart.chartType === "line" ? (
          <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            {showLegend && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {chart.series.map((series, index) => (
              <Line
                key={series.name}
                type="monotone"
                dataKey={series.name}
                stroke={CHART_COLORS[index % CHART_COLORS.length]}
              />
            ))}
          </LineChart>
        ) : (
          <PieChart margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Pie
              data={data}
              dataKey={chart.series[0]?.name ?? "value"}
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label
            >
              {data.map((_, index) => (
                <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        )}
      </ResponsiveContainer>
      {chart.sourceChunks.length > 0 && (
        <div className="aw-citations aw-chart-citations">
          {chart.sourceChunks.map((citation, index) => (
            <div key={index}>
              Chart source: {citation.documentTitle} (chunk {citation.chunkIndex})
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ message }: { message: DisplayMessage }) {
  const isUser = message.role === "user";
  const isError = message.role === "error";

  return (
    <div className={`aw-row ${isUser ? "aw-row-user" : "aw-row-other"}`}>
      <div className={`aw-bubble ${isUser ? "aw-bubble-user" : isError ? "aw-bubble-error" : "aw-bubble-assistant"}`}>
        <div>{message.content}</div>
        {message.citations && message.citations.length > 0 && (
          <div className="aw-citations">
            {message.citations.map((citation, index) => (
              <div key={index}>
                Source: {citation.documentTitle} (chunk {citation.chunkIndex})
              </div>
            ))}
          </div>
        )}
        {message.chart && <ChartView chart={message.chart} />}
        {message.finalizing && !message.chart && (
          <div className="aw-chart-pending">Generating chart…</div>
        )}
      </div>
    </div>
  );
}

export function AssistantWidgetApp({ assistantId, apiBase, authToken }: AssistantWidgetAppProps) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const finalizeTimerRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  useEffect(() => {
    return () => {
      if (finalizeTimerRef.current !== undefined) {
        window.clearTimeout(finalizeTimerRef.current);
      }
    };
  }, []);

  const isMisconfigured = !apiBase || !authToken;

  async function handleSend(): Promise<void> {
    const text = input.trim();
    if (!text || isSending || isMisconfigured) {
      return;
    }

    const assistantMessageId = makeId();
    setMessages((prev) => [
      ...prev,
      { id: makeId(), role: "user", content: text },
      { id: assistantMessageId, role: "assistant", content: "" },
    ]);
    setInput("");
    setIsSending(true);

    // Chart generation (when it happens at all) is a second, server-side
    // model call made only after text streaming finishes, so the client has
    // no upfront signal that a chart is coming. Instead of guessing, this
    // arms/re-arms a short debounce on every delta: as long as new text
    // keeps arriving, the timer never fires. Once deltas stop and this much
    // time passes without onChart/onDone/onError, the reply is very likely
    // just waiting on the chart call, so the "Generating chart…" indicator
    // appears. An ordinary, chart-less turn's onDone arrives well inside
    // this window, so the indicator never flashes on those turns.
    const FINALIZE_DEBOUNCE_MS = 500;

    function clearFinalizeTimer(): void {
      if (finalizeTimerRef.current !== undefined) {
        window.clearTimeout(finalizeTimerRef.current);
        finalizeTimerRef.current = undefined;
      }
    }

    function armFinalizeTimer(): void {
      clearFinalizeTimer();
      finalizeTimerRef.current = window.setTimeout(() => {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantMessageId ? { ...message, finalizing: true } : message,
          ),
        );
      }, FINALIZE_DEBOUNCE_MS);
    }

    await postChatStream(apiBase, authToken, {
      assistantId,
      message: text,
      conversationId,
      onDelta: (delta) => {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantMessageId
              ? { ...message, content: message.content + delta, finalizing: false }
              : message,
          ),
        );
        armFinalizeTimer();
      },
      onChart: (chart) => {
        clearFinalizeTimer();
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantMessageId ? { ...message, chart, finalizing: false } : message,
          ),
        );
      },
      onDone: (result) => {
        clearFinalizeTimer();
        setConversationId(result.conversationId);
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantMessageId
              ? { ...message, citations: result.citations, finalizing: false }
              : message,
          ),
        );
      },
      onError: (detail) => {
        clearFinalizeTimer();
        // The backend never persists a partial reply on failure, so the
        // widget doesn't show one either — drop the (possibly partially
        // filled) assistant bubble and show a single error bubble instead,
        // matching the pre-streaming error-bubble behavior.
        setMessages((prev) => [
          ...prev.filter((message) => message.id !== assistantMessageId),
          { id: makeId(), role: "error", content: detail },
        ]);
      },
    });

    setIsSending(false);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  }

  return (
    <div className="aw-shell">
      <style>{STYLES}</style>
      <header className="aw-header">
        <div className="aw-dot" />
        <strong>{assistantId || "(no assistant-id set)"}</strong>
      </header>

      {isMisconfigured ? (
        <div className="aw-config-error">
          This widget is missing required configuration: both{" "}
          <code>api-base</code> and <code>auth-token</code> attributes must be set on{" "}
          <code>&lt;assistant-widget&gt;</code>.
        </div>
      ) : (
        <>
          <div className="aw-messages">
            {messages.length === 0 && (
              <div className="aw-empty">Ask a question to get started.</div>
            )}
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form
            className="aw-form"
            onSubmit={(event) => {
              event.preventDefault();
              void handleSend();
            }}
          >
            <textarea
              className="aw-input"
              value={input}
              placeholder="Ask something... (Enter to send, Shift+Enter for newline)"
              rows={1}
              disabled={isSending}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              type="submit"
              className="aw-send"
              disabled={isSending || input.trim().length === 0}
            >
              {isSending ? "…" : "Send"}
            </button>
          </form>
        </>
      )}
    </div>
  );
}

const STYLES = `
  .aw-shell {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    display: flex;
    flex-direction: column;
    width: 380px;
    height: 560px;
    border: 1px solid #dfe3e8;
    border-radius: 12px;
    overflow: hidden;
    background: #ffffff;
    color: #1a1d21;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }
  .aw-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 14px;
    border-bottom: 1px solid #dfe3e8;
    font-size: 14px;
  }
  .aw-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
  }
  .aw-config-error {
    padding: 16px;
    font-size: 13px;
    color: #991b1b;
    background: #fee2e2;
    margin: 12px;
    border-radius: 8px;
  }
  .aw-messages {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .aw-empty {
    color: #6b7280;
    font-size: 13px;
    text-align: center;
    margin-top: 24px;
  }
  .aw-row {
    display: flex;
  }
  .aw-row-user {
    justify-content: flex-end;
  }
  .aw-row-other {
    justify-content: flex-start;
  }
  .aw-bubble {
    max-width: 82%;
    padding: 8px 12px;
    border-radius: 10px;
    font-size: 14px;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .aw-bubble-user {
    background: #4f46e5;
    color: #ffffff;
    border-bottom-right-radius: 3px;
  }
  .aw-bubble-assistant {
    background: #eef0f4;
    color: #1a1d21;
    border-bottom-left-radius: 3px;
  }
  .aw-bubble-error {
    background: #fee2e2;
    color: #991b1b;
  }
  .aw-citations {
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    font-size: 11px;
    opacity: 0.75;
  }
  .aw-chart {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
  }
  .aw-chart-title {
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .aw-chart-citations {
    border-top: none;
    padding-top: 4px;
    margin-top: 2px;
  }
  .aw-chart-pending {
    margin-top: 8px;
    font-size: 12px;
    font-style: italic;
    opacity: 0.65;
  }
  .aw-form {
    display: flex;
    gap: 8px;
    padding: 10px;
    border-top: 1px solid #dfe3e8;
  }
  .aw-input {
    flex: 1;
    resize: none;
    border: 1px solid #dfe3e8;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 14px;
    font-family: inherit;
    outline: none;
    max-height: 90px;
  }
  .aw-input:focus {
    border-color: #4f46e5;
  }
  .aw-input:disabled {
    background: #f4f5f7;
    opacity: 0.7;
  }
  .aw-send {
    background: #4f46e5;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0 16px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
  }
  .aw-send:disabled {
    opacity: 0.5;
    cursor: default;
  }
`;
