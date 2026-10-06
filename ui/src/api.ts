/**
 * Minimal client for the Generic AI Assistant Framework backend.
 *
 * Wire format matches app/api/chat.py exactly (snake_case JSON) — this
 * module is the only place that translates to/from the camelCase shapes
 * the rest of the widget uses. Nothing here invents an endpoint or field
 * that isn't already in app/api/chat.py / app/main.py.
 */

export interface Citation {
  documentTitle: string;
  chunkIndex: number;
}

export type ChartType = "bar" | "line" | "pie";

export interface ChartSeries {
  name: string;
  values: number[];
}

export interface Chart {
  chartType: ChartType;
  title: string;
  labels: string[];
  series: ChartSeries[];
  sourceChunks: Citation[];
}

export interface ChatReply {
  reply: string;
  citations: Citation[];
  conversationId: string;
  grounded: boolean;
  chart?: Chart;
}

export interface PostChatParams {
  assistantId: string;
  message: string;
  conversationId?: string;
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
  sequenceNo: number;
  citations: Citation[];
}

/** Thrown for 401/403 — the caller should show "not authorized", not a
 * generic failure. Kept distinct from ApiError so the UI can tell them apart
 * without inspecting status codes itself. */
export class ApiAuthError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(status === 401 ? "Not authenticated" : "Not authorized");
    this.name = "ApiAuthError";
    this.status = status;
  }
}

/** Everything else that isn't a 2xx: network failure, 404, 5xx, etc. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface CitationDto {
  document_title: string;
  chunk_index: number;
}

interface ChartSeriesDto {
  name: string;
  values: number[];
}

interface ChartDto {
  chart_type: ChartType;
  title: string;
  labels: string[];
  series: ChartSeriesDto[];
  source_chunks: CitationDto[];
}

interface ChatResponseDto {
  assistant_id: string;
  reply: string;
  conversation_id: string;
  citations: CitationDto[];
  grounded: boolean;
  chart?: ChartDto | null;
}

interface MessageDto {
  role: string;
  content: string;
  sequence_no: number;
  citations: CitationDto[];
}

function toCitation(dto: CitationDto): Citation {
  return { documentTitle: dto.document_title, chunkIndex: dto.chunk_index };
}

function toChart(dto: ChartDto): Chart {
  return {
    chartType: dto.chart_type,
    title: dto.title,
    labels: dto.labels,
    series: dto.series.map((s) => ({ name: s.name, values: s.values })),
    sourceChunks: dto.source_chunks.map(toCitation),
  };
}

/** Network-level failure message shared by request() and postChatStream() —
 * fires only when fetch() itself rejects (DNS, connection refused, wrong
 * port, CORS block), never for an HTTP error response from a reachable
 * backend (those are handled separately, after a Response actually comes
 * back). Includes the attempted base URL so a wrong-port/wrong-host mistake
 * is visible in the error text itself, not just "is it running?". */
function unreachableBackendMessage(baseUrl: string): string {
  return `Could not reach ${baseUrl} — check the API base URL and confirm the backend is running there.`;
}

async function request<T>(
  url: string,
  baseUrl: string,
  token: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError(0, unreachableBackendMessage(baseUrl));
  }

  if (response.status === 401 || response.status === 403) {
    throw new ApiAuthError(response.status);
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body: unknown = await response.json();
      if (
        body &&
        typeof body === "object" &&
        "detail" in body &&
        typeof (body as { detail: unknown }).detail === "string"
      ) {
        detail = (body as { detail: string }).detail;
      }
    } catch {
      // Response body wasn't JSON (or was empty) — keep the default detail.
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

/** POST /chat — one conversational turn. Omits conversation_id from the
 * request body entirely when not provided, matching ChatRequest's optional
 * field (rather than sending an explicit null). */
export async function postChat(
  baseUrl: string,
  token: string,
  params: PostChatParams,
): Promise<ChatReply> {
  const body: { assistant_id: string; message: string; conversation_id?: string } = {
    assistant_id: params.assistantId,
    message: params.message,
  };
  if (params.conversationId) {
    body.conversation_id = params.conversationId;
  }

  const data = await request<ChatResponseDto>(`${baseUrl}/chat`, baseUrl, token, {
    method: "POST",
    body: JSON.stringify(body),
  });

  return {
    reply: data.reply,
    citations: data.citations.map(toCitation),
    conversationId: data.conversation_id,
    grounded: data.grounded,
    chart: data.chart ? toChart(data.chart) : undefined,
  };
}

interface DeltaEventDto {
  type: "delta";
  text: string;
}

interface ChartEventDto {
  type: "chart";
  chart: ChartDto;
}

interface DoneEventDto {
  type: "done";
  conversation_id: string;
  citations: CitationDto[];
  grounded: boolean;
}

interface ErrorEventDto {
  type: "error";
  detail: string;
}

type StreamEventDto = DeltaEventDto | ChartEventDto | DoneEventDto | ErrorEventDto;

function parseStreamEventDto(raw: unknown): StreamEventDto | undefined {
  if (!raw || typeof raw !== "object" || !("type" in raw)) {
    return undefined;
  }
  const type = (raw as { type: unknown }).type;
  if (type === "delta" && "text" in raw && typeof (raw as DeltaEventDto).text === "string") {
    return raw as DeltaEventDto;
  }
  if (type === "chart" && "chart" in raw && (raw as ChartEventDto).chart) {
    return raw as ChartEventDto;
  }
  if (
    type === "done" &&
    "conversation_id" in raw &&
    typeof (raw as DoneEventDto).conversation_id === "string"
  ) {
    const done = raw as DoneEventDto;
    return {
      type: "done",
      conversation_id: done.conversation_id,
      citations: Array.isArray(done.citations) ? done.citations : [],
      grounded: Boolean(done.grounded),
    };
  }
  if (type === "error" && "detail" in raw && typeof (raw as ErrorEventDto).detail === "string") {
    return raw as ErrorEventDto;
  }
  return undefined;
}

export interface PostChatStreamParams extends PostChatParams {
  onDelta: (text: string) => void;
  /** Fired at most once per turn, only when the backend actually sent a
   * "chart" SSE event (see app/main.py's _encode_sse_event) — never called
   * with an empty/placeholder chart. Arrives after the last onDelta call
   * and before onDone (see backend/app/orchestration/chat_service.py's
   * ChatStreamChart docstring for the ordering rationale), so callers
   * should treat the gap between the last delta and this call (or onDone,
   * if no chart applies) as a possible "generating chart..." period. */
  onChart?: (chart: Chart) => void;
  onDone: (result: { conversationId: string; citations: Citation[]; grounded: boolean }) => void;
  onError: (detail: string) => void;
  /** Called instead of onError when the backend answers 401 (token
   * rejected), so the caller can end the session rather than keep sending
   * a token the backend no longer accepts. */
  onUnauthorized?: () => void;
}

/** POST /chat/stream — same turn as postChat, but the reply arrives as SSE
 * deltas instead of one blocking JSON response.
 *
 * Native EventSource can't be used here: it only supports GET requests with
 * no custom headers and no body, so it has no way to send the Bearer token
 * or the JSON request body this endpoint requires (verified against the
 * EventSource spec, not assumed). Instead this reads the fetch() response
 * body as a stream and parses the `data: {...}\n\n` frames by hand — the
 * same framing app/main.py's `_encode_sse_event` writes.
 *
 * Resolves once the stream ends (after onDone or onError fires) — it never
 * throws; every failure path goes through onError so callers can't forget
 * to handle a rejected promise.
 */
export async function postChatStream(
  baseUrl: string,
  token: string,
  params: PostChatStreamParams,
): Promise<void> {
  const { onDelta, onChart, onDone, onError, onUnauthorized, ...chatParams } = params;
  const body: { assistant_id: string; message: string; conversation_id?: string } = {
    assistant_id: chatParams.assistantId,
    message: chatParams.message,
  };
  if (chatParams.conversationId) {
    body.conversation_id = chatParams.conversationId;
  }

  let response: Response;
  try {
    response = await fetch(`${baseUrl}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
    });
  } catch {
    onError(unreachableBackendMessage(baseUrl));
    return;
  }

  if (response.status === 401) {
    if (onUnauthorized) {
      onUnauthorized();
    } else {
      onError("Not authenticated");
    }
    return;
  }
  if (response.status === 403) {
    onError("Not authorized");
    return;
  }

  if (!response.ok || !response.body) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const errorBody: unknown = await response.json();
      if (
        errorBody &&
        typeof errorBody === "object" &&
        "detail" in errorBody &&
        typeof (errorBody as { detail: unknown }).detail === "string"
      ) {
        detail = (errorBody as { detail: string }).detail;
      }
    } catch {
      // Response body wasn't JSON (or was empty) — keep the default detail.
    }
    onError(detail);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDone = false;

  function handleFrame(frame: string): void {
    const line = frame.trim();
    if (!line.startsWith("data:")) {
      return;
    }
    const jsonPart = line.slice("data:".length).trim();
    if (!jsonPart) {
      return;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(jsonPart);
    } catch {
      return;
    }

    const event = parseStreamEventDto(parsed);
    if (!event) {
      return;
    }

    if (event.type === "delta") {
      onDelta(event.text);
    } else if (event.type === "chart") {
      onChart?.(toChart(event.chart));
    } else if (event.type === "done") {
      sawDone = true;
      onDone({
        conversationId: event.conversation_id,
        citations: event.citations.map(toCitation),
        grounded: event.grounded,
      });
    } else {
      sawDone = true;
      onError(event.detail);
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        handleFrame(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
    }
  } catch {
    if (!sawDone) {
      onError("Connection to the assistant was lost.");
    }
    return;
  }

  if (!sawDone) {
    onError("The assistant is temporarily unavailable. Please try again.");
  }
}

/** GET /sessions/{conversationId}/messages — full history for a
 * conversation the caller owns, oldest first. */
export async function getMessages(
  baseUrl: string,
  token: string,
  conversationId: string,
): Promise<HistoryMessage[]> {
  const data = await request<MessageDto[]>(
    `${baseUrl}/sessions/${conversationId}/messages`,
    baseUrl,
    token,
    { method: "GET" },
  );

  return data.map((message) => ({
    role: message.role === "assistant" ? "assistant" : "user",
    content: message.content,
    sequenceNo: message.sequence_no,
    citations: message.citations.map(toCitation),
  }));
}

export interface AssistantSummary {
  assistantId: string;
  displayName: string;
}

interface AssistantSummaryDto {
  assistant_id: string;
  display_name: string;
}

/** GET /assistants — every assistant config currently loaded by the
 * backend. Used by AssistantWidgetApp to validate the widget's configured
 * assistant-id before letting the user send messages against it — an
 * assistant-id that's misspelled or points at the wrong (but still
 * existing) config fails silently otherwise, since /chat itself has no way
 * to know the caller "meant" a different assistant. */
export async function getAssistants(baseUrl: string, token: string): Promise<AssistantSummary[]> {
  const data = await request<AssistantSummaryDto[]>(`${baseUrl}/assistants`, baseUrl, token, {
    method: "GET",
  });
  return data.map((a) => ({ assistantId: a.assistant_id, displayName: a.display_name }));
}
