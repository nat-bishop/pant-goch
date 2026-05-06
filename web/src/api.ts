// Thin HTTP + SSE wrapper over the FastAPI sidecar. Vite proxies /api → :8000
// in dev (see vite.config.ts).

const API = "/api";

async function ensureOk(r: Response, op: string): Promise<void> {
  if (r.ok) return;
  // Surface a friendly message to UI banners while keeping the raw status +
  // response body in dev tools for debugging. Sidecar 5xx → "sidecar error";
  // 4xx (bad request, session ended, unknown thread) → "request rejected".
  const detail = r.status >= 500 ? "sidecar error" : "request rejected";
  const body = await r.text().catch(() => "");
  console.warn(`${op} failed`, r.status, body);
  throw new Error(detail);
}

export interface Email {
  id: string;
  from: string;
  to: string;
  subject: string;
  sent_at: string;
  is_read: boolean;
  preview: string;
  folder: string;
  thread_id: string | null;
  body?: string;
  beat_id?: string | null;
}

export interface InboxResponse {
  folders: Record<string, Email[]>;
  state: GameState;
}

export interface GameState {
  ended: boolean;
  end_reason: string | null;
  unread_counts: Record<string, number>;
  unread_counts_by_thread: Record<string, number>;
}

export interface ThreadResponse {
  thread_id: string;
  emails: Email[];
}

export interface NpcInfo {
  id: string;
  display_name: string;
  email_address: string;
  thread_id: string;
  replies_enabled: boolean;
}

export interface ScenarioResponse {
  title: string;
  premise: string;
  npcs: NpcInfo[];
}

export async function getScenario(): Promise<ScenarioResponse> {
  const r = await fetch(`${API}/scenario`);
  await ensureOk(r, "scenario");
  return r.json();
}

export async function getInbox(): Promise<InboxResponse> {
  const r = await fetch(`${API}/inbox`);
  await ensureOk(r, "inbox");
  return r.json();
}

export async function getThread(threadId: string): Promise<ThreadResponse> {
  const r = await fetch(`${API}/thread/${encodeURIComponent(threadId)}`);
  await ensureOk(r, "thread");
  return r.json();
}

export async function postOpen(emailId: string) {
  const r = await fetch(`${API}/open`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email_id: emailId }),
  });
  await ensureOk(r, "open");
  return r.json() as Promise<{ state: GameState; new_emails: Email[] }>;
}

// SSE helpers ─────────────────────────────────────────────────────────

export type SseEvent =
  | { type: "send_recorded"; email: Email }
  | { type: "new_email"; email: Email }
  | { type: "state"; state: GameState }
  | { type: "done" }
  | { type: "error"; message: string };

/** POST a JSON body and stream back parsed SSE events. */
export async function* postJsonSse(
  path: string,
  body: unknown,
): AsyncGenerator<SseEvent, void, void> {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
  });
  await ensureOk(r, path);
  if (!r.body) throw new Error("sidecar error");
  yield* readSseStream(r.body);
}

async function* readSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseEvent, void, void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            yield JSON.parse(line.slice(6)) as SseEvent;
          } catch (e) {
            // Don't tear the stream over a single malformed event, but make
            // sure dev tools see it — silent JSON corruption was costing us
            // hours of "why didn't the UI update?" debugging.
            console.warn("unparseable SSE line", line, e);
          }
        }
      }
    }
  }
}

export interface SendArgs {
  in_reply_to: string | null;
  recipient: string;
  recipient_npc_id: string;
  body: string;
}

export function streamSend(args: SendArgs) {
  return postJsonSse("/send", args);
}
