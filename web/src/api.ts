// Thin HTTP + SSE wrapper over the FastAPI sidecar. Vite proxies /api → :8000
// in dev (see vite.config.ts).

const API = "/api";

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
  in_reply_to_beat?: string | null;
  voice_band_used?: string | null;
}

export interface InboxResponse {
  folders: Record<string, Email[]>;
  state: GameState;
}

export interface GameState {
  ended: boolean;
  end_reason: string | null;
  unread_counts: Record<string, number>;
}

export interface ThreadResponse {
  thread_id: string;
  emails: Email[];
}

export async function getScenario() {
  const r = await fetch(`${API}/scenario`);
  if (!r.ok) throw new Error(`scenario: ${r.status}`);
  return (await r.json()) as { title: string; premise: string };
}

export async function getInbox(): Promise<InboxResponse> {
  const r = await fetch(`${API}/inbox`);
  if (!r.ok) throw new Error(`inbox: ${r.status}`);
  return r.json();
}

export async function getThread(threadId: string): Promise<ThreadResponse> {
  const r = await fetch(`${API}/thread/${encodeURIComponent(threadId)}`);
  if (!r.ok) throw new Error(`thread: ${r.status}`);
  return r.json();
}

export async function postOpen(emailId: string) {
  const r = await fetch(`${API}/open`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email_id: emailId }),
  });
  if (!r.ok) throw new Error(`open: ${r.status}`);
  return r.json() as Promise<{ state: GameState; new_emails: Email[] }>;
}

// SSE helpers ─────────────────────────────────────────────────────────

export type SseEvent =
  | { type: "send_recorded"; email: Email }
  | { type: "classified"; quality: string; value: string; score: number }
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
  if (!r.ok || !r.body) throw new Error(`${path}: ${r.status}`);
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
          } catch {
            // ignore unparseable
          }
        }
      }
    }
  }
}

export interface SendArgs {
  in_reply_to: string | null;
  recipient: string;
  body: string;
}

export function streamSend(args: SendArgs) {
  return postJsonSse("/send", args);
}
