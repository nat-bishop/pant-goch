// Thin HTTP + SSE wrapper over the HANDLER v2 FastAPI backend. Vite proxies
// /api → :8000 in dev (see vite.config.ts).

const API = "/api";

async function ensureOk(r: Response, op: string): Promise<void> {
  if (r.ok) return;
  const body = await r.text().catch(() => "");
  console.warn(`${op} failed`, r.status, body);
  throw new Error(`${op} failed: ${r.status}`);
}

// ─── REST endpoints ──────────────────────────────────────────────────────

export async function startSession(): Promise<void> {
  const r = await fetch(`${API}/start`, { method: "POST" });
  await ensureOk(r, "start");
}

export async function postPick(option_idx: number): Promise<void> {
  const r = await fetch(`${API}/pick`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ option_idx }),
  });
  await ensureOk(r, "pick");
}

export async function postWalk(to_location: string): Promise<void> {
  const r = await fetch(`${API}/walk`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ to_location }),
  });
  await ensureOk(r, "walk");
}

// ─── SSE event vocabulary (must match handler_v2/sse.py) ─────────────────

export interface OptionPayload {
  text: string;
  leaves_for: string | null;
}

export type SseEvent =
  | {
      event: "scene_start";
      scene_id: number;
      location_id: string;
      location_name: string;
    }
  | { event: "state_change"; kind: string; payload: Record<string, unknown> }
  | {
      event: "prose_word";
      idx: number;
      word: string;
      is_last?: boolean;
    }
  | { event: "options"; options: OptionPayload[] }
  | { event: "scene_exit" }
  | { event: "location_change"; from: string; to: string }
  | {
      event: "log_entry_word";
      scene_id: number;
      idx: number;
      word: string;
      is_last?: boolean;
    }
  | { event: "game_over"; ending_id: string }
  | { event: "error"; message: string }
  | { event: "session_done" };

export function streamUrl(): string {
  return `${API}/events`;
}

/** Consume an SSE response body into a typed event stream. Each
 * `data: { ... }\n\n` block parses as one SseEvent. Comment lines
 * (`: connected`) are skipped. Lifted from v1 web/src/api.ts:140-171. */
export async function* readSseStream(
  response: Response,
): AsyncGenerator<SseEvent, void, void> {
  if (!response.body) throw new Error("no SSE response body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const idx = buffer.indexOf("\n\n");
        if (idx < 0) break;
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        for (const line of block.split("\n")) {
          if (line.startsWith("data: ")) {
            try {
              yield JSON.parse(line.slice(6)) as SseEvent;
            } catch (err) {
              console.warn("malformed SSE payload", line, err);
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
