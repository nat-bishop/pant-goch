// Thin HTTP + SSE wrapper over the HANDLER FastAPI sidecar. Vite proxies
// /api → :8000 in dev (see vite.config.ts).

const API = "/api";

async function ensureOk(r: Response, op: string): Promise<void> {
  if (r.ok) return;
  const detail = r.status >= 500 ? "sidecar error" : "request rejected";
  const body = await r.text().catch(() => "");
  console.warn(`${op} failed`, r.status, body);
  throw new Error(detail);
}

// ─── shared types ────────────────────────────────────────────────────────

export function isWinOutcome(outcomeKey: string): boolean {
  return outcomeKey.startsWith("win_");
}

export interface AgentMeta {
  id: string;
  codename: string;
  file_number: string;
}

/** Full cast entry surfaced to the onboarding card. Covers agents,
 * handler (player), boss, and opposition — the four roles the spec
 * recognises. Agents also appear in `AgentMeta` for the sidebar, but
 * the onboarding roster needs the broader set (REDFIELD, MUNTEANU,
 * MERIWETHER) so they're known names before they appear in prose. */
export interface CastMember {
  id: string;
  codename: string;
  role: "agent" | "handler" | "boss" | "opposition" | "asset";
  file_number: string;
  situation: string;
}

export interface OptionPayload {
  archetype: string;
  agent: string | null;
  params: Record<string, string>;
  title: string;
  satisfies_event: string | null;
}

export type Quality = number | string | boolean;
export type GameState = Record<string, Quality>;
export type Delta = Record<string, { before: Quality | null; after: Quality }>;

/** Narrative descriptor for one quality in a cycle's delta. `before` is
 * null on first-time quality writes (e.g. authored mission qualities
 * surfacing for the first time); `agent_id` is the operative whose cover
 * changed (e.g. "LYNX") or null for mission-scoped qualities. The pair
 * always represents a *narrative* change — server-side filtering drops
 * same-bucket swings. */
export interface DeltaDescriptor {
  before: string | null;
  after: string | null;
  agent_id: string | null;
}
export type DeltaDescriptors = Record<string, DeltaDescriptor>;

// ─── REST endpoints ──────────────────────────────────────────────────────

export interface StartResponse {
  session_id: string;
  mission_id: string;
  mission_name: string;
  codename: string;
  max_cycles: number;
  opening_briefing: string;
  opening_prose: string;
  cast: CastMember[];
}

export async function startMission(): Promise<StartResponse> {
  const r = await fetch(`${API}/mission/start`, { method: "POST" });
  await ensureOk(r, "mission/start");
  return (await r.json()) as StartResponse;
}

export async function authorize(
  session_id: string,
  option_index: number,
): Promise<void> {
  const r = await fetch(`${API}/mission/${session_id}/authorize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ option_index }),
  });
  await ensureOk(r, "authorize");
}

export async function continueCycle(session_id: string): Promise<void> {
  const r = await fetch(`${API}/mission/${session_id}/continue`, {
    method: "POST",
  });
  await ensureOk(r, "continue");
}

export interface StateResponse {
  session_id: string;
  cycle_index: number | null;
  state: GameState;
  outcome: string | null;
}

export async function fetchState(session_id: string): Promise<StateResponse> {
  const r = await fetch(`${API}/mission/${session_id}/state`);
  await ensureOk(r, "state");
  return (await r.json()) as StateResponse;
}

// ─── SSE event vocabulary (must match handler/main.py) ───────────────────

export type SseEvent =
  | { event: "mission_started"; mission_id: string; mission_name: string; codename: string; max_cycles: number; agents: AgentMeta[]; cast: CastMember[]; opening_briefing: string; opening_prose: string; state: GameState }
  | { event: "cycle_opened"; cycle: number; state: GameState }
  | { event: "option_started"; cycle: number; option_idx: number; archetype: string; agent: string | null; params: Record<string, string>; title: string; satisfies_event: string | null }
  | { event: "option_word"; cycle: number; option_idx: number; idx: number; word: string; is_last?: boolean }
  | { event: "options_complete"; cycle: number; count: number; has_forced_event: boolean; event_id: string | null; engine_fallback: boolean }
  | { event: "awaiting_auth" }
  | { event: "awaiting_continue"; cycle: number }
  | { event: "effects_applied"; delta: Delta; delta_descriptors?: DeltaDescriptors }
  | { event: "events_fired"; events: string[]; delta: Delta; delta_descriptors?: DeltaDescriptors }
  | { event: "entry_started"; cycle: number; archetype: string }
  | { event: "entry_word"; cycle: number; idx: number; word: string; is_last?: boolean }
  | { event: "entry_shorthand"; cycle: number; shorthand: string }
  | { event: "cycle_closed"; cycle: number; state: GameState; delta_descriptors?: DeltaDescriptors }
  | { event: "mission_resolved"; outcome: string; summary: string; terminal_event: string; cycles_used: number; cycles_max: number; agents_lost: string[]; agents_alive: string[] }
  | { event: "session_closed" }
  | { event: "error"; code: string };

export function streamUrl(session_id: string): string {
  return `${API}/mission/${session_id}/stream`;
}

/** Consume an SSE response body into a typed event stream. The fetch returns
 * a `text/event-stream` body; each `data: { ... }\n\n` block parses as one
 * SseEvent. Comment lines (`: heartbeat`) are skipped. */
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
