import { create } from "zustand";
import type {
  AgentMeta,
  CastMember,
  Delta,
  DeltaDescriptors,
  GameState,
  SseEvent,
} from "./api";
import * as api from "./api";

// ─── domain shapes ───────────────────────────────────────────────────────

export interface MissionMeta {
  mission_id: string;
  mission_name: string;
  codename: string;
  max_days: number;
  agents: AgentMeta[];
  cast: CastMember[];
  opening_briefing: string;
  opening_prose: string;
}

export interface OptionDraft {
  action: string;
  agent: string | null;
  title: string;
  prose_words: string[];
  complete: boolean;
  forced: boolean;
}

/** One past case-file entry — full prose. Held in the store so the
 * case-file scroll can render any completed day inline without a
 * round-trip. */
export interface LogEntryRecord {
  day_index: number;
  action: string;
  title: string;
  text: string;
  delta: Delta;
  delta_descriptors: DeltaDescriptors;
}

export interface DayState {
  index: number;
  options: OptionDraft[];
  options_complete: boolean;
  selected_option_index: number | null;
  awaiting_auth: boolean;
  authorized: boolean;
  // Log entry streaming: present after authorize, until the entry completes.
  entry_action: string | null;
  entry_words: string[];
  entry_complete: boolean;
  // True after the server emits `awaiting_continue` — engine is paused
  // between days waiting for the player to click through.
  awaiting_continue: boolean;
  // Latest day delta (effects + reactions merged) — surfaced to the current-
  // day panel for the RESULT footer.
  delta: Delta;
  // Accumulated narrative-descriptor map across the day's mutation phases
  // (effects_applied + reactions_fired + day_closed). Powers the RESULT
  // footer in the current-day panel.
  delta_descriptors: DeltaDescriptors;
  reactions_fired: string[];
}

export interface Outcome {
  outcome_key: string;
  summary: string;
  terminal_reaction: string;
  days_used: number;
  days_max: number;
  agents_lost: string[];
  agents_alive: string[];
}

interface Store {
  session_id: string | null;
  mission_meta: MissionMeta | null;
  state: GameState;
  day: DayState | null;
  // Full prose per day. Click a case-file row to re-read any past day's entry.
  inbox: LogEntryRecord[];
  outcome: Outcome | null;
  errorMessage: string | null;
  bootStarted: boolean;
  // Onboarding modal — shown once at the start of a session, gated by user
  // dismissal. Reset on restart (no localStorage) so a replay is implicitly
  // first-time again.
  onboardingDismissed: boolean;

  startMission: () => Promise<void>;
  authorize: () => Promise<void>;
  continueDay: () => Promise<void>;
  selectOption: (idx: number | null) => void;
  dismissOnboarding: () => void;
  handleEvent: (event: SseEvent) => void;
  reset: () => void;
}

// ─── reducer-style helpers (kept outside the store to stay testable) ─────

function newDay(index: number): DayState {
  return {
    index,
    options: [],
    options_complete: false,
    selected_option_index: null,
    awaiting_auth: false,
    authorized: false,
    entry_action: null,
    entry_words: [],
    entry_complete: false,
    awaiting_continue: false,
    delta: {},
    delta_descriptors: {},
    reactions_fired: [],
  };
}

// Day accumulates mutation phases in order: effects → events → close.
// Both deltas and their narrative descriptors merge the same way — keep the
// first-seen `before` (the day's true starting value) and the latest
// `after` (the day's final value for that quality).
function mergeByQuality<T extends { before: unknown; after: unknown }>(
  a: Record<string, T>,
  b: Record<string, T> | undefined,
): Record<string, T> {
  if (!b || Object.keys(b).length === 0) return a;
  const out: Record<string, T> = { ...a };
  for (const [k, v] of Object.entries(b)) {
    out[k] = k in out ? { ...v, before: out[k].before } : v;
  }
  return out;
}

// ─── store ───────────────────────────────────────────────────────────────

// `useStore` is also attached to `window` in dev so Playwright tests can
// snapshot store state directly via `window.__STORE__.getState()`. No-op in
// production where `import.meta.env.DEV` is false.
const _store = create<Store>((set, get) => ({
  session_id: null,
  mission_meta: null,
  state: {},
  day: null,
  inbox: [],
  outcome: null,
  errorMessage: null,
  bootStarted: false,
  onboardingDismissed: false,

  reset: () =>
    set({
      session_id: null,
      mission_meta: null,
      state: {},
      day: null,
      inbox: [],
      outcome: null,
      errorMessage: null,
      bootStarted: false,
      onboardingDismissed: false,
    }),

  startMission: async () => {
    if (get().bootStarted) return;
    set({ bootStarted: true, errorMessage: null });
    // Sidecar boot waits for llama-server /health (~3-5s). Vite is up in
    // <200ms, so the first /mission/start typically hits ECONNREFUSED.
    // Retry with backoff so the user doesn't need to reload.
    const delays = [200, 400, 800, 1500, 2500, 4000];
    let lastErr: unknown = null;
    for (let attempt = 0; attempt <= delays.length; attempt++) {
      try {
        const r = await api.startMission();
        set({ session_id: r.session_id });
        // mission_meta + initial state will land via the first SSE event.
        return;
      } catch (e) {
        lastErr = e;
        if (attempt === delays.length) break;
        await new Promise((resolve) => setTimeout(resolve, delays[attempt]));
      }
    }
    const msg = lastErr instanceof Error ? lastErr.message : String(lastErr);
    set({ errorMessage: msg, bootStarted: false });
  },

  authorize: async () => {
    const { session_id, day } = get();
    if (!session_id || !day) return;
    if (day.selected_option_index === null) return;
    if (!day.awaiting_auth || day.authorized) return;
    try {
      await api.authorize(session_id, day.selected_option_index);
      // Re-read after the await — the server may have streamed
      // `entry_started`/`effects_applied` while we were awaiting the
      // /authorize response, and a stale-day spread here would clobber
      // those SSE-driven fields (entry_action, delta, etc.).
      const latest = get().day;
      if (!latest) return;
      set({
        day: { ...latest, authorized: true, awaiting_auth: false },
      });
    } catch (e: any) {
      set({ errorMessage: e.message ?? String(e) });
    }
  },

  continueDay: async () => {
    const { session_id, day } = get();
    if (!session_id || !day?.awaiting_continue) return;
    try {
      await api.continueDay(session_id);
      // No local state mutation — the server's `day_opened` for N+1 will
      // replace `day` wholesale, clearing `awaiting_continue`.
    } catch (e: any) {
      set({ errorMessage: e.message ?? String(e) });
    }
  },

  selectOption: (idx) => {
    const { day } = get();
    if (!day || day.authorized) return;
    const next = day.selected_option_index === idx ? null : idx;
    set({ day: { ...day, selected_option_index: next } });
  },

  dismissOnboarding: () => set({ onboardingDismissed: true }),

  handleEvent: (ev) => {
    switch (ev.event) {
      case "mission_started":
        set({
          mission_meta: {
            mission_id: ev.mission_id,
            mission_name: ev.mission_name,
            codename: ev.codename,
            max_days: ev.max_days,
            agents: ev.agents,
            cast: ev.cast,
            opening_briefing: ev.opening_briefing,
            opening_prose: ev.opening_prose,
          },
          state: ev.state,
        });
        break;

      case "day_opened":
        set({ day: newDay(ev.day), state: ev.state });
        break;

      case "option_started": {
        const c = get().day;
        if (!c || c.index !== ev.day) return;
        const opt: OptionDraft = {
          action: ev.action,
          agent: ev.agent,
          title: ev.title,
          prose_words: [],
          complete: false,
          forced: ev.forced,
        };
        const options = [...c.options];
        options[ev.option_idx] = opt;
        set({ day: { ...c, options } });
        break;
      }

      case "option_word": {
        const c = get().day;
        if (!c || c.index !== ev.day) return;
        const options = [...c.options];
        const opt = options[ev.option_idx];
        if (!opt) return;
        if (ev.is_last) {
          options[ev.option_idx] = { ...opt, complete: true };
        } else {
          options[ev.option_idx] = {
            ...opt,
            prose_words: [...opt.prose_words, ev.word],
          };
        }
        set({ day: { ...c, options } });
        break;
      }

      case "options_complete": {
        const c = get().day;
        if (!c) return;
        set({ day: { ...c, options_complete: true } });
        break;
      }

      case "awaiting_auth": {
        const c = get().day;
        if (!c) return;
        set({ day: { ...c, awaiting_auth: true } });
        break;
      }

      case "awaiting_continue": {
        const c = get().day;
        if (!c || c.index !== ev.day) return;
        set({ day: { ...c, awaiting_continue: true } });
        break;
      }

      case "effects_applied": {
        const c = get().day;
        if (!c) return;
        set({
          day: {
            ...c,
            delta: mergeByQuality(c.delta, ev.delta),
            delta_descriptors: mergeByQuality(
              c.delta_descriptors,
              ev.delta_descriptors,
            ),
          },
        });
        break;
      }

      case "reactions_fired": {
        const c = get().day;
        if (!c) return;
        set({
          day: {
            ...c,
            delta: mergeByQuality(c.delta, ev.delta),
            delta_descriptors: mergeByQuality(
              c.delta_descriptors,
              ev.delta_descriptors,
            ),
            reactions_fired: [...c.reactions_fired, ...ev.reactions],
          },
        });
        break;
      }

      case "entry_started": {
        const c = get().day;
        if (!c || c.index !== ev.day) return;
        set({
          day: {
            ...c,
            entry_action: ev.action,
            entry_words: [],
            entry_complete: false,
          },
        });
        break;
      }

      case "entry_word": {
        const c = get().day;
        if (!c || c.index !== ev.day) return;
        if (ev.is_last) {
          // Finalise: build inbox entry from accumulated text.
          const text = c.entry_words.join("");
          const title =
            c.selected_option_index !== null
              ? (c.options[c.selected_option_index]?.title ?? "")
              : "";
          const entry: LogEntryRecord = {
            day_index: c.index,
            action: c.entry_action ?? "?",
            title,
            text,
            delta: c.delta,
            delta_descriptors: c.delta_descriptors,
          };
          set({
            day: { ...c, entry_complete: true },
            inbox: [...get().inbox, entry],
            // Apply day delta to global state — covers + extraction_stage etc.
            state: applyDeltaToState(get().state, c.delta),
          });
        } else {
          set({
            day: { ...c, entry_words: [...c.entry_words, ev.word] },
          });
        }
        break;
      }

      case "day_closed": {
        const c = get().day;
        // Accumulate the day-close descriptors (days_remaining tick) into
        // the current-day panel's RESULT footer if the day is still on-screen.
        if (c && ev.delta_descriptors && Object.keys(ev.delta_descriptors).length > 0) {
          set({
            state: ev.state,
            day: {
              ...c,
              delta_descriptors: mergeByQuality(
                c.delta_descriptors,
                ev.delta_descriptors,
              ),
            },
          });
        } else {
          set({ state: ev.state });
        }
        break;
      }

      case "mission_resolved":
        set({
          outcome: {
            outcome_key: ev.outcome,
            summary: ev.summary,
            terminal_reaction: ev.terminal_reaction,
            days_used: ev.days_used,
            days_max: ev.days_max,
            agents_lost: ev.agents_lost,
            agents_alive: ev.agents_alive,
          },
        });
        break;

      case "session_closed":
        // nothing to do — the day loop ended; outcome (if any) is set.
        break;

      case "error":
        set({ errorMessage: `engine error: ${ev.code}` });
        break;

      default:
        // Unknown event kinds are ignored — the SSE stream might add new
        // events in future versions of the backend.
        console.warn("unknown SSE event", ev);
    }
  },
}));

export const useStore = _store;
if (import.meta.env.DEV) {
  (window as unknown as { __STORE__: typeof _store }).__STORE__ = _store;
}

function applyDeltaToState(state: GameState, delta: Delta): GameState {
  if (Object.keys(delta).length === 0) return state;
  const out = { ...state };
  for (const [k, v] of Object.entries(delta)) {
    out[k] = v.after;
  }
  return out;
}
