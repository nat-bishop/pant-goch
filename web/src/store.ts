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
  max_cycles: number;
  agents: AgentMeta[];
  cast: CastMember[];
  opening_briefing: string;
  opening_prose: string;
}

export interface OptionDraft {
  archetype: string;
  agent: string | null;
  params: Record<string, string>;
  title: string;
  prose_words: string[];
  complete: boolean;
  satisfies_event: string | null;
}

/** One past case-file entry — full prose. Held in the store so the
 * click-to-expand history modal can replay any cycle's entry without a
 * round-trip. */
export interface LogEntryRecord {
  cycle_index: number;
  archetype: string;
  text: string;
  delta: Delta;
}

/** One case-file history row — LLM-generated one-line shorthand for a cycle.
 * Server emits via `entry_shorthand` after the entry streams. Frontend
 * lists these in the case-file panel; clicking opens the full prose
 * (from `inbox` keyed by cycle_index). */
export interface LogShorthandRecord {
  cycle_index: number;
  shorthand: string;
}

export interface CycleState {
  index: number;
  options: OptionDraft[];
  options_complete: boolean;
  selected_option_index: number | null;
  awaiting_auth: boolean;
  authorized: boolean;
  // Log entry streaming: present after authorize, until the entry completes.
  entry_archetype: string | null;
  entry_words: string[];
  entry_complete: boolean;
  // True after the server emits `awaiting_continue` — engine is paused
  // between cycles waiting for the player to click through.
  awaiting_continue: boolean;
  // Latest cycle delta (effects + events merged) — surfaced to the current-
  // cycle panel for the RESULT footer.
  delta: Delta;
  // Accumulated narrative-descriptor map across the cycle's mutation phases
  // (effects_applied + events_fired + cycle_closed). Powers the RESULT
  // footer in the current-cycle panel: a one-line "what just moved" summary
  // pulled from server-side `_describe_*` helpers. Filtered: same-bucket
  // swings (e.g. 8→7 cover, both 'intact') don't surface.
  delta_descriptors: DeltaDescriptors;
  events_fired: string[];
}

export interface Outcome {
  outcome_key: string;
  summary: string;
  terminal_event: string;
  cycles_used: number;
  cycles_max: number;
  agents_lost: string[];
  agents_alive: string[];
}

interface Store {
  session_id: string | null;
  mission_meta: MissionMeta | null;
  state: GameState;
  cycle: CycleState | null;
  // Full prose per cycle. Click a case-file row to re-read any past cycle's entry.
  inbox: LogEntryRecord[];
  // History row shorthands. Server fires `entry_shorthand` after each
  // cycle's entry streams. Indexed by cycle_index — append-only.
  shorthands: LogShorthandRecord[];
  outcome: Outcome | null;
  errorMessage: string | null;
  bootStarted: boolean;
  modalDispatchIndex: number | null;
  // Onboarding modal — shown once at the start of a session, gated by user
  // dismissal. Reset on restart (no localStorage) so a replay is implicitly
  // first-time again.
  onboardingDismissed: boolean;

  startMission: () => Promise<void>;
  authorize: () => Promise<void>;
  continueCycle: () => Promise<void>;
  selectOption: (idx: number | null) => void;
  openModal: (cycleIndex: number) => void;
  closeModal: () => void;
  dismissOnboarding: () => void;
  handleEvent: (event: SseEvent) => void;
  reset: () => void;
}

// ─── reducer-style helpers (kept outside the store to stay testable) ─────

function newCycle(index: number): CycleState {
  return {
    index,
    options: [],
    options_complete: false,
    selected_option_index: null,
    awaiting_auth: false,
    authorized: false,
    entry_archetype: null,
    entry_words: [],
    entry_complete: false,
    awaiting_continue: false,
    delta: {},
    delta_descriptors: {},
    events_fired: [],
  };
}

// Cycle accumulates mutation phases in order: effects → events → close.
// Both deltas and their narrative descriptors merge the same way — keep the
// first-seen `before` (the cycle's true starting value) and the latest
// `after` (the cycle's final value for that quality).
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
  cycle: null,
  inbox: [],
  shorthands: [],
  outcome: null,
  errorMessage: null,
  bootStarted: false,
  modalDispatchIndex: null,
  onboardingDismissed: false,

  reset: () =>
    set({
      session_id: null,
      mission_meta: null,
      state: {},
      cycle: null,
      inbox: [],
      shorthands: [],
      outcome: null,
      errorMessage: null,
      modalDispatchIndex: null,
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
    const { session_id, cycle } = get();
    if (!session_id || !cycle) return;
    if (cycle.selected_option_index === null) return;
    if (!cycle.awaiting_auth || cycle.authorized) return;
    try {
      await api.authorize(session_id, cycle.selected_option_index);
      // Re-read after the await — the server may have streamed
      // `entry_started`/`effects_applied` while we were awaiting the
      // /authorize response, and a stale-cycle spread here would clobber
      // those SSE-driven fields (entry_archetype, delta, etc.).
      const latest = get().cycle;
      if (!latest) return;
      set({
        cycle: { ...latest, authorized: true, awaiting_auth: false },
      });
    } catch (e: any) {
      set({ errorMessage: e.message ?? String(e) });
    }
  },

  continueCycle: async () => {
    const { session_id, cycle } = get();
    if (!session_id || !cycle?.awaiting_continue) return;
    try {
      await api.continueCycle(session_id);
      // No local state mutation — the server's `cycle_opened` for N+1 will
      // replace `cycle` wholesale, clearing `awaiting_continue`.
    } catch (e: any) {
      set({ errorMessage: e.message ?? String(e) });
    }
  },

  selectOption: (idx) => {
    const { cycle } = get();
    if (!cycle || cycle.authorized) return;
    const next = cycle.selected_option_index === idx ? null : idx;
    set({ cycle: { ...cycle, selected_option_index: next } });
  },

  openModal: (cycleIndex) => set({ modalDispatchIndex: cycleIndex }),
  closeModal: () => set({ modalDispatchIndex: null }),

  dismissOnboarding: () => set({ onboardingDismissed: true }),

  handleEvent: (ev) => {
    switch (ev.event) {
      case "mission_started":
        set({
          mission_meta: {
            mission_id: ev.mission_id,
            mission_name: ev.mission_name,
            codename: ev.codename,
            max_cycles: ev.max_cycles,
            agents: ev.agents,
            cast: ev.cast,
            opening_briefing: ev.opening_briefing,
            opening_prose: ev.opening_prose,
          },
          state: ev.state,
        });
        break;

      case "cycle_opened":
        set({ cycle: newCycle(ev.cycle), state: ev.state });
        break;

      case "option_started": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
        const opt: OptionDraft = {
          archetype: ev.archetype,
          agent: ev.agent,
          params: ev.params,
          title: ev.title,
          prose_words: [],
          complete: false,
          satisfies_event: ev.satisfies_event,
        };
        const options = [...c.options];
        options[ev.option_idx] = opt;
        set({ cycle: { ...c, options } });
        break;
      }

      case "option_word": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
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
        set({ cycle: { ...c, options } });
        break;
      }

      case "options_complete": {
        const c = get().cycle;
        if (!c) return;
        set({ cycle: { ...c, options_complete: true } });
        break;
      }

      case "awaiting_auth": {
        const c = get().cycle;
        if (!c) return;
        set({ cycle: { ...c, awaiting_auth: true } });
        break;
      }

      case "awaiting_continue": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
        set({ cycle: { ...c, awaiting_continue: true } });
        break;
      }

      case "effects_applied": {
        const c = get().cycle;
        if (!c) return;
        set({
          cycle: {
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

      case "events_fired": {
        const c = get().cycle;
        if (!c) return;
        set({
          cycle: {
            ...c,
            delta: mergeByQuality(c.delta, ev.delta),
            delta_descriptors: mergeByQuality(
              c.delta_descriptors,
              ev.delta_descriptors,
            ),
            events_fired: [...c.events_fired, ...ev.events],
          },
        });
        break;
      }

      case "entry_started": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
        set({
          cycle: {
            ...c,
            entry_archetype: ev.archetype,
            entry_words: [],
            entry_complete: false,
          },
        });
        break;
      }

      case "entry_word": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
        if (ev.is_last) {
          // Finalise: build inbox entry from accumulated text.
          const text = c.entry_words.join("");
          const entry: LogEntryRecord = {
            cycle_index: c.index,
            archetype: c.entry_archetype ?? "?",
            text,
            delta: c.delta,
          };
          set({
            cycle: { ...c, entry_complete: true },
            inbox: [...get().inbox, entry],
            // Apply cycle delta to global state — covers + extraction_stage etc.
            state: applyDeltaToState(get().state, c.delta),
          });
        } else {
          set({
            cycle: { ...c, entry_words: [...c.entry_words, ev.word] },
          });
        }
        break;
      }

      case "entry_shorthand": {
        // Append the case-file history row. No cycle guard — shorthands
        // can land slightly after the cycle has rolled over (the LLM
        // call runs in parallel with the cycle-close sequence).
        set({
          shorthands: [
            ...get().shorthands,
            { cycle_index: ev.cycle, shorthand: ev.shorthand },
          ],
        });
        break;
      }

      case "cycle_closed": {
        const c = get().cycle;
        // Accumulate the cycle-close descriptors (cycles_remaining tick) into
        // the current-cycle panel's RESULT footer if the cycle is still on-screen.
        if (c && ev.delta_descriptors && Object.keys(ev.delta_descriptors).length > 0) {
          set({
            state: ev.state,
            cycle: {
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
            terminal_event: ev.terminal_event,
            cycles_used: ev.cycles_used,
            cycles_max: ev.cycles_max,
            agents_lost: ev.agents_lost,
            agents_alive: ev.agents_alive,
          },
        });
        break;

      case "session_closed":
        // nothing to do — the cycle loop ended; outcome (if any) is set.
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
