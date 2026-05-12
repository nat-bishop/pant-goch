import { create } from "zustand";
import type { AgentMeta, Delta, GameState, SseEvent } from "./api";
import { OPERATIONS_ACTOR } from "./api";
import * as api from "./api";

// ─── domain shapes ───────────────────────────────────────────────────────

export interface MissionMeta {
  mission_id: string;
  mission_name: string;
  codename: string;
  max_cycles: number;
  agents: AgentMeta[];
}

export interface OptionDraft {
  archetype: string;
  agent: string | null;
  params: Record<string, string>;
  title: string;
  prose_words: string[];
  complete: boolean;
  satisfies_moment: string | null;
}

export interface DispatchEntry {
  cycle_index: number;
  acting: string;
  archetype: string;
  text: string;
  delta: Delta;
}

export interface CycleState {
  index: number;
  sop_words: string[];
  sop_complete: boolean;
  options: OptionDraft[];
  options_complete: boolean;
  selected_option_index: number | null;
  awaiting_auth: boolean;
  authorized: boolean;
  // Dispatch streaming: present after authorize, until the dispatch completes.
  dispatch_acting: string | null;
  dispatch_archetype: string | null;
  dispatch_words: string[];
  dispatch_complete: boolean;
  // True after the server emits `awaiting_continue` — engine is paused
  // between cycles waiting for the player to click through.
  awaiting_continue: boolean;
  // Latest cycle delta (effects + beats merged) — surfaced to the dispatch modal.
  delta: Delta;
  beats_fired: string[];
}

export interface Outcome {
  outcome_key: string;
  summary: string;
  terminal_beat: string;
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
  inbox: DispatchEntry[];
  outcome: Outcome | null;
  errorMessage: string | null;
  bootStarted: boolean;
  modalDispatchIndex: number | null;

  startMission: () => Promise<void>;
  authorize: () => Promise<void>;
  continueCycle: () => Promise<void>;
  selectOption: (idx: number | null) => void;
  openModal: (cycleIndex: number) => void;
  closeModal: () => void;
  handleEvent: (event: SseEvent) => void;
  reset: () => void;
}

// ─── reducer-style helpers (kept outside the store to stay testable) ─────

function newCycle(index: number): CycleState {
  return {
    index,
    sop_words: [],
    sop_complete: false,
    options: [],
    options_complete: false,
    selected_option_index: null,
    awaiting_auth: false,
    authorized: false,
    dispatch_acting: null,
    dispatch_archetype: null,
    dispatch_words: [],
    dispatch_complete: false,
    awaiting_continue: false,
    delta: {},
    beats_fired: [],
  };
}

function mergeDelta(a: Delta, b: Delta): Delta {
  const out: Delta = { ...a };
  for (const [k, v] of Object.entries(b)) {
    if (k in out) {
      out[k] = { before: out[k].before, after: v.after };
    } else {
      out[k] = v;
    }
  }
  return out;
}

// ─── store ───────────────────────────────────────────────────────────────

export const useStore = create<Store>((set, get) => ({
  session_id: null,
  mission_meta: null,
  state: {},
  cycle: null,
  inbox: [],
  outcome: null,
  errorMessage: null,
  bootStarted: false,
  modalDispatchIndex: null,

  reset: () =>
    set({
      session_id: null,
      mission_meta: null,
      state: {},
      cycle: null,
      inbox: [],
      outcome: null,
      errorMessage: null,
      modalDispatchIndex: null,
      bootStarted: false,
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
      set({
        cycle: { ...cycle, authorized: true, awaiting_auth: false },
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
          },
          state: ev.state,
        });
        break;

      case "cycle_opened":
        set({ cycle: newCycle(ev.cycle), state: ev.state });
        break;

      case "sop_word": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
        if (ev.is_last) {
          set({ cycle: { ...c, sop_complete: true } });
        } else {
          set({ cycle: { ...c, sop_words: [...c.sop_words, ev.word] } });
        }
        break;
      }

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
          satisfies_moment: ev.satisfies_moment,
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
        set({ cycle: { ...c, delta: mergeDelta(c.delta, ev.delta) } });
        break;
      }

      case "beats_fired": {
        const c = get().cycle;
        if (!c) return;
        set({
          cycle: {
            ...c,
            delta: mergeDelta(c.delta, ev.delta),
            beats_fired: [...c.beats_fired, ...ev.beats],
          },
        });
        break;
      }

      case "dispatch_started": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
        set({
          cycle: {
            ...c,
            dispatch_acting: ev.acting,
            dispatch_archetype: ev.archetype,
            dispatch_words: [],
            dispatch_complete: false,
          },
        });
        break;
      }

      case "dispatch_word": {
        const c = get().cycle;
        if (!c || c.index !== ev.cycle) return;
        if (ev.is_last) {
          // Finalise: build inbox entry from accumulated text.
          const text = c.dispatch_words.join("");
          const entry: DispatchEntry = {
            cycle_index: c.index,
            acting: c.dispatch_acting ?? OPERATIONS_ACTOR,
            archetype: c.dispatch_archetype ?? "?",
            text,
            delta: c.delta,
          };
          set({
            cycle: { ...c, dispatch_complete: true },
            inbox: [...get().inbox, entry],
            // Apply cycle delta to global state — covers + extraction_stage etc.
            state: applyDeltaToState(get().state, c.delta),
          });
        } else {
          set({
            cycle: { ...c, dispatch_words: [...c.dispatch_words, ev.word] },
          });
        }
        break;
      }

      case "cycle_closed":
        set({ state: ev.state });
        break;

      case "mission_resolved":
        set({
          outcome: {
            outcome_key: ev.outcome,
            summary: ev.summary,
            terminal_beat: ev.terminal_beat,
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

function applyDeltaToState(state: GameState, delta: Delta): GameState {
  if (Object.keys(delta).length === 0) return state;
  const out = { ...state };
  for (const [k, v] of Object.entries(delta)) {
    out[k] = v.after;
  }
  return out;
}
