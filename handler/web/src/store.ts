// Zustand store for HANDLER v2. Pure event dispatch — UI reads from
// store, SSE events route through `handleEvent` to mutate. No imperative
// fetching here; api.ts owns HTTP.

import { create } from "zustand";
import type { OptionPayload, SseEvent } from "./api";

export type Phase =
  | "idle" // pre-start
  | "in_scene" // streaming prose, waiting for options
  | "awaiting_pick" // options shown, waiting on user click
  | "between_scenes" // EndScene happened; case file finishing; pick a walk
  | "game_over"
  | "stranded"; // EndScene at a location with no adjacents (shouldn't happen with Pant Goch)

export interface TurnPayload {
  /** Prose words appended as `prose_word` events arrive. */
  words: string[];
  options: OptionPayload[];
  player_pick: OptionPayload | null;
  /** True until the prose stream's sentinel arrives. */
  streaming: boolean;
}

export interface SceneSnapshot {
  scene_id: number;
  location_id: string;
  location_name: string;
  /** Turns recorded for this scene. The last entry is the currently
   * streaming turn (or the most-recently-completed turn after
   * scene_exit). */
  turns: TurnPayload[];
  /** Beats fired in this scene (UI flair). */
  unlocked_beats: string[];
  closed: boolean;
}

export interface CaseFileEntry {
  scene_id: number;
  words: string[];
  streaming: boolean;
}

interface Store {
  phase: Phase;
  /** All scenes the session has seen, in order. Renderer iterates this
   * to draw the running history (prior scenes dim, current bright). */
  scenes: SceneSnapshot[];
  /** Case-file entries, one per closed scene. Words append as
   * log_entry_word events arrive; `streaming` freezes false on
   * sentinel. */
  case_file: CaseFileEntry[];
  game_over_ending: string | null;
  error: string | null;

  resetForRestart: () => void;
  attachPick: (option: OptionPayload) => void;
  handleEvent: (event: SseEvent) => void;
}

function makeTurn(): TurnPayload {
  return { words: [], options: [], player_pick: null, streaming: true };
}

export const useStore = create<Store>((set) => ({
  phase: "idle",
  scenes: [],
  case_file: [],
  game_over_ending: null,
  error: null,

  resetForRestart: () =>
    set({
      phase: "idle",
      scenes: [],
      case_file: [],
      game_over_ending: null,
      error: null,
    }),

  attachPick: (option) =>
    set((s) => {
      const scenes = [...s.scenes];
      const cur = scenes[scenes.length - 1];
      if (!cur) return s;
      const turns = [...cur.turns];
      const last = turns[turns.length - 1];
      if (!last) return s;
      turns[turns.length - 1] = { ...last, player_pick: option };
      scenes[scenes.length - 1] = { ...cur, turns };
      return { scenes, phase: "in_scene" };
    }),

  handleEvent: (event) =>
    set((s) => {
      switch (event.event) {
        case "scene_start": {
          const snapshot: SceneSnapshot = {
            scene_id: event.scene_id,
            location_id: event.location_id,
            location_name: event.location_name,
            turns: [makeTurn()],
            unlocked_beats: [],
            closed: false,
          };
          return {
            phase: "in_scene",
            scenes: [...s.scenes, snapshot],
          };
        }

        case "state_change": {
          if (event.kind !== "unlock_beat") return s;
          const beat_id = event.payload.beat_id as string | undefined;
          if (!beat_id) return s;
          const scenes = [...s.scenes];
          const cur = scenes[scenes.length - 1];
          if (!cur) return s;
          scenes[scenes.length - 1] = {
            ...cur,
            unlocked_beats: [...cur.unlocked_beats, beat_id],
          };
          return { scenes };
        }

        case "prose_word": {
          const scenes = [...s.scenes];
          const cur = scenes[scenes.length - 1];
          if (!cur) return s;
          let turns = [...cur.turns];
          let turn = turns[turns.length - 1];
          // idx===0 marks the start of a turn. If the current turn
          // already has words, the previous turn is closed and we need
          // to push a new turn-in-progress.
          if (event.idx === 0 && !event.is_last && turn && turn.words.length > 0) {
            turns = [...turns, makeTurn()];
            turn = turns[turns.length - 1];
          }
          if (event.is_last) {
            turns[turns.length - 1] = { ...turn, streaming: false };
          } else {
            turns[turns.length - 1] = {
              ...turn,
              words: [...turn.words, event.word],
            };
          }
          scenes[scenes.length - 1] = { ...cur, turns };
          return { scenes, phase: "in_scene" };
        }

        case "options": {
          const scenes = [...s.scenes];
          const cur = scenes[scenes.length - 1];
          if (!cur) return s;
          const turns = [...cur.turns];
          turns[turns.length - 1] = {
            ...turns[turns.length - 1],
            options: event.options,
            streaming: false,
          };
          scenes[scenes.length - 1] = { ...cur, turns };
          return { scenes, phase: "awaiting_pick" };
        }

        case "scene_exit": {
          const scenes = [...s.scenes];
          const cur = scenes[scenes.length - 1];
          if (!cur) return s;
          scenes[scenes.length - 1] = { ...cur, closed: true };
          return { scenes, phase: "between_scenes" };
        }

        case "location_change": {
          // The next scene_start carries the new location; nothing to
          // do here other than clear walk_targets (already cleared on
          // scene_start). Treat as informational.
          return s;
        }

        case "log_entry_word": {
          const case_file = [...s.case_file];
          // Find or create the entry for this scene_id.
          let entry = case_file.find((e) => e.scene_id === event.scene_id);
          if (!entry) {
            entry = { scene_id: event.scene_id, words: [], streaming: true };
            case_file.push(entry);
          }
          const idx = case_file.indexOf(entry);
          if (event.is_last) {
            case_file[idx] = { ...entry, streaming: false };
          } else {
            case_file[idx] = { ...entry, words: [...entry.words, event.word] };
          }
          return { case_file };
        }

        case "game_over": {
          return { phase: "game_over", game_over_ending: event.ending_id };
        }

        case "error": {
          return { error: event.message };
        }

        case "session_done": {
          // Server-side play_session ended. If we haven't already moved
          // to game_over (e.g. via a stranded EndScene with no walk),
          // mark stranded.
          if (s.phase !== "game_over") {
            return { phase: "stranded" };
          }
          return s;
        }

        default:
          return s;
      }
    }),
}));
