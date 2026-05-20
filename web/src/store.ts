// Zustand store for HANDLER v2. Pure event dispatch — UI reads from
// store, SSE events route through `handleEvent` to mutate. No imperative
// fetching here; api.ts owns HTTP.

import { create } from "zustand";
import type { OptionPayload, SseEvent, WorldMeta } from "./api";

export type Phase =
  | "idle" // pre-start
  | "in_scene" // streaming prose, waiting for options
  | "awaiting_pick" // options shown, waiting on user click
  | "awaiting_free_text" // free-text request shown, waiting on user input
  | "between_scenes" // EndScene happened; case file finishing; pick a walk
  | "game_over"
  | "stranded"; // EndScene at a location with no adjacents (shouldn't happen with Pant Goch)

export interface FreeTextRequest {
  prompt: string;
  suggested: string | null;
  npc_id: string | null;
}

export type WorldMode = "location" | "map" | "notes";

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

/** Active walking interstitial. `fromId`/`toId` are location ids; the
 * MapView pauses the tracer at 75% in `arrive-hold` until scene_start
 * fires, then plays it out in `complete`. See plan §"Walking
 * interstitial: latency-aware". */
export interface WalkingState {
  fromId: string;
  toId: string;
  phase: "depart" | "arrive-hold" | "complete";
}

/** Currently-open gloss popup. Trigger rect is the source DOM rect
 * captured at hover time; popup positions relative to it. */
export interface ActiveGloss {
  glossId: string;
  rect: DOMRect;
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
  /** Set when the choice author asks for typed input; cleared when the
   * player submits or the turn moves on. Drives the FreeTextInput
   * component. */
  freeTextRequest: FreeTextRequest | null;
  game_over_ending: string | null;
  error: string | null;

  /** Static world metadata; loaded once on mount. Null until /world resolves. */
  world: WorldMeta | null;
  /** Live NPC-location map. Seeded from `world.npcs` defaults, mutated
   * by `npc_moved` events. */
  npcLocations: Record<string, string>;

  /** Left-pane mode (location backdrop / map / notes). Forced to "map"
   * during walks. */
  worldMode: WorldMode;
  /** Active walking interstitial state; null when not walking. */
  walking: WalkingState | null;
  /** True once the backend has armed the walk pick (after the chronicler
   * + case-file pacing finish, on the `walk_ready` event). Gates map-pin
   * clickability during `between_scenes` so an early POST /walk can't
   * 409. Cleared on scene_start / location_change. */
  walkReady: boolean;
  /** The chapter index whose body is centred in the right panel.
   * Drives which backdrop the left pane shows when scrolling back. */
  currentChapterIdx: number;
  /** Currently-open gloss popup (or null). */
  activeGloss: ActiveGloss | null;
  /** Click-to-reveal: when true, the StreamingProse renders all
   * words at full opacity, skipping per-word fade-in for the current
   * turn. Reset on each scene_start. */
  fastForward: boolean;

  setWorld: (w: WorldMeta) => void;
  setWorldMode: (m: WorldMode) => void;
  setCurrentChapterIdx: (i: number) => void;
  showGloss: (glossId: string, rect: DOMRect) => void;
  hideGloss: () => void;
  triggerFastForward: () => void;
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
  freeTextRequest: null,
  game_over_ending: null,
  error: null,

  world: null,
  npcLocations: {},
  worldMode: "location",
  walking: null,
  walkReady: false,
  currentChapterIdx: 0,
  activeGloss: null,
  fastForward: false,

  setWorld: (w) => {
    const npcLocations: Record<string, string> = {};
    for (const n of w.npcs) npcLocations[n.id] = n.default_location;
    set({ world: w, npcLocations });
  },

  setWorldMode: (m) => set({ worldMode: m }),
  setCurrentChapterIdx: (i) =>
    set((s) => (s.currentChapterIdx === i ? s : { currentChapterIdx: i })),
  showGloss: (glossId, rect) => set({ activeGloss: { glossId, rect } }),
  hideGloss: () => set({ activeGloss: null }),
  triggerFastForward: () => set({ fastForward: true }),

  resetForRestart: () =>
    set((s) => {
      // Reseed npcLocations from defaults if world is loaded.
      const npcLocations: Record<string, string> = {};
      if (s.world) for (const n of s.world.npcs) npcLocations[n.id] = n.default_location;
      return {
        phase: "idle",
        scenes: [],
        case_file: [],
        freeTextRequest: null,
        game_over_ending: null,
        error: null,
        npcLocations,
        worldMode: "location",
        walking: null,
        walkReady: false,
        currentChapterIdx: 0,
        activeGloss: null,
        fastForward: false,
      };
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
      return { scenes, phase: "in_scene", freeTextRequest: null };
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
          const scenes = [...s.scenes, snapshot];
          // If we were in a walk, advance to the "complete" phase. The
          // MapView's effect snaps tracer to 100% and triggers settle.
          const walking: WalkingState | null = s.walking
            ? { ...s.walking, phase: "complete" }
            : null;
          return {
            phase: "in_scene",
            scenes,
            walking,
            walkReady: false,
            currentChapterIdx: scenes.length - 1,
            fastForward: false,
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

        case "free_text_request": {
          // Close out the current turn's streaming, then surface the
          // typed-input UI. The phase change drives a swap from the
          // ChoicesBlock to a free-text input form.
          const scenes = [...s.scenes];
          const cur = scenes[scenes.length - 1];
          if (cur) {
            const turns = [...cur.turns];
            turns[turns.length - 1] = {
              ...turns[turns.length - 1],
              streaming: false,
            };
            scenes[scenes.length - 1] = { ...cur, turns };
          }
          return {
            scenes,
            phase: "awaiting_free_text",
            freeTextRequest: {
              prompt: event.prompt,
              suggested: event.suggested,
              npc_id: event.npc_id,
            },
          };
        }

        case "scene_exit": {
          // Scene closed. Swap the (now stale) location backdrop for the
          // map so the player sees where they can go next; the walk pick
          // isn't armed yet (walk_ready follows once the case file is
          // written), so pins stay inert until then.
          const scenes = [...s.scenes];
          const cur = scenes[scenes.length - 1];
          if (!cur) return s;
          scenes[scenes.length - 1] = { ...cur, closed: true };
          return {
            scenes,
            phase: "between_scenes",
            worldMode: "map",
            walkReady: false,
          };
        }

        case "walk_ready": {
          // Backend is now awaiting pick_walk — map pins go live.
          return { walkReady: true };
        }

        case "location_change": {
          // Start the walking interstitial. MapView watches `walking`
          // and animates the tracer. scene_start later advances to
          // "complete".
          return {
            walking: { fromId: event.from, toId: event.to, phase: "depart" },
            worldMode: "map",
            walkReady: false,
          };
        }

        case "npc_moved": {
          return {
            npcLocations: { ...s.npcLocations, [event.npc_id]: event.to },
          };
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

/** True only while the live turn is actively revealing words — i.e. the
 * server-generated prose is pacing out (words present + still streaming).
 * Gates click-to-complete and the reveal hint: there is nothing to skip
 * during the pre-word warm-up gap, nor once the turn has closed. */
export function useRevealing(): boolean {
  return useStore((s) => {
    if (s.phase !== "in_scene") return false;
    const scene = s.scenes[s.scenes.length - 1];
    const turn = scene?.turns[scene.turns.length - 1];
    return !!turn && turn.streaming && turn.words.length > 0;
  });
}
