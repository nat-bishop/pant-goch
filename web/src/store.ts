// Single Zustand store. Server is the source of truth; the store is a render
// cache plus a small amount of UI-only state (theme, pending typing indicator,
// first-open replay tracking).

import { create } from "zustand";
import {
  DebugScenario,
  Email,
  GameState,
  NpcInfo,
  ScenarioMeta,
  ScenarioResponse,
  TickDecision,
  getDebugScenario,
  getInbox,
  getScenario,
  getScenarioMeta,
  getThread,
  postOpen,
} from "./api";
import { detectInitialTheme, type Theme } from "./hooks/useTheme";
import { sleep, typingDelayMs } from "./util/pacing";

// Thread id; backend is authoritative. Was a `"aria"` literal pre-multi-NPC.
export type ChatId = string;

// Cap on how many TickDecisions the debug dock keeps in memory. Lost on page
// reload — matches SPEC §1 "no persistence across server restarts" non-goal
// and the existing pattern of session-only debug state.
const TICK_HISTORY_CAP = 100;

export type DebugTab = "pacing" | "ticks" | "canon" | "beats";
export type BeatsView = "list" | "graph";

export interface DebugSelection {
  tab: DebugTab;
  quality: string | null;  // pacing drilldown
  beat: string | null;     // beats inspector
  beatsView: BeatsView;    // beats tab sub-toggle
}

const LS_KEY_TAB = "debug.tab";
const LS_KEY_BEATS_VIEW = "debug.beatsView";

function loadSelection(): DebugSelection {
  let tab: DebugTab = "pacing";
  let beatsView: BeatsView = "list";
  try {
    const t = localStorage.getItem(LS_KEY_TAB);
    if (t === "pacing" || t === "ticks" || t === "canon" || t === "beats") tab = t;
    const v = localStorage.getItem(LS_KEY_BEATS_VIEW);
    if (v === "list" || v === "graph") beatsView = v;
  } catch {
    // localStorage may throw in private-browsing; defaults are fine.
  }
  return { tab, beatsView, quality: null, beat: null };
}

interface Store {
  bootStarted: boolean;
  bootError: string | null;
  scenario: { title: string; premise: string } | null;
  npcs: NpcInfo[];
  state: GameState | null;
  view: { chatId: ChatId | null };
  chatMessages: Record<string, Email[]>;
  pendingTypingFor: string | null;
  firstOpenForChat: Set<string>;
  theme: Theme;

  // Debug overlay (DEV-only consumer; backend always emits the source data).
  // `tickHistory` is the full session's tick log capped at TICK_HISTORY_CAP,
  // newest first; consumers read tickHistory[0] for the latest snapshot.
  scenarioMeta: ScenarioMeta | null;
  debugScenario: DebugScenario | null;
  tickHistory: TickDecision[];
  debugSelection: DebugSelection;
  playtestNoteCount: number;

  boot(): Promise<void>;
  refreshInbox(): Promise<void>;
  openChat(chatId: ChatId): Promise<void>;
  appendEmailOptimistic(email: Email): void;
  setPendingTypingFor(threadId: string | null): void;
  setTheme(theme: Theme): void;
  recordTickDecision(decision: TickDecision): void;
  setDebugSelection(patch: Partial<DebugSelection>): void;
  bumpPlaytestNoteCount(): void;
}

export const useStore = create<Store>()((set, get) => ({
  bootStarted: false,
  bootError: null,
  scenario: null,
  npcs: [],
  state: null,
  view: { chatId: null },
  chatMessages: {},
  pendingTypingFor: null,
  firstOpenForChat: new Set(),
  theme: detectInitialTheme(),
  scenarioMeta: null,
  debugScenario: null,
  tickHistory: [],
  debugSelection: loadSelection(),
  playtestNoteCount: 0,

  async boot() {
    // React StrictMode double-mounts in dev; without this guard, boot() runs
    // twice concurrently and the welcome typing-replay fires twice. Set
    // synchronously before any await so the second call sees it.
    if (get().bootStarted) return;
    set({ bootStarted: true });
    try {
      // scenario_meta + debug/scenario run in parallel with the other boot
      // fetches. Both DEV-only — production code never reads them.
      const [scenario, inbox, scenarioMeta, debugScenario] = await Promise.all([
        getScenario(),
        getInbox(),
        import.meta.env.DEV ? getScenarioMeta() : Promise.resolve(null),
        import.meta.env.DEV ? getDebugScenario() : Promise.resolve(null),
      ]);
      const sc = scenario as ScenarioResponse;
      set({
        scenario: { title: sc.title, premise: sc.premise },
        npcs: sc.npcs,
        state: inbox.state,
        scenarioMeta,
        debugScenario,
        bootError: null,
      });
      const initialChat = sc.npcs[0]?.thread_id ?? null;
      if (initialChat) {
        set({ view: { chatId: initialChat } });
        await get().openChat(initialChat);
      }
    } catch (err) {
      set({ bootError: err instanceof Error ? err.message : String(err) });
    }
  },

  async refreshInbox() {
    const inbox = await getInbox();
    set({ state: inbox.state });
  },

  async openChat(chatId) {
    const wasFirstOpen = !get().firstOpenForChat.has(chatId);
    set({
      view: { chatId },
      firstOpenForChat: new Set([...get().firstOpenForChat, chatId]),
    });

    const t = await getThread(chatId);
    const unread = t.emails.filter(
      (e) => !e.is_read && e.folder === "Inbox" && e.from !== "you",
    );
    const unreadIds = new Set(unread.map((e) => e.id));
    const seen = t.emails.filter((e) => !unreadIds.has(e.id));

    // First session-open of this chat: replay unread NPC messages through
    // the typing indicator. Applies to any NPC's first arrival, not just
    // Aria — a second NPC whose first beat fires mid-game also type-replays
    // when the player first opens that thread.
    if (wasFirstOpen && unread.length > 0) {
      set({
        chatMessages: { ...get().chatMessages, [chatId]: seen },
      });
      for (const email of unread) {
        set({ pendingTypingFor: chatId });
        await sleep(typingDelayMs(email.body ?? ""));
        const current = get().chatMessages[chatId] ?? [];
        set({
          chatMessages: { ...get().chatMessages, [chatId]: [...current, email] },
          pendingTypingFor: null,
        });
        await postOpen(email.id);
      }
      await get().refreshInbox();
      return;
    }

    set({ chatMessages: { ...get().chatMessages, [chatId]: t.emails } });
    if (unread.length > 0) {
      await Promise.all(unread.map((e) => postOpen(e.id)));
      await get().refreshInbox();
    }
  },

  appendEmailOptimistic(email) {
    const threadId = email.thread_id ?? email.folder;
    const current = get().chatMessages[threadId] ?? [];
    set({
      chatMessages: { ...get().chatMessages, [threadId]: [...current, email] },
    });
  },

  setPendingTypingFor(threadId) {
    set({ pendingTypingFor: threadId });
  },

  setTheme(theme) {
    set({ theme });
  },

  recordTickDecision(decision) {
    const next = [decision, ...get().tickHistory].slice(0, TICK_HISTORY_CAP);
    set({ tickHistory: next });
  },

  setDebugSelection(patch) {
    const prev = get().debugSelection;
    const next = { ...prev, ...patch };
    set({ debugSelection: next });
    try {
      // Skip persistence when the patch didn't actually change the value —
      // tab clicks repeat the current tab as users orient themselves and
      // each call would otherwise issue a localStorage write.
      if (patch.tab !== undefined && patch.tab !== prev.tab) {
        localStorage.setItem(LS_KEY_TAB, next.tab);
      }
      if (patch.beatsView !== undefined && patch.beatsView !== prev.beatsView) {
        localStorage.setItem(LS_KEY_BEATS_VIEW, next.beatsView);
      }
    } catch {
      // localStorage may throw in private-browsing; the in-memory selection
      // still updates so the dock keeps working without persistence.
    }
  },

  bumpPlaytestNoteCount() {
    set({ playtestNoteCount: get().playtestNoteCount + 1 });
  },
}));
