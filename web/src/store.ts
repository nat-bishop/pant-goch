// Single Zustand store. Server is the source of truth; the store is a render
// cache plus a small amount of UI-only state (theme, pending typing indicator,
// first-open replay tracking).

import { create } from "zustand";
import {
  Email,
  GameState,
  NpcInfo,
  QualityDelta,
  ScenarioMeta,
  ScenarioResponse,
  TickDecision,
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

// Debug overlay keeps a rolling history of quality changes across ticks
// so the author can scan "what moved engagement up over the last few
// turns" without trawling the terminal log. Cap is small — the panel is
// scannable, not an audit log.
const QUALITY_HISTORY_CAP = 50;

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
  // The panel reads canon / classifier / fired-beat snapshots directly
  // off `lastTickDecision`; only `qualityHistory` is accumulated locally
  // (capped) since the decision payload only carries the per-tick deltas.
  scenarioMeta: ScenarioMeta | null;
  lastTickDecision: TickDecision | null;
  qualityHistory: QualityDelta[];
  playtestNoteCount: number;
  debugCollapsed: boolean;

  boot(): Promise<void>;
  refreshInbox(): Promise<void>;
  openChat(chatId: ChatId): Promise<void>;
  appendEmailOptimistic(email: Email): void;
  setPendingTypingFor(threadId: string | null): void;
  setTheme(theme: Theme): void;
  recordTickDecision(decision: TickDecision): void;
  setDebugCollapsed(collapsed: boolean): void;
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
  lastTickDecision: null,
  qualityHistory: [],
  playtestNoteCount: 0,
  debugCollapsed: false,

  async boot() {
    // React StrictMode double-mounts in dev; without this guard, boot() runs
    // twice concurrently and the welcome typing-replay fires twice. Set
    // synchronously before any await so the second call sees it.
    if (get().bootStarted) return;
    set({ bootStarted: true });
    try {
      // scenario_meta runs in parallel with the other boot fetches — it's
      // a small payload and there's no dependency. DEV-only consumer
      // (debug panel); production code never reads `scenarioMeta`.
      const [scenario, inbox, scenarioMeta] = await Promise.all([
        getScenario(),
        getInbox(),
        import.meta.env.DEV ? getScenarioMeta() : Promise.resolve(null),
      ]);
      const sc = scenario as ScenarioResponse;
      set({
        scenario: { title: sc.title, premise: sc.premise },
        npcs: sc.npcs,
        state: inbox.state,
        scenarioMeta,
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
    const merged = [...get().qualityHistory, ...decision.quality_deltas];
    set({
      lastTickDecision: decision,
      qualityHistory: merged.slice(-QUALITY_HISTORY_CAP),
    });
  },

  setDebugCollapsed(collapsed) {
    set({ debugCollapsed: collapsed });
  },

  bumpPlaytestNoteCount() {
    set({ playtestNoteCount: get().playtestNoteCount + 1 });
  },
}));
