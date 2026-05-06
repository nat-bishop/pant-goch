// Single Zustand store. Server is the source of truth; the store is a render
// cache plus a small amount of UI-only state (theme, pending typing indicator,
// first-open replay tracking).

import { create } from "zustand";
import {
  Email,
  GameState,
  NpcInfo,
  ScenarioResponse,
  getInbox,
  getScenario,
  getThread,
  postOpen,
} from "./api";
import { detectInitialTheme, type Theme } from "./hooks/useTheme";
import { sleep, typingDelayMs } from "./util/pacing";

// Thread id; backend is authoritative. Was a `"aria"` literal pre-multi-NPC.
export type ChatId = string;

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

  boot(): Promise<void>;
  refreshInbox(): Promise<void>;
  openChat(chatId: ChatId): Promise<void>;
  appendEmailOptimistic(email: Email): void;
  setPendingTypingFor(threadId: string | null): void;
  setTheme(theme: Theme): void;
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

  async boot() {
    // React StrictMode double-mounts in dev; without this guard, boot() runs
    // twice concurrently and the welcome typing-replay fires twice. Set
    // synchronously before any await so the second call sees it.
    if (get().bootStarted) return;
    set({ bootStarted: true });
    try {
      const [scenario, inbox] = await Promise.all([getScenario(), getInbox()]);
      const sc = scenario as ScenarioResponse;
      set({
        scenario: { title: sc.title, premise: sc.premise },
        npcs: sc.npcs,
        state: inbox.state,
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
}));
