// Single Zustand store. Server is the source of truth; the store is a render
// cache plus a small amount of UI-only state (theme, pending typing indicator,
// first-open replay tracking).

import { create } from "zustand";
import {
  Email,
  GameState,
  getInbox,
  getScenario,
  getThread,
  postOpen,
} from "./api";
import { detectInitialTheme, type Theme } from "./hooks/useTheme";
import { sleep, typingDelayMs } from "./util/pacing";

export type ChatId = "aria";

interface Store {
  bootStarted: boolean;
  bootError: string | null;
  scenario: { title: string; premise: string } | null;
  state: GameState | null;
  view: { chatId: ChatId };
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
  state: null,
  view: { chatId: "aria" },
  chatMessages: {},
  pendingTypingFor: null,
  firstOpenForChat: new Set(),
  theme: detectInitialTheme(),

  async boot() {
    if (get().bootStarted) return;
    set({ bootStarted: true });
    try {
      const [scenario, inbox] = await Promise.all([getScenario(), getInbox()]);
      set({ scenario, state: inbox.state, bootError: null });
      await get().openChat(get().view.chatId);
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

    // Replay unread Aria messages through the typing indicator on first
    // session-open of the chat. The very first interaction begins with
    // "Aria is typing…" instead of the welcome already being there.
    if (wasFirstOpen && chatId === "aria" && unread.length > 0) {
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
