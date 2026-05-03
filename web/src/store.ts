// Single Zustand store mirroring server state. Don't try to be clever — the
// server is the source of truth, the store is a render cache. After every
// mutating call we reload /inbox.

import { create } from "zustand";
import {
  Email,
  GameState,
  getInbox,
  getScenario,
  getThread,
  postArchive,
  postOpen,
  postSleep,
  postWait,
} from "./api";

export type View =
  | { kind: "folder"; folder: string }
  | { kind: "thread"; threadId: string; folderReturnTo: string };

interface Store {
  bootError: string | null;
  scenario: { title: string; premise: string; max_hours: number } | null;
  state: GameState | null;
  folders: Record<string, Email[]>;
  view: View;
  threadEmails: Email[] | null;

  boot(): Promise<void>;
  refreshInbox(): Promise<void>;
  goFolder(folder: string): void;
  openThread(threadId: string, folderReturnTo: string): Promise<void>;
  back(): void;

  archiveEmail(emailId: string): Promise<void>;
  waitHours(): Promise<Email[]>;
  sleep(): Promise<Email[]>;
  _afterMutation(): Promise<void>;
}

export const useStore = create<Store>()((set, get) => ({
  bootError: null,
  scenario: null,
  state: null,
  folders: { Inbox: [], Sent: [], Archive: [] },
  view: { kind: "folder", folder: "Inbox" },
  threadEmails: null,

  async boot() {
    try {
      const [scenario, inbox] = await Promise.all([getScenario(), getInbox()]);
      set({
        scenario,
        state: inbox.state,
        folders: inbox.folders,
        bootError: null,
      });
    } catch (err) {
      set({ bootError: err instanceof Error ? err.message : String(err) });
    }
  },

  async refreshInbox() {
    const inbox = await getInbox();
    set({ folders: inbox.folders, state: inbox.state });
  },

  goFolder(folder) {
    set({ view: { kind: "folder", folder }, threadEmails: null });
  },

  async openThread(threadId, folderReturnTo) {
    const t = await getThread(threadId);
    set({
      view: { kind: "thread", threadId, folderReturnTo },
      threadEmails: t.emails,
    });
    const unreadIds = t.emails
      .filter((e) => !e.is_read && e.folder === "Inbox")
      .map((e) => e.id);
    if (unreadIds.length > 0) {
      await Promise.all(unreadIds.map((id) => postOpen(id)));
      await get().refreshInbox();
      set({
        threadEmails: t.emails.map((e) =>
          e.folder === "Inbox" ? { ...e, is_read: true } : e,
        ),
      });
    }
  },

  back() {
    const { view } = get();
    if (view.kind === "thread") {
      set({ view: { kind: "folder", folder: view.folderReturnTo }, threadEmails: null });
    }
  },

  async archiveEmail(emailId) {
    await postArchive(emailId);
    await get()._afterMutation();
  },

  async waitHours() {
    const r = await postWait();
    await get()._afterMutation();
    return r.new_emails;
  },

  async sleep() {
    const r = await postSleep();
    await get()._afterMutation();
    return r.new_emails;
  },

  async _afterMutation() {
    await get().refreshInbox();
    const { view } = get();
    if (view.kind === "thread") {
      await get().openThread(view.threadId, view.folderReturnTo);
    }
  },
}));
