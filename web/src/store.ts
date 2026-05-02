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
  SearchHit,
  getSearch,
} from "./api";

export type View =
  | { kind: "folder"; folder: string }
  | { kind: "thread"; threadId: string; folderReturnTo: string }
  | { kind: "search"; query: string }
  | { kind: "coda" };

interface Store {
  bootError: string | null;
  scenario: { title: string; premise: string; max_actions: number } | null;
  state: GameState | null;
  folders: Record<string, Email[]>;
  view: View;
  threadEmails: Email[] | null;
  searchHits: SearchHit[] | null;

  boot(): Promise<void>;
  refreshInbox(): Promise<void>;
  goFolder(folder: string): void;
  openThread(threadId: string, folderReturnTo: string): Promise<void>;
  goSearch(query: string): Promise<void>;
  goCoda(): void;
  back(): void;

  openEmail(emailId: string): Promise<Email[]>;
  archiveEmail(emailId: string): Promise<void>;
}

export const useStore = create<Store>()((set, get) => ({
  bootError: null,
  scenario: null,
  state: null,
  folders: { Inbox: [], Drafts: [], Sent: [], Archive: [] },
  view: { kind: "folder", folder: "Inbox" },
  threadEmails: null,
  searchHits: null,

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
    set({ view: { kind: "folder", folder }, threadEmails: null, searchHits: null });
  },

  async openThread(threadId, folderReturnTo) {
    const t = await getThread(threadId);
    set({
      view: { kind: "thread", threadId, folderReturnTo },
      threadEmails: t.emails,
    });
  },

  async goSearch(query) {
    if (!query.trim()) {
      set({ view: { kind: "folder", folder: "Inbox" }, searchHits: null });
      return;
    }
    const r = await getSearch(query);
    set({ view: { kind: "search", query }, searchHits: r.hits });
  },

  goCoda() {
    set({ view: { kind: "coda" } });
  },

  back() {
    const { view } = get();
    if (view.kind === "thread") {
      set({ view: { kind: "folder", folder: view.folderReturnTo }, threadEmails: null });
    } else if (view.kind === "search") {
      set({ view: { kind: "folder", folder: "Inbox" }, searchHits: null });
    }
  },

  async openEmail(emailId) {
    const r = await postOpen(emailId);
    await get().refreshInbox();
    const { view } = get();
    if (view.kind === "thread") {
      await get().openThread(view.threadId, view.folderReturnTo);
    }
    return r.new_emails;
  },

  async archiveEmail(emailId) {
    await postArchive(emailId);
    await get().refreshInbox();
  },
}));
