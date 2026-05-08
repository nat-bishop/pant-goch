// Pure-prop debug surface: tab strip + active panel. The single seam
// where DebugDock (store-bound) and DebugPreview (fixture-bound) both
// converge. No store calls, no side effects — every input arrives via
// `props`, every change leaves via `props.onSelectionChange`.

import clsx from "clsx";
import { TABS } from "./tabs";
import type { DebugPanelsProps } from "./shared/types";
import { postPlaytestNote } from "../api";
import { useState } from "react";

export function DebugPanels(props: DebugPanelsProps) {
  const active = TABS.find((t) => t.id === props.selection.tab) ?? TABS[0]!;
  const ActiveComponent = active.component;
  return (
    <aside className="flex h-full min-h-0 w-full flex-col bg-bg text-text">
      <header className="border-b border-border px-4 pt-3">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="font-mono text-[10px] uppercase tracking-[0.16em] text-text-muted">
            Debug Menu
          </h1>
          <NoteButton />
        </div>
        <nav className="flex gap-0">
          {TABS.map((t) => {
            const isActive = t.id === active.id;
            const badge = t.badge?.(props);
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => props.onSelectionChange({ tab: t.id })}
                className={clsx(
                  "-mb-px border-b-2 px-3 py-2 text-[12px] font-medium transition-colors",
                  isActive
                    ? "border-accent text-text"
                    : "border-transparent text-text-muted hover:text-text",
                )}
              >
                {t.label}
                {badge != null && (
                  <span className="ml-1 inline-block min-w-[16px] rounded-full bg-bg-elevated px-1.5 py-0.5 text-center font-mono text-[10px] text-text-muted">
                    {badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </header>
      <section className="min-h-0 flex-1 overflow-y-auto p-4">
        <ActiveComponent {...props} />
      </section>
    </aside>
  );
}

// Small playtest-note capture button — keeps the old NotesSection workflow
// alive without consuming a tab slot. Lives in the dock header so it's
// reachable from any tab. (postPlaytestNote is async + isolated; no need
// to thread it through props.)
function NoteButton() {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  async function save() {
    if (!note.trim() || busy) return;
    setBusy(true);
    setFeedback(null);
    try {
      const res = await postPlaytestNote(note.trim());
      setNote("");
      setFeedback(`saved → ${res.path}`);
    } catch (err) {
      setFeedback(err instanceof Error ? err.message : "save failed");
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded border border-border/60 px-2 py-0.5 text-[10px] uppercase tracking-wider text-text-muted hover:text-text"
        title="capture a playtest note"
      >+ note</button>
    );
  }
  return (
    <div className="flex flex-col gap-1">
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        rows={2}
        autoFocus
        placeholder="annotate this moment…"
        className="w-56 rounded border border-border bg-bg p-1 text-[11px] focus:border-accent focus:outline-none"
      />
      <div className="flex items-center justify-end gap-1">
        <button
          type="button"
          onClick={() => { setOpen(false); setNote(""); setFeedback(null); }}
          className="text-[10px] text-text-muted hover:text-text"
        >cancel</button>
        <button
          type="button"
          onClick={save}
          disabled={busy || !note.trim()}
          className={clsx(
            "rounded px-2 py-0.5 text-[10px] uppercase tracking-wider",
            "bg-accent text-text-on-accent disabled:bg-bg-elevated disabled:text-text-muted",
          )}
        >{busy ? "…" : "save"}</button>
      </div>
      {feedback && <div className="text-[9px] text-text-muted">{feedback}</div>}
    </div>
  );
}
