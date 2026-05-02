import { useState } from "react";
import { useStore } from "../store";

// Mirrors MAYA_TERMINAL_STATES in spine.py — the set the coda gates on.
const MAYA_TERMINAL_STATES = new Set(["warm", "ambivalent", "cold"]);

export function InboxHeader() {
  const { scenario, state, goSearch, view, goCoda } = useStore();
  const [q, setQ] = useState(view.kind === "search" ? view.query : "");

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    goSearch(q);
  };

  const codaReady =
    state &&
    (state.ended ||
      Object.values(state.thread_states).some((s) => MAYA_TERMINAL_STATES.has(s)));

  const endingNow = state?.ended && !state.coda_emitted;

  return (
    <header className="flex items-center gap-6 border-b border-line bg-paper px-6 py-3">
      <div className="shrink-0 text-base font-bold tracking-tight">
        {scenario?.title}
      </div>

      <form onSubmit={onSubmit} className="flex-1 max-w-2xl">
        <label className="flex items-center gap-2 rounded border border-line bg-paper-hover/60 px-3 py-1.5 focus-within:border-ink">
          <SearchGlyph />
          <input
            type="text"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="search Tess's mail"
            className="w-full bg-transparent text-sm placeholder-muted focus:outline-none"
          />
        </label>
      </form>

      <div className="flex shrink-0 items-center gap-4">
        {state && (
          <span
            className="rounded-full border border-cool/40 bg-cool/5 px-3 py-1 text-[11px] tracking-wide text-cool"
            title={`${state.actions_remaining} remaining`}
          >
            {state.action_count} / {state.max_actions} actions
          </span>
        )}
        {endingNow ? (
          <button
            onClick={goCoda}
            className="rounded border border-warm bg-warm px-3 py-1 text-xs font-bold tracking-wide text-paper hover:bg-warm/90"
          >
            view closing
          </button>
        ) : codaReady ? (
          <button
            onClick={goCoda}
            className="text-xs text-muted hover:text-ink"
            title="end the session and view the closing montage"
          >
            close session
          </button>
        ) : null}
      </div>
    </header>
  );
}

function SearchGlyph() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      className="shrink-0 text-muted"
      aria-hidden
    >
      <circle cx="6" cy="6" r="4.25" />
      <path d="m9 9 3.5 3.5" strokeLinecap="round" />
    </svg>
  );
}
