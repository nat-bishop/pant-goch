import { useState } from "react";
import { useStore } from "../store";
import { formatGameTime } from "../util/email";

export function InboxHeader() {
  const { scenario, state, waitHours, sleep } = useStore();
  const [busy, setBusy] = useState(false);

  const guard = (fn: () => Promise<unknown>) => async () => {
    if (busy) return;
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  const ended = state?.ended;

  return (
    <header className="flex items-center gap-6 border-b border-line bg-paper px-6 py-3">
      <div className="shrink-0 text-base font-bold tracking-tight">
        {scenario?.title}
      </div>

      <div className="flex-1" />

      <div className="flex shrink-0 items-center gap-3">
        {state && (
          <span
            className="rounded-full border border-cool/40 bg-cool/5 px-3 py-1 text-[11px] tracking-wide text-cool"
            title={`${state.game_time_hours.toFixed(1)} of ${state.max_hours} hours`}
          >
            {formatGameTime(state.game_time_hours)}
          </span>
        )}

        <button
          onClick={guard(waitHours)}
          disabled={busy || ended}
          className="rounded border border-line bg-paper px-3 py-1 text-xs tracking-wide text-ink-soft hover:border-ink hover:text-ink disabled:opacity-50"
          title="advance the clock by 2 hours"
        >
          wait 2h
        </button>

        <button
          onClick={guard(sleep)}
          disabled={busy || ended}
          className="rounded border border-line bg-paper px-3 py-1 text-xs tracking-wide text-ink-soft hover:border-ink hover:text-ink disabled:opacity-50"
          title="advance the clock by 8 hours (overnight)"
        >
          sleep 8h
        </button>

        {ended && (
          <span className="rounded-full border border-warm/40 bg-warm/5 px-3 py-1 text-[11px] tracking-wide text-warm">
            session ended
          </span>
        )}
      </div>
    </header>
  );
}
