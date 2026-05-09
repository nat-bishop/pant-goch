import { useEffect, useMemo, useState } from "react";
import {
  DebugSessionEvents,
  DebugSessionStateAt,
  SessionEvent,
  getDebugSessionEvents,
  getDebugSessionStateAt,
} from "../../api";
import type { PanelProps } from "../shared/types";

export function TimelinePanel(_: PanelProps) {
  const [events, setEvents] = useState<SessionEvent[] | null>(null);
  const [snap, setSnap] = useState<DebugSessionStateAt | null>(null);
  const [index, setIndex] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setError(null);
    try {
      const log: DebugSessionEvents = await getDebugSessionEvents();
      setEvents(log.events);
      // Snap the cursor to head when the log grows past it.
      setIndex((cur) => Math.min(cur, log.event_count));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const s = await getDebugSessionStateAt(index);
        if (!cancelled) setSnap(s);
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [index]);

  const eventAtCursor = useMemo(() => {
    if (events === null || index === 0) return null;
    return events[index - 1] ?? null;
  }, [events, index]);

  if (error) {
    return <div className="text-[12px] text-red-500">timeline error: {error}</div>;
  }
  if (events === null || snap === null) {
    return <div className="italic text-text-muted/70">loading timeline…</div>;
  }
  const total = events.length;
  return (
    <div className="space-y-3 text-[12px]">
      <div className="flex items-center gap-2">
        <button
          onClick={() => void refresh()}
          className="rounded border border-border px-2 py-0.5 text-[11px] hover:bg-bg-elevated/60"
        >
          refresh
        </button>
        <span className="font-mono text-[11px] text-text-muted">
          event {index} / {total}
        </span>
        <button
          onClick={() => setIndex(total)}
          disabled={index === total}
          className="ml-auto rounded border border-border px-2 py-0.5 text-[11px] disabled:opacity-50 hover:bg-bg-elevated/60"
        >
          jump to head
        </button>
      </div>
      <input
        type="range"
        min={0}
        max={total}
        value={index}
        onChange={(e) => setIndex(Number(e.target.value))}
        className="w-full"
      />
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          disabled={index === 0}
          className="rounded border border-border px-2 py-1 text-[11px] disabled:opacity-50 hover:bg-bg-elevated/60"
        >
          ← prev event
        </button>
        <button
          onClick={() => setIndex((i) => Math.min(total, i + 1))}
          disabled={index === total}
          className="rounded border border-border px-2 py-1 text-[11px] disabled:opacity-50 hover:bg-bg-elevated/60"
        >
          next event →
        </button>
      </div>
      <EventCard event={eventAtCursor} />
      <SnapshotCard snap={snap.state} />
    </div>
  );
}

function EventCard({ event }: { event: SessionEvent | null }) {
  if (event === null) {
    return (
      <section className="rounded border border-border/60 p-2">
        <h3 className="mb-1 font-mono text-[11px] text-text-muted">
          event at cursor
        </h3>
        <div className="italic text-text-muted/70">
          (initial state — no events folded yet)
        </div>
      </section>
    );
  }
  return (
    <section className="rounded border border-border/60 p-2">
      <h3 className="mb-1 font-mono text-[11px] text-text-muted">
        event at cursor — <span className="text-text">{event.kind}</span>
      </h3>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px]">
        {JSON.stringify(event, null, 2)}
      </pre>
    </section>
  );
}

function SnapshotCard({
  snap,
}: {
  snap: DebugSessionStateAt["state"];
}) {
  const fired = snap.fired_beat_ids.length;
  const canon = Object.keys(snap.player_canon).length;
  return (
    <section className="rounded border border-border/60 p-2">
      <h3 className="mb-1 font-mono text-[11px] text-text-muted">
        state at cursor
        {snap.ended ? (
          <span className="ml-2 rounded bg-red-500/20 px-1 text-[10px] text-red-500">
            ENDED — {snap.end_reason ?? "?"}
          </span>
        ) : null}
      </h3>
      <ul className="space-y-0.5 font-mono text-[11px]">
        <li>emails: {snap.emails.length}</li>
        <li>replies: {snap.reply_count}</li>
        <li>fired beats: {fired}</li>
        <li>canon entries: {canon}</li>
      </ul>
      {Object.keys(snap.qualities).length > 0 ? (
        <div className="mt-2">
          <h4 className="font-mono text-[11px] text-text-muted">qualities</h4>
          <ul className="space-y-0.5 font-mono text-[11px]">
            {Object.entries(snap.qualities).map(([k, v]) => (
              <li key={k}>
                {k} = {typeof v === "number" ? v.toFixed(2) : JSON.stringify(v)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {fired > 0 ? (
        <div className="mt-2">
          <h4 className="font-mono text-[11px] text-text-muted">fired_beat_ids</h4>
          <ul className="space-y-0.5 font-mono text-[11px]">
            {snap.fired_beat_ids.map((b) => (
              <li key={b}>{b}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {canon > 0 ? (
        <div className="mt-2">
          <h4 className="font-mono text-[11px] text-text-muted">player_canon</h4>
          <ul className="space-y-0.5 font-mono text-[11px]">
            {Object.entries(snap.player_canon).map(([k, v]) => (
              <li key={k}>
                <span className="text-text-muted">{k}:</span> {v.summary}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
