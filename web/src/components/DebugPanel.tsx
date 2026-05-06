// DEV-only overlay surfacing the engine's per-tick decisions inline with
// the game. Fed by the `tick_decision` SSE event recorded into the store
// from `useStreamSend`. Visibility is gated on `import.meta.env.DEV` at
// the mount site (App.tsx) — no env var, no URL param, no keyboard
// shortcut. Production bundles tree-shake this entire module.
//
// Three sections:
//   QUALITIES — current values + the most recent delta arrow per quality
//   LAST TICK — eligibility, selection, fired beat, blocked-by reasons
//   HISTORY   — scrollable list of every quality change across recent ticks
//
// Read-only inspection. No live mutation, no time-travel scrub — the
// panel exists to answer "what just happened?" not "what if?".

import clsx from "clsx";
import { useMemo } from "react";
import { QualityDelta } from "../api";
import { useStore } from "../store";

export function DebugPanel() {
  const lastTickDecision = useStore((s) => s.lastTickDecision);
  const qualityHistory = useStore((s) => s.qualityHistory);
  const collapsed = useStore((s) => s.debugCollapsed);
  const setCollapsed = useStore((s) => s.setDebugCollapsed);

  // Latest delta per quality, used by the QUALITIES section to show
  // "engagement 3.5 ▲ +1.0" with the most recent change inline. Walking
  // the history backwards is fine — it's capped at QUALITY_HISTORY_CAP.
  const latestDeltaByQuality = useMemo(() => {
    const map = new Map<string, QualityDelta>();
    for (let i = qualityHistory.length - 1; i >= 0; i--) {
      const d = qualityHistory[i]!;
      if (!map.has(d.quality)) map.set(d.quality, d);
    }
    return map;
  }, [qualityHistory]);

  // Current qualities are the union of any quality we've ever seen a delta
  // for. The backend's /state endpoint exposes a public_view that omits
  // qualities (deliberately — it's the player-facing view), so we
  // reconstruct from the deltas we've observed. After a page reload the
  // qualities list seeds from the next tick.
  const currentQualities = useMemo(() => {
    const out = new Map<string, number>();
    for (const d of qualityHistory) out.set(d.quality, d.after);
    return [...out.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [qualityHistory]);

  return (
    <div
      className={clsx(
        "fixed bottom-3 right-3 z-50 w-[320px] max-h-[80vh] flex flex-col",
        "rounded-md border border-border bg-bg text-text",
        "font-mono text-[11px] leading-snug shadow-lg",
      )}
    >
      <button
        type="button"
        onClick={() => setCollapsed(!collapsed)}
        className={clsx(
          "flex items-center justify-between px-2 py-1.5",
          "border-b border-border/60 bg-bg-elevated",
          "text-[10px] uppercase tracking-wider text-text-muted",
          "hover:text-text",
        )}
        aria-expanded={!collapsed}
      >
        <span>debug</span>
        <span aria-hidden>{collapsed ? "▸" : "▾"}</span>
      </button>
      {!collapsed && (
        <div className="overflow-y-auto p-2 space-y-3">
          <Section title="qualities">
            {currentQualities.length === 0 ? (
              <Empty>no quality writes yet this session</Empty>
            ) : (
              <ul className="space-y-0.5">
                {currentQualities.map(([q, value]) => {
                  const delta = latestDeltaByQuality.get(q);
                  return (
                    <li key={q} className="flex items-baseline justify-between gap-2">
                      <span className="truncate">{q}</span>
                      <span className="flex items-baseline gap-1.5">
                        <span>{fmt(value)}</span>
                        {delta && <DeltaArrow delta={delta} />}
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </Section>

          <Section title="last tick">
            {lastTickDecision ? (
              <LastTick />
            ) : (
              <Empty>no /send yet</Empty>
            )}
          </Section>

          <Section title={`history (${qualityHistory.length})`}>
            {qualityHistory.length === 0 ? (
              <Empty>no changes recorded</Empty>
            ) : (
              <ul className="space-y-0.5 max-h-[180px] overflow-y-auto">
                {[...qualityHistory].reverse().map((d, i) => (
                  <li key={i} className="text-text-muted">
                    <span className="text-text">{d.quality}</span>{" "}
                    {fmt(d.before)} → {fmt(d.after)}{" "}
                    <span className="text-text-muted/70">{d.source}</span>
                  </li>
                ))}
              </ul>
            )}
          </Section>
        </div>
      )}
    </div>
  );
}

function LastTick() {
  const decision = useStore((s) => s.lastTickDecision)!;

  return (
    <div className="space-y-1">
      <div className="text-text-muted">
        selection: <span className="text-text">{decision.selection}</span>
        {decision.fired_beat_id && (
          <>
            {" → "}
            <span className="text-text">{decision.fired_beat_id}</span>
            {decision.multi_turn_turn != null && (
              <span className="text-text-muted"> (turn {decision.multi_turn_turn})</span>
            )}
          </>
        )}
      </div>
      {decision.chosen_rationale && (
        <div className="text-text-muted italic">"{decision.chosen_rationale}"</div>
      )}
      {decision.eligible.length > 0 && (
        <ul className="space-y-0.5">
          {decision.eligible.map((be) => (
            <li
              key={be.beat_id}
              className={clsx(
                be.status === "eligible" ? "text-text" : "text-text-muted",
              )}
            >
              <span>{be.beat_id}</span>
              {be.multi_turn && <span className="text-text-muted/70"> [multi]</span>}
              {be.status === "blocked" && be.blocked_by && (
                <div className="pl-3 text-text-muted">↳ {be.blocked_by}</div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1 text-[9px] uppercase tracking-wider text-text-muted/80">
        {title}
      </div>
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="text-text-muted/70 italic">{children}</div>;
}

function DeltaArrow({ delta }: { delta: QualityDelta }) {
  const before = delta.before ?? 0;
  const diff = delta.after - before;
  if (diff === 0) {
    return <span className="text-text-muted">─</span>;
  }
  const sign = diff > 0 ? "▲" : "▼";
  const color = diff > 0 ? "text-presence-active" : "text-meter-hot";
  return (
    <span className={color}>
      {sign} {fmt(Math.abs(diff))}
    </span>
  );
}

function fmt(value: number | null): string {
  if (value === null) return "—";
  // Trim trailing zeros (3.50 → 3.5) but keep enough precision to show
  // sub-integer accumulator deltas (engagement +0.5).
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, "");
}
