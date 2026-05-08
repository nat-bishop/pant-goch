// Tick log tab — chronological list (newest first) of every TickDecision
// the dock has accumulated this session. Each row collapses to a one-line
// summary; expand reveals trigger / fired / classifier / selection /
// effects. Replaces the old "Last Tick" overlay section, broadened to the
// full session history (capped at TICK_HISTORY_CAP).

import clsx from "clsx";
import type { TickDecision } from "../../api";
import { BeatName } from "../shared/labels";
import type { PanelProps } from "../shared/types";

export function TickLogPanel({ tickHistory, scenarioMeta }: PanelProps) {
  if (tickHistory.length === 0) {
    return <div className="italic text-text-muted/70">no /send yet</div>;
  }
  return (
    <div className="space-y-1">
      {tickHistory.map((t, i) => (
        <TickRow
          key={i}
          decision={t}
          tickNumber={tickHistory.length - i}
          scenarioMeta={scenarioMeta}
        />
      ))}
    </div>
  );
}

function TickRow({
  decision, tickNumber, scenarioMeta,
}: {
  decision: TickDecision;
  tickNumber: number;
  scenarioMeta: PanelProps["scenarioMeta"];
}) {
  const fired = decision.fired_beat_id;
  const classify = topClassify(decision);
  return (
    <details className="overflow-hidden rounded border border-border/60">
      <summary
        className={clsx(
          "grid cursor-pointer grid-cols-[36px_1fr_auto_auto] items-baseline gap-2",
          "px-2 py-1.5 text-[12px] hover:bg-bg-elevated/60",
          "[&::-webkit-details-marker]:hidden",
        )}
      >
        <span className="font-mono text-[11px] text-text-muted">T{tickNumber}</span>
        <span className="truncate font-mono text-[12px]">
          {fired ? (
            <BeatName beatId={fired} scenarioMeta={scenarioMeta} />
          ) : (
            <span className="italic text-text-muted">no fire</span>
          )}
          {decision.scene_turn != null && fired && (
            <span className="ml-1 text-text-muted">· turn {decision.scene_turn}</span>
          )}
        </span>
        {classify ? (
          <span className="rounded bg-accent/10 px-1.5 py-0.5 font-mono text-[10px] text-accent">
            {classify}
          </span>
        ) : (
          <span className="font-mono text-[10px] text-text-muted">—</span>
        )}
        <span className="font-mono text-[10px] text-text-muted">{decision.selection}</span>
      </summary>
      <TickBody decision={decision} />
    </details>
  );
}

function TickBody({ decision }: { decision: TickDecision }) {
  return (
    <div className="space-y-2 bg-bg-elevated/40 p-2 font-mono text-[11px] leading-relaxed">
      <Block label="trigger">
        {decision.in_response_to ? (
          <>player reply (<span className="text-text-muted">{decision.in_response_to}</span>)</>
        ) : (
          "boot tick"
        )}
      </Block>
      {decision.fired_beat_id && (
        <Block label="fired">
          {decision.fired_beat_id}
          {decision.scene_turn != null && (
            <span className="text-text-muted"> · turn {decision.scene_turn}</span>
          )}
        </Block>
      )}
      <Block label="selection">
        method: {decision.selection}
        {decision.eligible.length > 0 && (
          <div className="text-text-muted">
            eligible: [{decision.eligible.map((e) => e.beat_id).join(", ")}]
          </div>
        )}
        {decision.chosen_rationale && (
          <div className="italic text-text-muted">"{decision.chosen_rationale}"</div>
        )}
      </Block>
      {decision.quality_deltas.length > 0 && (
        <Block label="effects">
          {decision.quality_deltas.map((d, i) => (
            <div key={i}>
              {d.quality}{" "}
              <span className={clsx(
                typeof d.after === "number" && typeof d.before === "number"
                  ? d.after - (d.before ?? 0) > 0 ? "text-presence-active" : "text-meter-hot"
                  : "text-text",
              )}>
                {fmt(d.before)} → {fmt(d.after)}
              </span>{" "}
              <span className="text-text-muted">({d.source})</span>
            </div>
          ))}
        </Block>
      )}
      {Object.keys(decision.classifier_history).length > 0 && (
        <Block label="classifier history">
          {Object.entries(decision.classifier_history).map(([axis, vals]) => (
            <div key={axis}>
              {axis}: <span className="text-text-muted">[{vals.join(" ← ")}]</span>
            </div>
          ))}
        </Block>
      )}
      {(decision.canon_writes.length > 0 || decision.canon_drops.length > 0) && (
        <Block label="canon">
          {decision.canon_writes.map((c, i) => (
            <div key={`w${i}`} className="text-presence-active">+ {c.key} = "{c.summary}" (★{c.salience.toFixed(2)})</div>
          ))}
          {decision.canon_drops.map((d, i) => (
            <div key={`d${i}`} className="text-meter-warn">drop: {d.candidate.key} ({d.reason})</div>
          ))}
        </Block>
      )}
    </div>
  );
}

function Block({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-wider text-text-muted">{label}</div>
      <div>{children}</div>
    </div>
  );
}

function fmt(v: number | string | null): string {
  if (v === null) return "—";
  if (typeof v === "string") return `"${v}"`;
  return Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/\.?0+$/, "");
}

function topClassify(decision: TickDecision): string | null {
  // Surface the most-recent verdict (any axis) in the row header. Useful
  // because the "selection method" pill is often "sole" / "continue" with
  // little signal — the classifier verdict carries the per-tick nuance.
  for (const vals of Object.values(decision.classifier_history)) {
    if (vals.length > 0) return vals[0]!;
  }
  return null;
}
