// Pacing tab — sparkline grid (overview) + per-quality drilldown.
// Reads tickHistory chronologically to build per-quality series, falling
// back to scenarioMeta defaults for qualities never written this session.
//
// The drilldown answers two questions for one quality:
//   1. "Why is it at this value?" — chronological cause attribution
//      (each tick's delta on this quality, with source label).
//   2. "What's gated on it?" — every unfired beat whose entry_when has
//      a numeric atom referencing this quality and currently failing.

import clsx from "clsx";
import { useMemo } from "react";
import type { DebugBeat, Predicate, QualityDelta, QualityMeta, ScenarioMeta, TickDecision } from "../../api";
import type { PanelProps } from "../shared/types";
import { Empty } from "../shared/Empty";
import { QualityName, BeatName } from "../shared/labels";
import { Sparkline } from "../shared/Sparkline";

interface QualitySeries {
  qualityId: string;
  meta: QualityMeta;
  current: number;
  lastDelta: number;
  series: number[];          // chronological (oldest first)
  causes: { tick: number; delta: number; source: string }[];
}

export function PacingPanel(props: PanelProps) {
  const { tickHistory, scenarioMeta, selection, onSelectionChange } = props;
  const numericSeries = useMemo(
    () => buildSeries(tickHistory, scenarioMeta),
    [tickHistory, scenarioMeta],
  );
  const latestQualities = useMemo(
    () => latestQualityValues(tickHistory),
    [tickHistory],
  );

  const drillTarget = selection.quality
    ? numericSeries.find((s) => s.qualityId === selection.quality) ?? null
    : null;

  if (drillTarget) {
    return (
      <PacingDrilldown
        series={drillTarget}
        onBack={() => onSelectionChange({ quality: null })}
        debugBeats={props.debugScenario.beats}
        scenarioMeta={scenarioMeta}
        latestQualities={latestQualities}
      />
    );
  }

  if (numericSeries.length === 0) {
    return <Empty>no qualities authored</Empty>;
  }

  return (
    <div className="flex flex-col">
      {numericSeries.map((s) => (
        <SparkRow
          key={s.qualityId}
          series={s}
          scenarioMeta={scenarioMeta}
          onClick={() => onSelectionChange({ quality: s.qualityId })}
        />
      ))}
    </div>
  );
}

function SparkRow({
  series, scenarioMeta, onClick,
}: {
  series: QualitySeries;
  scenarioMeta: ScenarioMeta;
  onClick: () => void;
}) {
  const range = inferRange(series.meta);
  const flat = series.series.every((v) => v === series.series[0]);
  const dCls = series.lastDelta > 0
    ? "text-presence-active"
    : series.lastDelta < 0
      ? "text-meter-hot"
      : "text-text-muted";
  const dStr = series.lastDelta === 0
    ? "±0"
    : (series.lastDelta > 0 ? "+" : "") + fmtNum(series.lastDelta);
  const arrow = series.lastDelta > 0 ? "↑" : series.lastDelta < 0 ? "↓" : "·";
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "grid grid-cols-[110px_1fr_44px_52px] items-center gap-2",
        "border-b border-border/40 px-1 py-2 text-left",
        "hover:bg-bg-elevated/60",
      )}
    >
      <span className="truncate"><QualityName quality={series.qualityId} scenarioMeta={scenarioMeta} /></span>
      <Sparkline series={series.series} range={range} flat={flat} />
      <span className="text-right font-mono text-[12px] text-text">{fmtNum(series.current)}</span>
      <span className={clsx("text-right font-mono text-[10px]", dCls)}>{dStr} {arrow}</span>
    </button>
  );
}

function PacingDrilldown({
  series, onBack, debugBeats, scenarioMeta, latestQualities,
}: {
  series: QualitySeries;
  onBack: () => void;
  debugBeats: Record<string, DebugBeat>;
  scenarioMeta: ScenarioMeta;
  latestQualities: Record<string, number | string>;
}) {
  const blocked = useMemo(
    () => collectBlockedBeats(series.qualityId, debugBeats, latestQualities),
    [series.qualityId, debugBeats, latestQualities],
  );
  return (
    <div>
      <div className="mb-3 flex items-baseline gap-2 border-b border-border/40 pb-2">
        <button
          type="button"
          onClick={onBack}
          className="text-[11px] text-accent hover:underline"
        >← back</button>
        <h3 className="flex-1 text-[14px] font-semibold text-text">
          <QualityName quality={series.qualityId} scenarioMeta={scenarioMeta} />
        </h3>
        <span className="font-mono text-[12px] text-text-muted">
          current: <span className="text-text">{fmtNum(series.current)}</span>
        </span>
      </div>
      <div className="font-mono text-[12px]">
        {series.causes.length === 0 ? (
          <Empty>no changes recorded yet</Empty>
        ) : (
          series.causes.map((c, i) => (
            <div
              key={i}
              className="grid grid-cols-[36px_50px_1fr] gap-2 border-b border-border/30 py-1"
            >
              <span className="text-text-muted">T{c.tick}</span>
              <span
                className={clsx(
                  "text-right font-mono",
                  c.delta > 0 && "text-presence-active",
                  c.delta < 0 && "text-meter-hot",
                  c.delta === 0 && "text-text-muted",
                )}
              >{(c.delta > 0 ? "+" : "")}{fmtNum(c.delta)}</span>
              <span className="truncate text-text-muted">{c.source}</span>
            </div>
          ))
        )}
      </div>
      <div className="mt-4 border-t border-dashed border-border/50 pt-2">
        <div className="mb-1 text-[9px] uppercase tracking-wider text-text-muted">
          blocked on this quality
        </div>
        {blocked.length === 0 ? (
          <Empty>nothing currently gated on this quality</Empty>
        ) : (
          <ul className="space-y-1 font-mono text-[12px]">
            {blocked.map((b, i) => (
              <li key={i} className="flex items-baseline justify-between gap-2">
                <span className="truncate">
                  <BeatName beatId={b.beatId} scenarioMeta={scenarioMeta} />
                  <span className="ml-2 text-text-muted">{b.threshold}</span>
                </span>
                <span className="text-meter-warn">{b.shortText}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ─── data shaping ────────────────────────────────────────────────────

function buildSeries(
  tickHistory: TickDecision[],
  scenarioMeta: ScenarioMeta,
): QualitySeries[] {
  // tickHistory is newest-first; chronological walk needs reverse.
  const oldestFirst = [...tickHistory].reverse();

  // Seed every numeric quality with its authored default.
  const out: QualitySeries[] = [];
  const seriesById = new Map<string, QualitySeries>();
  for (const [qid, meta] of Object.entries(scenarioMeta.qualities)) {
    if (meta.type === "string") continue;  // sparklines are numeric-only
    const seed = typeof meta.default === "number" ? meta.default : 0;
    const s: QualitySeries = {
      qualityId: qid,
      meta,
      current: seed,
      lastDelta: 0,
      series: [seed],
      causes: [],
    };
    out.push(s);
    seriesById.set(qid, s);
  }

  oldestFirst.forEach((tick, i) => {
    const tickIdx = i + 1;
    const seenThisTick = new Map<string, QualityDelta>();
    for (const d of tick.quality_deltas) {
      seenThisTick.set(d.quality, d);
    }
    for (const s of out) {
      const d = seenThisTick.get(s.qualityId);
      if (d && typeof d.after === "number") {
        s.series.push(d.after);
        const delta = d.after - (typeof d.before === "number" ? d.before : 0);
        s.causes.push({ tick: tickIdx, delta, source: d.source });
        s.current = d.after;
        s.lastDelta = delta;
      } else {
        s.series.push(s.current);
      }
    }
  });

  // If no ticks, lastDelta stays 0 and series is just [seed]; that's fine.
  // Reset lastDelta to reflect just the latest movement (not all-time).
  for (const s of out) {
    if (s.causes.length > 0) s.lastDelta = s.causes[s.causes.length - 1]!.delta;
  }
  return out;
}

function latestQualityValues(tickHistory: TickDecision[]): Record<string, number | string> {
  // Walk newest-first; first delta seen per quality wins (most recent).
  const out: Record<string, number | string> = {};
  for (const t of tickHistory) {
    for (const d of t.quality_deltas) {
      if (!(d.quality in out)) out[d.quality] = d.after;
    }
  }
  return out;
}

function inferRange(meta: QualityMeta): [number, number] {
  // Authoring may omit min/max for unbounded counters. Fall back to a
  // safe synthetic range so the sparkline still renders something
  // sensible (the line just hugs the bottom or top, which is honest).
  const lo = meta.min ?? 0;
  const hi = meta.max ?? Math.max(lo + 1, 10);
  return [lo, hi];
}

function fmtNum(v: number): string {
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2).replace(/\.?0+$/, "");
}

// ─── blocked-on-this-quality computation ─────────────────────────────

interface BlockedRow {
  beatId: string;
  threshold: string;
  shortText: string;
}

function collectBlockedBeats(
  qualityId: string,
  debugBeats: Record<string, DebugBeat>,
  qualities: Record<string, number | string>,
): BlockedRow[] {
  const out: BlockedRow[] = [];
  for (const beat of Object.values(debugBeats)) {
    for (const pred of beat.preconditions) {
      const row = describeQualityAtom(beat.id, pred, qualityId, qualities);
      if (row) out.push(row);
    }
    for (const pred of beat.exit_when) {
      const row = describeQualityAtom(beat.id, pred, qualityId, qualities);
      if (row) {
        // Tag exit-when atoms so author can tell them apart at a glance.
        out.push({ ...row, threshold: `${row.threshold} (exit)` });
      }
    }
  }
  return out;
}

function describeQualityAtom(
  beatId: string,
  pred: Predicate,
  qualityId: string,
  qualities: Record<string, number | string>,
): BlockedRow | null {
  // Walks Or_/Not_ wrappers shallowly so a quality_gte buried inside one
  // arm of an Or_ still surfaces. Doesn't honor full boolean logic — the
  // intent here is "which beats reference this quality"; the predicate
  // tab is where exact verdicts live.
  if (pred.kind === "quality_gte" && pred.quality === qualityId) {
    const cur = numericQuality(qualities[qualityId]);
    if (cur === null || cur < pred.value) {
      const short = cur === null ? pred.value : pred.value - cur;
      return {
        beatId,
        threshold: `≥ ${fmtNum(pred.value)}`,
        shortText: short > 0 ? `${fmtNum(short)} short` : "OK",
      };
    }
    return null;
  }
  if (pred.kind === "quality_lt" && pred.quality === qualityId) {
    const cur = numericQuality(qualities[qualityId]);
    if (cur === null || cur >= pred.value) {
      return {
        beatId,
        threshold: `< ${fmtNum(pred.value)}`,
        shortText: cur === null ? "needs first write" : `at ${fmtNum(cur)}`,
      };
    }
    return null;
  }
  if (pred.kind === "or") {
    for (const sub of pred.of) {
      const inner = describeQualityAtom(beatId, sub, qualityId, qualities);
      if (inner) return inner;
    }
    return null;
  }
  if (pred.kind === "not") {
    return describeQualityAtom(beatId, pred.of, qualityId, qualities);
  }
  return null;
}

function numericQuality(v: number | string | undefined): number | null {
  return typeof v === "number" ? v : null;
}
