// DEV-only overlay surfacing engine state per tick. Visibility is gated
// on `import.meta.env.DEV` at the mount site (App.tsx); production
// bundles tree-shake this entire module out.

import clsx from "clsx";
import { useMemo, useState } from "react";
import {
  BeatEligibility,
  CanonCandidate,
  CanonDropRecord,
  FactCheckTrace,
  PlayerCanonEntry,
  QualityDelta,
  postPlaytestNote,
} from "../api";
import { useStore } from "../store";
import { DebugTooltip } from "./DebugTooltip";
import { PredicateTree } from "./PredicateTree";
import { ScenarioGraph } from "./ScenarioGraph";

export function DebugPanel() {
  const lastTickDecision = useStore((s) => s.lastTickDecision);
  const qualityHistory = useStore((s) => s.qualityHistory);
  const collapsed = useStore((s) => s.debugCollapsed);
  const setCollapsed = useStore((s) => s.setDebugCollapsed);
  const playerCanon = lastTickDecision?.player_canon ?? {};
  const lastCanonWrites = lastTickDecision?.canon_writes ?? [];
  const lastCanonDrops = lastTickDecision?.canon_drops ?? [];
  const classifierHistory = lastTickDecision?.classifier_history ?? {};

  const latestDeltaByQuality = useMemo(() => {
    const map = new Map<string, QualityDelta>();
    for (let i = qualityHistory.length - 1; i >= 0; i--) {
      const d = qualityHistory[i]!;
      if (!map.has(d.quality)) map.set(d.quality, d);
    }
    return map;
  }, [qualityHistory]);

  // Current qualities reconstructed from observed deltas — `/state`'s
  // public_view deliberately omits qualities (it's the player-facing
  // view), so the panel seeds itself from the next tick after page load.
  const currentQualities = useMemo(() => {
    const out = new Map<string, number | string>();
    for (const d of qualityHistory) out.set(d.quality, d.after);
    return [...out.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [qualityHistory]);

  const eligibleCounts = useMemo(() => {
    if (!lastTickDecision) return { eligible: 0, blocked: 0 };
    let eligible = 0, blocked = 0;
    for (const e of lastTickDecision.eligible) {
      if (e.status === "eligible") eligible++;
      else blocked++;
    }
    return { eligible, blocked };
  }, [lastTickDecision]);

  return (
    <div
      className={clsx(
        "fixed bottom-3 right-3 z-50 flex max-h-[85vh] flex-col",
        "rounded-md border border-border bg-bg text-text",
        "font-mono text-[11px] leading-snug shadow-lg",
        // Wider when expanded so the new sections (canon ledger, predicate
        // trees, scenario graph) don't have to wrap awkwardly. Narrow
        // collapsed width keeps the bottom-right footprint small.
        collapsed ? "w-[200px]" : "w-[520px]",
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
          <Section
            title="last tick"
            badge={
              lastTickDecision
                ? `${eligibleCounts.eligible} eligible · ${eligibleCounts.blocked} blocked`
                : null
            }
          >
            {lastTickDecision ? <LastTick /> : <Empty>no /send yet</Empty>}
          </Section>

          <Section
            title="qualities"
            badge={currentQualities.length ? `${currentQualities.length}` : null}
          >
            {currentQualities.length === 0 ? (
              <Empty>no quality writes yet this session</Empty>
            ) : (
              <ul className="space-y-0.5">
                {currentQualities.map(([q, value]) => (
                  <QualityRow
                    key={q}
                    quality={q}
                    value={value}
                    delta={latestDeltaByQuality.get(q)}
                  />
                ))}
              </ul>
            )}
          </Section>

          <Section
            title="canon"
            badge={
              Object.keys(playerCanon).length
                ? `${Object.keys(playerCanon).length}`
                : null
            }
          >
            <CanonSection
              canon={playerCanon}
              writes={lastCanonWrites}
              drops={lastCanonDrops}
            />
          </Section>

          <Section
            title="classifier"
            badge={Object.keys(classifierHistory).length || null}
          >
            <ClassifierSection history={classifierHistory} />
          </Section>

          <Section title={`history (${qualityHistory.length})`}>
            {qualityHistory.length === 0 ? (
              <Empty>no changes recorded</Empty>
            ) : (
              <ul className="max-h-[180px] space-y-0.5 overflow-y-auto">
                {[...qualityHistory].reverse().map((d, i) => (
                  <li key={i} className="text-text-muted">
                    <QualityName quality={d.quality} />{" "}
                    {fmt(d.before)} → {fmt(d.after)}{" "}
                    <span className="text-text-muted/70">{d.source}</span>
                  </li>
                ))}
              </ul>
            )}
          </Section>

          <Section title="notes">
            <NotesSection />
          </Section>

          <Section title="scenario graph">
            <ScenarioGraph />
          </Section>
        </div>
      )}
    </div>
  );
}

// ─────────────────────── Last tick ───────────────────────

function LastTick() {
  const decision = useStore((s) => s.lastTickDecision)!;
  return (
    <div className="space-y-1">
      <div className="text-text-muted">
        selection: <span className="text-text">{decision.selection}</span>
        {decision.fired_beat_id && (
          <>
            {" → "}
            <BeatName beatId={decision.fired_beat_id} />
            {decision.scene_turn != null && (
              <span className="text-text-muted"> (turn {decision.scene_turn})</span>
            )}
          </>
        )}
      </div>
      {decision.fact_check && <FactCheckBadge trace={decision.fact_check} />}
      {decision.chosen_rationale && (
        <div className="italic text-text-muted">"{decision.chosen_rationale}"</div>
      )}
      {decision.eligible.length > 0 && (
        <ul className="space-y-0.5">
          {decision.eligible.map((be) => (
            <EligibleRow key={be.beat_id} entry={be} />
          ))}
        </ul>
      )}
    </div>
  );
}

function EligibleRow({ entry }: { entry: BeatEligibility }) {
  const [expanded, setExpanded] = useState(false);
  const isBlocked = entry.status === "blocked";
  const canExpand = isBlocked && entry.predicate_tree && entry.predicate_trace;
  return (
    <li>
      <button
        type="button"
        onClick={() => canExpand && setExpanded(!expanded)}
        disabled={!canExpand}
        className={clsx(
          "flex w-full items-center gap-1 text-left",
          canExpand ? "cursor-pointer hover:text-text" : "cursor-default",
        )}
      >
        <span
          aria-hidden
          className={clsx(
            "inline-block h-1.5 w-1.5 shrink-0 rounded-full",
            isBlocked ? "bg-meter-warn" : "bg-presence-active",
          )}
        />
        <BeatName beatId={entry.beat_id} />
        {entry.is_scene && (
          <span className="text-text-muted/70">[scene]</span>
        )}
        {canExpand && (
          <span className="ml-auto text-text-muted/70">{expanded ? "▾" : "▸"}</span>
        )}
      </button>
      {isBlocked && entry.blocked_by && !expanded && (
        <div className="pl-3 text-text-muted">↳ {entry.blocked_by}</div>
      )}
      {expanded && entry.predicate_tree && entry.predicate_trace && (
        <div className="ml-3 mt-0.5 rounded border border-border/40 bg-bg-elevated/40 p-1.5">
          <PredicateTree predicate={entry.predicate_tree} trace={entry.predicate_trace} />
        </div>
      )}
    </li>
  );
}

function FactCheckBadge({ trace }: { trace: FactCheckTrace }) {
  const last = trace.attempts[trace.attempts.length - 1];
  if (!last) return null;
  const passed = last.status === "passed";
  const totalAttempts = trace.attempts.length;
  return (
    <DebugTooltip
      wide
      content={
        <div className="space-y-1">
          <div className="font-mono">{passed ? "passed" : "refused"} after {totalAttempts} attempt{totalAttempts === 1 ? "" : "s"}</div>
          {trace.attempts.map((a, i) => (
            <div key={i} className="text-text-muted">
              <span className="text-text">attempt {i + 1}:</span> {a.status}
              {a.missing_facts.length > 0 && (
                <div className="pl-3">missing: {a.missing_facts.join(", ")}</div>
              )}
              {a.violations.length > 0 && (
                <div className="pl-3">violations: {a.violations.join(", ")}</div>
              )}
            </div>
          ))}
        </div>
      }
    >
      <span
        className={clsx(
          "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wider",
          passed
            ? "bg-presence-active/15 text-presence-active"
            : "bg-meter-hot/20 text-meter-hot",
        )}
      >
        fact-check {passed ? "✓" : "⚠"} {totalAttempts > 1 && `${totalAttempts}×`}
      </span>
    </DebugTooltip>
  );
}

// ─────────────────────── Qualities ───────────────────────

function QualityRow({
  quality, value, delta,
}: { quality: string; value: number | string; delta: QualityDelta | undefined }) {
  const scenarioMeta = useStore((s) => s.scenarioMeta);
  const meta = scenarioMeta?.qualities[quality];
  return (
    <li className="flex items-baseline justify-between gap-2">
      <span className="flex items-center gap-1 truncate">
        <QualityName quality={quality} />
        {meta?.surface_to_chat && (
          <span
            aria-label="surfaced to chat agent"
            title="surface_to_chat: true"
            className="text-text-muted"
          >📌</span>
        )}
      </span>
      <span className="flex items-baseline gap-1.5">
        <span>{fmt(value)}</span>
        {delta && <DeltaArrow delta={delta} />}
      </span>
    </li>
  );
}

function QualityName({ quality }: { quality: string }) {
  const scenarioMeta = useStore((s) => s.scenarioMeta);
  const meta = scenarioMeta?.qualities[quality];
  if (!meta || !meta.description) {
    return <span className="text-text">{quality}</span>;
  }
  return (
    <DebugTooltip
      wide
      content={
        <div className="space-y-0.5">
          <div className="font-mono text-text">{quality}</div>
          <div className="text-text-muted">type: {meta.type}{meta.min !== null && `, ${meta.min}–${meta.max}`}</div>
          {meta.surface_to_chat && (
            <div className="text-text-muted">surface_to_chat: true</div>
          )}
          <div className="mt-1 whitespace-pre-wrap text-text">{meta.description.trim()}</div>
        </div>
      }
    >
      <span className="text-text underline decoration-dotted decoration-text-muted/40 underline-offset-2">{quality}</span>
    </DebugTooltip>
  );
}

// ─────────────────────── Canon ───────────────────────

function CanonSection({
  canon, writes, drops,
}: {
  canon: Record<string, PlayerCanonEntry>;
  writes: CanonCandidate[];
  drops: CanonDropRecord[];
}) {
  const entries = Object.entries(canon).sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0 && writes.length === 0 && drops.length === 0) {
    return <Empty>no canon yet</Empty>;
  }
  return (
    <div className="space-y-1.5">
      {entries.length > 0 && (
        <ul className="space-y-0.5">
          {entries.map(([key, entry]) => (
            <li key={key} className="space-y-0.5">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-text">{key}</span>
                <span className="text-text-muted/70">★{entry.salience.toFixed(2)}</span>
              </div>
              <div className="pl-2 text-text">"{entry.summary}"</div>
              <div className="pl-2 text-text-muted">
                ↳ <mark className="bg-accent-glow/40 text-text">"{entry.literal_text_quote}"</mark>
              </div>
            </li>
          ))}
        </ul>
      )}
      {(writes.length > 0 || drops.length > 0) && (
        <div className="border-t border-border/40 pt-1.5">
          <div className="mb-0.5 text-[9px] uppercase tracking-wider text-text-muted/80">
            this tick: {writes.length} promoted · {drops.length} dropped
          </div>
          {drops.length > 0 && (
            <ul className="space-y-0.5">
              {drops.map((d, i) => (
                <li key={i} className="text-text-muted">
                  <span className="text-meter-warn">{d.reason}:</span>{" "}
                  {d.candidate.key} — "{d.candidate.summary}"
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────── Classifier ───────────────────────

function ClassifierSection({ history }: { history: Record<string, string[]> }) {
  const axes = Object.entries(history).sort(([a], [b]) => a.localeCompare(b));
  if (axes.length === 0) {
    return <Empty>no classifier verdicts yet</Empty>;
  }
  return (
    <ul className="space-y-1">
      {axes.map(([axis, values]) => (
        <li key={axis}>
          <div className="flex items-baseline gap-2">
            <QualityName quality={axis} />
          </div>
          <div className="ml-2 mt-0.5 flex flex-wrap gap-1">
            {values.map((v, i) => (
              <span
                key={i}
                className={clsx(
                  "rounded px-1.5 py-0.5 text-[9px]",
                  i === 0
                    ? "bg-accent/15 text-text"
                    : "bg-bg-elevated text-text-muted",
                )}
              >
                {v}
              </span>
            ))}
          </div>
        </li>
      ))}
    </ul>
  );
}

// ─────────────────────── Beat name ───────────────────────

function BeatName({ beatId }: { beatId: string }) {
  const scenarioMeta = useStore((s) => s.scenarioMeta);
  const meta = scenarioMeta?.beats[beatId];
  if (!meta) {
    return <span className="text-text">{beatId}</span>;
  }
  return (
    <DebugTooltip
      wide
      content={
        <div className="space-y-0.5">
          <div className="font-mono text-text">{beatId}</div>
          <div className="text-text-muted">npc: <span className="text-text">{meta.npc}</span></div>
          <div className="text-text-muted">thread: <span className="text-text">{meta.thread}</span></div>
          <div className="flex flex-wrap gap-1 text-text-muted">
            {meta.is_scene && <span>scene</span>}
            {meta.ends_session && <span className="text-meter-hot">terminal</span>}
            {meta.pre_authored && <span>pre-authored</span>}
            {meta.validate_facts && <span>fact-checked</span>}
          </div>
          {meta.summary && <div className="mt-1 whitespace-pre-wrap text-text">{meta.summary}</div>}
        </div>
      }
    >
      <span className="text-text underline decoration-dotted decoration-text-muted/40 underline-offset-2">{beatId}</span>
    </DebugTooltip>
  );
}

// ─────────────────────── Notes ───────────────────────

function NotesSection() {
  const count = useStore((s) => s.playtestNoteCount);
  const bump = useStore((s) => s.bumpPlaytestNoteCount);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  async function save() {
    if (!note.trim() || busy) return;
    setBusy(true);
    setFeedback(null);
    try {
      const res = await postPlaytestNote(note.trim());
      bump();
      setNote("");
      setFeedback(`saved → ${res.path}`);
    } catch (err) {
      setFeedback(err instanceof Error ? err.message : "save failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-1">
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="annotate this moment…"
        rows={2}
        className="w-full resize-y rounded border border-border bg-bg-elevated p-1.5 text-[10px] text-text placeholder:text-text-muted/60 focus:border-accent focus:outline-none"
      />
      <div className="flex items-center justify-between">
        <span className="text-text-muted/70">{count} saved this session</span>
        <button
          type="button"
          onClick={save}
          disabled={busy || !note.trim()}
          className={clsx(
            "rounded px-2 py-0.5 text-[10px] uppercase tracking-wider",
            "bg-accent text-text-on-accent disabled:bg-bg-elevated disabled:text-text-muted",
          )}
        >
          {busy ? "…" : "save note"}
        </button>
      </div>
      {feedback && <div className="text-[9px] text-text-muted">{feedback}</div>}
    </div>
  );
}

// ─────────────────────── shared building blocks ───────────────────────

function Section({
  title, badge, children,
}: {
  title: string;
  badge?: string | number | null;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-text-muted/80">
        <span className="text-[9px] uppercase tracking-wider">{title}</span>
        {badge != null && (
          <span className="text-[9px] text-text-muted/60">{badge}</span>
        )}
      </div>
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="italic text-text-muted/70">{children}</div>;
}

function DeltaArrow({ delta }: { delta: QualityDelta }) {
  const before = typeof delta.before === "number" ? delta.before : 0;
  const after = typeof delta.after === "number" ? delta.after : NaN;
  if (typeof delta.after !== "number") {
    return <span className="text-text-muted">→</span>;
  }
  const diff = after - before;
  if (diff === 0) return <span className="text-text-muted">─</span>;
  const sign = diff > 0 ? "▲" : "▼";
  const color = diff > 0 ? "text-presence-active" : "text-meter-hot";
  return (
    <span className={color}>
      {sign} {fmt(Math.abs(diff))}
    </span>
  );
}

function fmt(value: number | string | null): string {
  if (value === null) return "—";
  if (typeof value === "string") return `"${value}"`;
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, "");
}
