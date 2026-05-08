// Beats tab — answers "why didn't X fire?" by rendering each beat's
// entry_when + exit_when AST with the latest pass/fail trace per atom.
// Two view modes:
//   - list (default, mockup): left list of beats, right detail with
//     predicate trees zipped from /debug/scenario AST + tick_decision
//     beats_evaluation traces.
//   - graph: embedded ScenarioGraph for the structural view.

import clsx from "clsx";
import { useMemo, useState } from "react";
import type { BeatEvaluation, DebugBeat, Predicate, PredicateTrace } from "../../api";
import { PredicateTree } from "../shared/PredicateTree";
import type { PanelProps } from "../shared/types";
import { ScenarioGraph } from "./ScenarioGraph";

export function BeatsPanel(props: PanelProps) {
  const { selection, onSelectionChange, scenarioMeta, debugScenario, tickHistory } = props;
  const latestTick = tickHistory[0] ?? null;
  const evalsById = useMemo(
    () => indexEvaluations(latestTick?.beats_evaluation ?? []),
    [latestTick],
  );

  // First-render default — pick a sensible beat. Prefer a blocked terminal
  // (forces the predicate-tree renderer to actually have something to show);
  // fall back to the first beat in scenario order.
  const defaultBeatId = useMemo(() => {
    if (selection.beat) return selection.beat;
    const beats = Object.values(debugScenario.beats);
    const blockedTerminal = beats.find((b) => b.ends_session && evalsById.get(b.id)?.status === "blocked");
    if (blockedTerminal) return blockedTerminal.id;
    return beats[0]?.id ?? null;
  }, [selection.beat, debugScenario, evalsById]);

  const selectedBeat = defaultBeatId ? debugScenario.beats[defaultBeatId] ?? null : null;
  const selectedEval = defaultBeatId ? evalsById.get(defaultBeatId) ?? null : null;

  return (
    <div className="flex h-full flex-col gap-2">
      <ViewToggle
        view={selection.beatsView}
        onChange={(v) => onSelectionChange({ beatsView: v })}
      />
      {selection.beatsView === "graph" ? (
        <ScenarioGraph
          scenarioMeta={scenarioMeta}
          latestTick={latestTick}
          selectedBeatId={defaultBeatId}
          onSelectBeat={(id) => onSelectionChange({ beat: id })}
        />
      ) : (
        <div className="grid min-h-0 flex-1 grid-cols-[160px_1fr] gap-3">
          <BeatList
            beats={Object.values(debugScenario.beats)}
            evalsById={evalsById}
            selectedId={defaultBeatId}
            onSelect={(id) => onSelectionChange({ beat: id })}
          />
          <BeatDetail beat={selectedBeat} evaluation={selectedEval} />
        </div>
      )}
    </div>
  );
}

function ViewToggle({
  view, onChange,
}: { view: "list" | "graph"; onChange: (v: "list" | "graph") => void }) {
  return (
    <div className="flex gap-0 self-start rounded border border-border text-[11px]">
      {(["list", "graph"] as const).map((v) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          className={clsx(
            "px-2 py-0.5 font-mono uppercase tracking-wider",
            view === v ? "bg-accent text-text-on-accent" : "text-text-muted hover:bg-bg-elevated",
          )}
        >{v}</button>
      ))}
    </div>
  );
}

function BeatList({
  beats, evalsById, selectedId, onSelect,
}: {
  beats: DebugBeat[];
  evalsById: Map<string, BeatEvaluation>;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [filter, setFilter] = useState("");
  const f = filter.trim().toLowerCase();
  const filtered = beats.filter((b) => !f || b.id.toLowerCase().includes(f));
  return (
    <div className="flex min-h-0 flex-col gap-1 border-r border-border/40 pr-2">
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="search…"
        className="w-full rounded border border-border bg-bg px-1.5 py-1 text-[11px] focus:border-accent focus:outline-none"
      />
      <ul className="min-h-0 flex-1 overflow-y-auto">
        {filtered.map((b) => {
          const ev = evalsById.get(b.id);
          const status = ev?.status ?? "blocked";
          return (
            <li key={b.id}>
              <button
                type="button"
                onClick={() => onSelect(b.id)}
                className={clsx(
                  "flex w-full items-center justify-between gap-1 rounded px-1.5 py-1 text-left",
                  "font-mono text-[11px] leading-tight",
                  selectedId === b.id
                    ? "bg-accent/15 text-accent"
                    : "text-text hover:bg-bg-elevated/60",
                )}
              >
                <span className="truncate">{b.id}</span>
                <StatusGlyph status={status} ends_session={b.ends_session} />
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function StatusGlyph({
  status, ends_session,
}: { status: BeatEvaluation["status"]; ends_session: boolean }) {
  const label = status === "fired" ? "✓"
    : status === "eligible" ? "◆"
      : ends_session ? "○"
        : "✕";
  const cls = status === "fired" ? "text-presence-active"
    : status === "eligible" ? "text-accent"
      : ends_session ? "text-text-muted"
        : "text-meter-warn";
  return <span className={clsx("text-[10px]", cls)}>{label}</span>;
}

function BeatDetail({
  beat, evaluation,
}: { beat: DebugBeat | null; evaluation: BeatEvaluation | null }) {
  if (!beat) {
    return <div className="italic text-text-muted/70">select a beat</div>;
  }
  const fireStatus = evaluation?.status ?? "blocked";
  return (
    <div className="min-h-0 overflow-y-auto pr-1">
      <h3 className="font-mono text-[14px] font-semibold text-text">{beat.id}</h3>
      <div className="mb-3 font-mono text-[11px] text-text-muted">
        kind: {beat.kind} · npc: {beat.npc} · thread: {beat.thread} · ends_session: {String(beat.ends_session)}
        <br />
        <FireStatusLabel status={fireStatus} />
      </div>
      {beat.preconditions.length > 0 && (
        <PredicateBlock
          label="entry_when"
          predicates={beat.preconditions}
          traces={evaluation?.entry_traces ?? []}
        />
      )}
      {beat.exit_when.length > 0 && (
        <PredicateBlock
          label="exit_when"
          predicates={beat.exit_when}
          traces={evaluation?.exit_traces ?? []}
        />
      )}
      {beat.preconditions.length === 0 && beat.exit_when.length === 0 && (
        <div className="italic text-text-muted/70">no preconditions or exit_when</div>
      )}
      {beat.effects.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-[9px] uppercase tracking-wider text-text-muted">effects</div>
          <ul className="space-y-1 font-mono text-[11px]">
            {beat.effects.map((e, i) => (
              <li key={i}>
                <span className="text-text">{e.quality}</span>{" "}
                <span className="text-text-muted">{e.op}</span>{" "}
                {e.value !== null
                  ? <span>= {JSON.stringify(e.value)}</span>
                  : <span className="text-text-muted">via classify</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function PredicateBlock({
  label, predicates, traces,
}: { label: string; predicates: Predicate[]; traces: PredicateTrace[] }) {
  return (
    <div className="mb-2">
      <div className="mb-1 text-[9px] uppercase tracking-wider text-text-muted">{label}</div>
      <div className="rounded border border-border/60 bg-bg-elevated/40 p-2 font-mono text-[12px]">
        {predicates.map((p, i) => {
          const trace = traces[i];
          if (!trace) {
            // No trace yet (e.g. boot tick hasn't shipped beats_evaluation
            // — fixtures or fresh-load case). Render the AST with neutral
            // styling so the structure is still legible.
            return (
              <div key={i} className="opacity-50">
                <PredicateTree predicate={p} trace={{ passed: false, label: "(no trace)", children: [] }} />
              </div>
            );
          }
          return <PredicateTree key={i} predicate={p} trace={trace} />;
        })}
      </div>
    </div>
  );
}

function FireStatusLabel({ status }: { status: BeatEvaluation["status"] }) {
  const cls = status === "fired" ? "text-presence-active"
    : status === "eligible" ? "text-accent"
      : "text-meter-warn";
  return <span className={cls}>{status}</span>;
}

function indexEvaluations(evals: BeatEvaluation[]): Map<string, BeatEvaluation> {
  const m = new Map<string, BeatEvaluation>();
  for (const e of evals) m.set(e.beat_id, e);
  return m;
}
