// Scenario-graph view for the debug panel. Beats are laid out in
// horizontal lanes by thread, in authoring order (which is the rough
// intended fire order). Edges come in two flavours: structural
// (positive `BeatFired` prereq, solid + arrowed) and mutex
// (`Not_(BeatFired)`, dashed) — they're the same upstream reference
// rendered with different glyphs.

import clsx from "clsx";
import { useMemo, useState } from "react";
import type { BeatGraph, BeatMeta } from "../api";
import { useStore } from "../store";

type Status = "fired" | "eligible" | "blocked" | "structural" | "idle";

interface NodeLayout {
  id: string;
  thread: string;
  meta: BeatMeta;
  x: number;
  y: number;
  w: number;
  h: number;
  status: Status;
}

const NODE_W = 110;
const NODE_H = 22;
const NODE_GAP_X = 14;
const LANE_GAP_Y = 14;
const LANE_LABEL_W = 64;
const PADDING = 12;

export function ScenarioGraph() {
  const scenarioMeta = useStore((s) => s.scenarioMeta);
  const lastTick = useStore((s) => s.lastTickDecision);
  const [expanded, setExpanded] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const layout = useMemo(() => {
    if (!scenarioMeta) return null;
    return computeLayout(scenarioMeta.beat_graph, scenarioMeta.beats, {
      eligibleNow: lastTick?.eligible ?? [],
      firedBeatIds: new Set(lastTick?.fired_beat_ids ?? []),
    });
  }, [scenarioMeta, lastTick]);

  if (!scenarioMeta) {
    return <div className="text-text-muted/70 italic">scenario_meta not loaded</div>;
  }
  if (!layout) return null;

  const view = (
    <GraphSvg
      layout={layout}
      edges={scenarioMeta.beat_graph.edges}
      selectedId={selectedId}
      onSelect={setSelectedId}
    />
  );
  const selectedMeta = selectedId ? scenarioMeta.beats[selectedId] : null;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-text-muted">
        <span className="text-[9px] uppercase tracking-wider">
          {layout.nodes.length} beats · {scenarioMeta.beat_graph.edges.length} edges
        </span>
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="text-[10px] underline-offset-2 hover:underline"
        >
          fullscreen
        </button>
      </div>
      <div className="overflow-x-auto rounded border border-border/50 bg-bg-elevated/40 p-1">
        {view}
      </div>
      {selectedMeta && <NodeDetail meta={selectedMeta} onClose={() => setSelectedId(null)} />}
      {expanded && (
        <FullscreenModal onClose={() => setExpanded(false)}>
          <GraphSvg
            layout={layout}
            edges={scenarioMeta.beat_graph.edges}
            selectedId={selectedId}
            onSelect={setSelectedId}
            scale={1.6}
          />
          {selectedMeta && (
            <div className="mt-3">
              <NodeDetail meta={selectedMeta} onClose={() => setSelectedId(null)} />
            </div>
          )}
        </FullscreenModal>
      )}
    </div>
  );
}

interface ComputedLayout {
  nodes: NodeLayout[];
  byId: Record<string, NodeLayout>;
  width: number;
  height: number;
  threads: string[];
}

interface StatusInputs {
  eligibleNow: { beat_id: string; status: "eligible" | "blocked" }[];
  firedBeatIds: Set<string>;
}

function computeLayout(
  graph: BeatGraph,
  beatMeta: Record<string, BeatMeta>,
  status: StatusInputs,
): ComputedLayout {
  const eligibleStatus = new Map<string, "eligible" | "blocked">();
  for (const e of status.eligibleNow) eligibleStatus.set(e.beat_id, e.status);

  // Discover threads in first-appearance order. Stable for visual
  // consistency: re-renders won't shuffle lanes around.
  const threads: string[] = [];
  const beatsByThread = new Map<string, BeatMeta[]>();
  for (const node of graph.nodes) {
    const meta = beatMeta[node.id];
    if (!meta) continue;
    if (!beatsByThread.has(meta.thread)) {
      beatsByThread.set(meta.thread, []);
      threads.push(meta.thread);
    }
    beatsByThread.get(meta.thread)!.push(meta);
  }

  const nodes: NodeLayout[] = [];
  threads.forEach((thread, threadIdx) => {
    const beats = beatsByThread.get(thread) ?? [];
    beats.forEach((meta, beatIdx) => {
      const x = PADDING + LANE_LABEL_W + beatIdx * (NODE_W + NODE_GAP_X);
      const y = PADDING + threadIdx * (NODE_H + LANE_GAP_Y);
      let s: Status = "idle";
      if (status.firedBeatIds.has(meta.id)) s = "fired";
      else if (eligibleStatus.get(meta.id) === "eligible") s = "eligible";
      else if (eligibleStatus.get(meta.id) === "blocked") s = "blocked";
      else s = "structural";
      nodes.push({
        id: meta.id, thread, meta, x, y, w: NODE_W, h: NODE_H, status: s,
      });
    });
  });

  const maxBeatsInLane = Math.max(...threads.map((t) => beatsByThread.get(t)?.length ?? 0));
  const width =
    PADDING * 2 +
    LANE_LABEL_W +
    maxBeatsInLane * NODE_W +
    Math.max(0, maxBeatsInLane - 1) * NODE_GAP_X;
  const height = PADDING * 2 + threads.length * (NODE_H + LANE_GAP_Y) - LANE_GAP_Y;

  const byId: Record<string, NodeLayout> = {};
  for (const n of nodes) byId[n.id] = n;
  return { nodes, byId, width, height, threads };
}

function GraphSvg({
  layout,
  edges,
  selectedId,
  onSelect,
  scale = 1,
}: {
  layout: ComputedLayout;
  edges: { from: string; to: string; kind: "structural" | "mutex" }[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  scale?: number;
}) {
  // De-duplicate edges by (from,to,kind) — backend may emit duplicates if
  // a beat references the same upstream more than once across nested gates.
  const uniqEdges = useMemo(() => {
    const seen = new Set<string>();
    return edges.filter((e) => {
      const k = `${e.from}|${e.to}|${e.kind}`;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
  }, [edges]);

  return (
    <svg
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      width={layout.width * scale}
      height={layout.height * scale}
      className="block"
    >
      <defs>
        <marker
          id="arrow"
          markerWidth="6"
          markerHeight="6"
          refX="5"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M0,0 L0,6 L5,3 z" fill="currentColor" />
        </marker>
      </defs>
      {/* lane labels */}
      {layout.threads.map((thread, i) => {
        const y = PADDING + i * (NODE_H + LANE_GAP_Y) + NODE_H / 2;
        return (
          <text
            key={thread}
            x={PADDING}
            y={y}
            dominantBaseline="middle"
            className="fill-text-muted font-mono text-[8px] uppercase tracking-wider"
          >
            {thread}
          </text>
        );
      })}
      {/* edges first, so nodes paint on top */}
      {uniqEdges.map((edge, i) => {
        const a = layout.byId[edge.from];
        const b = layout.byId[edge.to];
        if (!a || !b) return null;
        return (
          <EdgePath
            key={i}
            a={a}
            b={b}
            kind={edge.kind}
            highlighted={selectedId === edge.from || selectedId === edge.to}
          />
        );
      })}
      {/* nodes */}
      {layout.nodes.map((n) => (
        <NodeRect key={n.id} node={n} selected={selectedId === n.id} onSelect={onSelect} />
      ))}
    </svg>
  );
}

function NodeRect({
  node, selected, onSelect,
}: { node: NodeLayout; selected: boolean; onSelect: (id: string | null) => void }) {
  const fill = nodeFill(node.status);
  const stroke = nodeStroke(node.status, selected);
  // Strip thread prefix for label compactness — the lane already
  // identifies the thread, repeating it on every node is noise.
  const localLabel = node.id.includes(".") ? node.id.split(".").slice(1).join(".") : node.id;
  return (
    <g
      className="cursor-pointer"
      onClick={(e) => {
        e.stopPropagation();
        onSelect(selected ? null : node.id);
      }}
    >
      <rect
        x={node.x}
        y={node.y}
        width={node.w}
        height={node.h}
        rx={3}
        ry={3}
        fill={fill}
        stroke={stroke}
        strokeWidth={selected ? 1.5 : 0.75}
      />
      <text
        x={node.x + node.w / 2}
        y={node.y + node.h / 2}
        dominantBaseline="middle"
        textAnchor="middle"
        className="pointer-events-none fill-text font-mono text-[8px]"
      >
        {localLabel.length > 14 ? localLabel.slice(0, 13) + "…" : localLabel}
      </text>
      {/* small flag glyphs in the corner */}
      {node.meta.multi_turn && (
        <text
          x={node.x + 3}
          y={node.y + 7}
          className="pointer-events-none fill-text-muted font-mono text-[6px]"
        >∞</text>
      )}
      {node.meta.ends_session && (
        <text
          x={node.x + node.w - 7}
          y={node.y + 7}
          className="pointer-events-none fill-meter-hot font-mono text-[6px]"
        >⊣</text>
      )}
    </g>
  );
}

function EdgePath({
  a, b, kind, highlighted,
}: { a: NodeLayout; b: NodeLayout; kind: "structural" | "mutex"; highlighted: boolean }) {
  // Source = right edge of A; target = left edge of B (with arrow head)
  const x1 = a.x + a.w;
  const y1 = a.y + a.h / 2;
  const x2 = b.x;
  const y2 = b.y + b.h / 2;
  // Cubic curve so cross-lane edges don't slice through nodes.
  const dx = Math.max(20, (x2 - x1) / 2);
  const d = `M ${x1},${y1} C ${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
  const color = kind === "mutex" ? "var(--meter-warn)" : "var(--text-muted)";
  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth={highlighted ? 1.5 : 0.75}
      strokeDasharray={kind === "mutex" ? "3 2" : undefined}
      style={{ color }}
      markerEnd={kind === "structural" ? "url(#arrow)" : undefined}
      opacity={highlighted ? 1 : 0.55}
    />
  );
}

function nodeFill(status: Status): string {
  switch (status) {
    case "fired": return "var(--bg-elevated)";
    case "eligible": return "var(--presence-active)";
    case "blocked": return "var(--meter-warn)";
    case "structural": return "var(--bg)";
    default: return "var(--bg)";
  }
}

function nodeStroke(status: Status, selected: boolean): string {
  if (selected) return "var(--accent)";
  switch (status) {
    case "fired": return "var(--text-muted)";
    case "eligible": return "var(--presence-active)";
    case "blocked": return "var(--meter-warn)";
    default: return "var(--border)";
  }
}

function NodeDetail({ meta, onClose }: { meta: BeatMeta; onClose: () => void }) {
  return (
    <div className="rounded border border-border/50 bg-bg-elevated/60 p-2 text-[10px]">
      <div className="flex items-center justify-between">
        <span className="font-mono text-text">{meta.id}</span>
        <button
          type="button"
          onClick={onClose}
          className="text-text-muted hover:text-text"
          aria-label="close"
        >×</button>
      </div>
      <div className="mt-1 space-y-0.5 text-text-muted">
        <div>npc: <span className="text-text">{meta.npc}</span></div>
        <div>thread: <span className="text-text">{meta.thread}</span></div>
        <div className="flex flex-wrap gap-1">
          {meta.multi_turn && <Pill>multi-turn</Pill>}
          {meta.ends_session && <Pill tone="hot">terminal</Pill>}
          {meta.pre_authored && <Pill>pre-authored</Pill>}
          {meta.validate_facts && <Pill>fact-checked</Pill>}
        </div>
        {meta.summary && <div className="mt-1 text-text">{meta.summary}</div>}
      </div>
    </div>
  );
}

function Pill({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "hot" }) {
  return (
    <span
      className={clsx(
        "rounded px-1.5 py-0.5 text-[9px] uppercase tracking-wider",
        tone === "hot"
          ? "bg-meter-hot/20 text-meter-hot"
          : "bg-bg/60 text-text-muted",
      )}
    >
      {children}
    </span>
  );
}

function FullscreenModal({
  children, onClose,
}: { children: React.ReactNode; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="max-h-[90vh] max-w-[90vw] overflow-auto rounded-lg border border-border bg-bg p-4 shadow-xl"
      >
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-wider text-text-muted">scenario graph</span>
          <button
            type="button"
            onClick={onClose}
            className="text-text-muted hover:text-text"
            aria-label="close"
          >×</button>
        </div>
        {children}
      </div>
    </div>
  );
}
