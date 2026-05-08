// Scenario-graph view (embedded into the Beats tab's `graph` mode).
// Beats are laid out in horizontal lanes by thread, in authoring order.
// Edges come in two flavours: structural (positive `BeatFired` prereq,
// solid + arrowed) and mutex (`Not_(BeatFired)`, dashed) — the same
// upstream reference rendered with different glyphs.
//
// Trimmed-down version of the previous fullscreen-modal component:
// renders inline since the dock now reserves its own viewport column.
// Selection state still local — the Beats tab uses `selection.beat`
// for the predicate-tree drilldown; the graph's selection is purely
// visual highlighting.

import clsx from "clsx";
import { useMemo } from "react";
import type { BeatGraph, BeatMeta, ScenarioMeta, TickDecision } from "../../api";

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

const NODE_W = 92;
const NODE_H = 22;
const NODE_GAP_X = 12;
const LANE_GAP_Y = 14;
const LANE_LABEL_W = 56;
const PADDING = 10;

interface ScenarioGraphProps {
  scenarioMeta: ScenarioMeta;
  latestTick: TickDecision | null;
  selectedBeatId: string | null;
  onSelectBeat: (id: string | null) => void;
}

export function ScenarioGraph({
  scenarioMeta, latestTick, selectedBeatId, onSelectBeat,
}: ScenarioGraphProps) {
  const layout = useMemo(
    () => computeLayout(scenarioMeta.beat_graph, scenarioMeta.beats, {
      eligibleNow: latestTick?.eligible ?? [],
      firedBeatIds: new Set(latestTick?.fired_beat_ids ?? []),
    }),
    [scenarioMeta, latestTick],
  );

  const counts = countByStatus(layout.nodes);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-text-muted">
        <span className="text-[9px] uppercase tracking-wider">
          {layout.nodes.length} beats · {scenarioMeta.beat_graph.edges.length} edges
        </span>
        <span className="text-[9px] text-text-muted/70">
          {counts.eligible} eligible · {counts.blocked} blocked · {counts.fired} fired
        </span>
      </div>
      <Legend />
      <GraphSvg
        layout={layout}
        edges={scenarioMeta.beat_graph.edges}
        selectedId={selectedBeatId}
        onSelect={onSelectBeat}
      />
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
  eligibleNow: { beat_id: string; status: "eligible" | "blocked" | "retired" | "deferred" }[];
  firedBeatIds: Set<string>;
}

function computeLayout(
  graph: BeatGraph,
  beatMeta: Record<string, BeatMeta>,
  status: StatusInputs,
): ComputedLayout {
  const eligibleStatus = new Map<string, "eligible" | "blocked" | "retired" | "deferred">();
  for (const e of status.eligibleNow) eligibleStatus.set(e.beat_id, e.status);

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
      let s: Status = "structural";
      if (eligibleStatus.get(meta.id) === "eligible") s = "eligible";
      else if (eligibleStatus.get(meta.id) === "blocked") s = "blocked";
      else if (status.firedBeatIds.has(meta.id)) s = "fired";
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

function countByStatus(nodes: NodeLayout[]) {
  let eligible = 0, blocked = 0, fired = 0;
  for (const n of nodes) {
    if (n.status === "eligible") eligible++;
    else if (n.status === "blocked") blocked++;
    else if (n.status === "fired") fired++;
  }
  return { eligible, blocked, fired };
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[9px] text-text-muted">
      <LegendSwatch className="bg-presence-active border-presence-active" label="eligible" />
      <LegendSwatch className="bg-meter-warn border-meter-warn" label="blocked" />
      <LegendSwatch className="bg-bg-elevated border-text-muted" label="fired" />
      <LegendSwatch className="bg-bg border-border" label="structural" />
    </div>
  );
}

function LegendSwatch({ className, label }: { className: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span aria-hidden className={clsx("inline-block h-2 w-3 rounded-sm border", className)} />
      <span>{label}</span>
    </span>
  );
}

function GraphSvg({
  layout, edges, selectedId, onSelect,
}: {
  layout: ComputedLayout;
  edges: { from: string; to: string; kind: "structural" | "mutex" }[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
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
      preserveAspectRatio="xMidYMid meet"
      style={{ width: "100%", height: "auto", maxHeight: "60vh", display: "block" }}
    >
      <defs>
        <marker id="arrow-dock" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L5,3 z" fill="currentColor" />
        </marker>
      </defs>
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
      {layout.nodes.map((n) => (
        <NodeRect
          key={n.id}
          node={n}
          selected={selectedId === n.id}
          onSelect={onSelect}
        />
      ))}
    </svg>
  );
}

function NodeRect({
  node, selected, onSelect,
}: { node: NodeLayout; selected: boolean; onSelect: (id: string | null) => void }) {
  const fill = nodeFill(node.status);
  const stroke = nodeStroke(node.status, selected);
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
        {localLabel.length > 12 ? localLabel.slice(0, 11) + "…" : localLabel}
      </text>
      {node.meta.is_scene && (
        <text x={node.x + 3} y={node.y + 7} className="pointer-events-none fill-text-muted font-mono text-[6px]">∞</text>
      )}
      {node.meta.ends_session && (
        <text x={node.x + node.w - 7} y={node.y + 7} className="pointer-events-none fill-meter-hot font-mono text-[6px]">⊣</text>
      )}
    </g>
  );
}

function EdgePath({
  a, b, kind, highlighted,
}: { a: NodeLayout; b: NodeLayout; kind: "structural" | "mutex"; highlighted: boolean }) {
  const x1 = a.x + a.w;
  const y1 = a.y + a.h / 2;
  const x2 = b.x;
  const y2 = b.y + b.h / 2;
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
      markerEnd={kind === "structural" ? "url(#arrow-dock)" : undefined}
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
