// Canon tab — current player_canon ledger plus a session-cumulative drops
// list. Search filters on key, value, or quote substring.
//
// The ledger reads off `tickHistory[0].player_canon` (the latest snapshot
// the engine emitted). Drops are accumulated across the last N=20 player
// turns by walking tickHistory; this lets the author see what the
// extractor proposed but the engine rejected even after several /sends.

import { useMemo, useState } from "react";
import type { CanonDropRecord, PlayerCanonEntry } from "../../api";
import { Empty } from "../shared/Empty";
import type { PanelProps } from "../shared/types";

const DROP_HISTORY_TURNS = 20;

export function CanonPanel({ tickHistory }: PanelProps) {
  const [filter, setFilter] = useState("");
  const f = filter.trim().toLowerCase();

  const latest = tickHistory[0];
  const canon = latest?.player_canon ?? {};
  const drops = useMemo(() => collectRecentDrops(tickHistory), [tickHistory]);

  const canonRows = Object.entries(canon)
    .filter(([k, v]) => !f || matches(k, v, f))
    .sort(([a], [b]) => a.localeCompare(b));

  return (
    <div className="flex flex-col gap-3">
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="filter by key, value, or quote…"
        className="w-full rounded border border-border bg-bg px-2 py-1.5 text-[12px] text-text placeholder:text-text-muted focus:border-accent focus:outline-none"
      />
      {canonRows.length === 0 ? (
        <Empty>no canon yet</Empty>
      ) : (
        <table className="w-full border-collapse text-left text-[12px]">
          <thead>
            <tr>
              <Th>key</Th>
              <Th>value</Th>
              <Th>quote</Th>
              <Th className="text-right">sal</Th>
              <Th>T</Th>
            </tr>
          </thead>
          <tbody>
            {canonRows.map(([key, entry]) => (
              <tr key={key} className="align-top">
                <Td className="font-mono text-text">{key}</Td>
                <Td className="text-text">{entry.summary}</Td>
                <Td>
                  <span className="rounded bg-bg-elevated px-1.5 py-0.5 text-[11px] italic text-text-muted">
                    "{entry.literal_text_quote}"
                  </span>
                </Td>
                <Td className="text-right">
                  <SalienceBar value={entry.salience} />
                  <span className="ml-1 font-mono text-[11px] text-text">{entry.salience.toFixed(2)}</span>
                </Td>
                <Td className="text-text-muted">T{entry.turn_of_origin}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div>
        <div className="mb-1 text-[9px] uppercase tracking-wider text-text-muted">
          dropped candidates · last {DROP_HISTORY_TURNS} ticks
        </div>
        {drops.length === 0 ? (
          <Empty>no drops</Empty>
        ) : (
          <ul className="space-y-1 text-[12px]">
            {drops.filter((d) =>
              !f || d.candidate.key.toLowerCase().includes(f)
                || d.candidate.summary.toLowerCase().includes(f)
            ).map((d, i) => (
              <li key={i} className="grid grid-cols-[30px_1fr_auto] gap-2">
                <span className="font-mono text-[11px] text-text-muted">T{d.tick}</span>
                <span>
                  <span className="font-mono text-text">{d.candidate.key}</span>
                  <span className="text-text-muted"> = "{d.candidate.summary}"</span>
                </span>
                <DropReason reason={d.reason} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function matches(key: string, entry: PlayerCanonEntry, q: string): boolean {
  return (
    key.toLowerCase().includes(q)
    || entry.summary.toLowerCase().includes(q)
    || entry.literal_text_quote.toLowerCase().includes(q)
  );
}

function SalienceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <span className="inline-block h-1 w-9 overflow-hidden rounded bg-bg-elevated align-middle">
      <span className="block h-full bg-accent" style={{ width: `${pct}%` }} />
    </span>
  );
}

function DropReason({ reason }: { reason: CanonDropRecord["reason"] }) {
  const cls = reason === "anchor_fail"
    ? "bg-meter-hot/15 text-meter-hot"
    : "bg-meter-warn/15 text-meter-warn";
  return (
    <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${cls}`}>
      {reason}
    </span>
  );
}

function Th({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <th className={`border-b border-border pb-1 text-[9px] uppercase tracking-wider text-text-muted ${className}`}>
      {children}
    </th>
  );
}

function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`border-b border-border/30 py-1.5 pr-2 ${className}`}>{children}</td>;
}

interface DropEntry {
  tick: number;
  candidate: CanonDropRecord["candidate"];
  reason: CanonDropRecord["reason"];
}

function collectRecentDrops(tickHistory: PanelProps["tickHistory"]): DropEntry[] {
  // tickHistory is newest-first. Tag each drop with a 1-indexed tick
  // number from the START of the visible history (tickHistory[0] = T_max).
  const out: DropEntry[] = [];
  const cap = Math.min(tickHistory.length, DROP_HISTORY_TURNS);
  for (let i = 0; i < cap; i++) {
    const t = tickHistory[i]!;
    const tickNumber = tickHistory.length - i;
    for (const d of t.canon_drops) {
      out.push({ tick: tickNumber, candidate: d.candidate, reason: d.reason });
    }
  }
  return out;
}
