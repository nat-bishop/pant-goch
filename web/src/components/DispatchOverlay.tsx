import { useEffect, useMemo, useState } from "react";
import { displayActor, OPERATIONS_ACTOR } from "../api";
import type { DeltaDescriptors } from "../api";
import { nowZulu } from "../lib/time";
import { useStore } from "../store";
import { StreamingText } from "./StreamingText";

// Hold the overlay back for this long after the player authorizes, so the
// AUTHORIZED stamp's 280ms `stamp-in` animation lands and the eye has a beat
// to register it before the scrim covers the paper.
const AUTH_TO_DISPATCH_DELAY_MS = 700;

// Stable fallback for the Zustand selector below. Creating a new `{}` inside
// the selector closure changes the reference on every read and trips
// `useSyncExternalStore`'s loop guard ("Maximum update depth exceeded").
const EMPTY_DESCRIPTORS: DeltaDescriptors = Object.freeze({});

interface ResultLine {
  key: string;
  text: string;
}

/** Build the RESULT footer lines from the cycle's accumulated narrative
 * descriptors. Cover changes render as `LYNX: cover intact → cover degraded`
 * (using `agent_id` from the server payload); mission qualities render the
 * new descriptor as a standalone clause. */
function buildResultLines(descriptors: DeltaDescriptors): ResultLine[] {
  const lines: ResultLine[] = [];
  for (const [q, d] of Object.entries(descriptors)) {
    if (d.after === null) continue;
    if (d.agent_id !== null) {
      lines.push({
        key: q,
        text: d.before
          ? `${d.agent_id}: ${d.before} → ${d.after}`
          : `${d.agent_id}: ${d.after}`,
      });
    } else {
      // Mission-scoped quality (cycles_remaining / extraction_stage) —
      // the descriptor already reads as a full clause.
      lines.push({ key: q, text: d.after });
    }
  }
  // Cover lines first (most player-visible), then mission-scoped lines.
  return lines.sort(
    (a, b) =>
      (descriptors[a.key].agent_id ? 0 : 1) -
      (descriptors[b.key].agent_id ? 0 : 1),
  );
}

export function DispatchOverlay() {
  const cycleIndex = useStore((s) => s.cycle?.index ?? null);
  const authorized = useStore((s) => s.cycle?.authorized ?? false);
  const acting = useStore((s) => s.cycle?.dispatch_acting ?? null);
  const words = useStore((s) => s.cycle?.dispatch_words);
  const dispatchComplete = useStore((s) => s.cycle?.dispatch_complete ?? false);
  const awaitingContinue = useStore((s) => s.cycle?.awaiting_continue ?? false);
  const descriptors = useStore(
    (s) => s.cycle?.delta_descriptors ?? EMPTY_DESCRIPTORS,
  );
  const continueCycle = useStore((s) => s.continueCycle);
  const time = useMemo(() => nowZulu(), [acting]);
  const [stampSettled, setStampSettled] = useState(false);

  useEffect(() => {
    setStampSettled(false);
    if (!authorized) return;
    const t = setTimeout(() => setStampSettled(true), AUTH_TO_DISPATCH_DELAY_MS);
    return () => clearTimeout(t);
  }, [cycleIndex, authorized]);

  // Only populated once the dispatch has finished streaming — otherwise the
  // player would read "RESULT" before reading the prose that produced it.
  // Computed before the visibility guard below so the hook count is stable
  // across renders.
  const resultLines = useMemo(
    () => (dispatchComplete ? buildResultLines(descriptors) : []),
    [dispatchComplete, descriptors],
  );

  if (acting === null || !words || !stampSettled) return null;

  const isOps = acting === OPERATIONS_ACTOR;
  const tag = isOps ? "STATION LOG" : "FIELD DISPATCH";
  const fromSub = isOps
    ? "Operations Desk"
    : `from ${displayActor(acting)} (in voice)`;

  return (
    <div className="dispatch-overlay" role="dialog" aria-label="incoming dispatch">
      <article className="warm-paper-card dispatch-card">
        <header className="dispatch-head">
          <span className="dispatch-tag">{tag}</span>
          <span className="dispatch-meta">
            {fromSub} · received {time}
          </span>
        </header>
        <div className="dispatch-body">
          <StreamingText
            words={words}
            showCaret={!dispatchComplete}
            complete={dispatchComplete}
          />
        </div>
        {resultLines.length > 0 && (
          <div className="dispatch-result" aria-label="cycle result">
            <span className="dispatch-result-label">RESULT</span>
            <ul className="dispatch-result-lines">
              {resultLines.map((line) => (
                <li key={line.key}>{line.text}</li>
              ))}
            </ul>
          </div>
        )}
        <footer className="dispatch-foot">
          <span className="paper-lbl">ACK</span>
          {awaitingContinue ? (
            <button
              type="button"
              className="paper-btn small"
              onClick={() => void continueCycle()}
            >
              Continue ▸
            </button>
          ) : (
            <span className="dispatch-hint">tap to skip</span>
          )}
        </footer>
      </article>
    </div>
  );
}
