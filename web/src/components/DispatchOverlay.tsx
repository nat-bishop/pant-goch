import { useEffect, useMemo, useState } from "react";
import { displayActor } from "../api";
import { nowZulu } from "../lib/time";
import { useStore } from "../store";
import { StreamingText } from "./StreamingText";

// Hold the overlay back for this long after the player authorizes, so the
// AUTHORIZED stamp's 280ms `stamp-in` animation lands and the eye has a beat
// to register it before the scrim covers the paper.
const AUTH_TO_DISPATCH_DELAY_MS = 700;

export function DispatchOverlay() {
  const cycleIndex = useStore((s) => s.cycle?.index ?? null);
  const authorized = useStore((s) => s.cycle?.authorized ?? false);
  const acting = useStore((s) => s.cycle?.dispatch_acting ?? null);
  const words = useStore((s) => s.cycle?.dispatch_words);
  const dispatchComplete = useStore((s) => s.cycle?.dispatch_complete ?? false);
  const awaitingContinue = useStore((s) => s.cycle?.awaiting_continue ?? false);
  const continueCycle = useStore((s) => s.continueCycle);
  const time = useMemo(() => nowZulu(), [acting]);
  const [stampSettled, setStampSettled] = useState(false);

  useEffect(() => {
    setStampSettled(false);
    if (!authorized) return;
    const t = setTimeout(() => setStampSettled(true), AUTH_TO_DISPATCH_DELAY_MS);
    return () => clearTimeout(t);
  }, [cycleIndex, authorized]);

  if (acting === null || !words || !stampSettled) return null;

  return (
    <div className="dispatch-overlay" role="dialog" aria-label="incoming dispatch">
      <article className="warm-paper-card dispatch-card">
        <header className="dispatch-head">
          <span className="dispatch-tag">DISPATCH</span>
          <span className="dispatch-meta">
            {displayActor(acting)} · {time} · received
          </span>
        </header>
        <div className="dispatch-body">
          <StreamingText
            words={words}
            showCaret={!dispatchComplete}
            complete={dispatchComplete}
          />
        </div>
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
