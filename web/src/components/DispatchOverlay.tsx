import { useMemo } from "react";
import { displayActor } from "../api";
import { nowZulu } from "../lib/time";
import { useStore } from "../store";
import { StreamingText } from "./StreamingText";

export function DispatchOverlay() {
  const acting = useStore((s) => s.cycle?.dispatch_acting ?? null);
  const words = useStore((s) => s.cycle?.dispatch_words);
  const dispatchComplete = useStore((s) => s.cycle?.dispatch_complete ?? false);
  const awaitingContinue = useStore((s) => s.cycle?.awaiting_continue ?? false);
  const continueCycle = useStore((s) => s.continueCycle);
  const time = useMemo(() => nowZulu(), [acting]);

  if (acting === null || !words) return null;

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
