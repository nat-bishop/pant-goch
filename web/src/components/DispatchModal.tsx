import { useEffect } from "react";
import { useStore } from "../store";
import { StreamingText } from "./StreamingText";
import { usePacedWords } from "../hooks/usePacedWords";

export function DispatchModal() {
  const modalIdx = useStore((s) => s.modalDispatchIndex);
  const inbox = useStore((s) => s.inbox);
  const closeModal = useStore((s) => s.closeModal);

  const d = modalIdx === null ? null : inbox.find((x) => x.cycle_index === modalIdx);
  const { words, complete, flush } = usePacedWords(d?.text ?? "");

  useEffect(() => {
    if (modalIdx === null) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") closeModal();
    }
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [modalIdx, closeModal]);

  if (modalIdx === null || !d) return null;

  return (
    <div
      className="scrim"
      onClick={(e) => {
        if (e.target === e.currentTarget) closeModal();
      }}
      role="dialog"
      aria-modal="true"
    >
      <article className="warm-paper-card archive-card" onClickCapture={flush}>
        <button
          type="button"
          className="archive-close"
          onClick={closeModal}
          aria-label="close"
        >
          ×
        </button>

        <div className="stamp stamp-corner">
          <div>FILE</div>
          <div>COPY</div>
        </div>

        <div className="archive-tag">
          <span className="archive-tag-rule" />
          <span>Case-file entry · cycle {d.cycle_index}</span>
        </div>
        <div className="archive-route">
          C{d.cycle_index} · {d.archetype}
        </div>

        <div className="archive-body">
          <StreamingText words={words} complete={complete} showCaret={!complete} />
        </div>

        {Object.keys(d.delta).length > 0 && (
          <div className="archive-delta">
            {Object.entries(d.delta).map(([q, { before, after }]) => (
              <span key={q}>
                <span className="archive-delta-key">{q}</span>
                {String(before)} → {String(after)}
              </span>
            ))}
          </div>
        )}
      </article>
    </div>
  );
}
