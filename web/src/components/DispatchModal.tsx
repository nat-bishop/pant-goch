import { useEffect } from "react";
import { displayActor, OPERATIONS_ACTOR } from "../api";
import { useStore } from "../store";

export function DispatchModal() {
  const modalIdx = useStore((s) => s.modalDispatchIndex);
  const inbox = useStore((s) => s.inbox);
  const closeModal = useStore((s) => s.closeModal);

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

  if (modalIdx === null) return null;
  const d = inbox.find((x) => x.cycle_index === modalIdx);
  if (!d) return null;

  const isOps = d.acting === OPERATIONS_ACTOR;
  const role = isOps ? "Station log" : "Field dispatch";

  return (
    <div
      className="scrim"
      onClick={(e) => {
        if (e.target === e.currentTarget) closeModal();
      }}
      role="dialog"
      aria-modal="true"
    >
      <article className="warm-paper-card archive-card">
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
          <span>
            {role} · cycle {d.cycle_index}
          </span>
        </div>
        <div className="archive-from">
          from <strong>{displayActor(d.acting)}</strong>
        </div>
        <div className="archive-route">
          C{d.cycle_index} · {d.archetype}
        </div>

        <div className="archive-body">{d.text}</div>

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
