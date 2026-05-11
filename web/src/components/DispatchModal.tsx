import { useEffect } from "react";
import { useStore } from "../store";

/** Full-dispatch reader. Opens when `modalDispatchIndex` is set; renders the
 * paper-style document with classification stamp, header/route, body, and
 * the recorded state delta at the bottom. */
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

  const isOps = d.acting === "operations";
  const acting = isOps ? "Operations Desk" : d.acting;
  const role = isOps ? "Station log" : "Field dispatch";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-6 py-12 overflow-y-auto"
      style={{ background: "rgba(5, 6, 4, 0.82)", backdropFilter: "blur(3px)" }}
      onClick={(e) => {
        if (e.target === e.currentTarget) closeModal();
      }}
      role="dialog"
      aria-modal="true"
    >
      <article
        className="bg-paper-onion text-ink w-full max-w-[720px] p-20 shadow-[0_24px_48px_rgba(0,0,0,0.3)] relative animate-paper-rise"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent 0px, transparent 22px, rgba(150, 140, 110, 0.05) 22px, rgba(150, 140, 110, 0.05) 23px), radial-gradient(ellipse 80% 20% at 0% 0%, rgba(160, 145, 100, 0.08) 0%, transparent 50%)",
        }}
      >
        <button
          onClick={closeModal}
          aria-label="close"
          className="absolute top-6 right-6 w-8 h-8 border border-ink-soft bg-transparent text-ink-soft cursor-pointer font-inherit text-base leading-none transition-all duration-200 hover:bg-ink hover:text-paper hover:border-ink"
        >
          ×
        </button>

        <span
          className="absolute top-7 right-[72px] text-stamp text-[9.5px] font-bold tracking-[0.28em] py-1 px-3 opacity-60"
          style={{
            border: "1.5px solid var(--stamp)",
            transform: "rotate(-3deg)",
            whiteSpace: "pre",
          }}
        >
          {"FILE\nCOPY"}
        </span>

        <div className="text-[10px] uppercase tracking-[0.32em] text-stamp mb-3.5 flex items-center gap-3.5">
          <span className="inline-block w-6 h-px bg-stamp" />
          <span>
            {role} · cycle {d.cycle_index}
          </span>
        </div>
        <div className="text-lg text-ink mb-1.5 tracking-[0.06em]">
          from <span className="font-bold">{acting}</span>
        </div>
        <div className="text-[11px] tracking-[0.18em] text-ink-soft uppercase pb-5 mb-7 border-b border-rule">
          C{d.cycle_index} · {d.archetype}
        </div>

        <div className="text-sm leading-[1.95] text-ink whitespace-pre-wrap">
          {d.text}
        </div>

        {Object.keys(d.delta).length > 0 && (
          <div className="mt-7 pt-4 border-t border-rule-soft border-dashed text-[11px] uppercase tracking-[0.16em] text-ink-soft flex gap-5 flex-wrap">
            {Object.entries(d.delta).map(([q, { before, after }]) => (
              <span key={q}>
                <span className="text-stamp mr-2 font-bold">{q}</span>
                {String(before)} → {String(after)}
              </span>
            ))}
          </div>
        )}
      </article>
    </div>
  );
}
