import { useStore } from "../store";

/** Past dispatches surfaced as cards below the paper. Click a card to open
 * the full dispatch modal. */
export function Inbox() {
  const inbox = useStore((s) => s.inbox);
  const openModal = useStore((s) => s.openModal);

  if (inbox.length === 0) return null;

  return (
    <section
      className="border-t border-desk-rule px-14 pt-7 pb-10"
      style={{ background: "linear-gradient(180deg, transparent 0%, rgba(0,0,0,0.25) 100%)" }}
      aria-label="prior dispatches"
    >
      <div className="flex justify-between items-baseline mb-4.5 text-[10.5px] uppercase tracking-[0.3em] text-chrome-dim">
        <span className="text-chrome-bright tracking-[0.32em]">
          — Inbox · prior dispatches —
        </span>
        <span className="text-chrome-faint">
          {inbox.length} {inbox.length === 1 ? "entry" : "entries"} · click to read
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4 max-w-[980px] mx-auto">
        {[...inbox].reverse().map((d) => (
          <button
            key={`${d.cycle_index}-${d.acting}`}
            onClick={() => openModal(d.cycle_index)}
            className="bg-transparent border border-desk-rule p-4 cursor-pointer transition-[border-color,background,transform] duration-200 text-left text-chrome font-inherit flex flex-col gap-2 min-w-0 hover:border-stamp hover:bg-[rgba(40,41,31,0.4)] hover:-translate-y-0.5"
          >
            <div className="flex justify-between items-baseline text-[9.5px] uppercase tracking-[0.26em] text-chrome-dim pb-2 border-b border-desk-rule border-dashed">
              <span className="text-stamp-soft font-bold">C{d.cycle_index}</span>
              <span>{d.acting === "operations" ? "sent" : "received"}</span>
            </div>
            <div className="text-sm tracking-[0.08em] text-chrome-bright font-bold uppercase flex items-baseline gap-3">
              {d.acting === "operations" ? "OPS" : d.acting}
              <span className="text-[9.5px] text-chrome-dim tracking-[0.22em] font-normal normal-case">
                — {d.acting === "operations" ? "station log" : "field dispatch"}
              </span>
            </div>
            <div className="text-[12.5px] leading-relaxed text-chrome-dim overflow-hidden line-clamp-2">
              {d.text.slice(0, 200)}
              {d.text.length > 200 && "…"}
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}
