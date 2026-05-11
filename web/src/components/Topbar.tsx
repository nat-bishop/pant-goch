import { useStore } from "../store";

/** Identity strip: HANDLER mark / station label / mission codename / cycle counter. */
export function Topbar() {
  const meta = useStore((s) => s.mission_meta);
  const cycle = useStore((s) => s.cycle);
  const state = useStore((s) => s.state);
  const cyclesRemaining = state["mission.cycles_remaining"];
  const maxCycles = meta?.max_cycles;
  const currentCycle =
    typeof cyclesRemaining === "number" && typeof maxCycles === "number"
      ? maxCycles - cyclesRemaining + (cycle?.dispatch_complete ? 1 : 0)
      : null;

  return (
    <header className="grid grid-cols-[1fr_auto_1fr] items-center gap-6 border-b border-desk-rule px-14 py-5 text-[10.5px] uppercase text-chrome-dim tracking-[0.28em]">
      <div className="flex gap-7 items-baseline">
        <span className="text-chrome-bright tracking-[0.32em]">HANDLER</span>
        <span className="text-chrome-faint">/</span>
        <span>STA · 1</span>
      </div>
      <div className="text-center text-chrome text-[11.5px] tracking-[0.22em]">
        {meta ? (
          <>
            Op. <em className="not-italic text-chrome-bright tracking-[0.24em]">{meta.codename}</em>
          </>
        ) : (
          <span>HANDLER</span>
        )}
      </div>
      <div className="flex gap-7 items-baseline justify-end">
        <span className="text-chrome-faint">/</span>
        <span className="text-chrome-bright tracking-[0.32em]">
          {currentCycle !== null && maxCycles
            ? `CYC ${String(Math.max(1, currentCycle)).padStart(2, "0")} / ${String(maxCycles).padStart(2, "0")}`
            : "—"}
        </span>
      </div>
    </header>
  );
}
