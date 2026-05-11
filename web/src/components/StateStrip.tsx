import { useStore } from "../store";
import { Pips } from "./Pips";
import { StageDots } from "./StageDots";

/** Four datums on the desk above the paper. Cycles remaining (warn-coloured
 * when ≤ 2), per-agent cover bars, extraction stage dots. */
export function StateStrip() {
  const meta = useStore((s) => s.mission_meta);
  const state = useStore((s) => s.state);
  if (!meta) {
    return (
      <section className="border-b border-desk-rule px-14 pt-7 pb-8 grid grid-cols-1 text-chrome-dim text-sm">
        <div>booting…</div>
      </section>
    );
  }

  const cyclesRemaining = (state["mission.cycles_remaining"] as number | undefined) ?? meta.max_cycles;
  const warn = cyclesRemaining <= 2;
  const stage = (state["mission.extraction_stage"] as number | undefined) ?? 0;

  return (
    <section
      aria-label="visible state"
      className="grid grid-cols-4 gap-0 border-b border-desk-rule px-14 pt-7 pb-8"
      style={{
        backgroundImage:
          "linear-gradient(180deg, rgba(255,255,255,0.012) 0%, transparent 100%)",
      }}
    >
      <Cell label="Cycles remaining" first>
        <span
          className={`text-[26px] leading-none tracking-[0.04em] tabular-nums ${
            warn ? "text-stamp-soft" : "text-chrome-bright"
          }`}
        >
          {String(cyclesRemaining).padStart(2, "0")}
          <span className="text-chrome-dim text-sm tracking-[0.04em]"> / {String(meta.max_cycles).padStart(2, "0")}</span>
        </span>
      </Cell>

      {meta.agents.map((agent) => {
        const cover = (state[`agent[${agent.id}].cover`] as number | undefined) ?? 0;
        return (
          <Cell key={agent.id} label={`${agent.id} · cover`}>
            <Pips count={cover} cap={10} />
            <span className="text-xs text-chrome tracking-[0.06em] tabular-nums">
              <strong className="text-chrome-bright font-normal">{cover}</strong> / 10
            </span>
          </Cell>
        );
      })}

      <Cell label="Extraction stage" last>
        <StageDots current={stage} total={3} />
        <span className="text-xs text-chrome tracking-[0.06em] tabular-nums">
          <strong className="text-chrome-bright font-normal">{stage}</strong> / 3
        </span>
      </Cell>
    </section>
  );
}

function Cell({
  label,
  children,
  first,
  last,
}: {
  label: string;
  children: React.ReactNode;
  first?: boolean;
  last?: boolean;
}) {
  return (
    <div
      className={`flex flex-col gap-2.5 px-9 ${
        first ? "pl-0 border-l-0" : "border-l border-desk-rule"
      } ${last ? "pr-0" : ""}`}
    >
      <div className="text-[10px] uppercase tracking-[0.3em] text-chrome-dim">{label}</div>
      <div className="flex items-baseline gap-3.5 flex-wrap">{children}</div>
    </div>
  );
}
