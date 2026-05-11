import { useStore } from "../store";
import { OptionsList } from "./OptionsList";
import { StreamingText } from "./StreamingText";

/** The document on the desk — classification, MINUTE heading, memo header,
 * State of Play body, options divider, options list, paper footer. The
 * .has-selection / .authorized state classes are applied to drive the
 * focus-dim treatment per the mockup. */
export function Paper() {
  const meta = useStore((s) => s.mission_meta);
  const cycle = useStore((s) => s.cycle);

  if (!cycle || !meta) {
    return (
      <article className="bg-paper text-ink w-full max-w-[820px] p-20 shadow-[0_40px_72px_rgba(0,0,0,0.4)] relative animate-paper-rise">
        <div className="text-ink-faint italic">Waiting for cycle…</div>
      </article>
    );
  }

  const hasSelection = cycle.selected_option_index !== null;
  const authorized = cycle.authorized;
  const cycleLabel = String(cycle.index).padStart(2, "0");

  return (
    <article
      className="bg-paper text-ink w-full max-w-[820px] px-24 py-20 shadow-[0_1px_2px_rgba(0,0,0,0.3),0_12px_28px_rgba(0,0,0,0.5),0_40px_72px_rgba(0,0,0,0.4)] relative animate-paper-rise"
      aria-label={`cycle ${cycle.index} minute`}
      style={{
        backgroundImage:
          "radial-gradient(ellipse 100% 100% at 50% 50%, transparent 75%, rgba(150, 130, 80, 0.10) 100%)",
      }}
      data-has-selection={hasSelection}
      data-authorized={authorized}
    >
      <span
        className="absolute top-9 right-14 text-stamp text-[10.5px] font-bold uppercase tracking-[0.3em] py-1.5 px-3.5 opacity-85"
        style={{
          border: "2px double var(--stamp)",
          background: "rgba(122, 31, 24, 0.04)",
          transform: "rotate(-2deg)",
        }}
      >
        TOP SECRET
      </span>

      <div className="text-[13px] font-bold tracking-[0.42em] text-ink mb-1">
        MINUTE
      </div>
      <div
        className="border-b border-ink mb-8"
        style={{ width: 92, borderBottomWidth: 1.5 }}
      />

      <div className="mb-9 text-[12.5px] leading-[1.95] text-ink">
        <MemoRow label="TO" value="Handler" />
        <MemoRow label="FROM" value="Operations Desk" />
        <MemoRow label="RE" value={`Cycle ${cycleLabel} — authorization required`} />
      </div>

      <div
        className="text-[14.5px] leading-[2.05] text-ink mb-12 transition-opacity duration-300"
        style={{
          maxWidth: "60ch",
          opacity: hasSelection ? 0.35 : 1,
        }}
      >
        <StreamingText
          words={cycle.sop_words}
          showCaret={!cycle.sop_complete}
          complete={cycle.sop_complete}
        />
      </div>

      <div
        className="grid grid-cols-[1fr_auto_1fr] gap-5 items-center mb-7 transition-opacity duration-300"
        style={{ opacity: hasSelection ? 0.45 : 1 }}
      >
        <div className="h-px bg-rule" />
        <span className="text-[11px] uppercase tracking-[0.34em] text-stamp whitespace-nowrap font-bold">
          — courses of action —
        </span>
        <div className="h-px bg-rule" />
      </div>

      <OptionsList />

      <div
        className="mt-11 pt-4 border-t border-rule-soft border-dashed flex justify-between items-baseline text-[10.5px] uppercase tracking-[0.22em] text-ink-soft"
      >
        <span>
          <span className="text-ink font-bold tracking-[0.14em]">N.B.</span>
          <span className="text-ink-faint mx-2.5">/</span>
          Handler
          <span className="text-ink-faint mx-2.5">/</span>
          STA-1
        </span>
        <span>
          {meta.codename}
          <span className="text-ink-faint mx-2"> · </span>
          PF.{meta.agents[0]?.file_number ?? "??-?"}
        </span>
      </div>
    </article>
  );
}

function MemoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[64px_1fr] gap-3">
      <span className="font-bold tracking-[0.1em]">{label}</span>
      <span>{value}</span>
    </div>
  );
}
