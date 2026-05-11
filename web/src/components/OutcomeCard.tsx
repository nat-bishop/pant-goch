import { useStore } from "../store";

/** Terminal outcome overlay — shown once `outcome` is set. Provides a
 * Restart button that triggers a fresh mission. */
export function OutcomeCard() {
  const outcome = useStore((s) => s.outcome);
  const meta = useStore((s) => s.mission_meta);
  const state = useStore((s) => s.state);
  const inbox = useStore((s) => s.inbox);

  if (!outcome || !meta) return null;

  const win = outcome.outcome_key.startsWith("win_");
  const cyclesUsed = inbox.length;
  const lostAgents = meta.agents
    .filter((a) => {
      const cover = state[`agent[${a.id}].cover`];
      return typeof cover === "number" && cover <= 0;
    })
    .map((a) => a.id);
  const liveAgents = meta.agents
    .filter((a) => !lostAgents.includes(a.id))
    .map((a) => a.id);

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center px-6 py-12 overflow-y-auto"
      style={{ background: "rgba(5, 6, 4, 0.92)", backdropFilter: "blur(4px)" }}
    >
      <article
        className="bg-paper text-ink w-full max-w-[640px] p-20 shadow-[0_40px_72px_rgba(0,0,0,0.45)] relative animate-paper-rise"
      >
        <div className="text-[10px] uppercase tracking-[0.34em] text-stamp mb-3 flex items-center gap-3">
          <span className="inline-block w-6 h-px bg-stamp" />
          Mission resolved
        </div>
        <h2 className="text-2xl font-bold tracking-[0.16em] text-ink mb-2">
          Op. {meta.codename} —{" "}
          <span className={win ? "text-ink" : "text-stamp"}>
            {win ? "SUCCESS" : "FAILURE"}
          </span>
        </h2>
        <div
          className="border-b border-ink mb-7"
          style={{ width: 92, borderBottomWidth: 1.5 }}
        />

        <div className="text-sm leading-[1.95] text-ink mb-9 whitespace-pre-wrap">
          {outcome.summary}
        </div>

        <div className="grid grid-cols-2 gap-6 text-[11px] uppercase tracking-[0.18em] text-ink-soft mb-9">
          <Stat label="Cycles used" value={`${cyclesUsed} / ${meta.max_cycles}`} />
          <Stat label="Agents lost" value={lostAgents.join(", ") || "none"} />
          <Stat label="Agents alive" value={liveAgents.join(", ") || "none"} />
          <Stat label="Outcome key" value={outcome.outcome_key} />
        </div>

        <div className="flex gap-4">
          <button
            onClick={() => window.location.reload()}
            className="bg-transparent px-7 py-3 font-inherit text-xs uppercase font-bold tracking-[0.32em] border-[1.5px] border-ink text-ink cursor-pointer hover:bg-ink hover:text-paper transition-all duration-200"
          >
            Restart mission
          </button>
        </div>
      </article>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[9.5px] tracking-[0.24em] text-ink-faint mb-1">{label}</div>
      <div className="text-ink tracking-[0.1em]">{value}</div>
    </div>
  );
}
