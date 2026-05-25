// Choices block — rendered once the latest turn has closed and options
// are present. Stays mounted after a pick (chosen/unchosen states) until
// the next turn begins streaming and unmounts it; this prevents the
// large height collapse that would otherwise yank the panel on click.

import type { OptionPayload } from "../api";

export function ChoicesBlock({
  options,
  picked,
  onPick,
  onLeave,
  busy,
}: {
  options: OptionPayload[];
  picked: OptionPayload | null;
  onPick: (idx: number) => void;
  onLeave: () => void;
  busy: boolean;
}) {
  const pickedIdx = picked
    ? options.findIndex((o) => o.text === picked.text)
    : -1;
  const hasPick = picked !== null;
  // Every option — including the diegetic exit (`leave`) — renders in the
  // numbered list and keeps its array index, so a non-leave option's number
  // still equals the PickOption index the backend expects. The leave option
  // also gets a number (so it reads as one of the choices), but its click is
  // routed to onLeave (open the map; the pin → walk flow does the leaving)
  // instead of consuming the turn's pick, and it carries a muted "leave the
  // location" treatment + a trailing arrow.

  return (
    <div className="choices-wrap">
      <ol className="choices">
        {options.map((opt, i) => {
          const state: "default" | "chosen" | "unchosen" = !hasPick
            ? "default"
            : i === pickedIdx
              ? "chosen"
              : "unchosen";
          // The <li> is the collapsible track: unchosen options animate
          // their height to zero so the chosen line rises into place.
          const itemCls =
            state === "unchosen" ? "choice-item unchosen" : "choice-item";
          const btnCls =
            (opt.leave ? "choice choice--leave" : "choice") +
            (state === "chosen" ? " chosen" : "");
          return (
            <li key={i} className={itemCls}>
              <button
                className={btnCls}
                onClick={() => (opt.leave ? onLeave() : onPick(i))}
                disabled={busy || hasPick}
              >
                <span className="choice-num">{i + 1}</span>
                <span>
                  {opt.text}
                  {opt.leave && <span className="choice-leave-arrow"> →</span>}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
