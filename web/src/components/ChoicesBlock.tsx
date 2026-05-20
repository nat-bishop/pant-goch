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
  // The engine appends leave entries after the real options, so the real
  // ones keep indices 0..N-1 (aligned with PickOption). Render the leave
  // entry apart from the numbered list; clicking it opens the map rather
  // than consuming the turn's pick.
  const leaveEntry = options.find((o) => o.leave);

  return (
    <div className="choices-wrap">
      <ol className="choices">
        {options.map((opt, i) => {
          if (opt.leave) return null;
          const state: "default" | "chosen" | "unchosen" = !hasPick
            ? "default"
            : i === pickedIdx
              ? "chosen"
              : "unchosen";
          // The <li> is the collapsible track: unchosen options animate
          // their height to zero so the chosen line rises into place.
          const itemCls =
            state === "unchosen" ? "choice-item unchosen" : "choice-item";
          const btnCls = state === "chosen" ? "choice chosen" : "choice";
          return (
            <li key={i} className={itemCls}>
              <button
                className={btnCls}
                onClick={() => onPick(i)}
                disabled={busy || hasPick}
              >
                <span className="choice-num">{i + 1}</span>
                <span>{opt.text}</span>
              </button>
            </li>
          );
        })}
      </ol>
      {leaveEntry && !hasPick && (
        <button
          type="button"
          className="choice-leave"
          onClick={onLeave}
          disabled={busy}
        >
          {leaveEntry.text} <span className="choice-leave-arrow">→</span>
        </button>
      )}
    </div>
  );
}
