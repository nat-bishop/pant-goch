// Choices block — rendered once the latest turn has closed and options
// are present. Stays mounted after a pick (chosen/unchosen states) until
// the next turn begins streaming and unmounts it; this prevents the
// large height collapse that would otherwise yank the panel on click.

import type { OptionPayload } from "../api";

export function ChoicesBlock({
  options,
  picked,
  onPick,
  busy,
}: {
  options: OptionPayload[];
  picked: OptionPayload | null;
  onPick: (idx: number) => void;
  busy: boolean;
}) {
  const pickedIdx = picked
    ? options.findIndex((o) => o.text === picked.text)
    : -1;
  const hasPick = picked !== null;

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
    </div>
  );
}
