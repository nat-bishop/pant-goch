// Choices block — rendered once the latest turn has closed and options
// are present. Stays mounted after a pick (chosen/unchosen states) until
// the next turn begins streaming and unmounts it; this prevents the
// large height collapse that would otherwise yank the panel on click.

import type { OptionPayload } from "../api";
import { LeaveButton } from "./LeaveButton";

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

  return (
    <div className="choices-wrap">
      <div className="choices-eyebrow">— how I went on —</div>
      <ol className="choices">
        {options.map((opt, i) => {
          const state: "default" | "chosen" | "unchosen" = !hasPick
            ? "default"
            : i === pickedIdx
              ? "chosen"
              : "unchosen";
          const cls = state === "default" ? "choice" : `choice ${state}`;
          return (
            <li key={i}>
              <button
                className={cls}
                onClick={() => onPick(i)}
                disabled={busy || hasPick}
              >
                <span className="choice-num">{i + 1}.</span>
                <span>{opt.text}</span>
              </button>
            </li>
          );
        })}
      </ol>
      {!hasPick && <LeaveButton onLeave={onLeave} busy={busy} />}
    </div>
  );
}
