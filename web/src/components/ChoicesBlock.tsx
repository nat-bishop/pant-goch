// Choices block — rendered once the latest turn has closed and options
// are present. Option is just {text}; the map handles navigation
// (between-scenes pick_walk). The persistent LeaveButton below the
// option list is the player's structural escape hatch: closes the scene
// via POST /leave and hands the floor to the map picker.

import type { OptionPayload } from "../api";
import { LeaveButton } from "./LeaveButton";

export function ChoicesBlock({
  options,
  onPick,
  onLeave,
  busy,
}: {
  options: OptionPayload[];
  onPick: (idx: number) => void;
  onLeave: () => void;
  busy: boolean;
}) {
  return (
    <div className="choices-wrap">
      <div className="choices-eyebrow">— how I went on —</div>
      <ol className="choices">
        {options.map((opt, i) => (
          <li key={i}>
            <button
              className="choice"
              onClick={() => onPick(i)}
              disabled={busy}
            >
              <span className="choice-num">{i + 1}.</span>
              <span>{opt.text}</span>
            </button>
          </li>
        ))}
      </ol>
      <LeaveButton onLeave={onLeave} busy={busy} />
    </div>
  );
}
