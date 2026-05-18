// Choices block — rendered once the latest turn has closed and options
// are present. Option is just {text}; the map handles navigation
// (between-scenes pick_walk).

import type { OptionPayload } from "../api";

export function ChoicesBlock({
  options,
  onPick,
  busy,
}: {
  options: OptionPayload[];
  onPick: (idx: number) => void;
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
    </div>
  );
}
