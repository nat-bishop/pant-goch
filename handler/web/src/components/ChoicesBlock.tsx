// Choices block — rendered once the latest turn has closed and options
// are present. "Leave for X" is the only derivable badge (the backend's
// Option schema is just {text, leaves_for}).

import type { OptionPayload } from "../api";

export function ChoicesBlock({
  options,
  onPick,
  busy,
  locationNameById,
}: {
  options: OptionPayload[];
  onPick: (idx: number) => void;
  busy: boolean;
  locationNameById: (id: string) => string;
}) {
  return (
    <div className="choices-wrap">
      <div className="choices-eyebrow">— how I went on —</div>
      <ol className="choices">
        {options.map((opt, i) => {
          const kind = opt.leaves_for ? "leave" : "engage";
          return (
            <li key={i}>
              <button
                className="choice"
                data-kind={kind}
                onClick={() => onPick(i)}
                disabled={busy}
              >
                <span className="choice-num">{i + 1}.</span>
                <span>
                  {opt.text}
                  {opt.leaves_for && (
                    <span className="choice-kind">
                      leave for {locationNameById(opt.leaves_for).toLowerCase()}
                    </span>
                  )}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
