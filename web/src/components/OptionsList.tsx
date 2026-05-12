import { useEffect } from "react";
import { useStore } from "../store";
import { StreamingText } from "./StreamingText";

const ROMAN = ["i.", "ii.", "iii.", "iv."];
const SLOT_INDICES = [0, 1, 2, 3];

interface Props {
  /** Forwarded from Paper so the call site can gate behavior; currently
   * unused inside the list because the parent .opts-block already fades
   * the whole block in (no per-row stagger). */
  visible: boolean;
}

export function OptionsList(_: Props) {
  const cycle = useStore((s) => s.cycle);
  const authorized = cycle?.authorized ?? false;
  const awaitingAuth = cycle?.awaiting_auth ?? false;
  const selectedIdx = cycle?.selected_option_index ?? null;
  const selectOption = useStore((s) => s.selectOption);
  const authorize = useStore((s) => s.authorize);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (authorized) return;
      if (["1", "2", "3", "4"].includes(e.key)) {
        selectOption(Number(e.key) - 1);
      }
      if (e.key === "Escape" && selectedIdx !== null) {
        selectOption(null);
      }
      if (e.key === "Enter" && awaitingAuth && selectedIdx !== null) {
        void authorize();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [authorized, awaitingAuth, selectedIdx, selectOption, authorize]);

  if (!cycle) return null;

  const hasSel = selectedIdx !== null;
  const optsClass = `opts ${hasSel ? "has-sel" : ""} ${authorized ? "authorized" : ""}`;

  // Render 4 fixed slots so the page height is stable from the moment the
  // options block fades in. Slots not yet populated by the drafter render
  // as hidden placeholders that still reserve their min-height.
  return (
    <div className={optsClass}>
      {SLOT_INDICES.map((idx) => {
        const opt = cycle.options[idx];
        const isPlaceholder = !opt;
        const selected = !isPlaceholder && selectedIdx === idx;
        return (
          <div
            key={idx}
            data-n={idx + 1}
            onClick={
              isPlaceholder || authorized
                ? undefined
                : () => selectOption(idx)
            }
            className={`opt-row ${selected ? "sel" : ""} ${isPlaceholder ? "placeholder" : ""}`}
          >
            {isPlaceholder ? null : (
              <>
                <span className="opt-n">{ROMAN[idx] ?? `${idx + 1}.`}</span>
                <span className="opt-check">{selected ? "[ × ]" : "[   ]"}</span>
                <span className="opt-body">
                  <span className="opt-title">{opt!.title}.</span>{" "}
                  <span className="opt-prose">
                    <StreamingText
                      words={opt!.prose_words}
                      showCaret={!opt!.complete}
                      complete={opt!.complete}
                    />
                  </span>
                </span>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
