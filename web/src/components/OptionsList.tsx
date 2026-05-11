import { useEffect } from "react";
import { useStore } from "../store";
import { StreamingText } from "./StreamingText";

const ROMAN = ["i.", "ii.", "iii.", "iv."];

/** Renders the 3–4 drafted options. Click an option to select; keyboard
 * shortcut 1–4 selects by index; Enter authorizes. Selected option shows the
 * AUTHORIZED stamp after authorize. */
export function OptionsList() {
  const cycle = useStore((s) => s.cycle);
  const selectOption = useStore((s) => s.selectOption);
  const authorize = useStore((s) => s.authorize);

  useEffect(() => {
    if (!cycle) return;
    function onKey(e: KeyboardEvent) {
      if (cycle?.authorized) return;
      if (["1", "2", "3", "4"].includes(e.key)) {
        selectOption(Number(e.key) - 1);
      }
      if (e.key === "Escape" && cycle?.selected_option_index !== null) {
        selectOption(null);
      }
      if (e.key === "Enter" && cycle?.awaiting_auth && cycle?.selected_option_index !== null) {
        authorize();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [cycle, selectOption, authorize]);

  if (!cycle) return null;

  return (
    <div className="opts">
      {cycle.options.map((opt, idx) => {
        const selected = cycle.selected_option_index === idx;
        const dimmed =
          cycle.authorized
            ? !selected
            : cycle.selected_option_index !== null && !selected;
        return (
          <div
            key={idx}
            data-n={idx + 1}
            onClick={() => selectOption(idx)}
            className={`grid grid-cols-[36px_38px_1fr] gap-2.5 py-4 px-3 -mx-3 cursor-pointer transition-opacity duration-300 items-start relative border-t border-rule-soft border-dashed first:border-t-transparent hover:bg-[rgba(122,31,24,0.04)]`}
            style={{
              opacity: dimmed ? (cycle.authorized ? 0.15 : 0.22) : 1,
              cursor: cycle.authorized ? "default" : "pointer",
            }}
          >
            <div className="text-ink text-[14.5px] leading-relaxed">
              {ROMAN[idx] ?? `${idx + 1}.`}
            </div>
            <div
              className={`text-[14.5px] leading-relaxed font-bold select-none ${
                selected ? "text-stamp" : "text-ink"
              }`}
            >
              {selected ? "[ X ]" : "[  ]"}
            </div>
            <div>
              <div className="text-[15px] font-bold text-ink leading-snug mb-1.5">
                {opt.title}
              </div>
              <div className="text-sm leading-relaxed text-ink-soft" style={{ maxWidth: "56ch" }}>
                <StreamingText
                  words={opt.prose_words}
                  showCaret={!opt.complete}
                  complete={opt.complete}
                />
              </div>
            </div>
            {selected && cycle.authorized && (
              <span
                className="absolute top-2 right-2.5 text-stamp text-[10px] font-bold uppercase tracking-[0.28em] px-3 py-1 border-[1.5px] border-stamp whitespace-nowrap pointer-events-none"
                style={{
                  background: "rgba(122, 31, 24, 0.06)",
                  transform: "rotate(-3.5deg)",
                }}
              >
                AUTHORIZED
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
