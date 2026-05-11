import { useStore } from "../store";

/** Authorize button + hint, below the paper. Disabled until a selection is
 * made; flips to "Continue to cycle N+1" once the dispatch completes. */
export function Authorize() {
  const cycle = useStore((s) => s.cycle);
  const authorize = useStore((s) => s.authorize);

  if (!cycle) return null;

  const canAuth = cycle.awaiting_auth && cycle.selected_option_index !== null && !cycle.authorized;
  const dispatchReady = cycle.authorized && cycle.dispatch_complete;
  const selectedTitle =
    cycle.selected_option_index !== null
      ? cycle.options[cycle.selected_option_index]?.title
      : null;

  let hint: React.ReactNode = "— select a course above —";
  let hintClass = "text-chrome-dim";
  if (cycle.authorized && cycle.dispatch_complete) {
    hint = (
      <>
        cycle complete <span className="text-stamp-soft ml-3.5">— continue when ready</span>
      </>
    );
    hintClass = "text-chrome-bright";
  } else if (cycle.authorized) {
    hint = (
      <>
        course authorized <span className="text-stamp-soft ml-3.5">· awaiting dispatch</span>
      </>
    );
    hintClass = "text-chrome-bright";
  } else if (canAuth && selectedTitle) {
    hint = (
      <>
        selected <span className="text-stamp-soft ml-3.5">· {selectedTitle}</span>
      </>
    );
    hintClass = "text-chrome-bright";
  }

  const buttonLabel = dispatchReady ? "Continue ▸" : "Authorize ▸";

  return (
    <div className="flex flex-col items-center gap-4 animate-paper-rise">
      <div className={`text-[10.5px] uppercase tracking-[0.3em] ${hintClass} min-h-[14px]`}>
        {hint}
      </div>
      <button
        onClick={canAuth ? authorize : undefined}
        disabled={!canAuth && !dispatchReady}
        className={`bg-transparent px-11 py-4 font-inherit text-xs uppercase font-bold tracking-[0.4em] transition-all duration-200 ${
          canAuth || dispatchReady
            ? "border-[1.5px] border-stamp text-stamp-soft cursor-pointer hover:bg-stamp hover:text-paper"
            : "border-[1.5px] border-chrome-faint text-chrome-dim cursor-not-allowed"
        }`}
      >
        {buttonLabel}
      </button>
    </div>
  );
}
