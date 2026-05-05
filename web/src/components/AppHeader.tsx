import { useStore } from "../store";
import { ThemeToggle } from "./ThemeToggle";

export function AppHeader() {
  const scenario = useStore((s) => s.scenario);
  const ended = useStore((s) => s.state?.ended ?? false);

  return (
    <header className="flex h-12 shrink-0 items-center justify-between bg-bg-sidebar px-4">
      <div className="flex items-center gap-2.5">
        <Mark />
        <span className="text-[12px] font-semibold uppercase tracking-[0.16em] text-text">
          {scenario?.title ?? "Continuity Program"}
        </span>
      </div>
      <div className="flex items-center gap-3">
        {ended && (
          <span className="rounded-full border border-meter-hot/40 bg-meter-hot/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-meter-hot">
            Transition Complete
          </span>
        )}
        <ThemeToggle />
      </div>
    </header>
  );
}

function Mark() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden>
      <defs>
        <linearGradient id="mark-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="var(--accent)" />
          <stop offset="100%" stopColor="var(--accent-hover)" />
        </linearGradient>
      </defs>
      <path
        d="M12 2 L21 7 L21 17 L12 22 L3 17 L3 7 Z"
        fill="url(#mark-grad)"
      />
      <path
        d="M12 6 L17 9 L17 15 L12 18 L7 15 L7 9 Z"
        fill="var(--bg-sidebar)"
      />
    </svg>
  );
}
