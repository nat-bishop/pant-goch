// Thin vertical handle that takes the dock's flex slot when the debug
// panel is collapsed. Clicking it flips `debugDockOpen` back to true.
// Mounted only in DEV — see App.tsx.

import { useStore } from "../store";

export function DebugReopenHandle() {
  const setDebugDockOpen = useStore((s) => s.setDebugDockOpen);
  return (
    <button
      type="button"
      onClick={() => setDebugDockOpen(true)}
      aria-label="open debug menu"
      title="open debug menu"
      className="h-full w-6 shrink-0 border-l border-border bg-bg text-text-muted hover:bg-bg-elevated hover:text-text"
    >
      <span
        className="block font-mono text-[10px] uppercase tracking-[0.16em]"
        style={{ writingMode: "vertical-rl" }}
      >
        Debug
      </span>
    </button>
  );
}
