// Store-bound shim. Reads the live engine snapshot off the Zustand store
// and forwards it as props to `DebugPanels`. Returns null until both
// scenarioMeta and debugScenario have loaded (boot fetches them in
// parallel; the panel can't render predicates without them).
//
// The dock is mounted only in DEV builds — see App.tsx for the gate.
// Production bundles tree-shake this entire module via the
// `import.meta.env.DEV` check at the mount site.

import { DebugPanels } from "./DebugPanels";
import { useStore } from "../store";

const DOCK_CLASS = "h-full w-[460px] shrink-0 border-l border-border";

export function DebugDock() {
  const tickHistory = useStore((s) => s.tickHistory);
  const scenarioMeta = useStore((s) => s.scenarioMeta);
  const debugScenario = useStore((s) => s.debugScenario);
  const selection = useStore((s) => s.debugSelection);
  const setDebugSelection = useStore((s) => s.setDebugSelection);

  if (!scenarioMeta || !debugScenario) {
    return (
      <aside className={`${DOCK_CLASS} flex items-center justify-center bg-bg`}>
        <span className="text-[11px] italic text-text-muted">debug — loading scenario…</span>
      </aside>
    );
  }
  return (
    <aside className={DOCK_CLASS}>
      <DebugPanels
        tickHistory={tickHistory}
        scenarioMeta={scenarioMeta}
        debugScenario={debugScenario}
        selection={selection}
        onSelectionChange={setDebugSelection}
      />
    </aside>
  );
}
