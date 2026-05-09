// Shared prop contract for debug panels. Every tab takes the same bundle
// of derived data + selection state — keeps the dock<->fixture seam thin
// and lets `tabs.ts` declare a uniform `component: (props: PanelProps) => JSX.Element`
// without per-tab generics.
//
// The actual store bridging lives in DebugDock.tsx. Fixtures pass these
// props directly to DebugPanels; both paths converge on the same render.

import type { DebugScenario, ScenarioMeta, TickDecision } from "../../api";
import type { DebugSelection } from "../../store";

export interface DebugPanelsProps {
  tickHistory: TickDecision[];          // newest-first
  scenarioMeta: ScenarioMeta;
  debugScenario: DebugScenario;
  selection: DebugSelection;
  onSelectionChange: (patch: Partial<DebugSelection>) => void;
  // Optional: only DebugDock supplies this. Fixtures (DebugPreview) leave
  // it undefined so cells don't render a close button.
  onClose?: () => void;
}

export type PanelProps = DebugPanelsProps;
