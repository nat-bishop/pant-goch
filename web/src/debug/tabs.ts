// Tab registry. Adding/deleting a tab touches one entry here plus one
// folder under web/src/debug/. The dock iterates this list to build the
// tab strip and pick the active panel — no hard-coded if/else per tab.

import type { ComponentType } from "react";
import { BeatsPanel } from "./beats/BeatsPanel";
import { CanonPanel } from "./canon/CanonPanel";
import { PacingPanel } from "./pacing/PacingPanel";
import { TickLogPanel } from "./ticks/TickLogPanel";
import { TimelinePanel } from "./timeline/TimelinePanel";
import type { PanelProps } from "./shared/types";
import type { DebugTab } from "../store";

export interface TabDef {
  id: DebugTab;
  label: string;
  component: ComponentType<PanelProps>;
  // Optional badge derived from props — rendered next to the label.
  // Keep cheap; called on every render.
  badge?: (p: PanelProps) => number | null;
}

export const TABS: TabDef[] = [
  { id: "pacing", label: "Pacing", component: PacingPanel },
  {
    id: "ticks",
    label: "Tick log",
    component: TickLogPanel,
    badge: (p) => p.tickHistory.length || null,
  },
  {
    id: "canon",
    label: "Canon",
    component: CanonPanel,
    badge: (p) => {
      const c = p.tickHistory[0]?.player_canon;
      return c ? Object.keys(c).length || null : null;
    },
  },
  {
    id: "beats",
    label: "Beats",
    component: BeatsPanel,
    badge: (p) => Object.keys(p.debugScenario.beats).length || null,
  },
  { id: "timeline", label: "Timeline", component: TimelinePanel },
];
