// Fixture catalog. Each entry is a complete `DebugPanelsProps` (minus
// the callbacks) that exercises a specific engine state. Used by
// DebugPreview.tsx to render every panel in isolation.

import type { TickDecision } from "../../api";
import type { DebugPanelsProps } from "../shared/types";
import { baseDebugScenario, baseScenarioMeta } from "./_base";
import { emptyTicks } from "./empty_session";
import { firstSendTicks } from "./first_send";
import { midBargainTicks } from "./mid_bargain";
import { allBlockedTicks } from "./all_blocked";
import { droppedCanonTicks } from "./dropped_canon";

export type FixtureProps = Omit<DebugPanelsProps, "selection" | "onSelectionChange">;

function fixture(tickHistory: TickDecision[]): FixtureProps {
  return {
    tickHistory,
    scenarioMeta: baseScenarioMeta,
    debugScenario: baseDebugScenario,
  };
}

export const FIXTURES: Record<string, FixtureProps> = {
  empty_session: fixture(emptyTicks),
  first_send: fixture(firstSendTicks),
  mid_bargain: fixture(midBargainTicks),
  all_blocked: fixture(allBlockedTicks),
  dropped_canon: fixture(droppedCanonTicks),
};
