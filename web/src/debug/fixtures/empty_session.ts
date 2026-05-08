// Empty session — just-booted, no /send yet. Verifies every panel
// renders empty-states without crashing on null lookups.

import type { TickDecision } from "../../api";

export const emptyTicks: TickDecision[] = [];
