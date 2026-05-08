// DEV-only fixture harness. Mounted in place of <App /> when the URL
// carries `?debug-preview=1`. Renders every fixture as a self-contained
// dock so each tab + state combo can be eyeballed without booting the
// full stack.
//
// Each fixture gets its own isolated `selection` state — clicking around
// in one preview cell doesn't disturb the others.

import { useState } from "react";
import { DebugPanels } from "./DebugPanels";
import { FIXTURES, type FixtureProps } from "./fixtures";
import type { DebugSelection } from "../store";

const DEFAULT_SELECTION: DebugSelection = {
  tab: "pacing", quality: null, beat: null, beatsView: "list",
};

export function DebugPreview() {
  return (
    <div className="min-h-screen bg-bg p-4 text-text">
      <header className="mb-4">
        <h1 className="font-mono text-[14px] uppercase tracking-[0.16em] text-text-muted">
          Debug Dock — Fixture Preview
        </h1>
        <p className="mt-1 text-[12px] text-text-muted">
          Each cell renders the same DebugPanels component against a different
          fixture. Drop the `?debug-preview=1` query param to switch back to
          the live app.
        </p>
      </header>
      <div className="grid gap-4 xl:grid-cols-2">
        {Object.entries(FIXTURES).map(([name, fx]) => (
          <FixtureCell key={name} name={name} fixture={fx} />
        ))}
      </div>
    </div>
  );
}

function FixtureCell({ name, fixture }: { name: string; fixture: FixtureProps }) {
  const [selection, setSelection] = useState<DebugSelection>(DEFAULT_SELECTION);
  return (
    <section className="overflow-hidden rounded border border-border bg-bg">
      <div className="border-b border-border bg-bg-elevated/50 px-3 py-1.5">
        <span className="font-mono text-[10px] uppercase tracking-wider text-text-muted">
          {name}
        </span>
      </div>
      <div className="h-[640px]">
        <DebugPanels
          {...fixture}
          selection={selection}
          onSelectionChange={(patch) => setSelection((s) => ({ ...s, ...patch }))}
        />
      </div>
    </section>
  );
}
