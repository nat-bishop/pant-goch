// Pure-prop debug surface: tab strip + active panel. The single seam
// where DebugDock (store-bound) and DebugPreview (fixture-bound) both
// converge. No store calls, no side effects — inputs arrive via `props`,
// changes leave via `props.onSelectionChange` and the optional `onClose`
// (only DebugDock supplies the latter; fixtures stay close-less).

import clsx from "clsx";
import { TABS } from "./tabs";
import type { DebugPanelsProps } from "./shared/types";

export function DebugPanels(props: DebugPanelsProps) {
  const active = TABS.find((t) => t.id === props.selection.tab) ?? TABS[0]!;
  const ActiveComponent = active.component;
  return (
    <aside className="flex h-full min-h-0 w-full flex-col bg-bg text-text">
      <header className="border-b border-border px-4 pt-3">
        <div className="mb-3 flex items-center justify-between">
          <h1 className="font-mono text-[10px] uppercase tracking-[0.16em] text-text-muted">
            Debug Menu
          </h1>
          {props.onClose && (
            <button
              type="button"
              onClick={props.onClose}
              aria-label="close debug menu"
              title="close debug menu"
              className="rounded border border-border/60 px-2 py-0.5 text-[11px] leading-none text-text-muted hover:text-text"
            >×</button>
          )}
        </div>
        <nav className="flex gap-0">
          {TABS.map((t) => {
            const isActive = t.id === active.id;
            const badge = t.badge?.(props);
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => props.onSelectionChange({ tab: t.id })}
                className={clsx(
                  "-mb-px border-b-2 px-3 py-2 text-[12px] font-medium transition-colors",
                  isActive
                    ? "border-accent text-text"
                    : "border-transparent text-text-muted hover:text-text",
                )}
              >
                {t.label}
                {badge != null && (
                  <span className="ml-1 inline-block min-w-[16px] rounded-full bg-bg-elevated px-1.5 py-0.5 text-center font-mono text-[10px] text-text-muted">
                    {badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </header>
      <section className="min-h-0 flex-1 overflow-y-auto p-4">
        <ActiveComponent {...props} />
      </section>
    </aside>
  );
}
