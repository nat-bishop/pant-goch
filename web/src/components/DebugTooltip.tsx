// Single hover-tooltip primitive used by every debug-panel surface so
// quality names, beat ids, and predicate leaves all share one render.
// Positioned below + right-aligned to the trigger; no collision
// detection, since the panel lives bottom-right and overflow can scroll.

import clsx from "clsx";
import type { ReactNode } from "react";

interface DebugTooltipProps {
  content: ReactNode;
  children: ReactNode;
  className?: string;
  wide?: boolean;
}

export function DebugTooltip({
  content,
  children,
  className,
  wide = false,
}: DebugTooltipProps) {
  return (
    <span className={clsx("group relative inline-block", className)}>
      {children}
      <span
        role="tooltip"
        className={clsx(
          "pointer-events-none absolute right-0 top-full z-[60] mt-1",
          "hidden whitespace-normal rounded-md border border-border bg-bg-elevated px-2 py-1.5",
          "text-[10px] leading-snug text-text shadow-lg",
          "group-hover:block",
          wide ? "w-72" : "w-56",
        )}
      >
        {content}
      </span>
    </span>
  );
}
