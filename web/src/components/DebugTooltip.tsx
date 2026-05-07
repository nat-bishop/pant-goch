// Single hover-tooltip primitive used by every debug-panel surface so
// quality names, beat ids, and predicate leaves all share one render.
// Rendered into document.body via portal with position: fixed — the
// debug panel's inner scroll container has overflow-y-auto, which (per
// CSS spec) coerces overflow-x to auto too, so an in-tree absolutely
// positioned tooltip gets clipped at the panel edges. The portal
// escapes that container entirely.

import clsx from "clsx";
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

interface DebugTooltipProps {
  content: ReactNode;
  children: ReactNode;
  className?: string;
  wide?: boolean;
}

const EDGE_PAD = 8;
const TRIGGER_GAP = 4;

export function DebugTooltip({
  content,
  children,
  className,
  wide = false,
}: DebugTooltipProps) {
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number } | null>(null);

  // Measure after the tooltip mounts so we can read its real size, then
  // place it below + right-aligned to the trigger by default and flip
  // off whichever viewport edge it would otherwise cross.
  useLayoutEffect(() => {
    if (!open) {
      setCoords(null);
      return;
    }
    const trigger = triggerRef.current;
    const tooltip = tooltipRef.current;
    if (!trigger || !tooltip) return;
    const t = trigger.getBoundingClientRect();
    const tt = tooltip.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let left = t.right - tt.width;
    if (left < EDGE_PAD) left = EDGE_PAD;
    if (left + tt.width > vw - EDGE_PAD) left = vw - EDGE_PAD - tt.width;

    let top = t.bottom + TRIGGER_GAP;
    if (top + tt.height > vh - EDGE_PAD) {
      top = t.top - TRIGGER_GAP - tt.height;
    }

    setCoords({ top, left });
  }, [open, content, wide]);

  return (
    <span
      ref={triggerRef}
      className={clsx("inline-block", className)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {children}
      {open &&
        createPortal(
          <span
            ref={tooltipRef}
            role="tooltip"
            style={{
              position: "fixed",
              top: coords?.top ?? -9999,
              left: coords?.left ?? -9999,
              visibility: coords ? "visible" : "hidden",
            }}
            className={clsx(
              "pointer-events-none z-[60] whitespace-normal",
              "rounded-md border border-border bg-bg-elevated px-2 py-1.5",
              "text-[10px] leading-snug text-text shadow-lg",
              wide ? "w-72" : "w-56",
            )}
          >
            {content}
          </span>,
          document.body,
        )}
    </span>
  );
}
