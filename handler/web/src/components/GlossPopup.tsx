// Floating gloss popup. One instance rendered at the stage root; the
// store holds the active gloss + trigger rect. Positioned to the LEFT
// of the trigger so it floats over the photograph, not the prose.

import { useEffect } from "react";
import { useStore } from "../store";

const POPUP_W = 280;
const POPUP_H = 130;
const GAP = 14;

export function GlossPopup() {
  const activeGloss = useStore((s) => s.activeGloss);
  const world = useStore((s) => s.world);
  const hideGloss = useStore((s) => s.hideGloss);

  // Escape dismisses.
  useEffect(() => {
    if (!activeGloss) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") hideGloss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeGloss, hideGloss]);

  if (!activeGloss || !world) return null;
  const gloss = world.glosses.find((g) => g.id === activeGloss.glossId);
  if (!gloss) return null;

  const { rect } = activeGloss;
  const above = rect.top > POPUP_H + GAP + 24;
  const top = above ? rect.top - POPUP_H - GAP : rect.bottom + GAP;
  // Float left into the photograph pane (the LEFT side of the viewport).
  let left = Math.max(20, rect.left - POPUP_W * 0.6);
  left = Math.min(left, window.innerWidth - POPUP_W - 20);

  return (
    <div
      className={"gloss-popup" + (above ? " above" : " below")}
      role="tooltip"
      style={{
        left: `${left}px`,
        top: `${top}px`,
        width: `${POPUP_W}px`,
      }}
      onMouseEnter={() => useStore.getState().showGloss(activeGloss.glossId, rect)}
      onMouseLeave={hideGloss}
    >
      <div className="gloss-popup-name">{gloss.name}</div>
      <div className="gloss-popup-body">{gloss.body}</div>
      {gloss.cross_ref && (
        <div className="gloss-popup-cross">↳ {gloss.cross_ref}</div>
      )}
    </div>
  );
}
