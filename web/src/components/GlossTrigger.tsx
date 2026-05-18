// Inline phrase that opens a hover/focus popup. Captures the trigger's
// bounding rect at hover time and hands it to the store so the popup
// can position itself relative to it (over the left photo, not the
// prose panel).

import { useCallback, useRef } from "react";
import { useStore } from "../store";

export function GlossTrigger({
  text,
  glossId,
}: {
  text: string;
  glossId: string;
}) {
  const showGloss = useStore((s) => s.showGloss);
  const hideGloss = useStore((s) => s.hideGloss);
  const ref = useRef<HTMLSpanElement | null>(null);

  const open = useCallback(() => {
    if (!ref.current) return;
    showGloss(glossId, ref.current.getBoundingClientRect());
  }, [glossId, showGloss]);

  const close = useCallback(() => {
    hideGloss();
  }, [hideGloss]);

  return (
    <span
      ref={ref}
      className="gloss-trigger"
      data-gloss-id={glossId}
      tabIndex={0}
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={close}
    >
      {text}
    </span>
  );
}
