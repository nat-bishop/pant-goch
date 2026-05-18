// Two-slot crossfader for backdrop swaps. When `currentUrl` changes, the
// inactive slot loads the new image and fades it in over 900ms (CSS
// transition on .world-img opacity). Backdrop URLs that 404 are caught
// by the onError handler and the slot is hidden so the fallback gradient
// shows through.

import { Fragment, useEffect, useState } from "react";

export function ImageSlots({ currentUrl }: { currentUrl: string | null }) {
  const [slotA, setSlotA] = useState<string | null>(currentUrl);
  const [slotB, setSlotB] = useState<string | null>(null);
  const [active, setActive] = useState<"a" | "b">("a");
  const [okA, setOkA] = useState(true);
  const [okB, setOkB] = useState(true);

  useEffect(() => {
    if (!currentUrl) {
      // Clear both slots when no backdrop should be shown.
      setSlotA(null);
      setSlotB(null);
      return;
    }
    if (active === "a" && slotA !== currentUrl) {
      setSlotB(currentUrl);
      setOkB(true);
      const t = setTimeout(() => {
        setActive("b");
        setSlotA(null);
      }, 50);
      return () => clearTimeout(t);
    }
    if (active === "b" && slotB !== currentUrl) {
      setSlotA(currentUrl);
      setOkA(true);
      const t = setTimeout(() => {
        setActive("a");
        setSlotB(null);
      }, 50);
      return () => clearTimeout(t);
    }
  }, [currentUrl, active, slotA, slotB]);

  const renderSlot = (
    url: string | null,
    ok: boolean,
    isActive: boolean,
    onErrorFn: () => void,
  ) => {
    if (!url || !ok) return null;
    return (
      <Fragment>
        {/* Hidden <img> probes for 404; on error, hide the layer. */}
        <img
          src={url}
          alt=""
          style={{ display: "none" }}
          onError={onErrorFn}
        />
        <div
          className="world-img"
          style={{
            backgroundImage: `url("${url}")`,
            opacity: isActive ? 1 : 0,
          }}
        />
      </Fragment>
    );
  };

  return (
    <Fragment>
      {/* Always-on fallback under both slots. */}
      <div className="world-img-fallback" />
      {renderSlot(slotA, okA, active === "a", () => setOkA(false))}
      {renderSlot(slotB, okB, active === "b", () => setOkB(false))}
    </Fragment>
  );
}
