// Top-level: two-pane grid, world fetch on mount, start/restart handling,
// SSE stream activation, gloss popup root.

import { useEffect, useState } from "react";
import { fetchWorld, startSession } from "../api";
import { useStream } from "../hooks/useStream";
import { useStore } from "../store";
import { GlossPopup } from "./GlossPopup";
import { PanelScroll } from "./PanelScroll";
import { WorldPane } from "./WorldPane";

export function Stage() {
  const phase = useStore((s) => s.phase);
  const setWorld = useStore((s) => s.setWorld);
  const resetForRestart = useStore((s) => s.resetForRestart);
  const triggerFastForward = useStore((s) => s.triggerFastForward);

  const [streaming, setStreaming] = useState(false);
  useStream(streaming);

  // Load world metadata once on mount. The frontispiece is shown
  // regardless; we just need glosses/locations/npcs ready before the
  // first scene streams.
  useEffect(() => {
    fetchWorld()
      .then((w) => setWorld(w))
      .catch((e) => console.warn("/world failed", e));
  }, [setWorld]);

  // `starting` gates the begin/restart buttons between the click and
  // the first scene event. Without it the user sees no feedback during
  // multi-second LLM warm-up and may click again, cancelling the prior
  // play_task and surfacing a stranded session.
  const [starting, setStarting] = useState(false);
  useEffect(() => {
    if (starting && phase !== "idle") setStarting(false);
  }, [starting, phase]);

  const onStart = async () => {
    if (starting) return;
    setStarting(true);
    // Tear down any active stream first so the new SSE opens against
    // the fresh session — without this a stale consumer receives the
    // session_done from cancellation.
    setStreaming(false);
    resetForRestart();
    try {
      await startSession();
    } catch (e) {
      console.warn("start failed", e);
      setStarting(false);
      return;
    }
    setTimeout(() => setStreaming(true), 0);
  };

  // Click-to-reveal: when the user clicks the panel during a streaming
  // turn, snap visible words to full opacity. The space key does the
  // same when the panel has focus. Server pacing is unaffected — this
  // collapses the *animation*, not the event stream.
  const onPanelClick = () => {
    if (phase === "in_scene") triggerFastForward();
  };
  const onPanelKey = (e: React.KeyboardEvent) => {
    if (e.key === " " && phase === "in_scene") {
      e.preventDefault();
      triggerFastForward();
    }
  };

  return (
    <div className="stage">
      <WorldPane />
      <PanelScroll
        starting={starting}
        onStart={onStart}
        onPanelClick={onPanelClick}
        onPanelKey={onPanelKey}
      />
      <GlossPopup />
    </div>
  );
}
