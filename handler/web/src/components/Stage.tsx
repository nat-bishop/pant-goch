// Top-level: two-pane grid, world fetch on mount, start/restart handling,
// SSE stream activation, gloss popup root.

import { useEffect, useState } from "react";
import { fetchWorld, postWalk, startSession } from "../api";
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

  // Auto-walk on leaves_for: when the player picks an option with a
  // `leaves_for` destination, the engine awaits POST /walk (after the
  // chronicler runs). The map-click between_scenes flow is reserved for
  // when the player did NOT pre-commit to a destination in fiction.
  //
  // Timing: scene_exit transitions phase → between_scenes, but the
  // walk_future is only created AFTER the chronicler completes its log
  // entry stream. So we gate on (phase = between_scenes) AND (the
  // latest case_file entry has finished streaming).
  const entryClosedKey = useStore((s) => {
    const last = s.case_file[s.case_file.length - 1];
    return last && !last.streaming ? last.scene_id : null;
  });
  useEffect(() => {
    if (phase !== "between_scenes" || entryClosedKey === null) return;
    const { scenes } = useStore.getState();
    const lastScene = scenes[scenes.length - 1];
    if (!lastScene || lastScene.scene_id !== entryClosedKey) return;
    const finalPick =
      [...lastScene.turns].reverse().find((t) => t.player_pick)?.player_pick ?? null;
    if (!finalPick?.leaves_for) return;
    postWalk(finalPick.leaves_for).catch((e) =>
      console.warn("auto-walk failed", e),
    );
  }, [phase, entryClosedKey]);

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
