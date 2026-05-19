// PanelScroll: right-hand vellum panel. Owns the panel-bar (case-file
// label + restart confirmation + connection banner) and the scrollable
// region containing either the Frontispiece (pre-start) or the chapters.

import { useEffect, useRef, useState } from "react";
import { useStore } from "../store";
import { Chapter } from "./Chapter";
import { Frontispiece } from "./Frontispiece";
import { buildGlossIndex } from "./glosses";
import { useMemo } from "react";

export function PanelScroll({
  starting,
  onStart,
  onPanelKey,
  onPanelClick,
}: {
  starting: boolean;
  onStart: () => void;
  onPanelKey: (e: React.KeyboardEvent) => void;
  onPanelClick: () => void;
}) {
  const phase = useStore((s) => s.phase);
  const scenes = useStore((s) => s.scenes);
  const error = useStore((s) => s.error);
  const gameOver = useStore((s) => s.game_over_ending);
  const world = useStore((s) => s.world);
  const setCurrentChapterIdx = useStore((s) => s.setCurrentChapterIdx);
  const resetForRestart = useStore((s) => s.resetForRestart);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const glossIndex = useMemo(
    () => (world ? buildGlossIndex(world.glosses) : null),
    [world],
  );

  // Track which chapter is centred so the world pane swaps backdrops.
  // Effect re-binds only when a scene is added — scene_id lookup inside
  // the handler stays current via the store ref each scroll tick, so
  // depending on `scenes.length` (not `scenes`) avoids rebinding the
  // listener on every prose_chunk.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || scenes.length === 0) return;
    const onScroll = () => {
      const live = useStore.getState().scenes;
      const panelRect = el.getBoundingClientRect();
      const center = panelRect.top + panelRect.height * 0.5;
      let best = 0;
      let bestDist = Infinity;
      live.forEach((s, i) => {
        const node = document.getElementById(`ch-${s.scene_id}`);
        if (!node) return;
        const r = node.getBoundingClientRect();
        const mid = (r.top + r.bottom) / 2;
        const d = Math.abs(mid - center);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      });
      setCurrentChapterIdx(best);
    };
    onScroll();
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [scenes.length, setCurrentChapterIdx]);

  // Auto-scroll into newest chapter on append.
  useEffect(() => {
    if (scenes.length === 0) return;
    const last = scenes[scenes.length - 1];
    const node = document.getElementById(`ch-${last.scene_id}`);
    if (node) {
      // Delay so the chapter has mounted.
      const t = setTimeout(
        () => node.scrollIntoView({ behavior: "smooth", block: "start" }),
        80,
      );
      return () => clearTimeout(t);
    }
  }, [scenes.length]);

  return (
    <div className="panel">
      <PanelBar
        starting={starting}
        onRestart={async () => {
          resetForRestart();
          onStart();
        }}
        error={error}
        sceneCount={scenes.length}
        hasSession={phase !== "idle"}
      />
      <div
        className="panel-scroll"
        ref={scrollRef}
        tabIndex={0}
        onKeyDown={onPanelKey}
        onClick={onPanelClick}
      >
        {phase === "idle" ? (
          <Frontispiece onStart={onStart} starting={starting} error={error} />
        ) : (
          <>
            {scenes.map((scene, i) => {
              const fromLocationName =
                i > 0 ? scenes[i - 1].location_name : null;
              return (
                <Chapter
                  key={scene.scene_id}
                  scene={scene}
                  chapterNo={i + 1}
                  isCurrent={i === scenes.length - 1}
                  fromLocationName={fromLocationName}
                  glossIndex={glossIndex}
                />
              );
            })}
            {phase === "game_over" && (
              <div className="colophon">
                <div className="colophon-rule" />
                — {gameOver?.replace(/_/g, " ")} —
              </div>
            )}
            {phase === "stranded" && (
              <div className="colophon">
                <div className="colophon-rule" />
                the road ends here
              </div>
            )}
            {phase !== "game_over" && phase !== "stranded" && (
              <div className="colophon">
                <div className="colophon-rule" />
                in progress
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function PanelBar({
  starting,
  onRestart,
  error,
  sceneCount,
  hasSession,
}: {
  starting: boolean;
  onRestart: () => void;
  error: string | null;
  sceneCount: number;
  hasSession: boolean;
}) {
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    if (!confirming) return;
    const t = setTimeout(() => setConfirming(false), 4000);
    return () => clearTimeout(t);
  }, [confirming]);

  const onClick = () => {
    if (starting) return;
    if (!confirming && hasSession) {
      setConfirming(true);
      return;
    }
    setConfirming(false);
    onRestart();
  };

  return (
    <div className="panel-bar">
      <span>case file · m. pritchard · 2026</span>
      {error && <span className="panel-bar-banner">{error}</span>}
      <div className="panel-bar-meta">
        <span>
          {hasSession
            ? `${sceneCount} ${sceneCount === 1 ? "entry" : "entries"}`
            : "—"}
        </span>
        <span className="panel-bar-sep" />
        <button
          onClick={onClick}
          disabled={starting}
          className={confirming ? "confirming" : ""}
        >
          {confirming ? "↺ confirm restart?" : "↺ restart"}
        </button>
      </div>
    </div>
  );
}
