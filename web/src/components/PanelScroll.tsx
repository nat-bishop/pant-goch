// PanelScroll: right-hand vellum panel. Holds the scrollable region —
// either the Frontispiece (pre-start) or the running chapters — plus the
// transient error notice and the bottom click-to-complete reveal hint.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useStore, useRevealing } from "../store";
import { Chapter } from "./Chapter";
import { Frontispiece } from "./Frontispiece";
import { CatchUpLink } from "./CatchUpLink";
import { buildGlossIndex } from "./glosses";

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
  const revealing = useRevealing();

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const glossIndex = useMemo(
    () => (world ? buildGlossIndex(world.glosses) : null),
    [world],
  );

  // Follow-the-live-edge. `following` means "auto-scroll to keep the live
  // edge in view." It flips off only on a genuine user scroll *gesture*
  // (wheel / trackpad / scroll keys), NOT on scrollTop deltas: the
  // page-turn scrolls programmatically and the choices-collapse shrinks
  // scrollHeight (clamping scrollTop), both of which a delta test misreads
  // as the reader moving. `followingRef` mirrors state for synchronous
  // reads in the page-turn gate.
  const [following, setFollowing] = useState(true);
  const followingRef = useRef(true);
  const setFollow = useCallback((v: boolean) => {
    if (followingRef.current === v) return;
    followingRef.current = v;
    setFollowing(v);
  }, []);

  const catchUp = useCallback(() => {
    const panel = scrollRef.current;
    if (!panel) return;
    // Explicit smooth overrides the element's scroll-behavior:auto.
    panel.scrollTo({
      top: panel.scrollHeight - panel.clientHeight,
      behavior: "smooth",
    });
    setFollow(true);
  }, [setFollow]);

  // User scroll-intent → disengage on an upward gesture ("I'm re-reading"),
  // re-engage on a downward one once the live edge is back in view.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const liveEdgeVisible = () => {
      const ref =
        el.querySelector<HTMLElement>(".chapter-prose .para.current") ??
        el.querySelector<HTMLElement>(".choices-wrap, .freetext-wrap");
      if (!ref) return true;
      return (
        ref.getBoundingClientRect().bottom <=
        el.getBoundingClientRect().bottom + 40
      );
    };
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < -2) setFollow(false);
      else if (e.deltaY > 2 && liveEdgeVisible()) setFollow(true);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowUp" || e.key === "PageUp" || e.key === "Home") {
        setFollow(false);
      } else if (
        (e.key === "ArrowDown" || e.key === "PageDown" || e.key === "End") &&
        liveEdgeVisible()
      ) {
        setFollow(true);
      }
    };
    el.addEventListener("wheel", onWheel, { passive: true });
    el.addEventListener("keydown", onKey);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("keydown", onKey);
    };
  }, [setFollow]);

  // Backdrop centring — which chapter sits at the panel's middle. Rebinds
  // only when a scene is added; reads the live store each tick, so
  // depending on scenes.length (not scenes) avoids rebinding per word.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || scenes.length === 0) return;
    const onScroll = () => {
      const live = useStore.getState().scenes;
      const center = el.getBoundingClientRect().top + el.clientHeight * 0.5;
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

  // Re-engage follow whenever fresh prose is about to stream — the player
  // just committed a pick / typed input (attachPick sets in_scene) or a new
  // scene started. They're at the live edge and want to follow the reply.
  // Fires once per transition into in_scene (prose_word keeps it in_scene
  // without re-firing). A mid-stream scroll-up still disengages via the
  // direction detector.
  useEffect(() => {
    if (phase === "in_scene") setFollow(true);
  }, [phase, setFollow]);

  // When prose closes, the choices / free-text block appears below the last
  // paragraph — which the page-turn (it only tracks the streaming edge) may
  // have left near the bottom. Nudge down just enough to reveal it.
  useEffect(() => {
    if (phase !== "awaiting_pick" && phase !== "awaiting_free_text") return;
    if (!followingRef.current) return;
    const panel = scrollRef.current;
    if (!panel) return;
    const t = setTimeout(() => {
      const block = panel.querySelector(".choices-wrap, .freetext-wrap");
      if (!block) return;
      const overflowBelow =
        block.getBoundingClientRect().bottom - panel.getBoundingClientRect().bottom;
      if (overflowBelow > 0) {
        panel.scrollTo({ top: panel.scrollTop + overflowBelow + 24, behavior: "auto" });
      }
    }, 60);
    return () => clearTimeout(t);
  }, [phase]);

  // Position the newest chapter's header at the top on append — instantly
  // (no animated scroll while reading). A new scene re-engages follow.
  useEffect(() => {
    if (scenes.length === 0) return;
    setFollow(true);
    const last = scenes[scenes.length - 1];
    const node = document.getElementById(`ch-${last.scene_id}`);
    if (!node) return;
    const t = setTimeout(
      () => node.scrollIntoView({ behavior: "auto", block: "start" }),
      80,
    );
    return () => clearTimeout(t);
  }, [scenes.length, setFollow]);

  return (
    <div className="panel">
      {/* Idle errors render inside the Frontispiece; avoid showing twice. */}
      {error && phase !== "idle" && <div className="panel-error">{error}</div>}
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
              const isCurrent = i === scenes.length - 1;
              const fromLocationName =
                i > 0 ? scenes[i - 1].location_name : null;
              return (
                <Chapter
                  key={scene.scene_id}
                  scene={scene}
                  chapterNo={i + 1}
                  isCurrent={isCurrent}
                  fromLocationName={fromLocationName}
                  glossIndex={glossIndex}
                  following={following}
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
            {!following && phase !== "game_over" && phase !== "stranded" && (
              <CatchUpLink onCatchUp={catchUp} />
            )}
          </>
        )}
      </div>
      {revealing && following && (
        <div className="reveal-hint" aria-hidden="true">
          click to complete
        </div>
      )}
    </div>
  );
}
