// Chapter = one scene in the running case file. Header (Chapter N ·
// Location), prose paragraphs streamed/segmented, choices block once
// streaming closes.
//
// Page-turn scroll: when a paragraph's bottom would sit past 68% of the
// panel height, smooth-scroll the panel so the previous paragraph sits
// at top with ~56px padding — BUT only when the player is "following"
// the live edge. If they have scrolled up to re-read, the scroll is
// suppressed; PanelScroll shows a "↓ catch up" link instead.

import { useEffect, useRef, useState } from "react";
import { postFreeText, postPick } from "../api";
import { useStore, type SceneSnapshot } from "../store";
import { StreamingProse } from "./StreamingProse";
import { ChoicesBlock } from "./ChoicesBlock";
import { FreeTextInput } from "./FreeTextInput";
import type { GlossIndex } from "./glosses";

function romanize(n: number): string {
  const map = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"];
  return map[n] ?? String(n);
}

export function Chapter({
  scene,
  chapterNo,
  isCurrent,
  fromLocationName,
  glossIndex,
  following,
}: {
  scene: SceneSnapshot;
  chapterNo: number;
  isCurrent: boolean;
  fromLocationName: string | null;
  glossIndex: GlossIndex | null;
  following: boolean;
}) {
  const sectionRef = useRef<HTMLElement | null>(null);
  const followingRef = useRef(following);
  followingRef.current = following;
  const fastForward = useStore((s) => s.fastForward);
  const attachPick = useStore((s) => s.attachPick);
  const phase = useStore((s) => s.phase);
  const freeTextRequest = useStore((s) => s.freeTextRequest);
  const [pickBusy, setPickBusy] = useState(false);
  const [freeTextBusy, setFreeTextBusy] = useState(false);

  const lastTurn = scene.turns[scene.turns.length - 1];
  const streamingThisChapter = isCurrent && lastTurn?.streaming;
  const wordCount = scene.turns.reduce((n, t) => n + t.words.length, 0);

  usePageTurn(sectionRef, wordCount, streamingThisChapter, followingRef);

  const liveClosed = isCurrent && !!lastTurn && !lastTurn.streaming;
  const showChoicesBlock = liveClosed && lastTurn.options.length > 0;
  const freeTextReady =
    liveClosed &&
    phase === "awaiting_free_text" &&
    freeTextRequest !== null &&
    !lastTurn.player_pick;

  const onPick = async (idx: number) => {
    if (pickBusy || !lastTurn || lastTurn.player_pick) return;
    setPickBusy(true);
    attachPick(lastTurn.options[idx]);
    try {
      await postPick(idx);
    } finally {
      setPickBusy(false);
    }
  };

  const onFreeTextSubmit = async (text: string) => {
    if (freeTextBusy) return;
    setFreeTextBusy(true);
    attachPick({ text });
    try {
      await postFreeText(text);
    } finally {
      setFreeTextBusy(false);
    }
  };

  return (
    <section
      className="chapter"
      id={`ch-${scene.scene_id}`}
      ref={sectionRef}
      role="article"
    >
      <header className="chapter-head">
        <span className="chapter-no">Chapter {romanize(chapterNo)}</span>
        <h2 className="chapter-title">{scene.location_name}</h2>
        {fromLocationName && (
          <span className="chapter-from">from {fromLocationName}</span>
        )}
      </header>
      <div className="chapter-prose">
        {scene.turns.map((turn, i) => {
          const isLastTurn = i === scene.turns.length - 1;
          // .player-pick is the historical log mark for a picked turn.
          // Hidden for the live picked turn because the still-mounted
          // ChoicesBlock (in chosen state) already represents it. Once
          // the next turn streams in, the ChoicesBlock unmounts and
          // this mark takes its place.
          const showPickMark =
            !!turn.player_pick && !(isCurrent && isLastTurn);
          return (
            <div key={i}>
              <StreamingProse
                words={turn.words}
                streaming={turn.streaming && isCurrent && isLastTurn}
                fastForward={fastForward && isCurrent}
                glossIndex={glossIndex}
              />
              {showPickMark && turn.player_pick && (
                <div className="player-pick">» {turn.player_pick.text}</div>
              )}
            </div>
          );
        })}
        {showChoicesBlock && (
          <ChoicesBlock
            options={lastTurn.options}
            picked={lastTurn.player_pick}
            onPick={onPick}
            busy={pickBusy}
          />
        )}
        {freeTextReady && freeTextRequest && (
          <FreeTextInput
            prompt={freeTextRequest.prompt}
            onSubmit={onFreeTextSubmit}
            busy={freeTextBusy}
          />
        )}
      </div>
    </section>
  );
}

function usePageTurn(
  sectionRef: React.RefObject<HTMLElement>,
  wordCount: number,
  active: boolean | undefined,
  followingRef: React.MutableRefObject<boolean>,
) {
  useEffect(() => {
    if (!active) return;
    // Respect the player when they have scrolled up to re-read. The
    // page-turn would yank them back to the live edge; suppress it. The
    // CatchUpLink in PanelScroll surfaces "↓ catch up" instead.
    if (!followingRef.current) return;
    const section = sectionRef.current;
    if (!section) return;
    const panel = section.closest(".panel-scroll") as HTMLElement | null;
    if (!panel) return;
    const paras = section.querySelectorAll<HTMLElement>(".chapter-prose .para");
    if (paras.length < 2) return;
    const current = paras[paras.length - 1];
    if (!current.classList.contains("current")) return;
    const prior = paras[paras.length - 2];
    const panelRect = panel.getBoundingClientRect();
    const priorRect = prior.getBoundingClientRect();
    const priorBottomInPanel = priorRect.bottom - panelRect.top;
    if (priorBottomInPanel > panelRect.height * 0.68) {
      const padding = 56;
      const newTop = panel.scrollTop + (priorRect.top - panelRect.top) - padding;
      // CSS `scroll-behavior: smooth` on `.panel-scroll` handles the
      // animation (with a @media (prefers-reduced-motion) override to
      // auto), so we pass behavior:auto here to defer to CSS rather
      // than stacking a second smooth-scroll on top of it.
      panel.scrollTo({ top: newTop, behavior: "auto" });
    }
  }, [wordCount, active, sectionRef, followingRef]);
}
