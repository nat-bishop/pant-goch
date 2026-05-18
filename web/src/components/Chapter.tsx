// Chapter = one scene in the running case file. Header (Chapter N ·
// Location), prose paragraphs streamed/segmented, choices block once
// streaming closes.
//
// Page-turn scroll: when a paragraph's bottom would sit past 68% of the
// panel height, smooth-scroll the panel so the previous paragraph sits
// at top with ~56px padding. Lifted from design/mockup-stage/app.jsx.

import { useEffect, useRef, useState } from "react";
import { postFreeText, postLeave, postPick } from "../api";
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
}: {
  scene: SceneSnapshot;
  chapterNo: number;
  isCurrent: boolean;
  fromLocationName: string | null;
  glossIndex: GlossIndex | null;
}) {
  const sectionRef = useRef<HTMLElement | null>(null);
  const fastForward = useStore((s) => s.fastForward);
  const attachPick = useStore((s) => s.attachPick);
  const phase = useStore((s) => s.phase);
  const freeTextRequest = useStore((s) => s.freeTextRequest);
  const [pickBusy, setPickBusy] = useState(false);
  const [freeTextBusy, setFreeTextBusy] = useState(false);

  const lastTurn = scene.turns[scene.turns.length - 1];
  const streamingThisChapter = isCurrent && lastTurn?.streaming;
  const allWords = scene.turns.flatMap((t) => t.words);

  usePageTurn(sectionRef, allWords.length, streamingThisChapter);

  const choicesReady =
    isCurrent &&
    lastTurn &&
    !lastTurn.streaming &&
    lastTurn.options.length > 0 &&
    !lastTurn.player_pick;

  const freeTextReady =
    isCurrent &&
    phase === "awaiting_free_text" &&
    freeTextRequest !== null &&
    lastTurn &&
    !lastTurn.player_pick;

  const onPick = async (idx: number) => {
    if (pickBusy || !lastTurn) return;
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

  const onLeave = async () => {
    if (pickBusy || freeTextBusy) return;
    await postLeave();
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
        {scene.turns.map((turn, i) => (
          <div key={i}>
            <StreamingProse
              words={turn.words}
              streaming={turn.streaming && isCurrent && i === scene.turns.length - 1}
              fastForward={fastForward && isCurrent}
              glossIndex={glossIndex}
            />
            {turn.player_pick && (
              <div className="player-pick">» {turn.player_pick.text}</div>
            )}
          </div>
        ))}
        {choicesReady && (
          <ChoicesBlock
            options={lastTurn.options}
            onPick={onPick}
            onLeave={onLeave}
            busy={pickBusy}
          />
        )}
        {freeTextReady && freeTextRequest && (
          <FreeTextInput
            prompt={freeTextRequest.prompt}
            onSubmit={onFreeTextSubmit}
            onLeave={onLeave}
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
) {
  useEffect(() => {
    if (!active) return;
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
      panel.scrollTo({ top: newTop, behavior: "smooth" });
    }
  }, [wordCount, active, sectionRef]);
}
