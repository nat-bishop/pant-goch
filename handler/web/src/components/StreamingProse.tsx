// StreamingProse renders an array of paragraphs. Each paragraph's words
// fade in via CSS as they arrive (no JS pacing — backend already paces
// at ≥80ms). When a paragraph is "closed" (the next paragraph has
// started OR `streaming=false`), it is post-processed by the gloss
// matcher so authored aliases become hover triggers.
//
// Per the design refinements: closed paragraphs get glosses retroactively;
// the underline fades in via the CSS transition on
// `.gloss-trigger { transition: text-decoration-color }`. The fade-in of
// the words themselves has already happened by then.

import { useMemo } from "react";
import { paragraphsFromWords, segmentText, type GlossIndex } from "./glosses";
import { GlossTrigger } from "./GlossTrigger";

export function StreamingProse({
  words,
  streaming,
  fastForward,
  glossIndex,
}: {
  words: string[];
  streaming: boolean;
  fastForward: boolean;
  glossIndex: GlossIndex | null;
}) {
  const paragraphs = useMemo(() => paragraphsFromWords(words), [words]);

  return (
    <div className={fastForward ? "fast-forward" : undefined}>
      {paragraphs.map((paraWords, i) => {
        const isLast = i === paragraphs.length - 1;
        // A paragraph is "closed" once another paragraph starts after
        // it, OR the stream sentinel has arrived.
        const closed = !isLast || !streaming;
        return (
          <Paragraph
            key={i}
            words={paraWords}
            closed={closed}
            isLast={isLast}
            streaming={streaming}
            glossIndex={glossIndex}
          />
        );
      })}
    </div>
  );
}

function Paragraph({
  words,
  closed,
  isLast,
  streaming,
  glossIndex,
}: {
  words: string[];
  closed: boolean;
  isLast: boolean;
  streaming: boolean;
  glossIndex: GlossIndex | null;
}) {
  const text = words.join("");
  const segments = useMemo(
    () =>
      closed && glossIndex
        ? segmentText(text, glossIndex)
        : null,
    [closed, text, glossIndex],
  );

  if (!closed) {
    return (
      <div className="para current">
        <p aria-live="polite">
          {words.map((w, i) => (
            <span key={i} className="word">
              {w}
            </span>
          ))}
          {isLast && streaming && <span className="caret" />}
        </p>
      </div>
    );
  }

  return (
    <div className="para prior">
      <p>
        {(segments ?? [{ kind: "text" as const, text }]).map((seg, i) =>
          seg.kind === "gloss" ? (
            <GlossTrigger key={i} text={seg.text} glossId={seg.glossId} />
          ) : (
            <span key={i}>{seg.text}</span>
          ),
        )}
      </p>
    </div>
  );
}
