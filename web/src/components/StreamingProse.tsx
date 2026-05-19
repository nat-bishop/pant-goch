// StreamingProse renders an array of paragraphs. Each paragraph's
// clauses fade in via CSS as they arrive (no JS pacing — backend already
// paces clauses by word count). When a paragraph is "closed" (the next
// paragraph has started OR `streaming=false`), it is post-processed by
// the gloss matcher so authored aliases become hover triggers.
//
// Per the design refinements: closed paragraphs get glosses retroactively;
// the underline fades in via the CSS transition on
// `.gloss-trigger { transition: text-decoration-color }`. The fade-in of
// the clauses themselves has already happened by then.

import { useMemo } from "react";
import { paragraphsFromChunks, segmentText, type GlossIndex } from "./glosses";
import { GlossTrigger } from "./GlossTrigger";

export function StreamingProse({
  chunks,
  streaming,
  fastForward,
  glossIndex,
}: {
  chunks: string[];
  streaming: boolean;
  fastForward: boolean;
  glossIndex: GlossIndex | null;
}) {
  const paragraphs = useMemo(() => paragraphsFromChunks(chunks), [chunks]);

  return (
    <div className={fastForward ? "fast-forward" : undefined}>
      {paragraphs.map((paraChunks, i) => {
        const isLast = i === paragraphs.length - 1;
        // A paragraph is "closed" once another paragraph starts after
        // it, OR the stream sentinel has arrived.
        const closed = !isLast || !streaming;
        return (
          <Paragraph
            key={i}
            chunks={paraChunks}
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
  chunks,
  closed,
  isLast,
  streaming,
  glossIndex,
}: {
  chunks: string[];
  closed: boolean;
  isLast: boolean;
  streaming: boolean;
  glossIndex: GlossIndex | null;
}) {
  const segments = useMemo(
    () =>
      closed && glossIndex ? segmentText(chunks.join(""), glossIndex) : null,
    [closed, chunks, glossIndex],
  );

  if (!closed) {
    return (
      <div className="para current">
        <p aria-live="polite">
          {chunks.map((c, i) => (
            <span key={i} className="chunk">
              {c}
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
        {(segments ?? [{ kind: "text" as const, text: chunks.join("") }]).map((seg, i) =>
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
