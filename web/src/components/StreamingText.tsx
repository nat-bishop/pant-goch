import { useEffect, useRef, useState } from "react";

interface Props {
  words: string[];
  className?: string;
  /** When true, append a blinking caret at the end while streaming. */
  showCaret?: boolean;
  /** When true, all words are present (no more arriving). Caret hides. */
  complete?: boolean;
}

// Shared style refs — hoisted so each word span doesn't allocate a fresh
// object on every render (the streaming path renders ~100s of spans).
const FLUSHED_STYLE = { opacity: 1 } as const;
const ANIMATING_STYLE = { animation: "word-in 80ms ease-out both" } as const;
const CARET_STYLE = {
  display: "inline-block",
  width: "0.5em",
  background: "currentColor",
  opacity: 0.5,
  animation: "word-in 80ms ease-out both",
  marginLeft: "2px",
} as const;

/** Renders `words` as a sequence of <span>s, each fading in over 80ms.
 * Backend paces word arrivals ≥80ms (HANDLER_SPEC_V2 §11). Clicking the
 * container flushes any in-flight animations to "all visible" instantly. */
export function StreamingText({ words, className, showCaret, complete }: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const [flushed, setFlushed] = useState(false);

  useEffect(() => {
    if (words.length === 0) setFlushed(false);
  }, [words.length]);

  function flush() {
    if (flushed || complete) return;
    setFlushed(true);
  }

  return (
    <span
      ref={ref}
      className={className}
      onClick={flush}
      data-flushed={flushed ? "true" : "false"}
    >
      {words.map((w, i) => (
        <span key={i} style={flushed ? FLUSHED_STYLE : ANIMATING_STYLE}>
          {w}
        </span>
      ))}
      {showCaret && !complete && (
        <span aria-hidden style={CARET_STYLE}>
          {" "}
        </span>
      )}
    </span>
  );
}
