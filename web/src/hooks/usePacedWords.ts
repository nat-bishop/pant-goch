import { useCallback, useEffect, useMemo, useState } from "react";

// Tokenizes a static authored prose string for staggered CSS-delay
// reveal. The companion render site lays out ALL tokens immediately at
// opacity 0 with per-word `animation-delay` so the container's size is
// stable from frame 1 — words fade in at the pace set here, but the box
// doesn't grow. Backend-streamed surfaces use <StreamingText> instead
// (words arrive incrementally and the live-write effect is intentional).
//
// `flush` jumps the whole list to fully visible for click-to-skip.
export function usePacedWords(text: string, paceMs = 80) {
  const tokens = useMemo(() => text.match(/\S+\s*/g) ?? [], [text]);
  const [flushed, setFlushed] = useState(false);
  const [complete, setComplete] = useState(false);

  useEffect(() => {
    setFlushed(false);
    setComplete(false);
    if (tokens.length === 0) {
      setComplete(true);
      return;
    }
    const timer = setTimeout(
      () => setComplete(true),
      tokens.length * paceMs,
    );
    return () => clearTimeout(timer);
  }, [tokens, paceMs]);

  const flush = useCallback(() => {
    setFlushed(true);
    setComplete(true);
  }, []);

  return { tokens, flushed, complete, flush };
}
