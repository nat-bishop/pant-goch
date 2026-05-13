import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Paces a static string out as word tokens to feed <StreamingText>. The
// backend's paced_word_emitter does this for LLM output (HANDLER_SPEC_V2
// §11 — server-side word-pacing is the single source of timing truth);
// this hook does the same for client-side authored prose (briefings,
// archive re-reads) so the reveal cadence is consistent across surfaces.
//
// Tokens keep their trailing whitespace so re-joining renders correctly.
// `flush` is idempotent — safe to wire directly as a click handler.
export function usePacedWords(
  text: string,
  paceMs = 80,
): { words: string[]; complete: boolean; flush: () => void } {
  const [words, setWords] = useState<string[]>([]);
  const [complete, setComplete] = useState(false);
  const tokens = useMemo(() => text.match(/\S+\s*/g) ?? [], [text]);
  const cancelRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    cancelRef.current?.();
    setWords([]);
    setComplete(false);
    if (tokens.length === 0) {
      setComplete(true);
      return;
    }
    let i = 0;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    function tick() {
      if (cancelled) return;
      setWords((w) => [...w, tokens[i]]);
      i += 1;
      if (i >= tokens.length) {
        setComplete(true);
        return;
      }
      timer = setTimeout(tick, paceMs);
    }
    timer = setTimeout(tick, paceMs);
    const cancel = () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    cancelRef.current = cancel;
    return cancel;
  }, [tokens, paceMs]);

  const flush = useCallback(() => {
    cancelRef.current?.();
    setWords(tokens);
    setComplete(true);
  }, [tokens]);

  return { words, complete, flush };
}
