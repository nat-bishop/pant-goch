interface Props {
  tokens: string[];
  flushed: boolean;
  paceMs: number;
}

// Renders pre-tokenized authored prose with per-word `animation-delay`
// so the container is sized for the final text from frame 1. Paired
// with usePacedWords. For backend-streamed prose (LLM output), use
// <StreamingText> instead — that handles words arriving over time.
export function PacedReveal({ tokens, flushed, paceMs }: Props) {
  return (
    <>
      {tokens.map((t, i) =>
        flushed ? (
          <span key={i}>{t}</span>
        ) : (
          <span
            key={i}
            style={{
              opacity: 0,
              animation: `word-in ${paceMs}ms ease-out ${i * paceMs}ms both`,
            }}
          >
            {t}
          </span>
        ),
      )}
    </>
  );
}
