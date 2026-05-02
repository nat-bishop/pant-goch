import { useEffect, useState } from "react";
import { streamCoda } from "../api";

interface CodaPiece {
  kind: "fragment" | "framing";
  index: number;
  label?: string;
  text: string;
}

export function Coda() {
  const [pieces, setPieces] = useState<CodaPiece[]>([]);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stream = streamCoda();
        for await (const ev of stream) {
          if (cancelled) break;
          if (ev.type === "fragment") {
            setPieces((p) => [
              ...p,
              { kind: "fragment", index: ev.index, label: ev.label, text: ev.excerpt },
            ]);
          } else if (ev.type === "framing") {
            setPieces((p) => [...p, { kind: "framing", index: ev.index, text: ev.line }]);
          } else if (ev.type === "error") {
            setError(ev.message);
          } else if (ev.type === "done") {
            setDone(true);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Server emits all fragments first, then all framings. Display order is
  // framing[0], fragment[0], framing[1], fragment[1], …, with one trailing
  // closing-framing line (framings.length === fragments.length + 1).
  const fragments = pieces.filter((p) => p.kind === "fragment").sort((a, b) => a.index - b.index);
  const framings = pieces.filter((p) => p.kind === "framing").sort((a, b) => a.index - b.index);

  const interleaved: CodaPiece[] = [];
  const max = Math.max(fragments.length, framings.length);
  for (let i = 0; i < max; i++) {
    if (framings[i]) interleaved.push(framings[i]);
    if (fragments[i]) interleaved.push(fragments[i]);
  }

  return (
    <div className="h-full overflow-y-auto bg-paper">
      <div className="mx-auto max-w-prose px-8 py-16">
        {error && <div className="mb-6 text-sm text-warm">{error}</div>}
        {interleaved.length === 0 && !error && (
          <div className="text-sm text-muted">…</div>
        )}
        <ol className="space-y-10">
          {interleaved.map((p, idx) => (
            <li key={idx}>
              {p.kind === "framing" ? (
                <p className="font-serif text-base italic leading-relaxed text-muted">
                  {p.text}
                </p>
              ) : (
                <blockquote className="prose-email border-l-2 border-warm pl-5 text-ink">
                  {p.text}
                </blockquote>
              )}
            </li>
          ))}
        </ol>
        {done && (
          <div className="mt-16 text-center text-xs text-muted">— end —</div>
        )}
      </div>
    </div>
  );
}
