interface Props {
  count: number;
  cap: number;
}

/** N filled rectangles + (cap-N) hollow ones — the cover meter style from the
 * mockup. Pure rendering; no state. */
export function Pips({ count, cap }: Props) {
  const safe = Math.max(0, Math.min(cap, count));
  return (
    <span className="inline-flex gap-[3px] items-center">
      {Array.from({ length: safe }, (_, i) => (
        <i
          key={`f-${i}`}
          className="inline-block bg-chrome-bright"
          style={{ width: 7, height: 16 }}
        />
      ))}
      {Array.from({ length: cap - safe }, (_, i) => (
        <i
          key={`e-${i}`}
          className="inline-block border border-chrome-faint"
          style={{ width: 5, height: 14, margin: 1 }}
        />
      ))}
    </span>
  );
}
