// Compact SVG sparkline — one row in the Pacing tab. Series y-values are
// normalized to [lo, hi]; flat series get a muted stroke so the user can
// tell at a glance which qualities have moved this session.

import clsx from "clsx";

interface SparklineProps {
  series: number[];     // chronological (oldest first)
  range: [number, number];
  flat?: boolean;
}

const W = 100;
const H = 24;
const PAD = 3;

export function Sparkline({ series, range, flat = false }: SparklineProps) {
  if (series.length === 0) {
    return <svg className="block h-6 w-full" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" />;
  }
  const [lo, hi] = range;
  const span = hi - lo || 1;
  // One-element series = flat dot at that value; two+ elements = polyline
  // across the viewport. Normalize y inside the [PAD, H-PAD] band so the
  // stroke doesn't clip on min/max boundary values.
  const points = series.map((v, i) => {
    const x = series.length === 1 ? W / 2 : (i / (series.length - 1)) * W;
    const y = H - PAD - ((v - lo) / span) * (H - 2 * PAD);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const areaPts = `0,${H} ${points} ${W},${H}`;

  return (
    <svg
      className={clsx("block h-6 w-full", flat && "opacity-60")}
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
    >
      <polygon points={areaPts} className={flat ? "fill-text-muted/10" : "fill-accent/15"} />
      <polyline
        points={points}
        fill="none"
        strokeWidth={1.5}
        className={flat ? "stroke-text-muted" : "stroke-accent"}
      />
    </svg>
  );
}
