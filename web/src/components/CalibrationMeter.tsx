import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

interface Props {
  value: number;
  ended: boolean;
}

const COOL = "var(--meter-cool)";
const WARN = "var(--meter-warn)";
const HOT = "var(--meter-hot)";

function colorFor(value: number): string {
  if (value < 40) return COOL;
  if (value < 75) return WARN;
  return HOT;
}

export function CalibrationMeter({ value, ended }: Props) {
  const clamped = Math.max(0, Math.min(100, Math.round(value)));
  const color = colorFor(clamped);
  const lastValue = useRef(clamped);
  const [glow, setGlow] = useState(false);

  useEffect(() => {
    if (clamped !== lastValue.current) {
      setGlow(true);
      const t = setTimeout(() => setGlow(false), 600);
      lastValue.current = clamped;
      return () => clearTimeout(t);
    }
  }, [clamped]);

  if (ended) {
    return (
      <div
        className="flex items-center gap-2 rounded-full border border-border px-3 py-1.5"
        style={{ background: `color-mix(in srgb, ${HOT} 12%, transparent)` }}
      >
        <span
          className="h-2 w-2 rounded-full animate-presence-pulse"
          style={{ background: HOT }}
        />
        <span
          className="text-[11px] font-semibold uppercase tracking-[0.14em]"
          style={{ color: HOT }}
        >
          Transition Complete
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-[12px] font-medium tabular-nums text-text">
        {clamped}%
      </span>
      <div
        className="relative h-1.5 w-44 overflow-hidden rounded-full bg-bg-bubble-them"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={clamped}
        aria-label="Counterpart fidelity"
      >
        <div
          className={clsx("h-full rounded-full transition-all duration-[600ms] ease-out", glow && "animate-meter-glow")}
          style={{
            width: `${clamped}%`,
            background: color,
            ["--meter-glow-color" as string]: color,
          }}
        />
      </div>
      <span className="hidden text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted lg:inline">
        Counterpart Fidelity
      </span>
    </div>
  );
}
