// Recursive renderer for Predicate trees, colored by per-node pass/fail.
// Tree shape is guaranteed to match `predicate_trace` — the backend
// builds them in lockstep via `evaluate_predicate_with_trace`, so this
// crashes loudly in DEV if they ever drift rather than degrading.

import clsx from "clsx";
import type { Predicate, PredicateTrace } from "../api";

interface PredicateTreeProps {
  predicate: Predicate;
  trace: PredicateTrace;
}

export function PredicateTree({ predicate, trace }: PredicateTreeProps) {
  if (predicate.kind === "not") {
    const [child] = trace.children;
    if (!child) return <Leaf passed={trace.passed} label={trace.label} />;
    return (
      <div className="space-y-0.5">
        <div className="flex items-center gap-1 text-text-muted">
          <Dot passed={trace.passed} />
          <span className="text-[9px] uppercase tracking-wider">not</span>
          <span className="text-text-muted/70">{trace.passed ? "(inner is false)" : "(inner is true)"}</span>
        </div>
        <div className="ml-3 border-l border-border/50 pl-2">
          <PredicateTree predicate={predicate.of} trace={child} />
        </div>
      </div>
    );
  }
  if (predicate.kind === "or") {
    return (
      <div className="space-y-0.5">
        <div className="flex items-center gap-1 text-text-muted">
          <Dot passed={trace.passed} />
          <span className="text-[9px] uppercase tracking-wider">or</span>
          <span className="text-text-muted/70">
            ({trace.children.filter((c) => c.passed).length}/{trace.children.length} passed)
          </span>
        </div>
        <div className="ml-3 space-y-0.5 border-l border-border/50 pl-2">
          {predicate.of.map((sub, i) => {
            const childTrace = trace.children[i];
            if (!childTrace) return null;
            return <PredicateTree key={i} predicate={sub} trace={childTrace} />;
          })}
        </div>
      </div>
    );
  }
  // Leaf — render the concrete-value label rendered server-side. The
  // label already carries names + numbers + verdict shape; no further
  // formatting needed here.
  return <Leaf passed={trace.passed} label={trace.label} />;
}

function Leaf({ passed, label }: { passed: boolean; label: string }) {
  return (
    <div className="flex items-baseline gap-1 leading-snug">
      <Dot passed={passed} />
      <span className={clsx(passed ? "text-text" : "text-text-muted")}>{label}</span>
    </div>
  );
}

function Dot({ passed }: { passed: boolean }) {
  return (
    <span
      aria-hidden
      className={clsx(
        "inline-block h-1.5 w-1.5 shrink-0 self-center rounded-full",
        passed ? "bg-presence-active" : "bg-meter-hot",
      )}
    />
  );
}
