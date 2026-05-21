"""Run report + iteration loop.

`emit()` writes a `summary.json` into the run's artifact dir and renders a
markdown report whose top is the five north-star metrics, each with a delta vs.
the *previous* run's `summary.json`. That diff is the iteration loop: run
`make playtest`, read whether the numbers moved.

`summary.json` shape (stable, so two runs diff cleanly):
    {"generated_at": ..., "metrics": {<CorpusMetrics.to_dict()>},
     "invariants": {<gate name>: <pass-rate 0..1>}}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .metrics import CorpusMetrics

# The five metrics shown at the top of every report, each with a ↑/↓ direction
# (whether higher is healthier) used only to annotate the delta, never to gate.
NORTH_STAR_KEYS = [
    "ending_entropy_normalized",
    "beats_fired",
    "filler_turn_ratio",
    "mean_pairwise_beat_jaccard_distance",
    "persona_outcome_separation",
]
_HIGHER_IS_BETTER = {
    "ending_entropy_normalized": True,
    "beats_fired": True,
    "filler_turn_ratio": False,
    "mean_pairwise_beat_jaccard_distance": True,
    "persona_outcome_separation": True,
}


def invariant_pass_rates(invariant_report) -> dict[str, float]:
    """Per-gate pass-rate (0..1) from a pydantic-evals `EvaluationReport`:
    the mean of each named boolean assertion across the run's cases."""
    totals: dict[str, list[bool]] = {}
    for case in invariant_report.cases:
        for name, result in case.assertions.items():
            totals.setdefault(name, []).append(bool(result.value))
    return {
        name: round(sum(vals) / len(vals), 3)
        for name, vals in sorted(totals.items())
        if vals
    }


def write_summary(out_dir: Path, metrics: CorpusMetrics, invariant_report) -> Path:
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": metrics.to_dict(),
        "invariants": invariant_pass_rates(invariant_report),
    }
    path = Path(out_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def previous_summary(artifacts_root: Path, current: Path) -> dict | None:
    """The most recent run (by timestamped dir name) strictly before `current`
    that has a `summary.json`. None if this is the first."""
    root = Path(artifacts_root)
    cur = Path(current).name
    candidates = sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name < cur and (d / "summary.json").exists()
    )
    if not candidates:
        return None
    return json.loads((candidates[-1] / "summary.json").read_text(encoding="utf-8"))


def _fmt_delta(key: str, cur: float, prev: float | None) -> str:
    if prev is None:
        return ""
    d = cur - prev
    if abs(d) < 1e-9:
        return "  (=)"
    arrow = "▲" if d > 0 else "▼"
    good = _HIGHER_IS_BETTER.get(key, True)
    mark = "✅" if (d > 0) == good else "⚠️"
    return f"  ({arrow}{abs(d):.3g} {mark})"


def render_report(summary: dict, prev: dict | None) -> str:
    m = summary["metrics"]
    pm = (prev or {}).get("metrics", {})
    lines: list[str] = []
    lines.append(f"# Playtest report — {summary['generated_at']}  ({m['sessions']} sessions)")
    if prev is None:
        lines.append("_(no previous run to diff against)_")
    lines.append("")
    lines.append("## North star")
    for key in NORTH_STAR_KEYS:
        cur = m.get(key)
        delta = _fmt_delta(key, cur, pm.get(key))
        lines.append(f"- **{key}**: {cur}{delta}")
    lines.append("")
    lines.append("## Coverage & depth")
    lines.append(f"- endings reached: {m['endings_reached']}")
    lines.append(f"- endings never reached: {m['endings_never_reached']}")
    lines.append(f"- beats fired: {m['beats_fired']}/{m['beats_total']}")
    lines.append(f"- beats never fired: {m['beats_never_fired']}")
    lines.append(f"- location coverage: {m['location_coverage']}  "
                 f"never visited: {m['locations_never_visited']}")
    lines.append(f"- beats per location: {m['beats_per_location']}")
    lines.append(f"- median scenes/session: {m['median_scenes_per_session']}  "
                 f"median locations/session: {m['median_distinct_locations_per_session']}")
    lines.append(f"- longest common beat prefix (>50%): {m['longest_common_beat_prefix']}")
    lines.append(f"- ladder compression: {m['ladder_compression']}")
    lines.append("")
    lines.append("## Invariant gates (pass-rate)")
    for name, rate in summary["invariants"].items():
        flag = "" if rate >= 1.0 else "  ⚠️"
        lines.append(f"- {name}: {rate}{flag}")
    return "\n".join(lines)


def emit(out_dir: Path, metrics: CorpusMetrics, invariant_report, *, console: bool = True) -> Path:
    """Write summary.json and (optionally) print the diffed markdown report.
    Returns the summary.json path."""
    out_dir = Path(out_dir)
    path = write_summary(out_dir, metrics, invariant_report)
    summary = json.loads(path.read_text(encoding="utf-8"))
    prev = previous_summary(out_dir.parent, out_dir)
    if console:
        print("\n" + render_report(summary, prev))
    return path
