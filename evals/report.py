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
    "post_exhaustion_turn_ratio",
    "mean_pairwise_beat_jaccard_distance",
    "persona_outcome_separation",
]
_HIGHER_IS_BETTER = {
    "ending_entropy_normalized": True,
    "beats_fired": True,
    "filler_turn_ratio": False,
    "post_exhaustion_turn_ratio": False,
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


def judge_results(judge_report) -> dict[str, dict]:
    """Per-judge stats from a pydantic-evals `EvaluationReport`. Each judge maps to
    a dict carrying the headline AND the floor — a mean alone hides a bimodal split
    (twelve 5s + a tail of {1,2,3} reads as a healthy 4.4 while a quarter of sessions
    are broken). Keyed by the judge's name (its `get_default_evaluation_name`).

      - PassFail: {"kind","value"=pass-rate,"n","n_fail"}
      - Scored:   {"kind","value"=mean,"n","min","n_below_3"}

    Errored cases (e.g. timeouts) land in `report.failures`, not `report.cases`, so
    they are excluded — matching the old scorer, which averaged non-error cases only."""
    bools: dict[str, list[bool]] = {}
    nums: dict[str, list[float]] = {}
    for case in judge_report.cases:
        for name, r in case.assertions.items():
            bools.setdefault(name, []).append(bool(r.value))
        for name, r in case.scores.items():
            nums.setdefault(name, []).append(float(r.value))
    out: dict[str, dict] = {}
    for name, v in bools.items():
        if v:
            out[name] = {"kind": "passfail", "value": round(sum(v) / len(v), 3),
                         "n": len(v), "n_fail": sum(1 for x in v if not x)}
    for name, v in nums.items():
        if v:
            out[name] = {"kind": "scored", "value": round(sum(v) / len(v), 3),
                         "n": len(v), "min": min(v),
                         "n_below_3": sum(1 for x in v if x <= 3)}
    return out


def _judge_value(entry) -> float | None:
    """The headline number from a judges-summary entry, tolerating the old flat
    `{name: float}` shape so this run can still diff against an older summary.json."""
    if isinstance(entry, dict):
        return entry.get("value")
    return entry  # legacy float


def _judge_flagged(st: dict) -> bool:
    """A floor breach worth a ⚠️: any PassFail failure, or any Scored case ≤3 —
    surfaced even when the mean looks healthy."""
    if st["kind"] == "passfail":
        return st["value"] < 1.0
    return st["n_below_3"] > 0


def write_summary_block(out_dir: Path, key: str, data: dict) -> Path:
    """Merge one named block into the run's existing summary.json under `key`,
    preserving the other top-level blocks (`metrics`, `invariants`, `judges`,
    isolation suites). A later pass over a finished run (judges, isolation evals)
    merges rather than clobbers what `make playtest` already wrote. Tolerates a
    missing summary.json (writes a block-only file)."""
    path = Path(out_dir) / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary[key] = data
    summary.setdefault("generated_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def write_judges(out_dir: Path, judges: dict[str, float]) -> Path:
    """Merge the per-judge numbers into summary.json under `judges`. judge_run is a
    separate pass over a finished run, so merge rather than clobber."""
    return write_summary_block(out_dir, "judges", judges)


def render_summary_block(title: str, key: str, data: dict[str, float], prev: dict | None) -> str:
    """Render a flat {name: float} block with a per-key delta vs. the previous
    run's same-keyed block, reusing `_fmt_delta` (higher-is-better for any key not
    in `_HIGHER_IS_BETTER` — true for both pass-rates and judge means). Used by the
    isolation-eval lane to land its numbers in the same diffed report surface."""
    pb = (prev or {}).get(key, {})
    lines = [f"## {title}"]
    for name, val in sorted(data.items()):
        lines.append(f"- {name}: {val}{_fmt_delta(name, val, pb.get(name))}")
    return "\n".join(lines)


def render_judges(judges: dict[str, dict], prev: dict | None) -> str:
    """Per-judge headline + floor, with a delta vs. the previous run (higher is
    better for every judge — pass-rate and mean score alike), reusing `_fmt_delta`.
    A ⚠️ marks a floor breach (a failing PassFail case or a Scored case ≤3) even
    when the mean reads healthy."""
    pj = prev or {}
    lines = ["## Judges (pass-rate / mean score)"]
    for name, st in sorted(judges.items()):
        val = st["value"]
        delta = _fmt_delta(name, val, _judge_value(pj.get(name)))
        if st["kind"] == "scored":
            floor = f"  [min={st['min']}, ≤3×{st['n_below_3']}/{st['n']}]"
        else:
            floor = f"  [fails={st['n_fail']}/{st['n']}]"
        flag = "  ⚠️" if _judge_flagged(st) else ""
        lines.append(f"- {name}: {val}{delta}{floor}{flag}")
    return "\n".join(lines)


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
