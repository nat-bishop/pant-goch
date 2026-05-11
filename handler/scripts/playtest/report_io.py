"""JSON dump + stdout summary for a playtest run.

Two artifacts per (persona, seed) case:
  - SessionRecord (engine trace: cycles, options, dispatches, final state)
  - list[EvalResult] (programmatic evaluator outputs)

Combined into a single JSON sidecar at `_analysis/playtest_<timestamp>.json`.
The schema is sectioned (`run` / `aggregate` / `cases`) so jq stays happy.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluators import EvalResult
from .session import SessionRecord


SCHEMA_VERSION = 2  # bumped from the chat-era schema_version=1


def build_payload(
    *,
    cases: list[tuple[SessionRecord, list[EvalResult]]],
    started_at: dt.datetime,
    ended_at: dt.datetime,
    cli_args: dict[str, Any],
) -> dict[str, Any]:
    aggregate = _aggregate(cases)
    summary = _summary(cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_s": (ended_at - started_at).total_seconds(),
            "cli_args": cli_args,
            "case_count": len(cases),
        },
        "aggregate": aggregate,
        "summary": summary,
        "cases": [
            {
                "session": rec.to_json(),
                "evaluators": [
                    {"name": e.name, "passed": e.passed, "rationale": e.rationale}
                    for e in evals
                ],
            }
            for rec, evals in cases
        ],
    }


def _aggregate(cases: list[tuple[SessionRecord, list[EvalResult]]]) -> dict[str, Any]:
    """Per-evaluator pass rate across all cases."""
    counts: dict[str, tuple[int, int]] = {}  # name -> (passed, total)
    for _rec, evals in cases:
        for e in evals:
            p, t = counts.get(e.name, (0, 0))
            counts[e.name] = (p + (1 if e.passed else 0), t + 1)
    return {
        name: {"passed": p, "total": t, "rate": (p / t) if t else 0.0}
        for name, (p, t) in counts.items()
    }


def _summary(cases: list[tuple[SessionRecord, list[EvalResult]]]) -> dict[str, Any]:
    """Per-persona rollup + global outcome distribution."""
    outcomes: Counter[str] = Counter()
    by_persona: dict[str, dict[str, Any]] = {}
    for rec, _evals in cases:
        outcomes[rec.outcome or "no_terminal"] += 1
        if rec.persona_name not in by_persona:
            by_persona[rec.persona_name] = {
                "archetype": rec.persona_archetype,
                "session_count": 0,
                "terminals": Counter(),
                "errors": 0,
            }
        p = by_persona[rec.persona_name]
        p["session_count"] += 1
        p["terminals"][rec.outcome or "no_terminal"] += 1
        if rec.error:
            p["errors"] += 1
    return {
        "outcomes": dict(outcomes),
        "by_persona": {
            name: {**v, "terminals": dict(v["terminals"])}
            for name, v in by_persona.items()
        },
    }


def write_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_stdout_summary(payload: dict[str, Any]) -> None:
    run = payload["run"]
    print(f"\n=== HANDLER playtest run · {run['started_at']} ===")
    print(f"Cases: {run['case_count']} · Duration: {run['duration_s']:.1f}s")
    print()
    print("Outcomes:")
    for outcome, n in sorted(payload["summary"]["outcomes"].items()):
        print(f"  {outcome}: {n}")
    print()
    print("Evaluator pass rates:")
    for name, agg in sorted(payload["aggregate"].items()):
        print(f"  {name:32s} {agg['passed']:3d}/{agg['total']:3d}  ({agg['rate']*100:5.1f}%)")
    print()
    print("Per-persona:")
    for name, p in sorted(payload["summary"]["by_persona"].items()):
        terminals = ", ".join(f"{k}={v}" for k, v in p["terminals"].items())
        print(f"  {name:24s} ({p['archetype']}) · {p['session_count']} sessions · {terminals}")
        if p["errors"]:
            print(f"    errors: {p['errors']}")
