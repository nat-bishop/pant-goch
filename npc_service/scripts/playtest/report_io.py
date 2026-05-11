"""Persist a playtest run to JSON + print a structured stdout summary.

A playtest run produces two parallel artifacts:
  - one `SessionRecord` per (persona, seed) — engine outputs (transcript,
    tick_decisions, final_state)
  - one pydantic-evals `EvaluationReport` — per-case evaluator results +
    aggregates

Both are serialized into a single JSON dump alongside run metadata. The
schema is sectioned (`run` / `aggregate` / `summary` / `cases`) so it stays
queryable from jq without flattening, and `schema_version` lets readers
detect drift if the shape evolves.

Kept in a separate module from `__main__.py` because the serialization logic
is non-trivial (EvaluationResult / ReportCase / EvaluatorFailure → dict)
and benefits from being independently testable. `__main__.py` stays
orchestration-only.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .session import SessionRecord


SCHEMA_VERSION = 1


# ───────────────────────── pydantic-evals → dict ─────────────────────────


def _eval_result_to_dict(result: Any) -> dict[str, Any]:
    """`EvaluationResult` → JSON-safe dict. The `source` field is an
    `EvaluatorSpec` whose `name` already lives at the parent dict key, so
    we drop it."""
    return {"value": result.value, "reason": result.reason}


def _case_to_dict(case: Any, session: SessionRecord | None) -> dict[str, Any]:
    """`ReportCase` → JSON-safe dict, embedding the matching SessionRecord
    so a case is self-contained (no cross-section joins needed to replay)."""
    metadata = case.metadata or {}
    failures = list(case.evaluator_failures or [])
    return {
        "case_name": case.name,
        "persona_archetype": metadata.get("archetype"),
        "seed": metadata.get("seed"),
        "task_duration_s": round(case.task_duration, 3),
        "total_duration_s": round(case.total_duration, 3),
        "scores": {k: _eval_result_to_dict(v) for k, v in case.scores.items()},
        "labels": {k: _eval_result_to_dict(v) for k, v in case.labels.items()},
        "assertions": {k: _eval_result_to_dict(v) for k, v in case.assertions.items()},
        "metrics": dict(case.metrics),
        "evaluator_failures": [
            {"name": f.name, "error_message": f.error_message}
            for f in failures
        ],
        "session": session.to_json() if session is not None else None,
    }


def _aggregate_to_dict(aggregate: Any) -> dict[str, Any]:
    """Pydantic-evals `ReportCaseAggregate` (returned by `report.averages()`)
    → JSON-safe dict. Drops `name` (always the literal "Averages")."""
    return {
        "scores": dict(aggregate.scores),
        "labels": dict(aggregate.labels),
        "assertions": aggregate.assertions,
        "metrics": dict(aggregate.metrics),
        "task_duration_s": round(aggregate.task_duration, 3),
        "total_duration_s": round(aggregate.total_duration, 3),
    }


# ───────────────────────── aggregations from SessionRecords ─────────────────────────


def _error_breakdown(records: list[SessionRecord]) -> dict[str, int]:
    """Group errored sessions by exception class. The error string is
    `"<phase>: <ExcClass>: <message>"` from `run_session`; we key on the
    first two segments so e.g. `ChatAgentFailure: 503...` and
    `ChatAgentFailure: 400...` aggregate together."""
    classes: Counter[str] = Counter()
    for r in records:
        if not r.error:
            continue
        parts = r.error.split(":")
        key = ":".join(p.strip() for p in parts[:2]) if len(parts) >= 2 else r.error
        classes[key] += 1
    return dict(classes)


def _terminal_distribution(records: list[SessionRecord]) -> dict[str, int]:
    """Count how often each terminal beat closed a session. Sessions that
    never ended bucket as `"<unterminated>"`."""
    counts: Counter[str] = Counter()
    for r in records:
        counts[r.reached_terminal() or "<unterminated>"] += 1
    return dict(counts)


def _beat_fire_frequency(records: list[SessionRecord]) -> dict[str, int]:
    """Across-run histogram of fired beats. Useful for cross-checking
    against `analyze_scenario.py` reachability (beats reachable but never
    fired = candidates for Director-bias investigation)."""
    counts: Counter[str] = Counter()
    for r in records:
        for beat_id in r.final_state.fired_beat_ids:
            counts[beat_id] += 1
    return dict(counts)


def _persona_summary(records: list[SessionRecord]) -> dict[str, dict[str, Any]]:
    """Per-persona rollup. Reach-rate is the headline metric — it answers
    "does my game finish for this kind of player?" — but we also surface
    error count and the set of fired beats so a struggling persona is
    visible at a glance."""
    by_persona: dict[str, list[SessionRecord]] = {}
    for r in records:
        by_persona.setdefault(r.persona_name, []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for name, group in by_persona.items():
        errored = sum(1 for r in group if r.error)
        ended = sum(1 for r in group if r.final_state.ended)
        terminals = Counter(
            r.reached_terminal() for r in group if r.reached_terminal()
        )
        all_beats: set[str] = set()
        for r in group:
            all_beats.update(r.final_state.fired_beat_ids)
        summary[name] = {
            "count": len(group),
            "errored": errored,
            "ended": ended,
            "terminal_reach_rate": round(ended / len(group), 3) if group else 0.0,
            "terminals": dict(terminals),
            "unique_beats_fired": sorted(all_beats),
        }
    return summary


# ───────────────────────── public API ─────────────────────────


def build_payload(
    *,
    records: list[SessionRecord],
    report: Any | None,
    run_args: dict[str, Any],
    engine_model: str,
    eval_model: str,
    scenario_path: str,
    started_at: dt.datetime,
    finished_at: dt.datetime,
) -> dict[str, Any]:
    """Assemble the full JSON payload. Pure function — no I/O — so tests
    can inspect the shape without writing files."""
    cases: list[dict[str, Any]] = []
    aggregate: dict[str, Any] | None = None

    if report is not None:
        # `records` is built as `[c.output for c in report.cases]` in
        # `__main__.py`, so zipping is exact lock-step alignment. If that
        # invariant ever changes, swap to an `id()`-keyed map.
        if len(report.cases) != len(records):
            raise RuntimeError(
                f"report has {len(report.cases)} cases but {len(records)} "
                "records were passed — caller broke the 1:1 invariant."
            )
        for case, session in zip(report.cases, records):
            cases.append(_case_to_dict(case, session))
        aggregate = _aggregate_to_dict(report.averages())
    else:
        # --no-evals path: still produce a `cases` section so the schema
        # is stable; just no scores/labels/assertions.
        for r in records:
            cases.append({
                "case_name": r.persona_name,
                "persona_archetype": None,
                "seed": None,
                "task_duration_s": None,
                "total_duration_s": None,
                "scores": {},
                "labels": {},
                "assertions": {},
                "metrics": {},
                "evaluator_failures": [],
                "session": r.to_json(),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "generated_at": finished_at.strftime("%Y%m%dT%H%M%S"),
            "started_at_iso": started_at.isoformat(timespec="seconds"),
            "finished_at_iso": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 1),
            "scenario_path": scenario_path,
            "engine_model": engine_model,
            "eval_model": eval_model,
            "args": run_args,
            "session_count": len(records),
            "errored_count": sum(1 for r in records if r.error),
        },
        "aggregate": aggregate,
        "summary": {
            "personas": _persona_summary(records),
            "terminal_distribution": _terminal_distribution(records),
            "beat_fire_frequency": _beat_fire_frequency(records),
            "error_breakdown": _error_breakdown(records),
        },
        "cases": cases,
    }


def write_dump(payload: dict[str, Any], out_dir: Path) -> Path:
    """Write `payload` to `_analysis/playtest_<ts>.json`. The timestamp
    comes from `payload["run"]["generated_at"]` so the filename and the
    embedded metadata match."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = payload["run"]["generated_at"]
    path = out_dir / f"playtest_{ts}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ───────────────────────── stdout summary ─────────────────────────


def _fmt_score(v: float | int | None) -> str:
    if v is None:
        return "  n/a"
    if isinstance(v, bool):
        return " PASS" if v else " FAIL"
    return f"{v:6.3f}"


def print_stdout_summary(payload: dict[str, Any]) -> None:
    """Print a compact, scriptable summary to stdout. Complements
    `report.print()` (pydantic-evals' rich-rendered per-case table) by
    surfacing run-level signal: error breakdown, terminal coverage,
    evaluator aggregates."""
    run = payload["run"]
    summary = payload["summary"]
    aggregate = payload["aggregate"]

    print()
    print("=" * 70)
    print(f"PLAYTEST RUN SUMMARY  ({run['generated_at']})")
    print("=" * 70)
    print(f"  duration:      {run['duration_seconds']:.1f}s")
    print(f"  engine model:  {run['engine_model']}")
    print(f"  eval model:    {run['eval_model']}")
    print(f"  sessions:      {run['session_count']} "
          f"({run['errored_count']} errored)")
    print()

    print("ERROR BREAKDOWN")
    if summary["error_breakdown"]:
        for key, n in sorted(summary["error_breakdown"].items(),
                             key=lambda kv: -kv[1]):
            print(f"  {n:>3}x  {key}")
    else:
        print("  (no errors)")
    print()

    print("TERMINAL DISTRIBUTION")
    for term, n in sorted(summary["terminal_distribution"].items(),
                          key=lambda kv: -kv[1]):
        print(f"  {n:>3}x  {term}")
    print()

    print("BEAT FIRE FREQUENCY")
    for beat, n in sorted(summary["beat_fire_frequency"].items(),
                          key=lambda kv: -kv[1]):
        print(f"  {n:>3}x  {beat}")
    print()

    print("PER-PERSONA")
    for name, p in sorted(summary["personas"].items()):
        reach = f"{p['ended']}/{p['count']} ended"
        err = f"({p['errored']} err)" if p["errored"] else ""
        print(f"  {name:<28} {reach:<14} {err}")
    print()

    if aggregate is not None:
        print("EVALUATOR AGGREGATES")
        # Assertions (bool) first, then scores (numeric). Reasons aren't
        # in the aggregate — they live per-case.
        if aggregate["assertions"] is not None:
            print(f"  assertion pass-rate: {aggregate['assertions']:.3f}")
        for name, val in sorted(aggregate["scores"].items()):
            print(f"  {name:<28} {_fmt_score(val)}")
        print()
