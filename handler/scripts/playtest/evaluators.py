"""Programmatic evaluators for HANDLER playtest sessions.

MVP keeps things programmatic — no LLM judges, no pydantic-evals dependency.
Each evaluator is a pure function `(SessionRecord) -> EvalResult` that returns
a pass/fail + rationale. Aggregation lives in `__main__.py`.

When prose-quality / voice-consistency judges become valuable (after
playtesting reveals which qualities need scoring), we'll re-introduce
pydantic-evals + LLM judges in a separate `judges.py` module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .session import SessionRecord


@dataclass
class EvalResult:
    name: str
    passed: bool
    rationale: str = ""


Evaluator = Callable[[SessionRecord], EvalResult]


def session_completed(rec: SessionRecord) -> EvalResult:
    """The session ran without errors and produced at least one day."""
    if rec.error:
        return EvalResult("session_completed", False, f"error: {rec.error}")
    if not rec.days:
        return EvalResult("session_completed", False, "no days recorded")
    return EvalResult("session_completed", True)


def terminal_reached(rec: SessionRecord) -> EvalResult:
    """A terminal event fired (mission resolved with an outcome)."""
    if rec.outcome is None:
        return EvalResult(
            "terminal_reached", False, f"no terminal; days={rec.days_used()}"
        )
    return EvalResult("terminal_reached", True, f"outcome={rec.outcome}")


def expected_outcome_reached(rec: SessionRecord, expected: str | None) -> EvalResult:
    """The persona reached its `expected_outcome` (if declared). Personas
    without an expected outcome always pass this evaluator."""
    if expected is None:
        return EvalResult("expected_outcome", True, "no expectation")
    if rec.outcome != expected:
        return EvalResult(
            "expected_outcome",
            False,
            f"expected {expected}, got {rec.outcome}",
        )
    return EvalResult("expected_outcome", True, f"matched {expected}")


def no_engine_fallback(rec: SessionRecord) -> EvalResult:
    """The drafter never exhausted retries — every day's options came from
    the LLM, not the engine fallback. Surfaces prompt instability."""
    n = rec.engine_fallback_count()
    if n > 0:
        return EvalResult(
            "no_engine_fallback",
            False,
            f"engine fallback fired in {n} day(s) — drafter prompt instability",
        )
    return EvalResult("no_engine_fallback", True)


def forced_event_satisfied_when_eligible(rec: SessionRecord) -> EvalResult:
    """If a forced event appeared in any day, exactly one option satisfied
    it. (The engine validates this server-side; the evaluator double-checks
    the recorded trace.)"""
    failures: list[int] = []
    for c in rec.days:
        if c.forced_event_id is None:
            continue
        satisfiers = [
            o for o in c.options if o.get("satisfies_event") == c.forced_event_id
        ]
        if len(satisfiers) != 1:
            failures.append(c.day_index)
    if failures:
        return EvalResult(
            "forced_event_satisfied_when_eligible",
            False,
            f"days with bad forced-event counts: {failures}",
        )
    return EvalResult("forced_event_satisfied_when_eligible", True)


def days_within_budget(rec: SessionRecord, max_days: int) -> EvalResult:
    """The session terminated within the authored day budget."""
    used = rec.days_used()
    if used > max_days:
        return EvalResult(
            "days_within_budget",
            False,
            f"used {used} days, budget was {max_days}",
        )
    return EvalResult("days_within_budget", True, f"{used}/{max_days}")


def options_tonal_spread(rec: SessionRecord) -> EvalResult:
    """Every day drafted at least 2 distinct archetypes — sanity check
    against the drafter collapsing to a single mode."""
    failures: list[int] = []
    for c in rec.days:
        archetypes = {o["archetype"] for o in c.options}
        if len(archetypes) < 2:
            failures.append(c.day_index)
    if failures:
        return EvalResult(
            "options_tonal_spread",
            False,
            f"days with <2 distinct archetypes: {failures}",
        )
    return EvalResult("options_tonal_spread", True)


# ───────────────────────── runner ─────────────────────────


def run_all(rec: SessionRecord, *, expected_outcome: str | None, max_days: int) -> list[EvalResult]:
    """Run the full evaluator suite against one record. Order matters for
    stdout summaries — keep the most informative results first."""
    return [
        session_completed(rec),
        terminal_reached(rec),
        expected_outcome_reached(rec, expected_outcome),
        days_within_budget(rec, max_days),
        forced_event_satisfied_when_eligible(rec),
        options_tonal_spread(rec),
        no_engine_fallback(rec),
    ]
