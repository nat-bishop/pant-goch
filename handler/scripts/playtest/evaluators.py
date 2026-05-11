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
    """The session ran without errors and produced at least one cycle."""
    if rec.error:
        return EvalResult("session_completed", False, f"error: {rec.error}")
    if not rec.cycles:
        return EvalResult("session_completed", False, "no cycles recorded")
    return EvalResult("session_completed", True)


def terminal_reached(rec: SessionRecord) -> EvalResult:
    """A terminal beat fired (mission resolved with an outcome)."""
    if rec.outcome is None:
        return EvalResult(
            "terminal_reached", False, f"no terminal; cycles={rec.cycles_used()}"
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
    """The drafter never exhausted retries — every cycle's options came from
    the LLM, not the engine fallback. Surfaces prompt instability."""
    n = rec.engine_fallback_count()
    if n > 0:
        return EvalResult(
            "no_engine_fallback",
            False,
            f"engine fallback fired in {n} cycle(s) — drafter prompt instability",
        )
    return EvalResult("no_engine_fallback", True)


def moment_fired_when_eligible(rec: SessionRecord) -> EvalResult:
    """If a forced moment appeared in any cycle, exactly one option satisfied
    it. (The engine validates this server-side; the evaluator double-checks
    the recorded trace.)"""
    failures: list[int] = []
    for c in rec.cycles:
        if c.forced_moment_id is None:
            continue
        satisfiers = [
            o for o in c.options if o.get("satisfies_moment") == c.forced_moment_id
        ]
        if len(satisfiers) != 1:
            failures.append(c.cycle_index)
    if failures:
        return EvalResult(
            "moment_fired_when_eligible",
            False,
            f"cycles with bad forced-moment counts: {failures}",
        )
    return EvalResult("moment_fired_when_eligible", True)


def cycles_within_budget(rec: SessionRecord, max_cycles: int) -> EvalResult:
    """The session terminated within the authored cycle budget."""
    used = rec.cycles_used()
    if used > max_cycles:
        return EvalResult(
            "cycles_within_budget",
            False,
            f"used {used} cycles, budget was {max_cycles}",
        )
    return EvalResult("cycles_within_budget", True, f"{used}/{max_cycles}")


def options_tonal_spread(rec: SessionRecord) -> EvalResult:
    """Every cycle drafted at least 2 distinct archetypes — sanity check
    against the drafter collapsing to a single mode."""
    failures: list[int] = []
    for c in rec.cycles:
        archetypes = {o["archetype"] for o in c.options}
        if len(archetypes) < 2:
            failures.append(c.cycle_index)
    if failures:
        return EvalResult(
            "options_tonal_spread",
            False,
            f"cycles with <2 distinct archetypes: {failures}",
        )
    return EvalResult("options_tonal_spread", True)


# ───────────────────────── runner ─────────────────────────


def run_all(rec: SessionRecord, *, expected_outcome: str | None, max_cycles: int) -> list[EvalResult]:
    """Run the full evaluator suite against one record. Order matters for
    stdout summaries — keep the most informative results first."""
    return [
        session_completed(rec),
        terminal_reached(rec),
        expected_outcome_reached(rec, expected_outcome),
        cycles_within_budget(rec, max_cycles),
        moment_fired_when_eligible(rec),
        options_tonal_spread(rec),
        no_engine_fallback(rec),
    ]
