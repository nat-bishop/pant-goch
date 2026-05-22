"""Isolation-eval lane — re-execute ONE engine agent against a frozen `State`.

The third eval category (see EVALS.md): a *deterministic assertion over a
stochastic subject*. The session harness (`evals.run`) buries a single agent's
behaviour in a 60-scene playthrough; the judge cases (`evals.judge_run`) freeze
*recorded output* and so cannot test a prompt change. This lane freezes the
**input** `State`, re-runs the real agent K times (Gemma is temp=1.0), and scores
the emitted output as a per-case **pass-rate** with a CI floor gate.

Members are `Suite`s (one per agent / property), each a pydantic-evals `Dataset`
over a task that builds the frozen state and calls the agent once. Deterministic
design rules are bool assertions (they gate); LLM judges reused from
`evals.judges` are Scored (they trend, never gate) — exactly the structural-vs-
behavioural split EVALS.md mandates.
"""
from __future__ import annotations

from .harness import (
    DEFAULT_FLOOR,
    IsoCase,
    Suite,
    case_pass_rates,
    floor_violations,
    run_and_report,
    run_suite,
    suite_metrics,
)

__all__ = [
    "DEFAULT_FLOOR",
    "IsoCase",
    "Suite",
    "case_pass_rates",
    "floor_violations",
    "run_and_report",
    "run_suite",
    "suite_metrics",
]
