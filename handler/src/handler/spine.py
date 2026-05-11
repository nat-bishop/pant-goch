"""Engine: predicate evaluation + state mutation. Pure deterministic functions.

The LLM never writes to state — every state mutation flows through engine
functions defined here and called from `cycle.py`. Handler tick primitives
(valid_archetype_combinations, apply_archetype_effects, apply_moment_effects,
fire_due_beats, tick_cycle_close) are authored in Phase 1.
"""
from __future__ import annotations

from typing import assert_never

from .models import (
    Not_,
    Or_,
    Predicate,
    QualityEquals,
    QualityGte,
    QualityLt,
    State,
)


def evaluate_predicate(state: State, pred: Predicate) -> bool:
    """Evaluate `pred` against `state`. Pure — no side effects."""
    if isinstance(pred, QualityGte):
        v = state.get_quality(pred.quality)
        return v is not None and float(v) >= pred.value
    if isinstance(pred, QualityLt):
        v = state.get_quality(pred.quality)
        return v is not None and float(v) < pred.value
    if isinstance(pred, QualityEquals):
        v = state.get_quality(pred.quality)
        if v is None:
            return False
        return v == pred.value
    if isinstance(pred, Not_):
        return not evaluate_predicate(state, pred.of)
    if isinstance(pred, Or_):
        return any(evaluate_predicate(state, sub) for sub in pred.of)
    assert_never(pred)
