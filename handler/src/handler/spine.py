"""Engine: predicate evaluation + state mutation. Pure deterministic functions.

The LLM never writes to state — every state mutation flows through one of
the spine entry points (`apply_archetype_effects`, `apply_moment_effects`,
`fire_due_beats`, `tick_cycle_close`) and is called from `cycle.py`.

No I/O, no LLM calls, no async. Every function is pure (returns a new value)
or mutates the passed-in `State` in place (well-marked). Predicate evaluation
never mutates.
"""
from __future__ import annotations

import copy
from typing import Any, assert_never

from .models import (
    And_,
    Archetype,
    Beat,
    BeatFired,
    Effect,
    Mission,
    Moment,
    MomentFired,
    Not_,
    Option,
    Or_,
    Predicate,
    QualityEquals,
    QualityGt,
    QualityGte,
    QualityLt,
    QualityLte,
    State,
    substitute_effect,
    substitute_predicate,
)


# ───────────────────────── predicate evaluation ─────────────────────────


def evaluate_predicate(state: State, pred: Predicate) -> bool:
    """Evaluate `pred` against `state`. Pure — no side effects."""
    if isinstance(pred, QualityGte):
        v = state.get_quality(pred.quality)
        return v is not None and not isinstance(v, bool) and float(v) >= pred.value
    if isinstance(pred, QualityGt):
        v = state.get_quality(pred.quality)
        return v is not None and not isinstance(v, bool) and float(v) > pred.value
    if isinstance(pred, QualityLte):
        v = state.get_quality(pred.quality)
        return v is not None and not isinstance(v, bool) and float(v) <= pred.value
    if isinstance(pred, QualityLt):
        v = state.get_quality(pred.quality)
        return v is not None and not isinstance(v, bool) and float(v) < pred.value
    if isinstance(pred, QualityEquals):
        v = state.get_quality(pred.quality)
        return v is not None and v == pred.value
    if isinstance(pred, BeatFired):
        return pred.beat in state.fired_beats
    if isinstance(pred, MomentFired):
        return pred.moment in state.fired_moments
    if isinstance(pred, Not_):
        return not evaluate_predicate(state, pred.of)
    if isinstance(pred, And_):
        return all(evaluate_predicate(state, sub) for sub in pred.of)
    if isinstance(pred, Or_):
        return any(evaluate_predicate(state, sub) for sub in pred.of)
    assert_never(pred)


# ───────────────────────── effect application ─────────────────────────


def _apply_one(state: State, eff: Effect) -> tuple[str, Any, Any] | None:
    """Apply a single effect to state IN PLACE, return (quality, before, after)
    delta tuple for audit. Returns None if the `when` predicate guards it off.
    """
    if eff.when is not None and not evaluate_predicate(state, eff.when):
        return None
    before = state.get_quality(eff.quality)
    if eff.op == "set":
        after = eff.value
    elif eff.op == "add":
        cur = float(before) if before is not None else 0.0
        after = cur + float(eff.value)
    elif eff.op == "max":
        cur = float(before) if before is not None else float("-inf")
        after = max(cur, float(eff.value))
    elif eff.op == "min":
        cur = float(before) if before is not None else float("inf")
        after = min(cur, float(eff.value))
    else:
        assert_never(eff.op)
    # Preserve int-ness for numeric ops on integer-typed qualities (cover,
    # cycles, extraction_stage). Mixed float/int qualities don't exist in MVP.
    if (
        eff.op != "set"
        and isinstance(before, int)
        and isinstance(eff.value, (int, float))
        and float(eff.value).is_integer()
    ):
        after = int(after)
    # Clamp cover qualities at 0/10 — cover is bounded.
    if eff.quality.endswith(".cover") and isinstance(after, (int, float)):
        after = max(0, min(10, int(after)))
    state.set_quality(eff.quality, after)
    return (eff.quality, before, after)


def _apply_many(state: State, effects: list[Effect]) -> dict[str, tuple[Any, Any]]:
    """Apply a list of effects in order, collecting deltas. The dict keeps
    only the net change per quality (first `before`, last `after`)."""
    delta: dict[str, tuple[Any, Any]] = {}
    for eff in effects:
        result = _apply_one(state, eff)
        if result is None:
            continue
        quality, before, after = result
        existing = delta.get(quality)
        if existing is None:
            delta[quality] = (before, after)
        else:
            delta[quality] = (existing[0], after)
    return delta


# ───────────────────────── archetype enumeration ─────────────────────────


def valid_archetype_combinations(
    state: State, mission: Mission
) -> list[tuple[str, str | None, dict[str, str]]]:
    """Enumerate every (archetype_id, agent_id, params) tuple whose
    precondition `when` is satisfied. For archetypes that take `params: [agent]`,
    expands over agents whose cover > 0.

    Returns the catalog the drafter receives.
    """
    catalog: list[tuple[str, str | None, dict[str, str]]] = []
    for arch in mission.archetypes:
        if "agent" in arch.params:
            for agent in mission.agents:
                cover = state.get_quality(f"agent[{agent.id}].cover")
                if cover is None or float(cover) <= 0:
                    continue
                # Substitute $agent in the precondition before evaluating.
                pred = substitute_predicate(arch.when, agent.id)
                if evaluate_predicate(state, pred):
                    catalog.append((arch.id, agent.id, {"agent": agent.id}))
        else:
            if evaluate_predicate(state, arch.when):
                catalog.append((arch.id, None, {}))
    return catalog


# ───────────────────────── moment enumeration ─────────────────────────


def due_moments(state: State, mission: Mission) -> list[Moment]:
    """Moments whose `when` holds and haven't exhausted `fires_at_most`."""
    out: list[Moment] = []
    for m in mission.moments:
        if m.fires_at_most == "once_per_mission" and m.id in state.fired_moments:
            continue
        # `once_per_agent` on moments is not exercised in MVP; the only MVP
        # moment (kestrel_sacrifice) fires once per mission.
        if evaluate_predicate(state, m.when):
            out.append(m)
    return out


# ───────────────────────── archetype application ─────────────────────────


def apply_archetype_effects(
    state: State, option: Option, mission: Mission
) -> dict[str, tuple[Any, Any]]:
    """Apply the chosen option's archetype effects to state IN PLACE. Returns
    the net delta. If the option binds an agent (via `params`), `$agent` is
    substituted in every effect's quality/when.
    """
    arch = mission.archetype(option.archetype)
    effects = list(arch.effects)
    if option.agent is not None:
        effects = [substitute_effect(e, option.agent) for e in effects]
    return _apply_many(state, effects)


def option_progresses_mission(delta: dict[str, tuple[Any, Any]], mission: Mission) -> bool:
    """True iff the delta moved `mission.extraction_stage` (or any
    non-cover, non-protected mission-qualities) forward. Used to gate
    passive-pressure beats (§7 `only_when_no_progress_this_cycle`).

    "Progress" definition: any mission.* quality changed AND at least one
    such change was an increase. Cover changes alone don't count as progress.
    """
    for quality, (before, after) in delta.items():
        if not quality.startswith("mission."):
            continue
        if quality == "mission.cycles_remaining":
            continue  # cycle countdown isn't progress
        if before is None and after is not None:
            return True
        if before is not None and after is not None and float(after) > float(before):
            return True
    return False


# ───────────────────────── moment application ─────────────────────────


def apply_moment_effects(
    state: State, moment: Moment, option: Option
) -> dict[str, tuple[Any, Any]]:
    """Apply a fired moment's effects to state IN PLACE. Returns net delta.
    The option's agent (if any) is substituted into `$agent` references.
    Records the moment in `state.fired_moments`.
    """
    effects = list(moment.effects)
    if option.agent is not None:
        effects = [substitute_effect(e, option.agent) for e in effects]
    delta = _apply_many(state, effects)
    state.fired_moments.append(moment.id)
    return delta


# ───────────────────────── beat firing ─────────────────────────


def _beat_fired_already(state: State, beat: Beat, agent_id: str | None) -> bool:
    if beat.fires_at_most == "unlimited":
        return False
    if beat.fires_at_most == "once_per_mission":
        return beat.id in state.fired_beats
    # once_per_agent
    if agent_id is None:
        return beat.id in state.fired_beats
    return f"{beat.id}:{agent_id}" in state.fired_per_agent


def merge_delta_into(
    target: dict[str, tuple[Any, Any]], source: dict[str, tuple[Any, Any]]
) -> None:
    """Merge `source` delta into `target` in place — first-seen `before` wins,
    latest `after` wins. Used to fold per-fire deltas into a cycle aggregate
    and to layer a moment's effects over an archetype's."""
    for q, (b, a) in source.items():
        existing = target.get(q)
        target[q] = (existing[0] if existing else b, a)


def fire_due_beats(
    state: State, mission: Mission, made_progress: bool
) -> tuple[list[tuple[Beat, str | None, dict[str, tuple[Any, Any]]]], dict[str, tuple[Any, Any]]]:
    """Fire every eligible beat in (priority, authored) order. Mutates state
    in place. Returns:
        - list of (beat, agent_id_or_None, per-beat delta) for each fire
        - aggregate delta across all fires

    Iterates beats in stable order. For `for_each_agent` beats, fires once per
    matching agent. Stops the cycle's beat phase when a `terminal` beat fires
    (subsequent beats do not fire that cycle).
    """
    fires: list[tuple[Beat, str | None, dict[str, tuple[Any, Any]]]] = []
    aggregate: dict[str, tuple[Any, Any]] = {}

    ordered = sorted(enumerate(mission.beats), key=lambda x: (x[1].priority, x[0]))

    for _, beat in ordered:
        if state.outcome is not None:
            break  # terminal short-circuit
        if beat.only_when_no_progress_this_cycle and made_progress:
            continue

        if beat.for_each_agent:
            for agent in mission.agents:
                if _beat_fired_already(state, beat, agent.id):
                    continue
                pred = substitute_predicate(beat.when, agent.id)
                if not evaluate_predicate(state, pred):
                    continue
                effects = [substitute_effect(e, agent.id) for e in beat.effects]
                delta = _apply_many(state, effects)
                state.fired_per_agent.append(f"{beat.id}:{agent.id}")
                # Also record in fired_beats — predicates that just ask "did
                # this beat fire at all" don't care about the agent split.
                state.fired_beats.append(beat.id)
                if beat.terminal and beat.outcome_key is not None:
                    state.outcome = beat.outcome_key
                fires.append((beat, agent.id, delta))
                merge_delta_into(aggregate, delta)
                if beat.terminal:
                    break  # one terminal fire is enough
        else:
            if _beat_fired_already(state, beat, None):
                continue
            if not evaluate_predicate(state, beat.when):
                continue
            delta = _apply_many(state, beat.effects)
            state.fired_beats.append(beat.id)
            if beat.terminal and beat.outcome_key is not None:
                state.outcome = beat.outcome_key
            fires.append((beat, None, delta))
            merge_delta_into(aggregate, delta)

    return fires, aggregate


# ───────────────────────── cycle close ─────────────────────────


def tick_cycle_close(state: State) -> dict[str, tuple[Any, Any]]:
    """End-of-cycle housekeeping. Decrements `mission.cycles_remaining` and
    clears one-cycle flags. Returns the cycle-close delta. Pure-ish: mutates
    `state` in place, doesn't fire beats (those run separately).
    """
    delta: dict[str, tuple[Any, Any]] = {}
    cur = state.get_quality("mission.cycles_remaining")
    if cur is not None:
        before = cur
        after = int(cur) - 1
        state.set_quality("mission.cycles_remaining", after)
        delta["mission.cycles_remaining"] = (before, after)
    state.one_cycle_flags.clear()
    return delta


# ───────────────────────── helpers ─────────────────────────


def snapshot(state: State) -> State:
    """Deep copy a State — used for replay and the debug timeline."""
    return copy.deepcopy(state)


