"""Engine: predicate evaluation + state mutation. Pure deterministic functions.

The LLM never writes to state — every state mutation flows through one of
the spine entry points (`apply_archetype_effects`, `apply_forced_event_effects`,
`fire_due_passive_events`, `tick_cycle_close`) and is called from `cycle.py`.

No I/O, no LLM calls, no async. Every function is pure (returns a new value)
or mutates the passed-in `State` in place (well-marked). Predicate evaluation
never mutates.
"""
from __future__ import annotations

import copy
from typing import Any, Literal, assert_never

from .models import (
    And_,
    Archetype,
    DramaticQuestion,
    Effect,
    Event,
    EventFired,
    Mission,
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
    iter_predicate_leaves,
    substitute_effect,
    substitute_predicate,
)


ActPhase = Literal["setup", "complication", "climax", "crisis"]
TensionBand = Literal["steady", "escalating", "crisis"]


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
    if isinstance(pred, EventFired):
        return pred.event in state.fired
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
                pred = substitute_predicate(arch.when, {"agent": agent.id})
                if evaluate_predicate(state, pred):
                    catalog.append((arch.id, agent.id, {"agent": agent.id}))
        else:
            if evaluate_predicate(state, arch.when):
                catalog.append((arch.id, None, {}))
    return catalog


# ───────────────────────── forced-event enumeration ─────────────────────────


def due_forced_events(state: State, mission: Mission) -> list[Event]:
    """Forced events whose `when` holds and haven't exhausted `fires_at_most`.
    Filters `mission.events` to those with `force_into_drafter=True` — these
    are surfaced to the drafter so one option this cycle must satisfy them.
    """
    out: list[Event] = []
    for e in mission.events:
        if not e.force_into_drafter:
            continue
        if e.fires_at_most == "once_per_mission" and e.id in state.fired:
            continue
        # `once_per_agent` on forced events is not exercised in MVP; the only
        # MVP forced event (kestrel_sacrifice) fires once per mission.
        if evaluate_predicate(state, e.when):
            out.append(e)
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
        effects = [substitute_effect(e, {"agent": option.agent}) for e in effects]
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


# ───────────────────────── forced-event application ─────────────────────────


def apply_forced_event_effects(
    state: State, event: Event, option: Option
) -> dict[str, tuple[Any, Any]]:
    """Apply a fired forced event's effects to state IN PLACE. Returns net
    delta. The option's agent (if any) is substituted into `$agent` references.
    Records the event in `state.fired`.
    """
    effects = list(event.effects)
    if option.agent is not None:
        effects = [substitute_effect(e, {"agent": option.agent}) for e in effects]
    delta = _apply_many(state, effects)
    state.fired.append(event.id)
    return delta


# ───────────────────────── passive-event firing ─────────────────────────


def _event_fired_already(state: State, event: Event, agent_id: str | None) -> bool:
    if event.fires_at_most == "unlimited":
        return False
    if event.fires_at_most == "once_per_mission":
        return event.id in state.fired
    # once_per_agent
    if agent_id is None:
        return event.id in state.fired
    return f"{event.id}:{agent_id}" in state.fired_per_agent


def merge_delta_into(
    target: dict[str, tuple[Any, Any]], source: dict[str, tuple[Any, Any]]
) -> None:
    """Merge `source` delta into `target` in place — first-seen `before` wins,
    latest `after` wins. Used to fold per-fire deltas into a cycle aggregate
    and to layer a forced event's effects over an archetype's."""
    for q, (b, a) in source.items():
        existing = target.get(q)
        target[q] = (existing[0] if existing else b, a)


def fire_due_passive_events(
    state: State, mission: Mission, made_progress: bool
) -> tuple[list[tuple[Event, str | None, dict[str, tuple[Any, Any]]]], dict[str, tuple[Any, Any]]]:
    """Fire every eligible passive event (`force_into_drafter=False`) in
    (priority, authored) order. Mutates state in place. Returns:
        - list of (event, agent_id_or_None, per-event delta) for each fire
        - aggregate delta across all fires

    For `for_each_agent` events, fires once per matching agent. Stops the
    cycle's passive-event phase when a `terminal` event fires (subsequent
    events do not fire that cycle).
    """
    fires: list[tuple[Event, str | None, dict[str, tuple[Any, Any]]]] = []
    aggregate: dict[str, tuple[Any, Any]] = {}

    passive = [(i, e) for i, e in enumerate(mission.events) if not e.force_into_drafter]
    ordered = sorted(passive, key=lambda x: (x[1].priority, x[0]))

    for _, event in ordered:
        if state.outcome is not None:
            break  # terminal short-circuit
        if event.only_when_no_progress_this_cycle and made_progress:
            continue

        if event.for_each_agent:
            for agent in mission.agents:
                if _event_fired_already(state, event, agent.id):
                    continue
                pred = substitute_predicate(event.when, {"agent": agent.id})
                if not evaluate_predicate(state, pred):
                    continue
                effects = [substitute_effect(e, {"agent": agent.id}) for e in event.effects]
                delta = _apply_many(state, effects)
                state.fired_per_agent.append(f"{event.id}:{agent.id}")
                # Also record in `state.fired` — predicates that just ask
                # "did this event fire at all" don't care about the agent split.
                state.fired.append(event.id)
                if event.terminal and event.outcome_key is not None:
                    state.outcome = event.outcome_key
                fires.append((event, agent.id, delta))
                merge_delta_into(aggregate, delta)
                if event.terminal:
                    break  # one terminal fire is enough
        else:
            if _event_fired_already(state, event, None):
                continue
            if not evaluate_predicate(state, event.when):
                continue
            delta = _apply_many(state, event.effects)
            state.fired.append(event.id)
            if event.terminal and event.outcome_key is not None:
                state.outcome = event.outcome_key
            fires.append((event, None, delta))
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


# ───────────────────────── narrative-shape derivations ─────────────────────────


# Cover at or below this value puts the mission in crisis act phase.
_COVER_BRINK = 2
# When `cycles_remaining / max_cycles` drops to / below this, climax begins.
_CLIMAX_THRESHOLD = 0.34
# When the ratio is at / above this, the setup phase still holds (provided
# no cover loss has happened yet).
_SETUP_THRESHOLD = 0.66
# Cover swing thresholds for derive_tension_band — values are absolute
# magnitudes summed across all .cover qualities this cycle.
_CRISIS_SWING = 4
_ESCALATING_SWING = 2


def _is_numeric(v: Any) -> bool:
    """True for int/float values that should be treated as numbers — excludes
    bool (which `isinstance(True, int)` returns True for) and None."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def derive_act_phase(state: State, mission: Mission) -> ActPhase:
    """Classify the current cycle into a four-act register so the renderer
    and SoP can adopt an appropriate dramatic tone. Mission-agnostic — uses
    universal qualities only (cycles_remaining + per-agent cover).

    Precedence: crisis > climax > setup > complication. The order matters
    because crisis (an agent on the brink) trumps "we still have time."
    """
    cycles_raw = state.get_quality("mission.cycles_remaining")
    cycles = int(cycles_raw) if _is_numeric(cycles_raw) else 0
    max_cycles = mission.max_cycles
    # crisis: any agent's cover is at the brink, OR one cycle left.
    for agent in mission.agents:
        cover = state.get_quality(f"agent[{agent.id}].cover")
        if _is_numeric(cover) and int(cover) <= _COVER_BRINK:
            return "crisis"
    if cycles <= 1:
        return "crisis"
    if max_cycles > 0 and cycles / max_cycles <= _CLIMAX_THRESHOLD:
        return "climax"
    # setup holds only while no agent has taken any cover loss yet.
    if max_cycles > 0 and cycles / max_cycles >= _SETUP_THRESHOLD:
        if all(
            not _is_numeric(state.get_quality(f"agent[{a.id}].cover"))
            or int(state.get_quality(f"agent[{a.id}].cover")) >= a.starting_cover
            for a in mission.agents
        ):
            return "setup"
    return "complication"


def derive_tension_band(prior_state: State | None, state: State) -> TensionBand:
    """Classify the magnitude of cover swing this cycle — a renderer-side
    signal so a quiet cycle reads measured and a sharp drop reads kinetic.
    `prior_state` is the state at the start of the cycle (before
    archetype/moment/beat application); `state` is post-cycle.
    """
    if prior_state is None:
        return "steady"
    total_swing = 0
    for q, after in state.qualities.items():
        if not q.endswith(".cover"):
            continue
        before = prior_state.qualities.get(q, after)
        if _is_numeric(after) and _is_numeric(before):
            total_swing += abs(int(after) - int(before))
    if total_swing >= _CRISIS_SWING:
        return "crisis"
    if total_swing >= _ESCALATING_SWING:
        return "escalating"
    return "steady"


def select_dramatic_question(state: State, mission: Mission) -> str | None:
    """Return the first dramatic question whose `when` matches current state,
    or None if no question fires. Author orders most-specific first."""
    for dq in mission.dramatic_questions:
        if evaluate_predicate(state, dq.when):
            return dq.question
    return None


def invalid_archetype_combinations(
    state: State, mission: Mission
) -> list[tuple[str, str | None, str]]:
    """For each (archetype, agent) combination whose precondition is NOT
    satisfied, return (archetype_id, agent_id_or_None, reason). Reason is a
    short human-readable description of which precondition leaf failed —
    surfaced to the drafter so it can reason about *why* an option is
    unavailable instead of only seeing the trimmed catalog.

    Sibling of `valid_archetype_combinations`; together they cover the full
    archetype × agent space without overlap.
    """
    invalid: list[tuple[str, str | None, str]] = []
    for arch in mission.archetypes:
        if "agent" in arch.params:
            for agent in mission.agents:
                cover = state.get_quality(f"agent[{agent.id}].cover")
                if not _is_numeric(cover) or float(cover) <= 0:
                    invalid.append((arch.id, agent.id, f"agent[{agent.id}] is lost"))
                    continue
                pred = substitute_predicate(arch.when, {"agent": agent.id})
                if evaluate_predicate(state, pred):
                    continue
                invalid.append((arch.id, agent.id, _describe_failing_predicate(state, pred)))
        else:
            if evaluate_predicate(state, arch.when):
                continue
            invalid.append((arch.id, None, _describe_failing_predicate(state, arch.when)))
    return invalid


def _describe_failing_predicate(state: State, pred: Predicate) -> str:
    """Walk the predicate tree; return a short description of the first leaf
    that evaluates False. If every leaf passes (which can happen for composite
    failures driven by Or_/Not_ logic), return a generic message."""
    for leaf in iter_predicate_leaves(pred):
        if not evaluate_predicate(state, leaf):
            return _format_leaf(state, leaf)
    return "preconditions not met"


# Comparison-leaf templates. Keyed by predicate class; the format string uses
# `q` (quality ref), `v` (actual value from state), `t` (required threshold).
_COMPARISON_LEAF_TEMPLATES: dict[type, str] = {
    QualityGte:    "{q} is {v} (requires >= {t})",
    QualityGt:     "{q} is {v} (requires > {t})",
    QualityLte:    "{q} is {v} (requires <= {t})",
    QualityLt:     "{q} is {v} (requires < {t})",
    QualityEquals: "{q} is {v} (requires == {t!r})",
}


def _format_leaf(state: State, leaf: Predicate) -> str:
    """Render one failing leaf as a short prose hint (e.g.
    'mission.extraction_stage is 0 (requires >= 2)')."""
    template = _COMPARISON_LEAF_TEMPLATES.get(type(leaf))
    if template is not None:
        return template.format(
            q=leaf.quality, v=state.get_quality(leaf.quality), t=leaf.value
        )
    if isinstance(leaf, EventFired):
        return f"event {leaf.event!r} has not fired"
    return "precondition not met"


