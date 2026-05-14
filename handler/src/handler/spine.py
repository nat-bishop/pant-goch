"""Engine: predicate evaluation + state mutation. Pure deterministic functions.

The LLM never writes to state — every state mutation flows through one of
the spine entry points (`apply_action_effects`, `fire_due_reactions`,
`tick_day_close`) and is called from `day.py`.

No I/O, no LLM calls, no async. Every function is pure (returns a new value)
or mutates the passed-in `State` in place (well-marked). Predicate evaluation
never mutates.
"""
from __future__ import annotations

import copy
from typing import Any, Literal, assert_never

from .models import (
    Action,
    And_,
    Beat,
    CastMember,
    Effect,
    EventFired,
    Mission,
    Not_,
    Option,
    Or_,
    Outcome,
    Predicate,
    QualityEquals,
    QualityGt,
    QualityGte,
    QualityLt,
    QualityLte,
    Reaction,
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
    for audit. Returns None if the `when` predicate guards it off."""
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
    # days, extraction_stage). Mixed float/int qualities don't exist in MVP.
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


def _apply_many(
    state: State, effects: list[Effect]
) -> dict[str, tuple[Any, Any]]:
    """Apply effects in order, returning a net delta map. The delta keeps the
    first `before` and the latest `after` per quality across the run."""
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


# ───────────────────────── action catalog ─────────────────────────


def valid_action_catalog(
    state: State, mission: Mission
) -> list[tuple[str, str | None, dict[str, str]]]:
    """Enumerate every (action_id, agent_id, params) tuple whose precondition
    `when` is satisfied. For actions with `for_each_agent=True`, expands over
    agents whose cover > 0.

    Returns the catalog the drafter receives.
    """
    catalog: list[tuple[str, str | None, dict[str, str]]] = []
    for action in mission.actions:
        # Forced actions whose id is already in state.fired drop out of the
        # catalog so they can't be re-authorized.
        if action.forced and action.id in state.fired:
            continue
        if action.for_each_agent:
            for agent in mission.agents:
                cover = state.get_quality(f"agent[{agent.id}].cover")
                if cover is None or float(cover) <= 0:
                    continue
                pred = substitute_predicate(action.when, {"agent": agent.id})
                if evaluate_predicate(state, pred):
                    catalog.append((action.id, agent.id, {"agent": agent.id}))
        else:
            if evaluate_predicate(state, action.when):
                catalog.append((action.id, None, {}))
    return catalog


def due_forced_actions(state: State, mission: Mission) -> list[Action]:
    """Forced actions whose `when` holds and which haven't fired yet. The
    drafter MUST surface one of these as an option per day — `day.py` picks
    the first due forced action and routes it through the drafter's prompt as
    a required satisfier."""
    return [
        a for a in mission.actions
        if a.forced
        and a.id not in state.fired
        and evaluate_predicate(state, a.when)
    ]


# ───────────────────────── action application ─────────────────────────


def apply_action_effects(
    state: State, option: Option, mission: Mission
) -> tuple[dict[str, tuple[Any, Any]], list[str]]:
    """Apply the chosen option's action effects to state IN PLACE. Returns
    `(delta, leans)`:

      - For flat actions (`effects:` authored), `leans` is empty.
      - For opposed-check actions (`outcomes:` authored), the first matching
        outcome's effects apply and `leans` is that outcome's `leans` list.

    Forced actions append their id to `state.fired` so the catalog filter and
    `EventFired` predicates can see the firing. Non-forced actions don't
    write to `state.fired` (no need — they're freely re-authorizable).

    If the option binds an agent, `$agent` is substituted in every effect's
    quality and `when`.
    """
    action = mission.action(option.action)
    placeholders = {"agent": option.agent} if option.agent is not None else {}

    if action.outcomes:
        delta, leans = _apply_outcomes(state, action.outcomes, placeholders)
    else:
        delta = _apply_with_substitution(state, action.effects, placeholders)
        leans = []

    if action.forced:
        state.fired.append(action.id)
    return delta, leans


def _apply_with_substitution(
    state: State, effects: list[Effect], placeholders: dict[str, str]
) -> dict[str, tuple[Any, Any]]:
    """Apply effects with optional `$name` substitution. Fast-path: skip the
    Effect-rebuild pass when `placeholders` is empty (the common case for
    unbound actions / reactions without `for_each_agent`)."""
    if placeholders:
        effects = [substitute_effect(e, placeholders) for e in effects]
    return _apply_many(state, effects)


def _apply_outcomes(
    state: State, outcomes: list[Outcome], placeholders: dict[str, str]
) -> tuple[dict[str, tuple[Any, Any]], list[str]]:
    """Walk outcomes, find the first whose `when:` matches, apply its effects.
    Returns `(delta, leans)`. The mutual-exclusion validator on `Action.outcomes`
    guarantees at most one outcome matches a given state — first-match-wins is
    safe."""
    for outcome in outcomes:
        pred = (
            substitute_predicate(outcome.when, placeholders) if placeholders else outcome.when
        )
        if not evaluate_predicate(state, pred):
            continue
        return _apply_with_substitution(state, outcome.effects, placeholders), list(outcome.leans)
    return {}, []


def option_progresses_mission(delta: dict[str, tuple[Any, Any]], mission: Mission) -> bool:
    """True iff the delta moved a non-cover, non-countdown `mission.*` quality
    forward. Used to gate passive-pressure reactions (`only_when_no_progress_this_day`)."""
    for quality, (before, after) in delta.items():
        if not quality.startswith("mission."):
            continue
        if quality == "mission.days_remaining":
            continue
        if before is None and after is not None:
            return True
        if before is not None and after is not None and float(after) > float(before):
            return True
    return False


# ───────────────────────── reaction firing ─────────────────────────


def per_agent_key(reaction_id: str, agent_id: str) -> str:
    """Canonical key format for `state.fired_per_agent`. The live spine write
    and the fold-replay write both go through this helper."""
    return f"{reaction_id}:{agent_id}"


def _reaction_fired_already(state: State, reaction: Reaction, agent_id: str | None) -> bool:
    if reaction.fires_at_most == "unlimited":
        return False
    if reaction.fires_at_most == "once_per_mission":
        return reaction.id in state.fired
    # once_per_agent — validator pins this to `for_each_agent=True`.
    assert agent_id is not None, "once_per_agent reaction fired without agent_id"
    return per_agent_key(reaction.id, agent_id) in state.fired_per_agent


def merge_delta_into(
    target: dict[str, tuple[Any, Any]], source: dict[str, tuple[Any, Any]]
) -> None:
    """Merge `source` delta into `target` in place — first-seen `before` wins,
    latest `after` wins."""
    for q, (b, a) in source.items():
        existing = target.get(q)
        target[q] = (existing[0] if existing else b, a)


def fire_due_reactions(
    state: State, mission: Mission, made_progress: bool
) -> tuple[list[tuple[Reaction, str | None, dict[str, tuple[Any, Any]]]], dict[str, tuple[Any, Any]]]:
    """Fire every eligible Reaction in (priority, authored) order. Mutates
    state in place. Returns:
        - list of (reaction, agent_id_or_None, per-reaction delta)
        - aggregate delta across all fires

    For `for_each_agent` reactions, fires once per matching agent. Stops the
    pass on a `terminal` fire.
    """
    fires: list[tuple[Reaction, str | None, dict[str, tuple[Any, Any]]]] = []
    aggregate: dict[str, tuple[Any, Any]] = {}

    ordered = sorted(
        enumerate(mission.reactions), key=lambda x: (x[1].priority, x[0])
    )

    for _, reaction in ordered:
        if state.outcome is not None:
            break
        if reaction.only_when_no_progress_this_day and made_progress:
            continue

        agent_ids: list[str | None] = (
            [a.id for a in mission.agents] if reaction.for_each_agent else [None]
        )

        for agent_id in agent_ids:
            if _reaction_fired_already(state, reaction, agent_id):
                continue
            placeholders = {"agent": agent_id} if agent_id is not None else {}
            pred = (
                substitute_predicate(reaction.when, placeholders) if placeholders else reaction.when
            )
            if not evaluate_predicate(state, pred):
                continue
            delta = _apply_with_substitution(state, reaction.effects, placeholders)
            state.fired.append(reaction.id)
            if agent_id is not None:
                state.fired_per_agent.append(per_agent_key(reaction.id, agent_id))
            if reaction.terminal and reaction.outcome_key is not None:
                state.outcome = reaction.outcome_key
            fires.append((reaction, agent_id, delta))
            merge_delta_into(aggregate, delta)
            if reaction.terminal:
                break

    return fires, aggregate


# ───────────────────────── day close ─────────────────────────


def tick_day_close(state: State) -> dict[str, tuple[Any, Any]]:
    """End-of-day housekeeping. Decrements `mission.days_remaining`. One-day
    flags (e.g. cover_protected) are cleared by explicit reset Reactions in
    the mission YAML — see `clear_cover_protected` in lynx_exfil.yaml for the
    pattern."""
    delta: dict[str, tuple[Any, Any]] = {}
    cur = state.get_quality("mission.days_remaining")
    if cur is not None:
        before = cur
        after = int(cur) - 1
        state.set_quality("mission.days_remaining", after)
        delta["mission.days_remaining"] = (before, after)
    return delta


# ───────────────────────── helpers ─────────────────────────


def snapshot(state: State) -> State:
    """Deep copy a State — used for replay and the debug timeline."""
    return copy.deepcopy(state)


# ───────────────────────── narrative-shape derivations ─────────────────────────


# Cover at or below this value puts the mission in crisis act phase.
_COVER_BRINK = 2
# When `days_remaining / max_days` drops to / below this, climax begins.
_CLIMAX_THRESHOLD = 0.34
# When the ratio is at / above this, the setup phase still holds (provided
# no cover loss has happened yet).
_SETUP_THRESHOLD = 0.66
# Cover swing thresholds for derive_tension_band — values are absolute
# magnitudes summed across all .cover qualities this day.
_CRISIS_SWING = 4
_ESCALATING_SWING = 2


def _is_numeric(v: Any) -> bool:
    """True for int/float values that should be treated as numbers — excludes
    bool (which `isinstance(True, int)` returns True for) and None."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def derive_act_phase(state: State, mission: Mission) -> ActPhase:
    """Classify the current day into a four-act register so the renderer and
    SoP can adopt an appropriate dramatic tone. Mission-agnostic — uses
    universal qualities only (days_remaining + per-agent cover).

    Precedence: crisis > climax > setup > complication.
    """
    days_raw = state.get_quality("mission.days_remaining")
    days = int(days_raw) if _is_numeric(days_raw) else 0
    max_days = mission.max_days
    covers: list[tuple[CastMember, Any]] = [
        (a, state.get_quality(f"agent[{a.id}].cover")) for a in mission.agents
    ]
    for _, cover in covers:
        if _is_numeric(cover) and int(cover) <= _COVER_BRINK:
            return "crisis"
    if days <= 1:
        return "crisis"
    if max_days > 0 and days / max_days <= _CLIMAX_THRESHOLD:
        return "climax"
    if max_days > 0 and days / max_days >= _SETUP_THRESHOLD:
        if all(
            not _is_numeric(cover) or int(cover) >= a.starting_cover
            for a, cover in covers
        ):
            return "setup"
    return "complication"


def derive_tension_band(prior_state: State | None, state: State) -> TensionBand:
    """Classify the magnitude of cover swing this day."""
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


def select_beat(state: State, mission: Mission) -> Beat | None:
    """Return the highest-priority `when`-matching beat that hasn't fired
    this mission, or None. Read-only; `day.fire_beat` is the single fire
    site that appends the beat id to `state.fired`."""
    candidates = sorted(
        enumerate(mission.beats), key=lambda x: (x[1].priority, x[0])
    )
    for _, beat in candidates:
        if (
            beat.fires_at_most == "once_per_mission"
            and beat.id in state.fired
        ):
            continue
        if evaluate_predicate(state, beat.when):
            return beat
    return None


def invalid_action_catalog(
    state: State, mission: Mission
) -> list[tuple[str, str | None, str]]:
    """For each (action, agent) combination whose precondition is NOT
    satisfied, return (action_id, agent_id_or_None, reason). Surfaced to the
    drafter so it can reason about *why* an option is unavailable instead of
    only seeing the trimmed catalog."""
    invalid: list[tuple[str, str | None, str]] = []
    for action in mission.actions:
        # Don't surface already-fired forced actions in the locked list —
        # they're done, not unavailable.
        if action.forced and action.id in state.fired:
            continue
        if action.for_each_agent:
            for agent in mission.agents:
                cover = state.get_quality(f"agent[{agent.id}].cover")
                if not _is_numeric(cover) or float(cover) <= 0:
                    invalid.append((action.id, agent.id, f"agent[{agent.id}] is lost"))
                    continue
                pred = substitute_predicate(action.when, {"agent": agent.id})
                if evaluate_predicate(state, pred):
                    continue
                invalid.append((action.id, agent.id, _describe_failing_predicate(state, pred)))
        else:
            if evaluate_predicate(state, action.when):
                continue
            invalid.append((action.id, None, _describe_failing_predicate(state, action.when)))
    return invalid


def _describe_failing_predicate(state: State, pred: Predicate) -> str:
    """Walk the predicate tree; return a short description of the first leaf
    that evaluates False."""
    for leaf in iter_predicate_leaves(pred):
        if not evaluate_predicate(state, leaf):
            return _format_leaf(state, leaf)
    return "preconditions not met"


_COMPARISON_LEAF_TEMPLATES: dict[type, str] = {
    QualityGte:    "{q} is {v} (requires >= {t})",
    QualityGt:     "{q} is {v} (requires > {t})",
    QualityLte:    "{q} is {v} (requires <= {t})",
    QualityLt:     "{q} is {v} (requires < {t})",
    QualityEquals: "{q} is {v} (requires == {t!r})",
}


def _format_leaf(state: State, leaf: Predicate) -> str:
    """Render one failing leaf as a short prose hint."""
    template = _COMPARISON_LEAF_TEMPLATES.get(type(leaf))
    if template is not None:
        return template.format(
            q=leaf.quality, v=state.get_quality(leaf.quality), t=leaf.value
        )
    if isinstance(leaf, EventFired):
        return f"event {leaf.event!r} has not fired"
    return "precondition not met"
