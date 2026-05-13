"""Engine unit tests against the tiny test fixture (and a few LYNX-EXFIL
spot checks for the opposed-check outcomes path).

After the Action/Reaction refactor, the public spine surface is:
  valid_action_catalog, due_forced_actions, apply_action_effects,
  fire_due_reactions, tick_day_close, plus derive_* / select_beat /
  invalid_action_catalog.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from handler.mission_loader import load_mission
from handler.models import Option, make_initial_state
from handler.spine import (
    apply_action_effects,
    derive_act_phase,
    derive_tension_band,
    due_forced_actions,
    evaluate_predicate,
    fire_due_reactions,
    invalid_action_catalog,
    option_progresses_mission,
    select_beat,
    snapshot,
    tick_day_close,
    valid_action_catalog,
)


_TINY = Path(__file__).parent / "fixtures" / "tiny_mission.yaml"
_LYNX = Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml"


@pytest.fixture
def mission():
    return load_mission(_TINY)


@pytest.fixture
def state(mission):
    return make_initial_state(mission)


# ───────────────────────── predicate evaluation ─────────────────────────


def test_evaluate_predicate_quality_gte(mission, state):
    win = next(r for r in mission.reactions if r.id == "win")
    assert not evaluate_predicate(state, win.when)
    state.set_quality("mission.progress", 2)
    assert evaluate_predicate(state, win.when)


# ───────────────────────── action catalog ─────────────────────────


def test_valid_action_catalog_expands_over_agents(mission, state):
    catalog = valid_action_catalog(state, mission)
    sacrifice = [c for c in catalog if c[0] == "SACRIFICE_BY_AGENT"]
    agents = {ag for (_, ag, _) in sacrifice}
    assert agents == {"ALPHA", "BRAVO"}
    push = [c for c in catalog if c[0] == "PUSH"]
    assert push and push[0][1] is None


def test_valid_action_catalog_excludes_zero_cover_agents(mission, state):
    state.set_quality("agent[ALPHA].cover", 0)
    catalog = valid_action_catalog(state, mission)
    agents = {ag for (a, ag, _) in catalog if a == "SACRIFICE_BY_AGENT"}
    assert agents == {"BRAVO"}


def test_valid_action_catalog_respects_when(mission, state):
    state.set_quality("mission.progress", 2)
    catalog = valid_action_catalog(state, mission)
    assert not [c for c in catalog if c[0] == "PUSH"]


def test_valid_action_catalog_omits_fired_forced_actions(mission, state):
    state.set_quality("mission.progress", 1)
    state.fired.append("turning_point")
    catalog = valid_action_catalog(state, mission)
    assert not [c for c in catalog if c[0] == "turning_point"]


# ───────────────────────── action effects ─────────────────────────


def test_apply_action_effects_flat(mission, state):
    opt = Option(action="PUSH", title="t", prose="p")
    delta, leans = apply_action_effects(state, opt, mission)
    assert state.get_quality("mission.progress") == 1
    assert delta["mission.progress"] == (0, 1)
    assert leans == []


def test_apply_action_effects_substitutes_agent(mission, state):
    opt = Option(action="SACRIFICE_BY_AGENT", agent="ALPHA", title="t", prose="p")
    apply_action_effects(state, opt, mission)
    assert state.get_quality("agent[ALPHA].cover") == 3
    assert state.get_quality("mission.progress") == 1
    assert state.get_quality("agent[BRAVO].cover") == 5


def test_apply_action_effects_forced_appends_to_fired(mission, state):
    state.set_quality("mission.progress", 1)
    opt = Option(action="turning_point", title="t", prose="p")
    apply_action_effects(state, opt, mission)
    assert "turning_point" in state.fired


def test_apply_action_effects_non_forced_does_not_append(mission, state):
    opt = Option(action="PUSH", title="t", prose="p")
    apply_action_effects(state, opt, mission)
    assert "PUSH" not in state.fired


def test_apply_action_effects_outcomes_clean_branch():
    """LYNX-EXFIL RUN_EXFIL has outcomes — clean branch at low naming_progress."""
    m = load_mission(_LYNX)
    s = make_initial_state(m)
    s.set_quality("mission.extraction_stage", 2)
    s.set_quality("agent[LYNX].cover", 7)
    s.set_quality("mission.opposition.naming_progress", 2)
    opt = Option(action="RUN_EXFIL", title="t", prose="p")
    delta, leans = apply_action_effects(s, opt, m)
    assert s.get_quality("mission.extraction_stage") == 3
    assert s.get_quality("agent[LYNX].cover") == 5  # 7 - 2
    assert leans  # clean-branch leans surfaced


def test_apply_action_effects_outcomes_partial_branch():
    m = load_mission(_LYNX)
    s = make_initial_state(m)
    s.set_quality("mission.extraction_stage", 2)
    s.set_quality("agent[LYNX].cover", 7)
    s.set_quality("mission.opposition.naming_progress", 7)
    opt = Option(action="RUN_EXFIL", title="t", prose="p")
    delta, leans = apply_action_effects(s, opt, m)
    assert s.get_quality("agent[LYNX].cover") == 3  # 7 - 4
    assert s.get_quality("mission.extraction_stage") == 3
    assert s.get_quality("agent[KESTREL].cover") == 0
    assert any("KESTREL is the cost" in lean for lean in leans)


def test_apply_action_effects_outcomes_failure_branch():
    m = load_mission(_LYNX)
    s = make_initial_state(m)
    s.set_quality("mission.extraction_stage", 2)
    s.set_quality("agent[LYNX].cover", 7)
    s.set_quality("mission.opposition.naming_progress", 9)
    opt = Option(action="RUN_EXFIL", title="t", prose="p")
    delta, leans = apply_action_effects(s, opt, m)
    assert s.get_quality("mission.opposition.naming_progress") == 10
    assert any("platform" in lean for lean in leans)


def test_cover_clamp_at_zero(mission, state):
    state.set_quality("agent[ALPHA].cover", 1)
    opt = Option(action="SACRIFICE_BY_AGENT", agent="ALPHA", title="t", prose="p")
    apply_action_effects(state, opt, mission)
    assert state.get_quality("agent[ALPHA].cover") == 0


# ───────────────────────── progress detection ─────────────────────────


def test_option_progresses_mission_on_stage_increase(mission):
    assert option_progresses_mission({"mission.progress": (0, 1)}, mission)


def test_option_progresses_mission_ignores_cover_only(mission):
    assert not option_progresses_mission({"agent[ALPHA].cover": (5, 3)}, mission)


def test_option_progresses_mission_ignores_day_countdown(mission):
    assert not option_progresses_mission({"mission.days_remaining": (4, 3)}, mission)


# ───────────────────────── forced action enumeration ─────────────────────────


def test_due_forced_actions_fires_when_predicate_holds(mission, state):
    assert due_forced_actions(state, mission) == []
    state.set_quality("mission.progress", 1)
    forced = due_forced_actions(state, mission)
    assert [a.id for a in forced] == ["turning_point"]


def test_due_forced_actions_skips_already_fired(mission, state):
    state.set_quality("mission.progress", 1)
    state.fired.append("turning_point")
    assert due_forced_actions(state, mission) == []


# ───────────────────────── reaction firing ─────────────────────────


def test_terminal_reaction_short_circuits(mission, state):
    state.set_quality("mission.progress", 2)
    fires, _ = fire_due_reactions(state, mission, made_progress=True)
    fired_ids = [r.id for (r, _, _) in fires]
    assert "win" in fired_ids
    assert state.outcome == "win"
    assert "passive_decay" not in fired_ids


def test_passive_reaction_only_fires_without_progress(mission, state):
    fires, _ = fire_due_reactions(state, mission, made_progress=True)
    assert all(r.id != "passive_decay" for (r, _, _) in fires)


def test_passive_reaction_fires_on_no_progress(mission, state):
    fires, _ = fire_due_reactions(state, mission, made_progress=False)
    decay = [(r, ag, d) for (r, ag, d) in fires if r.id == "passive_decay"]
    assert decay
    assert state.get_quality("agent[ALPHA].cover") == 4


def test_for_each_agent_reaction_fires_once_per_matching_agent(mission, state):
    state.set_quality("agent[ALPHA].cover", 0)
    fires, _ = fire_due_reactions(state, mission, made_progress=False)
    agent_lost = [(r, ag) for (r, ag, _) in fires if r.id == "agent_lost"]
    assert len(agent_lost) == 1
    assert agent_lost[0][1] == "ALPHA"
    # Re-firing same day: doesn't fire again.
    fires2, _ = fire_due_reactions(state, mission, made_progress=False)
    assert all(r.id != "agent_lost" for (r, _, _) in fires2)


# ───────────────────────── day close ─────────────────────────


def test_tick_day_close_decrements_days_remaining(mission, state):
    delta = tick_day_close(state)
    assert state.get_quality("mission.days_remaining") == 3
    assert delta["mission.days_remaining"] == (4, 3)


# ───────────────────────── act phase ─────────────────────────


def test_derive_act_phase_setup_at_full_window(mission, state):
    assert derive_act_phase(state, mission) == "setup"


def test_derive_act_phase_complication_after_partial_loss(mission, state):
    state.set_quality("agent[ALPHA].cover", 4)
    assert derive_act_phase(state, mission) == "complication"


def test_derive_act_phase_crisis_when_agent_brink(mission, state):
    state.set_quality("agent[ALPHA].cover", 2)
    assert derive_act_phase(state, mission) == "crisis"


def test_derive_act_phase_crisis_at_one_day_left(mission, state):
    state.set_quality("mission.days_remaining", 1)
    assert derive_act_phase(state, mission) == "crisis"


# ───────────────────────── tension band ─────────────────────────


def test_derive_tension_band_steady_when_no_prior(state):
    assert derive_tension_band(None, state) == "steady"


def test_derive_tension_band_escalating_on_two_cover_drop(state):
    prior = snapshot(state)
    state.set_quality("agent[ALPHA].cover", 3)  # was 5
    assert derive_tension_band(prior, state) == "escalating"


def test_derive_tension_band_crisis_on_four_cover_swing(state):
    prior = snapshot(state)
    state.set_quality("agent[ALPHA].cover", 1)
    assert derive_tension_band(prior, state) == "crisis"


# ───────────────────────── beat selection ─────────────────────────


def test_select_beat_returns_none_when_no_beats_authored(mission, state):
    assert select_beat(state, mission) is None  # tiny mission has no beats


# ───────────────────────── invalid catalog ─────────────────────────


def test_invalid_action_catalog_reports_failing_predicate(mission, state):
    state.set_quality("mission.progress", 2)
    invalid = invalid_action_catalog(state, mission)
    push = [r for (a, _, r) in invalid if a == "PUSH"]
    assert push  # PUSH is now locked because progress >= 2
