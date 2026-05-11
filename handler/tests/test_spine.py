"""Engine unit tests against the tiny test fixture."""
from __future__ import annotations

from pathlib import Path

import pytest

from handler.mission_loader import load_mission
from handler.models import Option, make_initial_state
from handler.spine import (
    apply_archetype_effects,
    apply_moment_effects,
    due_moments,
    evaluate_predicate,
    fire_due_beats,
    option_progresses_mission,
    tick_cycle_close,
    valid_archetype_combinations,
)


_TINY = Path(__file__).parent / "fixtures" / "tiny_mission.yaml"


@pytest.fixture
def mission():
    return load_mission(_TINY)


@pytest.fixture
def state(mission):
    return make_initial_state(mission)


# ───────────────────────── predicate evaluation ─────────────────────────


def test_evaluate_predicate_quality_gte(mission, state):
    pred = mission.beats[0].when  # "win" beat: progress >= 2
    assert not evaluate_predicate(state, pred)
    state.set_quality("mission.progress", 2)
    assert evaluate_predicate(state, pred)


# ───────────────────────── archetype enumeration ─────────────────────────


def test_valid_archetype_combinations_expands_over_agents(mission, state):
    catalog = valid_archetype_combinations(state, mission)
    # SACRIFICE_BY_AGENT has params:[agent] — expands over both agents whose cover > 0.
    sacrifice_entries = [c for c in catalog if c[0] == "SACRIFICE_BY_AGENT"]
    agents = {ag for (_, ag, _) in sacrifice_entries}
    assert agents == {"ALPHA", "BRAVO"}
    # PUSH and HOLD are unbound (no agent param).
    push = [c for c in catalog if c[0] == "PUSH"]
    assert push and push[0][1] is None


def test_valid_archetype_combinations_excludes_zero_cover_agents(mission, state):
    state.set_quality("agent[ALPHA].cover", 0)
    catalog = valid_archetype_combinations(state, mission)
    agents = {ag for (a, ag, _) in catalog if a == "SACRIFICE_BY_AGENT"}
    assert agents == {"BRAVO"}


def test_valid_archetype_combinations_respects_archetype_when(mission, state):
    # Push out of stage: at progress=2 PUSH's `when: progress < 2` is false.
    state.set_quality("mission.progress", 2)
    catalog = valid_archetype_combinations(state, mission)
    push = [c for c in catalog if c[0] == "PUSH"]
    assert not push


# ───────────────────────── effect application ─────────────────────────


def test_apply_archetype_effects_unbound(mission, state):
    opt = Option(archetype="PUSH", title="t", prose="p")
    delta = apply_archetype_effects(state, opt, mission)
    assert state.get_quality("mission.progress") == 1
    assert delta["mission.progress"] == (0, 1)


def test_apply_archetype_effects_substitutes_agent(mission, state):
    opt = Option(
        archetype="SACRIFICE_BY_AGENT",
        agent="ALPHA",
        params={"agent": "ALPHA"},
        title="t",
        prose="p",
    )
    delta = apply_archetype_effects(state, opt, mission)
    # ALPHA cover: 5 → 3 (-2); progress: 0 → 1
    assert state.get_quality("agent[ALPHA].cover") == 3
    assert state.get_quality("mission.progress") == 1
    assert state.get_quality("agent[BRAVO].cover") == 5  # untouched


def test_apply_archetype_effects_with_conditional_branches(state):
    # The LYNX-Exfil mission's RUN_EXFIL is the canonical conditional case.
    from pathlib import Path
    from handler.mission_loader import load_mission

    m = load_mission(Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml")
    s = make_initial_state(m)
    s.set_quality("mission.extraction_stage", 2)
    s.set_quality("agent[LYNX].cover", 7)
    opt = Option(archetype="RUN_EXFIL", title="t", prose="p")
    apply_archetype_effects(s, opt, m)
    # Clean exit (cover ≥ 5) costs 2.
    assert s.get_quality("agent[LYNX].cover") == 5
    assert s.get_quality("mission.extraction_stage") == 3


def test_cover_clamp_at_zero(mission, state):
    state.set_quality("agent[ALPHA].cover", 1)
    opt = Option(archetype="SACRIFICE_BY_AGENT", agent="ALPHA", params={"agent": "ALPHA"}, title="t", prose="p")
    apply_archetype_effects(state, opt, mission)
    # 1 + (-2) = -1, clamped to 0.
    assert state.get_quality("agent[ALPHA].cover") == 0


# ───────────────────────── progress detection ─────────────────────────


def test_option_progresses_mission_on_stage_increase(mission):
    delta = {"mission.progress": (0, 1)}
    assert option_progresses_mission(delta, mission)


def test_option_progresses_mission_ignores_cover_only(mission):
    delta = {"agent[ALPHA].cover": (5, 3)}
    assert not option_progresses_mission(delta, mission)


def test_option_progresses_mission_ignores_cycle_countdown(mission):
    delta = {"mission.cycles_remaining": (4, 3)}
    assert not option_progresses_mission(delta, mission)


# ───────────────────────── moment evaluation ─────────────────────────


def test_due_moments_fires_when_predicate_holds(mission, state):
    assert due_moments(state, mission) == []  # progress=0, predicate wants 1
    state.set_quality("mission.progress", 1)
    moments = due_moments(state, mission)
    assert [m.id for m in moments] == ["turning_point"]


def test_apply_moment_effects_records_fire(mission, state):
    state.set_quality("mission.progress", 1)
    moment = due_moments(state, mission)[0]
    opt = Option(archetype="PUSH", title="t", prose="p", satisfies_moment="turning_point")
    apply_moment_effects(state, moment, opt)
    assert "turning_point" in state.fired_moments
    assert state.get_quality("mission.progress") == 2  # 1 + 1
    assert state.get_quality("agent[ALPHA].cover") == 2  # 5 - 3


def test_due_moments_skips_already_fired(mission, state):
    state.set_quality("mission.progress", 1)
    state.fired_moments.append("turning_point")
    assert due_moments(state, mission) == []


# ───────────────────────── beat firing ─────────────────────────


def test_terminal_beat_short_circuits(mission, state):
    state.set_quality("mission.progress", 2)
    fires, _ = fire_due_beats(state, mission, made_progress=True)
    fired_ids = [b.id for (b, _, _) in fires]
    assert "win" in fired_ids
    assert state.outcome == "win"
    # passive_decay shouldn't fire after terminal short-circuit.
    assert "passive_decay" not in fired_ids


def test_passive_beat_only_fires_without_progress(mission, state):
    fires, _ = fire_due_beats(state, mission, made_progress=True)
    assert all(b.id != "passive_decay" for (b, _, _) in fires)


def test_passive_beat_fires_on_no_progress(mission, state):
    fires, _ = fire_due_beats(state, mission, made_progress=False)
    # passive_decay drops ALPHA cover by 1 (BRAVO unaffected — predicate guards).
    decay = [(b, ag, d) for (b, ag, d) in fires if b.id == "passive_decay"]
    assert decay
    assert state.get_quality("agent[ALPHA].cover") == 4


def test_for_each_agent_beat_fires_once_per_matching_agent(mission, state):
    state.set_quality("agent[ALPHA].cover", 0)
    fires, _ = fire_due_beats(state, mission, made_progress=False)
    agent_lost_fires = [(b, ag) for (b, ag, _) in fires if b.id == "agent_lost"]
    assert len(agent_lost_fires) == 1
    assert agent_lost_fires[0][1] == "ALPHA"
    assert state.get_quality("agent[ALPHA].lost") is True
    # Re-firing same cycle: doesn't fire again.
    fires2, _ = fire_due_beats(state, mission, made_progress=False)
    assert all(b.id != "agent_lost" for (b, _, _) in fires2)


# ───────────────────────── cycle close ─────────────────────────


def test_tick_cycle_close_decrements_cycles_remaining(mission, state):
    delta = tick_cycle_close(state)
    assert state.get_quality("mission.cycles_remaining") == 3
    assert delta["mission.cycles_remaining"] == (4, 3)


def test_tick_cycle_close_clears_one_cycle_flags(state):
    state.one_cycle_flags["protected"] = True
    tick_cycle_close(state)
    assert state.one_cycle_flags == {}
