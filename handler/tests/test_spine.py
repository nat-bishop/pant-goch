"""Engine unit tests against the tiny test fixture."""
from __future__ import annotations

from pathlib import Path

import pytest

from handler.mission_loader import load_mission
from handler.models import (
    Beat,
    Mission,
    Option,
    QualityGte,
    QualityLte,
    make_initial_state,
)
from handler.spine import (
    apply_archetype_effects,
    apply_forced_event_effects,
    derive_act_phase,
    derive_tension_band,
    due_forced_events,
    evaluate_predicate,
    fire_due_passive_events,
    invalid_archetype_combinations,
    option_progresses_mission,
    select_beat,
    snapshot,
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
    # Find the "win" event in mission.events (passive terminal: progress >= 2).
    win_event = next(e for e in mission.events if e.id == "win")
    pred = win_event.when
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
    delta, branches = apply_archetype_effects(state, opt, mission)
    assert state.get_quality("mission.progress") == 1
    assert delta["mission.progress"] == (0, 1)
    assert branches == set()  # PUSH has no branch-tagged effects


def test_apply_archetype_effects_substitutes_agent(mission, state):
    opt = Option(
        archetype="SACRIFICE_BY_AGENT",
        agent="ALPHA",
        title="t",
        prose="p",
    )
    delta, _branches = apply_archetype_effects(state, opt, mission)
    # ALPHA cover: 5 → 3 (-2); progress: 0 → 1
    assert state.get_quality("agent[ALPHA].cover") == 3
    assert state.get_quality("mission.progress") == 1
    assert state.get_quality("agent[BRAVO].cover") == 5  # untouched


def test_apply_archetype_effects_with_conditional_branches(state):
    # The LYNX-Exfil mission's RUN_EXFIL is the canonical opposed-check case.
    # Default initial state has naming_progress=2 (clean band, < 6).
    from pathlib import Path
    from handler.mission_loader import load_mission

    m = load_mission(Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml")
    s = make_initial_state(m)
    s.set_quality("mission.extraction_stage", 2)
    s.set_quality("agent[LYNX].cover", 7)
    opt = Option(archetype="RUN_EXFIL", title="t", prose="p")
    delta, branches = apply_archetype_effects(s, opt, m)
    # Clean branch (naming_progress < 6): -2 cover, stage advances.
    assert s.get_quality("agent[LYNX].cover") == 5
    assert s.get_quality("mission.extraction_stage") == 3
    assert branches == {"clean"}


def test_apply_archetype_effects_partial_branch(state):
    """Mid-band naming_progress (6..8) triggers the partial branch on
    RUN_EXFIL — LYNX cover takes a -4 hit and KESTREL is the cost."""
    from pathlib import Path
    from handler.mission_loader import load_mission

    m = load_mission(Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml")
    s = make_initial_state(m)
    s.set_quality("mission.extraction_stage", 2)
    s.set_quality("agent[LYNX].cover", 7)
    s.set_quality("mission.opposition.naming_progress", 7)
    opt = Option(archetype="RUN_EXFIL", title="t", prose="p")
    delta, branches = apply_archetype_effects(s, opt, m)
    assert s.get_quality("agent[LYNX].cover") == 3  # 7 - 4
    assert s.get_quality("mission.extraction_stage") == 3
    assert s.get_quality("agent[KESTREL].cover") == 0
    assert branches == {"partial"}


def test_apply_archetype_effects_failure_branch(state):
    """High-band naming_progress (>= 9) on RUN_EXFIL forces naming = 10 so
    the lynx_named terminal fires next passive pass. Cover unchanged in this
    branch — the loss is the file closing, not LYNX collapsing in the field."""
    from pathlib import Path
    from handler.mission_loader import load_mission

    m = load_mission(Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml")
    s = make_initial_state(m)
    s.set_quality("mission.extraction_stage", 2)
    s.set_quality("agent[LYNX].cover", 7)
    s.set_quality("mission.opposition.naming_progress", 9)
    opt = Option(archetype="RUN_EXFIL", title="t", prose="p")
    delta, branches = apply_archetype_effects(s, opt, m)
    assert s.get_quality("mission.opposition.naming_progress") == 10
    assert branches == {"failure"}


def test_cover_clamp_at_zero(mission, state):
    state.set_quality("agent[ALPHA].cover", 1)
    opt = Option(archetype="SACRIFICE_BY_AGENT", agent="ALPHA", title="t", prose="p")
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


# ───────────────────────── forced-event evaluation ─────────────────────────


def test_due_forced_events_fires_when_predicate_holds(mission, state):
    assert due_forced_events(state, mission) == []  # progress=0, predicate wants 1
    state.set_quality("mission.progress", 1)
    events = due_forced_events(state, mission)
    assert [e.id for e in events] == ["turning_point"]


def test_apply_forced_event_effects_records_fire(mission, state):
    state.set_quality("mission.progress", 1)
    event = due_forced_events(state, mission)[0]
    opt = Option(archetype="PUSH", title="t", prose="p", satisfies_event="turning_point")
    apply_forced_event_effects(state, event, opt)
    assert "turning_point" in state.fired
    assert state.get_quality("mission.progress") == 2  # 1 + 1
    assert state.get_quality("agent[ALPHA].cover") == 2  # 5 - 3


def test_due_forced_events_skips_already_fired(mission, state):
    state.set_quality("mission.progress", 1)
    state.fired.append("turning_point")
    assert due_forced_events(state, mission) == []


# ───────────────────────── passive-event firing ─────────────────────────


def test_terminal_event_short_circuits(mission, state):
    state.set_quality("mission.progress", 2)
    fires, _ = fire_due_passive_events(state, mission, made_progress=True)
    fired_ids = [ev.id for (ev, _, _) in fires]
    assert "win" in fired_ids
    assert state.outcome == "win"
    # passive_decay shouldn't fire after terminal short-circuit.
    assert "passive_decay" not in fired_ids


def test_passive_event_only_fires_without_progress(mission, state):
    fires, _ = fire_due_passive_events(state, mission, made_progress=True)
    assert all(ev.id != "passive_decay" for (ev, _, _) in fires)


def test_passive_event_fires_on_no_progress(mission, state):
    fires, _ = fire_due_passive_events(state, mission, made_progress=False)
    # passive_decay drops ALPHA cover by 1 (BRAVO unaffected — predicate guards).
    decay = [(ev, ag, d) for (ev, ag, d) in fires if ev.id == "passive_decay"]
    assert decay
    assert state.get_quality("agent[ALPHA].cover") == 4


def test_for_each_agent_event_fires_once_per_matching_agent(mission, state):
    state.set_quality("agent[ALPHA].cover", 0)
    fires, _ = fire_due_passive_events(state, mission, made_progress=False)
    agent_lost_fires = [(ev, ag) for (ev, ag, _) in fires if ev.id == "agent_lost"]
    assert len(agent_lost_fires) == 1
    assert agent_lost_fires[0][1] == "ALPHA"
    assert state.get_quality("agent[ALPHA].lost") is True
    # Re-firing same cycle: doesn't fire again.
    fires2, _ = fire_due_passive_events(state, mission, made_progress=False)
    assert all(ev.id != "agent_lost" for (ev, _, _) in fires2)


# ───────────────────────── cycle close ─────────────────────────


def test_tick_cycle_close_decrements_cycles_remaining(mission, state):
    delta = tick_cycle_close(state)
    assert state.get_quality("mission.cycles_remaining") == 3
    assert delta["mission.cycles_remaining"] == (4, 3)


def test_tick_cycle_close_clears_one_cycle_flags(state):
    state.one_cycle_flags["protected"] = True
    tick_cycle_close(state)
    assert state.one_cycle_flags == {}


# ───────────────────────── act phase ─────────────────────────


def test_derive_act_phase_setup_at_full_window(mission, state):
    """Cycle 1: full window, no cover loss → setup."""
    assert derive_act_phase(state, mission) == "setup"


def test_derive_act_phase_complication_after_partial_loss(mission, state):
    """Once any cover has dropped, we're no longer in setup even with the
    window still wide open."""
    state.set_quality("agent[ALPHA].cover", 4)  # was 5
    assert derive_act_phase(state, mission) == "complication"


def test_derive_act_phase_climax_when_window_closing():
    """Cycles below the 0.34 ratio → climax (provided no agent is on the
    brink and we're not at the final cycle). Tiny mission has max_cycles=4,
    which collapses climax into crisis; build a longer mission inline."""
    from handler.models import CastMember, Archetype, Event
    m = Mission(
        id="long",
        name="long",
        max_cycles=10,
        cast=[
            CastMember(id="A", codename="A", starting_cover=8),
            CastMember(id="B", codename="B", starting_cover=8),
        ],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        events=[Event(id="end", terminal=True, outcome_key="w", when=QualityGte(quality="x", value=0))],
        outcomes={"w": ""},
    )
    s = make_initial_state(m)
    s.set_quality("mission.cycles_remaining", 3)  # 3/10 = 0.3 → climax
    assert derive_act_phase(s, m) == "climax"


def test_derive_act_phase_crisis_when_agent_brink(mission, state):
    """Any agent at cover <= 2 → crisis regardless of cycles."""
    state.set_quality("agent[ALPHA].cover", 2)
    assert derive_act_phase(state, mission) == "crisis"


def test_derive_act_phase_crisis_at_one_cycle_left(mission, state):
    """One cycle left is crisis even if cover is fine."""
    state.set_quality("mission.cycles_remaining", 1)
    assert derive_act_phase(state, mission) == "crisis"


# ───────────────────────── tension band ─────────────────────────


def test_derive_tension_band_steady_when_no_prior(mission, state):
    assert derive_tension_band(None, state) == "steady"


def test_derive_tension_band_steady_when_no_swing(mission, state):
    prior = snapshot(state)
    assert derive_tension_band(prior, state) == "steady"


def test_derive_tension_band_escalating_on_two_cover_drop(mission, state):
    prior = snapshot(state)
    state.set_quality("agent[ALPHA].cover", 3)  # -2
    assert derive_tension_band(prior, state) == "escalating"


def test_derive_tension_band_crisis_on_four_cover_swing(mission, state):
    prior = snapshot(state)
    state.set_quality("agent[ALPHA].cover", 1)  # -4
    assert derive_tension_band(prior, state) == "crisis"


def test_derive_tension_band_sums_across_agents(mission, state):
    prior = snapshot(state)
    state.set_quality("agent[ALPHA].cover", 3)  # -2
    state.set_quality("agent[BRAVO].cover", 3)  # -2 → total 4
    assert derive_tension_band(prior, state) == "crisis"


# ───────────────────────── beat selection ─────────────────────────


def test_select_beat_returns_highest_priority_match(mission, state):
    """Lowest `priority` int wins among `when`-matching beats; authored
    order breaks ties (handled by zip-with-enumerate in select_beat)."""
    mission_with_beats = mission.model_copy(
        update={
            "beats": [
                Beat(
                    id="alpha_brink",
                    when=QualityLte(quality="agent[ALPHA].cover", value=2),
                    fragment="ALPHA is at the brink.",
                    priority=1,
                ),
                Beat(
                    id="ambient",
                    when=QualityGte(quality="mission.cycles_remaining", value=1),
                    fragment="The day is still open.",
                    priority=5,
                ),
            ]
        }
    )
    # Cover=5 → brink doesn't match, ambient does
    selected = select_beat(state, mission_with_beats)
    assert selected is not None and selected.id == "ambient"
    # Drop cover → brink matches and (priority=1) wins over ambient (priority=5)
    state.set_quality("agent[ALPHA].cover", 1)
    selected = select_beat(state, mission_with_beats)
    assert selected is not None and selected.id == "alpha_brink"


def test_select_beat_skips_already_fired(mission, state):
    """Once a beat's id is in state.fired, it never re-selects (fire-once)."""
    mission_with_beats = mission.model_copy(
        update={
            "beats": [
                Beat(
                    id="ambient",
                    when=QualityGte(quality="mission.cycles_remaining", value=1),
                    fragment="The day is open.",
                    priority=5,
                ),
            ]
        }
    )
    selected = select_beat(state, mission_with_beats)
    assert selected is not None and selected.id == "ambient"
    state.fired.append("ambient")
    assert select_beat(state, mission_with_beats) is None


def test_select_beat_unlimited_refires(mission, state):
    """`fires_at_most="unlimited"` beats refire every cycle their predicate matches."""
    mission_with_beats = mission.model_copy(
        update={
            "beats": [
                Beat(
                    id="ambient",
                    when=QualityGte(quality="mission.cycles_remaining", value=1),
                    fragment="The day is open.",
                    priority=5,
                    fires_at_most="unlimited",
                ),
            ]
        }
    )
    state.fired.append("ambient")  # simulate prior fire — should NOT skip
    selected = select_beat(state, mission_with_beats)
    assert selected is not None and selected.id == "ambient"


def test_select_beat_returns_none_when_none_match(mission, state):
    """No beats defined → None (caller falls back gracefully)."""
    assert select_beat(state, mission) is None


# ───────────────────────── locked-archetype rationale ─────────────────────────


def test_invalid_archetype_combinations_reports_failing_predicate(mission, state):
    """PUSH requires progress < 2; if we set progress to 2 it should appear
    as invalid with a rationale."""
    state.set_quality("mission.progress", 2)
    invalid = invalid_archetype_combinations(state, mission)
    push_entry = next(((a, ag, r) for (a, ag, r) in invalid if a == "PUSH"), None)
    assert push_entry is not None
    assert "mission.progress" in push_entry[2]
    assert "2" in push_entry[2]


def test_invalid_archetype_combinations_reports_lost_agents(mission, state):
    """An agent at cover 0 makes their agent-bound combinations invalid with
    a clean 'agent is lost' rationale (skips the predicate walk)."""
    state.set_quality("agent[ALPHA].cover", 0)
    invalid = invalid_archetype_combinations(state, mission)
    sacrifice_alpha = next(
        ((a, ag, r) for (a, ag, r) in invalid if a == "SACRIFICE_BY_AGENT" and ag == "ALPHA"),
        None,
    )
    assert sacrifice_alpha is not None
    assert "lost" in sacrifice_alpha[2]


def test_invalid_archetype_combinations_excludes_currently_valid(mission, state):
    """An archetype that's in the valid catalog must NOT appear in invalid."""
    valid = {(a, ag) for (a, ag, _) in valid_archetype_combinations(state, mission)}
    invalid = {(a, ag) for (a, ag, _) in invalid_archetype_combinations(state, mission)}
    assert valid.isdisjoint(invalid)
