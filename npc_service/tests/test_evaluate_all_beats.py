"""Tests for `evaluate_all_beats` — the per-beat predicate trace builder
that drives the debug Beats inspector.

Differs from the per-tick scheduler tests (test_app.py): no FastAPI client,
no agent overrides — just the pure spine function against a constructed
InboxState and the test-fixture scenario.
"""
from pathlib import Path

import pytest

from npc_service.models import (
    Email,
    InboxState,
    PendingBeat,
    PlayerCanonEntry,
    Reply,
    load_scenario,
)
from npc_service.spine import evaluate_all_beats


@pytest.fixture
def scenario():
    fixture_path = Path(__file__).parent / "fixtures" / "test_scenario.yaml"
    return load_scenario(str(fixture_path))


def _initial_state(scenario) -> InboxState:
    return InboxState(
        qualities=dict(scenario.quality_defaults),
        fired_beat_ids=set(),
        ever_fired_beat_ids=set(),
        pending_beats=[PendingBeat(beat_id=b.id) for b in scenario.beats],
        emails=[],
        replies=[],
        thread_messages={},
        player_canon={},
        ended=False,
        end_reason=None,
    )


def test_covers_every_beat(scenario):
    """Unlike find_due_beats (which prunes structural-blocked beats), every
    scenario beat appears in the evaluation."""
    state = _initial_state(scenario)
    evals = evaluate_all_beats(state, scenario)
    assert {e.beat_id for e in evals} == {b.id for b in scenario.beats}
    assert len(evals) == len(scenario.beats)


def test_traces_parallel_to_preconditions(scenario):
    """entry_traces[i] corresponds to beat.preconditions[i]; exit_traces
    parallel to spec.exit_when. Empty for one_shots."""
    state = _initial_state(scenario)
    evals_by_id = {e.beat_id: e for e in evaluate_all_beats(state, scenario)}

    welcome = evals_by_id["aria.welcome"]
    assert welcome.entry_traces == []  # no preconditions
    assert welcome.exit_traces == []  # one_shot
    assert welcome.is_scene is False

    loop = evals_by_id["aria.loop"]
    loop_beat = scenario.beat_by_id["aria.loop"]
    assert len(loop.entry_traces) == len(loop_beat.preconditions)
    assert len(loop.exit_traces) == len(loop_beat.spec.exit_when)
    assert loop.is_scene is True


def test_status_eligible_when_no_preconditions(scenario):
    """aria.welcome has no preconditions — always eligible until fired."""
    state = _initial_state(scenario)
    evals_by_id = {e.beat_id: e for e in evaluate_all_beats(state, scenario)}
    assert evals_by_id["aria.welcome"].status == "eligible"


def test_status_blocked_when_predicate_fails(scenario):
    """Initial state: 0 engagement, 0 replies. complete_warm and
    complete_standard are gated on replies >= 5; complete_warm also wants
    engagement >= 4. Both blocked."""
    state = _initial_state(scenario)
    evals_by_id = {e.beat_id: e for e in evaluate_all_beats(state, scenario)}
    assert evals_by_id["aria.complete_warm"].status == "blocked"
    assert evals_by_id["aria.complete_standard"].status == "blocked"
    assert evals_by_id["aria.loop"].status == "blocked"  # replies < 1

    # Atom-level verdict — at least one entry trace failed.
    warm_traces = evals_by_id["aria.complete_warm"].entry_traces
    assert any(not t.passed for t in warm_traces)


def test_status_fired_when_in_fired_beat_ids(scenario):
    """aria.welcome enters fired_beat_ids → status switches to 'fired'."""
    state = _initial_state(scenario)
    state.fired_beat_ids.add("aria.welcome")
    evals_by_id = {e.beat_id: e for e in evaluate_all_beats(state, scenario)}
    assert evals_by_id["aria.welcome"].status == "fired"


def test_status_eligible_when_all_preconditions_pass(scenario):
    """Push state forward: 5 replies, engagement 5.0. complete_warm should
    flip to eligible (and so should standard, since neither terminal
    has fired)."""
    state = _initial_state(scenario)
    state.qualities["engagement"] = 5.0
    state.replies = [
        Reply(
            id=f"r{i}", thread_id="aria", beat_responded_to=None,
            recipient="Aria", body="hi", classifier_results=[],
        )
        for i in range(5)
    ]
    evals_by_id = {e.beat_id: e for e in evaluate_all_beats(state, scenario)}
    assert evals_by_id["aria.complete_warm"].status == "eligible"
    assert evals_by_id["aria.complete_standard"].status == "eligible"
    # All entry_traces should be passing for an eligible beat.
    for t in evals_by_id["aria.complete_warm"].entry_traces:
        assert t.passed is True


def test_exit_traces_only_for_scenes(scenario):
    """One_shot beats have empty exit_traces regardless of authoring."""
    state = _initial_state(scenario)
    evals_by_id = {e.beat_id: e for e in evaluate_all_beats(state, scenario)}
    assert evals_by_id["aria.welcome"].exit_traces == []
    assert evals_by_id["aria.complete_warm"].exit_traces == []
    # The scene beat has exit_when authored, so non-empty.
    assert len(evals_by_id["aria.loop"].exit_traces) > 0
