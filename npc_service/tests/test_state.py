"""Unit tests for state mutation and end-condition logic.

Pure-Python — no LLM, no FastAPI, no llama-server.
"""
from npc_service.main import apply_response, check_end_conditions
from npc_service.models import CaptorResponse, GameState, HostageState


def _make_state(
    stress: int = 50,
    trust: int = 50,
    desperation: int = 50,
    action_count: int = 0,
    max_actions: int = 30,
) -> GameState:
    return GameState(
        max_actions=max_actions,
        stress=stress,
        trust=trust,
        desperation=desperation,
        action_count=action_count,
        hostages=[
            HostageState(name="Janet", description="hostage 1"),
            HostageState(name="David", description="hostage 2"),
        ],
    )


def test_apply_response_clamps_meters_to_0_100():
    state = _make_state(stress=10, trust=95, desperation=5)
    apply_response(state, CaptorResponse(stress_delta=-20, trust_delta=20, desperation_delta=-20))
    assert state.stress == 0
    assert state.trust == 100
    assert state.desperation == 0


def test_apply_response_release_hostage_marks_released():
    state = _make_state()
    apply_response(state, CaptorResponse(action="release_hostage", action_target="Janet"))
    janet = state.hostage_by_name("Janet")
    assert janet is not None
    assert janet.status == "released"
    # game does not end on a single release
    assert not state.ended


def test_apply_response_release_unknown_target_is_noop():
    state = _make_state()
    apply_response(state, CaptorResponse(action="release_hostage", action_target="Mystery"))
    assert all(h.status == "held" for h in state.hostages)
    assert not state.ended


def test_apply_response_harm_hostage_ends_game():
    state = _make_state()
    apply_response(state, CaptorResponse(action="harm_hostage", action_target="David"))
    assert state.ended
    assert state.captor_outcome == "harmed"
    assert state.end_reason == "hostage_harmed"


def test_apply_response_surrender_ends_game():
    state = _make_state()
    apply_response(state, CaptorResponse(action="surrender"))
    assert state.ended
    assert state.captor_outcome == "surrendered"
    assert state.end_reason == "captor_surrendered"


def test_releasing_both_hostages_ends_game_with_all_released():
    state = _make_state()
    apply_response(state, CaptorResponse(action="release_hostage", action_target="Janet"))
    assert not state.ended
    apply_response(state, CaptorResponse(action="release_hostage", action_target="David"))
    assert state.ended
    assert state.end_reason == "all_released"


def test_high_desperation_triggers_auto_breach():
    state = _make_state(desperation=85)
    apply_response(state, CaptorResponse(desperation_delta=10))  # → 95
    assert state.ended
    assert state.captor_outcome == "breached"
    assert state.end_reason == "auto_breach"


def test_action_budget_exhaustion_ends_game():
    state = _make_state(action_count=30, max_actions=30)
    check_end_conditions(state)
    assert state.ended
    assert state.captor_outcome == "breached"
    assert state.end_reason == "budget_exhausted"


def test_under_budget_does_not_end():
    state = _make_state(action_count=29, max_actions=30)
    check_end_conditions(state)
    assert not state.ended


def test_already_ended_state_is_not_overwritten():
    state = _make_state()
    state.ended = True
    state.captor_outcome = "surrendered"
    state.end_reason = "captor_surrendered"
    # re-running end checks must not flip a settled outcome
    for h in state.hostages:
        h.status = "released"
    check_end_conditions(state)
    assert state.captor_outcome == "surrendered"
    assert state.end_reason == "captor_surrendered"
