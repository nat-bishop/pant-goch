"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, effect application, beat scheduler, voice band selection, time
helpers, and reply scheduling.

No LLM, no FastAPI. Runs fast.
"""
from pathlib import Path

import pytest

from npc_service.models import (
    AnchoredBeat,
    InboxState,
    Reply,
    Scenario,
    SetQuality,
    ScheduleBeat,
    load_scenario,
)
from npc_service.spine import (
    advance_time,
    apply_effect,
    apply_texture_score,
    check_end_conditions,
    find_due_beats,
    initialize_boot_inbox,
    schedule_npc_reply,
    select_voice_band,
)


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    path = Path(__file__).resolve().parents[1] / "scenario.yaml"
    return load_scenario(path)


@pytest.fixture
def state(scenario: Scenario) -> InboxState:
    return InboxState.from_scenario(scenario)


# ───────────────────────── scenario validation ─────────────────────────


def test_scenario_loads_and_is_internally_consistent(scenario: Scenario):
    assert scenario.title == "Threads"
    assert scenario.max_hours == 96.0
    assert {n.id for n in scenario.npcs} == {"sam", "alex"}
    assert {t.id for t in scenario.threads} >= {"bounces", "sam", "alex"}
    boot_count = sum(1 for b in scenario.beats if getattr(b, "in_boot_inbox", False))
    assert boot_count == 2, f"expected 2 boot-inbox beats, got {boot_count}"


def test_initial_state_starts_at_zero(state: InboxState):
    assert state.game_time_hours == 0.0
    assert state.qualities["warmth.sam"] == 0.0
    assert state.qualities["register.sam"] == 0.0
    assert state.qualities["warmth.alex"] == 0.0
    assert state.ended is False


def test_boot_inbox_returns_two_beats(state: InboxState, scenario: Scenario):
    boot = initialize_boot_inbox(state, scenario)
    assert len(boot) == 2
    ids = {b.beat.id for b in boot}
    assert ids == {"bounces.001", "sam.opening"}
    for b in boot:
        assert isinstance(b.beat, AnchoredBeat)


# ───────────────────────── effect application ─────────────────────────


def test_set_quality_add_op_clamps_to_max(state: InboxState, scenario: Scenario):
    apply_effect(SetQuality(quality="register.sam", op="set", value=2.5), state, scenario)
    apply_effect(SetQuality(quality="register.sam", op="add", value=10.0), state, scenario)
    assert state.qualities["register.sam"] == 3.0  # max=3


def test_drift_halfway_op(state: InboxState, scenario: Scenario):
    state.qualities["register.sam"] = 2.0
    apply_effect(SetQuality(quality="register.sam", op="drift_halfway", value=0.0), state, scenario)
    assert state.qualities["register.sam"] == 1.0


def test_schedule_beat_appears_in_pending(state: InboxState, scenario: Scenario):
    apply_effect(ScheduleBeat(beat="sam.first_response", after_hours=1.5), state, scenario)
    assert any(pb.beat_id == "sam.first_response" for pb in state.pending_beats)


def test_schedule_beat_keeps_earlier_deadline(state: InboxState, scenario: Scenario):
    state.game_time_hours = 5.0
    apply_effect(ScheduleBeat(beat="sam.first_response", after_hours=2.0), state, scenario)
    apply_effect(ScheduleBeat(beat="sam.first_response", after_hours=10.0), state, scenario)
    pending = [pb for pb in state.pending_beats if pb.beat_id == "sam.first_response"]
    assert len(pending) == 1
    assert pending[0].earliest_time_hours == 7.0  # 5 + 2, not 5 + 10


# ───────────────────────── texture / register loop ─────────────────────────


def test_warm_reply_increases_warmth_and_drifts_register(state: InboxState, scenario: Scenario):
    apply_texture_score(state, scenario, recipient_npc_id="sam", warmth=2.0)
    assert state.qualities["warmth.sam"] == 2.0
    # Register drifts halfway from 0 toward 2 → 1.0.
    assert state.qualities["register.sam"] == 1.0


def test_curt_reply_pulls_register_negative(state: InboxState, scenario: Scenario):
    apply_texture_score(state, scenario, recipient_npc_id="sam", warmth=-2.0)
    assert state.qualities["register.sam"] == -1.0


def test_register_drift_visible_within_one_round(state: InboxState, scenario: Scenario):
    """The texture loop must close per-round. Two replies should produce
    visibly different register values."""
    apply_texture_score(state, scenario, recipient_npc_id="sam", warmth=3.0)
    after_warm = state.qualities["register.sam"]
    apply_texture_score(state, scenario, recipient_npc_id="sam", warmth=-3.0)
    after_cold = state.qualities["register.sam"]
    assert after_warm > after_cold
    assert after_cold < 0


def test_unknown_recipient_does_not_mutate_qualities(state: InboxState, scenario: Scenario):
    apply_texture_score(state, scenario, recipient_npc_id=None, warmth=2.0)
    assert state.qualities["warmth.sam"] == 0.0
    assert state.qualities["register.sam"] == 0.0


# ───────────────────────── voice band selection ─────────────────────────


def test_voice_band_helpful_at_zero(state: InboxState, scenario: Scenario):
    sam = next(n for n in scenario.npcs if n.id == "sam")
    band = select_voice_band(sam, state)
    assert band is not None and band.name == "helpful"


def test_voice_band_enthusiastic_when_register_high(state: InboxState, scenario: Scenario):
    state.qualities["register.sam"] = 2.0
    sam = next(n for n in scenario.npcs if n.id == "sam")
    band = select_voice_band(sam, state)
    assert band is not None and band.name == "enthusiastic"


def test_voice_band_terse_when_register_low(state: InboxState, scenario: Scenario):
    state.qualities["register.sam"] = -2.0
    sam = next(n for n in scenario.npcs if n.id == "sam")
    band = select_voice_band(sam, state)
    assert band is not None and band.name == "terse"


# ───────────────────────── time advancement ─────────────────────────


def test_advance_time_increments_clock(state: InboxState):
    advance_time(state, 2.5)
    assert state.game_time_hours == 2.5
    advance_time(state, 1.0)
    assert state.game_time_hours == 3.5


# ───────────────────────── beat scheduler ─────────────────────────


def test_time_gated_anchored_beat_fires_at_threshold(state: InboxState, scenario: Scenario):
    state.game_time_hours = 1.0
    due = find_due_beats(state, scenario)
    assert not any(bf.beat.id == "alex.checking_in" for bf in due)
    state.game_time_hours = 2.0
    due = find_due_beats(state, scenario)
    assert any(bf.beat.id == "alex.checking_in" for bf in due)


def test_scheduled_beat_only_fires_once_time_reached(state: InboxState, scenario: Scenario):
    apply_effect(ScheduleBeat(beat="sam.first_response", after_hours=2.0), state, scenario)
    state.game_time_hours = 1.0
    due = find_due_beats(state, scenario)
    assert not any(bf.beat.id == "sam.first_response" for bf in due)
    state.game_time_hours = 2.5
    due = find_due_beats(state, scenario)
    assert any(bf.beat.id == "sam.first_response" for bf in due)


# ───────────────────────── reply scheduling ─────────────────────────


def test_schedule_npc_reply_picks_first_eligible_beat(state: InboxState, scenario: Scenario):
    state.fired_beat_ids.add("sam.opening")  # boot beat fired
    state.game_time_hours = 0.25
    scheduled = schedule_npc_reply(state, scenario, "sam")
    assert scheduled is True
    # First eligible Sam beat is sam.first_response (no preconditions).
    pending = [pb for pb in state.pending_beats if pb.beat_id == "sam.first_response"]
    assert len(pending) == 1
    # Sam's reply latency is [0.5, 2.0]; midpoint = 1.25; from t=0.25 → 1.5h.
    assert pending[0].earliest_time_hours == pytest.approx(1.5)


def test_schedule_npc_reply_skips_when_pending(state: InboxState, scenario: Scenario):
    """Subsequent sends to an NPC with a pending reply don't reschedule."""
    state.fired_beat_ids.add("sam.opening")
    schedule_npc_reply(state, scenario, "sam")
    initial_pending = list(state.pending_beats)
    state.game_time_hours = 0.5
    scheduled_again = schedule_npc_reply(state, scenario, "sam")
    assert scheduled_again is False
    assert state.pending_beats == initial_pending


def test_schedule_npc_reply_respects_preconditions(state: InboxState, scenario: Scenario):
    """sam.with_evidence requires player_replies_to_thread_gte sam, 2.
    Without enough player replies it is skipped; sam.first_response wins."""
    state.fired_beat_ids.update({"sam.opening"})
    schedule_npc_reply(state, scenario, "sam")
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert "sam.first_response" in pending_ids
    assert "sam.with_evidence" not in pending_ids


def test_player_replies_to_thread_predicate(state: InboxState, scenario: Scenario):
    """Once two player replies sit in the sam thread, sam.with_evidence
    becomes the next eligible beat."""
    state.fired_beat_ids.update({"sam.opening", "sam.first_response"})
    state.replies.extend([
        _make_reply(thread="sam", warmth=1.0),
        _make_reply(thread="sam", warmth=1.5),
    ])
    schedule_npc_reply(state, scenario, "sam")
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert "sam.with_evidence" in pending_ids


# ───────────────────────── end conditions ─────────────────────────


def test_session_ends_when_clock_hits_max(state: InboxState):
    state.game_time_hours = 96.0
    check_end_conditions(state)
    assert state.ended
    assert state.end_reason == "time_exhausted"


def test_session_does_not_end_before_max(state: InboxState):
    state.game_time_hours = 50.0
    check_end_conditions(state)
    assert state.ended is False


# ───────────────────────── helpers ─────────────────────────


def _make_reply(
    thread: str | None,
    warmth: float,
    beat: str | None = None,
    body: str = "(test reply)",
) -> Reply:
    return Reply(
        id=f"reply_{warmth}",
        thread_id=thread,
        beat_responded_to=beat,
        recipient="Sam Park" if thread == "sam" else "other",
        body=body,
        warmth=warmth,
        rationale="(test)",
        sent_at_hours=0.0,
    )
