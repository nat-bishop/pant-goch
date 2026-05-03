"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, effect application, beat scheduler, voice band selection,
calibration application, and reply scheduling.

No LLM, no FastAPI. Runs fast.
"""
from pathlib import Path

import pytest

from npc_service.models import (
    AnchoredBeat,
    BeatFired,
    InboxState,
    Not_,
    PendingBeat,
    QualityGte,
    Reply,
    Scenario,
    SetQuality,
    load_scenario,
)
from npc_service.spine import (
    apply_calibration_delta,
    apply_effect,
    apply_texture_score,
    evaluate_predicate,
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
    assert scenario.title == "Continuity Program"
    assert {n.id for n in scenario.npcs} == {"aria"}
    assert {t.id for t in scenario.threads} >= {"aria", "system"}
    boot_count = sum(1 for b in scenario.beats if getattr(b, "in_boot_inbox", False))
    assert boot_count == 1, f"expected 1 boot-inbox beat, got {boot_count}"


def test_initial_state_starts_at_zero(state: InboxState):
    assert state.qualities["calibration"] == 0.0
    assert state.qualities["warmth.aria"] == 0.0
    assert state.qualities["register.aria"] == 0.0
    assert state.ended is False


def test_boot_inbox_returns_one_beat(state: InboxState, scenario: Scenario):
    boot = initialize_boot_inbox(state, scenario)
    assert len(boot) == 1
    assert {b.beat.id for b in boot} == {"aria.welcome"}
    assert isinstance(boot[0].beat, AnchoredBeat)


def test_from_scenario_auto_pends_non_boot_anchored_beats(state: InboxState):
    """`InboxState.from_scenario` auto-pends every anchored beat that is not
    in the boot inbox. Their authored preconditions are then evaluated by
    find_due_beats as runtime predicates."""
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert "system.midpoint_report" in pending_ids
    assert "system.transition_notice" in pending_ids
    # The boot beat is fired by initialize_boot_inbox, not auto-pended.
    assert "aria.welcome" not in pending_ids
    # Rendered beats are scheduled by schedule_npc_reply on player send,
    # not auto-pended.
    assert "aria.first_response" not in pending_ids


# ───────────────────────── effect application ─────────────────────────


def test_set_quality_add_op_clamps_to_max(state: InboxState, scenario: Scenario):
    apply_effect(SetQuality(quality="register.aria", op="set", value=2.5), state, scenario)
    apply_effect(SetQuality(quality="register.aria", op="add", value=10.0), state, scenario)
    assert state.qualities["register.aria"] == 3.0  # max=3


def test_drift_halfway_op(state: InboxState, scenario: Scenario):
    state.qualities["register.aria"] = 2.0
    apply_effect(SetQuality(quality="register.aria", op="drift_halfway", value=0.0), state, scenario)
    assert state.qualities["register.aria"] == 1.0


# ───────────────────────── calibration ─────────────────────────


def test_calibration_delta_adds_and_clamps(state: InboxState, scenario: Scenario):
    apply_calibration_delta(state, scenario, 30.0)
    assert state.qualities["calibration"] == 30.0
    apply_calibration_delta(state, scenario, 80.0)
    assert state.qualities["calibration"] == 100.0  # clamped to max


def test_calibration_at_cap_ends_session(state: InboxState, scenario: Scenario):
    apply_calibration_delta(state, scenario, 100.0)
    assert state.ended is True
    assert state.end_reason == "calibration_complete"


def test_calibration_below_cap_does_not_end(state: InboxState, scenario: Scenario):
    apply_calibration_delta(state, scenario, 50.0)
    assert state.ended is False


# ───────────────────────── texture / register loop ─────────────────────────


def test_warm_reply_increases_warmth_and_drifts_register(state: InboxState, scenario: Scenario):
    apply_texture_score(state, scenario, recipient_npc_id="aria", warmth=2.0)
    assert state.qualities["warmth.aria"] == 2.0
    # Register drifts halfway from 0 toward 2 → 1.0.
    assert state.qualities["register.aria"] == 1.0


def test_curt_reply_pulls_register_negative(state: InboxState, scenario: Scenario):
    apply_texture_score(state, scenario, recipient_npc_id="aria", warmth=-2.0)
    assert state.qualities["register.aria"] == -1.0


def test_register_drift_visible_within_one_round(state: InboxState, scenario: Scenario):
    """The texture loop must close per-round. Two replies should produce
    visibly different register values."""
    apply_texture_score(state, scenario, recipient_npc_id="aria", warmth=3.0)
    after_warm = state.qualities["register.aria"]
    apply_texture_score(state, scenario, recipient_npc_id="aria", warmth=-3.0)
    after_cold = state.qualities["register.aria"]
    assert after_warm > after_cold
    assert after_cold < 0


def test_unknown_recipient_does_not_mutate_qualities(state: InboxState, scenario: Scenario):
    apply_texture_score(state, scenario, recipient_npc_id=None, warmth=2.0)
    assert state.qualities["warmth.aria"] == 0.0
    assert state.qualities["register.aria"] == 0.0


# ───────────────────────── voice band selection ─────────────────────────


def test_voice_band_standard_at_zero(state: InboxState, scenario: Scenario):
    aria = scenario.npc_by_id["aria"]
    band = select_voice_band(aria, state)
    assert band is not None and band.name == "standard"


def test_voice_band_affirming_when_register_high(state: InboxState, scenario: Scenario):
    state.qualities["register.aria"] = 2.0
    aria = scenario.npc_by_id["aria"]
    band = select_voice_band(aria, state)
    assert band is not None and band.name == "affirming"


def test_voice_band_corrective_when_register_low(state: InboxState, scenario: Scenario):
    state.qualities["register.aria"] = -2.0
    aria = scenario.npc_by_id["aria"]
    band = select_voice_band(aria, state)
    assert band is not None and band.name == "corrective"


# ───────────────────────── predicate evaluation ─────────────────────────


def test_beat_fired_predicate_true_after_fire(state: InboxState):
    state.fired_beat_ids.add("aria.terminal_warm")
    assert evaluate_predicate(BeatFired(beat="aria.terminal_warm"), state) is True


def test_beat_fired_predicate_false_when_unfired(state: InboxState):
    assert evaluate_predicate(BeatFired(beat="aria.terminal_warm"), state) is False


def test_not_predicate_inverts(state: InboxState):
    p = Not_(of=BeatFired(beat="aria.terminal_warm"))
    assert evaluate_predicate(p, state) is True
    state.fired_beat_ids.add("aria.terminal_warm")
    assert evaluate_predicate(p, state) is False


# ───────────────────────── beat scheduler ─────────────────────────


def test_calibration_gated_anchored_beat_fires_at_threshold(
    state: InboxState, scenario: Scenario
):
    """system.midpoint_report fires when calibration crosses 40."""
    state.qualities["calibration"] = 30.0
    due = find_due_beats(state, scenario)
    assert not any(bf.beat.id == "system.midpoint_report" for bf in due)
    state.qualities["calibration"] = 40.0
    due = find_due_beats(state, scenario)
    assert any(bf.beat.id == "system.midpoint_report" for bf in due)


def test_pending_beat_with_runtime_predicate_waits(
    state: InboxState, scenario: Scenario
):
    """A runtime predicate AND'd with the beat's authored preconditions
    must all pass before the beat fires."""
    state.pending_beats = [
        PendingBeat(
            beat_id="aria.first_response",
            runtime_preconditions=[QualityGte(quality="calibration", value=10.0)],
        )
    ]
    state.replies.append(_make_reply(thread="aria", warmth=0.0))  # satisfies authored
    state.qualities["calibration"] = 5.0
    due = find_due_beats(state, scenario)
    assert not any(bf.beat.id == "aria.first_response" for bf in due)
    state.qualities["calibration"] = 12.0
    due = find_due_beats(state, scenario)
    assert any(bf.beat.id == "aria.first_response" for bf in due)


def test_terminal_warm_and_standard_are_mutually_exclusive(
    state: InboxState, scenario: Scenario
):
    """Both terminals share calibration + reply-count gates. terminal_standard
    has a Not(BeatFired terminal_warm) precondition so warm wins when warmth
    is high; standard fires alone when warmth is low."""
    # High warmth path: warm fires, standard suppressed.
    state.qualities["calibration"] = 80.0
    state.qualities["warmth.aria"] = 5.0
    state.replies.extend(_make_reply(thread="aria", warmth=2.0) for _ in range(4))
    # Force scheduling of both candidates so find_due_beats sees them pending.
    state.pending_beats.extend([
        PendingBeat(beat_id="aria.terminal_warm"),
        PendingBeat(beat_id="aria.terminal_standard"),
    ])
    due = find_due_beats(state, scenario)
    due_ids = {bf.beat.id for bf in due}
    assert "aria.terminal_warm" in due_ids
    assert "aria.terminal_standard" not in due_ids


def test_terminal_standard_fires_when_warm_did_not(
    state: InboxState, scenario: Scenario
):
    state.qualities["calibration"] = 80.0
    state.qualities["warmth.aria"] = 0.0  # below 3.0 — warm fails
    state.replies.extend(_make_reply(thread="aria", warmth=0.0) for _ in range(4))
    state.pending_beats.extend([
        PendingBeat(beat_id="aria.terminal_warm"),
        PendingBeat(beat_id="aria.terminal_standard"),
    ])
    due = find_due_beats(state, scenario)
    due_ids = {bf.beat.id for bf in due}
    assert "aria.terminal_warm" not in due_ids
    assert "aria.terminal_standard" in due_ids


# ───────────────────────── reply scheduling ─────────────────────────


def test_schedule_npc_reply_picks_first_eligible_beat(
    state: InboxState, scenario: Scenario
):
    state.fired_beat_ids.add("aria.welcome")
    state.replies.append(_make_reply(thread="aria", warmth=0.0))  # for ≥1 gate
    state.qualities["calibration"] = 5.0
    scheduled = schedule_npc_reply(state, scenario, "aria")
    assert scheduled is True
    pending = [pb for pb in state.pending_beats if pb.beat_id == "aria.first_response"]
    assert len(pending) == 1
    # Aria's reply delay is [2.0, 5.0]; midpoint = 3.5; from cal=5 → threshold 8.5.
    assert len(pending[0].runtime_preconditions) == 1
    pred = pending[0].runtime_preconditions[0]
    assert isinstance(pred, QualityGte)
    assert pred.quality == "calibration"
    assert pred.value == pytest.approx(8.5)


def test_schedule_npc_reply_skips_when_pending(
    state: InboxState, scenario: Scenario
):
    """Subsequent sends to an NPC with a pending reply don't reschedule."""
    state.fired_beat_ids.add("aria.welcome")
    state.replies.append(_make_reply(thread="aria", warmth=0.0))
    schedule_npc_reply(state, scenario, "aria")
    initial_pending = list(state.pending_beats)
    state.qualities["calibration"] = 7.0
    scheduled_again = schedule_npc_reply(state, scenario, "aria")
    assert scheduled_again is False
    assert state.pending_beats == initial_pending


def test_schedule_npc_reply_respects_preconditions(
    state: InboxState, scenario: Scenario
):
    """aria.midprogram requires player_replies_to_thread_gte aria, 3.
    Without enough player replies it is skipped; aria.first_response wins."""
    state.fired_beat_ids.add("aria.welcome")
    state.replies.append(_make_reply(thread="aria", warmth=0.0))  # only 1 reply
    schedule_npc_reply(state, scenario, "aria")
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert "aria.first_response" in pending_ids
    assert "aria.midprogram" not in pending_ids


def test_player_replies_to_thread_predicate_unlocks_midprogram(
    state: InboxState, scenario: Scenario
):
    """Once three player replies sit in the aria thread, aria.midprogram
    becomes the next eligible beat."""
    state.fired_beat_ids.update({"aria.welcome", "aria.first_response"})
    state.replies.extend([
        _make_reply(thread="aria", warmth=1.0),
        _make_reply(thread="aria", warmth=1.5),
        _make_reply(thread="aria", warmth=0.5),
    ])
    schedule_npc_reply(state, scenario, "aria")
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert "aria.midprogram" in pending_ids


# ───────────────────────── helpers ─────────────────────────


def _make_reply(
    thread: str | None,
    warmth: float,
    beat: str | None = None,
    body: str = "(test reply)",
) -> Reply:
    return Reply(
        id=f"reply_{warmth}_{len(body)}",
        thread_id=thread,
        beat_responded_to=beat,
        recipient="Aria" if thread == "aria" else "other",
        body=body,
        warmth=warmth,
        rationale="(test)",
    )
