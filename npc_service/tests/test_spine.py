"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, beat scheduler, voice band selection, calibration application,
PlotBeat eligibility, and reply scheduling.

No LLM, no FastAPI. Runs fast.
"""
from pathlib import Path

import pytest

from npc_service.models import (
    AnchoredBeat,
    BeatFired,
    Email,
    InboxState,
    Not_,
    PendingBeat,
    PlotBeat,
    QualityGte,
    Reply,
    Scenario,
    load_scenario,
)
from npc_service.spine import (
    apply_calibration_delta,
    apply_calibration_from_warmth,
    apply_texture_score,
    evaluate_predicate,
    find_due_beats,
    find_eligible_plot_beat,
    latest_npc_email_in_thread,
    reply_subject_for_email,
    schedule_thread_reply,
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
    assert "aria.welcome" in scenario.beat_by_id
    # aria.first_response was deleted in the chat-turn refactor — first reply
    # is now produced by the chat-turn fallback, not an authored PlotBeat.
    assert "aria.first_response" not in scenario.beat_by_id
    # aria has the email_address field for chat-turn rendering.
    assert scenario.npc_by_id["aria"].email_address.startswith("Aria")


def test_initial_state_starts_at_zero(state: InboxState):
    assert state.qualities["calibration"] == 0.0
    assert state.qualities["warmth.aria"] == 0.0
    assert state.qualities["register.aria"] == 0.0
    assert state.ended is False
    assert state.thread_messages == {}


def test_from_scenario_auto_pends_all_anchored_beats(state: InboxState):
    """`InboxState.from_scenario` auto-pends every anchored beat — including
    aria.welcome (no preconditions, so it fires on the first scheduler tick
    at boot). PlotBeats are NOT auto-pended; they're queued by
    schedule_thread_reply on player send."""
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert "aria.welcome" in pending_ids
    assert "system.midpoint_report" in pending_ids
    assert "system.transition_notice" in pending_ids
    # PlotBeats stay out until schedule_thread_reply queues them.
    assert "aria.midprogram" not in pending_ids
    assert "aria.terminal_warm" not in pending_ids


def test_boot_tick_fires_aria_welcome(state: InboxState, scenario: Scenario):
    """An anchored beat with no preconditions fires on the first scheduler
    tick — bootstrap just calls find_due_beats once."""
    due = find_due_beats(state, scenario)
    due_ids = [b.id for b in due]
    assert due_ids[0] == "aria.welcome"
    assert isinstance(due[0], AnchoredBeat)


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


@pytest.mark.parametrize(
    "warmth,base,coef,expected",
    [
        (0.0, 2.0, 1.5, 2.0),     # neutral: just the base
        (1.0, 2.0, 1.5, 3.5),
        (-2.0, 2.0, 1.5, 5.0),    # cold scores absolute, same as warm
        (3.0, 2.0, 1.5, 6.5),
    ],
)
def test_apply_calibration_from_warmth(
    state: InboxState, scenario: Scenario,
    warmth: float, base: float, coef: float, expected: float,
):
    delta = apply_calibration_from_warmth(state, scenario, warmth, base=base, coef=coef)
    assert delta == pytest.approx(expected)
    assert state.qualities["calibration"] == pytest.approx(expected)


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


def test_warmth_and_register_are_clamped(state: InboxState, scenario: Scenario):
    for _ in range(20):
        apply_texture_score(state, scenario, recipient_npc_id="aria", warmth=3.0)
    assert state.qualities["warmth.aria"] == 10.0  # max=10
    assert state.qualities["register.aria"] <= 3.0  # max=3


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
    find_due_beats(state, scenario)  # drain boot tick
    state.qualities["calibration"] = 30.0
    due = find_due_beats(state, scenario)
    assert not any(b.id == "system.midpoint_report" for b in due)
    state.qualities["calibration"] = 40.0
    due = find_due_beats(state, scenario)
    assert any(b.id == "system.midpoint_report" for b in due)


def test_pending_beat_with_runtime_predicate_waits(
    state: InboxState, scenario: Scenario
):
    """A runtime predicate AND'd with the beat's authored preconditions
    must all pass before the beat fires."""
    _drain_boot(state, scenario)
    # Use system.midpoint_report (authored: calibration ≥ 40), AND'd with a
    # runtime calibration ≥ 50 predicate.
    state.pending_beats = [
        PendingBeat(
            beat_id="system.midpoint_report",
            runtime_preconditions=[QualityGte(quality="calibration", value=50.0)],
        )
    ]
    state.qualities["calibration"] = 45.0  # authored passes, runtime fails
    due = find_due_beats(state, scenario)
    assert not any(b.id == "system.midpoint_report" for b in due)
    state.qualities["calibration"] = 55.0
    due = find_due_beats(state, scenario)
    assert any(b.id == "system.midpoint_report" for b in due)


# ───────────────────────── PlotBeat eligibility & mutual exclusion ─────────────────────────


def test_find_eligible_plot_beat_returns_none_when_no_preconditions_pass(
    state: InboxState, scenario: Scenario
):
    """No replies, no calibration: no PlotBeat eligible."""
    _drain_boot(state, scenario)
    assert find_eligible_plot_beat(state, scenario, "aria") is None


def test_find_eligible_plot_beat_returns_midprogram_when_replies_satisfy(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria", warmth=1.0) for _ in range(3))
    plot = find_eligible_plot_beat(state, scenario, "aria")
    assert plot is not None
    assert plot.id == "aria.midprogram"
    assert isinstance(plot, PlotBeat)


def test_find_eligible_plot_beat_skips_already_pending(
    state: InboxState, scenario: Scenario
):
    """Once a PlotBeat is queued, find_eligible_plot_beat doesn't re-pick
    it — prevents the /send schedule loop from queueing the same beat
    twice."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria", warmth=1.0) for _ in range(3))
    state.pending_beats.append(PendingBeat(beat_id="aria.midprogram"))
    assert find_eligible_plot_beat(state, scenario, "aria") is None


def test_find_eligible_plot_beat_skips_already_fired(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria", warmth=1.0) for _ in range(3))
    state.fired_beat_ids.add("aria.midprogram")
    assert find_eligible_plot_beat(state, scenario, "aria") is None


def test_terminal_mutual_exclusion_warm_wins_when_warmth_high(
    state: InboxState, scenario: Scenario
):
    """High warmth: terminal_warm comes first in authoring order; it's
    eligible and fires. terminal_standard's Not_(BeatFired terminal_warm)
    keeps it suppressed once warm has fired."""
    _drain_boot(state, scenario)
    state.fired_beat_ids.add("aria.midprogram")
    state.qualities["calibration"] = 80.0
    state.qualities["warmth.aria"] = 5.0
    state.replies.extend(_make_reply(thread="aria", warmth=2.0) for _ in range(4))

    plot = find_eligible_plot_beat(state, scenario, "aria")
    assert plot is not None and plot.id == "aria.terminal_warm"

    # Simulate firing terminal_warm.
    state.fired_beat_ids.add("aria.terminal_warm")
    plot2 = find_eligible_plot_beat(state, scenario, "aria")
    assert plot2 is None  # terminal_standard's Not_(BeatFired) suppresses it


def test_terminal_standard_eligible_when_warmth_low(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.fired_beat_ids.add("aria.midprogram")
    state.qualities["calibration"] = 80.0
    state.qualities["warmth.aria"] = 0.0  # below 3.0 — terminal_warm fails
    state.replies.extend(_make_reply(thread="aria", warmth=0.0) for _ in range(4))

    plot = find_eligible_plot_beat(state, scenario, "aria")
    assert plot is not None and plot.id == "aria.terminal_standard"


# ───────────────────────── reply scheduling ─────────────────────────


def test_schedule_thread_reply_queues_with_calibration_threshold(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria", warmth=1.0) for _ in range(3))
    state.qualities["calibration"] = 50.0

    midprogram = find_eligible_plot_beat(state, scenario, "aria")
    assert midprogram is not None
    schedule_thread_reply(state, scenario, "aria", midprogram)

    pending = [pb for pb in state.pending_beats if pb.beat_id == "aria.midprogram"]
    assert len(pending) == 1
    # Aria's reply_delay_calibration = 3.5; from cal=50 → threshold 53.5.
    assert len(pending[0].runtime_preconditions) == 1
    pred = pending[0].runtime_preconditions[0]
    assert isinstance(pred, QualityGte)
    assert pred.quality == "calibration"
    assert pred.value == pytest.approx(53.5)


def test_scheduled_plot_beat_does_not_fire_same_send(
    state: InboxState, scenario: Scenario
):
    """Threshold sits above current calibration → queued PlotBeat doesn't
    fire on its own scheduling tick. This is the deliberate-pacing pattern
    for plot moments."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria", warmth=1.0) for _ in range(3))
    state.qualities["calibration"] = 50.0
    midprogram = find_eligible_plot_beat(state, scenario, "aria")
    schedule_thread_reply(state, scenario, "aria", midprogram)
    # Same tick: threshold is 53.5, calibration is 50 → not due.
    due = find_due_beats(state, scenario)
    assert not any(b.id == "aria.midprogram" for b in due)
    # Crossing the threshold: now due.
    state.qualities["calibration"] = 54.0
    due = find_due_beats(state, scenario)
    assert any(b.id == "aria.midprogram" for b in due)


# ───────────────────────── reply-subject helper ─────────────────────────


def test_reply_subject_prefixes_re():
    state = InboxState()
    state.emails.append(_make_email(id="e1", subject="Calibration Update"))
    assert reply_subject_for_email(state, "e1") == "Re: Calibration Update"


def test_reply_subject_idempotent_on_existing_re():
    state = InboxState()
    state.emails.append(_make_email(id="e1", subject="Re: Calibration Update"))
    assert reply_subject_for_email(state, "e1") == "Re: Calibration Update"


def test_reply_subject_unknown_email_id():
    state = InboxState()
    assert reply_subject_for_email(state, "nope") == "(no subject)"
    assert reply_subject_for_email(state, None) == "(no subject)"


# ───────────────────────── latest_npc_email lookup ─────────────────────────


def test_latest_npc_email_picks_most_recent_by_sender():
    state = InboxState()
    state.emails.append(_make_email(id="e1", thread_id="aria", sender="aria@x", recipient="player@x"))
    state.emails.append(_make_email(id="e2", thread_id="aria", sender="you", recipient="aria@x"))
    state.emails.append(_make_email(id="e3", thread_id="aria", sender="aria@x", recipient="player@x"))
    found = latest_npc_email_in_thread(state, "aria", "aria@x")
    assert found is not None and found.id == "e3"


def test_latest_npc_email_returns_none_when_none_exist():
    state = InboxState()
    assert latest_npc_email_in_thread(state, "aria", "aria@x") is None


# ───────────────────────── helpers ─────────────────────────


def _drain_boot(state: InboxState, scenario: Scenario) -> None:
    for beat in find_due_beats(state, scenario):
        state.fired_beat_ids.add(beat.id)


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


def _make_email(
    *,
    id: str,
    subject: str = "(no subject)",
    thread_id: str | None = None,
    sender: str = "aria@x",
    recipient: str = "player@x",
) -> Email:
    return Email(
        id=id,
        beat_id=None,
        thread_id=thread_id,
        folder="Inbox",
        sender=sender,
        recipient=recipient,
        subject=subject,
        body="",
        sent_at_label="just now",
    )
