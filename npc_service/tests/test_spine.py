"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, beat scheduler, calibration application, classifier-driven
quality writes, and multi-turn beat completion.

No LLM, no FastAPI. Runs fast.
"""
import uuid
from pathlib import Path

import pytest

from npc_service.models import (
    AnchoredBeat,
    BeatFired,
    ClassifyReply,
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
    apply_classification,
    evaluate_completes_when,
    evaluate_predicate,
    find_due_beats,
    latest_npc_email_in_thread,
    reply_subject_for_email,
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
    assert state.ended is False
    assert state.thread_messages == {}


def test_from_scenario_auto_pends_every_beat(state: InboxState):
    """`InboxState.from_scenario` auto-pends every beat — anchored AND plot.
    Authored preconditions are evaluated by `find_due_beats` each tick;
    beats whose preconditions don't yet pass stay pending."""
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert "aria.welcome" in pending_ids
    assert "aria.first_calibration" in pending_ids
    assert "aria.midprogram" in pending_ids
    assert "aria.terminal_warm" in pending_ids
    assert "aria.terminal_standard" in pending_ids
    assert "system.midpoint_report" in pending_ids
    assert "system.transition_notice" in pending_ids


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


def test_pending_beat_with_runtime_predicate_waits(
    state: InboxState, scenario: Scenario
):
    """A runtime predicate AND'd with the beat's authored preconditions
    must all pass before the beat fires. The PendingBeat field stays in the
    schema for future extensibility even though no runtime code currently
    sets it."""
    _drain_boot(state, scenario)
    pending = next(pb for pb in state.pending_beats if pb.beat_id == "system.midpoint_report")
    pending.runtime_preconditions = [QualityGte(quality="calibration", value=50.0)]
    state.fired_beat_ids.add("aria.midprogram")  # authored precondition met
    state.qualities["calibration"] = 45.0  # authored passes, runtime fails
    due = find_due_beats(state, scenario)
    assert not any(b.id == "system.midpoint_report" for b in due)
    state.qualities["calibration"] = 55.0
    due = find_due_beats(state, scenario)
    assert any(b.id == "system.midpoint_report" for b in due)


# ───────────────────────── classifier writes ─────────────────────────


def test_apply_classification_writes_mapped_score(
    state: InboxState, scenario: Scenario
):
    classify = ClassifyReply(
        quality="engaged_initial",
        question="(test)",
        allowed_values=["engaged", "deflected", "unclear"],
        map={"engaged": 1.0, "deflected": 0.0, "unclear": 0.5},
    )
    applied = apply_classification(state, scenario, classify, "engaged")
    assert applied is True
    assert state.qualities["engaged_initial"] == 1.0


def test_apply_classification_clamps_to_quality_bounds(
    state: InboxState, scenario: Scenario
):
    """Classifier scores route through `_clamp_quality`, so out-of-bound
    mapped values get clipped."""
    classify = ClassifyReply(
        quality="engaged_initial",
        question="(test)",
        allowed_values=["over", "under"],
        map={"over": 5.0, "under": -1.0},  # quality bounds are 0..1
    )
    apply_classification(state, scenario, classify, "over")
    assert state.qualities["engaged_initial"] == 1.0  # clamped to max
    apply_classification(state, scenario, classify, "under")
    assert state.qualities["engaged_initial"] == 0.0  # clamped to min


def test_apply_classification_off_spec_returns_false(
    state: InboxState, scenario: Scenario
):
    """Off-spec value (LLM returned something not in allowed_values) leaves
    the quality untouched and returns False so the caller can log it."""
    classify = ClassifyReply(
        quality="engaged_initial",
        question="(test)",
        allowed_values=["engaged", "deflected"],
        map={"engaged": 1.0, "deflected": 0.0},
    )
    state.qualities["engaged_initial"] = 0.5  # default
    applied = apply_classification(state, scenario, classify, "garbage")
    assert applied is False
    assert state.qualities["engaged_initial"] == 0.5  # unchanged


# ───────────────────────── multi-turn beat lifecycle ─────────────────────────


def test_evaluate_completes_when_empty_returns_true(
    state: InboxState, scenario: Scenario
):
    """An empty completes_when means fire-once: the beat completes on first
    render. evaluate_completes_when returns True so the caller marks done."""
    welcome = scenario.beat_by_id["aria.welcome"]
    assert welcome.completes_when == []
    assert evaluate_completes_when(state, welcome) is True


def test_evaluate_completes_when_unmet_returns_false(
    state: InboxState, scenario: Scenario
):
    """Multi-turn beat with an unmet gate: not yet complete."""
    first = scenario.beat_by_id["aria.first_calibration"]
    state.qualities["engaged_initial"] = 0.5  # default; gate is ≥0.7
    assert evaluate_completes_when(state, first) is False


def test_evaluate_completes_when_met_returns_true(
    state: InboxState, scenario: Scenario
):
    first = scenario.beat_by_id["aria.first_calibration"]
    state.qualities["engaged_initial"] = 1.0
    assert evaluate_completes_when(state, first) is True


def test_first_calibration_eligible_when_player_has_replied_once(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.append(_make_reply(thread="aria"))
    due = find_due_beats(state, scenario)
    assert any(b.id == "aria.first_calibration" for b in due)


def test_first_calibration_re_fires_until_gate_met(
    state: InboxState, scenario: Scenario
):
    """The multi-turn beat stays in pending and keeps appearing in
    find_due_beats until completes_when passes. We simulate `_fire_beat`'s
    behavior: don't add to fired_beat_ids when completes_when is unmet."""
    _drain_boot(state, scenario)
    state.replies.append(_make_reply(thread="aria"))

    # Tick 1: beat is due. Simulate _fire_beat refusing to mark complete.
    due = find_due_beats(state, scenario)
    assert any(b.id == "aria.first_calibration" for b in due)
    # Gate is engaged_initial ≥ 0.7; default is 0.5 → not met → no mark.
    assert "aria.first_calibration" not in state.fired_beat_ids

    # Tick 2 (same gate): beat still pending, still due.
    state.replies.append(_make_reply(thread="aria"))
    due = find_due_beats(state, scenario)
    assert any(b.id == "aria.first_calibration" for b in due)


def test_first_calibration_completes_when_gate_met(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.append(_make_reply(thread="aria"))
    state.qualities["engaged_initial"] = 1.0  # classifier wrote "engaged"

    first = scenario.beat_by_id["aria.first_calibration"]
    assert evaluate_completes_when(state, first) is True

    # Caller marks complete; subsequent find_due_beats prunes from pending.
    state.fired_beat_ids.add("aria.first_calibration")
    due = find_due_beats(state, scenario)
    assert not any(b.id == "aria.first_calibration" for b in due)


# ───────────────────────── chain progression ─────────────────────────


def test_midprogram_blocked_until_first_calibration_completes(
    state: InboxState, scenario: Scenario
):
    """midprogram has `beat_fired(aria.first_calibration)` as a precondition.
    Even with replies≥3, midprogram doesn't fire until first_calibration is
    completed."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(3))
    due = find_due_beats(state, scenario)
    assert not any(b.id == "aria.midprogram" for b in due)

    state.fired_beat_ids.add("aria.first_calibration")  # complete the loop
    due = find_due_beats(state, scenario)
    assert any(b.id == "aria.midprogram" for b in due)


def test_terminal_warm_blocked_until_midprogram_fired(
    state: InboxState, scenario: Scenario
):
    """terminal_warm requires `beat_fired(aria.midprogram)`. Without first
    completing the chain through midprogram, terminal_warm stays out of the
    due list — even with replies≥5 and partnership_warmth≥0.7."""
    _drain_boot(state, scenario)
    state.qualities["partnership_warmth"] = 1.0
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    due = find_due_beats(state, scenario)
    due_ids = {b.id for b in due}
    assert "aria.terminal_warm" not in due_ids
    assert "aria.midprogram" not in due_ids  # blocked on first_calibration

    # Complete the chain manually; on the next tick midprogram fires AND
    # terminal_warm fires same-tick via the extra_fired mutex (this is the
    # intentional chain-of-beats behavior — once preconditions cascade,
    # downstream beats become eligible immediately).
    state.fired_beat_ids.update({"aria.first_calibration", "aria.midprogram"})
    due = find_due_beats(state, scenario)
    assert any(b.id == "aria.terminal_warm" for b in due)


def test_terminal_mutual_exclusion_warm_wins_when_warmth_high(
    state: InboxState, scenario: Scenario
):
    """High partnership_warmth: terminal_warm comes first in authoring
    order; it's eligible and fires. terminal_standard's
    Not_(BeatFired terminal_warm) keeps it suppressed once warm has fired
    (via tick-local extra_fired)."""
    _drain_boot(state, scenario)
    state.fired_beat_ids.update({"aria.first_calibration", "aria.midprogram"})
    state.qualities["partnership_warmth"] = 1.0
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))

    due = find_due_beats(state, scenario)
    due_ids = [b.id for b in due]
    # terminal_warm fires; terminal_standard's Not_(BeatFired terminal_warm)
    # catches the tick-local fire via `extra_fired` and suppresses standard.
    assert "aria.terminal_warm" in due_ids
    assert "aria.terminal_standard" not in due_ids


def test_terminal_standard_fires_when_warmth_low(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.fired_beat_ids.update({"aria.first_calibration", "aria.midprogram"})
    state.qualities["partnership_warmth"] = 0.0  # below 0.7 — warm fails
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))

    due = find_due_beats(state, scenario)
    due_ids = [b.id for b in due]
    assert "aria.terminal_warm" not in due_ids
    assert "aria.terminal_standard" in due_ids


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
    beat: str | None = None,
    body: str = "(test reply)",
) -> Reply:
    return Reply(
        id=f"reply_{uuid.uuid4().hex[:6]}",
        thread_id=thread,
        beat_responded_to=beat,
        recipient="Aria" if thread == "aria" else "other",
        body=body,
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
