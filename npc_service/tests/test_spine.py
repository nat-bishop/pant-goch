"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, beat scheduler, classifier-driven quality writes (set + accumulate),
and reply/email lookup helpers.

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
    PlotBeat,
    Reply,
    Scenario,
    load_scenario,
)
from npc_service.spine import (
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
    assert {t.id for t in scenario.threads} == {"aria"}
    # 4 beats: welcome + two endings + the ongoing loop.
    assert {b.id for b in scenario.beats} == {
        "aria.welcome",
        "aria.complete_warm",
        "aria.complete_standard",
        "aria.loop",
    }


def test_loop_is_multi_turn_with_classifier(scenario: Scenario):
    loop = scenario.beat_by_id["aria.loop"]
    assert isinstance(loop, PlotBeat)
    assert loop.multi_turn is True
    assert loop.ends_session is False
    assert loop.classify_reply is not None
    assert loop.classify_reply.accumulate is True
    assert loop.classify_reply.quality == "engagement"


def test_terminals_carry_ends_session_flag(scenario: Scenario):
    warm = scenario.beat_by_id["aria.complete_warm"]
    standard = scenario.beat_by_id["aria.complete_standard"]
    assert isinstance(warm, PlotBeat) and warm.ends_session is True
    assert isinstance(standard, PlotBeat) and standard.ends_session is True


def test_initial_state_starts_clean(state: InboxState, scenario: Scenario):
    assert state.qualities["engagement"] == 0.0
    assert state.ended is False
    assert state.thread_messages == {}


def test_from_scenario_auto_pends_every_beat(state: InboxState):
    """`InboxState.from_scenario` auto-pends every beat. Authored
    preconditions are evaluated by `find_due_beats` each tick; beats whose
    preconditions don't yet pass stay pending."""
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert pending_ids == {
        "aria.welcome",
        "aria.complete_warm",
        "aria.complete_standard",
        "aria.loop",
    }


def test_boot_tick_fires_aria_welcome(state: InboxState, scenario: Scenario):
    """An anchored beat with no preconditions fires on the first scheduler
    tick. The loop and terminals stay pending — they require player replies."""
    due = find_due_beats(state, scenario)
    due_ids = [b.id for b in due]
    assert due_ids == ["aria.welcome"]
    assert isinstance(due[0], AnchoredBeat)


# ───────────────────────── predicate evaluation ─────────────────────────


def test_beat_fired_predicate_true_after_fire(state: InboxState):
    state.fired_beat_ids.add("aria.complete_warm")
    assert evaluate_predicate(BeatFired(beat="aria.complete_warm"), state) is True


def test_beat_fired_predicate_false_when_unfired(state: InboxState):
    assert evaluate_predicate(BeatFired(beat="aria.complete_warm"), state) is False


def test_not_predicate_inverts(state: InboxState):
    p = Not_(of=BeatFired(beat="aria.complete_warm"))
    assert evaluate_predicate(p, state) is True
    state.fired_beat_ids.add("aria.complete_warm")
    assert evaluate_predicate(p, state) is False


# ───────────────────────── classifier writes ─────────────────────────


def test_apply_classification_set_mode_overwrites(
    state: InboxState, scenario: Scenario
):
    """Default `accumulate=False`: the mapped score replaces the quality.
    Repeated applies do not stack — subsequent calls overwrite."""
    classify = ClassifyReply(
        quality="engagement",
        question="(test)",
        allowed_values=["engaged", "deflected", "unclear"],
        map={"engaged": 1.0, "deflected": -0.5, "unclear": 0.5},
    )
    apply_classification(state, scenario, classify, "engaged")
    assert state.qualities["engagement"] == 1.0
    apply_classification(state, scenario, classify, "engaged")
    assert state.qualities["engagement"] == 1.0  # no stacking


def test_apply_classification_accumulate_adds(
    state: InboxState, scenario: Scenario
):
    """With `accumulate=True`, each apply adds the mapped score to the
    current value. Negative deltas (deflected → -0.5) reduce it; the
    quality bound clamps at 0 so a long string of deflections won't go
    negative."""
    classify = ClassifyReply(
        quality="engagement",
        question="(test)",
        allowed_values=["engaged", "deflected", "unclear"],
        map={"engaged": 1.0, "deflected": -0.5, "unclear": 0.5},
        accumulate=True,
    )
    apply_classification(state, scenario, classify, "engaged")
    apply_classification(state, scenario, classify, "engaged")
    assert state.qualities["engagement"] == 2.0
    apply_classification(state, scenario, classify, "deflected")
    assert state.qualities["engagement"] == 1.5
    # Floor clamp at 0 (engagement.min = 0).
    for _ in range(10):
        apply_classification(state, scenario, classify, "deflected")
    assert state.qualities["engagement"] == 0.0


def test_apply_classification_clamps_to_quality_bounds(
    state: InboxState, scenario: Scenario
):
    """Out-of-bound mapped values get clipped via _clamp_quality. Engagement
    is bounded 0..10."""
    classify = ClassifyReply(
        quality="engagement",
        question="(test)",
        allowed_values=["over", "under"],
        map={"over": 50.0, "under": -10.0},
    )
    apply_classification(state, scenario, classify, "over")
    assert state.qualities["engagement"] == 10.0  # clamped to max
    apply_classification(state, scenario, classify, "under")
    assert state.qualities["engagement"] == 0.0  # clamped to min (set mode)


def test_apply_classification_off_spec_returns_false(
    state: InboxState, scenario: Scenario
):
    """LLM returned a value not in allowed_values: leave quality untouched
    and return False so the caller can log it."""
    classify = ClassifyReply(
        quality="engagement",
        question="(test)",
        allowed_values=["engaged", "deflected"],
        map={"engaged": 1.0, "deflected": 0.0},
    )
    state.qualities["engagement"] = 3.0
    applied = apply_classification(state, scenario, classify, "garbage")
    assert applied is False
    assert state.qualities["engagement"] == 3.0


# ───────────────────────── multi-turn / completes_when ─────────────────────────


def test_evaluate_completes_when_empty_returns_true(
    state: InboxState, scenario: Scenario
):
    """An empty completes_when means the beat completes on first render."""
    welcome = scenario.beat_by_id["aria.welcome"]
    assert welcome.completes_when == []
    assert evaluate_completes_when(state, welcome) is True


# ───────────────────────── loop scheduling ─────────────────────────


def test_loop_eligible_after_first_player_reply(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.append(_make_reply(thread="aria"))
    due = find_due_beats(state, scenario)
    assert any(b.id == "aria.loop" for b in due)


def test_loop_blocked_when_terminal_already_fired(
    state: InboxState, scenario: Scenario
):
    """Once a terminal has fired, the loop's mutex preconditions stop
    holding — loop is no longer due regardless of reply count."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(3))
    state.fired_beat_ids.add("aria.complete_warm")
    due = find_due_beats(state, scenario)
    assert not any(b.id == "aria.loop" for b in due)


# ───────────────────────── terminals ─────────────────────────


def test_complete_warm_fires_at_engagement_threshold(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 4.0  # threshold met
    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.complete_warm" in due_ids


def test_complete_warm_blocked_below_threshold(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 3.5  # below threshold
    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.complete_warm" not in due_ids


def test_complete_warm_blocked_below_reply_count(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(4))
    state.qualities["engagement"] = 8.0  # plenty
    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.complete_warm" not in due_ids


def test_complete_mutex_warm_wins_when_engagement_high(
    state: InboxState, scenario: Scenario
):
    """Both endings become eligible at replies≥5; complete_warm comes first
    in authoring order. When it fires, complete_standard's
    Not_(BeatFired complete_warm) catches the same-tick fire via
    `extra_fired` and stays out of the due list."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 5.0

    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.complete_warm" in due_ids
    assert "aria.complete_standard" not in due_ids


def test_complete_standard_fires_when_engagement_low(
    state: InboxState, scenario: Scenario
):
    """5 replies, engagement below threshold → complete_warm blocked,
    complete_standard fires."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 1.0

    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.complete_warm" not in due_ids
    assert "aria.complete_standard" in due_ids


def test_endings_block_loop_same_tick(
    state: InboxState, scenario: Scenario
):
    """The ending beats are authored before the loop, so when both become
    eligible the same tick, find_due_beats evaluates the ending first; it
    enters `extra_fired`; the loop's Not_(BeatFired complete_*) preconditions
    catch it and the loop is not in the due list this tick."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 5.0

    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.complete_warm" in due_ids
    assert "aria.loop" not in due_ids


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
