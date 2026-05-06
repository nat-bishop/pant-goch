"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, beat scheduler, classifier-driven quality writes (set + accumulate),
recent-window predicates, BeatSpec.writes (commitments), and reply/email
lookup helpers.

No LLM, no FastAPI. Runs fast.
"""
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from npc_service.models import (
    AxisResult,
    Beat,
    BeatFired,
    BeatSpec,
    ClassifyReply,
    Email,
    InboxState,
    Not_,
    QualityDef,
    RecentClassifierValue,
    Reply,
    Scenario,
    load_scenario,
)
from npc_service.spine import (
    apply_classification,
    apply_writes,
    evaluate_predicate,
    find_due_beats,
    reply_subject_for_email,
)


def _build_minimal_scenario() -> Scenario:
    """A tiny synthesized scenario that exercises every engine path the spine
    tests need: one accumulating quality (`engagement`), one anchored welcome,
    a multi-turn loop, and two mutex-chained terminals (warm + standard).

    Decoupled from the production scenario.yaml so spine tests stay focused
    on engine mechanics rather than on whichever scenario happens to ship."""
    return Scenario.model_validate({
        "title": "Test Scenario",
        "premise": "Synthetic fixture for spine tests.",
        "player_email": "p@x",
        "qualities": [
            {"id": "engagement", "type": "float", "min": 0.0, "max": 10.0, "default": 0.0},
        ],
        "npcs": [
            {"id": "aria", "display_name": "Aria", "email_address": "aria@x"},
        ],
        "threads": [{"id": "aria", "npc": "aria"}],
        "beats": [
            {
                "id": "aria.welcome",
                "thread": "aria",
                "npc": "aria",
                "subject": "Welcome",
                "spec": {"pre_authored_prose": "Welcome."},
            },
            {
                "id": "aria.complete_warm",
                "thread": "aria",
                "npc": "aria",
                "ends_session": True,
                "subject": "Done — warm",
                "preconditions": [
                    {"kind": "player_replies_to_thread_gte", "thread": "aria", "value": 5},
                    {"kind": "quality_gte", "quality": "engagement", "value": 4.0},
                ],
                "spec": {"intent": "warm finale"},
            },
            {
                "id": "aria.complete_standard",
                "thread": "aria",
                "npc": "aria",
                "ends_session": True,
                "subject": "Done — standard",
                "preconditions": [
                    {"kind": "player_replies_to_thread_gte", "thread": "aria", "value": 5},
                    {"kind": "not", "of": {"kind": "beat_fired", "beat": "aria.complete_warm"}},
                ],
                "spec": {"intent": "standard finale"},
            },
            {
                "id": "aria.loop",
                "thread": "aria",
                "npc": "aria",
                "multi_turn": True,
                "subject": "Continuing",
                "preconditions": [
                    {"kind": "player_replies_to_thread_gte", "thread": "aria", "value": 1},
                    {"kind": "not", "of": {"kind": "beat_fired", "beat": "aria.complete_warm"}},
                    {"kind": "not", "of": {"kind": "beat_fired", "beat": "aria.complete_standard"}},
                ],
                "spec": {"intent": "ask another question"},
                "classify_reply_axes": [{
                    "quality": "engagement",
                    "question": "Did the player engage?",
                    "allowed_values": ["engaged", "deflected"],
                    "map": {"engaged": 1.0, "deflected": -0.5},
                    "accumulate": True,
                }],
            },
        ],
    })


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return _build_minimal_scenario()


@pytest.fixture
def state(scenario: Scenario) -> InboxState:
    return InboxState.from_scenario(scenario)


# ───────────────────────── scenario validation ─────────────────────────


def test_scenario_loads_and_is_internally_consistent(scenario: Scenario):
    assert scenario.title == "Test Scenario"
    assert {n.id for n in scenario.npcs} == {"aria"}
    assert {t.id for t in scenario.threads} == {"aria"}
    assert {b.id for b in scenario.beats} == {
        "aria.welcome",
        "aria.complete_warm",
        "aria.complete_standard",
        "aria.loop",
    }


def test_loop_is_multi_turn_with_classifier(scenario: Scenario):
    loop = scenario.beat_by_id["aria.loop"]
    assert loop.multi_turn is True
    assert loop.ends_session is False
    assert len(loop.classify_reply_axes) == 1
    axis = loop.classify_reply_axes[0]
    assert axis.accumulate is True
    assert axis.quality == "engagement"


def test_terminals_carry_ends_session_flag(scenario: Scenario):
    warm = scenario.beat_by_id["aria.complete_warm"]
    standard = scenario.beat_by_id["aria.complete_standard"]
    assert warm.ends_session is True
    assert standard.ends_session is True


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
    """A pre-authored beat with no preconditions fires on the first scheduler
    tick. The loop and terminals stay pending — they require player replies."""
    due = find_due_beats(state, scenario)
    due_ids = [b.id for b in due]
    assert due_ids == ["aria.welcome"]
    assert due[0].spec.pre_authored_prose is not None


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
    assert applied is None
    assert state.qualities["engagement"] == 3.0


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


def test_both_terminals_eligible_when_paths_overlap(
    state: InboxState, scenario: Scenario
):
    """With extra_fired removed, find_due_beats returns every beat whose
    preconditions hold — including both terminals when their gates overlap.
    The Director (in `_tick`, not exercised here) picks one. The same-tick
    mutex via Not_(BeatFired) in the existing scenario stops short of
    suppressing the sibling — both come back in the eligible list, and
    selection happens at the Director layer."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 5.0

    due_ids = [b.id for b in find_due_beats(state, scenario)]
    # Both terminals are individually eligible (their authored preconditions
    # all pass — the Not_(BeatFired complete_warm) on complete_standard
    # only catches once complete_warm has actually been fired).
    assert "aria.complete_warm" in due_ids
    assert "aria.complete_standard" in due_ids
    # The loop is also eligible (Not_(BeatFired) on terminals isn't yet
    # satisfied). Director arbitrates.
    assert "aria.loop" in due_ids


def test_complete_standard_fires_when_engagement_low(
    state: InboxState, scenario: Scenario
):
    """5 replies, engagement below threshold → complete_warm blocked
    (its quality_gte engagement >= 4.0 fails), complete_standard remains
    eligible."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 1.0

    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.complete_warm" not in due_ids
    assert "aria.complete_standard" in due_ids


def test_loop_blocked_after_terminal_added_to_fired_ids(
    state: InboxState, scenario: Scenario
):
    """Once a terminal id lands in `state.fired_beat_ids`, the loop's
    Not_(BeatFired complete_warm) precondition catches it. This is the
    PERSISTENT mutex (post-fire), not the same-tick one — same-tick mutex
    is now Director-handled."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 5.0
    # Simulate a Director pick + fire of complete_warm.
    state.fired_beat_ids.add("aria.complete_warm")

    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert "aria.loop" not in due_ids
    # complete_standard's Not_(BeatFired complete_warm) also catches it now.
    assert "aria.complete_standard" not in due_ids


# ───────────────────────── apply_classification → Reply.classifier_results ─────────────────────────


def test_apply_classification_returns_axis_result_on_success(
    state: InboxState, scenario: Scenario
):
    """Returns an AxisResult that the caller appends onto a Reply so
    RecentClassifierValue predicates can walk state.replies directly."""
    classify = ClassifyReply(
        quality="engagement",
        question="(test)",
        allowed_values=["engaged", "deflected"],
        map={"engaged": 1.0, "deflected": 0.0},
        accumulate=True,
    )
    result = apply_classification(state, scenario, classify, "engaged", "rationale x")
    assert result is not None
    assert result.quality == "engagement"
    assert result.value == "engaged"
    assert result.rationale == "rationale x"


def test_apply_classification_off_spec_returns_none(
    state: InboxState, scenario: Scenario
):
    """An off-spec value returns None — the caller skips the append, so
    the recent-window list never sees values that didn't write the
    quality."""
    classify = ClassifyReply(
        quality="engagement",
        question="(test)",
        allowed_values=["engaged", "deflected"],
        map={"engaged": 1.0, "deflected": 0.0},
    )
    result = apply_classification(state, scenario, classify, "garbage")
    assert result is None


# ───────────────────────── RecentClassifierValue predicate ─────────────────────────


def test_recent_classifier_value_empty_replies_returns_false(
    state: InboxState,
):
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3
    )
    assert evaluate_predicate(pred, state) is False


def test_recent_classifier_value_any_match(state: InboxState):
    """match='any' (default) → ≥1 hit is enough."""
    state.replies = [
        _reply_with_axis("aria", "engagement", "engaged"),
        _reply_with_axis("aria", "engagement", "deflected"),
        _reply_with_axis("aria", "engagement", "deflected"),
    ]
    pred_hit = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3, match="any"
    )
    pred_miss = RecentClassifierValue(
        quality="engagement", values=["unclear"], window=3, match="any"
    )
    assert evaluate_predicate(pred_hit, state) is True
    assert evaluate_predicate(pred_miss, state) is False


def test_recent_classifier_value_all_requires_full_window(state: InboxState):
    """match='all' requires every slot in the window to match — under-filled
    cannot satisfy."""
    state.replies = [
        _reply_with_axis("aria", "engagement", "engaged"),
        _reply_with_axis("aria", "engagement", "engaged"),
    ]
    # Window 3, only 2 replies recorded → False even though both hit.
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3, match="all"
    )
    assert evaluate_predicate(pred, state) is False
    state.replies.append(_reply_with_axis("aria", "engagement", "engaged"))
    assert evaluate_predicate(pred, state) is True
    state.replies.append(_reply_with_axis("aria", "engagement", "deflected"))
    # Now last 3 are engaged, engaged, deflected → False.
    assert evaluate_predicate(pred, state) is False


def test_recent_classifier_value_int_match(state: InboxState):
    """match=N → ≥N matches required."""
    state.replies = [
        _reply_with_axis("aria", "engagement", "engaged"),
        _reply_with_axis("aria", "engagement", "deflected"),
        _reply_with_axis("aria", "engagement", "engaged"),
    ]
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3, match=2
    )
    assert evaluate_predicate(pred, state) is True
    pred_3 = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3, match=3
    )
    assert evaluate_predicate(pred_3, state) is False


def test_recent_classifier_value_any_of_values(state: InboxState):
    """`values: list[str]` is any-of: a window of [clinical, engaged,
    horrified] with values=[clinical, horrified] match=2 → True."""
    state.replies = [
        _reply_with_axis("aria", "affect", "clinical"),
        _reply_with_axis("aria", "affect", "engaged"),
        _reply_with_axis("aria", "affect", "horrified"),
    ]
    pred = RecentClassifierValue(
        quality="affect", values=["clinical", "horrified"], window=3, match=2
    )
    assert evaluate_predicate(pred, state) is True


def test_recent_classifier_value_skips_replies_without_axis(state: InboxState):
    """Replies whose classifier_results don't include the target quality
    are skipped — they don't fill a window slot. Window walks "the last N
    replies that were classified on this quality," not "the last N replies
    overall."""
    state.replies = [
        _reply_with_axis("aria", "engagement", "engaged"),
        _make_reply(thread="aria"),  # no classifier_results
        _reply_with_axis("aria", "engagement", "engaged"),
    ]
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=2, match="all"
    )
    # Two replies with engagement axis, both engaged → True.
    assert evaluate_predicate(pred, state) is True


def test_recent_classifier_value_under_not_(state: InboxState):
    """RecentClassifierValue evaluates correctly nested under Not_."""
    state.replies = [
        _reply_with_axis("aria", "engagement", "deflected"),
    ]
    pred = Not_(of=RecentClassifierValue(
        quality="engagement", values=["engaged"], window=1, match="any",
    ))
    assert evaluate_predicate(pred, state) is True


def test_recent_classifier_value_match_int_validation():
    """match int must be 1..window."""
    with pytest.raises(ValidationError):
        RecentClassifierValue(
            quality="engagement", values=["x"], window=3, match=5
        )
    with pytest.raises(ValidationError):
        RecentClassifierValue(
            quality="engagement", values=["x"], window=3, match=0
        )


# ───────────────────────── apply_writes (commitments ledger) ─────────────────────────


def test_apply_writes_set_semantics(state: InboxState, scenario: Scenario):
    """SET semantics: writes overwrite the current value (no accumulate)."""
    apply_writes(state, scenario, {"engagement": 5.0})
    assert state.qualities["engagement"] == 5.0
    apply_writes(state, scenario, {"engagement": 2.0})
    assert state.qualities["engagement"] == 2.0


def test_apply_writes_clamps_to_quality_bounds(
    state: InboxState, scenario: Scenario
):
    """Clamping is shared with apply_classification via _clamp_quality."""
    apply_writes(state, scenario, {"engagement": 99.0})
    assert state.qualities["engagement"] == 10.0  # max
    apply_writes(state, scenario, {"engagement": -3.0})
    assert state.qualities["engagement"] == 0.0  # min


def test_apply_writes_multi_key(state: InboxState, scenario: Scenario):
    """Multi-key writes apply atomically — every key is set in one call."""
    apply_writes(state, scenario, {"engagement": 3.0})
    # Use a quality that exists; engagement is the only one in the loaded
    # scenario, so a one-key call exercises the loop path. Multi-key is
    # exercised in the synthetic-scenario test below.
    assert state.qualities["engagement"] == 3.0


# ───────────────────────── load-time validation ─────────────────────────


_BASE_QUALITIES = [
    {"id": "engagement", "type": "float", "min": 0.0, "max": 10.0, "default": 0.0},
]
_BASE_NPC = {
    "id": "aria",
    "display_name": "Aria",
    "email_address": "aria@x",
}
_BASE_THREAD = {"id": "aria", "npc": "aria"}


def _scenario_dict(*, beats: list[dict], qualities: list[dict] | None = None) -> dict:
    return {
        "title": "T",
        "premise": "P",
        "player_email": "p@x",
        "qualities": qualities or _BASE_QUALITIES,
        "npcs": [_BASE_NPC],
        "threads": [_BASE_THREAD],
        "beats": beats,
    }


def _classify_axis(quality: str = "engagement") -> dict:
    return {
        "quality": quality,
        "question": "?",
        "allowed_values": ["a", "b"],
        "map": {"a": 1.0, "b": 0.0},
    }


def _plot_beat(
    beat_id: str = "aria.test",
    *,
    spec: dict | None = None,
    classify_reply_axes: list[dict] | None = None,
) -> dict:
    return {
        "id": beat_id,
        "type": "plot",
        "thread": "aria",
        "npc": "aria",
        "subject": "Re: x",
        "spec": spec or {"intent": "be brief"},
        "classify_reply_axes": classify_reply_axes or [],
    }


def test_load_rejects_duplicate_axis_quality_within_one_beat():
    """Two axes targeting the same quality on one beat is an authoring
    error — the second write would always clobber the first."""
    beat = _plot_beat(classify_reply_axes=[_classify_axis(), _classify_axis()])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_rejects_duplicate_axis_identifier_when_axis_name_used():
    """Two axes on one beat sharing the same axis_name (or quality) are
    rejected — the dedup key is `axis.identifier`, which is axis_name or
    quality."""
    a1 = {
        "axis_name": "verdict",
        "question": "?",
        "allowed_values": ["a", "b"],
        "map": {"a": 1.0, "b": 0.0},
    }
    a2 = {
        "axis_name": "verdict",
        "question": "?",
        "allowed_values": ["x", "y"],
        "map": {"x": 1.0, "y": 0.0},
    }
    beat = _plot_beat(classify_reply_axes=[a1, a2])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_accepts_verdict_only_axis_with_axis_name():
    """An axis with `quality=None, axis_name=<id>` is the verdict-only
    pattern — no quality write, but the AxisResult still records on the
    Reply for RecentClassifierValue predicates."""
    axis = {
        "axis_name": "bargain_response",
        "question": "?",
        "allowed_values": ["accepted", "refused"],
        "map": {"accepted": 1.0, "refused": 0.0},
    }
    beat = _plot_beat(classify_reply_axes=[axis])
    Scenario.model_validate(_scenario_dict(beats=[beat]))  # no raise


def test_classify_reply_requires_quality_or_axis_name():
    """A ClassifyReply with neither `quality` nor `axis_name` is invalid."""
    with pytest.raises(ValidationError):
        ClassifyReply(
            question="?",
            allowed_values=["a", "b"],
            map={"a": 1.0, "b": 0.0},
        )


def test_apply_classification_quality_none_does_not_write():
    """Verdict-only axis: no quality write, but AxisResult returns so the
    caller can append to Reply.classifier_results."""
    scenario = _scenario_with_commitment_qualities()
    state = InboxState.from_scenario(scenario)
    classify = ClassifyReply(
        axis_name="bargain_response",
        question="?",
        allowed_values=["accepted", "refused"],
        map={"accepted": 1.0, "refused": 0.0},
    )
    qualities_before = dict(state.qualities)
    result = apply_classification(state, scenario, classify, "accepted")
    assert result is not None
    assert result.quality == "bargain_response"  # axis identifier
    assert result.value == "accepted"
    assert state.qualities == qualities_before  # no write happened


def test_recent_classifier_value_matches_axis_name(state: InboxState):
    """RecentClassifierValue.quality matches the axis identifier — for
    verdict-only axes that's the axis_name."""
    state.replies = [
        _reply_with_axis("aria", "bargain_response", "accepted"),
    ]
    pred = RecentClassifierValue(
        quality="bargain_response", values=["accepted"], window=1, match="any"
    )
    assert evaluate_predicate(pred, state) is True
    pred_miss = RecentClassifierValue(
        quality="bargain_response", values=["refused"], window=1, match="any"
    )
    assert evaluate_predicate(pred_miss, state) is False


def test_load_rejects_writes_collision_with_axis_on_same_beat():
    """A beat whose `writes` key collides with one of its own classifier
    axes is rejected — multi_turn re-fire would clobber the classifier
    each tick."""
    beat = _plot_beat(
        spec={"intent": "x", "writes": {"engagement": 1.0}},
        classify_reply_axes=[_classify_axis()],
    )
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_rejects_writes_unknown_quality():
    beat = _plot_beat(spec={"intent": "x", "writes": {"nonexistent": 1.0}})
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_rejects_recent_classifier_value_unknown_quality():
    beat = _plot_beat()
    beat["preconditions"] = [{
        "kind": "recent_classifier_value",
        "quality": "nonexistent",
        "values": ["a"],
        "window": 1,
    }]
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_rejects_validate_facts_without_constraints():
    beat = _plot_beat(spec={"intent": "x", "validate_facts": True})
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_accepts_validate_facts_with_must_include_facts():
    beat = _plot_beat(spec={
        "intent": "x",
        "must_include_facts": ["thank the player"],
        "validate_facts": True,
    })
    Scenario.model_validate(_scenario_dict(beats=[beat]))  # no raise


# ───────────────────────── multi-key writes + writes-on-silent-complete ─────────────────────────


def _scenario_with_commitment_qualities() -> Scenario:
    """Synthetic two-quality scenario — engagement plus a commitment.* —
    used to exercise multi-key writes and the silent-complete-with-writes
    path."""
    return Scenario.model_validate(_scenario_dict(
        qualities=_BASE_QUALITIES + [
            {
                "id": "commitment.bargain_state",
                "type": "float",
                "min": 0.0,
                "max": 10.0,
                "default": 0.0,
            },
        ],
        beats=[_plot_beat(beat_id="aria.test")],
    ))


def test_apply_writes_multi_key_atomic():
    scenario = _scenario_with_commitment_qualities()
    state = InboxState.from_scenario(scenario)
    apply_writes(state, scenario, {
        "engagement": 4.0,
        "commitment.bargain_state": 2.0,
    })
    assert state.qualities["engagement"] == 4.0
    assert state.qualities["commitment.bargain_state"] == 2.0


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


def _reply_with_axis(thread: str | None, quality: str, value: str) -> Reply:
    """Reply seeded with one classifier axis result — used for
    RecentClassifierValue evaluation tests."""
    r = _make_reply(thread)
    r.classifier_results.append(
        AxisResult(quality=quality, value=value, rationale="(t)")
    )
    return r


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
