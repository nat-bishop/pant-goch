"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, beat scheduler, classifier-driven quality writes (set + add +
max + min), recent-window predicates, literal effects, and reply/email
lookup helpers.

No LLM, no FastAPI. Runs fast.
"""
import uuid

import pytest
from pydantic import ValidationError

from npc_service.models import (
    AxisResult,
    Beat,
    BeatFired,
    BeatSpec,
    CanonCategory,
    ClassifyDirective,
    Effect,
    Email,
    InboxState,
    Not_,
    Or_,
    PlayerCanonEntry,
    PlayerCanonExists,
    PlayerRepliesToThreadGte,
    Quality,
    QualityEquals,
    QualityGte,
    QualityLt,
    RecentClassifierValue,
    Reply,
    Scenario,
)
from npc_service.spine import (
    apply_classify_effect,
    apply_literal_effects,
    describe_predicate,
    evaluate_predicate,
    find_due_beats,
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
        "email_domain": "x",
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
                "spec": {"pre_authored_prose": "Welcome."},
            },
            {
                "id": "aria.complete_warm",
                "thread": "aria",
                "npc": "aria",
                "ends_session": True,
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
                "preconditions": [
                    {"kind": "player_replies_to_thread_gte", "thread": "aria", "value": 1},
                    {"kind": "not", "of": {"kind": "beat_fired", "beat": "aria.complete_warm"}},
                    {"kind": "not", "of": {"kind": "beat_fired", "beat": "aria.complete_standard"}},
                ],
                "spec": {
                    "intent": "ask another question",
                    "effects": [{
                        "quality": "engagement",
                        "op": "add",
                        "classify": {
                            "question": "Did the player engage?",
                            "allowed_values": ["engaged", "deflected"],
                            "map": {"engaged": 1.0, "deflected": -0.5},
                        },
                    }],
                },
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
    classify_effects = [e for e in loop.spec.effects if e.is_classify]
    assert len(classify_effects) == 1
    effect = classify_effects[0]
    assert effect.op == "add"
    assert effect.quality == "engagement"


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
    pending_ids = {pb.beat_id for pb in state.pending_beats}
    assert pending_ids == {
        "aria.welcome",
        "aria.complete_warm",
        "aria.complete_standard",
        "aria.loop",
    }


def test_boot_tick_fires_aria_welcome(state: InboxState, scenario: Scenario):
    due, _report = find_due_beats(state, scenario)
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


def test_quality_equals_string(state: InboxState):
    """QualityEquals works on string-typed qualities (used by bargain_state
    branching). Verdict strings written by classify effects are read by
    QualityEquals predicates."""
    state.qualities["bargain_state"] = "refused"
    assert evaluate_predicate(
        QualityEquals(quality="bargain_state", value="refused"), state,
    ) is True
    assert evaluate_predicate(
        QualityEquals(quality="bargain_state", value="accepted"), state,
    ) is False


def test_quality_equals_numeric(state: InboxState):
    state.qualities["engagement"] = 4.0
    assert evaluate_predicate(
        QualityEquals(quality="engagement", value=4.0), state,
    ) is True
    assert evaluate_predicate(
        QualityEquals(quality="engagement", value=3.0), state,
    ) is False


# ───────────────────────── describe_predicate ─────────────────────────


def test_describe_quality_gte_renders_threshold_and_current(
    state: InboxState,
):
    state.qualities["engagement"] = 2.5
    out = describe_predicate(QualityGte(quality="engagement", value=4.0), state)
    assert out == "engagement>=4 got 2.5"


def test_describe_quality_gte_renders_unset_quality(state: InboxState):
    state.qualities.pop("engagement", None)
    out = describe_predicate(QualityGte(quality="engagement", value=4.0), state)
    assert "(unset)" in out


def test_describe_quality_lt_renders_threshold_and_current(
    state: InboxState,
):
    state.qualities["engagement"] = 3.5
    out = describe_predicate(QualityLt(quality="engagement", value=2.0), state)
    assert out == "engagement<2 got 3.5"


def test_describe_player_replies_renders_thread_and_count(
    state: InboxState,
):
    state.replies = [_make_reply(thread="aria"), _make_reply(thread="aria")]
    out = describe_predicate(
        PlayerRepliesToThreadGte(thread="aria", value=5), state,
    )
    assert out == "replies(aria)>=5 got 2"


def test_describe_beat_fired_unfired_says_not_yet(state: InboxState):
    out = describe_predicate(BeatFired(beat="aria.welcome"), state)
    assert "not yet" in out
    assert "aria.welcome" in out


def test_describe_beat_fired_says_just_the_id_when_fired(
    state: InboxState,
):
    state.fired_beat_ids.add("aria.welcome")
    out = describe_predicate(BeatFired(beat="aria.welcome"), state)
    assert "not yet" not in out
    assert "aria.welcome" in out


def test_describe_recent_classifier_value_includes_window_and_count(
    state: InboxState,
):
    state.replies = [
        _reply_with_axis("aria", "engagement", "engaged"),
        _reply_with_axis("aria", "engagement", "deflected"),
    ]
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3, match="any",
    )
    out = describe_predicate(pred, state)
    assert "engagement" in out and "last 3" in out
    assert "engaged" in out
    assert "1/2" in out


def test_describe_not_unwraps_inner(state: InboxState):
    state.qualities["engagement"] = 2.5
    out = describe_predicate(
        Not_(of=QualityGte(quality="engagement", value=4.0)), state,
    )
    assert out.startswith("not(")
    assert "engagement>=4 got 2.5" in out


def test_describe_or_joins_subpredicates(state: InboxState):
    state.qualities["engagement"] = 2.5
    pred = Or_(of=[
        QualityGte(quality="engagement", value=4.0),
        QualityLt(quality="engagement", value=1.0),
    ])
    out = describe_predicate(pred, state)
    assert out.startswith("or(")
    assert " | " in out
    assert "engagement>=4 got 2.5" in out
    assert "engagement<1 got 2.5" in out


def test_describe_or_recurses_into_arms(state: InboxState):
    pred = Or_(of=[
        PlayerCanonExists(name="name"),
        PlayerRepliesToThreadGte(thread="aria", value=4),
    ])
    out = describe_predicate(pred, state)
    assert "player_canon(name)" in out
    assert "replies(aria)>=4 got 0" in out


def test_describe_player_canon_exists_unset_says_not_yet(state: InboxState):
    state.player_canon.pop("name", None)
    out = describe_predicate(PlayerCanonExists(name="name"), state)
    assert "not yet" in out
    assert "name" in out


def test_describe_player_canon_exists_set_includes_value(state: InboxState):
    state.player_canon["name"] = PlayerCanonEntry(
        summary="Henry",
        literal_text_quote="I'm Henry",
        salience=0.9,
        category=CanonCategory.NAME,
        turn_of_origin=1,
    )
    out = describe_predicate(PlayerCanonExists(name="name"), state)
    assert "Henry" in out
    assert "not yet" not in out


# ───────────────────────── find_due_beats eligibility report ─────────────────────────


def test_find_due_beats_reports_value_blocked_with_concrete_predicate(
    state: InboxState, scenario: Scenario,
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 3.0  # below complete_warm's 4.0 threshold

    _eligible, report = find_due_beats(state, scenario)
    by_id = {be.beat_id: be for be in report}
    warm = by_id.get("aria.complete_warm")
    assert warm is not None
    assert warm.status == "blocked"
    assert warm.blocked_by is not None
    assert "engagement>=4" in warm.blocked_by
    assert "got 3" in warm.blocked_by


def test_find_due_beats_omits_structurally_blocked_beats_from_report(
    state: InboxState, scenario: Scenario,
):
    sc = Scenario.model_validate({
        "title": "T", "premise": "P", "email_domain": "x",
        "qualities": [{"id": "engagement", "type": "float", "min": 0.0, "max": 10.0, "default": 0.0}],
        "npcs": [{"id": "aria", "display_name": "Aria", "email_address": "aria@x"}],
        "threads": [{"id": "aria", "npc": "aria"}],
        "beats": [
            {
                "id": "aria.welcome",
                "thread": "aria",
                "npc": "aria",
                "spec": {"pre_authored_prose": "Welcome."},
            },
            {
                "id": "aria.first",
                "thread": "aria",
                "npc": "aria",
                "spec": {"intent": "x"},
                "preconditions": [
                    {"kind": "player_replies_to_thread_gte", "thread": "aria", "value": 1},
                ],
            },
            {
                "id": "aria.downstream",
                "thread": "aria",
                "npc": "aria",
                "spec": {"intent": "y"},
                "preconditions": [
                    {"kind": "beat_fired", "beat": "aria.first"},
                ],
            },
        ],
    })
    s = InboxState.from_scenario(sc)
    s.fired_beat_ids.add("aria.welcome")

    _eligible, report = find_due_beats(s, sc)
    by_id = {be.beat_id: be for be in report}
    assert "aria.first" in by_id
    assert by_id["aria.first"].status == "blocked"
    assert "aria.downstream" not in by_id


# ───────────────────────── classify-effect writes ─────────────────────────


def _engagement_classify_effect(*, op: str = "set") -> Effect:
    return Effect(
        quality="engagement",
        op=op,
        classify=ClassifyDirective(
            question="(test)",
            allowed_values=["engaged", "deflected", "unclear"],
            map={"engaged": 1.0, "deflected": -0.5, "unclear": 0.5},
        ),
    )


def test_apply_classify_effect_set_mode_overwrites(
    state: InboxState, scenario: Scenario
):
    """Default `op=set`: the mapped score replaces the quality. Repeated
    applies do not stack — subsequent calls overwrite."""
    effect = _engagement_classify_effect(op="set")
    apply_classify_effect(state, scenario, effect, "engaged")
    assert state.qualities["engagement"] == 1.0
    apply_classify_effect(state, scenario, effect, "engaged")
    assert state.qualities["engagement"] == 1.0  # no stacking


def test_apply_classify_effect_add_accumulates(
    state: InboxState, scenario: Scenario
):
    """With `op=add`, each apply adds the mapped score to the current value.
    Negative deltas reduce it; the quality bound clamps at 0 so a long string
    of deflections won't go negative."""
    effect = _engagement_classify_effect(op="add")
    apply_classify_effect(state, scenario, effect, "engaged")
    apply_classify_effect(state, scenario, effect, "engaged")
    assert state.qualities["engagement"] == 2.0
    apply_classify_effect(state, scenario, effect, "deflected")
    assert state.qualities["engagement"] == 1.5
    for _ in range(10):
        apply_classify_effect(state, scenario, effect, "deflected")
    assert state.qualities["engagement"] == 0.0


def test_apply_classify_effect_clamps_to_quality_bounds(
    state: InboxState, scenario: Scenario
):
    """Out-of-bound mapped values get clipped via _clamp_quality. Engagement
    is bounded 0..10."""
    effect = Effect(
        quality="engagement",
        classify=ClassifyDirective(
            question="(test)",
            allowed_values=["over", "under"],
            map={"over": 50.0, "under": -10.0},
        ),
    )
    apply_classify_effect(state, scenario, effect, "over")
    assert state.qualities["engagement"] == 10.0  # clamped to max
    apply_classify_effect(state, scenario, effect, "under")
    assert state.qualities["engagement"] == 0.0  # clamped to min (set mode)


def test_apply_classify_effect_off_spec_returns_none(
    state: InboxState, scenario: Scenario
):
    """LLM returned a value not in allowed_values: leave quality untouched
    and return (None, None) so the caller can log it."""
    effect = Effect(
        quality="engagement",
        classify=ClassifyDirective(
            question="(test)",
            allowed_values=["engaged", "deflected"],
            map={"engaged": 1.0, "deflected": 0.0},
        ),
    )
    state.qualities["engagement"] = 3.0
    applied, delta = apply_classify_effect(state, scenario, effect, "garbage")
    assert applied is None
    assert delta is None
    assert state.qualities["engagement"] == 3.0


def test_apply_classify_effect_returns_axis_result_and_delta_on_success(
    state: InboxState, scenario: Scenario
):
    """Returns (AxisResult, QualityDelta) — the caller appends the AxisResult
    onto a Reply so RecentClassifierValue predicates can walk state.replies,
    and folds the QualityDelta into TickDecision.quality_deltas."""
    effect = _engagement_classify_effect(op="add")
    applied, delta = apply_classify_effect(
        state, scenario, effect, "engaged", "rationale x",
    )
    assert applied is not None
    assert applied.quality == "engagement"
    assert applied.value == "engaged"
    assert applied.rationale == "rationale x"
    assert delta is not None
    assert delta.quality == "engagement"
    assert delta.before == 0.0
    assert delta.after == 1.0
    assert delta.source == "classify:engagement"


def test_apply_classify_effect_no_op_returns_axis_result_but_no_delta(
    state: InboxState, scenario: Scenario
):
    """When the value lands on the same quality value (e.g. op=set with the
    same mapped score), AxisResult still returns but delta is None."""
    effect = _engagement_classify_effect(op="set")
    state.qualities["engagement"] = 1.0
    applied, delta = apply_classify_effect(state, scenario, effect, "engaged")
    assert applied is not None
    assert delta is None  # value was already 1.0


def test_apply_classify_effect_string_quality_writes_verdict_verbatim():
    """For string-typed qualities, the verdict string IS the value written —
    no `map` field, no numeric mapping. Replaces the old verdict-only axis_name
    pattern."""
    sc = Scenario.model_validate({
        "title": "T", "premise": "P", "email_domain": "x",
        "qualities": [{
            "id": "bargain_state",
            "type": "string",
            "default": "untouched",
            "allowed_values": ["untouched", "offered", "accepted", "refused", "hedged"],
        }],
        "npcs": [{"id": "aria", "display_name": "A", "email_address": "a@x"}],
        "threads": [{"id": "aria", "npc": "aria"}],
        "beats": [{
            "id": "aria.test",
            "thread": "aria",
            "npc": "aria",
            "spec": {"intent": "x"},
        }],
    })
    s = InboxState.from_scenario(sc)
    effect = Effect(
        quality="bargain_state",
        classify=ClassifyDirective(
            question="?",
            allowed_values=["accepted", "refused", "hedged"],
        ),
    )
    applied, delta = apply_classify_effect(s, sc, effect, "refused", "rationale")
    assert applied is not None
    assert applied.quality == "bargain_state"
    assert applied.value == "refused"
    assert s.qualities["bargain_state"] == "refused"
    assert delta is not None
    assert delta.before == "untouched"
    assert delta.after == "refused"


# ───────────────────────── loop scheduling ─────────────────────────


def test_loop_eligible_after_first_player_reply(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.append(_make_reply(thread="aria"))
    due, _report = find_due_beats(state, scenario)
    assert any(b.id == "aria.loop" for b in due)


def test_loop_blocked_when_terminal_already_fired(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(3))
    state.fired_beat_ids.add("aria.complete_warm")
    due, _report = find_due_beats(state, scenario)
    assert not any(b.id == "aria.loop" for b in due)


# ───────────────────────── terminals ─────────────────────────


def test_complete_warm_fires_at_engagement_threshold(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 4.0
    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
    assert "aria.complete_warm" in due_ids


def test_complete_warm_blocked_below_threshold(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 3.5
    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
    assert "aria.complete_warm" not in due_ids


def test_complete_warm_blocked_below_reply_count(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(4))
    state.qualities["engagement"] = 8.0
    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
    assert "aria.complete_warm" not in due_ids


def test_both_terminals_eligible_when_paths_overlap(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 5.0

    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
    assert "aria.complete_warm" in due_ids
    assert "aria.complete_standard" in due_ids
    assert "aria.loop" in due_ids


def test_complete_standard_fires_when_engagement_low(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 1.0

    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
    assert "aria.complete_warm" not in due_ids
    assert "aria.complete_standard" in due_ids


def test_loop_blocked_after_terminal_added_to_fired_ids(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply(thread="aria") for _ in range(5))
    state.qualities["engagement"] = 5.0
    state.fired_beat_ids.add("aria.complete_warm")

    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
    assert "aria.loop" not in due_ids
    assert "aria.complete_standard" not in due_ids


# ───────────────────────── RecentClassifierValue predicate ─────────────────────────


def test_recent_classifier_value_empty_replies_returns_false(
    state: InboxState,
):
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3
    )
    assert evaluate_predicate(pred, state) is False


def test_recent_classifier_value_any_match(state: InboxState):
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
    state.replies = [
        _reply_with_axis("aria", "engagement", "engaged"),
        _reply_with_axis("aria", "engagement", "engaged"),
    ]
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=3, match="all"
    )
    assert evaluate_predicate(pred, state) is False
    state.replies.append(_reply_with_axis("aria", "engagement", "engaged"))
    assert evaluate_predicate(pred, state) is True
    state.replies.append(_reply_with_axis("aria", "engagement", "deflected"))
    assert evaluate_predicate(pred, state) is False


def test_recent_classifier_value_int_match(state: InboxState):
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
    state.replies = [
        _reply_with_axis("aria", "engagement", "deflected"),
        _reply_with_axis("aria", "engagement", "engaged"),
        _reply_with_axis("aria", "engagement", "unclear"),
    ]
    pred = RecentClassifierValue(
        quality="engagement", values=["deflected", "unclear"], window=3, match=2
    )
    assert evaluate_predicate(pred, state) is True


def test_recent_classifier_value_skips_replies_without_axis(state: InboxState):
    state.replies = [
        _reply_with_axis("aria", "engagement", "engaged"),
        _make_reply(thread="aria"),
        _reply_with_axis("aria", "engagement", "engaged"),
    ]
    pred = RecentClassifierValue(
        quality="engagement", values=["engaged"], window=2, match="all"
    )
    assert evaluate_predicate(pred, state) is True


def test_recent_classifier_value_under_not_(state: InboxState):
    state.replies = [
        _reply_with_axis("aria", "engagement", "deflected"),
    ]
    pred = Not_(of=RecentClassifierValue(
        quality="engagement", values=["engaged"], window=1, match="any",
    ))
    assert evaluate_predicate(pred, state) is True


def test_recent_classifier_value_match_int_validation():
    with pytest.raises(ValidationError):
        RecentClassifierValue(
            quality="engagement", values=["x"], window=3, match=5
        )
    with pytest.raises(ValidationError):
        RecentClassifierValue(
            quality="engagement", values=["x"], window=3, match=0
        )


def test_recent_classifier_value_walks_string_quality_verdicts(state: InboxState):
    """RecentClassifierValue walks AxisResult.value strings — works
    identically whether the underlying quality is numeric or string-typed.
    Used by aria.bargain_followup to read the bargain_state verdict
    written by aria.bargain_offer's classifier."""
    state.replies = [
        _reply_with_axis("aria", "bargain_state", "accepted"),
    ]
    pred = RecentClassifierValue(
        quality="bargain_state", values=["accepted"], window=1, match="any"
    )
    assert evaluate_predicate(pred, state) is True
    pred_miss = RecentClassifierValue(
        quality="bargain_state", values=["refused"], window=1, match="any"
    )
    assert evaluate_predicate(pred_miss, state) is False


# ───────────────────────── apply_literal_effects ─────────────────────────


def _engagement_literal_effect(value: float, *, op: str = "set", when=None) -> Effect:
    return Effect(quality="engagement", value=value, op=op, when=when)


def test_apply_literal_effects_set_semantics(
    state: InboxState, scenario: Scenario
):
    """`op=set` (default): writes overwrite the current value."""
    apply_literal_effects(state, scenario, [_engagement_literal_effect(5.0)])
    assert state.qualities["engagement"] == 5.0
    apply_literal_effects(state, scenario, [_engagement_literal_effect(2.0)])
    assert state.qualities["engagement"] == 2.0


def test_apply_literal_effects_clamps_to_quality_bounds(
    state: InboxState, scenario: Scenario
):
    """Clamping is shared with classify-effect writes via _clamp_quality."""
    apply_literal_effects(state, scenario, [_engagement_literal_effect(99.0)])
    assert state.qualities["engagement"] == 10.0  # max
    apply_literal_effects(state, scenario, [_engagement_literal_effect(-3.0)])
    assert state.qualities["engagement"] == 0.0  # min


def test_apply_literal_effects_op_max_acts_as_floor(
    state: InboxState, scenario: Scenario
):
    """`op=max` writes the larger of (current, incoming) — idempotent floor."""
    state.qualities["engagement"] = 5.0
    apply_literal_effects(state, scenario, [_engagement_literal_effect(3.0, op="max")])
    assert state.qualities["engagement"] == 5.0  # floor doesn't lower
    apply_literal_effects(state, scenario, [_engagement_literal_effect(7.0, op="max")])
    assert state.qualities["engagement"] == 7.0  # raises


def test_apply_literal_effects_op_min_acts_as_cap(
    state: InboxState, scenario: Scenario
):
    """`op=min` writes the smaller of (current, incoming) — cap."""
    state.qualities["engagement"] = 5.0
    apply_literal_effects(state, scenario, [_engagement_literal_effect(7.0, op="min")])
    assert state.qualities["engagement"] == 5.0  # cap doesn't raise
    apply_literal_effects(state, scenario, [_engagement_literal_effect(3.0, op="min")])
    assert state.qualities["engagement"] == 3.0  # lowers


def test_apply_literal_effects_op_add_accumulates(
    state: InboxState, scenario: Scenario
):
    apply_literal_effects(state, scenario, [_engagement_literal_effect(2.0, op="add")])
    apply_literal_effects(state, scenario, [_engagement_literal_effect(3.0, op="add")])
    assert state.qualities["engagement"] == 5.0


def test_apply_literal_effects_when_predicate_skips_when_false(
    state: InboxState, scenario: Scenario
):
    """Effect's `when:` predicate gates the write. False → skip."""
    when_true = QualityGte(quality="engagement", value=0.0)
    when_false = QualityGte(quality="engagement", value=999.0)
    apply_literal_effects(
        state, scenario,
        [_engagement_literal_effect(5.0, when=when_true)],
    )
    assert state.qualities["engagement"] == 5.0
    apply_literal_effects(
        state, scenario,
        [_engagement_literal_effect(7.0, when=when_false)],
    )
    assert state.qualities["engagement"] == 5.0  # unchanged — when was false


def test_apply_literal_effects_skips_classify_effects(
    state: InboxState, scenario: Scenario
):
    """Classify-typed effects are scheduled for reply-time — the literal
    effect applier ignores them."""
    classify = _engagement_classify_effect()
    literal = _engagement_literal_effect(3.0)
    apply_literal_effects(state, scenario, [classify, literal])
    assert state.qualities["engagement"] == 3.0  # only the literal landed


def test_apply_literal_effects_multi_effect_atomic():
    """Multiple effects in one call all apply."""
    sc = Scenario.model_validate(_scenario_dict(
        qualities=_BASE_QUALITIES + [
            {"id": "bargain_state", "type": "string", "default": "untouched",
             "allowed_values": ["untouched", "offered"]},
        ],
        beats=[_plot_beat(beat_id="aria.test")],
    ))
    s = InboxState.from_scenario(sc)
    apply_literal_effects(s, sc, [
        Effect(quality="engagement", value=4.0),
        Effect(quality="bargain_state", value="offered"),
    ])
    assert s.qualities["engagement"] == 4.0
    assert s.qualities["bargain_state"] == "offered"


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
        "email_domain": "x",
        "qualities": qualities or _BASE_QUALITIES,
        "npcs": [_BASE_NPC],
        "threads": [_BASE_THREAD],
        "beats": beats,
    }


def _classify_effect_dict(quality: str = "engagement") -> dict:
    return {
        "quality": quality,
        "classify": {
            "question": "?",
            "allowed_values": ["a", "b"],
            "map": {"a": 1.0, "b": 0.0},
        },
    }


def _plot_beat(
    beat_id: str = "aria.test",
    *,
    spec: dict | None = None,
    effects: list[dict] | None = None,
) -> dict:
    base_spec = {"intent": "be brief"}
    if spec is not None:
        base_spec = {**base_spec, **spec}
    if effects is not None:
        base_spec["effects"] = effects
    return {
        "id": beat_id,
        "thread": "aria",
        "npc": "aria",
        "spec": base_spec,
    }


def test_load_rejects_duplicate_classify_effects_on_same_quality():
    """Two classify effects on one beat targeting the same quality is an
    authoring error — the classifier returns one verdict per quality, so
    two effects on the same target are ambiguous."""
    beat = _plot_beat(effects=[_classify_effect_dict(), _classify_effect_dict()])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_accepts_string_quality_classify_effect_no_map():
    """A classify effect targeting a string quality may omit `map` —
    the verdict string IS the value. Replaces the old verdict-only axis
    pattern."""
    beat = _plot_beat(effects=[{
        "quality": "bargain_state",
        "classify": {
            "question": "?",
            "allowed_values": ["accepted", "refused"],
        },
    }])
    Scenario.model_validate(_scenario_dict(
        qualities=[
            {"id": "bargain_state", "type": "string", "default": "untouched",
             "allowed_values": ["untouched", "accepted", "refused"]},
        ],
        beats=[beat],
    ))


def test_load_rejects_classify_effect_numeric_without_map():
    """Numeric qualities require a `map:` from verdict → float."""
    beat = _plot_beat(effects=[{
        "quality": "engagement",
        "classify": {
            "question": "?",
            "allowed_values": ["a", "b"],
            # missing `map`
        },
    }])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_rejects_classify_effect_string_with_map():
    """String qualities forbid `map:` — the verdict string IS the value."""
    beat = _plot_beat(effects=[{
        "quality": "bargain_state",
        "classify": {
            "question": "?",
            "allowed_values": ["accepted", "refused"],
            "map": {"accepted": 1.0, "refused": 0.0},
        },
    }])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(
            qualities=[
                {"id": "bargain_state", "type": "string", "default": "untouched",
                 "allowed_values": ["untouched", "accepted", "refused"]},
            ],
            beats=[beat],
        ))


def test_load_rejects_effect_unknown_quality():
    beat = _plot_beat(effects=[{"quality": "nonexistent", "value": 1.0}])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_rejects_effect_missing_value_and_classify():
    """An Effect with neither `value` nor `classify` is invalid."""
    with pytest.raises(ValidationError):
        Effect(quality="engagement")


def test_load_rejects_effect_with_both_value_and_classify():
    """An Effect with both value AND classify is invalid (mutually exclusive)."""
    with pytest.raises(ValidationError):
        Effect(
            quality="engagement",
            value=1.0,
            classify=ClassifyDirective(
                question="?", allowed_values=["a", "b"],
                map={"a": 1.0, "b": 0.0},
            ),
        )


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
    beat = _plot_beat(spec={"validate_facts": True})
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_accepts_validate_facts_with_must_include_facts():
    beat = _plot_beat(spec={
        "must_include_facts": ["thank the player"],
        "validate_facts": True,
    })
    Scenario.model_validate(_scenario_dict(beats=[beat]))


def test_load_rejects_string_quality_default_not_string():
    """String-typed qualities require string default."""
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(
            qualities=[{
                "id": "bargain_state",
                "type": "string",
                "default": 0,  # not a string
            }],
            beats=[_plot_beat()],
        ))


def test_load_rejects_numeric_quality_with_allowed_values():
    """allowed_values is a string-quality concept — numeric forbids it."""
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(
            qualities=[{
                "id": "engagement",
                "type": "float",
                "min": 0.0, "max": 10.0, "default": 0.0,
                "allowed_values": ["high", "low"],
            }],
            beats=[_plot_beat()],
        ))


def test_load_rejects_string_quality_op_other_than_set():
    """String qualities only support op=set."""
    beat = _plot_beat(effects=[{
        "quality": "bargain_state",
        "value": "offered",
        "op": "add",
    }])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(
            qualities=[
                {"id": "bargain_state", "type": "string", "default": "untouched",
                 "allowed_values": ["untouched", "offered"]},
            ],
            beats=[beat],
        ))


def test_load_rejects_string_quality_value_not_in_allowed_values():
    beat = _plot_beat(effects=[{
        "quality": "bargain_state",
        "value": "phantom",
    }])
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario_dict(
            qualities=[
                {"id": "bargain_state", "type": "string", "default": "untouched",
                 "allowed_values": ["untouched", "offered"]},
            ],
            beats=[beat],
        ))


# ───────────────────────── multi-turn self-stop on Not_(<exit>) ─────────────────────────


def test_multi_turn_beat_self_stops_when_exit_predicate_flips():
    """The calibration-scene pattern: a multi-turn beat with `Not_(<exit>)`
    in its preconditions remains eligible while <exit> is False, and stops
    being eligible the moment <exit> flips True."""
    scenario = Scenario.model_validate({
        "title": "T",
        "premise": "p",
        "email_domain": "x",
        "qualities": [],
        "npcs": [{"id": "aria", "display_name": "A", "email_address": "a@x"}],
        "threads": [{"id": "aria", "npc": "aria"}],
        "beats": [
            {
                "id": "aria.calibration",
                "thread": "aria",
                "npc": "aria",
                "multi_turn": True,
                "preconditions": [
                    {
                        "kind": "player_replies_to_thread_gte",
                        "thread": "aria",
                        "value": 1,
                    },
                    {
                        "kind": "not",
                        "of": {
                            "kind": "or",
                            "of": [
                                {"kind": "player_canon_exists", "name": "name"},
                                {
                                    "kind": "player_replies_to_thread_gte",
                                    "thread": "aria",
                                    "value": 4,
                                },
                            ],
                        },
                    },
                ],
                "spec": {"intent": "elicit name"},
            },
        ],
    })
    state = InboxState.from_scenario(scenario)

    assert [b.id for b in find_due_beats(state, scenario)[0]] == []

    state.replies.append(Reply(
        id="r1", thread_id="aria", beat_responded_to=None, recipient="A", body="hi",
    ))
    assert [b.id for b in find_due_beats(state, scenario)[0]] == ["aria.calibration"]

    state.replies.append(Reply(
        id="r2", thread_id="aria", beat_responded_to=None, recipient="A", body="ok",
    ))
    assert [b.id for b in find_due_beats(state, scenario)[0]] == ["aria.calibration"]

    state.player_canon["name"] = PlayerCanonEntry(
        summary="Marcus", literal_text_quote="I'm Marcus", salience=0.9,
        category="name", turn_of_origin=2,
    )
    assert [b.id for b in find_due_beats(state, scenario)[0]] == []

    del state.player_canon["name"]
    for i in range(3, 5):
        state.replies.append(Reply(
            id=f"r{i}", thread_id="aria", beat_responded_to=None, recipient="A",
            body=f"reply {i}",
        ))
    assert [b.id for b in find_due_beats(state, scenario)[0]] == []


def test_or_predicate_evaluation_in_preconditions():
    scenario = Scenario.model_validate({
        "title": "T",
        "premise": "p",
        "email_domain": "x",
        "qualities": [],
        "npcs": [{"id": "aria", "display_name": "A", "email_address": "a@x"}],
        "threads": [{"id": "aria", "npc": "aria"}],
        "beats": [
            {
                "id": "aria.body",
                "thread": "aria",
                "npc": "aria",
                "preconditions": [
                    {
                        "kind": "or",
                        "of": [
                            {"kind": "player_canon_exists", "name": "name"},
                            {
                                "kind": "player_replies_to_thread_gte",
                                "thread": "aria",
                                "value": 4,
                            },
                        ],
                    },
                ],
                "spec": {"intent": "body fires when calibration done"},
            },
        ],
    })
    state = InboxState.from_scenario(scenario)
    assert [b.id for b in find_due_beats(state, scenario)[0]] == []
    state.replies.append(Reply(
        id="r1", thread_id="aria", beat_responded_to=None, recipient="A", body="hi",
    ))
    assert [b.id for b in find_due_beats(state, scenario)[0]] == []
    for i in range(2, 5):
        state.replies.append(Reply(
            id=f"r{i}", thread_id="aria", beat_responded_to=None, recipient="A", body="x",
        ))
    assert [b.id for b in find_due_beats(state, scenario)[0]] == ["aria.body"]


# ───────────────────────── helpers ─────────────────────────


def _drain_boot(state: InboxState, scenario: Scenario) -> None:
    for beat in find_due_beats(state, scenario)[0]:
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
    """Reply seeded with one classifier result — used for
    RecentClassifierValue evaluation tests."""
    r = _make_reply(thread)
    r.classifier_results.append(
        AxisResult(quality=quality, value=value, rationale="(t)")
    )
    return r


def _make_email(
    *,
    id: str,
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
        body="",
        sent_at_label="just now",
    )
