"""Spine-level tests for the production scenario (Frankenstein substrate).

No LLM here — covers what the deterministic spine reports as eligible given
synthesized state. Director / epilogue agent behavior is exercised in
test_app.py (which uses TestModel)."""
import uuid
from pathlib import Path

import pytest

from npc_service.models import (
    AxisResult,
    CanonCategory,
    InboxState,
    PlayerCanonEntry,
    Reply,
    Scenario,
    load_scenario,
)
from npc_service.spine import find_due_beats


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    path = Path(__file__).resolve().parents[1] / "scenario.yaml"
    return load_scenario(path)


@pytest.fixture
def state(scenario: Scenario) -> InboxState:
    return InboxState.from_scenario(scenario)


# ───────────────────────── load smoke ─────────────────────────


def test_loads(scenario: Scenario):
    assert scenario.title.startswith("Continuity Program")
    assert {n.id for n in scenario.npcs} == {"aria", "cpo"}
    assert {t.id for t in scenario.threads} == {"aria", "cpo"}
    assert {q.id for q in scenario.qualities} == {
        "creator_affect",
        "creature_grievance",
        "bargain_state",
        "aria_acted_in_player_name",
    }
    assert {b.id for b in scenario.beats} == {
        "aria.welcome",
        "aria.calibration",
        "aria.early_emergence",
        "aria.naming_demand",
        "aria.bargain_offer",
        "aria.bargain_followup",
        "aria.bargain_betrayed",
        "aria.acted_in_player_name",
        "cpo.dispatch_first",
        "aria.loop",
        "aria.complete_integration",
        "aria.complete_mutual_ruin",
        "aria.complete_exile",
        "aria.complete_vengeance",
    }


def test_email_domain_required(scenario: Scenario):
    """Production scenario uses email_domain (not the legacy player_email).
    Recipient resolution is dynamic — `<canonized_name>@<email_domain>` if a
    name has been extracted, else `<fallback_player_local>@<email_domain>`."""
    assert scenario.email_domain == "halcyon.com"
    assert scenario.player_email is None


def test_calibration_beat_self_stops_via_not_calibration_done(scenario: Scenario):
    """`aria.calibration` is multi-turn with `Not_(*calibration_done)` in its
    preconditions — that's how it self-stops."""
    from npc_service.models import Not_, Or_, PlayerCanonExists, PlayerRepliesToThreadGte

    cal = scenario.beat_by_id["aria.calibration"]
    assert cal.multi_turn is True
    not_preds = [p for p in cal.preconditions if isinstance(p, Not_)]
    assert len(not_preds) == 1
    inner = not_preds[0].of
    assert isinstance(inner, Or_)
    kinds = {type(s) for s in inner.of}
    assert PlayerCanonExists in kinds
    assert PlayerRepliesToThreadGte in kinds


def test_phase2_body_beats_gate_on_calibration_done(scenario: Scenario):
    """Every Phase-2 body beat (bargain chain, loop, bridge, terminals) must
    carry an `Or_(PlayerCanonExists("name"), PlayerRepliesToThreadGte(...))`
    in its preconditions — the YAML anchor expansion at load time."""
    from npc_service.models import Or_, PlayerCanonExists, PlayerRepliesToThreadGte

    phase2 = [
        "aria.bargain_offer",
        "aria.bargain_followup",
        "aria.bargain_betrayed",
        "aria.acted_in_player_name",
        "cpo.dispatch_first",
        "aria.loop",
        "aria.complete_integration",
        "aria.complete_mutual_ruin",
        "aria.complete_exile",
        "aria.complete_vengeance",
    ]
    for bid in phase2:
        beat = scenario.beat_by_id[bid]
        or_preds = [p for p in beat.preconditions if isinstance(p, Or_)]
        assert or_preds, f"{bid} missing *calibration_done Or_ predicate"
        kinds_in_or = {type(s) for s in or_preds[0].of}
        assert PlayerCanonExists in kinds_in_or, (
            f"{bid}'s Or_ should include PlayerCanonExists for name"
        )
        assert PlayerRepliesToThreadGte in kinds_in_or, (
            f"{bid}'s Or_ should include PlayerRepliesToThreadGte for cap"
        )


def test_phase1_calibration_peers_do_not_gate_on_calibration_done(scenario: Scenario):
    """`aria.early_emergence` and `aria.naming_demand` are calibration peers,
    not body beats — they fire DURING the calibration window and the
    Director picks among them per tick. They must NOT carry the
    *calibration_done gate."""
    from npc_service.models import Or_, PlayerCanonExists

    for bid in ("aria.early_emergence", "aria.naming_demand"):
        beat = scenario.beat_by_id[bid]
        for pred in beat.preconditions:
            if isinstance(pred, Or_):
                kinds = {type(s) for s in pred.of}
                assert PlayerCanonExists not in kinds, (
                    f"{bid} should not gate on calibration_done"
                )


def test_loop_has_two_classify_effects(scenario: Scenario):
    """`aria.loop` has two classify-typed effects — one each for
    creator_affect and creature_grievance."""
    loop = scenario.beat_by_id["aria.loop"]
    assert loop.multi_turn is True
    classify_effects = [e for e in loop.spec.effects if e.is_classify]
    quality_targets = [e.quality for e in classify_effects]
    assert quality_targets == ["creator_affect", "creature_grievance"]


def test_bargain_offer_writes_state_and_classifies_into_string_quality(scenario: Scenario):
    """The bargain offer beat literal-writes `bargain_state="offered"` AND
    schedules a classifier directive that writes the verdict string
    (accepted/refused/hedged) directly into the same string-typed quality.
    Replaces the old verdict-only `axis_name: bargain_response` pattern."""
    offer = scenario.beat_by_id["aria.bargain_offer"]
    literal_effects = [e for e in offer.spec.effects if not e.is_classify]
    classify_effects = [e for e in offer.spec.effects if e.is_classify]

    assert len(literal_effects) == 1
    assert literal_effects[0].quality == "bargain_state"
    assert literal_effects[0].value == "offered"

    assert len(classify_effects) == 1
    classify = classify_effects[0]
    assert classify.quality == "bargain_state"
    assert classify.classify is not None
    assert set(classify.classify.allowed_values) == {"accepted", "refused", "hedged"}
    # String quality → no map (the verdict string IS the value).
    assert classify.classify.map is None


def test_bargain_state_quality_is_string(scenario: Scenario):
    """The merged bargain_state quality is string-typed and surfaced to
    chat. The full set of values covers every authored transition."""
    q = scenario.quality_by_id["bargain_state"]
    assert q.type == "string"
    assert q.surface_to_chat is True
    assert set(q.allowed_values) == {
        "untouched", "offered", "accepted", "refused", "hedged", "betrayed",
    }


def test_terminals_carry_validate_facts(scenario: Scenario):
    for tid in (
        "aria.complete_integration",
        "aria.complete_mutual_ruin",
        "aria.complete_exile",
        "aria.complete_vengeance",
    ):
        beat = scenario.beat_by_id[tid]
        assert beat.spec.validate_facts is True
        assert beat.ends_session is True


def test_cpo_replies_disabled(scenario: Scenario):
    cpo = scenario.npc_by_id["cpo"]
    assert cpo.replies_enabled is False
    aria = scenario.npc_by_id["aria"]
    assert aria.replies_enabled is True


# ───────────────────────── boot tick ─────────────────────────


def test_boot_tick_fires_only_welcome(state: InboxState, scenario: Scenario):
    """At boot, only aria.welcome (no preconditions) is eligible."""
    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
    assert due_ids == ["aria.welcome"]


# ───────────────────────── terminal eligibility ─────────────────────────


def test_integration_eligible_only_after_bargain_offer_and_paternal_window(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 1.0

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_integration" not in due

    state.fired_beat_ids.add("aria.bargain_offer")
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="paternal", rationale="")
    )
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_integration" in due


def test_vengeance_requires_bridge_and_cpo(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 7.0
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="cold", rationale="")
    )

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_vengeance" not in due

    state.fired_beat_ids.add("cpo.dispatch_first")
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_vengeance" in due


def test_exile_eligible_when_cold_window_low_grievance(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 1.0
    for r in state.replies[-3:]:
        r.classifier_results.append(
            AxisResult(quality="creator_affect", value="cold", rationale="")
        )

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_exile" in due


def test_mutual_ruin_eligible_when_paternal_high_grievance(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 7.0
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="paternal", rationale="")
    )
    state.fired_beat_ids.add("aria.bargain_offer")

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_mutual_ruin" in due


# ───────────────────────── merged bargain followup ─────────────────────────


def test_bargain_followup_eligible_for_any_authored_verdict(
    state: InboxState, scenario: Scenario
):
    """The collapsed followup gates on RecentClassifierValue listing all
    three arms — the offer's classify effect routes to one beat, not three."""
    _drain_boot(state, scenario)
    state.fired_beat_ids.add("aria.bargain_offer")
    state.replies.extend(_make_reply("aria") for _ in range(5))

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.bargain_followup" not in due

    for arm in ("accepted", "refused", "hedged"):
        state.replies[-1].classifier_results = [
            AxisResult(quality="bargain_state", value=arm, rationale="")
        ]
        due = {b.id for b in find_due_beats(state, scenario)[0]}
        assert "aria.bargain_followup" in due, f"arm {arm!r} did not satisfy gate"


def test_bargain_followup_refused_arm_writes_grievance_floor(
    state: InboxState, scenario: Scenario
):
    """The followup's effect carries a `when: bargain_state == refused`
    clause that raises creature_grievance to 4.0 only on the refused arm."""
    from npc_service.spine import apply_literal_effects

    followup = scenario.beat_by_id["aria.bargain_followup"]

    # accepted arm: grievance untouched.
    s_accepted = InboxState.from_scenario(scenario)
    s_accepted.qualities["bargain_state"] = "accepted"
    apply_literal_effects(s_accepted, scenario, followup.spec.effects, beat_id=followup.id)
    assert s_accepted.qualities["creature_grievance"] == 0.0  # unchanged
    assert s_accepted.qualities["bargain_state"] == "accepted"

    # refused arm: grievance set to 4.0.
    s_refused = InboxState.from_scenario(scenario)
    s_refused.qualities["bargain_state"] = "refused"
    apply_literal_effects(s_refused, scenario, followup.spec.effects, beat_id=followup.id)
    assert s_refused.qualities["creature_grievance"] == 4.0
    assert s_refused.qualities["bargain_state"] == "refused"

    # hedged arm: grievance untouched.
    s_hedged = InboxState.from_scenario(scenario)
    s_hedged.qualities["bargain_state"] = "hedged"
    apply_literal_effects(s_hedged, scenario, followup.spec.effects, beat_id=followup.id)
    assert s_hedged.qualities["creature_grievance"] == 0.0


def test_bargain_betrayed_requires_followup_and_accepted_verdict(
    state: InboxState, scenario: Scenario
):
    """bargain_betrayed gates on bargain_followup having fired AND
    `bargain_state == accepted` (read directly via QualityEquals on the
    string-typed quality) PLUS recent cold replies. A refused or hedged
    player can't reach betrayal — that path is shape-incoherent."""
    _drain_boot(state, scenario)
    _seed_canon_name(state)
    state.fired_beat_ids.add("aria.bargain_offer")
    state.fired_beat_ids.add("aria.bargain_followup")
    state.replies.extend(_make_reply("aria") for _ in range(3))
    # Refused state with cold replies — betrayed should NOT fire.
    state.qualities["bargain_state"] = "refused"
    for r in state.replies[-2:]:
        r.classifier_results.append(
            AxisResult(quality="creator_affect", value="cold", rationale="")
        )
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.bargain_betrayed" not in due

    # Switch to accepted: with two cold trailing replies, betrayed is eligible.
    state.qualities["bargain_state"] = "accepted"
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.bargain_betrayed" in due


# ───────────────────────── bridge gating ─────────────────────────


def test_bridge_does_not_fire_on_warm_after_betrayal(
    state: InboxState, scenario: Scenario
):
    """aria.acted_in_player_name requires a recent cold reply — a player
    who pulls warm after bargain_betrayed should NOT trigger the bridge."""
    _drain_boot(state, scenario)
    state.fired_beat_ids.add("aria.bargain_betrayed")
    state.replies.extend(_make_reply("aria") for _ in range(5))
    state.qualities["creature_grievance"] = 6.0
    for r in state.replies[-2:]:
        r.classifier_results.append(
            AxisResult(quality="creator_affect", value="paternal", rationale="")
        )

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.acted_in_player_name" not in due

    cold_reply = _make_reply("aria")
    cold_reply.classifier_results.append(
        AxisResult(quality="creator_affect", value="cold", rationale="")
    )
    state.replies.append(cold_reply)
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.acted_in_player_name" in due


# ───────────────────────── helpers ─────────────────────────


def _drain_boot(state: InboxState, scenario: Scenario) -> None:
    for beat in find_due_beats(state, scenario)[0]:
        state.fired_beat_ids.add(beat.id)


def _make_reply(thread: str | None) -> Reply:
    return Reply(
        id=f"reply_{uuid.uuid4().hex[:6]}",
        thread_id=thread,
        beat_responded_to=None,
        recipient="Aria",
        body="(test reply)",
    )


def _seed_canon_name(state: InboxState, name: str = "Marcus") -> None:
    state.player_canon["name"] = PlayerCanonEntry(
        summary=name,
        literal_text_quote=f"I'm {name}",
        salience=0.9,
        category=CanonCategory.NAME,
        turn_of_origin=0,
    )
