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
    assert {n.id for n in scenario.npcs} == {"aria", "cpo", "marcus"}
    assert {t.id for t in scenario.threads} == {"aria", "cpo", "marcus"}
    assert {q.id for q in scenario.qualities} >= {
        "creator_affect",
        "creature_grievance",
        "bargain_state",
        "aria_acted_in_player_name",
    }
    expected_aria_beats = {
        "aria.welcome",
        "aria.calibration_scene",
        "aria.bargain_scene",
        "aria.recognition_scene",
        "aria.acted_in_player_name",
        "aria.complete_integration",
        "aria.complete_mutual_ruin",
        "aria.complete_exile",
        "aria.complete_vengeance",
    }
    assert expected_aria_beats <= {b.id for b in scenario.beats}
    assert "cpo.dispatch_first" in {b.id for b in scenario.beats}
    assert "marcus.intro_scene" in {b.id for b in scenario.beats}
    assert "marcus.advice_scene" in {b.id for b in scenario.beats}


def test_email_domain_required(scenario: Scenario):
    """Production scenario uses email_domain (not the legacy player_email).
    Recipient resolution is dynamic — `<canonized_name>@<email_domain>` if a
    name has been extracted, else `<fallback_player_local>@<email_domain>`."""
    assert scenario.email_domain == "halcyon.com"
    assert scenario.player_email is None


def test_calibration_scene_exits_via_exit_when(scenario: Scenario):
    """`aria.calibration_scene` is kind=scene with `exit_when` carrying the
    *calibration_done predicate. Its preconditions do NOT carry a
    Not_(*calibration_done) gate any longer — that responsibility moved to
    exit_when."""
    from npc_service.models import Not_, Or_, PlayerCanonExists, PlayerRepliesToThreadGte

    cal = scenario.beat_by_id["aria.calibration_scene"]
    assert cal.kind == "scene"
    assert cal.spec.exit_when, "scene must have exit_when"
    # exit_when contains the calibration_done OR predicate.
    or_exits = [p for p in cal.spec.exit_when if isinstance(p, Or_)]
    assert or_exits, "calibration_scene exit_when should include the Or_ calibration_done"
    inner_kinds = {type(s) for s in or_exits[0].of}
    assert PlayerCanonExists in inner_kinds
    assert PlayerRepliesToThreadGte in inner_kinds
    # And the precondition list does NOT contain Not_(*calibration_done).
    not_preds = [p for p in cal.preconditions if isinstance(p, Not_)]
    for n in not_preds:
        if isinstance(n.of, Or_):
            inner = {type(s) for s in n.of.of}
            assert not (PlayerCanonExists in inner and PlayerRepliesToThreadGte in inner), (
                "calibration_scene should NOT have Not_(*calibration_done) "
                "in preconditions; that lives in exit_when now."
            )


def test_phase2_body_beats_gate_on_calibration_done(scenario: Scenario):
    """Every Phase-2 body beat (bargain scene, recognition scene, bridge,
    terminals) must carry an `Or_(PlayerCanonExists("name"),
    PlayerRepliesToThreadGte(...))` in its preconditions — the YAML anchor
    expansion at load time."""
    from npc_service.models import Or_, PlayerCanonExists, PlayerRepliesToThreadGte

    phase2 = [
        "aria.bargain_scene",
        "aria.recognition_scene",
        "aria.acted_in_player_name",
        "cpo.dispatch_first",
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


def test_calibration_scene_has_two_classify_effects(scenario: Scenario):
    """`aria.calibration_scene` has classify effects on both creator_affect
    and creature_grievance — they ride on every reply during calibration."""
    cal = scenario.beat_by_id["aria.calibration_scene"]
    assert cal.kind == "scene"
    classify_effects = [e for e in cal.spec.effects if e.is_classify]
    quality_targets = sorted(e.quality for e in classify_effects)
    assert quality_targets == ["creator_affect", "creature_grievance"]


def test_bargain_scene_writes_state_and_classifies_into_string_quality(scenario: Scenario):
    """The bargain scene literal-writes `bargain_state="offered"` AND
    schedules a classifier directive that writes the verdict string
    (accepted/refused/hedged) directly into the same string-typed quality.
    Replaces the old verdict-only `axis_name: bargain_response` pattern."""
    bargain = scenario.beat_by_id["aria.bargain_scene"]
    assert bargain.kind == "scene"
    literal_effects = [e for e in bargain.spec.effects if not e.is_classify]
    classify_effects = [e for e in bargain.spec.effects if e.is_classify]

    # At least one literal "offered" write — extra literal effects (e.g.
    # bargain_replies counter) are allowed.
    offered_writes = [
        e for e in literal_effects
        if e.quality == "bargain_state" and e.value == "offered"
    ]
    assert offered_writes, "bargain_scene must literal-write bargain_state=offered"

    bargain_state_classify = [
        e for e in classify_effects if e.quality == "bargain_state"
    ]
    assert len(bargain_state_classify) == 1
    classify = bargain_state_classify[0]
    assert classify.classify is not None
    assert set(classify.classify.allowed_values) >= {"accepted", "refused", "hedged"}
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


def test_integration_eligible_only_after_bargain_scene_and_paternal_window(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    _seed_canon_name(state)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 1.0

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_integration" not in due

    state.fired_beat_ids.add("aria.bargain_scene")
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="paternal", rationale="")
    )
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_integration" in due


def test_vengeance_requires_bridge_and_cpo(
    state: InboxState, scenario: Scenario
):
    _drain_boot(state, scenario)
    _seed_canon_name(state)
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
    _seed_canon_name(state)
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
    _seed_canon_name(state)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 7.0
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="paternal", rationale="")
    )
    state.fired_beat_ids.add("aria.bargain_scene")

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_mutual_ruin" in due


# ───────────────────────── bridge gating ─────────────────────────


def test_bridge_does_not_fire_on_warm_after_betrayal(
    state: InboxState, scenario: Scenario
):
    """aria.acted_in_player_name requires a recent cold reply — a player
    who pulls warm after aria.bargain_betrayed should NOT trigger the bridge."""
    _drain_boot(state, scenario)
    _seed_canon_name(state)
    state.fired_beat_ids.add("aria.bargain_scene")
    state.fired_beat_ids.add("aria.bargain_betrayed")
    state.replies.extend(_make_reply("aria") for _ in range(5))
    state.qualities["creature_grievance"] = 6.0
    state.qualities["bargain_state"] = "betrayed"
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
