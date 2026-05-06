"""Spine-level tests for the production scenario (Frankenstein substrate).

No LLM here — covers what the deterministic spine reports as eligible given
synthesized state. Director / epilogue agent behavior is exercised in
test_app.py (which uses TestModel)."""
import uuid
from pathlib import Path

import pytest

from npc_service.models import (
    AxisResult,
    InboxState,
    PlotBeat,
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
        "commitment.bargain_state",
        "commitment.aria_acted_in_player_name",
    }
    assert {b.id for b in scenario.beats} == {
        "aria.welcome",
        "aria.calibration",
        "aria.early_emergence",
        "aria.naming_demand",
        "aria.bargain_offer",
        "aria.bargain_accepted_followup",
        "aria.bargain_refused_followup",
        "aria.bargain_hedged_followup",
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
    preconditions — that's how it self-stops without using completes_when.
    The calibration_done predicate is `Or_(PlayerCanonExists("name"),
    PlayerRepliesToThreadGte("aria", 4))`."""
    from npc_service.models import Not_, Or_, PlayerCanonExists, PlayerRepliesToThreadGte

    cal = scenario.beat_by_id["aria.calibration"]
    assert isinstance(cal, PlotBeat)
    assert cal.multi_turn is True
    # Find the Not_(Or_(...)) precondition.
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
        "aria.bargain_accepted_followup",
        "aria.bargain_refused_followup",
        "aria.bargain_hedged_followup",
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
        # Every one of these starts with the anchor expansion.
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


def test_loop_has_three_axes_including_register(scenario: Scenario):
    loop = scenario.beat_by_id["aria.loop"]
    assert isinstance(loop, PlotBeat)
    assert loop.multi_turn is True
    identifiers = [a.identifier for a in loop.classify_reply_axes]
    assert identifiers == ["creator_affect", "creature_grievance", "register"]
    # `register` is a verdict-only axis — quality is None, axis_name is "register".
    register_axis = loop.classify_reply_axes[2]
    assert register_axis.quality is None
    assert register_axis.axis_name == "register"


def test_bargain_offer_uses_verdict_only_axis(scenario: Scenario):
    offer = scenario.beat_by_id["aria.bargain_offer"]
    assert isinstance(offer, PlotBeat)
    assert len(offer.classify_reply_axes) == 1
    axis = offer.classify_reply_axes[0]
    assert axis.quality is None
    assert axis.axis_name == "bargain_response"


def test_terminals_carry_validate_facts(scenario: Scenario):
    for tid in (
        "aria.complete_integration",
        "aria.complete_mutual_ruin",
        "aria.complete_exile",
        "aria.complete_vengeance",
    ):
        beat = scenario.beat_by_id[tid]
        assert isinstance(beat, PlotBeat)
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
    due_ids = [b.id for b in find_due_beats(state, scenario)]
    assert due_ids == ["aria.welcome"]


# ───────────────────────── terminal eligibility ─────────────────────────


def test_integration_eligible_only_after_bargain_offer_and_paternal_window(
    state: InboxState, scenario: Scenario
):
    """complete_integration requires: 8 replies + paternal-in-last-3 +
    grievance < 3.0 + bargain_offer fired."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 1.0

    # No bargain fired AND no paternal in recent — not eligible.
    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.complete_integration" not in due

    # Mark bargain_offer fired AND seed paternal classifier history.
    state.fired_beat_ids.add("aria.bargain_offer")
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="paternal", rationale="")
    )
    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.complete_integration" in due


def test_vengeance_requires_bridge_and_cpo(
    state: InboxState, scenario: Scenario
):
    """complete_vengeance gates on cpo.dispatch_first having fired AND
    grievance >= 6.0 AND a recent cold reply. Bridge (aria.acted_in_player_name)
    is the precondition for cpo.dispatch_first becoming eligible — so the
    chain must traverse fully."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 7.0
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="cold", rationale="")
    )

    # Without cpo.dispatch_first fired, vengeance isn't eligible.
    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.complete_vengeance" not in due

    # Fire cpo.dispatch_first → vengeance becomes eligible.
    state.fired_beat_ids.add("cpo.dispatch_first")
    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.complete_vengeance" in due


def test_exile_eligible_when_cold_window_low_grievance(
    state: InboxState, scenario: Scenario
):
    """complete_exile: 8 replies + last-3-cold (any) + grievance < 4.0.
    No bargain or CPO required."""
    _drain_boot(state, scenario)
    state.replies.extend(_make_reply("aria") for _ in range(8))
    state.qualities["creature_grievance"] = 1.0
    # Three cold replies in a row to satisfy the recent-cold window.
    for r in state.replies[-3:]:
        r.classifier_results.append(
            AxisResult(quality="creator_affect", value="cold", rationale="")
        )

    due = {b.id for b in find_due_beats(state, scenario)}
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

    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.complete_mutual_ruin" in due


# ───────────────────────── bargain followups via verdict axis ─────────────────────────


def test_bargain_accepted_followup_requires_recent_accepted(
    state: InboxState, scenario: Scenario
):
    """The accepted followup gates on RecentClassifierValue against the
    bargain_response axis_name — the engine's verdict-only path."""
    _drain_boot(state, scenario)
    state.fired_beat_ids.add("aria.bargain_offer")
    state.replies.extend(_make_reply("aria") for _ in range(5))

    # No recent bargain_response classification → not eligible.
    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.bargain_accepted_followup" not in due

    # Land "accepted" on the most recent reply's classifier_results.
    state.replies[-1].classifier_results.append(
        AxisResult(quality="bargain_response", value="accepted", rationale="")
    )
    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.bargain_accepted_followup" in due


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
    # Two warm/paternal replies — no cold in window.
    for r in state.replies[-2:]:
        r.classifier_results.append(
            AxisResult(quality="creator_affect", value="paternal", rationale="")
        )

    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.acted_in_player_name" not in due

    # A fresh cold reply lands (RecentClassifierValue walks per-reply,
    # taking ≤1 result per quality per reply — so the cold needs its own
    # reply slot, not piggybacked on the existing paternal-classified one).
    cold_reply = _make_reply("aria")
    cold_reply.classifier_results.append(
        AxisResult(quality="creator_affect", value="cold", rationale="")
    )
    state.replies.append(cold_reply)
    due = {b.id for b in find_due_beats(state, scenario)}
    assert "aria.acted_in_player_name" in due


# ───────────────────────── helpers ─────────────────────────


def _drain_boot(state: InboxState, scenario: Scenario) -> None:
    for beat in find_due_beats(state, scenario):
        state.fired_beat_ids.add(beat.id)


def _make_reply(thread: str | None) -> Reply:
    return Reply(
        id=f"reply_{uuid.uuid4().hex[:6]}",
        thread_id=thread,
        beat_responded_to=None,
        recipient="Aria",
        body="(test reply)",
    )
