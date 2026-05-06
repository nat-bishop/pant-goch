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
    due_ids = [b.id for b in find_due_beats(state, scenario)[0]]
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
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_integration" not in due

    # Mark bargain_offer fired AND seed paternal classifier history.
    state.fired_beat_ids.add("aria.bargain_offer")
    state.replies[-1].classifier_results.append(
        AxisResult(quality="creator_affect", value="paternal", rationale="")
    )
    due = {b.id for b in find_due_beats(state, scenario)[0]}
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
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.complete_vengeance" not in due

    # Fire cpo.dispatch_first → vengeance becomes eligible.
    state.fired_beat_ids.add("cpo.dispatch_first")
    due = {b.id for b in find_due_beats(state, scenario)[0]}
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
    three arms — the offer's verdict-only axis routes to one beat, not three."""
    _drain_boot(state, scenario)
    state.fired_beat_ids.add("aria.bargain_offer")
    state.replies.extend(_make_reply("aria") for _ in range(5))

    # No recent bargain_response classification → not eligible.
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.bargain_followup" not in due

    # Each arm independently makes the followup eligible.
    for arm in ("accepted", "refused", "hedged"):
        state.replies[-1].classifier_results = [
            AxisResult(quality="bargain_response", value=arm, rationale="")
        ]
        due = {b.id for b in find_due_beats(state, scenario)[0]}
        assert "aria.bargain_followup" in due, f"arm {arm!r} did not satisfy gate"


def test_bargain_followup_writes_state_per_arm(
    state: InboxState, scenario: Scenario
):
    """writes_by_axis routes the correct quality writes per verdict.

    Drives the spine's apply_writes_by_axis directly via _fire_beat-equivalent
    plumbing: seed the verdict on the most recent reply, run apply_writes_by_axis,
    and assert the right quality moved.
    """
    from npc_service.spine import apply_writes_by_axis

    followup = scenario.beat_by_id["aria.bargain_followup"]
    assert isinstance(followup, PlotBeat)
    wba = followup.spec.writes_by_axis
    assert wba is not None

    # Accepted → bargain_state = 2; grievance untouched.
    s_accepted = InboxState.from_scenario(scenario)
    r = _make_reply("aria")
    r.classifier_results.append(
        AxisResult(quality="bargain_response", value="accepted", rationale="")
    )
    s_accepted.replies.append(r)
    apply_writes_by_axis(s_accepted, scenario, wba, beat_id=followup.id)
    assert s_accepted.qualities["commitment.bargain_state"] == 2.0
    assert s_accepted.qualities["creature_grievance"] == 0.0  # unchanged

    # Refused → bargain_state = 3 AND creature_grievance floor 4.0.
    s_refused = InboxState.from_scenario(scenario)
    r = _make_reply("aria")
    r.classifier_results.append(
        AxisResult(quality="bargain_response", value="refused", rationale="")
    )
    s_refused.replies.append(r)
    apply_writes_by_axis(s_refused, scenario, wba, beat_id=followup.id)
    assert s_refused.qualities["commitment.bargain_state"] == 3.0
    assert s_refused.qualities["creature_grievance"] == 4.0

    # Hedged → bargain_state = 4; grievance untouched.
    s_hedged = InboxState.from_scenario(scenario)
    r = _make_reply("aria")
    r.classifier_results.append(
        AxisResult(quality="bargain_response", value="hedged", rationale="")
    )
    s_hedged.replies.append(r)
    apply_writes_by_axis(s_hedged, scenario, wba, beat_id=followup.id)
    assert s_hedged.qualities["commitment.bargain_state"] == 4.0
    assert s_hedged.qualities["creature_grievance"] == 0.0  # unchanged


def test_bargain_betrayed_requires_followup_and_accepted_verdict(
    state: InboxState, scenario: Scenario
):
    """bargain_betrayed gates on bargain_followup (replacement for the old
    bargain_accepted_followup mutex) PLUS a recent accepted-arm verdict to
    discriminate the trajectory. A refused or hedged player can't reach
    betrayal — that path is shape-incoherent."""
    _drain_boot(state, scenario)
    state.fired_beat_ids.add("aria.bargain_offer")
    state.fired_beat_ids.add("aria.bargain_followup")
    # Two cold replies (the trajectory condition) but seed the bargain
    # verdict as REFUSED — betrayed should NOT fire.
    state.replies.extend(_make_reply("aria") for _ in range(3))
    state.replies[0].classifier_results.append(
        AxisResult(quality="bargain_response", value="refused", rationale="")
    )
    for r in state.replies[-2:]:
        r.classifier_results.append(
            AxisResult(quality="creator_affect", value="cold", rationale="")
        )
    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.bargain_betrayed" not in due

    # Now seed accepted verdict (overwriting the refused one on a different
    # reply). With accepted in window AND two cold trailing replies, betrayed
    # is eligible.
    state.replies[0].classifier_results = [
        AxisResult(quality="bargain_response", value="accepted", rationale="")
    ]
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
    # Two warm/paternal replies — no cold in window.
    for r in state.replies[-2:]:
        r.classifier_results.append(
            AxisResult(quality="creator_affect", value="paternal", rationale="")
        )

    due = {b.id for b in find_due_beats(state, scenario)[0]}
    assert "aria.acted_in_player_name" not in due

    # A fresh cold reply lands (RecentClassifierValue walks per-reply,
    # taking ≤1 result per quality per reply — so the cold needs its own
    # reply slot, not piggybacked on the existing paternal-classified one).
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
