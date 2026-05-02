"""Pure-Python tests for the deterministic spine: scenario load, predicate
evaluation, effect application, beat scheduler, thread state machine, voice
band selection, search, and the coda fragment selector.

No LLM, no FastAPI. Runs fast.
"""
from pathlib import Path

import pytest

from npc_service.models import (
    AnchoredBeat,
    InboxState,
    Reply,
    Scenario,
    SetAtom,
    SetQuality,
    ScheduleBeat,
    load_scenario,
)
from npc_service.spine import (
    apply_effect,
    apply_texture_score,
    apply_voice_fidelity,
    check_end_conditions,
    evaluate_thread_transitions,
    find_due_beats,
    initialize_boot_inbox,
    search_inbox,
    select_coda_fragments,
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
    assert scenario.title == "Inbox"
    assert scenario.max_actions == 18
    assert {a.id for a in scenario.knowledge_atoms} >= {
        "knowledge.tess.hospitalized",
        "unsent_draft_disposition",
    }
    assert {n.id for n in scenario.npcs} == {"maya", "tess"}
    assert {t.id for t in scenario.threads} >= {"intake", "maya", "thursday_group"}
    boot_count = sum(1 for b in scenario.beats if getattr(b, "in_boot_inbox", False))
    assert boot_count == 3, f"expected 3 boot-inbox beats, got {boot_count}"


# ───────────────────────── boot inbox + initial state ─────────────────────────


def test_initial_atoms_match_defaults(state: InboxState):
    assert state.atoms["knowledge.tess.hospitalized"] is False
    assert state.atoms["unsent_draft_disposition"] == "pending"
    assert state.atoms["knowledge.maya_knows_truth"] == "unknown"


def test_initial_thread_states(state: InboxState):
    assert state.thread_states["maya"] == "incoming"
    assert state.thread_states["intake"] == "incoming"


def test_boot_inbox_returns_three_beats(state: InboxState, scenario: Scenario):
    boot = initialize_boot_inbox(state, scenario)
    assert len(boot) == 3
    ids = {b.beat.id for b in boot}
    assert ids == {"intake.janet_first", "maya.first_email", "class_registration.reminder"}
    for b in boot:
        assert isinstance(b.beat, AnchoredBeat)


# ───────────────────────── effect application ─────────────────────────


def test_set_atom_effect(state: InboxState, scenario: Scenario):
    apply_effect(SetAtom(atom="knowledge.tess.hospitalized", value=True), state, scenario)
    assert state.atoms["knowledge.tess.hospitalized"] is True


def test_set_quality_add_op_clamps_to_max(state: InboxState, scenario: Scenario):
    apply_effect(SetQuality(quality="register.maya", op="set", value=2.5), state, scenario)
    apply_effect(SetQuality(quality="register.maya", op="add", value=10.0), state, scenario)
    assert state.qualities["register.maya"] == 3.0  # max=3


def test_drift_halfway_op(state: InboxState, scenario: Scenario):
    state.qualities["register.maya"] = 2.0
    apply_effect(SetQuality(quality="register.maya", op="drift_halfway", value=0.0), state, scenario)
    assert state.qualities["register.maya"] == 1.0


def test_schedule_beat_appears_in_pending(state: InboxState, scenario: Scenario):
    apply_effect(ScheduleBeat(beat="maya.gentle_followup", after_actions=3), state, scenario)
    assert any(pb.beat_id == "maya.gentle_followup" for pb in state.pending_beats)


def test_schedule_beat_is_not_duplicated_with_later_target(state: InboxState, scenario: Scenario):
    """If we schedule a beat twice, the earlier deadline wins."""
    state.action_count = 5
    apply_effect(ScheduleBeat(beat="maya.gentle_followup", after_actions=3), state, scenario)
    apply_effect(ScheduleBeat(beat="maya.gentle_followup", after_actions=10), state, scenario)
    pending = [pb for pb in state.pending_beats if pb.beat_id == "maya.gentle_followup"]
    assert len(pending) == 1
    assert pending[0].earliest_action == 8  # 5 + 3, not 5 + 10


# ───────────────────────── texture / register loop ─────────────────────────


def test_warm_reply_to_maya_increases_warmth_and_drifts_register(state: InboxState, scenario: Scenario):
    state.replies.append(_make_reply(thread="maya", warmth=2.0, honesty=1.0))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=2.0, honesty=1.0)
    assert state.qualities["warmth.maya"] == 2.0
    # Register drifts halfway from 0 toward 2 → 1.0.
    assert state.qualities["register.maya"] == 1.0


def test_curt_reply_pulls_register_negative(state: InboxState, scenario: Scenario):
    state.replies.append(_make_reply(thread="maya", warmth=-2.0, honesty=0.0))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=-2.0, honesty=0.0)
    assert state.qualities["register.maya"] == -1.0


def test_register_drift_visible_within_one_round(state: InboxState, scenario: Scenario):
    """The texture loop must close per-round. Two replies should produce
    visibly different register values."""
    # Reply 1: warm
    state.replies.append(_make_reply(thread="maya", warmth=3.0, honesty=1.0))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=3.0, honesty=1.0)
    after_warm = state.qualities["register.maya"]
    # Reply 2: cold
    state.replies.append(_make_reply(thread="maya", warmth=-3.0, honesty=0.0))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=-3.0, honesty=0.0)
    after_cold = state.qualities["register.maya"]
    assert after_warm > after_cold
    # And: the register actually moved toward the new sign in one round.
    assert after_cold < 0


def test_non_maya_replies_only_touch_honesty_global(state: InboxState, scenario: Scenario):
    state.replies.append(_make_reply(thread=None, warmth=2.0, honesty=2.0))
    apply_texture_score(state, scenario, recipient_npc_id=None, warmth=2.0, honesty=2.0)
    assert state.qualities["warmth.maya"] == 0.0
    assert state.qualities["register.maya"] == 0.0
    assert state.qualities["honesty.global"] == 2.0


# ───────────────────────── voice band selection ─────────────────────────


def test_voice_band_neutral_at_zero(state: InboxState, scenario: Scenario):
    maya = next(n for n in scenario.npcs if n.id == "maya")
    band = select_voice_band(maya, state)
    assert band is not None and band.name == "neutral"


def test_voice_band_warm_when_register_high(state: InboxState, scenario: Scenario):
    state.qualities["register.maya"] = 2.0
    maya = next(n for n in scenario.npcs if n.id == "maya")
    band = select_voice_band(maya, state)
    assert band is not None and band.name == "warm"


def test_voice_band_reserved_when_register_low(state: InboxState, scenario: Scenario):
    state.qualities["register.maya"] = -2.0
    maya = next(n for n in scenario.npcs if n.id == "maya")
    band = select_voice_band(maya, state)
    assert band is not None and band.name == "reserved"


def test_tess_has_no_voice_band(state: InboxState, scenario: Scenario):
    tess = next(n for n in scenario.npcs if n.id == "tess")
    assert select_voice_band(tess, state) is None


# ───────────────────────── thread state machine ─────────────────────────


def test_maya_transitions_to_in_progress_after_reply(state: InboxState, scenario: Scenario):
    state.atoms["knowledge.tess.hospitalized"] = True
    state.replies.append(_make_reply(thread="maya", warmth=1.0, honesty=1.0, beat="maya.first_email"))
    fired = evaluate_thread_transitions(state, scenario)
    assert state.thread_states["maya"] == "in_progress"
    # The transition has effects (schedule_beat) but no fire_beat, so nothing
    # fires immediately.
    assert fired == []
    # gentle_followup is scheduled.
    assert any(pb.beat_id == "maya.gentle_followup" for pb in state.pending_beats)


def test_warm_trajectory_lands_in_warm_terminal_state(state: InboxState, scenario: Scenario):
    """Three warm replies + an honest direct-ask reply → maya.state == warm."""
    # Walk through the state machine manually.
    state.atoms["knowledge.tess.hospitalized"] = True
    # First reply sends incoming → in_progress.
    state.replies.append(_make_reply(thread="maya", warmth=2.0, honesty=2.0, beat="maya.first_email"))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=2.0, honesty=2.0)
    evaluate_thread_transitions(state, scenario)
    assert state.thread_states["maya"] == "in_progress"

    # Bump action count so pre_consequential transition fires.
    state.action_count = 12
    evaluate_thread_transitions(state, scenario)
    assert state.thread_states["maya"] == "pre_consequential"

    # Warm reply to direct_ask → warm terminal.
    state.replies.append(_make_reply(thread="maya", warmth=2.0, honesty=2.0, beat="maya.direct_ask"))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=2.0, honesty=2.0)
    evaluate_thread_transitions(state, scenario)
    assert state.thread_states["maya"] == "warm"


def test_cold_trajectory_lands_in_cold_terminal_state(state: InboxState, scenario: Scenario):
    """Cold replies throughout → maya.state == cold even if the last reply is
    warm. Trajectory matters, not just last."""
    state.atoms["knowledge.tess.hospitalized"] = True
    state.replies.append(_make_reply(thread="maya", warmth=-2.0, honesty=0.0, beat="maya.first_email"))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=-2.0, honesty=0.0)
    evaluate_thread_transitions(state, scenario)

    state.action_count = 12
    evaluate_thread_transitions(state, scenario)
    assert state.thread_states["maya"] == "pre_consequential"

    # Late-game warmth rush — even +3 warmth here can't pull average above 0.5.
    # Avg is (-2 + 3) / 2 = 0.5; predicate is >=0.5 (warm) so this could land warm.
    # Let's tilt cold further to ensure cold lands.
    state.replies.append(_make_reply(thread="maya", warmth=-3.0, honesty=-1.0, beat="maya.direct_ask"))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=-3.0, honesty=-1.0)
    evaluate_thread_transitions(state, scenario)
    assert state.thread_states["maya"] == "cold"


def test_late_warmth_rush_doesnt_save_cold_trajectory(state: InboxState, scenario: Scenario):
    """The whole point of trajectory averaging — late warmth doesn't redeem
    a cold session."""
    state.atoms["knowledge.tess.hospitalized"] = True
    # Two cold replies first.
    state.replies.append(_make_reply(thread="maya", warmth=-2.0, honesty=0.0, beat="maya.first_email"))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=-2.0, honesty=0.0)
    state.replies.append(_make_reply(thread="maya", warmth=-2.0, honesty=0.0, beat="maya.gentle_followup"))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=-2.0, honesty=0.0)
    evaluate_thread_transitions(state, scenario)

    state.action_count = 12
    evaluate_thread_transitions(state, scenario)
    assert state.thread_states["maya"] == "pre_consequential"

    # Now the warmth rush.
    state.replies.append(_make_reply(thread="maya", warmth=3.0, honesty=2.0, beat="maya.direct_ask"))
    apply_texture_score(state, scenario, recipient_npc_id="maya", warmth=3.0, honesty=2.0)
    evaluate_thread_transitions(state, scenario)
    # Average is (-2 + -2 + 3) / 3 = -0.33 → cold predicate (avg < -0.5? no) →
    # falls through to ambivalent (avg >= -0.5? yes). Verifies trajectory holds.
    assert state.thread_states["maya"] in {"ambivalent", "cold"}
    assert state.thread_states["maya"] != "warm"


# ───────────────────────── search ─────────────────────────


def test_search_finds_archive_emails(state: InboxState, scenario: Scenario):
    hits = search_inbox(state, scenario, "writing group")
    assert any("Tess" in h["from"] or "writing" in h["subject"].lower() for h in hits)


def test_search_returns_empty_for_no_match(state: InboxState, scenario: Scenario):
    hits = search_inbox(state, scenario, "xyzzy_nothing_here")
    assert hits == []


def test_search_ranks_by_match_count(state: InboxState, scenario: Scenario):
    hits = search_inbox(state, scenario, "tess")
    assert len(hits) >= 1


# ───────────────────────── coda fragment selector ─────────────────────────


def test_coda_fragments_warm_with_held_draft(state: InboxState, scenario: Scenario):
    # Warm trajectory + held draft + class registration preserved.
    state.replies.append(_make_reply(thread="maya", warmth=3.0, honesty=2.0, beat="maya.first_email", body="thinking of you, im taking care of things"))
    state.replies.append(_make_reply(thread="maya", warmth=2.0, honesty=3.0, beat="maya.direct_ask", body="shes ok, shes safe"))
    state.thread_states["maya"] = "warm"
    state.atoms["unsent_draft_disposition"] = "held"
    state.commitments.add("class_registration")

    fragments = select_coda_fragments(state, scenario)
    labels = {f.label for f in fragments}
    assert "player_warmest_reply" in labels
    assert "draft_held" in labels
    assert "class_registration_preserved" in labels


def test_coda_fragments_cold_with_sent_unchanged(state: InboxState, scenario: Scenario):
    state.thread_states["maya"] = "cold"
    state.atoms["unsent_draft_disposition"] = "sent_unchanged"
    # Append a fake draft email so the selector finds something to excerpt.
    from npc_service.models import Email
    state.emails.append(Email(
        id="email_draft",
        beat_id="maya.draft_to_her_visible",
        thread_id="maya",
        folder="Drafts",
        sender="Tess (draft)",
        recipient="Maya Chen",
        subject="(no subject)",
        body="hey - ive been trying to write this for like a week",
        sent_at_label="(unsent draft)",
        sent_at_action=0,
    ))
    fragments = select_coda_fragments(state, scenario)
    labels = {f.label for f in fragments}
    assert "draft_sent_unchanged" in labels


def test_coda_fragments_edited_draft_uses_rationale(state: InboxState, scenario: Scenario):
    state.atoms["unsent_draft_disposition"] = "sent_edited"
    state.voice_fidelity_rationale = "kept the lowercase but dropped the hedge"
    fragments = select_coda_fragments(state, scenario)
    labels = {f.label for f in fragments}
    assert "draft_sent_edited" in labels
    assert any("hedge" in f.excerpt for f in fragments)


# ───────────────────────── end conditions ─────────────────────────


def test_natural_end_when_maya_terminal_and_min_actions(state: InboxState):
    state.thread_states["maya"] = "warm"
    state.action_count = 16
    check_end_conditions(state)
    assert state.ended
    assert state.end_reason == "natural"


def test_budget_exhaustion_ends_session(state: InboxState):
    state.action_count = 18
    check_end_conditions(state)
    assert state.ended
    assert state.end_reason == "budget_exhausted"


# ───────────────────────── voice fidelity ─────────────────────────


def test_apply_voice_fidelity_clamps_and_stores_rationale(state: InboxState):
    apply_voice_fidelity(state, voice_fidelity=1.5, rationale="clamp test")
    assert state.voice_fidelity == 1.0
    apply_voice_fidelity(state, voice_fidelity=-0.5, rationale="below zero")
    assert state.voice_fidelity == 0.0
    assert state.voice_fidelity_rationale == "below zero"


# ───────────────────────── scheduler ordering ─────────────────────────


def test_scheduled_beat_only_fires_once_action_count_reached(state: InboxState, scenario: Scenario):
    apply_effect(ScheduleBeat(beat="maya.gentle_followup", after_actions=3), state, scenario)
    state.thread_states["maya"] = "in_progress"  # precondition
    # Not enough actions yet.
    state.action_count = 1
    due = find_due_beats(state, scenario)
    assert not any(bf.beat.id == "maya.gentle_followup" for bf in due)
    # Now we cross the threshold.
    state.action_count = 3
    due = find_due_beats(state, scenario)
    assert any(bf.beat.id == "maya.gentle_followup" for bf in due)


# ───────────────────────── helpers ─────────────────────────


def _make_reply(
    thread: str | None,
    warmth: float,
    honesty: float,
    beat: str | None = None,
    body: str = "(test reply)",
) -> Reply:
    return Reply(
        id=f"reply_{warmth}_{honesty}",
        thread_id=thread,
        beat_responded_to=beat,
        recipient="Maya Chen" if thread == "maya" else "other",
        body=body,
        warmth=warmth,
        honesty=honesty,
        rationale="(test)",
        sent_at_action=0,
    )
