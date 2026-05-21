"""Prompt builders.

Verifies KV-cache discipline (per-agent system prompts byte-stable), the
Phase 1 / prose writer / choice author split (prose writer sees voice
samples + few-shots but NOT beats / notes_to_llm / NPC knowledge /
endings; choice author sees neither bible nor voice), and per-turn
user-prompt shape for each.
"""
from __future__ import annotations

from pathlib import Path

from handler.prompts import (
    build_choice_author_system_prompt,
    build_choice_author_user_message,
    build_chronicler_system_prompt,
    build_caption_author_system_prompt,
    build_caption_author_user_message,
    build_free_text_gate_system_prompt,
    build_free_text_gate_user_message,
    build_phase1_system_prompt,
    build_phase1_user_prompt,
    build_prose_writer_system_prompt,
    build_prose_writer_user_message,
)
from handler.schemas import Option, StateChangeBundle, UnlockBeat
from handler.state import TurnRecord, make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── system prompts: byte stability ─────────────────────────


def test_phase1_system_prompt_is_byte_stable():
    world = load_world(FIXTURE_VALID)
    a = build_phase1_system_prompt(world)
    b = build_phase1_system_prompt(world)
    assert a == b


def test_prose_writer_system_prompt_is_byte_stable():
    world = load_world(FIXTURE_VALID)
    a = build_prose_writer_system_prompt(world)
    b = build_prose_writer_system_prompt(world)
    assert a == b


def test_choice_author_system_prompt_is_byte_stable():
    world = load_world(FIXTURE_VALID)
    a = build_choice_author_system_prompt(world)
    b = build_choice_author_system_prompt(world)
    assert a == b


# ───────────────────────── phase 1 system prompt: world surface + crime only ─────────────────────────


def test_phase1_system_prompt_carries_world_surface_and_crime_only():
    """Phase 1 system prompt holds locations + cast (with knowledge) + the
    slim ## What Happened crime narrative. Beats and endings are NOT here —
    beats live in the per-turn user prompt's ## Reachable Beats, endings
    nowhere (engine-only data)."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_phase1_system_prompt(world)
    for loc in world.locations.values():
        assert loc.id in sys_prompt
        assert loc.name in sys_prompt
    for npc in world.npcs.values():
        assert npc.id in sys_prompt
        assert npc.name in sys_prompt
        assert npc.knowledge.strip() in sys_prompt
    # Crime narrative under ## What Happened.
    assert "## What Happened" in sys_prompt
    assert world.crime.strip() in sys_prompt
    assert "Megan Pritchard" in sys_prompt
    # Beats: NOT in the system prompt (they live in the user prompt only).
    for beat in world.beats.values():
        assert beat.notes_to_llm.strip() not in sys_prompt, (
            f"phase1 system leaks beat notes_to_llm {beat.id!r}"
        )
    # Endings: removed everywhere.
    for ending in world.endings.values():
        assert ending.id not in sys_prompt, (
            f"phase1 system leaks ending id {ending.id!r}"
        )
    assert "## The Truth" not in sys_prompt


# ───────────────────────── prose writer system prompt: slim, voice-focused ─────────────────────────


def test_prose_writer_system_prompt_excludes_beats_and_notes_to_llm():
    """The prose writer must NOT see beat ids, notes_to_llm, or beat
    structure. Voice attention; beat-aware prose was the failure mode
    this split fixes."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    for beat in world.beats.values():
        assert beat.id not in sys_prompt, (
            f"prose writer prompt leaks beat id {beat.id!r}"
        )
        assert beat.notes_to_llm.strip() not in sys_prompt
    assert "## The Truth" not in sys_prompt
    assert "requires_beats" not in sys_prompt
    assert "blocked_by_beats" not in sys_prompt


def test_prose_writer_system_prompt_excludes_endings():
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    for ending in world.endings.values():
        assert ending.id not in sys_prompt, (
            f"prose writer prompt leaks ending id {ending.id!r}"
        )


def test_prose_writer_system_prompt_excludes_npc_knowledge():
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    tom = world.npcs["tom"]
    assert tom.knowledge.strip() not in sys_prompt


def test_prose_writer_system_prompt_includes_crime_overview():
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    assert world.crime.strip() in sys_prompt
    assert "## What Happened" in sys_prompt


def test_prose_writer_system_prompt_includes_voice_samples_when_authored():
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    any_voice = any(npc.voice_sample for npc in world.npcs.values())
    if any_voice:
        for npc in world.npcs.values():
            for line in npc.voice_sample:
                assert line.strip() in sys_prompt
    for npc in world.npcs.values():
        assert npc.id in sys_prompt


def test_prose_writer_system_prompt_includes_few_shots_when_authored():
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    assert "## Style Examples" in sys_prompt
    if world.few_shots:
        for fs in world.few_shots:
            assert fs.setup.strip() in sys_prompt
            assert fs.player.strip() in sys_prompt
    else:
        assert "(no examples authored)" in sys_prompt


# ───────────────────────── choice author system prompt: slim, no bible ─────────────────────────


def test_choice_author_system_prompt_excludes_bible():
    """The choice author sees NO bible — its context is the just-written
    prose, supplied in the per-call user message."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_choice_author_system_prompt(world)
    for beat in world.beats.values():
        assert beat.id not in sys_prompt
        assert beat.name not in sys_prompt
    for ending in world.endings.values():
        assert ending.id not in sys_prompt
    for npc in world.npcs.values():
        assert npc.knowledge.strip() not in sys_prompt
    assert "## The Truth" not in sys_prompt
    assert "## What Happened" not in sys_prompt
    assert "## Style Examples" not in sys_prompt


def test_choice_author_system_prompt_describes_variants():
    """The choice author's vocabulary is now Continue | Leave only —
    FreeText moved to the free-text gate + engine."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_choice_author_system_prompt(world)
    assert "Continue" in sys_prompt
    assert "Leave" in sys_prompt
    assert "FreeText" not in sys_prompt


def test_free_text_gate_system_prompt_is_byte_stable_and_minimal():
    """Byte-stable for KV-cache; mechanics-only (no bible, no voice)."""
    world = load_world(FIXTURE_VALID)
    a = build_free_text_gate_system_prompt(world)
    b = build_free_text_gate_system_prompt(world)
    assert a == b
    assert "offer" in a
    assert "## The Truth" not in a
    assert "## Style Examples" not in a


def test_free_text_gate_user_message_is_prose_only():
    """Pure-prose judgment: the prose is the only context."""
    world = load_world(FIXTURE_VALID)
    msg = build_free_text_gate_user_message(
        make_initial_state(world), world, "Tom sets down his pint.",
    )
    assert "Tom sets down his pint." in msg
    assert "FreeTextGate" in msg


def test_caption_author_system_prompt_is_byte_stable_and_minimal():
    """Byte-stable for KV-cache; mechanics-only (no bible, no voice)."""
    world = load_world(FIXTURE_VALID)
    a = build_caption_author_system_prompt(world)
    b = build_caption_author_system_prompt(world)
    assert a == b
    assert "caption" in a
    assert "suggested" in a
    assert "## The Truth" not in a
    assert "## Style Examples" not in a


def test_caption_author_user_message_is_prose_only():
    """Pure-prose judgment: the prose is the only context."""
    world = load_world(FIXTURE_VALID)
    msg = build_caption_author_user_message(
        make_initial_state(world), world, "Tom sets down his pint.",
    )
    assert "Tom sets down his pint." in msg
    assert "FreeTextCaption" in msg


def test_prose_writer_user_message_distinguishes_typed_from_menu_pick():
    """A typed free-text line is rendered as the player's own words; a
    clicked option is rendered as an intent to dramatize. Never 'said'."""
    world = load_world(FIXTURE_VALID)
    applied = StateChangeBundle()

    # Typed (verbatim).
    state = make_initial_state(world)
    state.dialogue_history.append(TurnRecord(
        prose="Tom waits.", applied_state_changes=[],
        player_pick=Option(text="Where were you that night?"),
        player_pick_typed=True,
    ))
    typed_msg = build_prose_writer_user_message(
        state, world, applied, is_first_turn=False,
    )
    assert 'own words: "Where were you that night?"' in typed_msg
    assert "chose to:" not in typed_msg

    # Clicked menu option (intent).
    state2 = make_initial_state(world)
    state2.dialogue_history.append(TurnRecord(
        prose="Tom waits.", applied_state_changes=[],
        player_pick=Option(text="press him on the alibi"),
        player_pick_typed=False,
    ))
    menu_msg = build_prose_writer_user_message(
        state2, world, applied, is_first_turn=False,
    )
    assert "chose to: press him on the alibi" in menu_msg
    assert "own words" not in menu_msg


def test_prose_writer_system_prompt_forbids_travel():
    """Prose writer must explicitly stay in the current location — the
    structural defence against the prose-teleport failure mode (commit
    history: 06:55 session at the_old_vicarage where the narrator
    walked the player to Lower Hafod without an engine close)."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    # Some phrasing that pins "body stays inside" should be present.
    assert "does NOT cross the threshold" in sys_prompt
    assert "Do NOT name a destination" in sys_prompt


# ───────────────────────── phase 1 user prompt ─────────────────────────


def test_phase1_user_prompt_includes_reachable_beats_and_no_endings():
    """Endings are removed from Phase 1 entirely (no agent emits EndGame).
    Reachable beats are the load-bearing block."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    prompt = build_phase1_user_prompt(state, world)
    assert "## Reachable Beats" in prompt
    assert "## All Endings" not in prompt
    assert "## State" in prompt
    assert "## Case File" in prompt
    assert "## This Scene So Far" in prompt
    assert "## Generate State Changes" in prompt


def test_phase1_user_prompt_scene_anchor_is_compact():
    """## You Are At and ## Present are name-only in Phase 1's user prompt —
    the full location description and cast knowledge live in the cached
    system prompt, so the user prompt does not re-render them."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    prompt = build_phase1_user_prompt(state, world)
    loc = world.locations["the_slaughtered_lamb"]
    # Name is mentioned, but the long description is NOT re-rendered.
    assert loc.name in prompt
    assert loc.description.strip() not in prompt
    # Present NPCs are listed compactly without their knowledge.
    tom = world.npcs["tom"]
    assert tom.id in prompt
    assert tom.knowledge.strip() not in prompt


def test_phase1_reachable_beats_shows_only_fireable_now():
    """## Reachable Beats lists only beats fireable RIGHT NOW: prerequisites
    unlocked (not merely reachable), required NPCs present, valid location.
    Renders use the LLM-facing beat NAME."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # Tom is here

    prompt = build_phase1_user_prompt(state, world)
    # tom_named_a_vehicle fires now (Tom present, no prereq). tom_admits_more
    # needs tom_named_a_vehicle UNLOCKED first, so it is NOT yet fireable —
    # the strict gate stops the ladder from skipping ahead.
    assert world.beats["tom_named_a_vehicle"].name in prompt
    assert world.beats["tom_closes_off"].name in prompt
    reachable_block = prompt.split("## Reachable Beats")[1].split("\n\n")[0]
    assert world.beats["tom_admits_more"].name not in reachable_block

    # Once the vehicle is named, the next rung becomes fireable.
    state.unlocked_beats.add("tom_named_a_vehicle")
    prompt = build_phase1_user_prompt(state, world)
    assert world.beats["tom_admits_more"].name in prompt

    # Fire tom_closes_off → the whole Tom ladder is blocked; nothing fireable.
    state2 = make_initial_state(world)
    state2.player_location = "the_slaughtered_lamb"
    state2.unlocked_beats.add("tom_closes_off")
    prompt = build_phase1_user_prompt(state2, world)
    reachable_block = prompt.split("## Reachable Beats")[1].split("\n\n")[0]
    assert "(none" in reachable_block


def test_phase1_present_block_lists_npcs_at_current_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    prompt = build_phase1_user_prompt(state, world)
    present_block = prompt.split("## Present")[1].split("## State")[0]
    assert "tom" in present_block
    assert "maud" not in present_block


def test_phase1_present_block_says_alone_at_npc_free_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)  # vicarage, no NPCs
    prompt = build_phase1_user_prompt(state, world)
    present_block = prompt.split("## Present")[1].split("## State")[0]
    assert "alone" in present_block.lower()


# ───────────────────────── prose writer user message ─────────────────────────


def test_prose_writer_user_message_first_turn_includes_scene_anchor():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=True
    )
    assert "## You Are At" in msg
    assert "## Present" in msg
    assert "the_slaughtered_lamb" in msg


def test_prose_writer_user_message_subsequent_turn_drops_scene_anchor():
    """Turn 2+ only carries the delta — scene anchor lives in conversation
    history."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=False
    )
    assert "## You Are At" not in msg
    assert "## Present" not in msg


def test_prose_writer_user_message_excludes_reachable_beats_and_endings():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    for first in (True, False):
        msg = build_prose_writer_user_message(
            state, world, StateChangeBundle(), is_first_turn=first
        )
        assert "## Reachable Beats" not in msg
        assert "## All Endings" not in msg
        assert "## Case File" not in msg
        assert "## This Scene So Far" not in msg


def test_prose_writer_user_message_includes_applied_state_changes():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    applied = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")]
    )
    msg = build_prose_writer_user_message(
        state, world, applied, is_first_turn=True
    )
    assert "## State Changes Just Applied" in msg
    assert world.beats["tom_named_a_vehicle"].name in msg


def test_prose_writer_user_message_has_no_options_or_leave_directive():
    """Plain-text output: the user message must not mention options,
    next directives, or leave semantics."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=True
    )
    assert "MUST be EndScene" not in msg
    assert "EndScene" not in msg
    assert "EndGame" not in msg


def test_prose_writer_user_message_includes_player_pick_on_subsequent_turn():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    from handler.state import TurnRecord
    state.dialogue_history.append(TurnRecord(
        prose="t1",
        applied_state_changes=[],
        player_pick=Option(text="Ask Tom about the night"),
    ))
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=False
    )
    assert "## Player" in msg
    assert "Ask Tom about the night" in msg


# ───────────────────────── choice author user message ─────────────────────────


def test_choice_author_user_message_carries_prose_as_load_bearing_context():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    prose = "Tom looks at you and sets down his pint."
    msg = build_choice_author_user_message(state, world, prose)
    assert "## Prose" in msg
    assert prose in msg
    assert "## Choose" in msg


def test_choice_author_user_message_lists_present_npcs():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    msg = build_choice_author_user_message(state, world, "anything")
    assert "NPCs present" in msg
    # Tom is at the pub by default.
    assert world.npcs["tom"].name in msg


def test_choice_author_user_message_when_alone():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)  # vicarage; no NPCs
    msg = build_choice_author_user_message(state, world, "the door clicks shut")
    assert "alone" in msg.lower()


def test_choice_author_user_message_omits_pacing_block_early_turns():
    """With an NPC present, turns 1-3 should not render the Scene Pacing
    nudge (an NPC scene needs room to breathe)."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # Tom present → not solo
    # Turn 1: empty history. No nudge.
    msg = build_choice_author_user_message(state, world, "Anything.")
    assert "## Scene Pacing" not in msg
    # Turn 3: two prior turns. Still no nudge.
    state.dialogue_history = [
        TurnRecord(prose="a", applied_state_changes=[]),
        TurnRecord(prose="b", applied_state_changes=[]),
    ]
    msg = build_choice_author_user_message(state, world, "Anything.")
    assert "## Scene Pacing" not in msg


def test_choice_author_user_message_nudges_solo_from_turn_2():
    """Solo scenes (no NPC present) have shallow depth, so the Leave nudge
    appears from turn 2 — keeps the vicarage prologue from sprawling into an
    object hunt."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)  # vicarage; alone
    state.dialogue_history = [TurnRecord(prose="a", applied_state_changes=[])]
    msg = build_choice_author_user_message(state, world, "Anything.")
    assert "## Scene Pacing" in msg
    assert "turn 2" in msg


def test_choice_author_user_message_nudges_solo_after_quiet_turns():
    """Turn 5+ (PACING_QUIET_TURNS=4) with no NPCs present → render the
    soft solo Scene Pacing nudge."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)  # vicarage; alone
    state.dialogue_history = [
        TurnRecord(prose="t", applied_state_changes=[]) for _ in range(4)
    ]
    msg = build_choice_author_user_message(state, world, "Anything.")
    assert "## Scene Pacing" in msg
    assert "turn 5" in msg
    assert "Solo exploration" in msg


def test_choice_author_user_message_hard_nudge_after_hard_turns():
    """Turn 8+ (PACING_HARD_NUDGE_TURNS=7) → render the hard Leave nudge
    regardless of NPC presence."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # NPCs present
    state.dialogue_history = [
        TurnRecord(prose="t", applied_state_changes=[]) for _ in range(7)
    ]
    msg = build_choice_author_user_message(state, world, "Anything.")
    assert "## Scene Pacing" in msg
    assert "turn 8" in msg
    assert "filler" in msg


def test_choice_author_user_message_no_nudge_for_npc_scene_midrange():
    """Turns 4-5 with NPCs present → no nudge (NPC scenes legitimately
    run longer)."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # Tom is here
    state.dialogue_history = [
        TurnRecord(prose="t", applied_state_changes=[]) for _ in range(4)
    ]
    msg = build_choice_author_user_message(state, world, "Anything.")
    assert "## Scene Pacing" not in msg


# ───────────────────────── npc_memory + beat description ─────────────────────────


def test_prose_writer_user_message_renders_npc_history_when_present():
    """The prose writer's ## Present NPC card includes 'history with
    player' when state.npc_memory has entries for that NPC. Phase 1's
    ## Present does NOT (it sees the prose case file separately)."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    state.npc_memory["tom"] = [
        "Met Tom; guarded but warming. Did not name what he saw.",
        "Tom on visit two; he asked after my editor.",
    ]
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=True
    )
    assert "history with player" in msg
    assert "Did not name what he saw" in msg


def test_phase1_user_prompt_omits_npc_history():
    """Phase 1 must NOT render per-NPC history in its NPC card — Phase 1
    reads the prose case file via _render_case_file instead. Per-NPC
    memory is Phase 2-only."""
    from handler.prompts import build_phase1_user_prompt
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    state.npc_memory["tom"] = ["Tom note that must not appear in Phase 1."]
    prompt = build_phase1_user_prompt(state, world)
    assert "Tom note that must not appear" not in prompt


def test_prose_writer_applied_changes_includes_beat_description():
    """When a beat fires, the prose writer's ## State Changes Just Applied
    block surfaces the beat description (not just the slug). This is the
    load-bearing channel for the narrator to write the reveal moment
    with specifics — the prose writer's system prompt has no beat
    catalog."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    applied = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")]
    )
    msg = build_prose_writer_user_message(state, world, applied, is_first_turn=True)
    assert "## State Changes Just Applied" in msg
    beat = world.beats["tom_named_a_vehicle"]
    assert beat.name in msg
    description_words = beat.description.strip().split()
    assert " ".join(description_words[:4]) in msg


# ───────────────────────── populator prompts ─────────────────────────


def test_populator_system_prompt_carries_cast_knowledge_and_compact_beats():
    """Populator needs cast knowledge for dramatic placement reasoning. Beats
    appear in compact form (name + description) — no firing guidance, no
    endings."""
    from handler.prompts import build_populator_system_prompt
    world = load_world(FIXTURE_VALID)
    prompt = build_populator_system_prompt(world)
    # Cast knowledge appears.
    assert "Old Tom Bevan" in prompt
    # Compact beats: names appear, but firing guidance (notes_to_llm) does not.
    assert "## Story Beats" in prompt
    for beat in world.beats.values():
        assert beat.name in prompt
        assert beat.notes_to_llm.strip() not in prompt
    # Endings are dropped.
    for ending in world.endings.values():
        assert ending.id not in prompt
    # Job directive establishes restraint.
    assert "should be empty" in prompt or "empty bundle" in prompt


def test_populator_user_prompt_includes_current_placements_and_case_file():
    from handler.prompts import build_populator_user_prompt
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    state.case_file_entries = ["A prior scene at the pub."]
    prompt = build_populator_user_prompt(state, world)
    assert "## Player Just Entered" in prompt
    assert "the_slaughtered_lamb" in prompt
    assert "## Current NPC Locations" in prompt
    assert "tom:" in prompt
    # Case file accumulation surfaces (the populator reads it for context).
    assert "A prior scene at the pub" in prompt


# ───────────────────────── chronicler (unchanged) ─────────────────────────


def test_chronicler_system_prompt_byte_stable():
    world = load_world(FIXTURE_VALID)
    a = build_chronicler_system_prompt(world)
    b = build_chronicler_system_prompt(world)
    assert a == b


def test_chronicler_system_prompt_describes_dual_output():
    """The dual-output instruction (case_file_entry + npc_memory_updates)
    must surface in the system prompt; the chronicler needs to know about
    both fields. A worked case-file example anchors the terse voice."""
    world = load_world(FIXTURE_VALID)
    prompt = build_chronicler_system_prompt(world)
    assert "case_file_entry" in prompt
    assert "npc_memory_updates" in prompt
    assert "## Case File Example" in prompt


def test_chronicler_system_prompt_withholds_the_hidden_truth():
    """The chronicler writes only what the journalist witnessed, so it must
    NOT receive the solution: no crime narrative, no NPC knowledge, no beat
    catalog. Cast names stay (it renders NPC voice); knowledge does not."""
    world = load_world(FIXTURE_VALID)
    prompt = build_chronicler_system_prompt(world)
    assert "## What Happened" not in prompt
    assert "## Story Beats" not in prompt
    assert world.crime.strip() not in prompt
    for npc in world.npcs.values():
        assert npc.name in prompt  # cast names still present
        assert npc.knowledge.strip() not in prompt  # knowledge withheld
    for beat in world.beats.values():
        assert beat.description.strip() not in prompt


def test_prose_writer_system_prompt_uses_compact_locations():
    """Narrator's system prompt carries a one-line catalogue of locations;
    the long descriptions and notes live in the per-turn user prompt's
    ## You Are At block (current location only)."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    # Names + ids are present.
    for loc in world.locations.values():
        assert loc.id in sys_prompt
        assert loc.name in sys_prompt
    # But location NOTES are not rendered in the system prompt — the compact
    # catalogue takes only the first sentence of each description.
    pub = world.locations["the_slaughtered_lamb"]
    for note in pub.notes:
        assert note.strip() not in sys_prompt
