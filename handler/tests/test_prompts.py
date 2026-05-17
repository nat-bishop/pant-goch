"""M3 — Prompt builders.

Verifies KV-cache discipline (per-phase system prompts byte-stable), the
phase 1 / phase 2 split (Phase 2 sees voice samples + few-shots but NOT
beats / notes_to_llm / NPC knowledge / endings), and per-turn user-prompt
shape for both phases.
"""
from __future__ import annotations

from pathlib import Path

from handler.prompts import (
    build_chronicler_system_prompt,
    build_phase1_system_prompt,
    build_phase1_user_prompt,
    build_prose_writer_system_prompt,
    build_prose_writer_user_message,
)
from handler.schemas import Option, StateChangeBundle, UnlockBeat
from handler.state import make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── system prompts: byte stability ─────────────────────────


def test_phase1_system_prompt_is_byte_stable():
    world = load_world(FIXTURE_VALID)
    a = build_phase1_system_prompt(world)
    b = build_phase1_system_prompt(world)
    assert a == b


def test_phase2_system_prompt_is_byte_stable():
    world = load_world(FIXTURE_VALID)
    a = build_prose_writer_system_prompt(world)
    b = build_prose_writer_system_prompt(world)
    assert a == b


# ───────────────────────── phase 1 system prompt: full bible ─────────────────────────


def test_phase1_system_prompt_includes_full_world_bible():
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_phase1_system_prompt(world)
    for loc in world.locations.values():
        assert loc.id in sys_prompt
        assert loc.name in sys_prompt
    for npc in world.npcs.values():
        assert npc.id in sys_prompt
        assert npc.name in sys_prompt
    # Phase 1 sees beat NAMES (not ids) — the LLM emits the name and the
    # prompt's [THE TRUTH] / [REACHABLE BEATS] sections render names too.
    for beat in world.beats.values():
        assert beat.name in sys_prompt
        assert beat.notes_to_llm.strip() in sys_prompt
    for ending in world.endings.values():
        assert ending.id in sys_prompt
    assert "Megan Pritchard" in sys_prompt


# ───────────────────────── phase 2 system prompt: slim, voice-focused ─────────────────────────


def test_prose_writer_system_prompt_excludes_beats_and_notes_to_llm():
    """Prose writer must NOT see beat ids, notes_to_llm, or beat
    structure. Beat-aware prose was the failure mode this split fixes.
    The prose writer DOES see endings (because end_game is its tool),
    but never beats."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    for beat in world.beats.values():
        assert beat.id not in sys_prompt, (
            f"Prose writer system prompt leaks beat id {beat.id!r}"
        )
        assert beat.notes_to_llm.strip() not in sys_prompt, (
            f"Prose writer system prompt leaks notes_to_llm for {beat.id!r}"
        )
    assert "[THE TRUTH]" not in sys_prompt
    # `requires_beats` appears in the [ENDINGS] block (endings reference
    # their requires_beats by name) — that's intentional, prose writer
    # needs to know when end_game is fireable. But the beat catalog
    # itself (with `blocked_by_beats`) must NOT be present.
    assert "blocked_by_beats" not in sys_prompt


def test_prose_writer_system_prompt_includes_endings():
    """Prose writer sees endings — end_game(ending_id) is its tool. The
    `[ENDINGS]` block carries ending ids + their requires_beats so the
    model knows what's satisfied."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    assert "[ENDINGS]" in sys_prompt
    for ending in world.endings.values():
        assert ending.id in sys_prompt, (
            f"Prose writer system prompt missing ending id {ending.id!r}"
        )


def test_prose_writer_system_prompt_includes_tools_block():
    """Prose writer's [TOOLS] section documents all four tools."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    assert "[TOOLS]" in sys_prompt
    assert "pause_for_continue" in sys_prompt
    assert "request_free_text" in sys_prompt
    assert "end_scene" in sys_prompt
    assert "end_game" in sys_prompt


def test_phase2_system_prompt_excludes_npc_knowledge():
    """Phase 2 sees voice/attitude/themes/voice_sample but NOT knowledge
    (which carries truth specifics)."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    # The fixture has Tom's knowledge: "Tom saw a Land Rover..."
    tom = world.npcs["tom"]
    assert tom.knowledge.strip() not in sys_prompt, (
        "Phase 2 system prompt leaks Tom's knowledge — truth specifics"
    )


def test_phase2_system_prompt_includes_crime_overview():
    """Phase 2 sees the high-level crime so the narrator doesn't write
    contradictions. The fixture's crime mentions Megan Pritchard."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    assert world.crime.strip() in sys_prompt
    assert "[WHAT HAPPENED]" in sys_prompt


def test_phase2_system_prompt_includes_voice_samples_when_authored():
    """When NPCs have voice_sample populated, Phase 2's cast block must
    render them. The fixture has voice samples on at least one NPC (we
    enrich the fixture in this test if needed)."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    # The fixture NPCs may not have voice_samples yet — accept either:
    # if voice_sample is populated, the lines appear; if not, the marker
    # block at minimum shows up.
    any_voice = any(npc.voice_sample for npc in world.npcs.values())
    if any_voice:
        for npc in world.npcs.values():
            for line in npc.voice_sample:
                assert line.strip() in sys_prompt, (
                    f"voice_sample line for {npc.id!r} not in phase 2 prompt"
                )
    # Either way the cast block exists.
    for npc in world.npcs.values():
        assert npc.id in sys_prompt


def test_phase2_system_prompt_includes_few_shots_when_authored():
    """When world.few_shots is non-empty, [STYLE EXAMPLES] must render
    them. With empty list (fixture), the empty marker appears."""
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_prose_writer_system_prompt(world)
    assert "[STYLE EXAMPLES]" in sys_prompt
    if world.few_shots:
        for fs in world.few_shots:
            assert fs.setup.strip() in sys_prompt
            assert fs.player.strip() in sys_prompt
    else:
        assert "(no examples authored)" in sys_prompt


# ───────────────────────── phase 1 user prompt ─────────────────────────


def test_phase1_user_prompt_includes_reachable_beats_and_endings():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    prompt = build_phase1_user_prompt(state, world)
    assert "[REACHABLE BEATS]" in prompt
    assert "[ALL ENDINGS]" in prompt
    assert "[STATE]" in prompt
    assert "[CASE FILE]" in prompt
    assert "[THIS SCENE SO FAR]" in prompt
    assert "GENERATE STATE CHANGES" in prompt


def test_phase1_reachable_beats_hides_fired_and_blocked():
    """[REACHABLE BEATS] must omit fired beats and beats whose blockers
    have fired (or whose chain is transitively blocked). Renders use the
    LLM-facing beat NAME."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    prompt = build_phase1_user_prompt(state, world)
    # All three fixture beats reachable initially — assert by name.
    assert world.beats["tom_named_a_vehicle"].name in prompt
    assert world.beats["tom_admits_more"].name in prompt
    assert world.beats["tom_closes_off"].name in prompt

    # Fire tom_closes_off → tom_named_a_vehicle is blocked, tom_admits_more
    # transitively unreachable.
    state.unlocked_beats.add("tom_closes_off")
    prompt = build_phase1_user_prompt(state, world)
    reachable_block = prompt.split("[REACHABLE BEATS]")[1].split("\n\n")[0]
    assert "(none" in reachable_block


def test_phase1_present_block_lists_npcs_at_current_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    prompt = build_phase1_user_prompt(state, world)
    present_block = prompt.split("[PRESENT]")[1].split("[STATE]")[0]
    assert "tom" in present_block
    assert "maud" not in present_block


def test_phase1_present_block_says_alone_at_npc_free_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)  # vicarage, no NPCs
    prompt = build_phase1_user_prompt(state, world)
    present_block = prompt.split("[PRESENT]")[1].split("[STATE]")[0]
    assert "alone" in present_block.lower()


# ───────────────────────── phase 2 user message ─────────────────────────


def test_phase2_user_message_first_turn_includes_scene_anchor():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=True
    )
    assert "[YOU ARE AT]" in msg
    assert "[PRESENT]" in msg
    assert "the_slaughtered_lamb" in msg


def test_phase2_user_message_subsequent_turn_drops_scene_anchor():
    """Turn 2+ only carries the delta — scene anchor lives in conversation
    history."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=False
    )
    assert "[YOU ARE AT]" not in msg
    assert "[PRESENT]" not in msg


def test_phase2_user_message_excludes_reachable_beats_and_endings():
    """Phase 2 must not see beat structure or endings in its per-turn
    user message either."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    for first in (True, False):
        msg = build_prose_writer_user_message(
            state, world, StateChangeBundle(), is_first_turn=first
        )
        assert "[REACHABLE BEATS]" not in msg
        assert "[ALL ENDINGS]" not in msg
        assert "[CASE FILE]" not in msg
        assert "[THIS SCENE SO FAR]" not in msg


def test_phase2_user_message_includes_applied_state_changes():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    applied = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")]
    )
    msg = build_prose_writer_user_message(
        state, world, applied, is_first_turn=True
    )
    assert "[STATE CHANGES JUST APPLIED]" in msg
    # Render uses the LLM-facing beat name (ids never appear in prompts).
    assert world.beats["tom_named_a_vehicle"].name in msg


def test_phase2_user_message_leave_pick_directive():
    """Leave-pick directive tells the narrator the engine will close the
    scene — narrator writes departure prose, engine handles the rest."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    leave = Option(text="leave for the post office", leaves_for="the_post_office")
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(),
        leave_pick=leave, is_first_turn=True,
    )
    assert "[LEAVE DIRECTIVE]" in msg
    assert "the_post_office" in msg
    # The engine handles termination; narrator does NOT call end_scene
    # itself when leave_pick is set.
    assert "engine will close" in msg or "engine has it covered" in msg


def test_phase2_user_message_no_leave_pick_no_directive():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=True
    )
    assert "[LEAVE DIRECTIVE]" not in msg


def test_phase2_user_message_includes_player_pick_on_subsequent_turn():
    """On turn 2+, the player's last pick surfaces as a [PLAYER] block —
    it's their 'speaking turn' in the conversation."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    # Simulate a turn 1 with the player having picked an option.
    from handler.state import TurnRecord
    state.dialogue_history.append(TurnRecord(
        prose="t1",
        applied_state_changes=[],
        player_pick=Option(text="Ask Tom about the night"),
    ))
    msg = build_prose_writer_user_message(
        state, world, StateChangeBundle(), is_first_turn=False
    )
    assert "[PLAYER]" in msg
    assert "Ask Tom about the night" in msg


# ───────────────────────── npc_memory + beat description ─────────────────────────


def test_phase2_user_message_renders_npc_history_when_present():
    """Phase 2's [PRESENT] NPC card includes 'history with player' when
    state.npc_memory has entries for that NPC. Phase 1's [PRESENT] does
    NOT (it sees the prose case file separately)."""
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


def test_phase2_applied_changes_includes_beat_description():
    """When a beat fires, Phase 2's [STATE CHANGES JUST APPLIED] block
    surfaces the beat description (not just the slug). This is the
    load-bearing channel for Phase 2 to write the reveal moment with
    specifics — Phase 2's system prompt has no beat catalog."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    applied = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")]
    )
    msg = build_prose_writer_user_message(state, world, applied, is_first_turn=True)
    assert "[STATE CHANGES JUST APPLIED]" in msg
    beat = world.beats["tom_named_a_vehicle"]
    # Render uses the LLM-facing beat name.
    assert beat.name in msg
    # Beat description should surface in some form — at minimum a phrase
    # from the fixture beat's description.
    description_words = beat.description.strip().split()
    # First few words of the description should appear in the prompt.
    assert " ".join(description_words[:4]) in msg


# ───────────────────────── populator prompts ─────────────────────────


def test_populator_system_prompt_carries_full_bible():
    """Populator needs cast knowledge + beats + endings to reason
    dramatically about where NPCs should be."""
    from handler.prompts import build_populator_system_prompt
    world = load_world(FIXTURE_VALID)
    prompt = build_populator_system_prompt(world)
    # Cast knowledge appears.
    assert "Old Tom Bevan" in prompt
    # Beats + ending structure appears.
    assert "BEATS" in prompt
    # Job directive establishes restraint.
    assert "empty bundle" in prompt or "empty placements" in prompt


def test_populator_user_prompt_includes_current_placements_and_case_file():
    from handler.prompts import build_populator_user_prompt
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    state.case_file_entries = ["A prior scene at the pub."]
    prompt = build_populator_user_prompt(state, world)
    assert "[PLAYER JUST ENTERED]" in prompt
    assert "the_slaughtered_lamb" in prompt
    assert "[CURRENT NPC LOCATIONS]" in prompt
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
    both fields."""
    world = load_world(FIXTURE_VALID)
    prompt = build_chronicler_system_prompt(world)
    assert "case_file_entry" in prompt
    assert "npc_memory_updates" in prompt
