"""M3 — Prompt builders.

Verifies KV-cache discipline (system prompt byte-stable, phase1/phase2
share the dynamic-suffix prefix), reachability filter integration, and
that all rendering paths handle the valid fixture without error.
"""
from __future__ import annotations

from pathlib import Path

from handler.prompts import (
    build_phase1_tail,
    build_phase1_user_prompt,
    build_phase2_tail,
    build_phase2_user_prompt,
    build_prompt_prefix,
    build_scene_runner_system_prompt,
)
from handler.schemas import Option, StateChangeBundle, UnlockBeat
from handler.state import make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def test_system_prompt_is_byte_stable_across_calls():
    """System prefix MUST be byte-identical across calls — KV cache
    relies on this. Build twice from the same world; bytes must match."""
    world = load_world(FIXTURE_VALID)
    a = build_scene_runner_system_prompt(world)
    b = build_scene_runner_system_prompt(world)
    assert a == b


def test_system_prompt_includes_full_world_bible():
    world = load_world(FIXTURE_VALID)
    sys_prompt = build_scene_runner_system_prompt(world)
    # Locations
    for loc in world.locations.values():
        assert loc.id in sys_prompt
        assert loc.name in sys_prompt
    # NPCs
    for npc in world.npcs.values():
        assert npc.id in sys_prompt
        assert npc.name in sys_prompt
    # Beats
    for beat in world.beats.values():
        assert beat.id in sys_prompt
    # Endings
    for ending in world.endings.values():
        assert ending.id in sys_prompt
    # Crime
    assert "Megan Pritchard" in sys_prompt


def test_phase1_and_phase2_prompts_share_prefix():
    """KV cache discipline: phase 1 and phase 2 must share their byte
    prefix up to a single divergence point. Compute both, assert the
    longer one starts with the shorter shared prefix (returned by
    build_prompt_prefix)."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)

    prefix = build_prompt_prefix(state, world)
    phase1 = build_phase1_user_prompt(state, world)
    phase2 = build_phase2_user_prompt(state, world, StateChangeBundle())

    # Both phase prompts start with the byte-identical prefix.
    assert phase1.startswith(prefix)
    assert phase2.startswith(prefix)
    # And after the prefix, they diverge.
    p1_tail = phase1[len(prefix):]
    p2_tail = phase2[len(prefix):]
    assert p1_tail != p2_tail
    # Phase 1 tail mentions GENERATE STATE CHANGES; phase 2 mentions
    # GENERATE PROSE AND NEXT.
    assert "GENERATE STATE CHANGES" in p1_tail
    assert "GENERATE PROSE AND NEXT" in p2_tail


def test_phase2_prompt_includes_applied_state_changes():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    applied = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")]
    )
    phase2 = build_phase2_user_prompt(state, world, applied)
    assert "[STATE CHANGES JUST APPLIED]" in phase2
    assert "tom_named_a_vehicle" in phase2


def test_phase2_leave_pick_directive():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    applied = StateChangeBundle()
    leave = Option(text="leave for the post office", leaves_for="the_post_office")
    phase2 = build_phase2_user_prompt(state, world, applied, leave_pick=leave)
    # The leave directive forces EndScene.
    assert "MUST be EndScene" in phase2
    assert "the_post_office" in phase2


def test_phase2_no_leave_pick_no_directive():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    applied = StateChangeBundle()
    phase2 = build_phase2_user_prompt(state, world, applied)
    assert "MUST be EndScene" not in phase2


def test_reachable_beats_hides_fired_and_blocked():
    """[REACHABLE BEATS] must omit fired beats and beats whose blockers
    have fired (or whose chain is transitively blocked)."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    # Initially: all 3 beats reachable.
    suffix = build_prompt_prefix(state, world)
    assert "tom_named_a_vehicle" in suffix
    assert "tom_admits_more" in suffix
    assert "tom_closes_off" in suffix

    # Fire tom_closes_off → tom_named_a_vehicle is blocked, tom_admits_more
    # transitively unreachable (requires the blocked beat).
    state.unlocked_beats.add("tom_closes_off")
    suffix = build_prompt_prefix(state, world)
    # tom_closes_off is fired → not in reachable.
    # tom_named_a_vehicle is blocked → not in reachable.
    # tom_admits_more requires tom_named_a_vehicle → transitively unreachable.
    # The [REACHABLE BEATS] section should say "(none)".
    # Find the reachable-beats section and verify it's the empty marker.
    reachable_block = suffix.split("[REACHABLE BEATS]")[1].split("\n\n")[0]
    assert "(none" in reachable_block


def test_opening_seed_only_on_first_turn():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    with_seed = build_prompt_prefix(state, world, opening_seed="just arrived")
    without_seed = build_prompt_prefix(state, world)
    assert "[OPENING SEED]" in with_seed
    assert "just arrived" in with_seed
    assert "[OPENING SEED]" not in without_seed


def test_present_block_lists_npcs_at_current_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # tom is here
    suffix = build_prompt_prefix(state, world)
    assert "[PRESENT]" in suffix
    # Verify Tom is present, Maud isn't (she's at post office).
    present_block = suffix.split("[PRESENT]")[1].split("[STATE]")[0]
    assert "tom" in present_block
    assert "maud" not in present_block


def test_present_block_says_alone_at_npc_free_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)  # vicarage, no NPCs
    suffix = build_prompt_prefix(state, world)
    present_block = suffix.split("[PRESENT]")[1].split("[STATE]")[0]
    assert "alone" in present_block.lower()
