"""M2 — State + scene lifecycle tests."""
from __future__ import annotations

from pathlib import Path

from handler.engine import enter_scene
from handler.state import GameOver, State, TurnRecord, make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def test_make_initial_state_sets_player_at_brief_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    assert state.player_location == "the_old_vicarage"


def test_make_initial_state_defaults():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    assert state.unlocked_beats == set()
    assert state.dialogue_history == []
    assert state.case_file_entries == []
    assert state.scene_id == 0
    assert state.game_over is None


def test_make_initial_state_seeds_npc_locations_from_defaults():
    """The populator overrides `state.npc_locations`; before it runs,
    defaults are the starting positions. `state_aware npcs_at` reads from
    this dict, so seeding it at init is load-bearing."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    for npc_id, npc in world.npcs.items():
        assert state.npc_locations[npc_id] == npc.default_location


def test_make_initial_state_initialises_empty_npc_memory_lists():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    for npc_id in world.npcs:
        assert state.npc_memory[npc_id] == []


def test_enter_scene_increments_scene_id_and_sets_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")
    assert state.player_location == "the_slaughtered_lamb"
    assert state.scene_id == 1
    enter_scene(state, world, "the_post_office")
    assert state.player_location == "the_post_office"
    assert state.scene_id == 2


def test_enter_scene_clears_dialogue_history():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.dialogue_history.append(
        TurnRecord(prose="stale", applied_state_changes=[])
    )
    enter_scene(state, world, "the_slaughtered_lamb")
    assert state.dialogue_history == []


def test_game_over_dataclass_carries_ending_id():
    over = GameOver(ending_id="published_story")
    assert over.ending_id == "published_story"
