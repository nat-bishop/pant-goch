"""M2 — State + scene lifecycle tests."""
from __future__ import annotations

from pathlib import Path

from handler.engine import enter_scene, record_scene_co_presence
from handler.state import GameOver, State, TurnRecord, make_initial_state
from handler.world import load_world


def _enter_and_count(state, world, location_id):
    """Helper: enter_scene + record_scene_co_presence in the order that
    scene_loop.play_scene calls them (around the populator, which the
    state-level tests don't exercise)."""
    enter_scene(state, world, location_id)
    record_scene_co_presence(state, world)


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def test_make_initial_state_sets_player_at_brief_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    assert state.player_location == "the_old_vicarage"


def test_make_initial_state_zeroes_co_presence_for_every_npc():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    assert state.npc_scenes_co_present == {"tom": 0, "maud": 0}
    assert state.unlocked_beats == set()
    assert state.dialogue_history == []
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


def test_enter_scene_increments_co_presence_for_npcs_at_location():
    """Co-presence increment is now `record_scene_co_presence`, called
    after `enter_scene` (and after the populator, which doesn't run in
    state-level tests). Helper `_enter_and_count` mirrors the production
    ordering in `scene_loop.play_scene`."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    # Player starts at vicarage (no NPCs). Walk to pub (tom is there).
    _enter_and_count(state, world, "the_slaughtered_lamb")
    assert state.npc_scenes_co_present == {"tom": 1, "maud": 0}
    assert state.player_location == "the_slaughtered_lamb"
    assert state.scene_id == 1
    # Walk to post office (maud is there).
    _enter_and_count(state, world, "the_post_office")
    assert state.npc_scenes_co_present == {"tom": 1, "maud": 1}
    assert state.scene_id == 2
    # Back to pub.
    _enter_and_count(state, world, "the_slaughtered_lamb")
    assert state.npc_scenes_co_present == {"tom": 2, "maud": 1}
    assert state.scene_id == 3


def test_enter_scene_clears_dialogue_history():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.dialogue_history.append(
        TurnRecord(
            scene_id=99,
            location_id="x",
            npcs_present=[],
            prose="stale",
            applied_state_changes=[],
            next_kind="continue",
            options=[],
        )
    )
    enter_scene(state, world, "the_slaughtered_lamb")
    assert state.dialogue_history == []


def test_enter_scene_at_npc_free_location_does_not_change_counts():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    _enter_and_count(state, world, "the_old_vicarage")  # no NPCs there
    assert state.npc_scenes_co_present == {"tom": 0, "maud": 0}


def test_game_over_dataclass_carries_ending_id():
    over = GameOver(ending_id="published_story")
    assert over.ending_id == "published_story"
