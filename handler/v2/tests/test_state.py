"""M2 — State + scene lifecycle tests."""
from __future__ import annotations

from pathlib import Path

from handler_v2.engine import enter_scene
from handler_v2.state import GameOver, State, TurnRecord, make_initial_state
from handler_v2.world import load_world


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


def test_enter_scene_increments_co_presence_for_npcs_at_location():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    # Player starts at vicarage (no NPCs). Walk to pub (tom is there).
    enter_scene(state, world, "the_slaughtered_lamb")
    assert state.npc_scenes_co_present == {"tom": 1, "maud": 0}
    assert state.player_location == "the_slaughtered_lamb"
    assert state.scene_id == 1
    # Walk to post office (maud is there).
    enter_scene(state, world, "the_post_office")
    assert state.npc_scenes_co_present == {"tom": 1, "maud": 1}
    assert state.scene_id == 2
    # Back to pub.
    enter_scene(state, world, "the_slaughtered_lamb")
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
    enter_scene(state, world, "the_old_vicarage")  # no NPCs there
    assert state.npc_scenes_co_present == {"tom": 0, "maud": 0}


def test_game_over_dataclass_carries_ending_id():
    over = GameOver(ending_id="published_story")
    assert over.ending_id == "published_story"
