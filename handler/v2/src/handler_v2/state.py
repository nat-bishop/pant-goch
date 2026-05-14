"""Engine state.

Per plan §"State (refines §3.1)" — minimal:
  - player_location
  - npc_scenes_co_present (counter for soft prereqs)
  - unlocked_beats
  - dialogue_history (cleared on scene start)
  - scene_id (monotonic counter)
  - game_over

Drops from SPEC §3.1: `npc_alive` (no kill_npc in v1), `npc_locations`
(no MoveNPC in v1; renderer queries cast directly).

`npc_scenes_co_present` measures co-presence (player was in the room),
NOT conversation. Soft prereqs that consume it must phrase that way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .schemas import Option, UnlockBeat
from .world import World


@dataclass
class GameOver:
    ending_id: str


@dataclass
class TurnRecord:
    """One completed turn within a scene. Carries the scene anchor (scene_id,
    location, npcs_present) so the renderer can group consecutive same-scene
    turns under a single header."""

    scene_id: int
    location_id: str
    npcs_present: list[str]
    prose: str
    applied_state_changes: list[UnlockBeat]
    next_kind: Literal["continue", "end_scene", "end_game"]
    options: list[Option]
    player_pick: Option | None = None  # set by engine when player picks


@dataclass
class State:
    player_location: str
    npc_scenes_co_present: dict[str, int] = field(default_factory=dict)
    unlocked_beats: set[str] = field(default_factory=set)
    dialogue_history: list[TurnRecord] = field(default_factory=list)
    scene_id: int = 0
    game_over: GameOver | None = None


def make_initial_state(world: World) -> State:
    """Build the State for a fresh game from the loaded world. NPCs start
    with co-presence counter 0; the first enter_scene increments for any
    NPC at the start location."""
    return State(
        player_location=world.brief.player_starts_at,
        npc_scenes_co_present={npc_id: 0 for npc_id in world.npcs},
    )
