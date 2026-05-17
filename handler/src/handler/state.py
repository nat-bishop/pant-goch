"""Engine state.

Minimal scene-loop state:
  - player_location
  - npc_locations (override map; absent NPCs sit at default_location)
  - npc_scenes_co_present (counter for soft prereqs)
  - npc_memory (per-NPC short notes, one per scene the NPC was present)
  - unlocked_beats
  - dialogue_history (cleared on scene start)
  - scene_id (monotonic counter)
  - game_over

`npc_locations` is the populator agent's output channel — it writes
overrides at scene entry, e.g. "Dylan is at bryn_iago_wood for this
scene". Queries via `world.npcs_at(location_id, state=state)` fall
back to `default_location` for any NPC not in the dict.

`npc_memory` is the structured memory surfaced to Phase 2's NPC card
(separate from the prose case file, which is the UI artifact). The log
writer emits one short note per NPC present at scene-end; the engine
appends it to the per-NPC list. Phase 2's `_render_npc_card(voice_only=True)`
renders the recent entries inline with the NPC briefing.

Drops `npc_alive` from SPEC §3.1 (no kill_npc in v1).

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
class LogEntry:
    """One case-file entry. Written by the log writer after a scene ends;
    the text is what streams to the player's case-file sidebar AND folds
    back into the next scene's `[CASE FILE]` prompt block.

    Anchors (scene_id, location_id, npcs_present) come from the
    just-completed scene's first dialogue_history turn — all turns in a
    scene share these. v1 had an `action` field for the picked Option;
    v2's logs are scene-keyed (multiple turns per entry) so it's dropped.

    `npc_memory_updates` carries the structured per-NPC memory channel
    the log writer emits alongside the prose case file (one LLM call,
    two artifacts). `scene_loop.play_session` merges these into
    `state.npc_memory[npc_id]` (one append per NPC); Phase 2's
    [PRESENT] NPC card then renders them on subsequent scenes.
    """

    scene_id: int
    location_id: str
    npcs_present: list[str]
    text: str
    npc_memory_updates: dict[str, str] = field(default_factory=dict)


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
    npc_locations: dict[str, str] = field(default_factory=dict)
    npc_scenes_co_present: dict[str, int] = field(default_factory=dict)
    npc_memory: dict[str, list[str]] = field(default_factory=dict)
    unlocked_beats: set[str] = field(default_factory=set)
    dialogue_history: list[TurnRecord] = field(default_factory=list)
    scene_id: int = 0
    game_over: GameOver | None = None


def make_initial_state(world: World) -> State:
    """Build the State for a fresh game from the loaded world. NPCs start
    at their `default_location` (seeded into `npc_locations`); the
    populator can override on scene entry. Co-presence counter 0; the
    first enter_scene increments for any NPC at the start location.
    `npc_memory` starts empty; the log writer populates it at scene-end."""
    return State(
        player_location=world.brief.player_starts_at,
        npc_locations={npc.id: npc.default_location for npc in world.npcs.values()},
        npc_scenes_co_present={npc_id: 0 for npc_id in world.npcs},
        npc_memory={npc_id: [] for npc_id in world.npcs},
    )
