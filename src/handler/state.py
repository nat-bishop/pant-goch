"""Engine state.

The single source of truth for everything session-persistent. A
`dataclasses.asdict(state)` is the complete save-game shape.

  - player_location
  - npc_locations (override map; absent NPCs sit at default_location)
  - npc_memory (per-NPC short notes, one per scene the NPC was present)
  - unlocked_beats
  - case_file_entries (one prose entry per closed scene)
  - dialogue_history (scene-bounded; cleared on scene start)
  - scene_id (monotonic counter)
  - game_over

`npc_locations` is the populator agent's output channel — it writes
overrides at scene entry, e.g. "Dylan is at bryn_iago_wood for this
scene". Queries via `world.npcs_at(location_id, state=state)` fall
back to `default_location` for any NPC not in the dict.

`npc_memory` is the structured memory surfaced to Phase 2's NPC card
(separate from the prose case file, which is the UI artifact). The log
writer emits one short note per NPC present at scene-end; the engine
appends it to the per-NPC list. Phase 2's [PRESENT] card renders the
recent entries inline with the NPC briefing.

`case_file_entries` is the journalist's prose record, written by the
chronicler at scene close. Phase 1 reads it as cross-scene context; the
SSE frontend renders it as the case-file sidebar. Symmetric with
`npc_memory` — both are session-persistent, both written by the same
chronicler call, both live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Option, UnlockBeat
from .world import World


@dataclass
class GameOver:
    ending_id: str


@dataclass
class TurnRecord:
    """One completed turn within the current scene. The scene anchor
    (scene_id, location, npcs_present) is constant across all turns in
    `dialogue_history` (which is cleared per scene), so it lives on
    `state` and renderers derive it once per scene rather than per turn.
    """

    prose: str
    applied_state_changes: list[UnlockBeat]
    player_pick: Option | None = None  # set by engine when player picks
    # True when player_pick is a verbatim free-text line the player typed
    # (vs. a clicked menu option). Lets the next turn's prompts render it
    # as the player's own words rather than an intent to dramatize.
    player_pick_typed: bool = False


@dataclass
class State:
    player_location: str
    npc_locations: dict[str, str] = field(default_factory=dict)
    npc_memory: dict[str, list[str]] = field(default_factory=dict)
    unlocked_beats: set[str] = field(default_factory=set)
    case_file_entries: list[str] = field(default_factory=list)
    dialogue_history: list[TurnRecord] = field(default_factory=list)
    scene_id: int = 0
    game_over: GameOver | None = None

    @property
    def turn_n(self) -> int:
        """1-indexed turn number within the current scene — the prose
        about to be written is turn N. `dialogue_history` holds the N-1
        completed turns; the player's pending pick is on the last
        entry. Cleared with the rest of dialogue_history at scene entry."""
        return len(self.dialogue_history) + 1


def make_initial_state(world: World) -> State:
    """Build the State for a fresh game from the loaded world. NPCs start
    at their `default_location` (seeded into `npc_locations`); the
    populator can override on scene entry. `npc_memory` and
    `case_file_entries` start empty; the chronicler populates them at
    scene-end."""
    return State(
        player_location=world.brief.player_starts_at,
        npc_locations={npc.id: npc.default_location for npc in world.npcs.values()},
        npc_memory={npc_id: [] for npc_id in world.npcs},
    )
