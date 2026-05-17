"""Scene-loop helpers — reachability, state application, scene lifecycle.

Pure functions (no LLM, no SSE). All structural / state-dependent rules
that used to live in `validate_state_change_bundle` and
`validate_prose_part` have moved:

  - State-INDEPENDENT rules (≤2 unlock_beat cap, no duplicate beats,
    ≥1 leave option in ContinueScene) → `@model_validator` /
    `Field(...)` on the canonical schemas in `schemas.py`.
  - State-DEPENDENT enum constraints (`beat_id` ∈ reachable beats,
    `ending_id` ∈ satisfied endings, `leaves_for` ∈ adjacent locations,
    `next == EndScene` when leave_pick is set) → per-call `Literal[...]`
    narrowing in `agents._build_state_change_bundle_type` /
    `_build_prose_part_type`.

`reachable_beat_ids` stays here — used by both grammar narrowing
(`agents.py`) and prompt rendering (`prompts.py`).
"""
from __future__ import annotations

from .schemas import StateChangeBundle, UnlockBeat
from .state import State, TurnRecord
from .world import World


def apply_state_changes(bundle: StateChangeBundle, state: State) -> None:
    """Apply validated state changes to state. Validation happens upstream
    via schema @model_validator + per-call Literal narrowing in
    `agents._build_state_change_bundle_type`."""
    for change in bundle.state_changes:
        if isinstance(change, UnlockBeat):
            state.unlocked_beats.add(change.beat_id)


# ───────────────────────── reachability filter ─────────────────────────


def reachable_beat_ids(world: World, state: State) -> set[str]:
    """Return ids of beats that are *reachable* — unfired AND can eventually
    fire given current state. Used by prompts.py to render [REACHABLE BEATS]
    and by agents._build_state_change_bundle_type to narrow the
    grammar-sampled Literal so the model cannot emit a transitively-blocked
    or transitively-unreachable beat.

    Reachability (memoized recursion):
      A beat B is reachable iff:
        - B is not in unlocked_beats, AND
        - no id in B.blocked_by_beats is in unlocked_beats, AND
        - every id in B.requires_beats is either in unlocked_beats OR reachable.

    Cycles in requires_beats are forbidden by the world loader (invariant 9),
    so the recursion terminates.
    """
    cache: dict[str, bool] = {}

    def is_reachable(beat_id: str) -> bool:
        if beat_id in cache:
            return cache[beat_id]
        beat = world.beats.get(beat_id)
        if beat is None:
            cache[beat_id] = False
            return False
        if beat_id in state.unlocked_beats:
            cache[beat_id] = False
            return False
        if any(b in state.unlocked_beats for b in beat.blocked_by_beats):
            cache[beat_id] = False
            return False
        for r in beat.requires_beats:
            if r in state.unlocked_beats:
                continue
            if not is_reachable(r):
                cache[beat_id] = False
                return False
        cache[beat_id] = True
        return True

    return {bid for bid in world.beats if is_reachable(bid)}


# ───────────────────────── scene lifecycle ─────────────────────────


def enter_scene(state: State, world: World, location_id: str) -> None:
    """Begin a new scene at `location_id`. Increments scene_id, clears
    dialogue_history, sets player_location.

    SSE / log-writer / populator triggering happens at the scene_loop
    layer; this function only mutates state.
    """
    state.scene_id += 1
    state.dialogue_history = []
    state.player_location = location_id


def append_turn_record(
    state: State,
    *,
    prose: str,
    applied_state_changes: list[UnlockBeat],
) -> None:
    """Pack a completed turn's data into a TurnRecord and append it to
    dialogue_history. The scene anchor (scene_id, location, npcs_present)
    is constant across `dialogue_history` and lives on `state`, so
    renderers derive it once per scene."""
    state.dialogue_history.append(
        TurnRecord(
            prose=prose,
            applied_state_changes=list(applied_state_changes),
        )
    )
