"""Scene-loop helpers — reachability, state application, scene lifecycle.

Pure functions (no LLM, no SSE). All structural / state-dependent rules
that used to live in `validate_state_change_bundle` and
`validate_prose_part` have moved:

  - State-INDEPENDENT rules (≤2 unlock_beat cap, no duplicate beats,
    Continue option count 2-4) → `@model_validator` / `Field(...)` on
    the canonical schemas in `schemas.py`.
  - State-DEPENDENT enum constraints (`beat_id` ∈ reachable beats) →
    per-call `Literal[...]` narrowing in
    `agents._build_state_change_bundle_type`.

`reachable_beat_ids` and `resolved_ending_id_at_close` stay here — used
by both grammar narrowing (`agents.py`) and prompt rendering
(`prompts.py`), plus the scene_loop dispatch on choice-author Leave.
"""
from __future__ import annotations

from .schemas import FreeText, StateChangeBundle, UnlockBeat
from .state import State, TurnRecord
from .world import World


# The free-text box surfaced when the gate offers it. General agency: a
# fixed, neutral caption (the move may be speech or action) and no
# addressee (`npc_id=None`). Built by the engine, never by the LLM —
# the gate emits only a bool. See `scene_loop._assemble_prose_part`.
SCENE_FREE_TEXT = FreeText(prompt="What do you do next?", npc_id=None)


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
    fire given current state. Used by prompts.py to render ## Reachable Beats
    and by agents._build_state_change_bundle_type to narrow the
    grammar-sampled Literal so the model cannot emit a transitively-blocked
    or transitively-unreachable beat.

    Reachability (memoized recursion):
      A beat B is reachable iff:
        - B is not in unlocked_beats, AND
        - no id in B.blocked_by_beats is in unlocked_beats, AND
        - every id in B.requires_beats is either in unlocked_beats OR reachable, AND
        - if B.requires_any_beats is non-empty, at least one id in it is
          either in unlocked_beats OR reachable.

    Cycles in requires_beats / requires_any_beats are forbidden by the world
    loader (invariant 9), so the recursion terminates.
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
        if beat.requires_any_beats:
            any_ok = any(
                r in state.unlocked_beats or is_reachable(r)
                for r in beat.requires_any_beats
            )
            if not any_ok:
                cache[beat_id] = False
                return False
        cache[beat_id] = True
        return True

    return {bid for bid in world.beats if is_reachable(bid)}


def present_npc_ids(world: World, state: State) -> tuple[str, ...]:
    """NPC ids at the player's current location. Stable across a scene
    (the populator runs only at scene entry), so callers can compute
    on demand without caching."""
    return tuple(n.id for n in world.npcs_at(state.player_location, state=state))


def resolved_ending_id_at_close(world: World, state: State) -> str | None:
    """Pick the ending that fires when a scene closes. Iterates endings in
    author (YAML) order; returns the first whose `requires_beats` are all
    in `state.unlocked_beats`, or None if no ending is satisfied.

    By design, decision beats are mutually exclusive (pairwise
    `blocked_by_beats`), so at most one ending should ever be satisfied
    simultaneously. The author-order tie-break is defensive.

    Used by `scene_loop._assemble_prose_part` when the choice author emits
    a Leave variant: if an ending is satisfied, the engine constructs
    EndGame with that id; otherwise EndScene.
    """
    for eid, ending in world.endings.items():
        if not set(ending.requires_beats).issubset(state.unlocked_beats):
            continue
        if ending.requires_any_beats and not any(
            b in state.unlocked_beats for b in ending.requires_any_beats
        ):
            continue
        return eid
    return None


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
