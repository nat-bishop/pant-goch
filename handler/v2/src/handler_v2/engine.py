"""Scene loop + state-change validation + reachability filter.

M2 scope: pure functions (no LLM, no SSE). The functions here are
called by:
  - agents.py output_validators (M3) — they raise StateChangeError, which
    the validator wraps in pydantic-ai's ModelRetry(msg).
  - prompts.py [REACHABLE BEATS] renderer (M3) — uses reachable_beat_ids().
  - the scene loop (M3) — calls enter_scene, apply_state_changes.

Everything here is unit-testable without a running llama-server.
"""
from __future__ import annotations

from .schemas import (
    ContinueScene,
    EndGame,
    EndScene,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from .state import State, TurnRecord
from .world import World


# Plan: ≤2 unlock_beat per turn cap. Allows naturally-paired beats (e.g.
# "Helen hands over records AND admits she suspected Eluned" can land in
# one moment without a synthetic delay); rejects bursty 3+ firings.
UNLOCK_BEAT_CAP_PER_BUNDLE = 2


class StateChangeError(ValueError):
    """One of the engine's structural invariants on a state-change bundle was
    violated. Message is suitable for ModelRetry — terse, names the offender."""


class ProsePartError(ValueError):
    """A ProsePart violated an engine invariant (ending prereqs, leave-pick
    forces EndScene, ContinueScene needs ≥1 leave option). Message is
    suitable for ModelRetry."""


# ───────────────────────── state-change bundle validation ─────────────────────────


def validate_state_change_bundle(
    bundle: StateChangeBundle, world: World, state: State
) -> None:
    """Raise StateChangeError if the bundle violates engine rules. Used by:
      - agents.py output_validator on the phase-1 agent (M3).
      - apply_state_changes (defense-in-depth before mutation).

    Rules:
      - ≤2 UnlockBeat per bundle.
      - Each UnlockBeat.beat_id resolves to a defined beat.
      - Each beat's requires_beats ⊆ state.unlocked_beats.
      - No beat in blocked_by_beats has fired.
    """
    unlock_count = sum(1 for c in bundle.state_changes if isinstance(c, UnlockBeat))
    if unlock_count > UNLOCK_BEAT_CAP_PER_BUNDLE:
        raise StateChangeError(
            f"too many unlock_beat changes in one turn: {unlock_count} > "
            f"{UNLOCK_BEAT_CAP_PER_BUNDLE}; pace beats across turns"
        )

    # Within a single bundle the LLM might also try to fire the same beat
    # twice — defensive check.
    seen_in_bundle: set[str] = set()
    for change in bundle.state_changes:
        if isinstance(change, UnlockBeat):
            if change.beat_id in seen_in_bundle:
                raise StateChangeError(
                    f"unlock_beat: {change.beat_id!r} appears twice in this bundle"
                )
            seen_in_bundle.add(change.beat_id)
            _validate_unlock_beat(change, world, state)


def _validate_unlock_beat(change: UnlockBeat, world: World, state: State) -> None:
    beat = world.beats.get(change.beat_id)
    if beat is None:
        raise StateChangeError(
            f"unlock_beat: beat {change.beat_id!r} is not defined in the world"
        )
    if change.beat_id in state.unlocked_beats:
        # Idempotent in principle but suspicious in practice — surface it.
        raise StateChangeError(
            f"unlock_beat: beat {change.beat_id!r} has already fired"
        )
    missing_requires = [r for r in beat.requires_beats if r not in state.unlocked_beats]
    if missing_requires:
        raise StateChangeError(
            f"unlock_beat {change.beat_id!r}: requires_beats not yet satisfied: "
            f"{missing_requires}; fire them first or don't fire this beat yet"
        )
    active_blockers = [b for b in beat.blocked_by_beats if b in state.unlocked_beats]
    if active_blockers:
        raise StateChangeError(
            f"unlock_beat {change.beat_id!r}: blocked by already-fired "
            f"{active_blockers}; this beat is permanently unreachable"
        )


def apply_state_changes(bundle: StateChangeBundle, state: State) -> None:
    """Apply validated state changes to state. Call validate_state_change_bundle
    first; this function does not re-validate (the M3 output_validator handles
    that)."""
    for change in bundle.state_changes:
        if isinstance(change, UnlockBeat):
            state.unlocked_beats.add(change.beat_id)


# ───────────────────────── prose-part validation ─────────────────────────


def validate_prose_part(
    prose_part: ProsePart,
    world: World,
    state: State,
    *,
    leave_pick: Option | None = None,
) -> None:
    """Raise ProsePartError if the ProsePart violates engine rules. Used by
    the phase-2 output_validator (M3).

    Rules:
      - If `next` is EndGame: ending exists AND ending.requires_beats ⊆ state.unlocked_beats.
      - If `next` is ContinueScene: at least one option has leaves_for set,
        and every leaves_for value resolves to an adjacent location of state.player_location.
      - If `leave_pick` is not None (player picked a leave option last turn):
        `next` MUST be EndScene.
    """
    if leave_pick is not None and not isinstance(prose_part.next, EndScene):
        raise ProsePartError(
            f"player chose to leave for {leave_pick.leaves_for!r}; you MUST "
            f"emit EndScene as next, not {type(prose_part.next).__name__}"
        )

    if isinstance(prose_part.next, EndGame):
        ending = world.endings.get(prose_part.next.ending_id)
        if ending is None:
            raise ProsePartError(
                f"end_game: ending {prose_part.next.ending_id!r} is not defined"
            )
        missing = [r for r in ending.requires_beats if r not in state.unlocked_beats]
        if missing:
            raise ProsePartError(
                f"end_game {ending.id!r}: requires_beats not yet satisfied: "
                f"{missing}; the player has not committed to this ending yet"
            )

    if isinstance(prose_part.next, ContinueScene):
        leave_options = [o for o in prose_part.next.options if o.leaves_for is not None]
        if not leave_options:
            raise ProsePartError(
                "continue: at least one option must have leaves_for set "
                "(player must always have a walk-away choice)"
            )
        location = world.locations.get(state.player_location)
        if location is None:
            # Defensive — should be impossible if state is consistent.
            raise ProsePartError(
                f"player_location {state.player_location!r} is not defined"
            )
        adjacent = set(location.adjacent_to)
        for opt in leave_options:
            if opt.leaves_for not in adjacent:
                raise ProsePartError(
                    f"option leaves_for={opt.leaves_for!r}: not adjacent to "
                    f"current location {state.player_location!r} (adjacents: "
                    f"{sorted(adjacent)})"
                )


# ───────────────────────── reachability filter ─────────────────────────


def reachable_beat_ids(world: World, state: State) -> set[str]:
    """Return ids of beats that are *reachable* — unfired AND can eventually
    fire given current state. Used by prompts.py to render [REACHABLE BEATS],
    omitting transitively-blocked or transitively-unreachable beats entirely.

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
    dialogue_history, increments npc_scenes_co_present for each NPC at
    the new location.

    SSE / log-writer triggering happens in the engine layer (M3); this
    function only mutates state.
    """
    state.scene_id += 1
    state.dialogue_history = []
    state.player_location = location_id
    for npc in world.npcs_at(location_id):
        state.npc_scenes_co_present[npc.id] = (
            state.npc_scenes_co_present.get(npc.id, 0) + 1
        )


def append_turn_record(
    state: State,
    *,
    location_id: str,
    npcs_present: list[str],
    prose: str,
    applied_state_changes: list,  # validated to be UnlockBeat-only by the bundle validator
    prose_part: ProsePart,
) -> None:
    """Pack a completed turn's data into a TurnRecord and append it to
    dialogue_history. The renderer groups records by scene_id for the
    [THIS SCENE SO FAR] block."""
    if isinstance(prose_part.next, ContinueScene):
        next_kind = "continue"
        options = list(prose_part.next.options)
    elif isinstance(prose_part.next, EndScene):
        next_kind = "end_scene"
        options = []
    else:  # EndGame
        next_kind = "end_game"
        options = []
    state.dialogue_history.append(
        TurnRecord(
            scene_id=state.scene_id,
            location_id=location_id,
            npcs_present=npcs_present,
            prose=prose,
            applied_state_changes=list(applied_state_changes),
            next_kind=next_kind,
            options=options,
        )
    )
