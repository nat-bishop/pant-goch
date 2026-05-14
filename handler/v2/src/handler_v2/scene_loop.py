"""Scene-loop orchestration.

Wires the two phases together with silent fallbacks (per plan §"Drift
mitigation"). engine.py is pure functions; agents.py is the LLM. This
module crosses the boundary — sequencing phase 1 → apply → phase 2,
appending TurnRecord, and recovering from final-failure pydantic-ai
exceptions.

The fallback policy:
  - Phase 1 retries exhausted → drop the bundle (empty state_changes).
  - Phase 2 retries exhausted → emit a generic prose + next:
      - leave_pick set → EndScene (force scene to end at the player's
        chosen adjacent).
      - otherwise → ContinueScene with two generic options ("Wait",
        "Look around") + a leave option for an arbitrary adjacent.

Server-side warnings via logfire; never bubble engine metatext to the
player (per plan §"Drift mitigation").
"""
from __future__ import annotations

from collections.abc import Callable

import logfire
from pydantic_ai.exceptions import UnexpectedModelBehavior

from .agents import AgentBundle, emit_prose, emit_state_changes
from .engine import (
    apply_state_changes,
    append_turn_record,
    enter_scene,
)
from .prompts import build_prompt_prefix
from .schemas import (
    ContinueScene,
    EndScene,
    Option,
    ProsePart,
    StateChangeBundle,
)
from .state import State


async def run_turn(
    bundle: AgentBundle,
    state: State,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
    leave_pick: Option | None = None,
) -> ProsePart:
    """One full turn end-to-end. Returns the ProsePart so the caller can
    dispatch on `prose_part.next` (ContinueScene → wait for player pick;
    EndScene → close + maybe location_change; EndGame → terminal).

    State is mutated: state_changes applied, dialogue_history appended.
    """
    # Build the byte-identical prefix once and pass it to both phases — the
    # phase 1 / phase 2 prompts share everything up to the divergence point
    # (per plan §"Context construction"). Halves prompt-string construction
    # per turn and gives the server's KV cache the same bytes both calls.
    prefix = build_prompt_prefix(
        state, bundle.world,
        case_file_entries=case_file_entries,
        opening_seed=opening_seed,
    )

    # ── Phase 1 ────────────────────────────────────────────────────────────
    try:
        applied = await emit_state_changes(bundle, state, prefix=prefix)
    except UnexpectedModelBehavior as e:
        logfire.warn(
            "phase1 fallback: retries exhausted; dropping state_changes",
            error=str(e),
        )
        applied = StateChangeBundle()
    apply_state_changes(applied, state)

    # ── Phase 2 ────────────────────────────────────────────────────────────
    try:
        prose_part = await emit_prose(
            bundle, state, applied,
            prefix=prefix,
            leave_pick=leave_pick,
        )
    except UnexpectedModelBehavior as e:
        logfire.warn(
            "phase2 fallback: retries exhausted; emitting generic prose",
            error=str(e),
        )
        prose_part = _fallback_prose_part(state, bundle, leave_pick=leave_pick)

    # ── Append turn record ────────────────────────────────────────────────
    npcs_present = [n.id for n in bundle.world.npcs_at(state.player_location)]
    append_turn_record(
        state,
        location_id=state.player_location,
        npcs_present=npcs_present,
        prose=prose_part.prose,
        applied_state_changes=list(applied.state_changes),
        prose_part=prose_part,
    )

    return prose_part


def _fallback_prose_part(
    state: State,
    bundle: AgentBundle,
    *,
    leave_pick: Option | None = None,
) -> ProsePart:
    """Engine-authored fallback when phase 2 retries exhaust. Never shown
    as engine metatext — looks like ordinary prose to the player.
    Plan §"Failed-ending fallback": silent generic options."""
    if leave_pick is not None:
        # Force scene to end at the chosen adjacent.
        return ProsePart(
            prose="A pause. The conversation has run its course for now.",
            next=EndScene(),
        )
    location = bundle.world.locations[state.player_location]
    leave_options = [
        Option(text=f"Walk to {bundle.world.locations[adj].name}", leaves_for=adj)
        for adj in location.adjacent_to
    ]
    if not leave_options:
        # Stranded location — only EndScene is valid.
        return ProsePart(
            prose="A pause. There's nothing more to do here right now.",
            next=EndScene(),
        )
    options: list[Option] = [
        Option(text="Wait."),
        Option(text="Look around."),
        leave_options[0],
    ]
    return ProsePart(
        prose="A pause. The moment passes without movement.",
        next=ContinueScene(options=options),
    )


async def play_scene(
    bundle: AgentBundle,
    state: State,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
    pick_option: Callable[[ProsePart], int],
) -> ProsePart:
    """Run a scene from the current player_location until EndScene or
    EndGame. `pick_option(prose_part) -> int` is the caller-supplied
    interaction (CLI, frontend, test harness) that returns the index of
    the player's pick when a ContinueScene is emitted.

    Returns the terminating ProsePart (EndScene or EndGame).
    """
    enter_scene(state, bundle.world, state.player_location)
    seed = opening_seed
    leave_pick: Option | None = None
    while True:
        prose_part = await run_turn(
            bundle, state,
            case_file_entries=case_file_entries,
            opening_seed=seed,
            leave_pick=leave_pick,
        )
        seed = None  # only on first turn

        if isinstance(prose_part.next, ContinueScene):
            idx = pick_option(prose_part)
            picked = prose_part.next.options[idx]
            # Stamp the pick onto the just-appended turn record.
            state.dialogue_history[-1].player_pick = picked
            leave_pick = picked if picked.leaves_for else None
            continue

        # EndScene or EndGame — terminal for this scene.
        return prose_part
