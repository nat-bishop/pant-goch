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

import inspect
from collections.abc import Awaitable, Callable
from typing import Union

import logfire
from pydantic_ai.exceptions import UnexpectedModelBehavior

from .agents import (
    AgentBundle,
    ChroniclerRuntime,
    SceneRunnerRuntime,
    emit_chronicle_entry,
    emit_placements,
    emit_prose,
    emit_state_changes,
)
from .engine import (
    apply_state_changes,
    append_turn_record,
    enter_scene,
)
from .schemas import (
    ChronicleEntry,
    ContinueScene,
    EndGame,
    EndScene,
    NpcPlacementBundle,
    Option,
    ProsePart,
    StateChangeBundle,
)
from .sse import (
    _words_from_text,
    game_over_event,
    location_change_event,
    npc_moved_event,
    options_event,
    paced_word_emitter,
    scene_exit_event,
    scene_start_event,
    state_change_event,
)
from .state import GameOver, State
from .world import World


EmitEvent = Callable[[dict], Awaitable[None]]


# Callback shape: sync OR async. The driver picks. CLI uses sync `input()`;
# SSE backend uses async (await a Future resolved by POST /pick).
PickOption = Callable[["ProsePart"], Union[int, Awaitable[int]]]
PickWalk = Callable[[State, World], Union[str, Awaitable[str]]]
OnChronicleEntry = Callable[
    [State, ChronicleEntry], Union[None, Awaitable[None]]
]
OnSceneClose = Callable[["ProsePart"], Union[None, Awaitable[None]]]


async def _maybe_await(value):
    """If the callable returned an awaitable, await it. Otherwise return
    as-is. Keeps the same call sites usable by sync (test stub, CLI) and
    async (SSE backend / future-resolution) callbacks alike."""
    if inspect.isawaitable(value):
        return await value
    return value


async def run_turn(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    *,
    leave_pick: Option | None = None,
) -> ProsePart:
    """One full turn end-to-end. Returns the ProsePart so the caller can
    dispatch on `prose_part.next` (ContinueScene → wait for player pick;
    EndScene → close + maybe location_change; EndGame → terminal).

    State is mutated: state_changes applied, dialogue_history appended.
    Phase 2 conversation history (`runtime.phase2_message_history`) is
    mutated on successful Phase 2; left UNCHANGED on Phase 2 fallback.
    """
    # ── Phase 1 ────────────────────────────────────────────────────────────
    try:
        applied = await emit_state_changes(bundle, state)
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
            bundle, runtime, state, applied, leave_pick=leave_pick,
        )
    except UnexpectedModelBehavior as e:
        # Why: fallback prose is engine-authored, generic, and intentionally
        # NOT appended to runtime.phase2_message_history. Persisting it would
        # taint the next turn's conversation context with a stand-in line that
        # the narrator didn't actually write — breaking voice continuity.
        # Player still sees the fallback prose; the conversation just behaves
        # as if the failed turn didn't exist on the next call.
        logfire.warn(
            "phase2 fallback: retries exhausted; emitting generic prose",
            error=str(e),
        )
        prose_part = _fallback_prose_part(state, bundle, leave_pick=leave_pick)

    append_turn_record(
        state,
        prose=prose_part.prose,
        applied_state_changes=list(applied.state_changes),
    )

    return prose_part


async def _apply_placements(
    state: State,
    bundle: AgentBundle,
    placements: NpcPlacementBundle,
    *,
    emit_event: EmitEvent | None,
) -> None:
    """Apply the populator's NPC placements to `state.npc_locations` and
    fire one `npc_moved` SSE event per actual change. Silently ignores
    no-op placements (NPC moved to where they already were). Defensive
    against unknown npc_ids / location_ids — those shouldn't reach here
    given the populator's grammar-narrowed Literal, but skip rather than
    crash.
    """
    for placement in placements.placements:
        if placement.npc_id not in bundle.world.npcs:
            continue
        if placement.location_id not in bundle.world.locations:
            continue
        npc = bundle.world.npcs[placement.npc_id]
        prev = state.npc_locations.get(placement.npc_id, npc.default_location)
        if prev == placement.location_id:
            continue
        state.npc_locations[placement.npc_id] = placement.location_id
        if emit_event is not None:
            await emit_event(npc_moved_event(
                npc_id=placement.npc_id,
                from_id=prev,
                to_id=placement.location_id,
            ))


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
    runtime: SceneRunnerRuntime,
    state: State,
    *,
    pick_option: PickOption,
    emit_event: EmitEvent | None = None,
) -> ProsePart:
    """Run a scene from the current player_location until EndScene or
    EndGame. `pick_option(prose_part) -> int` is the caller-supplied
    interaction (CLI, frontend, test harness, eval policy) that returns
    the index of the player's pick when a ContinueScene is emitted.

    If `emit_event` is supplied, the scene also emits SSE events at each
    boundary (scene_start, state_change, prose_word, options/scene_exit/
    game_over). Prose words are paced at 80 ms by `paced_word_emitter`
    so the SSE client gets steady cadence regardless of LLM speed.

    When `emit_event` is None (tests, CLI), no events fire and no
    pacing happens — caller is responsible for surfacing the prose.

    Scene-entry sequence: (1) enter_scene → init scene_id, clear
    dialogue_history, set player_location. (2) populator (emit_placements)
    decides NPC overrides; we apply them and emit npc_moved events.
    (3) reset Phase 2 message history. (4) first run_turn.

    On populator failure (retries exhausted), defaults stand — scene
    continues without crashing.

    Returns the terminating ProsePart (EndScene or EndGame).
    """
    enter_scene(state, bundle.world, state.player_location)

    # ── populator: resolve scene-entry NPC placements ──
    try:
        placements = await emit_placements(bundle, state)
    except UnexpectedModelBehavior as e:
        logfire.warn(
            "populator fallback: retries exhausted; using default placements",
            error=str(e),
        )
        placements = NpcPlacementBundle()
    await _apply_placements(state, bundle, placements, emit_event=emit_event)

    runtime.phase2_message_history = []
    if emit_event is not None:
        loc = bundle.world.locations[state.player_location]
        await emit_event(scene_start_event(state.scene_id, loc.id, loc.name))

    leave_pick: Option | None = None
    while True:
        prose_part = await run_turn(
            bundle, runtime, state, leave_pick=leave_pick,
        )

        # ── emit state_change events ──
        if emit_event is not None:
            last_turn = state.dialogue_history[-1]
            for change in last_turn.applied_state_changes:
                await emit_event(state_change_event(
                    kind="unlock_beat", payload={"beat_id": change.beat_id}
                ))

        # ── pace prose words ──
        if emit_event is not None:
            await paced_word_emitter(
                _words_from_text(prose_part.prose),
                emit_event,
                event_kind="prose_word",
            )

        # ── dispatch on next ──
        if isinstance(prose_part.next, ContinueScene):
            if emit_event is not None:
                await emit_event(options_event([
                    {"text": o.text, "leaves_for": o.leaves_for}
                    for o in prose_part.next.options
                ]))
            idx = await _maybe_await(pick_option(prose_part))
            picked = prose_part.next.options[idx]
            # Stamp the pick onto the just-appended turn record.
            state.dialogue_history[-1].player_pick = picked
            leave_pick = picked if picked.leaves_for else None
            continue

        # EndScene or EndGame — terminal for this scene.
        if emit_event is not None:
            if isinstance(prose_part.next, EndGame):
                await emit_event(game_over_event(prose_part.next.ending_id))
            else:
                await emit_event(scene_exit_event())
        return prose_part


# ───────────────────────── session loop ─────────────────────────


async def play_session(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    chronicler: ChroniclerRuntime,
    state: State,
    world: World,
    *,
    pick_option: PickOption,
    pick_walk: PickWalk,
    on_chronicle_entry: OnChronicleEntry | None = None,
    on_scene_close: OnSceneClose | None = None,
    emit_event: EmitEvent | None = None,
) -> ProsePart:
    """Play one full session (multiple scenes) from `state` until the
    LLM fires EndGame or the player gets stranded at a location with no
    adjacents. Returns the terminal ProsePart (always EndGame in the
    normal path).

    Two policy callbacks make this driver-agnostic — supply CLI/stdin
    variants from `__main__.py`, SSE/HTTP variants from `main.py`'s
    FastAPI routes, or scripted/random/LLM-as-player variants from an
    eval harness. The shape is the same.

    Args:
        pick_option: called when a ContinueScene needs a player pick.
            Receives the just-streamed ProsePart; returns the index of
            the chosen option.
        pick_walk: called between scenes (after EndScene) to choose
            which adjacent location to walk to. Receives the live state
            and world; returns the destination location id (MUST be in
            the current location's adjacent_to).
        on_chronicle_entry: optional callback fired after each scene-end
            chronicler call. Receives `(state, ChronicleEntry)`; useful
            for eval traces, SSE streaming, or printing in the CLI. The
            engine has already merged case_file_entry into
            `state.case_file_entries` and per-NPC notes into
            `state.npc_memory` by the time the callback fires.
        on_scene_close: optional callback fired with the terminal
            ProsePart (EndScene or EndGame) BEFORE the chronicler runs.
            Lets the driver surface the closing prose to the player.
            ContinueScene prose is surfaced via pick_option, which sees
            every non-terminal ProsePart.

    Side effects:
        - Mutates `state` (location, scene_id, dialogue_history,
          unlocked_beats, npc_locations, npc_memory, case_file_entries,
          game_over).
        - Mutates `chronicler.message_history`.
    """
    while state.game_over is None:
        terminal = await play_scene(
            bundle, runtime, state,
            pick_option=pick_option,
            emit_event=emit_event,
        )

        # Surface the closing prose before the chronicler runs — the
        # player should read the scene's last line before the case file
        # gets its commentary.
        if on_scene_close is not None:
            await _maybe_await(on_scene_close(terminal))

        # Chronicler fires after every scene close — both EndScene and
        # EndGame. The just-completed scene is still in dialogue_history;
        # the next enter_scene will clear it. Decompose into the two
        # state channels right here so the callback sees a settled state.
        output = await emit_chronicle_entry(chronicler, world, state)
        state.case_file_entries.append(output.case_file_entry)
        for upd in output.npc_memory_updates:
            state.npc_memory[upd.npc_id].append(upd.note)
        if on_chronicle_entry is not None:
            await _maybe_await(on_chronicle_entry(state, output))

        # Pace the case-file entry's words out as SSE log_entry_word
        # events. Frontend renders into the sidebar in cadence with the
        # prose stream.
        if emit_event is not None:
            await paced_word_emitter(
                _words_from_text(output.case_file_entry),
                emit_event,
                event_kind="log_entry_word",
                extra={"scene_id": state.scene_id},
            )

        if isinstance(terminal.next, EndGame):
            state.game_over = GameOver(ending_id=terminal.next.ending_id)
            return terminal

        # EndScene — between-scenes. Caller picks adjacent.
        location = world.locations[state.player_location]
        if not location.adjacent_to:
            # Stranded — only EndScene was reachable from a dead-end
            # location and the player can't move. Return the terminal
            # ProsePart so the caller can render whatever closing UI it
            # wants. game_over stays None.
            return terminal

        to_location = await _maybe_await(pick_walk(state, world))
        assert to_location in location.adjacent_to, (
            f"pick_walk returned {to_location!r}, not in "
            f"{location.adjacent_to!r}"
        )
        if emit_event is not None:
            await emit_event(location_change_event(
                state.player_location, to_location
            ))
        state.player_location = to_location

    # game_over set externally (defensive — current engine only sets it
    # via the EndGame path above, but be permissive).
    return terminal  # type: ignore[possibly-undefined]
