"""Scene-loop orchestration.

Wires Phase 1 + the two Phase 2 calls (prose writer, choice author)
together with silent fallbacks. engine.py is pure functions; agents.py is
the LLM. This module crosses the boundary — sequencing phase 1 → apply →
prose writer → choice author → assemble ProsePart → append TurnRecord,
and recovering from final-failure pydantic-ai exceptions.

The fallback policy:
  - Phase 1 retries exhausted → drop the bundle (empty state_changes).
  - Prose writer retries exhausted → use a generic prose string. Not
    persisted to the runtime's message history (voice continuity).
  - Choice author retries exhausted → use a generic Continue with two
    safe options.
  - Departure prose retries exhausted → use a generic threshold line.

Scene exits flow through `close_scene`, called from `_close_via_departure`.
Two triggers:
  - the player raises `LeaveRequested` from `pick_option` / `pick_free_text`
    (driven by POST /leave on the SSE backend);
  - the engine's `MAX_TURNS_PER_SCENE` guardrail fires inside `play_scene`.
Both write a short "moment of leaving" prose, then `close_scene` resolves
which ending (if any) is satisfied via `resolved_ending_id_at_close` and
assembles EndGame or EndScene. The LLM never emits an `ending_id`.

Server-side warnings via logfire; never bubble engine metatext to the
player.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Union

import logfire
from pydantic_ai.exceptions import UnexpectedModelBehavior

from .agents import (
    AgentBundle,
    ChroniclerRuntime,
    SceneRunnerRuntime,
    emit_chronicle_entry,
    emit_choices,
    emit_departure_prose,
    emit_placements,
    emit_prose_text,
    emit_state_changes,
)
from .engine import (
    apply_state_changes,
    append_turn_record,
    enter_scene,
    resolved_ending_id_at_close,
)
from .schemas import (
    ChoiceBundle,
    ChronicleEntry,
    Continue,
    EndGame,
    EndScene,
    FreeText,
    NpcPlacementBundle,
    Option,
    ProsePart,
    StateChangeBundle,
)
from .sse import (
    _words_from_text,
    free_text_request_event,
    game_over_event,
    location_change_event,
    npc_moved_event,
    options_event,
    paced_clause_emitter,
    scene_exit_event,
    scene_start_event,
    state_change_event,
)
from .state import GameOver, State
from .world import World


EmitEvent = Callable[[dict], Awaitable[None]]


async def _emit_paced(
    emit_event: EmitEvent | None,
    text: str,
    *,
    event_kind: str,
    extra: dict | None = None,
) -> None:
    """No-op if `emit_event` is None (CLI / test drivers don't take an
    emitter); otherwise stream `text` as paced clause events."""
    if emit_event is None:
        return
    await paced_clause_emitter(
        _words_from_text(text),
        emit_event,
        event_kind=event_kind,
        extra=extra,
    )


# Callback shapes: sync OR async. The driver picks. CLI uses sync
# `input()`; SSE backend uses async (await a Future resolved by POST).
# `pick_option` and `pick_free_text` may raise `LeaveRequested` to signal
# that the player chose to leave the scene rather than pick/type — the
# scene loop catches and runs the departure flow.
PickOption = Callable[["ProsePart"], Union[int, Awaitable[int]]]
PickFreeText = Callable[["FreeText"], Union[str, Awaitable[str]]]
PickWalk = Callable[[State, World], Union[str, Awaitable[str]]]
OnChronicleEntry = Callable[
    [State, ChronicleEntry], Union[None, Awaitable[None]]
]
OnSceneClose = Callable[["ProsePart"], Union[None, Awaitable[None]]]


# Soft cap. After this many turns without a player-driven close, the
# engine synthesises EndScene via the departure-prose flow. Well past the
# 4-8 turn target so it never trips on healthy play; exists purely as a
# backstop against runaway Continue loops.
MAX_TURNS_PER_SCENE = 10


class LeaveRequested(Exception):
    """Raised by `pick_option` / `pick_free_text` when the player chose
    to leave the scene rather than pick or type. `play_scene` catches it
    and runs the departure-prose + close_scene path. Uses the existing
    callback await rather than threading a separate "leave channel"
    through the engine."""


@contextmanager
def _root_span(name: str, **attrs):
    """Open a logfire span that starts a NEW root trace, severing the
    OTEL context this is invoked under.

    Why: scenes/turns/chronicler all execute inside an asyncio.Task
    spawned from POST /start. Without detaching, OTEL nests every span
    they create under the long-dead POST /start request span — turns
    minutes apart all collapse into one bloated request trace. The
    `attach_context({})` call clears the captured context so the span
    inside becomes a root.
    """
    with logfire.attach_context({}), logfire.span(name, **attrs):
        yield


async def _maybe_await(value):
    """If the callable returned an awaitable, await it. Otherwise return
    as-is. Keeps the same call sites usable by sync (test stub, CLI) and
    async (SSE backend / future-resolution) callbacks alike."""
    if inspect.isawaitable(value):
        return await value
    return value


def close_scene(world: World, state: State, prose: str) -> ProsePart:
    """Synthesise the terminal ProsePart when a scene closes. EndGame if
    an ending's prereqs are satisfied at this point; otherwise EndScene
    (between-scenes walk follows).

    Called from `_close_via_departure` — the single path that ends a
    scene, used by both the player /leave action and the engine's
    `MAX_TURNS_PER_SCENE` guardrail. The LLM never emits an `ending_id`;
    the engine resolves it from `state.unlocked_beats`.
    """
    ending_id = resolved_ending_id_at_close(world, state)
    if ending_id is not None:
        return ProsePart(prose=prose, next=EndGame(ending_id=ending_id))
    return ProsePart(prose=prose, next=EndScene())


def _fallback_prose(*, departure: bool = False) -> str:
    """Engine-authored prose when the prose writer's retries exhaust.
    Two flavours: a general in-scene pause, and a threshold-only line for
    the closing turn. Both look like ordinary prose to the player."""
    if departure:
        return "You stand. The air shifts as you step toward the door."
    return "A pause. The moment passes without movement."


def _fallback_choices() -> ChoiceBundle:
    """Engine-authored ChoiceBundle when the choice author's retries
    exhaust. Generic Continue with two safe options."""
    return ChoiceBundle(next=Continue(options=[
        Option(text="Wait."),
        Option(text="Look around."),
    ]))


async def _close_via_departure(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    *,
    emit_event: EmitEvent | None,
) -> ProsePart:
    """Close the current scene with a short 'moment of leaving' prose.
    Single path used by both the player's /leave action (raised
    `LeaveRequested` from a pick callback) and the engine's turn-count
    guardrail. Emits the prose word-by-word over SSE (when supplied),
    appends a `TurnRecord`, then assembles EndScene / EndGame via
    `close_scene`. Fires the matching scene-end SSE event.
    """
    with _root_span("scene_close", scene_id=state.scene_id, location=state.player_location):
        try:
            prose = await emit_departure_prose(bundle, runtime, state)
        except UnexpectedModelBehavior as e:
            logfire.warn(
                "departure_prose fallback: retries exhausted; emitting generic prose",
                error=str(e),
            )
            prose = _fallback_prose(departure=True)

        append_turn_record(state, prose=prose, applied_state_changes=[])

        await _emit_paced(emit_event, prose, event_kind="prose_chunk")

        prose_part = close_scene(bundle.world, state, prose)

        if emit_event is not None:
            if isinstance(prose_part.next, EndGame):
                await emit_event(game_over_event(prose_part.next.ending_id))
            else:
                await emit_event(scene_exit_event())

        return prose_part


async def run_turn(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
) -> ProsePart:
    """One full turn end-to-end. Returns the engine-assembled ProsePart
    so the caller can dispatch on `prose_part.next` — always Continue or
    FreeText, since the choice author cannot emit a close. Scene closure
    is handled by `_close_via_departure`, called separately from
    `play_scene` on player leave or turn-count cap.

    State is mutated: state_changes applied, dialogue_history appended.
    Prose writer's conversation history (`runtime.prose_writer_message_history`)
    is mutated on successful prose-writer call; left UNCHANGED on
    fallback.
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

    # ── Phase 2a: prose writer ─────────────────────────────────────────────
    try:
        prose = await emit_prose_text(bundle, runtime, state, applied)
    except UnexpectedModelBehavior as e:
        # Why: fallback prose is engine-authored, generic, and intentionally
        # NOT appended to runtime.prose_writer_message_history. Persisting
        # it would taint the next turn's conversation context with a
        # stand-in line that the narrator didn't actually write — breaking
        # voice continuity. Player still sees the fallback prose; the
        # conversation just behaves as if the failed turn didn't exist on
        # the next call.
        logfire.warn(
            "prose_writer fallback: retries exhausted; emitting generic prose",
            error=str(e),
        )
        prose = _fallback_prose()

    # ── Phase 2b: choice author ────────────────────────────────────────────
    try:
        choices = await emit_choices(bundle, state, prose)
    except UnexpectedModelBehavior as e:
        logfire.warn(
            "choice_author fallback: retries exhausted; emitting generic Continue",
            error=str(e),
        )
        choices = _fallback_choices()

    prose_part = ProsePart(prose=prose, next=choices.next)

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


async def play_scene(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    *,
    pick_option: PickOption,
    pick_free_text: PickFreeText | None = None,
    emit_event: EmitEvent | None = None,
) -> ProsePart:
    """Run a scene from the current player_location until EndScene or
    EndGame. `pick_option(prose_part) -> int` returns the index of the
    player's pick when a Continue is emitted. `pick_free_text(req)
    -> str` returns the typed string when a FreeText is emitted;
    required only if the choice author may emit FreeText (eval / test
    drivers that don't expect it can leave it None).

    If `emit_event` is supplied, the scene also emits SSE events at each
    boundary (scene_start, state_change, prose_chunk, options /
    free_text_request / scene_exit / game_over). Prose is paced clause
    by clause by `paced_clause_emitter` so the SSE client gets steady
    cadence regardless of LLM speed.

    Scene-entry sequence: (1) enter_scene → init scene_id, clear
    dialogue_history, set player_location. (2) populator (emit_placements)
    decides NPC overrides; we apply them and emit npc_moved events.
    (3) reset prose writer message history. (4) first run_turn.

    On populator failure (retries exhausted), defaults stand — scene
    continues without crashing.

    Returns the terminating ProsePart (EndScene or EndGame).
    """
    enter_scene(state, bundle.world, state.player_location)

    # ── populator: resolve scene-entry NPC placements ──
    with _root_span("scene_enter", scene_id=state.scene_id, location=state.player_location):
        try:
            placements = await emit_placements(bundle, state)
        except UnexpectedModelBehavior as e:
            logfire.warn(
                "populator fallback: retries exhausted; using default placements",
                error=str(e),
            )
            placements = NpcPlacementBundle()
        await _apply_placements(state, bundle, placements, emit_event=emit_event)

        runtime.prose_writer_message_history = []
        if emit_event is not None:
            loc = bundle.world.locations[state.player_location]
            await emit_event(scene_start_event(state.scene_id, loc.id, loc.name))

    while True:
        # Soft cap fires before the LLM call so we never waste tokens
        # on a Phase 2 we'd immediately discard.
        if len(state.dialogue_history) >= MAX_TURNS_PER_SCENE:
            logfire.info(
                "scene turn cap hit; engine-driven close",
                scene_id=state.scene_id,
                turns=len(state.dialogue_history),
            )
            return await _close_via_departure(
                bundle, runtime, state, emit_event=emit_event,
            )

        # The pick_option / pick_free_text wait below stays OUT of the
        # turn span — span duration should measure LLM work, not the idle
        # wait for the player to click.
        with _root_span(
            "turn",
            scene_id=state.scene_id,
            turn_n=len(state.dialogue_history) + 1,
            location=state.player_location,
        ):
            prose_part = await run_turn(bundle, runtime, state)

            # ── emit state_change events ──
            if emit_event is not None:
                last_turn = state.dialogue_history[-1]
                for change in last_turn.applied_state_changes:
                    await emit_event(state_change_event(
                        kind="unlock_beat", payload={"beat_id": change.beat_id}
                    ))

        # ── pace prose clauses ──
        await _emit_paced(emit_event, prose_part.prose, event_kind="prose_chunk")

        # ── dispatch on next ──
        if isinstance(prose_part.next, Continue):
            if emit_event is not None:
                await emit_event(options_event([
                    {"text": o.text} for o in prose_part.next.options
                ]))
            try:
                idx = await _maybe_await(pick_option(prose_part))
            except LeaveRequested:
                return await _close_via_departure(
                    bundle, runtime, state, emit_event=emit_event,
                )
            picked = prose_part.next.options[idx]
            state.dialogue_history[-1].player_pick = picked
            continue

        if isinstance(prose_part.next, FreeText):
            if emit_event is not None:
                await emit_event(free_text_request_event(prose_part.next))
            assert pick_free_text is not None, (
                "choice author emitted FreeText but no pick_free_text "
                "callback was supplied"
            )
            try:
                typed = await _maybe_await(pick_free_text(prose_part.next))
            except LeaveRequested:
                return await _close_via_departure(
                    bundle, runtime, state, emit_event=emit_event,
                )
            state.dialogue_history[-1].player_pick = Option(text=typed)
            continue

        # Future schema additions shouldn't loop. Close cleanly instead.
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
    pick_free_text: PickFreeText | None = None,
    on_chronicle_entry: OnChronicleEntry | None = None,
    on_scene_close: OnSceneClose | None = None,
    emit_event: EmitEvent | None = None,
) -> ProsePart:
    """Play one full session (multiple scenes) from `state` until a
    close resolves to EndGame or the player gets stranded at a location
    with no adjacents. Returns the terminal ProsePart (always EndGame in
    the normal path).

    Scenes close via `_close_via_departure` — triggered by the player
    raising `LeaveRequested` from a pick callback (POST /leave on the
    SSE backend) or by the `MAX_TURNS_PER_SCENE` guardrail.

    Three policy callbacks make this driver-agnostic — supply CLI/stdin
    variants from `__main__.py`, SSE/HTTP variants from `main.py`'s
    FastAPI routes, or scripted/random/LLM-as-player variants from an
    eval harness. The shape is the same.

    Args:
        pick_option: called when a Continue needs a player pick.
            Receives the just-streamed ProsePart; returns the index of
            the chosen option.
        pick_walk: called between scenes (after EndScene) to choose
            which adjacent location to walk to. Receives the live state
            and world; returns the destination location id (MUST be in
            the current location's adjacent_to).
        pick_free_text: called when the choice author emits a FreeText
            variant. Receives the FreeText; returns the typed string.
            Required only if the choice author may emit FreeText;
            pass-through to play_scene.
        on_chronicle_entry: optional callback fired after each scene-end
            chronicler call.
        on_scene_close: optional callback fired with the terminal
            ProsePart (EndScene or EndGame) BEFORE the chronicler runs.

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
            pick_free_text=pick_free_text,
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
        with _root_span("chronicler_call", scene_id=state.scene_id):
            output = await emit_chronicle_entry(bundle, chronicler, world, state)
        state.case_file_entries.append(output.case_file_entry)
        for upd in output.npc_memory_updates:
            state.npc_memory[upd.npc_id].append(upd.note)
        if on_chronicle_entry is not None:
            await _maybe_await(on_chronicle_entry(state, output))

        # Pace the case-file entry's clauses out so the sidebar fills in
        # cadence with the prose stream.
        await _emit_paced(
            emit_event,
            output.case_file_entry,
            event_kind="log_entry_chunk",
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
