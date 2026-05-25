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

Scene exits flow through the choice author's Leave variant: the engine
resolves which ending (if any) is satisfied via
`resolved_ending_id_at_close`, and assembles EndGame or EndScene
accordingly. The LLM never emits an ending_id.

Server-side warnings via logfire; never bubble engine metatext to the
player.
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Union

import logfire
from pydantic_ai.exceptions import UnexpectedModelBehavior

from .agents import (
    AgentBundle,
    SceneRunnerRuntime,
    emit_caption,
    emit_chronicle_entry,
    emit_choices,
    emit_closing_prose,
    emit_free_text_gate,
    emit_placements,
    emit_state_changes,
    emit_village_attention,
    stream_prose_text,
)
from .engine import (
    FREE_TEXT_FALLBACK,
    apply_state_changes,
    apply_village_attention,
    append_turn_record,
    band_name,
    effective_deltas,
    enter_scene,
    free_text_cadence_ok,
    is_decision,
    present_npc_ids,
    resolved_ending_id_at_close,
    seed_present_npcs,
    scene_exhausted_here,
    scene_is_spent,
)
from .schemas import (
    ChoiceBundle,
    ChronicleEntry,
    Continue,
    EndGame,
    EndScene,
    FreeText,
    FreeTextCaption,
    Leave,
    NpcPlacementBundle,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from .sse import (
    _words_from_stream,
    _words_from_text,
    free_text_request_event,
    game_over_event,
    location_change_event,
    npc_moved_event,
    npc_standing_event,
    options_event,
    paced_word_emitter,
    scene_exit_event,
    scene_spent_event,
    scene_start_event,
    state_change_event,
    walk_ready_event,
)
from .state import GameOver, State
from .world import World


EmitEvent = Callable[[dict], Awaitable[None]]


# ── dossier "People" standings ──
# Per-NPC confession band, surfaced on the frontend dossier as the journalist's
# read on a person (see sse.npc_standing_event). Only NPCs with an authored
# `disposition` are tracked — auxiliary NPCs stay pure prose and off the board.
# Emitted at scene entry (seed) and on band CHANGE within a scene (not per turn).


def _dispositioned_present(world: World, state: State) -> list[str]:
    """Present NPC ids that carry a `disposition` — the only ones with a tracked
    relationship, hence the only ones on the dossier board."""
    return [
        nid for nid in present_npc_ids(world, state)
        if world.npcs[nid].disposition is not None
    ]


def _impression_for(world: World, state: State, npc_id: str, band: str) -> str:
    """The authored journalist's impression for `npc_id` at `band`. "" when the
    band is unauthored (the card then shows no impression line)."""
    disp = world.npcs[npc_id].disposition
    return disp.impression_by_band.get(band, "") if disp else ""


async def _emit_standing(
    world: World, state: State, npc_id: str, emit_event: EmitEvent
) -> None:
    band = band_name(world, state, npc_id)
    await emit_event(
        npc_standing_event(npc_id, band, _impression_for(world, state, npc_id, band))
    )


# ── player action: the single input a scene awaits ──
# One tagged union covers every way a turn can resolve: pick a canned
# option, type a free-text move, or walk out via the map (mid-scene or at
# a soft-close). `request_player_input` returns one of these. The map is
# live at every input-await point, so a WalkTo can end any turn.
@dataclass(frozen=True)
class PickOption:
    """Player chose the canned option at `index` from a Continue."""

    index: int


@dataclass(frozen=True)
class TypeText:
    """Player typed their own move (the free-text box)."""

    text: str


@dataclass(frozen=True)
class WalkTo:
    """Player clicked the map to leave for `location_id` (always adjacent
    to the current location). Ends the scene from any turn."""

    location_id: str


PlayerAction = Union[PickOption, TypeText, WalkTo]


# Sync OR async. The driver picks. CLI parses one stdin line; the SSE
# backend awaits a single Future resolved by POST /pick|/walk|/free_text;
# tests/evals script it.
RequestPlayerInput = Callable[
    ["ProsePart", State, World],
    Union[PlayerAction, Awaitable[PlayerAction]],
]
OnChronicleEntry = Callable[
    [State, ChronicleEntry], Union[None, Awaitable[None]]
]
OnSceneClose = Callable[["ProsePart"], Union[None, Awaitable[None]]]


@dataclass(frozen=True)
class RunConfig:
    """Per-session engine feature flags. Threaded through `play_session` →
    `play_scene`; the eval A/B harness splats one per arm
    (`RunConfig(**arm.run_config)`), so a future knob is a new field here plus a
    new experiment file — not new plumbing.

    Currently empty. Leaving a scene is fixed to two paths — a choice-author
    `leave: true` option the player picks, or the narrator soft-close (`Leave`)
    — with no per-session variant, so there is nothing to flag. `RunConfig()` is
    the only shape there is, and the harness still accepts `run_config: {}` arms.
    """


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


async def _update_village_attention_at_close(bundle: AgentBundle, state: State) -> None:
    """Run the dedicated village-attention classifier over the just-closed scene
    and ratchet `state.village_attention` up to its verdict.

    MUST run BEFORE `resolved_ending_id_at_close` at either close path, or the
    `run_out_of_the_village` ending can't fire on the closing turn (attention is
    what gates it). On retry-exhaustion we leave the heat unchanged: the channel
    only ever ratchets up, so the safe failure direction is under-counting — a
    dropped scene never spuriously ejects the player.
    """
    if not state.dialogue_history:
        return
    try:
        level = await emit_village_attention(bundle, bundle.world, state)
    except UnexpectedModelBehavior as e:
        logfire.warn(
            "village_attention fallback: retries exhausted; leaving heat unchanged",
            error=str(e),
        )
        return
    apply_village_attention(state, level)


def _assemble_prose_part(
    world: World, state: State, prose: str, choices: ChoiceBundle,
    *, free_text_offer: bool = False, caption: FreeTextCaption | None = None,
) -> ProsePart:
    """Compose the choice author's ChoiceBundle, the resolved free-text
    offer, the caption author's output, and the prose writer's text into
    the engine-facing ProsePart.

    Precedence: **Leave > free-text > Continue**.
      - Leave → EndGame (when an ending is satisfied) or EndScene. A
        closing scene never gets a free-text box, even if free text was
        offered.
      - else, free text offered → per-turn FreeText built from the caption
        author's caption + suggested move; `FREE_TEXT_FALLBACK` when `caption` is
        None (caption author failed but the offer already stands).
      - else → the choice author's Continue, verbatim.
    """
    nxt = choices.next
    if isinstance(nxt, Leave):
        ending_id = resolved_ending_id_at_close(world, state)
        if ending_id is not None:
            return ProsePart(prose=prose, next=EndGame(ending_id=ending_id))
        return ProsePart(prose=prose, next=EndScene())
    if free_text_offer:
        free_text = (
            FreeText(prompt=caption.caption, suggested=caption.suggested)
            if caption is not None
            else FREE_TEXT_FALLBACK
        )
        return ProsePart(prose=prose, next=free_text)
    return ProsePart(prose=prose, next=nxt)


def _fallback_prose() -> str:
    """Engine-authored prose when the prose writer's retries exhaust.
    Never shown as engine metatext — looks like ordinary prose to the
    player."""
    return "A pause. The moment passes without movement."


# Hard ceiling on player turns within a single scene. A scene normally ends
# when the player leaves or the choice author soft-closes a spent scene; but
# nothing forces either, so a player who keeps picking in-scene options can
# mill indefinitely (one observed eval session ran 47 turns in one location).
# At the cap we convert the turn into a narrator soft-close (EndScene with no
# destination), so the normal leave path runs — the player picks an adjacent
# location and any already-met ending still fires on the walk. This bounds a
# session's wall-clock structurally (cap × max_scenes × per-turn cost), so the
# eval harness's wall-clock timeout is a pure hang-backstop, never a length gate.
MAX_TURNS_PER_SCENE = 20

# Minimum completed turns before the content-derived soft-close (engine.scene_is_spent)
# may fire. A scene gets at least this many turns to breathe — arrive, take the
# room in — before a dry, beat-less turn can end it.
SPENT_CLOSE_MIN_TURNS = 2


def _fallback_choices() -> ChoiceBundle:
    """Engine-authored ChoiceBundle when the choice author's retries
    exhaust. Generic Continue with two safe options."""
    return ChoiceBundle(next=Continue(options=[
        Option(text="Wait."),
        Option(text="Look around."),
    ]))


def _authored_leave_present(nxt: object) -> bool:
    """True when the choice author marked one of this turn's options as the
    diegetic exit (`leave: true`). When so, the engine lets the PLAYER pick that
    in-fiction departure rather than force-closing the scene itself: the spent
    auto-close is skipped and the generic 'Step outside' entry is not appended
    (no duplicate exits)."""
    return isinstance(nxt, Continue) and any(o.leave for o in nxt.options)


def _fallback_closing_prose() -> str:
    """Engine-authored closing line when the prose writer's retries
    exhaust on a leave. Voice-neutral; never shown as engine metatext."""
    return "You step out, and the door falls shut behind you."


async def _stream_pace_and_author(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    applied_beats: list[str],
    *,
    emit_event: EmitEvent | None,
    skip_event: asyncio.Event | None,
    leave_required: bool = False,
    npc_present: bool = True,
) -> tuple[str, object, object]:
    """Phase 2a + 2b/2c, fused so the reveal overlaps the authoring.

    The prose writer is streamed and its words paced to the client as the
    LLM produces them (the player starts reading at `prose TTFT`, not after
    the whole turn assembles). The choice author and free-text gate need the
    finished prose, so they are kicked off the *moment generation completes*
    — concurrently with the reveal's remaining 150 ms holds. The model is
    free during pacing (those are pure `asyncio.sleep`s), so the authoring
    decode overlaps the reveal and options are ready as it ends.

    `npc_present` is the structural precondition for free text: it is the
    "words to a person" channel, so when no one is present the gate
    coroutine is never scheduled (no LLM call) and `gate_res` is a literal
    `False`. Object/place moments in a solo scene are click-driven.

    Returns `(prose, choices_res, gate_res)` where the two results are the
    raw `asyncio.gather(..., return_exceptions=True)` outputs (or, for the
    gate on a solo turn, a bare `False`); the caller interprets their
    fallbacks. `prose` is the streamed text (or the engine fallback on
    prose-writer failure).

    Mechanics: a producer task drains `stream_prose_text` eagerly into a
    queue (so the LLM is NOT throttled to pacing speed by TCP back-pressure
    — that would pin the model for the whole reveal and block the authoring
    overlap). A pacer task pulls from the queue and either paces to the
    client (`emit_event` set) or drains quietly (CLI / tests / evals). The
    producer's `finally` always enqueues the `None` sentinel so the pacer
    unblocks even when the prose writer fails mid-stream.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    acc: list[str] = []

    async def _produce() -> None:
        try:
            async for delta in stream_prose_text(bundle, runtime, state, applied_beats):
                acc.append(delta)
                await queue.put(delta)
        finally:
            await queue.put(None)

    async def _queue_iter() -> AsyncIterator[str]:
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    async def _drain_quiet() -> str:
        full = ""
        async for word in _words_from_stream(_queue_iter()):
            full += word
        return full

    producer = asyncio.create_task(_produce())
    if emit_event is not None:
        pacer = asyncio.create_task(paced_word_emitter(
            _words_from_stream(_queue_iter()), emit_event,
            event_kind="prose_word", skip_event=skip_event,
        ))
    else:
        pacer = asyncio.create_task(_drain_quiet())

    try:
        try:
            await producer
        except UnexpectedModelBehavior as e:
            # Why: fallback prose is engine-authored, generic, and intentionally
            # NOT appended to runtime.prose_writer_message_history (stream_prose_text
            # skips that on failure). Persisting it would taint the next turn's
            # conversation with a line the narrator never wrote — breaking voice
            # continuity. The producer's finally already released the pacer.
            logfire.warn(
                "prose_writer fallback: retries exhausted; emitting generic prose",
                error=str(e),
            )
            await pacer  # drains whatever streamed + emits its is_last sentinel
            if acc:
                # A partial line already reached the player; keep it rather than
                # appending fallback prose on top of it.
                prose = "".join(acc)
            else:
                prose = _fallback_prose()
                if emit_event is not None:
                    await paced_word_emitter(
                        _words_from_text(prose), emit_event,
                        event_kind="prose_word", skip_event=skip_event,
                    )
            tasks = [emit_choices(bundle, state, prose, leave_required=leave_required)]
            if npc_present:
                tasks.append(emit_free_text_gate(bundle, state, prose))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            choices_res = results[0]
            gate_res = results[1] if npc_present else False  # solo → no LLM call
            return prose, choices_res, gate_res

        # Generation finished cleanly: the full prose is known and the model
        # is free. Author the choices/gate now, overlapping the reveal tail.
        # gather() schedules both coroutines immediately, so they run while we
        # await the pacer's remaining holds.
        prose = "".join(acc)
        tasks = [emit_choices(bundle, state, prose, leave_required=leave_required)]
        if npc_present:
            tasks.append(emit_free_text_gate(bundle, state, prose))
        authoring = asyncio.gather(*tasks, return_exceptions=True)
        await pacer
        results = await authoring
        choices_res = results[0]
        gate_res = results[1] if npc_present else False  # solo → no LLM call
        return prose, choices_res, gate_res
    finally:
        # Safety net: a non-UnexpectedModelBehavior escaping above must not
        # leave the pacer task pending ("Task was destroyed but is pending").
        if not pacer.done():
            pacer.cancel()
            with suppress(asyncio.CancelledError):
                await pacer


async def run_turn(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    *,
    emit_event: EmitEvent | None = None,
    skip_event: asyncio.Event | None = None,
) -> ProsePart:
    """One full turn end-to-end. Returns the engine-assembled ProsePart
    so the caller can dispatch on `prose_part.next` (Continue → wait
    for player pick; FreeText → wait for typed input; EndScene → close
    + maybe location_change; EndGame → terminal).

    State is mutated: state_changes applied, dialogue_history appended.
    Prose writer's conversation history (`runtime.prose_writer_message_history`)
    is mutated on successful prose-writer call; left UNCHANGED on
    fallback.

    When `emit_event` is supplied, the turn's SSE events are emitted here:
    `state_change` (right after Phase 1, before any prose word) and the
    paced `prose_word` stream (live, as the prose writer produces it — see
    `_stream_pace_and_author`). `skip_event` flushes the reveal at full
    speed. Both default to None for CLI / tests / evals, which drive the
    turn without SSE.
    """
    # Confession bands for present, tracked NPCs BEFORE this turn moves state —
    # so we can emit a dossier standing only for those whose band actually
    # changes (a thaw, a breach), never every turn. Present set is stable across
    # a scene, so this snapshot pairs with the post-apply read below.
    present_tracked = _dispositioned_present(bundle.world, state)
    bands_before = {
        nid: band_name(bundle.world, state, nid) for nid in present_tracked
    }

    # ── Phase 1 ────────────────────────────────────────────────────────────
    try:
        applied = await emit_state_changes(bundle, state)
    except UnexpectedModelBehavior as e:
        logfire.warn(
            "phase1 fallback: retries exhausted; dropping state_changes",
            error=str(e),
            location=state.player_location,
            present_npcs=list(present_npc_ids(bundle.world, state)),
        )
        applied = StateChangeBundle()
    # Snapshot trust/guard BEFORE applying, so `effective_deltas` can report what
    # the engine ACTUALLY did (a +1 zeroed by opens_up_only_at, a clamp-saturated
    # nudge) rather than what the classifier asked for — the transcript/metrics
    # must read realized state, not emitted intent (the Dylan "+10" that was 0).
    trust_before = dict(state.trust)
    guard_before = dict(state.guard)
    breached_before = set(state.guard_breached)
    newly_unlocked = apply_state_changes(applied, state, bundle.world)
    # Only the deltas that actually moved standing: an all-zero realized delta
    # changed nothing, so the turn reads dry (scene soft-close + filler metrics).
    moved_deltas = [
        d
        for d in effective_deltas(
            applied,
            state,
            trust_before=trust_before,
            guard_before=guard_before,
            breached_before=breached_before,
        )
        if d.trust != "0" or d.guard != "0"
    ]
    if newly_unlocked or moved_deltas:
        # One queryable record per turn that moved state: which beats unlocked
        # (commitments + engine-derived), the quality nudges actually applied, and
        # the resulting set. Pairs with the per-action logs in main.py so a
        # session's progress is reconstructable from Logfire alone.
        logfire.info(
            "state applied",
            applied_beats=newly_unlocked,
            applied_deltas=[d.model_dump() for d in moved_deltas],
            unlocked_beats=sorted(state.unlocked_beats),
            village_attention=state.village_attention,
            location=state.player_location,
        )

    # state_change events go out before any prose word, so the frontend
    # unlocks beats ahead of the reveal. Both commitments and engine-derived
    # milestones surface here, so the frontend contract is unchanged.
    if emit_event is not None:
        for beat_id in newly_unlocked:
            await emit_event(state_change_event(
                kind="unlock_beat", payload={"beat_id": beat_id}
            ))
        # Dossier standings: only NPCs whose band crossed this turn (thaw/breach).
        for nid in present_tracked:
            if band_name(bundle.world, state, nid) != bands_before[nid]:
                await _emit_standing(bundle.world, state, nid, emit_event)

    # Should the choice author offer a diegetic exit this turn? Two triggers:
    #  - a decision just fired — the player committed, so the scene winds down
    #    (one closing exchange, then a clean way out) rather than letting them
    #    keep interrogating past the turning point;
    #  - the scene is spent — this turn moved nothing AND nothing here can still
    #    pay off (mirrors scene_is_spent, but evaluated from THIS turn's movement
    #    since the turn record isn't appended until below). Completed turns after
    #    this one == len(dialogue_history)+1, matching scene_is_spent's grace read.
    # When this fires the author MUST write a `leave: true` option — with no
    # generic engine exit, the player-picked diegetic departure is the only way
    # out short of the narrator soft-close.
    decision_fired = any(is_decision(b) for b in newly_unlocked)
    this_turn_moved = bool(newly_unlocked) or bool(moved_deltas)
    leave_required = decision_fired or (
        not this_turn_moved
        and scene_exhausted_here(
            bundle.world, state, len(state.dialogue_history) + 1
        )
    )

    # Structural precondition for free text — the "words to a person" channel.
    # No one present → solo scene → the gate never runs (object/place moments
    # are click-driven). Cheap, deterministic, and shields the gate from ever
    # needing to know who is present.
    npc_present = bool(present_npc_ids(bundle.world, state))

    # ── Phase 2a + 2b/2c: stream prose live, author over the reveal tail ────
    prose, choices_res, gate_res = await _stream_pace_and_author(
        bundle, runtime, state, newly_unlocked,
        emit_event=emit_event, skip_event=skip_event,
        leave_required=leave_required, npc_present=npc_present,
    )

    if isinstance(choices_res, UnexpectedModelBehavior):
        logfire.warn(
            "choice_author fallback: retries exhausted; emitting generic Continue",
            error=str(choices_res),
        )
        choices = _fallback_choices()
    elif isinstance(choices_res, BaseException):
        raise choices_res
    else:
        choices = choices_res

    if isinstance(gate_res, UnexpectedModelBehavior):
        logfire.warn(
            "free_text_gate fallback: retries exhausted; not offering free text",
            error=str(gate_res),
        )
        gate_offered = False
    elif isinstance(gate_res, BaseException):
        raise gate_res
    else:
        gate_offered = gate_res

    # ── Phase 2d: caption author (only when free text actually surfaces) ────
    # The LLM gate only judges whether the moment hands the floor to the
    # player; the engine cadence (a cooldown — no probability) suppresses a
    # fresh box too soon after the last typed move. We apply cadence BEFORE
    # the caption call so its cost is paid only on turns that show a box. On
    # caption failure the offer still stands — _assemble falls back to a
    # fixed caption.
    free_text_offer = gate_offered and free_text_cadence_ok(state)
    caption: FreeTextCaption | None = None
    if free_text_offer:
        try:
            caption = await emit_caption(bundle, state, prose)
        except UnexpectedModelBehavior as e:
            logfire.warn(
                "caption_author fallback: retries exhausted; using fixed caption",
                error=str(e),
            )
            caption = None

    # A Leave closes the scene inside _assemble_prose_part (it resolves the
    # ending). Update village attention FIRST so a run_out can fire on this turn.
    if isinstance(choices.next, Leave):
        await _update_village_attention_at_close(bundle, state)

    prose_part = _assemble_prose_part(
        bundle.world, state, prose, choices,
        free_text_offer=free_text_offer, caption=caption,
    )

    append_turn_record(
        state,
        prose=prose_part.prose,
        applied_state_changes=[UnlockBeat(beat_id=b) for b in newly_unlocked],
        applied_deltas=moved_deltas,
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
    request_player_input: RequestPlayerInput,
    emit_event: EmitEvent | None = None,
    skip_event: asyncio.Event | None = None,
    session_id: str = "",
    run_config: RunConfig = RunConfig(),
) -> ProsePart:
    """Run a scene from the current player_location until it ends.

    Each turn, `request_player_input(prose_part, state, world)` returns one
    `PlayerAction`: a `PickOption` / `TypeText` continues the scene; a
    `WalkTo` ends it. A `WalkTo` is only on the table when leaving is — a
    soft-close (`EndScene` from the choice author's Leave, which removes the
    options and leaves the map as the only way on) or a `Continue` whose author
    marked an option `leave: true` (the player's in-fiction exit). A plain
    `Continue` / `FreeText` turn has no walk: leaving comes from the fiction.

    If `emit_event` is supplied, the scene emits SSE at each boundary
    (scene_start, state_change, the live-paced prose_word stream, then the
    input cue: options / free_text_request / scene_spent — each followed by
    walk_ready to arm the map). Prose is paced at a 150 ms floor by
    `paced_word_emitter`; `skip_event` flushes the reveal at full speed.

    Scene-entry sequence: (1) enter_scene → init scene_id, clear
    dialogue_history, set player_location. (2) populator (emit_placements)
    decides NPC overrides; we apply them and emit npc_moved events.
    (3) reset prose writer message history. (4) first run_turn.

    On populator failure (retries exhausted), defaults stand — scene
    continues without crashing.

    Returns the terminating ProsePart:
      - `EndGame` — the LLM soft-closed into a satisfied ending, or the
        player walked out while one was satisfied.
      - `EndScene(to_location=...)` — the player walked; the closing prose
        has already streamed and `to_location` names where they went.
    """
    enter_scene(state, bundle.world, state.player_location)

    # ── populator: resolve scene-entry NPC placements ──
    with _root_span("scene_enter", session_id=session_id, scene_id=state.scene_id, location=state.player_location):
        try:
            placements = await emit_placements(bundle, state)
        except UnexpectedModelBehavior as e:
            logfire.warn(
                "populator fallback: retries exhausted; using default placements",
                error=str(e),
            )
            placements = NpcPlacementBundle()
        await _apply_placements(state, bundle, placements, emit_event=emit_event)

        # Deterministic reachability: force any location-pinned NPC (Dylan) present
        # where the player can actually warm them, so their findable doesn't hinge
        # on the stay-biased populator. Done AFTER the populator so it overrides a
        # stale placement, and emits npc_moved like any other arrival.
        seed_prev = {nid: state.npc_locations.get(nid) for nid in bundle.world.npcs}
        for nid in seed_present_npcs(bundle.world, state):
            if emit_event is not None:
                await emit_event(npc_moved_event(
                    npc_id=nid,
                    from_id=seed_prev[nid] or bundle.world.npcs[nid].default_location,
                    to_id=state.player_location,
                ))

        runtime.prose_writer_message_history = []
        if emit_event is not None:
            loc = bundle.world.locations[state.player_location]
            await emit_event(scene_start_event(state.scene_id, loc.id, loc.name))
            # Seed the dossier with the journalist's read on whoever is present
            # and tracked — covers first meetings and re-syncs on every entry.
            for nid in _dispositioned_present(bundle.world, state):
                await _emit_standing(bundle.world, state, nid, emit_event)

    turns_this_scene = 0
    while True:
        # The pick_option / pick_free_text wait below stays OUT of the
        # turn span — span duration should measure the turn's work, not the
        # idle wait for the player to click. run_turn now emits the turn's
        # SSE (state_change, then the live-paced prose_word stream) itself,
        # so the span also covers the reveal pacing; per-agent LLM latency
        # is still isolated by pydantic-ai's own instrumentation spans.
        with _root_span(
            "turn",
            session_id=session_id,
            scene_id=state.scene_id,
            turn_n=state.turn_n,
            location=state.player_location,
        ):
            prose_part = await run_turn(
                bundle, runtime, state,
                emit_event=emit_event, skip_event=skip_event,
            )
        turns_this_scene += 1

        nxt = prose_part.next

        # LLM-resolved terminal: Leave + a satisfied ending. No player
        # input — the climax already committed in fiction.
        if isinstance(nxt, EndGame):
            if emit_event is not None:
                await emit_event(game_over_event(nxt.ending_id))
            return prose_part

        # Per-scene turn cap (MAX_TURNS_PER_SCENE): a scene that keeps offering
        # in-scene options never ends on its own. Once we hit the cap, convert
        # this turn into a narrator soft-close (EndScene, no destination) so the
        # spent-scene branch below runs exactly as if the choice author had
        # closed the scene — the driver surfaces the map and the player walks
        # out (and `resolved_ending_id_at_close` still fires any met ending).
        if turns_this_scene >= MAX_TURNS_PER_SCENE and isinstance(nxt, (Continue, FreeText)):
            logfire.info(
                "scene hit per-turn cap; forcing soft-close",
                scene_id=state.scene_id,
                location=state.player_location,
                turns_this_scene=turns_this_scene,
            )
            nxt = EndScene()
            prose_part = ProsePart(prose=prose_part.prose, next=nxt)

        # Content-derived soft-close (engine.scene_is_spent): end a dead room the
        # moment it is dead — no beat can still fire here, no present NPC can still
        # open up, and the last turn moved nothing — rather than milling to the cap
        # above. This is what stops the "investigate inert scenery in circles"
        # tail (e.g. the vicarage map dead-end). Same conversion as the cap: a
        # destinationless EndScene drops into the spent-scene branch below, the
        # player walks out, and any met ending still fires on the walk.
        elif (
            isinstance(nxt, (Continue, FreeText))
            and turns_this_scene >= SPENT_CLOSE_MIN_TURNS
            and scene_is_spent(bundle.world, state)
            and not _authored_leave_present(nxt)
        ):
            logfire.info(
                "scene spent; forcing soft-close",
                scene_id=state.scene_id,
                location=state.player_location,
                turns_this_scene=turns_this_scene,
            )
            nxt = EndScene()
            prose_part = ProsePart(prose=prose_part.prose, next=nxt)

        # Every other branch awaits one PlayerAction. Surface the matching
        # in-scene cue, then arm the map (walk_ready) ONLY when leaving is on the
        # table — a narrator soft-close (the player must pick a destination) or a
        # Continue carrying an authored `leave: true` option (picking it opens
        # the map to walk). On a plain Continue / free-text turn the map stays
        # inert: leaving comes from the fiction, not an always-live escape.
        if emit_event is not None:
            soft_close = not isinstance(nxt, (Continue, FreeText))
            if isinstance(nxt, Continue):
                # Carry each option's `leave` flag to the client so a diegetic
                # exit the choice author wrote renders (and routes) as the exit.
                payload = [
                    {"text": o.text, "leave": True} if o.leave else {"text": o.text}
                    for o in nxt.options
                ]
                await emit_event(options_event(payload))
            elif isinstance(nxt, FreeText):
                await emit_event(free_text_request_event(nxt))
            else:  # EndScene soft-close — scene spent, only a walk remains
                await emit_event(scene_spent_event())
            if soft_close or _authored_leave_present(nxt):
                await emit_event(walk_ready_event())

        action = await _maybe_await(
            request_player_input(prose_part, state, bundle.world)
        )

        if isinstance(action, PickOption):
            assert isinstance(nxt, Continue), "PickOption with no options offered"
            state.dialogue_history[-1].player_pick = nxt.options[action.index]
            continue
        if isinstance(action, TypeText):
            state.dialogue_history[-1].player_pick = Option(text=action.text)
            state.dialogue_history[-1].player_pick_typed = True
            continue

        # WalkTo — the player leaves; the scene ends from whatever branch.
        assert isinstance(action, WalkTo)
        # Update village attention before resolving, so walking out can trip run_out.
        await _update_village_attention_at_close(bundle, state)
        ending_id = resolved_ending_id_at_close(bundle.world, state)
        if ending_id is not None:
            # An ending's beats are already met — the game ends as they
            # leave. No destination line (they aren't arriving anywhere);
            # the last turn's prose stands as the terminal beat.
            if emit_event is not None:
                await emit_event(game_over_event(ending_id))
            return ProsePart(prose=prose_part.prose, next=EndGame(ending_id=ending_id))

        try:
            closing = await emit_closing_prose(
                bundle, runtime, state, bundle.world, action.location_id
            )
        except UnexpectedModelBehavior as e:
            logfire.warn(
                "closing_prose fallback: retries exhausted; using fixed line",
                error=str(e),
            )
            closing = _fallback_closing_prose()
        if emit_event is not None:
            await paced_word_emitter(
                _words_from_text(closing), emit_event,
                event_kind="prose_word", skip_event=skip_event,
            )
            await emit_event(scene_exit_event())
        return ProsePart(
            prose=closing, next=EndScene(to_location=action.location_id)
        )


# ───────────────────────── session loop ─────────────────────────


async def play_session(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    world: World,
    *,
    request_player_input: RequestPlayerInput,
    on_chronicle_entry: OnChronicleEntry | None = None,
    on_scene_close: OnSceneClose | None = None,
    emit_event: EmitEvent | None = None,
    skip_event: asyncio.Event | None = None,
    session_id: str = "",
    run_config: RunConfig = RunConfig(),
) -> ProsePart:
    """Play one full session (multiple scenes) from `state` until an ending
    fires. Returns the terminal EndGame ProsePart.

    `request_player_input` makes this driver-agnostic — supply a CLI/stdin
    variant from `__main__.py`, the SSE/HTTP variant from `main.py`, or a
    scripted/random/LLM-as-player variant from an eval harness. The shape
    is the same: it returns one `PlayerAction` per await.

    Args:
        request_player_input: called every turn that awaits the player —
            returns a `PickOption` / `TypeText` (continue the scene) or a
            `WalkTo` (leave; the WalkTo's location_id MUST be adjacent).
        on_chronicle_entry: optional callback fired after each scene-end
            chronicler call.
        on_scene_close: optional callback fired with the terminal ProsePart
            (the closing prose / EndGame) after the scene ends, before the
            chronicler runs.

    Side effects:
        - Mutates `state` (location, scene_id, dialogue_history,
          unlocked_beats, npc_locations, npc_memory, case_file_entries,
          game_over).
    """
    while state.game_over is None:
        terminal = await play_scene(
            bundle, runtime, state,
            request_player_input=request_player_input,
            emit_event=emit_event,
            skip_event=skip_event,
            session_id=session_id,
            run_config=run_config,
        )

        # The closing prose already streamed inside play_scene (EndScene),
        # or game_over already fired (EndGame). on_scene_close lets a driver
        # render the terminal ProsePart (the CLI prints it).
        if on_scene_close is not None:
            await _maybe_await(on_scene_close(terminal))

        # Chronicler runs while player_location still points at the scene
        # just closed, so present_npc_ids is correct. The completed scene
        # is still in dialogue_history; the next enter_scene clears it.
        with _root_span("chronicler_call", session_id=session_id, scene_id=state.scene_id):
            output = await emit_chronicle_entry(bundle, world, state)
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
                skip_event=skip_event,
            )

        if isinstance(terminal.next, EndGame):
            state.game_over = GameOver(ending_id=terminal.next.ending_id)
            return terminal

        # EndScene — the player already walked; terminal.next.to_location
        # names the destination (validated adjacent at the /walk route and
        # in play_scene). Move them now that the chronicler has settled the
        # closing state; the frontend animates the tracer off this event.
        assert isinstance(terminal.next, EndScene)
        to_location = terminal.next.to_location
        assert (
            to_location is not None
            and to_location in world.locations[state.player_location].adjacent_to
        ), (
            f"finalized EndScene.to_location={to_location!r} is not adjacent "
            f"to {state.player_location!r}"
        )
        if emit_event is not None:
            await emit_event(location_change_event(
                state.player_location, to_location
            ))
        state.player_location = to_location

    # game_over set externally (defensive — current engine only sets it
    # via the EndGame path above, but be permissive).
    return terminal  # type: ignore[possibly-undefined]
