"""Scene-runner pydantic-ai agents + dynamic per-call output schemas.

Per turn, three LLM calls plus side agents:

  Phase 1 (`emit_state_changes`) — classifier emitting `StateChangeBundle`.
      Stateless per turn (no message_history). Bible-loaded system prompt.

  Prose writer (`emit_prose`) — narrator with a tool loop. Output type is
      `str` (PoL 2026-05-17 confirmed `NativeOutput` + tools is incompatible
      on Gemma 4 + llama.cpp). Scene-bounded message history in
      `SceneRunnerRuntime.phase2_message_history` accumulates prior prose
      as assistant content; voice continuity rides on it.

      Four tools registered on the prose writer:
        - `pause_for_continue()` — suspend mid-prose, resume on player tap.
        - `request_free_text(prompt, npc_id?)` — suspend, return player text.
        - `end_scene(location_id?)` — set termination flag on deps.
        - `end_game(ending_id)` — set termination flag on deps.

      Termination is a side effect on `ProseWriterDeps.termination`; the
      engine inspects it after the agent.run completes and assembles the
      composite ProsePart.next accordingly.

  Choice author (`emit_choices`) — runs only when prose writer terminated
      without an end_scene/end_game tool call AND leave_pick was not set.
      Emits `ChoiceBundle.options` (a list of 2-4 options, ≥1 leave).
      Stateless per cycle (no message_history). Slim system prompt — no
      bible, no voice samples, no endings.

Per-call dynamic types narrow `Option.leaves_for` to
`Literal[*adjacent_ids] | None` (choice author) — same pattern as Phase 1.

Tool args (npc_id, location_id, ending_id) are validated post-hoc inside
the tool function bodies in v1 (str type at the agent level), not via
per-call grammar narrowing. Promoting to prepare-callback narrowing is a
follow-up if the looser tool-arg surface causes drift.

State-INDEPENDENT structural rules live on the canonical schemas in
`schemas.py` (`ChoiceBundle._has_at_least_one_leave_option`,
`StateChangeBundle.max_length=2`). State-DEPENDENT enum constraints live
in the per-call Literal narrowing below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Awaitable, Callable, Literal, Union

import logfire
from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel

from .engine import reachable_beat_ids
from .prompts import (
    build_choice_author_system_prompt,
    build_choice_author_user_message,
    build_chronicler_system_prompt,
    build_chronicler_user_prompt,
    build_phase1_system_prompt,
    build_phase1_user_prompt,
    build_populator_system_prompt,
    build_populator_user_prompt,
    build_prose_writer_system_prompt,
    build_prose_writer_user_message,
)
from .schemas import (
    ChoiceBundle,
    ChronicleEntry,
    ContinueScene,
    EndGame,
    EndScene,
    NpcMemoryUpdate,
    NpcPlacementBundle,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from .state import State
from .world import World


# ───────────────────────── prose writer deps ─────────────────────────


AwaitContinue = Callable[[], Awaitable[None]]
AwaitFreeText = Callable[[str, str | None], Awaitable[str]]


@dataclass
class ProseWriterDeps:
    """Injected per-call via `agent.run(deps=...)`. The async callbacks
    (`await_continue`, `await_free_text`) are how the pause_for_continue
    and request_free_text tools suspend until the engine resolves them
    (typically via a future bound to an HTTP POST in the SSE backend, or
    an asyncio.Event in tests / CLI).

    `termination` is set in place by the `end_scene` / `end_game` tools
    via `terminate_scene` / `terminate_game`. After the agent.run
    completes, the engine reads this field to decide which composite
    `ProsePart.next` variant to construct. Priority is in `emit_prose`.

    `present_npc_ids`, `adjacent_ids`, `satisfied_ending_ids` are passed
    per-call so the tool bodies can validate args against the current
    state (v1 uses str-typed tool args + post-hoc validation rather than
    per-call grammar narrowing).
    """

    await_continue: AwaitContinue
    await_free_text: AwaitFreeText
    present_npc_ids: tuple[str, ...] = ()
    adjacent_ids: tuple[str, ...] = ()
    satisfied_ending_ids: tuple[str, ...] = ()
    termination: tuple[Literal["scene", "game"], str | None] | None = None

    def terminate_scene(self, location_id: str | None) -> None:
        self.termination = ("scene", location_id)

    def terminate_game(self, ending_id: str) -> None:
        self.termination = ("game", ending_id)


# ───────────────────────── agent bundle ─────────────────────────


@dataclass
class AgentBundle:
    """One world's agents. Built once in lifespan via
    `build_scene_runner_bundle(model, world)`.

    All five agents (Phase 1 classifier, prose writer, choice author,
    populator, chronicler) have world-scoped system prompts and are
    constructed once at bundle-build time. Per-call dynamic schemas are
    passed at `agent.run()` time via NativeOutput (for Phase 1, choice
    author, populator, chronicler — output_type=str for prose writer).
    System prompts are byte-stable across the session → KV cache hits
    across turns / scenes for each agent independently.
    """

    phase1: Agent[None, StateChangeBundle]
    prose_writer: Agent[ProseWriterDeps, str]
    choice_author: Agent[None, ChoiceBundle]
    populator: Agent[None, NpcPlacementBundle]
    chronicler: Agent[None, ChronicleEntry]
    world: World


@dataclass
class SceneRunnerRuntime:
    """Scene-scoped prose-writer conversation state. Reset at scene
    boundary (engine.enter_scene). One per session.

    Prose writer is conversation-based within a scene: the model sees
    its own prior turns' prose as assistant messages. Voice continuity
    (sentence rhythm, narrator persona, dialogue tag style) accumulates
    naturally over the 4-8 turns of a scene. Reset to empty at scene
    boundary so the next scene's KV cache starts fresh from the (cached)
    system prompt.

    On prose-writer retry exhaustion (UnexpectedModelBehavior → fallback
    prose path in scene_loop), the runtime is NOT mutated. The fallback
    prose is surfaced to the player but never persisted to message
    history — preserves voice continuity by treating the failed turn as
    if it didn't exist in the conversation, even though the player saw
    generic stand-in prose.
    """

    phase2_message_history: list[ModelMessage] = field(default_factory=list)


def make_scene_runner_runtime() -> SceneRunnerRuntime:
    """Construct a fresh scene-runner runtime. Call once per session
    (mirrors make_chronicler_runtime)."""
    return SceneRunnerRuntime()


# ───────────────────────── dynamic type builders ─────────────────────────


def _build_state_change_bundle_type(world: World, state: State) -> type[BaseModel]:
    """StateChangeBundle whose UnlockBeat.beat_name is
    `Literal[*reachable_beat_names(world, state)]`. The grammar sampler
    rejects any beat_name not currently reachable at decode time — covers
    requires_beats/blocked_by_beats/already-fired in one Literal.

    Names (not ids) on the wire: small LLMs reason better with short
    human-language phrases ("Tom speaks of Megan") than snake_case slugs.
    `emit_state_changes` translates name → canonical `UnlockBeat.beat_id`
    after decode so engine/SSE/frontend ride on stable id keys.

    The dynamic bundle does NOT inherit from canonical `StateChangeBundle`:
    canonical's `_no_duplicate_beat_ids` reads `change.beat_id`, which
    dynamic items lack. The inline `_no_duplicate_beat_names` validator
    below catches duplicates during the agent's retry loop; canonical
    validators still fire at the round-trip in `emit_state_changes`
    (defense in depth).
    """
    reachable_ids = tuple(sorted(reachable_beat_ids(world, state)))
    reachable_names = tuple(world.beats[bid].name for bid in reachable_ids)
    if not reachable_names:
        # Nothing fireable from this state → bundle must be empty.
        return create_model(
            "StateChangeBundle",
            state_changes=(
                list[Any],
                Field(default_factory=list, max_length=0),
            ),
        )
    NameLiteral = Literal[reachable_names]  # type: ignore[valid-type]
    DynamicUnlockBeat = create_model(
        "UnlockBeat",
        type=(Literal["unlock_beat"], "unlock_beat"),
        beat_name=(NameLiteral, ...),
    )
    StateChange = Annotated[DynamicUnlockBeat, Field(discriminator="type")]

    def _no_duplicate_beat_names(self):  # type: ignore[no-untyped-def]
        seen: set[str] = set()
        for change in self.state_changes:
            name = getattr(change, "beat_name", None)
            if name is None:
                continue
            if name in seen:
                raise ValueError(
                    f"beat_name {name!r} appears twice in bundle"
                )
            seen.add(name)
        return self

    return create_model(
        "StateChangeBundle",
        state_changes=(
            list[StateChange],
            Field(default_factory=list, max_length=2),
        ),
        __validators__={
            "_no_duplicate_beat_names": model_validator(mode="after")(
                _no_duplicate_beat_names
            ),
        },
    )


def _build_choice_bundle_type(world: World, state: State) -> type[BaseModel]:
    """ChoiceBundle whose Option.leaves_for is
    `Literal[*adjacent_ids] | None`. The grammar sampler physically
    cannot emit a non-adjacent walk target.

    State-independent rules (≥1 leave option) live on the canonical
    ChoiceBundle as @model_validator and fire on round-trip in
    `emit_choices`.

    Stranded-location edge: when adjacent_ids is empty, leaves_for
    Literal collapses to None — choice author cannot emit a leave option
    at all. The canonical `_has_at_least_one_leave_option` validator
    will then ALWAYS fail; the engine fallback path handles this case.
    """
    location = world.locations[state.player_location]
    adjacent_ids = tuple(location.adjacent_to)
    if adjacent_ids:
        LeavesForType = Union[Literal[adjacent_ids], None]  # type: ignore[valid-type]
    else:
        LeavesForType = None
    DynamicOption = create_model(
        "Option",
        text=(str, ...),
        leaves_for=(LeavesForType, None),
    )
    return create_model(
        "ChoiceBundle",
        __base__=ChoiceBundle,
        options=(list[DynamicOption], ...),
    )


# ───────────────────────── populator dynamic schema ─────────────────────────


def _build_npc_placement_bundle_type(world: World) -> type[BaseModel]:
    """NpcPlacementBundle whose NpcPlacement.npc_id is `Literal[*npc_ids]`
    and .location_id is `Literal[*location_ids]`. Grammar sampler rejects
    invalid ids at decode time.

    No reachability-style state narrowing here: any NPC can be moved to
    any location in principle (the populator's prompt teaches it
    restraint).
    """
    npc_ids = tuple(world.npcs.keys())
    location_ids = tuple(world.locations.keys())
    if not npc_ids or not location_ids:
        # Edge: empty cast or world — placements must be empty.
        return create_model(
            "NpcPlacementBundle",
            __base__=NpcPlacementBundle,
            placements=(list[Any], Field(default_factory=list, max_length=0)),
        )
    DynamicNpcPlacement = create_model(
        "NpcPlacement",
        npc_id=(Literal[npc_ids], ...),  # type: ignore[valid-type]
        location_id=(Literal[location_ids], ...),  # type: ignore[valid-type]
    )
    return create_model(
        "NpcPlacementBundle",
        __base__=NpcPlacementBundle,
        placements=(list[DynamicNpcPlacement], Field(default_factory=list)),
    )


# ───────────────────────── chronicler dynamic schema ─────────────────────────


def _build_chronicle_entry_type(
    present_npc_ids: tuple[str, ...],
) -> type[BaseModel]:
    """ChronicleEntry whose `npc_memory_updates` is a length-N list of
    NpcMemoryUpdate, where `npc_id` is `Literal[*present_npc_ids]` and
    the list is constrained to exactly N entries
    (`min_length=max_length=N`).

    The grammar sampler enforces both the per-item id constraint and the
    list length, so the model literally cannot emit an empty list when
    NPCs were present nor an out-of-cast id. `__base__=ChronicleEntry`
    keeps the canonical `_no_duplicate_npc_ids` validator firing during
    the agent's retry loop (ValidationError → ModelRetry, with
    `output_retries=3` on the agent).

    Empty present_npc_ids (rare: scene with no NPCs at all) → constrain
    list to max_length=0, mirroring the empty-reachable path in
    `_build_state_change_bundle_type`.
    """
    if not present_npc_ids:
        return create_model(
            "ChronicleEntry",
            __base__=ChronicleEntry,
            npc_memory_updates=(
                list[Any],
                Field(default_factory=list, max_length=0),
            ),
        )
    NpcIdLiteral = Literal[present_npc_ids]  # type: ignore[valid-type]
    DynamicNpcMemoryUpdate = create_model(
        "NpcMemoryUpdate",
        npc_id=(NpcIdLiteral, ...),
        note=(str, ...),
    )
    n = len(present_npc_ids)
    return create_model(
        "ChronicleEntry",
        __base__=ChronicleEntry,
        npc_memory_updates=(
            list[DynamicNpcMemoryUpdate],
            Field(min_length=n, max_length=n),
        ),
    )


# ───────────────────────── factory ─────────────────────────


def build_scene_runner_bundle(
    model: OpenAIChatModel, world: World
) -> AgentBundle:
    """Construct the agent bundle for `world`. Bakes five system prompts
    (Phase 1 classifier, prose writer, choice author, populator,
    chronicler) and builds one agent per role. Per-call dynamic
    output_type / tool-arg validation happens at agent.run() time.
    """
    phase1 = Agent[None, StateChangeBundle](
        model=model,
        system_prompt=build_phase1_system_prompt(world),
        output_retries=3,
    )
    prose_writer: Agent[ProseWriterDeps, str] = Agent(
        model=model,
        deps_type=ProseWriterDeps,
        system_prompt=build_prose_writer_system_prompt(world),
        output_retries=3,
    )
    _register_prose_writer_tools(prose_writer)
    choice_author = Agent[None, ChoiceBundle](
        model=model,
        system_prompt=build_choice_author_system_prompt(world),
        output_retries=3,
    )
    populator = Agent[None, NpcPlacementBundle](
        model=model,
        system_prompt=build_populator_system_prompt(world),
        output_retries=3,
    )
    chronicler = build_chronicler_agent(model, world)
    return AgentBundle(
        phase1=phase1,
        prose_writer=prose_writer,
        choice_author=choice_author,
        populator=populator,
        chronicler=chronicler,
        world=world,
    )


# ───────────────────────── prose writer tools ─────────────────────────


def _register_prose_writer_tools(agent: Agent[ProseWriterDeps, str]) -> None:
    """Register the four prose-writer tools on `agent`. Called once at
    bundle build. Tool args are str-typed; per-cycle valid sets
    (present_npc_ids, adjacent_ids, satisfied_ending_ids) live on
    `ProseWriterDeps` and the tool bodies validate against them — invalid
    args raise `ModelRetry` so pydantic-ai's retry loop kicks in.

    Promoting to per-call grammar narrowing via the `prepare` callback
    is a follow-up if the looser surface causes drift in practice.
    """

    @agent.tool
    async def pause_for_continue(ctx: RunContext[ProseWriterDeps]) -> str:
        """Pause the scene mid-prose. The player sees what you've written,
        taps Continue, and you resume. Returns an empty string when the
        player has tapped continue (pydantic-ai tools must return JSON-
        serialisable; '' is the no-payload signal)."""
        await ctx.deps.await_continue()
        return ""

    @agent.tool
    async def request_free_text(
        ctx: RunContext[ProseWriterDeps],
        prompt: str,
        npc_id: str | None = None,
    ) -> str:
        """Ask the player to type a response. Returns the player's typed
        text. `npc_id` (optional) scopes the input to an NPC currently
        present in the scene; raises ModelRetry if the id isn't
        present."""
        if npc_id is not None and npc_id not in ctx.deps.present_npc_ids:
            raise ModelRetry(
                f"request_free_text: npc_id {npc_id!r} is not present in "
                f"this scene. Present NPCs: "
                f"{list(ctx.deps.present_npc_ids)!r}. Omit npc_id or use "
                f"one of these ids."
            )
        return await ctx.deps.await_free_text(prompt, npc_id)

    @agent.tool
    async def end_scene(
        ctx: RunContext[ProseWriterDeps],
        location_id: str | None = None,
    ) -> str:
        """Close the scene. Optional `location_id` is the adjacent
        location the player walks to; raises ModelRetry if non-adjacent.
        Omit location_id to close in place — the player picks next
        location via the map."""
        if location_id is not None and location_id not in ctx.deps.adjacent_ids:
            raise ModelRetry(
                f"end_scene: location_id {location_id!r} is not adjacent. "
                f"Adjacent locations: {list(ctx.deps.adjacent_ids)!r}. "
                f"Omit location_id to close in place, or use one of these."
            )
        ctx.deps.terminate_scene(location_id)
        return ""

    @agent.tool
    async def end_game(
        ctx: RunContext[ProseWriterDeps],
        ending_id: str,
    ) -> str:
        """End the game with one of the currently satisfied endings.
        Raises ModelRetry if the ending's requires_beats aren't met."""
        if ending_id not in ctx.deps.satisfied_ending_ids:
            raise ModelRetry(
                f"end_game: ending_id {ending_id!r} is not currently "
                f"satisfied. Satisfied endings right now: "
                f"{list(ctx.deps.satisfied_ending_ids)!r}. The player must "
                f"trigger a decision beat for an ending before you can "
                f"emit it."
            )
        ctx.deps.terminate_game(ending_id)
        return ""


# ───────────────────────── per-call run wrappers ─────────────────────────


async def emit_state_changes(
    bundle: AgentBundle,
    state: State,
) -> StateChangeBundle:
    """Phase 1 LLM call. Stateless per turn (no message_history). Builds a
    full rich user prompt with reachable beats, endings, scene-so-far, case
    file — everything the classifier needs to judge what just fired.

    `output_type=NativeOutput(DynamicBundle, strict=True)` baked per call.
    Dynamic items carry `beat_name` (grammar-narrowed to reachable beat
    names); this function translates each name back to the canonical
    `beat_id` via `World.beat_id_for_name` and constructs the canonical
    `StateChangeBundle`, which fires `_no_duplicate_beat_ids` + the
    `max_length=2` cap at the boundary.
    """
    prompt = build_phase1_user_prompt(state, bundle.world)
    DynamicBundle = _build_state_change_bundle_type(bundle.world, state)
    result = await bundle.phase1.run(
        prompt,
        output_type=NativeOutput(DynamicBundle, strict=True),
    )
    raw = result.output
    canonical_changes = [
        UnlockBeat(beat_id=bundle.world.beat_id_for_name(c.beat_name))
        for c in getattr(raw, "state_changes", [])
        if getattr(c, "beat_name", None) is not None
    ]
    return StateChangeBundle(state_changes=canonical_changes)


async def emit_placements(
    bundle: AgentBundle,
    state: State,
) -> NpcPlacementBundle:
    """Populator LLM call. Runs ONCE at scene entry (before Phase 1 of
    turn 1). Decides which NPCs are at which locations now in light of
    the case file, beats so far, and dramatic logic.

    Empty bundle is the common case — most scene entries don't need
    movement.
    """
    prompt = build_populator_user_prompt(state, bundle.world)
    DynamicBundle = _build_npc_placement_bundle_type(bundle.world)
    result = await bundle.populator.run(
        prompt,
        output_type=NativeOutput(DynamicBundle, strict=True),
    )
    return NpcPlacementBundle.model_validate(result.output.model_dump())


def _satisfied_ending_ids(world: World, state: State) -> tuple[str, ...]:
    return tuple(sorted(
        eid for eid, e in world.endings.items()
        if set(e.requires_beats).issubset(state.unlocked_beats)
    ))


async def emit_prose(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    applied: StateChangeBundle,
    *,
    leave_pick: Option | None = None,
    emit_event: Callable[[dict], Awaitable[None]] | None = None,
    await_continue: AwaitContinue | None = None,
    await_free_text: AwaitFreeText | None = None,
) -> ProsePart:
    """Per-turn prose generation with tool loop + choice-author follow-up.

    Pipeline:
      1. Run prose writer (output_type=str) with the 4 tools. Tools fire
         inline; pause / free_text suspend until the engine resolves them.
         end_scene / end_game set deps.termination as a side effect.
      2. Stream prose words via `paced_word_emitter` as the agent
         produces them (run_stream + TextPartDeltaEvent → words →
         emit_event). One continuous stream across all tool boundaries.
      3. Assemble composite `ProsePart.next` in priority order:
           a. deps.termination == ("game", eid) → EndGame
           b. deps.termination == ("scene", loc?) → EndScene
           c. leave_pick is not None → EndScene
           d. otherwise → run choice author → ContinueScene

    Streaming via emit_event was previously done by scene_loop; it moves
    here because the agent.run emits prose across tool-call boundaries
    and needs to pace each chunk as it arrives.

    `SceneRunnerRuntime.phase2_message_history` is updated on success
    (atomic — pydantic-ai's internal ModelRetry loop doesn't pollute,
    only the final accepted result is appended). On
    UnexpectedModelBehavior (retries exhausted), this function raises;
    caller (scene_loop.run_turn) catches and emits fallback prose.

    `is_first_turn` is derived from runtime.phase2_message_history being
    empty. The opening seed from world.brief.opening_seed renders into
    the user message on the first turn of scene 1.

    `await_continue` / `await_free_text` MUST be supplied if the prose
    writer might call those tools. Tests can pass no-op callbacks. The
    SSE backend wires them through `SessionRuntime` futures.
    """
    is_first_turn = not runtime.phase2_message_history
    is_session_opening = is_first_turn and state.scene_id == 1
    opening_seed = (
        bundle.world.brief.opening_seed if is_session_opening else None
    )
    user_msg = build_prose_writer_user_message(
        state, bundle.world, applied,
        leave_pick=leave_pick,
        opening_seed=opening_seed,
        is_first_turn=is_first_turn,
    )

    location = bundle.world.locations[state.player_location]
    deps = ProseWriterDeps(
        await_continue=await_continue or _noop_await_continue,
        await_free_text=await_free_text or _noop_await_free_text,
        present_npc_ids=tuple(
            n.id for n in bundle.world.npcs_at(state.player_location, state=state)
        ),
        adjacent_ids=tuple(location.adjacent_to),
        satisfied_ending_ids=_satisfied_ending_ids(bundle.world, state),
    )

    full_prose = await _run_prose_writer_streamed(
        bundle.prose_writer,
        user_msg,
        runtime=runtime,
        deps=deps,
        emit_event=emit_event,
    )

    # Assemble next in priority order.
    if deps.termination is not None:
        kind, payload = deps.termination
        if kind == "game":
            assert payload is not None  # tool body validates non-empty
            return ProsePart(prose=full_prose, next=EndGame(ending_id=payload))
        # kind == "scene"; payload may be None (close in place) or an
        # adjacent location id (engine reads via state.player_location vs
        # the loop's next-walk logic — handled in scene_loop).
        return ProsePart(prose=full_prose, next=EndScene())
    if leave_pick is not None:
        return ProsePart(prose=full_prose, next=EndScene())

    # Choice author runs only when prose writer terminated cleanly and
    # the scene continues.
    choices = await emit_choices(bundle, state, full_prose)
    return ProsePart(
        prose=full_prose,
        next=ContinueScene(options=list(choices.options)),
    )


async def _run_prose_writer_streamed(
    agent: Agent[ProseWriterDeps, str],
    user_msg: str,
    *,
    runtime: SceneRunnerRuntime,
    deps: ProseWriterDeps,
    emit_event: Callable[[dict], Awaitable[None]] | None,
) -> str:
    """Run the prose writer with streaming, pace prose words via
    emit_event as they arrive, return the accumulated prose. Updates
    runtime.phase2_message_history on success."""
    from .sse import paced_word_emitter, _words_from_stream

    async def text_chunks():
        async with agent.run_stream(
            user_msg,
            message_history=runtime.phase2_message_history,
            deps=deps,
        ) as result:
            async for delta_text in result.stream_text(delta=True):
                yield delta_text
            # Capture the final messages for history persistence after
            # the stream completes.
            runtime.phase2_message_history = result.all_messages()

    if emit_event is not None:
        return await paced_word_emitter(
            _words_from_stream(text_chunks()),
            emit_event,
            event_kind="prose_word",
        )
    # No event emitter — just drain the stream for the prose text.
    parts: list[str] = []
    async for chunk in text_chunks():
        parts.append(chunk)
    return "".join(parts)


async def emit_choices(
    bundle: AgentBundle,
    state: State,
    prose: str,
) -> ChoiceBundle:
    """Choice author LLM call. Stateless per cycle (no message_history).
    Sees the just-written prose + current state; emits a list of 2-4
    options with adjacency-narrowed leaves_for.

    Stranded-location edge: when player has no adjacent locations,
    `_build_choice_bundle_type` collapses leaves_for to None, so the
    canonical `_has_at_least_one_leave_option` validator always fails
    here. Caller (emit_prose) doesn't currently special-case this; the
    fallback path in scene_loop.run_turn fires generic options.
    """
    user_msg = build_choice_author_user_message(state, bundle.world, prose)
    DynamicChoiceBundle = _build_choice_bundle_type(bundle.world, state)
    result = await bundle.choice_author.run(
        user_msg,
        output_type=NativeOutput(DynamicChoiceBundle, strict=True),
    )
    return ChoiceBundle.model_validate(result.output.model_dump())


async def _noop_await_continue() -> None:
    """Default await_continue when no callback is supplied (tests / CLI
    that don't expect pause tool calls). If the prose writer DOES call
    pause_for_continue under this, the tool returns immediately — the
    scene proceeds without an actual UI pause. Tests that need to
    verify pause semantics supply a real callback."""
    return None


async def _noop_await_free_text(prompt: str, npc_id: str | None) -> str:
    """Default await_free_text when no callback is supplied. Returns a
    sentinel so the model sees a non-empty tool result and doesn't loop.
    Tests that exercise free_text supply a real callback."""
    logfire.warn(
        "request_free_text called without an await_free_text callback; "
        "returning sentinel",
        prompt=prompt, npc_id=npc_id,
    )
    return "(no response)"


# ───────────────────────── chronicler agent ─────────────────────────


@dataclass
class ChroniclerRuntime:
    """Session-scoped state for the conversation-based chronicler.
    Symmetric with `SceneRunnerRuntime` — both are thin holders for the
    accumulating `message_history`. The chronicler agent lives on the
    world-scoped `AgentBundle` (its system prompt is baked from the
    world bible).

    `message_history` accumulates one user + one assistant message per
    scene-end; the conversation IS the case file so far. Replaced on
    each successful call with `result.all_messages()`.
    """

    message_history: list[ModelMessage] = field(default_factory=list)


def build_chronicler_agent(
    model: OpenAIChatModel, world: World
) -> Agent[None, ChronicleEntry]:
    """Conversation-based chronicler agent. System prompt is baked once
    (full bible + voice rules); each call appends one user message (scene
    summary) and gets one assistant message back. Output is structured
    `ChronicleEntry` = case_file_entry (UI prose) + npc_memory_updates
    (per-NPC notes surfaced to the prose writer).

    The OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)
    flag is set on the model in lifespan / _build_model — load-bearing
    for this agent. Without it, pydantic-ai's default 'auto' round-trips
    `ThinkingPart`s into prior assistant content as <think>...</think>
    blocks, which Gemma 4 cannot tolerate in multi-turn.
    """
    return Agent[None, ChronicleEntry](
        model=model,
        system_prompt=build_chronicler_system_prompt(world),
        # Match the other agents. Needed so the canonical
        # _no_duplicate_npc_ids validator triggers a retry rather than
        # bubbling on the first duplicate.
        output_retries=3,
    )


def make_chronicler_runtime() -> ChroniclerRuntime:
    """Construct a fresh ChroniclerRuntime. Call once per session.
    Mirrors `make_scene_runner_runtime` — the agent itself lives on
    `AgentBundle.chronicler`."""
    return ChroniclerRuntime()


async def emit_chronicle_entry(
    bundle: AgentBundle,
    runtime: ChroniclerRuntime,
    world: World,
    state: State,
) -> ChronicleEntry:
    """Generate one case-file entry (+ structured per-NPC memory updates)
    for the just-completed scene. Returns the canonical `ChronicleEntry`
    directly; caller (scene_loop.play_session) decomposes it into
    `state.case_file_entries.append(...)` and per-NPC memory merges.

    Scene anchor (scene_id, location, npcs present) is read from `state`
    — at the point this is called (after play_scene returns
    EndScene/EndGame), the just-completed scene's turns are still in
    `state.dialogue_history` and `state.player_location` still points
    at the scene's location.

    Side effects:
      - Mutates `runtime.message_history` with the just-rendered
        user+assistant pair. Subsequent calls see this scene's entry in
        their conversation context.
      - Does NOT mutate `state` — caller merges fields explicitly.
    """
    assert state.dialogue_history, "emit_chronicle_entry: empty dialogue_history"
    # The populator only runs at scene entry, so the live state matches
    # who was there for the whole scene.
    present_npc_ids = tuple(
        n.id for n in world.npcs_at(state.player_location, state=state)
    )
    user_prompt = build_chronicler_user_prompt(state, world)
    DynamicChronicleEntry = _build_chronicle_entry_type(present_npc_ids)
    result = await bundle.chronicler.run(
        user_prompt,
        message_history=runtime.message_history,
        output_type=NativeOutput(DynamicChronicleEntry, strict=True),
    )
    runtime.message_history = result.all_messages()
    # Single canonical construction: fires _no_duplicate_npc_ids and applies
    # the engine-side whitespace filter (the one rule grammar can't enforce)
    # in one pass.
    raw = result.output
    return ChronicleEntry(
        case_file_entry=raw.case_file_entry.strip(),
        npc_memory_updates=[
            NpcMemoryUpdate(npc_id=u.npc_id, note=u.note.strip())
            for u in raw.npc_memory_updates
            if u.note.strip()
        ],
    )
