"""Scene-runner pydantic-ai agents + dynamic per-call output schemas.

Phase 1 (`emit_state_changes`) emits `StateChangeBundle`; Phase 2 is split
into two sequential calls:

  - `emit_prose_text` (prose writer) — plain text output. Scene-bounded
    message history on `SceneRunnerRuntime.prose_writer_message_history`.
    Voice continuity accumulates within a scene, resets at scene boundary.

  - `emit_choices` (choice author) — structured `ChoiceBundle` output
    (Continue | FreeText). Stateless per call; the just-written prose is
    the load-bearing context. System prompt holds neither bible nor voice
    rules.

The engine composes the two outputs into a `ProsePart` for downstream
dispatch — see `scene_loop.run_turn`. Scene closure is engine + player
only; the choice author cannot emit a close.

Per-call dynamic types narrow `beat_name` to `Literal[*reachable_beat_names]`
(Phase 1) and `FreeText.npc_id` to `Literal[*present_npc_ids]` (choice
author) so llama.cpp's grammar sampler rejects invalid ids at decode time.

State-INDEPENDENT structural rules (≤2 unlock_beat cap, Continue option
count 2-4, no duplicate beat ids) live on the canonical schemas in
`schemas.py` as @model_validator / Field(...). State-DEPENDENT enum
constraints live in the per-call Literal narrowing below.

Why NativeOutput rather than the output-function pattern: empirical
testing on Gemma 4 E4B via llama-server's tool-calling path showed the
agent hung indefinitely; NativeOutput routes through
`response_format.json_schema` which Gemma 4 handles reliably. The prose
writer is the exception — its output is plain text (`output_type=str`),
no JSON schema involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel

from .engine import present_npc_ids, reachable_beat_ids
from .prompts import (
    build_choice_author_system_prompt,
    build_choice_author_user_message,
    build_chronicler_system_prompt,
    build_chronicler_user_prompt,
    build_phase1_system_prompt,
    build_phase1_user_prompt,
    build_populator_system_prompt,
    build_populator_user_prompt,
    build_prose_writer_departure_message,
    build_prose_writer_system_prompt,
    build_prose_writer_user_message,
)
from .schemas import (
    ChoiceBundle,
    ChronicleEntry,
    NpcMemoryUpdate,
    NpcPlacementBundle,
    StateChangeBundle,
    UnlockBeat,
)
from .state import State
from .world import World


# ───────────────────────── agent bundle ─────────────────────────


@dataclass
class AgentBundle:
    """One world's agents. Built once in lifespan via
    `build_scene_runner_bundle(model, world)`.

    Five agents (Phase 1 classifier, prose writer, choice author,
    populator, chronicler) have world-scoped system prompts and are
    constructed once at bundle-build time. Per-call dynamic schemas are
    passed at `agent.run()` time via NativeOutput (or `output_type=str`
    for the prose writer). System prompts are byte-stable across the
    session → KV cache hits across turns / scenes for each agent
    independently.
    """

    phase1: Agent[None, StateChangeBundle]
    prose_writer: Agent[None, str]
    choice_author: Agent[None, ChoiceBundle]
    populator: Agent[None, NpcPlacementBundle]
    chronicler: Agent[None, ChronicleEntry]
    world: World


@dataclass
class SceneRunnerRuntime:
    """Scene-scoped prose-writer conversation state. Reset at scene
    boundary (engine.enter_scene). One per session.

    The prose writer is conversation-based within a scene: the model
    sees its own prior turns' prose as assistant messages. Voice
    continuity (sentence rhythm, narrator persona, dialogue tag style)
    accumulates naturally over the 4-8 turns of a scene. Reset to empty
    at scene boundary so the next scene's KV cache starts fresh from
    the (cached) system prompt.

    On prose-writer retry exhaustion (UnexpectedModelBehavior → fallback
    path in scene_loop), the runtime is NOT mutated. The fallback prose
    is surfaced to the player but never persisted to message history —
    preserves voice continuity by treating the failed turn as if it
    didn't exist in the conversation, even though the player saw
    generic stand-in prose.

    The choice author is stateless per call (no history field needed).
    """

    prose_writer_message_history: list[ModelMessage] = field(default_factory=list)


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
    """ChoiceBundle whose FreeText.npc_id is
    `Literal[*present_npc_ids] | None`. No narrowing on Continue.options
    (no leaves_for anymore).

    The discriminated union reuses the canonical `Continue` so its
    structural validators (2-4 options) fire during the agent's retry
    loop. The dynamic FreeText is a fresh model — its only mutation is
    the per-call npc_id Literal.
    """
    npc_ids = present_npc_ids(world, state)
    if npc_ids:
        NpcIdType = Union[Literal[npc_ids], None]  # type: ignore[valid-type]
    else:
        # No NPCs present → npc_id must be None.
        NpcIdType = type(None)  # type: ignore[assignment]

    from .schemas import Continue

    DynamicFreeText = create_model(
        "FreeText",
        type=(Literal["free_text"], "free_text"),
        prompt=(str, ...),
        npc_id=(NpcIdType, None),
    )
    NextUnion = Union[Continue, DynamicFreeText]  # type: ignore[valid-type]
    NextField = Annotated[NextUnion, Field(discriminator="type")]
    return create_model("ChoiceBundle", next=(NextField, ...))


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
    output_type is passed at agent.run() time via NativeOutput, except
    the prose writer which uses `output_type=str`.
    """
    phase1 = Agent[None, StateChangeBundle](
        model=model,
        system_prompt=build_phase1_system_prompt(world),
        output_retries=3,
    )
    prose_writer = Agent[None, str](
        model=model,
        system_prompt=build_prose_writer_system_prompt(world),
        output_retries=3,
    )
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
    movement. Caller (scene_loop.play_scene) applies non-empty
    placements to `state.npc_locations` before the populator-aware
    Phase 1 / Phase 2 prompts render `[PRESENT]`.

    Stateless across scenes (no message_history); the case file
    (`state.case_file_entries`) is the cross-scene context the populator
    reads.
    """
    prompt = build_populator_user_prompt(state, bundle.world)
    DynamicBundle = _build_npc_placement_bundle_type(bundle.world)
    result = await bundle.populator.run(
        prompt,
        output_type=NativeOutput(DynamicBundle, strict=True),
    )
    return NpcPlacementBundle.model_validate(result.output.model_dump())


async def emit_prose_text(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    applied: StateChangeBundle,
) -> str:
    """Prose writer LLM call. Scene-bounded conversation: the model sees
    its own prior turns' prose as assistant messages via
    `runtime.prose_writer_message_history`. On success, history is updated
    with `result.all_messages()` (atomic: pydantic-ai's internal
    ModelRetry loop doesn't pollute — only the final accepted result is
    appended).

    On UnexpectedModelBehavior (retries exhausted), this function raises;
    caller (scene_loop.run_turn) catches and uses fallback prose, which
    does NOT mutate runtime.prose_writer_message_history.

    `is_first_turn` is derived from runtime.prose_writer_message_history
    being empty. On the first turn of the first scene, the opening seed
    from `world.brief.opening_seed` is rendered into the user message.
    """
    is_first_turn = not runtime.prose_writer_message_history
    is_session_opening = is_first_turn and state.scene_id == 1
    opening_seed = (
        bundle.world.brief.opening_seed if is_session_opening else None
    )
    user_msg = build_prose_writer_user_message(
        state, bundle.world, applied,
        opening_seed=opening_seed,
        is_first_turn=is_first_turn,
    )
    result = await bundle.prose_writer.run(
        user_msg,
        message_history=runtime.prose_writer_message_history,
        output_type=str,
    )
    runtime.prose_writer_message_history = result.all_messages()
    return result.output


async def emit_departure_prose(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
) -> str:
    """Closing-prose LLM call. Same agent as `emit_prose_text` (system
    prompt is byte-stable → KV cache stays warm); user message instructs
    "write the moment of leaving" — short, threshold-only, no destination.

    Used by `scene_loop._close_via_departure` from two trigger paths: the
    player hitting /leave (raised LeaveRequested) and the engine's
    `MAX_TURNS_PER_SCENE` guardrail. Conversation history is updated on
    success like a normal turn.
    """
    user_msg = build_prose_writer_departure_message(state, bundle.world)
    result = await bundle.prose_writer.run(
        user_msg,
        message_history=runtime.prose_writer_message_history,
        output_type=str,
    )
    runtime.prose_writer_message_history = result.all_messages()
    return result.output


async def emit_choices(
    bundle: AgentBundle,
    state: State,
    prose: str,
) -> ChoiceBundle:
    """Choice author LLM call. Stateless (no message_history). The
    just-written prose is the load-bearing context; the system prompt
    has no bible/voice rules.

    `output_type=NativeOutput(DynamicChoiceBundle, strict=True)` baked
    per call. Dynamic FreeText.npc_id is grammar-narrowed to present
    NPCs.
    """
    user_msg = build_choice_author_user_message(state, bundle.world, prose)
    DynamicChoiceBundle = _build_choice_bundle_type(bundle.world, state)
    result = await bundle.choice_author.run(
        user_msg,
        output_type=NativeOutput(DynamicChoiceBundle, strict=True),
    )
    return ChoiceBundle.model_validate(result.output.model_dump())


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
    (per-NPC notes surfaced to Phase 2).

    Output spec is passed at `agent.run()` time (in `emit_chronicle_entry`)
    via `NativeOutput(ChronicleEntry, strict=True)` — same pattern as
    Phase 1/2. Keeping the agent's construction free of `NativeOutput`
    lets TestModel-based unit tests bypass the native-output profile
    check by monkeypatching `emit_chronicle_entry` or supplying
    `custom_output_args`.

    The OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)
    flag is set on the model in lifespan / _build_model — load-bearing
    for this agent. Without it, pydantic-ai's default 'auto' round-trips
    `ThinkingPart`s into prior assistant content as <think>...</think>
    blocks, which Gemma 4 cannot tolerate in multi-turn.
    """
    return Agent[None, ChronicleEntry](
        model=model,
        system_prompt=build_chronicler_system_prompt(world),
        # Match phase1/prose_writer/choice_author/populator. Needed so the
        # canonical _no_duplicate_npc_ids validator triggers a retry rather
        # than bubbling on the first duplicate.
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
    user_prompt = build_chronicler_user_prompt(state, world)
    DynamicChronicleEntry = _build_chronicle_entry_type(
        present_npc_ids(world, state)
    )
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
