"""Scene-runner pydantic-ai agents + dynamic per-call output schemas.

Phase 1 (`emit_state_changes`) emits `StateChangeBundle`; Phase 2 is split
into two sequential calls:

  - `stream_prose_text` (prose writer) — plain text output, streamed as
    deltas so the caller can pace words to the client live. Scene-bounded
    message history on `SceneRunnerRuntime.prose_writer_message_history`.
    Voice continuity accumulates within a scene, resets at scene boundary.

  - `emit_choices` (choice author) — structured `ChoiceBundle` output
    (Continue | Leave). Stateless per call; the just-written prose is the
    load-bearing context. System prompt holds neither bible nor voice
    rules.

  - `emit_free_text_gate` (Phase 2c) — `FreeTextGate` (a single bool).
    Runs concurrently with the choice author (both read only the prose).
    Decides whether the player writes their own move this turn.

  - `emit_caption` (Phase 2d) — `FreeTextCaption` (caption + suggested
    move). Stateless, prose-only. Runs only when the gate offered AND the
    engine cadence allows (see `scene_loop.run_turn`), so its cost is paid
    only on the turns that actually surface a free-text box.

The engine composes the outputs into a `ProsePart` for downstream
dispatch — see `scene_loop._assemble_prose_part`.

Per-call dynamic types narrow `beat_name` to `Literal[*reachable_beat_names]`
(Phase 1) so llama.cpp's grammar sampler rejects invalid ids at decode
time. The choice author and gate need no dynamic narrowing — their
output schemas are static.

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

import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import logfire
from pydantic import BaseModel, Field, create_model, model_validator
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

from .config import THINKING_OFF_EXTRA_BODY
from .engine import fireable_beat_ids, present_npc_ids
from .prompts import (
    build_caption_author_system_prompt,
    build_caption_author_user_message,
    build_choice_author_system_prompt,
    build_choice_author_user_message,
    build_closing_prose_user_message,
    build_chronicler_system_prompt,
    build_chronicler_user_prompt,
    build_free_text_gate_system_prompt,
    build_free_text_gate_user_message,
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
    FreeTextCaption,
    FreeTextGate,
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

    Seven agents (Phase 1 classifier, prose writer, choice author, free-text
    gate, caption author, populator, chronicler) have world-scoped system
    prompts and are constructed once at bundle-build time. Per-call dynamic
    schemas are passed at `agent.run()` time via NativeOutput (or
    `output_type=str` for the prose writer). System prompts are byte-stable
    across the session → KV cache hits across turns / scenes for each agent
    independently.
    """

    phase1: Agent[None, StateChangeBundle]
    prose_writer: Agent[None, str]
    choice_author: Agent[None, ChoiceBundle]
    free_text_gate: Agent[None, FreeTextGate]
    caption_author: Agent[None, FreeTextCaption]
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

    `rng` is the session-scoped source for the free-text cadence draw
    (`engine.free_text_cadence_ok`). It is NOT reset at scene boundary —
    cadence is a session-long property. Seedable via
    `make_scene_runner_runtime(seed=...)` so tests are deterministic.
    """

    prose_writer_message_history: list[ModelMessage] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)


def make_scene_runner_runtime(seed: int | None = None) -> SceneRunnerRuntime:
    """Construct a fresh scene-runner runtime. Call once per session.
    `seed` makes the free-text cadence RNG deterministic (default: system
    entropy)."""
    return SceneRunnerRuntime(rng=random.Random(seed))


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
    reachable_ids = tuple(sorted(fireable_beat_ids(world, state)))
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
    model: OpenAIChatModel, world: World, *, decision_max_tokens: int = 500
) -> AgentBundle:
    """Construct the agent bundle for `world`. Bakes seven system prompts
    (Phase 1 classifier, prose writer, choice author, free-text gate,
    caption author, populator, chronicler) and builds one agent per role.
    Per-call dynamic output_type is passed at agent.run() time via
    NativeOutput, except the prose writer which uses `output_type=str`.

    `decision_max_tokens` caps the output of the two empty-list "decision"
    agents (phase1, populator) — see `decision_settings` below. Defaulted so
    tests constructing the bundle with `TestModel()` need not thread config;
    production passes `Settings.decision_agent_max_tokens`.
    """
    # The two "decide whether to act" agents (phase1, populator) intermittently
    # open Gemma 4's ungrammared reasoning channel and loop to the provider
    # token ceiling without ever emitting answer JSON. Capping their output
    # converts that runaway into a fast `length` finish (→ UnexpectedModelBehavior
    # → retried by `_run_decision_agent`). MUST re-include THINKING_OFF_EXTRA_BODY:
    # a per-agent extra_body REPLACES the model-level one (pydantic-ai merges
    # ModelSettings shallowly per key), so omitting it would re-enable thinking
    # on the very agents that leak. The content agents keep the model default
    # (no cap) — prose/chronicle legitimately need long output.
    decision_settings = ModelSettings(
        max_tokens=decision_max_tokens,
        extra_body=THINKING_OFF_EXTRA_BODY,
    )
    # `name=` is the suffix in the per-agent span name
    # (`invoke_agent {name}` under instrumentation version 3 — set in
    # main.py's `instrument_pydantic_ai` call). Without it, all five
    # agents would share one indistinguishable span name.
    phase1 = Agent[None, StateChangeBundle](
        model=model,
        name="phase1",
        system_prompt=build_phase1_system_prompt(world),
        output_retries=3,
        model_settings=decision_settings,
    )
    prose_writer = Agent[None, str](
        model=model,
        name="prose_writer",
        system_prompt=build_prose_writer_system_prompt(world),
        output_retries=3,
    )
    choice_author = Agent[None, ChoiceBundle](
        model=model,
        name="choice_author",
        system_prompt=build_choice_author_system_prompt(world),
        output_retries=3,
    )
    free_text_gate = Agent[None, FreeTextGate](
        model=model,
        name="free_text_gate",
        system_prompt=build_free_text_gate_system_prompt(world),
        output_retries=3,
    )
    caption_author = Agent[None, FreeTextCaption](
        model=model,
        name="caption_author",
        system_prompt=build_caption_author_system_prompt(world),
        output_retries=3,
    )
    populator = Agent[None, NpcPlacementBundle](
        model=model,
        name="populator",
        system_prompt=build_populator_system_prompt(world),
        output_retries=3,
        model_settings=decision_settings,
    )
    chronicler = build_chronicler_agent(model, world)
    return AgentBundle(
        phase1=phase1,
        prose_writer=prose_writer,
        choice_author=choice_author,
        free_text_gate=free_text_gate,
        caption_author=caption_author,
        populator=populator,
        chronicler=chronicler,
        world=world,
    )


# ───────────────────────── per-call run wrappers ─────────────────────────


# Attempts for the two capped decision agents (phase1, populator). Their cap
# turns a runaway reasoning loop into a fast `length` finish that pydantic-ai
# raises as UnexpectedModelBehavior; since the leak is intermittent (~6-18% per
# call), re-rolling recovers the overwhelming majority before the scene_loop
# falls back to an empty bundle. `output_retries` does NOT cover this — it
# retries output *validation*, not a token-limit UnexpectedModelBehavior.
_DECISION_ATTEMPTS = 3


async def _run_decision_agent(
    agent: Agent[None, Any],
    prompt: str,
    output_type: NativeOutput,
    *,
    attempts: int = _DECISION_ATTEMPTS,
):
    """Run a capped decision agent, re-rolling on a runaway (the model opened
    its reasoning channel and hit the token cap before emitting JSON). Returns
    the AgentRunResult on success; re-raises the last UnexpectedModelBehavior
    once attempts are exhausted, so the caller's existing fallback still fires.
    """
    last_exc: UnexpectedModelBehavior | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await agent.run(prompt, output_type=output_type)
        except UnexpectedModelBehavior as e:
            last_exc = e
            logfire.warn(
                "decision agent runaway; re-rolling",
                agent=agent.name,
                attempt=attempt,
                attempts=attempts,
                error=str(e),
            )
    assert last_exc is not None  # loop ran ≥1 time, so a failure was recorded
    raise last_exc


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
    result = await _run_decision_agent(
        bundle.phase1,
        prompt,
        NativeOutput(DynamicBundle, strict=True),
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
    Phase 1 / Phase 2 prompts render `## Present`.

    Stateless across scenes (no message_history); the case file
    (`state.case_file_entries`) is the cross-scene context the populator
    reads.
    """
    prompt = build_populator_user_prompt(state, bundle.world)
    DynamicBundle = _build_npc_placement_bundle_type(bundle.world)
    result = await _run_decision_agent(
        bundle.populator,
        prompt,
        NativeOutput(DynamicBundle, strict=True),
    )
    return NpcPlacementBundle.model_validate(result.output.model_dump())


async def stream_prose_text(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    applied: StateChangeBundle,
) -> AsyncIterator[str]:
    """Prose writer LLM call, streamed. Yields text DELTAS as the model
    produces them, so the caller can pace them to the client live instead
    of waiting for the whole turn to assemble.

    Scene-bounded conversation: the model sees its own prior turns' prose
    as assistant messages via `runtime.prose_writer_message_history`. On
    clean completion (the `async for` exhausting), history is updated with
    `result.all_messages()` — atomic, only the final accepted result.

    On UnexpectedModelBehavior (retries exhausted), the `async for` raises
    BEFORE the history line, so it is skipped: the caller
    (scene_loop._stream_pace_and_author) catches and uses fallback prose,
    and `runtime.prose_writer_message_history` is left UNCHANGED. Same
    contract as the prior blocking `emit_prose_text`.

    Uses cumulative `stream_text()` (not `delta=True`): per pydantic-ai,
    `delta=True` omits the final assistant message from `all_messages()`,
    which would break voice continuity. We diff against the previous
    cumulative chunk to recover deltas ourselves.

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
    async with bundle.prose_writer.run_stream(
        user_msg,
        message_history=runtime.prose_writer_message_history,
        output_type=str,
    ) as result:
        prev = ""
        async for cumulative in result.stream_text():
            yield cumulative[len(prev):]
            prev = cumulative
        runtime.prose_writer_message_history = result.all_messages()


async def emit_closing_prose(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    world: World,
    to_location_id: str,
) -> str:
    """One closing line as the player leaves for `to_location_id`. Reuses
    the prose writer with the scene's `prose_writer_message_history` so the
    exit keeps the scene's voice; the user message names the destination
    (via `Location.display_short`) and asks for a single line — see
    `build_closing_prose_user_message`. Single-phase (no Phase 1, no
    choices) per SPEC §3.2.

    Blocking (`.run`, not streamed): it's one short line and the caller
    (scene_loop.play_scene) paces it to the client itself. On
    UnexpectedModelBehavior (retries exhausted) this raises like the prose
    writer; the caller catches and uses a fixed fallback line. History is
    updated only on success, so a failed close leaves the (about-to-be-
    reset) scene history untouched.
    """
    user_msg = build_closing_prose_user_message(state, world, to_location_id)
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

    The output schema is static (`Continue | Leave`, no per-call id
    narrowing) so the canonical `ChoiceBundle` is passed directly — its
    `Continue` 2-4-option validator fires during the agent's retry loop.
    """
    user_msg = build_choice_author_user_message(state, bundle.world, prose)
    result = await bundle.choice_author.run(
        user_msg,
        output_type=NativeOutput(ChoiceBundle, strict=True),
    )
    return ChoiceBundle.model_validate(result.output.model_dump())


async def emit_free_text_gate(
    bundle: AgentBundle,
    state: State,
    prose: str,
) -> bool:
    """Free-text gate LLM call (Phase 2c). Stateless. Pure-prose judgment:
    should the player write their own move this turn instead of picking a
    canned option? Runs concurrently with `emit_choices` (both read only
    the prose). Returns the `offer` bool.

    Static output schema (`FreeTextGate`, a single bool) → fast decode.
    """
    user_msg = build_free_text_gate_user_message(state, bundle.world, prose)
    result = await bundle.free_text_gate.run(
        user_msg,
        output_type=NativeOutput(FreeTextGate, strict=True),
    )
    return result.output.offer


async def emit_caption(
    bundle: AgentBundle,
    state: State,
    prose: str,
) -> FreeTextCaption:
    """Caption author LLM call (Phase 2d). Stateless. Pure-prose judgment
    like the gate: from the just-written prose, write the eyebrow caption
    above the free-text box and a suggested first-person move (the editable
    backup). Called only when the gate offered AND cadence allows, so its
    cost lands only on turns that actually show a box.

    Static output schema (`FreeTextCaption`, no per-call id narrowing) — so
    `result.output` is already the canonical model (no re-validation round-
    trip needed, unlike the dynamically-narrowed choice/placement schemas).
    """
    user_msg = build_caption_author_user_message(state, bundle.world, prose)
    result = await bundle.caption_author.run(
        user_msg,
        output_type=NativeOutput(FreeTextCaption, strict=True),
    )
    return result.output


# ───────────────────────── chronicler agent ─────────────────────────


def build_chronicler_agent(
    model: OpenAIChatModel, world: World
) -> Agent[None, ChronicleEntry]:
    """Stateless chronicler agent. System prompt is baked once (full bible +
    voice rules); each call is independent — the per-call user prompt opens
    with a `## Case File So Far` block (all prior entries) and then this
    scene's transcript, so no `message_history` is threaded. Output is
    structured `ChronicleEntry` = case_file_entry (UI prose) +
    npc_memory_updates (per-NPC notes surfaced to Phase 2).

    Output spec is passed at `agent.run()` time (in `emit_chronicle_entry`)
    via `NativeOutput(ChronicleEntry, strict=True)` — same pattern as
    Phase 1/2. Keeping the agent's construction free of `NativeOutput`
    lets TestModel-based unit tests bypass the native-output profile
    check by monkeypatching `emit_chronicle_entry` or supplying
    `custom_output_args`.
    """
    return Agent[None, ChronicleEntry](
        model=model,
        name="chronicler",
        system_prompt=build_chronicler_system_prompt(world),
        # Match phase1/prose_writer/choice_author/populator. Needed so the
        # canonical _no_duplicate_npc_ids validator triggers a retry rather
        # than bubbling on the first duplicate.
        output_retries=3,
    )


async def emit_chronicle_entry(
    bundle: AgentBundle,
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

    Stateless per call: the user prompt opens with a `## Case File So Far`
    block (all prior entries, append-only) and then this scene's transcript,
    so the chronicler sees its accumulating record without a conversation
    history. The append-only framing keeps the prompt prefix cache-stable
    across scenes — the previous design rewrote a past message down to an
    anchor stub on every call, which shifted every following token and
    invalidated the chronicler's whole KV prefix each time. Does NOT mutate
    `state` — caller merges fields explicitly.
    """
    assert state.dialogue_history, "emit_chronicle_entry: empty dialogue_history"
    user_prompt = build_chronicler_user_prompt(state, world)
    DynamicChronicleEntry = _build_chronicle_entry_type(
        present_npc_ids(world, state)
    )
    result = await bundle.chronicler.run(
        user_prompt,
        output_type=NativeOutput(DynamicChronicleEntry, strict=True),
    )
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
