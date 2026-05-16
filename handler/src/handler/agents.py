"""Scene-runner pydantic-ai agents + dynamic per-call output schemas.

Phase 1 (`emit_state_changes`) emits `StateChangeBundle`; Phase 2
(`emit_prose`) emits `ProsePart`. Each has its own system prompt:
classifier (full bible) for Phase 1, narrator (slim, voice-focused) for
Phase 2. Phase 1 is stateless per turn; Phase 2 carries a scene-bounded
message history in `SceneRunnerRuntime.phase2_message_history` so the
narrator sees its own prior turns as assistant messages — voice continuity
accumulates within a scene, resets at scene boundary.

Per-call dynamic types narrow `beat_id` to `Literal[*reachable_beat_ids]`,
`Option.leaves_for` to `Literal[*adjacent_ids]`, and `EndGame.ending_id` to
`Literal[*satisfied_ending_ids]` so llama.cpp's grammar sampler rejects
invalid ids at decode time. When `leave_pick` is set, `ProsePart.next` is
narrowed to `EndScene` (no union) — the leave-pick-forces-end-scene rule
is grammar-enforced too.

State-INDEPENDENT structural rules live on the canonical schemas in
`schemas.py` as @model_validator / Field(...). State-DEPENDENT enum
constraints live in the per-call Literal narrowing below. The dynamic
StateChangeBundle and ContinueScene subclass their canonical counterparts
(`__base__=`) so the @model_validator inheritance fires when pydantic-ai
parses the LLM output — on ValidationError pydantic-ai's retry loop kicks
in automatically. The remaining dynamic classes (UnlockBeat, Option,
EndScene, EndGame, ProsePart) have no canonical-side validators, so they
don't need `__base__=`.

Why NativeOutput rather than the output-function pattern: empirical
testing on Gemma 4 E4B via llama-server's tool-calling path showed the
agent hung indefinitely; NativeOutput routes through
`response_format.json_schema` which Gemma 4 handles reliably. See the
plan's Phase 0 fallback note.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel

from .engine import reachable_beat_ids
from .prompts import (
    build_log_writer_system_prompt,
    build_log_writer_user_prompt,
    build_phase1_system_prompt,
    build_phase1_user_prompt,
    build_phase2_system_prompt,
    build_phase2_user_message,
)
from .schemas import (
    ContinueScene,
    EndGame,
    EndScene,
    Option,
    ProsePart,
    StateChangeBundle,
)
from .state import LogEntry, State
from .world import World


# ───────────────────────── agent bundle ─────────────────────────


@dataclass
class AgentBundle:
    """One world's scene-runner agents. Built once in lifespan via
    `build_scene_runner_bundle(model, world)`.

    Single phase-1 + single phase-2 agent. Per-call dynamic schemas are
    passed at `agent.run()` time via the output-function pattern (see
    emit_state_changes / emit_prose). System prompts are byte-stable across
    the session → KV cache hits across turns. The phase-1 and phase-2
    prompts differ → no inter-phase cache sharing within a turn (deliberate
    trade for Phase 2 attention focus).
    """

    phase1: Agent[None, StateChangeBundle]
    phase2: Agent[None, ProsePart]
    world: World


@dataclass
class SceneRunnerRuntime:
    """Scene-scoped Phase 2 conversation state. Reset at scene boundary
    (engine.enter_scene). One per session.

    Phase 2 is conversation-based within a scene: the model sees its own
    prior turns' prose as assistant messages. Voice continuity (sentence
    rhythm, narrator persona, dialogue tag style) accumulates naturally
    over the 4-8 turns of a scene. Reset to empty at scene boundary so
    the next scene's KV cache starts fresh from the (cached) system
    prompt.

    On Phase 2 retry exhaustion (UnexpectedModelBehavior → fallback
    prose path in scene_loop), the runtime is NOT mutated. The fallback
    prose is surfaced to the player but never persisted to message
    history — preserves voice continuity by treating the failed turn as
    if it didn't exist in the conversation, even though the player saw
    generic stand-in prose.
    """

    phase2_message_history: list[ModelMessage] = field(default_factory=list)


def make_scene_runner_runtime() -> SceneRunnerRuntime:
    """Construct a fresh scene-runner runtime. Call once per session
    (mirrors make_log_writer_runtime)."""
    return SceneRunnerRuntime()


# ───────────────────────── dynamic type builders ─────────────────────────


def _build_state_change_bundle_type(world: World, state: State) -> type[BaseModel]:
    """StateChangeBundle whose UnlockBeat.beat_id is
    `Literal[*reachable_beat_ids(world, state)]`. The grammar sampler will
    reject any beat_id not currently reachable at decode time — covers
    requires_beats/blocked_by_beats/already-fired in one Literal. `__base__=`
    inherits `_no_duplicate_beat_ids` from canonical so the rule fires
    during the agent's retry loop (ValidationError → ModelRetry).
    """
    reachable = tuple(sorted(reachable_beat_ids(world, state)))
    if not reachable:
        # Nothing fireable from this state → bundle must be empty.
        return create_model(
            "StateChangeBundle",
            __base__=StateChangeBundle,
            state_changes=(
                list[Any],
                Field(default_factory=list, max_length=0),
            ),
        )
    BeatLiteral = Literal[reachable]  # type: ignore[valid-type]
    DynamicUnlockBeat = create_model(
        "UnlockBeat",
        type=(Literal["unlock_beat"], "unlock_beat"),
        beat_id=(BeatLiteral, ...),
    )
    StateChange = Annotated[DynamicUnlockBeat, Field(discriminator="type")]
    return create_model(
        "StateChangeBundle",
        __base__=StateChangeBundle,
        state_changes=(
            list[StateChange],
            Field(default_factory=list, max_length=2),
        ),
    )


def _build_prose_part_type(
    world: World,
    state: State,
    *,
    leave_pick: Option | None = None,
) -> type[BaseModel]:
    """ProsePart whose Option.leaves_for is `Literal[*adjacent_ids] | None`
    and EndGame.ending_id is `Literal[*satisfied_ending_ids]`. When
    `leave_pick` is set, `next` is narrowed to `EndScene` (no union) so the
    grammar physically prevents emitting ContinueScene/EndGame on a
    leave-pick turn.

    State-independent rules (≥1 leave option in ContinueScene) live on the
    canonical schema as @model_validator and fire when the output function
    round-trips dynamic → canonical.
    """
    if leave_pick is not None:
        return create_model("ProsePart", prose=(str, ...), next=(EndScene, ...))

    location = world.locations[state.player_location]
    adjacent_ids = tuple(location.adjacent_to)
    satisfied_endings = tuple(sorted(
        eid for eid, e in world.endings.items()
        if set(e.requires_beats).issubset(state.unlocked_beats)
    ))

    if adjacent_ids:
        LeavesForType = Union[Literal[adjacent_ids], None]  # type: ignore[valid-type]
    else:
        LeavesForType = None
    DynamicOption = create_model(
        "Option",
        text=(str, ...),
        leaves_for=(LeavesForType, None),
    )
    DynamicContinue = create_model(
        "ContinueScene",
        __base__=ContinueScene,
        options=(list[DynamicOption], ...),
    )
    next_variants: list[type] = [DynamicContinue, EndScene]
    if satisfied_endings:
        DynamicEndGame = create_model(
            "EndGame",
            type=(Literal["end_game"], "end_game"),
            ending_id=(Literal[satisfied_endings], ...),  # type: ignore[valid-type]
        )
        next_variants.append(DynamicEndGame)
    NextUnion = Union[tuple(next_variants)]  # type: ignore[valid-type]
    SceneNext = Annotated[NextUnion, Field(discriminator="type")]
    return create_model("ProsePart", prose=(str, ...), next=(SceneNext, ...))


# ───────────────────────── factory ─────────────────────────


def build_scene_runner_bundle(
    model: OpenAIChatModel, world: World
) -> AgentBundle:
    """Construct the scene-runner agent bundle for `world`. Bakes two
    system prompts (phase 1 classifier, phase 2 narrator) and builds one
    phase-1 agent + one phase-2 agent. Per-call dynamic output_type is
    passed at agent.run() time via the output-function pattern.
    """
    phase1 = Agent[None, StateChangeBundle](
        model=model,
        system_prompt=build_phase1_system_prompt(world),
        output_retries=3,
    )
    phase2 = Agent[None, ProsePart](
        model=model,
        system_prompt=build_phase2_system_prompt(world),
        output_retries=3,
    )
    return AgentBundle(phase1=phase1, phase2=phase2, world=world)


# ───────────────────────── per-call run wrappers ─────────────────────────


async def emit_state_changes(
    bundle: AgentBundle,
    state: State,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
) -> StateChangeBundle:
    """Phase 1 LLM call. Stateless per turn (no message_history). Builds a
    full rich user prompt with reachable beats, endings, scene-so-far, case
    file — everything the classifier needs to judge what just fired.

    `output_type=NativeOutput(DynamicBundle, strict=True)` baked per call;
    DynamicBundle subclasses canonical so @model_validators fire during
    pydantic-ai's retry loop. The post-run round-trip canonicalizes the
    instance for downstream isinstance() dispatch.
    """
    prompt = build_phase1_user_prompt(
        state, bundle.world,
        case_file_entries=case_file_entries,
        opening_seed=opening_seed,
    )
    DynamicBundle = _build_state_change_bundle_type(bundle.world, state)
    result = await bundle.phase1.run(
        prompt,
        output_type=NativeOutput(DynamicBundle, strict=True),
    )
    return StateChangeBundle.model_validate(result.output.model_dump())


async def emit_prose(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    applied: StateChangeBundle,
    *,
    leave_pick: Option | None = None,
    opening_seed: str | None = None,
) -> ProsePart:
    """Phase 2 LLM call. Scene-bounded conversation: the model sees its
    own prior turns' prose as assistant messages via
    `runtime.phase2_message_history`. On success, history is updated with
    `result.all_messages()` (atomic: pydantic-ai's internal ModelRetry
    loop doesn't pollute — only the final accepted result is appended).

    On UnexpectedModelBehavior (retries exhausted), this function raises;
    caller (scene_loop.run_turn) catches and emits fallback prose, which
    does NOT mutate runtime.phase2_message_history.

    `is_first_turn` is computed from the history (empty → first turn).
    Whether the user message renders the scene anchor or just the
    incremental delta follows from that.
    """
    is_first_turn = not runtime.phase2_message_history
    user_msg = build_phase2_user_message(
        state, bundle.world, applied,
        leave_pick=leave_pick,
        opening_seed=opening_seed,
        is_first_turn=is_first_turn,
    )
    DynamicProsePart = _build_prose_part_type(
        bundle.world, state, leave_pick=leave_pick
    )
    result = await bundle.phase2.run(
        user_msg,
        message_history=runtime.phase2_message_history,
        output_type=NativeOutput(DynamicProsePart, strict=True),
    )
    runtime.phase2_message_history = result.all_messages()
    return ProsePart.model_validate(result.output.model_dump())


# ───────────────────────── log writer agent ─────────────────────────


@dataclass
class LogWriterRuntime:
    """Session-scoped state for the conversation-based log writer.

    The agent is stateless across instances of the runtime (system prompt
    baked once for the world). `message_history` accumulates one user +
    one assistant message per scene-end; the conversation IS the case file
    so far. Replaced on each successful call with `result.all_messages()`.

    Why a dataclass and not a global: tests need to construct fresh
    runtimes per test; multiple sessions (if v2 grows beyond single-
    session) would each carry their own history.
    """

    agent: Agent[None, str]
    message_history: list[ModelMessage] = field(default_factory=list)


def build_log_writer_agent(
    model: OpenAIChatModel, world: World
) -> Agent[None, str]:
    """Conversation-based log writer agent. System prompt is baked once
    (full bible + voice rules); each call appends one user message (scene
    summary) and gets one assistant message (the case-file entry text)
    back. Output is plain `str` — no NativeOutput / Literal grammar
    constraint, since the case file is freeform prose.

    The OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)
    flag is set on the model in lifespan / _build_model — load-bearing
    for this agent. Without it, pydantic-ai's default 'auto' round-trips
    `ThinkingPart`s into prior assistant content as <think>...</think>
    blocks, which Gemma 4 cannot tolerate in multi-turn.
    """
    return Agent[None, str](
        model=model,
        output_type=str,
        system_prompt=build_log_writer_system_prompt(world),
    )


def make_log_writer_runtime(
    model: OpenAIChatModel, world: World
) -> LogWriterRuntime:
    """Construct a fresh LogWriterRuntime. Call once per session."""
    return LogWriterRuntime(agent=build_log_writer_agent(model, world))


async def emit_log_entry(
    runtime: LogWriterRuntime,
    world: World,
    state: State,
) -> LogEntry:
    """Generate one case-file entry for the just-completed scene.

    Reads scene anchor (scene_id, location, npcs present) from
    `state.dialogue_history` — at the point this is called (after
    play_scene returns EndScene/EndGame), the just-completed scene's
    turns are still in dialogue_history; the next enter_scene will
    clear it.

    Side effect: mutates `runtime.message_history` with the just-rendered
    user+assistant pair. Subsequent calls see this scene's entry in
    their conversation context.
    """
    # Caller (play_session) only fires this after play_scene returns;
    # the just-completed scene guarantees ≥1 turn in dialogue_history.
    assert state.dialogue_history, "emit_log_entry: empty dialogue_history"
    user_prompt = build_log_writer_user_prompt(state, world)
    result = await runtime.agent.run(
        user_prompt, message_history=runtime.message_history
    )
    runtime.message_history = result.all_messages()
    anchor = state.dialogue_history[0]
    return LogEntry(
        scene_id=anchor.scene_id,
        location_id=anchor.location_id,
        npcs_present=list(anchor.npcs_present),
        text=result.output.strip(),
    )
