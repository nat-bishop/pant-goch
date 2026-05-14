"""Scene-runner pydantic-ai agent + dynamic per-call output schemas.

Two-stage emission per turn (plan §"Agents (refines §3.3, §3.4, §4.1)"):
  Phase 1 — emit_state_changes(): structured `StateChangeBundle`. Validated
            via @agent.output_validator → ModelRetry on requires/blocks
            violations, ≤2-cap, undefined ids. output_retries=3.
  Phase 2 — emit_prose(): structured `ProsePart`. Validated for ending
            prereqs, ≥1 leave option per ContinueScene, leave-pick forces
            EndScene. output_retries=3.

The two phases share one stable system prompt (the world bible, baked at
`build_scene_runner_bundle()`). Per-call dynamic types narrow the str-typed
schemas in `schemas.py` to `Literal[*beat_ids]`, `Literal[*adjacent_ids]`,
`Literal[*ending_ids]` so the llama.cpp grammar sampler rejects invalid ids
at decode time.

Pattern lift: stable-system / dynamic-user split (handler/src/handler/agents.py:206-390),
NativeOutput strict=True (handler/src/handler/agents.py:1230-1232), dynamic
Literal type builders (handler/src/handler/agents.py:919-961). The retry
orchestration is REPLACED by pydantic-ai's @output_validator + output_retries=3
— per `select_then_write_with_retries` lift note in plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel

from .engine import (
    ProsePartError,
    StateChangeError,
    validate_prose_part,
    validate_state_change_bundle,
)
from .prompts import (
    build_log_writer_system_prompt,
    build_log_writer_user_prompt,
    build_phase1_tail,
    build_phase2_tail,
    build_prompt_prefix,
    build_scene_runner_system_prompt,
)
from .schemas import (
    ContinueScene,
    EndGame,
    EndScene,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from .state import LogEntry, State
from .world import World


# ───────────────────────── deps for output_validators ─────────────────────────


@dataclass
class Phase1Deps:
    state: State


@dataclass
class Phase2Deps:
    state: State
    leave_pick: Option | None = None


# ───────────────────────── agent bundle ─────────────────────────


@dataclass
class AgentBundle:
    """One world's scene-runner agents. Built once in lifespan via
    `build_scene_runner_bundle(model, world)`.

    Why N+1 agents: pydantic-ai 1.93+ forbids passing `output_type=` at
    `agent.run()` when `@agent.output_validator` is registered (it raises
    `UserError("Cannot set a custom run output_type when the agent has
    output validators")`). So we bake the per-call dynamic type at
    construction. Phase 1's `Literal[*beat_ids]` is world-scoped (one
    agent). Phase 2's `Literal[*adjacent_ids]` is location-scoped (one
    agent per location). All N+1 agents share byte-identical system
    prompts → KV cache hits on the system prefix regardless of which
    agent issues the HTTP request.
    """

    phase1: Agent[Phase1Deps, Any]
    phase2_by_location: dict[str, Agent[Phase2Deps, Any]]
    world: World


# ───────────────────────── dynamic type builders ─────────────────────────


def _build_state_change_bundle_type(world: World) -> type[BaseModel]:
    """StateChangeBundle whose UnlockBeat.beat_id is `Literal[*beat_ids]`.
    The grammar sampler will reject any beat_id not in this enum at decode
    time. Application-layer rules (requires_beats, blocked_by_beats, ≤2 cap)
    are checked in the output_validator post-decode.
    """
    beat_ids = tuple(world.beats.keys())
    if not beat_ids:
        # Edge: world with no beats. The schema can't have an empty Literal,
        # but the LLM can still emit StateChangeBundle(state_changes=[]).
        # Use the str-typed UnlockBeat from schemas.py — engine validator will
        # reject any beat_id since no beats are defined.
        DynamicUnlockBeat = UnlockBeat
    else:
        BeatLiteral = Literal[beat_ids]  # type: ignore[valid-type]
        DynamicUnlockBeat = create_model(
            "UnlockBeat",
            type=(Literal["unlock_beat"], "unlock_beat"),
            beat_id=(BeatLiteral, ...),
        )
    StateChange = Annotated[DynamicUnlockBeat, Field(discriminator="type")]
    return create_model(
        "StateChangeBundle",
        state_changes=(list[StateChange], Field(default_factory=list)),
    )


def _build_prose_part_type(world: World, location_id: str) -> type[BaseModel]:
    """ProsePart whose Option.leaves_for is `Literal[*adjacent_loc_ids] | None`
    and EndGame.ending_id is `Literal[*ending_ids]`. Adjacencies are
    location-specific, so this is built per location_id (cached in the
    bundle's prose_part_types_by_loc dict)."""
    location = world.locations[location_id]
    adjacent_ids = tuple(location.adjacent_to)
    ending_ids = tuple(world.endings.keys())

    if adjacent_ids:
        LeavesForType = Union[Literal[adjacent_ids], None]  # type: ignore[valid-type]
    else:
        # No adjacents from this location: leaves_for can only be None.
        # The validator will then reject ContinueScene (no leave option
        # possible) — caller should expect EndScene from this location.
        LeavesForType = None
    DynamicOption = create_model(
        "Option",
        text=(str, ...),
        leaves_for=(LeavesForType, None),
    )

    DynamicContinue = create_model(
        "ContinueScene",
        type=(Literal["continue"], "continue"),
        options=(list[DynamicOption], ...),
    )
    DynamicEndScene = create_model(
        "EndScene",
        type=(Literal["end_scene"], "end_scene"),
    )
    if ending_ids:
        EndingLiteral = Literal[ending_ids]  # type: ignore[valid-type]
        DynamicEndGame = create_model(
            "EndGame",
            type=(Literal["end_game"], "end_game"),
            ending_id=(EndingLiteral, ...),
        )
        NextUnion = Union[DynamicContinue, DynamicEndScene, DynamicEndGame]
    else:
        # No endings authored: drop EndGame from the union entirely.
        NextUnion = Union[DynamicContinue, DynamicEndScene]
    SceneNext = Annotated[NextUnion, Field(discriminator="type")]

    return create_model(
        "ProsePart",
        prose=(str, ...),
        next=(SceneNext, ...),
    )


# ───────────────────────── factory ─────────────────────────


def _build_phase1_agent(
    model: OpenAIChatModel, world: World, system_prompt: str
) -> Agent[Phase1Deps, Any]:
    state_change_bundle_type = _build_state_change_bundle_type(world)
    agent = Agent[Phase1Deps, Any](
        model=model,
        deps_type=Phase1Deps,
        output_type=NativeOutput(state_change_bundle_type, strict=True),
        system_prompt=system_prompt,
        output_retries=3,
    )

    @agent.output_validator
    async def _validate(
        ctx: RunContext[Phase1Deps], output: BaseModel
    ) -> BaseModel:
        try:
            validate_state_change_bundle(output, world, ctx.deps.state)  # type: ignore[arg-type]
        except StateChangeError as e:
            raise ModelRetry(str(e))
        return output

    return agent


def _build_phase2_agent(
    model: OpenAIChatModel,
    world: World,
    system_prompt: str,
    location_id: str,
) -> Agent[Phase2Deps, Any]:
    prose_part_type = _build_prose_part_type(world, location_id)
    agent = Agent[Phase2Deps, Any](
        model=model,
        deps_type=Phase2Deps,
        output_type=NativeOutput(prose_part_type, strict=True),
        system_prompt=system_prompt,
        output_retries=3,
    )

    @agent.output_validator
    async def _validate(
        ctx: RunContext[Phase2Deps], output: BaseModel
    ) -> BaseModel:
        try:
            validate_prose_part(
                output,  # type: ignore[arg-type]
                world,
                ctx.deps.state,
                leave_pick=ctx.deps.leave_pick,
            )
        except ProsePartError as e:
            raise ModelRetry(str(e))
        return output

    return agent


def build_scene_runner_bundle(model: OpenAIChatModel, world: World) -> AgentBundle:
    """Construct the scene-runner agent bundle for `world`. Bakes the
    system prompt with the full bible. Builds one phase-1 agent (output_type
    locked to the per-world StateChangeBundle) and N phase-2 agents (one
    per location, output_type locked to that location's per-call ProsePart
    type with adjacents-Literal in `leaves_for`).

    `strict=True` on NativeOutput is non-negotiable — without it pydantic-ai
    sends `response_format.json_schema.strict=false` and llama.cpp ignores
    the schema (lift-comment from v1 agents.py:976-981).
    """
    system_prompt = build_scene_runner_system_prompt(world)
    phase1 = _build_phase1_agent(model, world, system_prompt)
    phase2_by_location = {
        loc_id: _build_phase2_agent(model, world, system_prompt, loc_id)
        for loc_id in world.locations
    }
    return AgentBundle(
        phase1=phase1,
        phase2_by_location=phase2_by_location,
        world=world,
    )


# ───────────────────────── per-call run wrappers ─────────────────────────


async def emit_state_changes(
    bundle: AgentBundle,
    state: State,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
    prefix: str | None = None,
) -> StateChangeBundle:
    """Phase 1 LLM call. Returns the validated StateChangeBundle, round-tripped
    through the canonical str-typed model from schemas.py (the dynamic type's
    Literal-typed beat_id is a str at runtime).

    Pass `prefix` (from `build_prompt_prefix`) to share the byte-identical
    prefix bytes with phase 2 — caller (scene_loop.run_turn) builds it once
    per turn and passes it to both phases.
    """
    if prefix is None:
        prefix = build_prompt_prefix(
            state, bundle.world,
            case_file_entries=case_file_entries,
            opening_seed=opening_seed,
        )
    prompt = prefix + build_phase1_tail()
    result = await bundle.phase1.run(prompt, deps=Phase1Deps(state=state))
    return StateChangeBundle.model_validate(result.output.model_dump())


async def emit_prose(
    bundle: AgentBundle,
    state: State,
    applied: StateChangeBundle,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
    leave_pick: Option | None = None,
    prefix: str | None = None,
) -> ProsePart:
    """Phase 2 LLM call. Picks the per-location agent (whose output_type
    has `Literal[*adjacent_ids]` baked in) and returns the validated
    ProsePart in canonical form.

    Pass `prefix` (from `build_prompt_prefix`) to share the byte-identical
    prefix bytes with phase 1.
    """
    if prefix is None:
        prefix = build_prompt_prefix(
            state, bundle.world,
            case_file_entries=case_file_entries,
            opening_seed=opening_seed,
        )
    prompt = prefix + build_phase2_tail(applied, leave_pick=leave_pick)
    agent = bundle.phase2_by_location[state.player_location]
    result = await agent.run(
        prompt, deps=Phase2Deps(state=state, leave_pick=leave_pick)
    )
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
    runtime.message_history = list(result.all_messages())
    anchor = state.dialogue_history[0]
    return LogEntry(
        scene_id=anchor.scene_id,
        location_id=anchor.location_id,
        npcs_present=list(anchor.npcs_present),
        text=result.output.strip(),
    )
