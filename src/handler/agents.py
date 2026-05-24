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
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import logfire
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings

from .config import THINKING_OFF_EXTRA_BODY
from .engine import emittable_beat_ids, present_npc_ids
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
    build_village_attention_system_prompt,
    build_village_attention_user_prompt,
)
from .schemas import (
    ChoiceBundle,
    ChronicleEntry,
    FreeTextCaption,
    FreeTextGate,
    GuardDelta,
    NpcMemoryUpdate,
    NpcPlacement,
    NpcPlacementBundle,
    QualityDelta,
    StateChangeBundle,
    TrustDelta,
    UnlockBeat,
    VillageAttentionVerdict,
)
from .state import State
from .world import World


# ───────────────────────── agent bundle ─────────────────────────


@dataclass
class AgentBundle:
    """One world's agents. Built once in lifespan via
    `build_scene_runner_bundle(model, world)`.

    Eight agents (Phase 1 classifier, prose writer, choice author, free-text
    gate, caption author, populator, chronicler, village-attention) have
    world-scoped system prompts and are constructed once at bundle-build time.
    Per-call dynamic schemas are passed at `agent.run()` time via NativeOutput
    (or `output_type=str` for the prose writer). System prompts are byte-stable
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
    village_attention: Agent[None, VillageAttentionVerdict]
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
    """StateChangeBundle narrowed per call for the quality engine:

      - `npc_deltas`: a list of QualityDelta (npc_id narrowed to present,
        non-breached NPCs), at most one per present NPC. The grammar sampler
        physically rejects a delta for an absent or breached NPC, and the
        trust/guard magnitudes are already Literal-bounded on the canonical
        schema.
      - `commitments`: at most one UnlockBeat whose `beat_name` is narrowed to
        `emittable_beat_ids` (observation milestones + currently-available
        decisions). Evidence/outcome milestones are engine-derived and never
        appear here.

    Village attention is NOT a channel here. It was once a scalar `attention`
    field on this bundle, but the model anchored on the salient per-NPC delta and
    left it at its default — private aggression never registered. It now has its
    own dedicated scene-close agent (`emit_village_attention`), asked in isolation,
    which classifies village suspicion reliably.

    Names (not ids) on the commitment wire — `emit_state_changes` translates
    `beat_name → beat_id` after decode, same as before.
    """
    adjustable = tuple(
        n
        for n in present_npc_ids(world, state)
        if n not in state.guard_breached and not world.npcs[n].background
    )

    # ── npc_deltas: one optional QualityDelta per present, non-breached NPC ──
    if adjustable:
        NpcLiteral = Literal[adjustable]  # type: ignore[valid-type]
        DynQuality = create_model(
            "QualityDelta",
            npc_id=(NpcLiteral, ...),
            trust=(TrustDelta, "0"),
            guard=(GuardDelta, "0"),
            # Required: the prose moment that earns a non-zero nudge. Forces the
            # classifier to articulate a real disclosure / careless move before
            # moving standing — the discipline that ends the +1-every-turn drip.
            reason=(str, ...),
        )
        npc_deltas_field: Any = (
            list[DynQuality],
            Field(default_factory=list, max_length=len(adjustable)),
        )
    else:
        npc_deltas_field = (list[Any], Field(default_factory=list, max_length=0))

    # ── commitments (≤1, narrowed to emittable names) ──
    emittable_ids = tuple(sorted(emittable_beat_ids(world, state)))
    emittable_names = tuple(world.beats[bid].name for bid in emittable_ids)
    if emittable_names:
        CommitLiteral = Literal[emittable_names]  # type: ignore[valid-type]
        DynCommit = create_model(
            "UnlockBeat",
            type=(Literal["unlock_beat"], "unlock_beat"),
            beat_name=(CommitLiteral, ...),
            # A short quote of the PLAYER'S own words / chosen action this turn
            # that performs this commitment. Forces the classifier to ground the
            # fire in what the player did, and lets `emit_state_changes` drop a
            # commitment the narrator's prose drifted into but the player never
            # actually did (judge: beat_fired_appropriately).
            warrant=(str, ...),
        )
        commitments_field = (
            list[Annotated[DynCommit, Field(discriminator="type")]],
            Field(default_factory=list, max_length=1),
        )
    else:
        commitments_field = (list[Any], Field(default_factory=list, max_length=0))

    return create_model(
        "StateChangeBundle",
        npc_deltas=npc_deltas_field,
        commitments=commitments_field,
    )


# ───────────────────────── populator dynamic schema ─────────────────────────


# Sentinel value for "keep this NPC's current location". A real location whose
# id equals this would collide in the wire enum, so the builder rejects it.
_STAY = "stay"


def _build_npc_placement_bundle_type(world: World) -> type[BaseModel]:
    """Populator output as a FIXED-SHAPE OBJECT — one field PER cast NPC, keyed
    by `npc_id`, whose value is `Literal[*location_ids, "stay"]` (default
    `"stay"` = keep current location).

    No array, so llama.cpp's grammar sampler — which enforces object keys and
    enum values but NOT array length — cannot loop filling a `placements` list
    to the token ceiling (the multi-minute runaway the old list shape allowed).
    Keying by id also makes "one placement per NPC" structurally true.

    `emit_placements` reads each field, skips `"stay"`, and builds the canonical
    list-shaped `NpcPlacementBundle` the engine consumes.

    Empty cast or world → an empty object (no fields); `emit_placements` then
    yields no placements, mirroring the old `max_length=0` path.
    """
    npc_ids = tuple(world.npcs.keys())
    location_ids = tuple(world.locations.keys())
    if not npc_ids or not location_ids:
        return create_model(
            "NpcPlacementBundle", __config__=ConfigDict(extra="forbid")
        )
    # A location whose id is the keep-current sentinel would be silently
    # un-targetable (emit_placements treats `_STAY` as "no move"); fail loud.
    if _STAY in location_ids:
        raise ValueError(
            f"a location id is {_STAY!r}, which collides with the populator "
            "keep-current sentinel"
        )
    LocLiteral = Literal[(*location_ids, _STAY)]  # type: ignore[valid-type]
    fields: dict[str, Any] = {npc_id: (LocLiteral, _STAY) for npc_id in npc_ids}
    return create_model(
        "NpcPlacementBundle", __config__=ConfigDict(extra="forbid"), **fields
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
    """Construct the agent bundle for `world`. Bakes eight system prompts
    (Phase 1 classifier, prose writer, choice author, free-text gate,
    caption author, populator, chronicler, village-attention) and builds one
    agent per role.
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
    village_attention = build_village_attention_agent(model, world)
    return AgentBundle(
        phase1=phase1,
        prose_writer=prose_writer,
        choice_author=choice_author,
        free_text_gate=free_text_gate,
        caption_author=caption_author,
        populator=populator,
        chronicler=chronicler,
        village_attention=village_attention,
        world=world,
    )


# ───────────────────────── per-call run wrappers ─────────────────────────


# Stopwords stripped before comparing a commitment's `warrant` to the player's
# pick — connective / framing words that would create spurious overlap.
_WARRANT_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "it", "is",
    "was", "you", "your", "they", "them", "this", "that", "with", "for",
    "about", "ask", "asks", "asked", "say", "says", "said", "tell", "tells",
    "told", "what", "who", "her", "him", "his", "she", "he", "from", "into",
})


def _warrant_supported(
    warrant: str, pick_text: str, triggers: tuple[str, ...] = ()
) -> bool:
    """True when the commitment is genuinely warranted by the player's pick this
    turn — a guard that the beat reflects what the PLAYER did, not what the
    narrator's prose drifted into.

    Two modes:
      - `triggers` set (object-anchored observation beats — the letter, the map,
        the grave): the player's pick MUST reference one of those object tokens.
        This is the strict gate that stops "Read Iwan's letter" firing when the
        player walked to the study to look at the map (judge:
        beat_fired_appropriately) — the letter beat's triggers don't appear in a
        map action, so it is dropped.
      - no `triggers` (NPC findables, decisions): the looser fallback — the
        `warrant` must share a content word (4-char stem) with the pick.

    Conservative by design: a dropped commitment is recoverable — the beat stays
    emittable and the choice author re-steers toward it next turn — so the
    asymmetry favors dropping over a false fire.
    """
    if triggers:
        p = pick_text.lower()
        return any(t.lower() in p for t in triggers if t)

    def stems(s: str) -> set[str]:
        return {
            w[:4]
            for w in re.findall(r"[a-z]+", s.lower())
            if len(w) > 2 and w not in _WARRANT_STOPWORDS
        }

    w, p = stems(warrant), stems(pick_text)
    if not w or not p:
        return False
    return bool(w & p)


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
    `StateChangeBundle` (whose validators enforce the `max_length` caps and
    per-NPC delta uniqueness at the boundary).

    Two cleanups happen here before the canonical bundle is built, because
    the grammar sampler constrains *which* ids/values are emittable but not
    *how many* or whether they move anything:
      - no-op deltas (trust=guard="0") are dropped — they shift no state but
        would otherwise pollute the per-turn ledger and read as "movement" in
        the filler metric;
      - at most one delta per NPC is kept, so a single turn can never move
        trust/guard by more than the ±1/turn the band design rests on (the
        grammar narrows npc_id but does not forbid repeats within the list).
    """
    prompt = build_phase1_user_prompt(state, bundle.world)
    DynamicBundle = _build_state_change_bundle_type(bundle.world, state)
    result = await _run_decision_agent(
        bundle.phase1,
        prompt,
        NativeOutput(DynamicBundle, strict=True),
    )
    raw = result.output
    deltas: list = []
    seen_npcs: set[str] = set()
    for d in getattr(raw, "npc_deltas", []):
        if d.trust == "0" and d.guard == "0":
            continue  # no-op nudge — moves nothing
        reason = (getattr(d, "reason", "") or "").strip()
        if not reason:
            # A non-zero nudge with no stated justification is the drip the
            # rework targets — drop it. The NPC stays warmable, so a genuine
            # move next turn still lands; only the unjustified tick is lost.
            logfire.info(
                "delta dropped: no reason for non-zero nudge",
                npc_id=d.npc_id,
                trust=d.trust,
                guard=d.guard,
                location=state.player_location,
            )
            continue
        if d.npc_id in seen_npcs:
            continue  # keep ±1/turn: first delta per NPC wins
        seen_npcs.add(d.npc_id)
        deltas.append(
            QualityDelta(npc_id=d.npc_id, trust=d.trust, guard=d.guard, reason=reason)
        )

    # Commitments are warranted against the player's pick THIS turn: the action
    # that triggered this turn is on the last dialogue_history record (set in
    # play_scene before the next run_turn). A commitment whose `warrant` shares
    # no content word with that pick is dropped — it fired on the narrator's
    # prose, not the player's action (or it is turn 1, which has no pick and so
    # cannot warrant anything). The beat stays emittable, so a false drop simply
    # defers the fire by a turn.
    last_pick = None
    if state.dialogue_history and state.dialogue_history[-1].player_pick is not None:
        last_pick = state.dialogue_history[-1].player_pick.text
    commitments: list[UnlockBeat] = []
    for c in getattr(raw, "commitments", []):
        if getattr(c, "beat_name", None) is None:
            continue
        beat_id = bundle.world.beat_id_for_name(c.beat_name)
        triggers = tuple(bundle.world.beats[beat_id].triggers)
        warrant = getattr(c, "warrant", "") or ""
        if last_pick is None or not _warrant_supported(warrant, last_pick, triggers):
            logfire.info(
                "commitment dropped: warrant unsupported by player action",
                beat_name=c.beat_name,
                warrant=warrant,
                player_pick=last_pick,
                triggers=triggers,
                location=state.player_location,
            )
            continue
        commitments.append(UnlockBeat(beat_id=beat_id))
    return StateChangeBundle(deltas=deltas, commitments=commitments)


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

    The wire output is a fixed-shape object (one `Literal[*location_ids,
    "stay"]` field per NPC); this reads each field, skips `"stay"` (the
    keep-current sentinel), and builds the canonical list-shaped
    `NpcPlacementBundle` the engine consumes.
    """
    prompt = build_populator_user_prompt(state, bundle.world)
    DynamicBundle = _build_npc_placement_bundle_type(bundle.world)
    result = await _run_decision_agent(
        bundle.populator,
        prompt,
        NativeOutput(DynamicBundle, strict=True),
    )
    raw = result.output
    placements = [
        NpcPlacement(npc_id=npc_id, location_id=getattr(raw, npc_id))
        for npc_id in bundle.world.npcs
        if getattr(raw, npc_id, _STAY) != _STAY
    ]
    return NpcPlacementBundle(placements=placements)


async def stream_prose_text(
    bundle: AgentBundle,
    runtime: SceneRunnerRuntime,
    state: State,
    applied_beats: list[str],
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
        state, bundle.world, applied_beats,
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
    *,
    leave_required: bool = False,
) -> ChoiceBundle:
    """Choice author LLM call. Stateless (no message_history). The
    just-written prose is the load-bearing context; the system prompt
    has no bible/voice rules.

    The output schema is static (`Continue | Leave`, no per-call id
    narrowing) so the canonical `ChoiceBundle` is passed directly — its
    `Continue` 2-4-option validator fires during the agent's retry loop.

    `leave_required` (the engine judged the scene spent / a decision just fired)
    adds the "Offer The Exit" directive so one option is the diegetic departure.
    """
    user_msg = build_choice_author_user_message(
        state, bundle.world, prose, leave_required=leave_required
    )
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


# ───────────────────────── village-attention agent ─────────────────────────


def build_village_attention_agent(
    model: OpenAIChatModel, world: World
) -> Agent[None, VillageAttentionVerdict]:
    """Dedicated scene-close classifier for the global `village_attention` heat.

    Split out of Phase 1: bundled with the per-NPC trust/guard deltas the model
    anchored on those and left attention at its default, so private aggression
    never registered. Asked in isolation — only "how wary is the village now?" —
    the same model classifies it reliably. System prompt is baked once (world +
    voice-only cast, NO knowledge); each call is independent (no message_history).

    Output spec (`NativeOutput(VillageAttentionVerdict, strict=True)`) is passed at
    `agent.run()` time in `emit_village_attention`, mirroring the chronicler so
    TestModel-based unit tests can bypass the native-output profile check.
    Content/judgement agent, so NO `decision_settings` token cap.
    """
    return Agent[None, VillageAttentionVerdict](
        model=model,
        name="village_attention",
        system_prompt=build_village_attention_system_prompt(world),
        output_retries=3,
    )


async def emit_village_attention(
    bundle: AgentBundle,
    world: World,
    state: State,
) -> str:
    """Judge the village's attention level for the just-closed scene; returns the
    level word (`none`|`noticed`|`talked_about`|`hostile`). The caller ratchets
    `state.village_attention` up via `engine.apply_village_attention` — this stays
    side-effect-free, like `emit_chronicle_entry`.

    Read off the live `state`, same as the chronicler: at scene close the scene's
    turns are still in `dialogue_history` and `player_location` still points at
    the scene's location. The static 4-level schema needs no per-call narrowing.
    """
    assert state.dialogue_history, "emit_village_attention: empty dialogue_history"
    user_prompt = build_village_attention_user_prompt(state, world)
    result = await bundle.village_attention.run(
        user_prompt,
        output_type=NativeOutput(VillageAttentionVerdict, strict=True),
    )
    return result.output.level
