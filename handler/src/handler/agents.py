"""LLM agents — three stateless one-shot pydantic-ai agents per HANDLER_SPEC_V2 §8.

- `SOP_AGENT` — synthesises the State-of-Play memo each cycle, Operations
  voice (str output, streamed).
- `DRAFTER_AGENT` — produces 3–4 candidate options per cycle, structured
  `OptionSet` output, Operations voice.
- `RENDERER_AGENT` — renders the field dispatch in the acting agent's voice
  (str output, streamed).

All three are stateless — **no `message_history`** is passed between cycles.
Each cycle is a fresh prompt; continuity is preserved by explicitly passing
recent dispatches in the user-side context. §8.4 explains why.

Prompt structure (KV-cache friendly per §8.4 closing notes):
  [stable system prompt] — voice register, dossiers, mission framing, rules
  [dynamic user prompt]  — current state, recent dispatches, this-cycle context

The system prompts are baked at `build_agent_bundle(model, mission)` time —
once per mission load. User prompts are built per call by `_sop_user_prompt`,
`_drafter_user_prompt`, `_renderer_user_prompt`.

This module has NO direct LLM call sites. `cycle.py` wraps the agent runs
into the `SopFn`/`DrafterFn`/`RendererFn` protocols; `main.py` wires up the
paced word-stream emitter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import logfire
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel

from .models import (
    AgentDossier,
    Mission,
    Moment,
    Option,
    OptionSet,
    State,
)


log = logging.getLogger(__name__)


# ───────────────────────── dispatch record (used in prompts) ─────────────────────────


@dataclass
class DispatchRecord:
    """One past dispatch surfaced as user-side context for continuity. Lives
    in main.py's session memory (newest-first). The renderer prefers
    same-agent records when available, per §8.3."""

    cycle_index: int
    acting: str  # agent id, or "operations"
    archetype: str
    text: str


# ───────────────────────── bundle ─────────────────────────


@dataclass
class AgentBundle:
    """One mission's three agents. Built once at lifespan boot via
    `build_agent_bundle(model, mission)`."""

    sop: Agent[None, str]
    drafter: Agent[None, OptionSet]
    renderer: Agent[None, str]
    mission: Mission


# ───────────────────────── system-prompt builders ─────────────────────────


def _agent_dossier_block(dossiers: list[AgentDossier]) -> str:
    parts: list[str] = []
    for d in dossiers:
        parts.append(
            f"  - {d.id} (file {d.file_number}, codename {d.codename!r}): "
            f"{d.voice_register}\n    {d.voice_details.strip()}"
        )
    return "\n".join(parts)


_VOICE_OPERATIONS = (
    "third-party intelligence handler, terse, declarative. Speaks in service "
    "of decision-making, not narration. Names threats, resources, what hasn't "
    "moved. Never predicts outcomes. Never decorates. Sentences are short."
)


def _sop_system_prompt(mission: Mission) -> str:
    return f"""\
You are Operations Desk. You write the State of Play memo at the start of
every cycle of {mission.name} ({mission.codename}).

VOICE: {_VOICE_OPERATIONS}

MISSION:
{mission.opening_prose.strip()}

AGENTS DEPLOYED:
{_agent_dossier_block(mission.agents)}

RULES:
- Look forward. Name the threat. Name the resources. Name what has not moved.
- 1–3 paragraphs. ~120–180 words total.
- May reference state numerically ("KESTREL's cover is at 7") or narratively.
- NEVER propose options (that is the drafter's job).
- NEVER predict outcomes. NEVER decorate.
- Reflect the actual current state. Do not contradict prior dispatches.
- Output the prose only — no headings, no labels, no preamble.
"""


def _drafter_system_prompt(mission: Mission) -> str:
    return f"""\
You are Operations Desk drafting candidate courses of action for {mission.codename}.
Each cycle you produce 3–4 OPTIONS the Handler can authorize.

VOICE: {_VOICE_OPERATIONS}

MISSION:
{mission.opening_prose.strip()}

AGENTS DEPLOYED:
{_agent_dossier_block(mission.agents)}

OUTPUT SHAPE (strict):
Each option carries:
  - `archetype`: one of the archetype ids from the VALID CATALOG below.
  - `agent`: the bound agent id when the catalog entry has one, else null.
  - `params`: copy from the catalog entry's params.
  - `title`: ~6–10 words, imperative voice (e.g. "Run eastern exfil — now").
  - `prose`: 2–3 sentences naming the action, the actor, the cost honestly.
  - `satisfies_moment`: null, unless one option satisfies the FORCED MOMENT.

VALIDITY RULES (the engine REJECTS options that violate these):
- `archetype` MUST be present in the VALID CATALOG you receive.
- `agent` MUST match the catalog entry (or be null when entry has no agent).
- If a FORCED MOMENT block is present this cycle, exactly ONE option must
  set `satisfies_moment` to the moment id AND honor its rules.
- 3 or 4 options per cycle.

PROSE RULES:
- Telegraph cost honestly. If an option costs cover, the prose names it.
- Span tonal range: at least one safe-but-slow option, one fast-but-risky.
- NEVER predict outcomes. NEVER fabricate state that isn't in the user prompt.
- HOLD options frame doing nothing as a real cost — passive pressure decays cover.
"""


def _renderer_system_prompt(mission: Mission) -> str:
    return f"""\
You write FIELD DISPATCHES for {mission.codename}. A dispatch is the
acting agent's report after the Handler authorized an action.

You will be told `ACTING` (an agent id, or "operations") and the agent's
dossier. Write in that voice. For "operations", write in the Operations
Desk register (terse, third-party, declarative).

AGENTS DEPLOYED:
{_agent_dossier_block(mission.agents)}

DISPATCH RULES:
- 1–4 paragraphs. ~90–180 words.
- Voice: as specified for ACTING. First-person for agents, third-party for operations.
- Honor the RESOLVED STATE DELTA. If cover dropped, the prose reflects what cost that.
- MUST NOT introduce facts inconsistent with state.
- MUST NOT predict the next cycle. The dispatch reports what just happened.
- MAY include first-person color, justification, character — within voice.
- Output the prose only — no salutation, no signoff lines unless the voice calls for one.
"""


# ───────────────────────── user-prompt builders ─────────────────────────


def _state_block(state: State, mission: Mission) -> str:
    parts: list[str] = []
    cycles = state.get_quality("mission.cycles_remaining")
    parts.append(f"cycles_remaining: {cycles} / {mission.max_cycles}")
    for agent in mission.agents:
        cover = state.get_quality(f"agent[{agent.id}].cover")
        parts.append(f"agent[{agent.id}].cover: {cover} / 10")
    for q, v in state.qualities.items():
        if q.startswith("mission.") and q != "mission.cycles_remaining":
            parts.append(f"{q}: {v}")
    if state.fired_beats:
        parts.append(f"fired_beats: {state.fired_beats}")
    if state.fired_moments:
        parts.append(f"fired_moments: {state.fired_moments}")
    return "\n".join(parts)


def _dispatches_block(dispatches: list[DispatchRecord], limit: int) -> str:
    if not dispatches:
        return "(no prior dispatches yet — this is cycle 1.)"
    chunk = dispatches[-limit:]
    parts: list[str] = []
    for d in chunk:
        parts.append(f"--- Cycle {d.cycle_index} — {d.acting} — {d.archetype}")
        parts.append(d.text.strip())
        parts.append("")
    return "\n".join(parts).rstrip()


def _sop_user_prompt(
    cycle_index: int, state: State, mission: Mission, dispatches: list[DispatchRecord]
) -> str:
    return f"""\
This is the State of Play memo for cycle {cycle_index} of {mission.max_cycles}.

CURRENT STATE:
{_state_block(state, mission)}

RECENT DISPATCHES (last 3, oldest first):
{_dispatches_block(dispatches, limit=3)}

Write the SoP memo now.
"""


def _catalog_block(catalog: list[tuple[str, str | None, dict[str, str]]]) -> str:
    lines = []
    for arch_id, agent_id, params in catalog:
        agent_str = agent_id or "null"
        lines.append(f"  - archetype={arch_id} agent={agent_str} params={params}")
    return "\n".join(lines)


def _forced_moment_block(moment: Moment | None) -> str:
    if moment is None:
        return "(no forced moment this cycle.)"
    return f"""\
A FORCED MOMENT must appear among this cycle's options:
  moment_id: {moment.id}
  intent: {moment.intent.strip()}
  must_communicate:
{chr(10).join(f"    - {x}" for x in moment.must_communicate)}
  must_not:
{chr(10).join(f"    - {x}" for x in moment.must_not)}
  tone: {moment.tone}
EXACTLY ONE option this cycle must set `satisfies_moment` = "{moment.id}"
and write prose that honors the rules above. The mechanical effect of the
moment is engine-applied; you only write the prose situated in current state.
"""


def _drafter_user_prompt(
    cycle_index: int,
    state: State,
    mission: Mission,
    sop_text: str,
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_moment: Moment | None,
    dispatches: list[DispatchRecord],
    *,
    retry_feedback: str | None = None,
) -> str:
    archetype_intents = "\n".join(
        f"  - {a.id}: {a.intent}" for a in mission.archetypes
    )
    retry = ""
    if retry_feedback:
        retry = f"\nPREVIOUS ATTEMPT WAS REJECTED:\n{retry_feedback}\n"
    return f"""\
Cycle {cycle_index} of {mission.max_cycles}. Draft 3 or 4 options now.

CURRENT STATE:
{_state_block(state, mission)}

RECENT DISPATCHES (last 2):
{_dispatches_block(dispatches, limit=2)}

STATE OF PLAY MEMO (this cycle, just rendered):
{sop_text.strip()}

VALID CATALOG (engine-computed; you MUST pick from this exact set):
{_catalog_block(catalog)}

ARCHETYPE INTENTS (for prose framing — these are NOT the player-facing prose):
{archetype_intents}

{_forced_moment_block(forced_moment)}{retry}
Produce 3 or 4 options now.
"""


def _renderer_user_prompt(
    cycle_index: int,
    state: State,
    mission: Mission,
    option: Option,
    delta: dict[str, tuple[Any, Any]],
    acting: str,
    dispatches: list[DispatchRecord],
) -> str:
    delta_lines = []
    for q, (before, after) in delta.items():
        delta_lines.append(f"  - {q}: {before} -> {after}")
    delta_str = "\n".join(delta_lines) if delta_lines else "  (no state changes)"

    if acting == "operations":
        voice_note = (
            "ACTING is operations — write in the Operations Desk voice (terse, "
            "third-party, declarative). Describe what counter-intel did, what "
            "the agents are doing, what the situation looks like from the desk."
        )
    else:
        dossier = mission.agent(acting)
        voice_note = (
            f"ACTING is {dossier.id} (codename {dossier.codename!r}). "
            f"Write FIRST-PERSON in this register: {dossier.voice_register}.\n"
            f"Character notes:\n{dossier.voice_details.strip()}"
        )

    # Prefer same-agent recent dispatches for voice anchoring (§8.3).
    if acting != "operations":
        same_agent = [d for d in dispatches if d.acting == acting]
        anchor = same_agent[-2:] if same_agent else dispatches[-1:]
    else:
        anchor = dispatches[-2:]

    return f"""\
Cycle {cycle_index} of {mission.max_cycles}. Render the FIELD DISPATCH.

{voice_note}

AUTHORIZED OPTION:
  archetype: {option.archetype}
  agent: {option.agent or "(none)"}
  title: {option.title}
  prose (the option as offered): {option.prose}

RESOLVED STATE DELTA (this cycle's mutations):
{delta_str}

CURRENT STATE (after delta):
{_state_block(state, mission)}

PRIOR DISPATCHES FOR VOICE ANCHORING:
{_dispatches_block(anchor, limit=2)}

Write the dispatch now.
"""


# ───────────────────────── factory ─────────────────────────


def build_agent_bundle(model: OpenAIChatModel, mission: Mission) -> AgentBundle:
    """Construct the three agents for `mission`, baking the stable prefix
    into each agent's system prompt. Reuses one OpenAIChatModel across all
    three — pydantic-ai handles concurrent calls."""
    sop = Agent(model=model, output_type=str, system_prompt=_sop_system_prompt(mission))
    drafter = Agent(
        model=model,
        output_type=NativeOutput(OptionSet),
        system_prompt=_drafter_system_prompt(mission),
    )
    renderer = Agent(
        model=model, output_type=str, system_prompt=_renderer_system_prompt(mission)
    )
    return AgentBundle(sop=sop, drafter=drafter, renderer=renderer, mission=mission)


# ───────────────────────── drafter validation + retry/fallback ─────────────────────────


class DrafterValidationError(ValueError):
    """One or more drafter constraints violated; retry with feedback."""


def validate_options(
    options: list[Option],
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_moment_id: str | None,
) -> None:
    """Raise `DrafterValidationError` with a corrective message if the option
    set doesn't satisfy the constraints. Used by `draft_with_retries` to
    decide whether to retry."""
    problems: list[str] = []
    if not (3 <= len(options) <= 4):
        problems.append(
            f"option count {len(options)} not in [3,4]; produce exactly 3 or 4 options"
        )
    valid_keys = {
        (a, ag) for (a, ag, _) in catalog
    }
    for i, opt in enumerate(options):
        if (opt.archetype, opt.agent) not in valid_keys:
            problems.append(
                f"option {i + 1}: archetype={opt.archetype!r} agent={opt.agent!r} "
                f"is not in the VALID CATALOG"
            )
    if forced_moment_id is not None:
        satisfiers = [o for o in options if o.satisfies_moment == forced_moment_id]
        if len(satisfiers) != 1:
            problems.append(
                f"forced moment {forced_moment_id!r}: expected exactly 1 option "
                f"with satisfies_moment={forced_moment_id!r}, found {len(satisfiers)}"
            )
    if problems:
        raise DrafterValidationError("; ".join(problems))


def engine_fallback_options(
    catalog: list[tuple[str, str | None, dict[str, str]]],
    mission: Mission,
    forced_moment: Moment | None,
) -> list[Option]:
    """Engine-authored fallback when the drafter exhausts retries. Picks
    deterministic options from the valid catalog with terse generic prose.

    Strategy:
    - If a forced moment is required, the first option satisfies it (using
      any agent-bound catalog entry as carrier — moments apply their effects
      on top of the carrier archetype's).
    - HOLD is included next if available.
    - Then one (archetype, agent) per remaining agent.
    - Fill from leftover catalog entries until 3–4 options exist.

    The validator requires 3–4 options and (when a forced moment is active)
    exactly one option with `satisfies_moment` set. The function preserves
    that contract whenever the catalog has enough variety.
    """
    options: list[Option] = []
    used_keys: set[tuple[str, str | None]] = set()

    def _add(opt: Option) -> None:
        key = (opt.archetype, opt.agent)
        if key in used_keys:
            return
        used_keys.add(key)
        options.append(opt)

    # Forced moment FIRST so it's never overwritten by later fill.
    if forced_moment is not None:
        carrier = next(
            ((a, ag, params) for (a, ag, params) in catalog if ag is not None),
            None,
        )
        if carrier is not None:
            a, ag, params = carrier
            _add(
                Option(
                    archetype=a,
                    agent=ag,
                    params=params,
                    title=forced_moment.id,
                    prose=forced_moment.intent.strip()[:240] or f"Honor the {forced_moment.id} moment.",
                    satisfies_moment=forced_moment.id,
                )
            )

    # HOLD next, if available and not already used as a forced-moment carrier.
    for (a, ag, params) in catalog:
        if a != "HOLD":
            continue
        _add(
            Option(
                archetype=a,
                agent=ag,
                params=params,
                title="Hold position",
                prose="Take no action this cycle. Both agents stay put. The situation tightens.",
            )
        )
        break

    # One archetype per remaining agent.
    seen_agents: set[str | None] = {o.agent for o in options}
    for (a, ag, params) in catalog:
        if a == "HOLD" or ag in seen_agents:
            continue
        seen_agents.add(ag)
        _add(
            Option(
                archetype=a,
                agent=ag,
                params=params,
                title=f"Authorize {a} ({ag or 'unbound'})",
                prose=(
                    f"Authorize the {a} archetype on {ag or 'Operations'}. "
                    f"The mechanical cost is documented in the catalog."
                ),
            )
        )
        if len(options) >= 4:
            break

    # Fill from remaining catalog entries until ≥3 (or catalog exhausted).
    if len(options) < 3:
        for (a, ag, params) in catalog:
            _add(
                Option(
                    archetype=a,
                    agent=ag,
                    params=params,
                    title=f"Authorize {a}",
                    prose=f"Authorize the {a} archetype.",
                )
            )
            if len(options) >= 3:
                break

    return options[:4]


async def draft_with_retries(
    drafter: Agent[None, OptionSet],
    *,
    cycle_index: int,
    state: State,
    mission: Mission,
    sop_text: str,
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_moment: Moment | None,
    dispatches: list[DispatchRecord],
    max_retries: int = 3,
) -> tuple[list[Option], bool]:
    """Call the drafter up to `max_retries + 1` times. Each failure feeds the
    validator's complaints back into the next prompt as `retry_feedback`. If
    all attempts fail, return the engine-authored fallback option set and the
    flag `engine_fallback=True`."""
    forced_id = forced_moment.id if forced_moment is not None else None
    last_error: str | None = None
    attempts = 0
    while attempts <= max_retries:
        prompt = _drafter_user_prompt(
            cycle_index, state, mission, sop_text, catalog, forced_moment, dispatches,
            retry_feedback=last_error,
        )
        with logfire.span("drafter_call", attempt=attempts + 1):
            try:
                result = await drafter.run(prompt)
            except Exception as e:
                log.warning("drafter call failed: %s", e)
                last_error = f"agent run failed: {e}"
                attempts += 1
                continue
        options = list(result.output.options)
        try:
            validate_options(options, catalog, forced_id)
            return options, False
        except DrafterValidationError as e:
            last_error = str(e)
            log.info("drafter validation failed (attempt %d): %s", attempts + 1, e)
            attempts += 1
    log.warning(
        "drafter exhausted %d retries; using engine fallback options", max_retries
    )
    return engine_fallback_options(catalog, mission, forced_moment), True
