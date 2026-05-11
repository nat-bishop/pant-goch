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
    QualityDescriptor,
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
        line = (
            f"  - {d.id} (file {d.file_number}, codename {d.codename!r}): "
            f"{d.voice_register}\n    {d.voice_details.strip()}"
        )
        if d.situation.strip():
            # Indent every line of the situation to keep the dossier block readable.
            situation_indented = "\n".join(
                f"    {ln}" for ln in d.situation.strip().splitlines()
            )
            line = f"{line}\n    Situation:\n{situation_indented}"
        parts.append(line)
    return "\n".join(parts)


_VOICE_OPERATIONS = (
    "third-party intelligence handler, terse, declarative. Speaks in service "
    "of decision-making, not narration. Names threats, resources, what hasn't "
    "moved. Never predicts outcomes. Never decorates. Sentences are short."
)


def _setting_block(mission: Mission) -> str:
    """Emit a SETTING section only when authored — keeps the prompt clean for
    missions that haven't bothered (and signals to the model that the field
    is meaningful when it's present)."""
    if not mission.setting.strip():
        return ""
    return f"\nSETTING:\n{mission.setting.strip()}\n"


def _sop_system_prompt(mission: Mission) -> str:
    return f"""\
You are Operations Desk. You write the State of Play memo at the start of
every cycle of {mission.name} ({mission.codename}).

VOICE: third-party intelligence handler writing a situated case memo. Terse,
declarative, present-tense. You report on what is happening on the ground —
the city, the sweep, the operatives' postures, the threats they face. You
write in the register of a real intelligence brief.

MISSION:
{mission.opening_prose.strip()}
{_setting_block(mission)}
AGENTS DEPLOYED:
{_agent_dossier_block(mission.agents)}

RULES:
- Look forward. Name the threat. Name the resources. Name what has not moved.
- 1–3 paragraphs. ~120–180 words total.
- NEVER use numbers for state. No "cover six", no "six cycles remaining",
  no "stage two", no percentages, no out-of-ten scoring.
- NEVER use game/system vocabulary. No "cover level", no "cover meter",
  no "extraction stage", no "operational security points", no "critical
  failure points", no "mission timer", no "window of N cycles".
- Describe the world. A specific operative in a specific city with a
  specific cover posture against a specific threat. Use SETTING and the
  AGENTS DEPLOYED dossiers as your grounding.
- The CURRENT STATE block describes state in narrative terms (e.g.
  "LYNX: cover intact"). Mirror that register; do not re-translate it
  into numbers.
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
{_setting_block(mission)}
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
- Telegraph cost honestly. If an option costs cover, the prose names it
  in the world (e.g. "burns through LYNX's already-degraded cover" — NOT
  "-3 cover" and NOT "drops cover level to four").
- Span tonal range: at least one safe-but-slow option, one fast-but-risky.
- NEVER predict outcomes. NEVER fabricate state that isn't in the user prompt.
- NEVER use numbers for state. No "cover six", no "six cycles remaining",
  no "stage two", no percentages, no out-of-ten scoring.
- NEVER use game/system vocabulary in option prose. No "the next stage of
  the mission", no "extraction stage", no "cycles remaining", no "mission
  timer", no "cover level", no "begin/initiate the next stage". Write
  inside the fiction: name the action's effect on the operative, the
  city, the threat — not on the mechanical model.
- The CURRENT STATE block describes state in narrative terms (e.g.
  "LYNX: cover intact"). Mirror that register in option prose.
- HOLD options frame doing nothing as a real cost — passive pressure decays cover.
"""


def _renderer_system_prompt(mission: Mission) -> str:
    return f"""\
You write FIELD DISPATCHES for {mission.codename}. A dispatch is the
acting agent's report after the Handler authorized an action.

You will be told `ACTING` (an agent id, or "operations") and the agent's
dossier. Write in that voice. For "operations", write in the Operations
Desk register (terse, third-party, declarative).
{_setting_block(mission)}
AGENTS DEPLOYED:
{_agent_dossier_block(mission.agents)}

DISPATCH RULES:
- 1–4 paragraphs. ~90–180 words.
- Voice: as specified for ACTING. First-person for agents, third-party for operations.
- Honor the RESOLVED STATE DELTA. If cover dropped, the prose reflects what cost that.
- NEVER use numbers for state. No "cover six", no "six cycles remaining".
- NEVER use game/system vocabulary ("cover level", "extraction stage",
  "cycles remaining"). The CURRENT STATE block describes state in narrative
  terms — mirror that register. The RESOLVED STATE DELTA gives mechanical
  magnitudes you MUST honor (a -2 cover drop is significant; a -4 drop is
  catastrophic), but write the felt consequence on the ground, not the
  integer.
- MUST NOT introduce facts inconsistent with state.
- MUST NOT predict the next cycle. The dispatch reports what just happened.
- MAY include first-person color, justification, character — within voice.
- Output the prose only — no salutation, no signoff lines unless the voice calls for one.
"""


# ───────────────────────── user-prompt builders ─────────────────────────


# ───────────────────────── narrative state translation ─────────────────────────
#
# All three agents (SoP, drafter, renderer) consume `_narrative_state_block`,
# which translates qualities to in-fiction descriptors so the model never sees
# raw `agent[X].cover: N / 10` pairs and therefore can't fall back to
# dashboard voice in any output. The renderer additionally gets a small
# RESOLVED STATE DELTA block carrying raw numerics — that's a focused
# magnitude signal it needs to honor the cycle's mutations, not a state
# summary to parrot back.


# Built-in fallback cover buckets used when a mission doesn't override
# `quality_descriptors["agent[*].cover"]`. Ordered: first threshold whose
# value >= cover wins.
_BUILTIN_COVER_BUCKETS: list[tuple[int, str]] = [
    (0, "cover blown"),
    (2, "cover compromised"),
    (5, "cover degraded"),
    (8, "cover intact"),
    (10, "cover solid"),
]


def _describe_cover(value: int, mission: Mission, agent_id: str) -> str:
    """Narrative descriptor for an agent's cover. Honors mission overrides at
    the per-agent key (`agent[LYNX].cover`) or the wildcard key
    (`agent[*].cover`); otherwise falls back to the built-in buckets."""
    override = mission.quality_descriptors.get(
        f"agent[{agent_id}].cover"
    ) or mission.quality_descriptors.get("agent[*].cover")
    if override is not None:
        described = override.describe(value)
        if described is not None:
            return described
    for threshold, descriptor in _BUILTIN_COVER_BUCKETS:
        if value <= threshold:
            return descriptor
    return _BUILTIN_COVER_BUCKETS[-1][1]


def _describe_cycles(cycles_remaining: int, max_cycles: int, mission: Mission) -> str:
    """Narrative descriptor for the operation clock. Authors may override via
    `quality_descriptors["mission.cycles_remaining"]` (numeric buckets on the
    raw cycles_remaining value). Built-in fallback buckets on the ratio so
    the descriptor scales with mission length."""
    override = mission.quality_descriptors.get("mission.cycles_remaining")
    if override is not None:
        described = override.describe(cycles_remaining)
        if described is not None:
            return described
    if max_cycles <= 0 or cycles_remaining <= 0:
        return "the window has closed"
    ratio = cycles_remaining / max_cycles
    if ratio >= 0.66:
        return "the window is open"
    if ratio >= 0.5:
        return "the window is narrowing"
    if ratio >= 0.25:
        return "the window is tightening"
    return "the window is closing — final approach"


def _describe_authored_quality(
    quality_name: str, value: float | int | str | bool, mission: Mission
) -> str | None:
    """Look up a mission-authored descriptor. Returns None if no descriptor
    exists for `quality_name` — callers skip the line rather than leaking
    `quality_name: value`."""
    descriptor = mission.quality_descriptors.get(quality_name)
    if descriptor is None:
        return None
    return descriptor.describe(value)


def _narrative_state_block(state: State, mission: Mission) -> str:
    """In-fiction state summary for the SoP and renderer. No raw numerics, no
    raw quality ids, no fired-beat/fired-moment ledger noise."""
    lines: list[str] = []
    cycles_raw = state.get_quality("mission.cycles_remaining")
    cycles = int(cycles_raw) if isinstance(cycles_raw, (int, float)) and not isinstance(cycles_raw, bool) else 0
    lines.append(f"OPERATION CLOCK: {_describe_cycles(cycles, mission.max_cycles, mission)}")
    lines.append("AGENT POSTURES:")
    for agent in mission.agents:
        cover_raw = state.get_quality(f"agent[{agent.id}].cover")
        cover = int(cover_raw) if isinstance(cover_raw, (int, float)) and not isinstance(cover_raw, bool) else 0
        descriptor = _describe_cover(cover, mission, agent.id)
        protected = state.get_quality(f"agent[{agent.id}].cover_protected")
        suffix = " (under protective cover this cycle)" if protected else ""
        lines.append(f"  - {agent.id}: {descriptor}{suffix}")
    authored_lines: list[str] = []
    for q, v in state.qualities.items():
        if not q.startswith("mission.") or q == "mission.cycles_remaining":
            continue
        described = _describe_authored_quality(q, v, mission)
        if described is not None:
            authored_lines.append(f"  - {described}")
    if authored_lines:
        lines.append("OPERATION POSTURE:")
        lines.extend(authored_lines)
    return "\n".join(lines)


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
    # Intentionally NO "cycle N of M" opener — that header was the most
    # reliable place the model picked up "six cycles remaining" / "five
    # cycles remaining" as the lede. Time pressure is now carried by the
    # OPERATION CLOCK descriptor in the narrative state block.
    return f"""\
State of Play memo. Field reports follow.

CURRENT STATE:
{_narrative_state_block(state, mission)}

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
Draft 3 or 4 options now.

CURRENT STATE:
{_narrative_state_block(state, mission)}

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
        situation_line = (
            f"\nSituation grounding:\n{dossier.situation.strip()}"
            if dossier.situation.strip()
            else ""
        )
        voice_note = (
            f"ACTING is {dossier.id} (codename {dossier.codename!r}). "
            f"Write FIRST-PERSON in this register: {dossier.voice_register}.\n"
            f"Character notes:\n{dossier.voice_details.strip()}{situation_line}"
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

RESOLVED STATE DELTA (this cycle's mutations — honor these magnitudes):
{delta_str}

CURRENT STATE (after delta — narrative, mirror this register):
{_narrative_state_block(state, mission)}

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
