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
from typing import Any, Sequence

import logfire
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel

from .models import (
    CastMember,
    Event,
    Mission,
    Option,
    OptionSet,
    State,
)
from .spine import (
    ActPhase,
    TensionBand,
    derive_act_phase,
    derive_tension_band,
    invalid_archetype_combinations,
    select_dramatic_question,
)


# Map act phase → renderer tone band (4k). Centralised so the SoP and the
# renderer agree on how each phase reads tonally.
_TONE_BAND: dict[ActPhase, str] = {
    "setup":        "measured — the operation is establishing; cadence is unhurried, observation precedes action",
    "complication": "kinetic — the operation is in motion; sentences shorter, moves are observed and named",
    "climax":       "urgent — decisive cycles; verbs land, hesitation reads as cost",
    "crisis":       "spare — collapse is one mistake away; cut everything that isn't load-bearing",
}


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


_ROLE_LABEL: dict[str, str] = {
    "agent": "agent",
    "handler": "handler (player)",
    "boss": "boss",
    "opposition": "opposition",
    "asset": "asset",
}


def _cast_block(members: list[CastMember]) -> str:
    """Render the cast block. Agent-role entries keep the legacy
    `(file N, codename 'X')` shape so the KV-cache prefix stays stable for
    agent-only missions; non-agent roles carry their role label inline."""
    parts: list[str] = []
    for d in members:
        if d.role == "agent":
            prefix = f"file {d.file_number}, codename {d.codename!r}"
        else:
            file_segment = f"file {d.file_number}, " if d.file_number else ""
            prefix = f"{_ROLE_LABEL[d.role]}; {file_segment}codename {d.codename!r}"
        header = f"  - {d.id} ({prefix}): {d.voice_register}"
        body_lines: list[str] = []
        if d.voice_details.strip():
            body_lines.append(f"    {d.voice_details.strip()}")
        if d.situation.strip():
            situation_indented = "\n".join(
                f"    {ln}" for ln in d.situation.strip().splitlines()
            )
            body_lines.append(f"    Situation:\n{situation_indented}")
        parts.append("\n".join([header, *body_lines]))
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


def _bullet_block(
    header: str,
    lines: Sequence[str],
    *,
    leading_blank: bool = True,
    trailing_blank: bool = False,
) -> str:
    """Render a header line followed by `  - line` bullets. Empty `lines`
    collapses to an empty string — callers don't have to guard against the
    empty case. `leading_blank=True` (default) prepends a newline so the
    block visually separates from preceding prose; `trailing_blank=True`
    appends an extra newline (used by `_what_moved_block` so the block
    sits one blank line above the dramatic question that follows it).
    """
    if not lines:
        return ""
    bullets = "\n".join(f"  - {line}" for line in lines)
    prefix = "\n" if leading_blank else ""
    suffix = "\n\n" if trailing_blank else "\n"
    return f"{prefix}{header}\n{bullets}{suffix}"


def _standing_orders_block(mission: Mission) -> str:
    """Per-mission persistent prose constraints (4e). Injected into all three
    agent system prompts so the renderer/SoP/drafter honour them every cycle.
    Empty when the mission authored none."""
    return _bullet_block(
        "STANDING ORDERS (honour every cycle — do not contradict):",
        list(mission.standing_orders),
    )


def _leans_block(leans: Sequence[str]) -> str:
    """Per-cycle renderer directives collected by `cycle._collect_leans`
    from the cycle's fired events. Injected verbatim — author wrote them."""
    return _bullet_block(
        "LEAN-ON THIS CYCLE (honour these — they tell you what to "
        "foreground in the prose):",
        list(leans),
    )


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
MISSION CAST:
{_cast_block(mission.cast)}
{_standing_orders_block(mission)}
RULES:
- Look forward. Name the threat. Name the resources. Name what has not moved.
- If WHAT MOVED LAST CYCLE is provided, open the memo by naming this cycle's
  pivot — what changed, what didn't — in one sentence, before the dramatic
  question. If no WHAT MOVED block is provided (cycle 1), open on the
  dramatic question directly.
- If a DRAMATIC QUESTION is supplied this cycle, the memo is *about* that
  question — lead with it in prose (not as a header), after the pivot line
  if one was written.
- 1–2 short paragraphs. ~60–110 words total. Cut anything that doesn't
  move the memo forward.
- NEVER use numbers for state. No "cover six", no "six cycles remaining",
  no "stage two", no percentages, no out-of-ten scoring.
- NEVER use game/system vocabulary. No "cover level", no "cover meter",
  no "extraction stage", no "operational security points", no "critical
  failure points", no "mission timer", no "window of N cycles".
- Describe the world. A specific operative in a specific city with a
  specific cover posture against a specific threat. Use SETTING and the
  MISSION CAST dossiers as your grounding.
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
MISSION CAST:
{_cast_block(mission.cast)}
{_standing_orders_block(mission)}
OUTPUT SHAPE (strict):
Each option carries:
  - `archetype`: one of the archetype ids from the VALID CATALOG below.
  - `agent`: the bound agent id when the catalog entry has one, else null.
  - `params`: copy from the catalog entry's params.
  - `title`: ~6–10 words, imperative voice (e.g. "Run eastern exfil — now").
  - `prose`: 1 sentence, ~12–20 words. Read at a glance. The title carries
     the verb; the agent label names who. Prose carries ONLY the cost,
     the risk, or the stake. No restating the verb. No scene-setting.
  - `satisfies_event`: null, unless one option satisfies the FORCED EVENT.

VALIDITY RULES (the engine REJECTS options that violate these):
- `archetype` MUST be present in the VALID CATALOG you receive.
- `agent` MUST match the catalog entry (or be null when entry has no agent).
- If a FORCED EVENT block is present this cycle, exactly ONE option must
  set `satisfies_event` to the event id AND honor its rules.
- 3 or 4 options per cycle.

PROSE RULES:
- Telegraph cost honestly. If an option costs cover, the prose names it
  in the world (e.g. "burns through LYNX's already-degraded cover" — NOT
  "-3 cover" and NOT "drops cover level to four").
- Span tonal range: at least one safe-but-slow option, one fast-but-risky.
- When RECENTLY TRIED is provided, your option set MUST span a different
  tonal range than the prior cycle's: a different bound agent, a different
  cost shape, or a different point on the safe-↔-risky axis. Do not propose
  the same (archetype, agent) pair the player just authorized.
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
MISSION CAST:
{_cast_block(mission.cast)}
{_standing_orders_block(mission)}
DISPATCH RULES:
- 1–2 short paragraphs. ~60–110 words. Cut anything that doesn't carry
  voice, cost, or the closing beat. No padding.
- Voice: as specified for ACTING. First-person for agents, third-party for operations.
- Honor the RESOLVED STATE DELTA. If cover dropped, the prose reflects what cost that.
- Adopt the TONE BAND given each cycle (measured / kinetic / urgent / spare)
  — sentence rhythm must match the current dramatic register.
- End on a single short observational sentence — a closing beat the reader
  carries into the next cycle. No summary, no signoff: a thing seen, a
  thing felt, a thing left unmoved.
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
# `qualities["agent[*].cover"]`. Ordered: first threshold whose value >= cover wins.
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
    override = mission.qualities.get(
        f"agent[{agent_id}].cover"
    ) or mission.qualities.get("agent[*].cover")
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
    a `qualities["mission.cycles_remaining"]` entry (numeric buckets on the
    raw cycles_remaining value). Built-in fallback buckets on the ratio so
    the descriptor scales with mission length."""
    override = mission.qualities.get("mission.cycles_remaining")
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
    """Look up a mission-authored descriptor. Returns None if no entry exists
    for `quality_name` — callers skip the line rather than leaking
    `quality_name: value`."""
    quality = mission.qualities.get(quality_name)
    if quality is None:
        return None
    return quality.describe(value)


# Per-cycle delta type, shared with cycle.py's aggregate computation.
DeltaMap = dict[str, tuple[Any, Any]]


def _agent_id_from_cover_quality(quality: str) -> str | None:
    """Extract `LYNX` from `agent[LYNX].cover`. None for non-cover or
    non-agent-scoped quality refs (e.g. `agent[LYNX].cover_protected` or
    `mission.cycles_remaining`)."""
    if not quality.startswith("agent[") or not quality.endswith(".cover"):
        return None
    closing = quality.find("]")
    if closing < 0:
        return None
    return quality[len("agent["):closing]


def _is_numeric_value(v: Any) -> bool:
    """Numeric, but not bool — `isinstance(True, int)` is True in Python."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def describe_quality_pair(
    quality: str, before: Any, after: Any, mission: Mission
) -> tuple[str | None, str | None]:
    """Return (before_descriptor, after_descriptor) for a delta entry.

    Used by both the SoP "WHAT MOVED LAST CYCLE" block and the SSE
    `delta_descriptors` field. Returns (None, None) for qualities whose
    descriptors don't change (e.g. cover dropped 7→6 within the same
    bucket) or for qualities with no authored / built-in descriptor
    (e.g. one-cycle flags, lost flags).
    """
    if quality == "mission.cycles_remaining":
        before_desc = (
            _describe_cycles(int(before), mission.max_cycles, mission)
            if _is_numeric_value(before)
            else None
        )
        after_desc = (
            _describe_cycles(int(after), mission.max_cycles, mission)
            if _is_numeric_value(after)
            else None
        )
        return before_desc, after_desc

    agent_id = _agent_id_from_cover_quality(quality)
    if agent_id is not None:
        before_desc = (
            _describe_cover(int(before), mission, agent_id)
            if _is_numeric_value(before)
            else None
        )
        after_desc = (
            _describe_cover(int(after), mission, agent_id)
            if _is_numeric_value(after)
            else None
        )
        return before_desc, after_desc

    # Authored mission qualities (mission.extraction_stage etc.) — only those
    # with a `qualities:` entry get descriptors. Skip transient agent flags
    # (`cover_protected`, `lost`) which have no descriptor by design.
    before_desc = (
        _describe_authored_quality(quality, before, mission)
        if before is not None
        else None
    )
    after_desc = (
        _describe_authored_quality(quality, after, mission)
        if after is not None
        else None
    )
    return before_desc, after_desc


def describe_delta(delta: DeltaMap, mission: Mission) -> dict[str, dict[str, str | None]]:
    """Render a delta as `{quality: {before, after, agent_id}}` narrative
    descriptors. `agent_id` is the operative whose cover changed (e.g.
    "LYNX") or None for mission-scoped qualities — the client uses it to
    prefix the RESULT line without re-parsing the quality ref.

    Drops entries whose descriptors are absent or unchanged — the result is
    only the moves with a *narrative* signal, suitable for both the SoP
    "what moved last cycle" prompt block and the client-facing RESULT
    footer (via SSE).
    """
    out: dict[str, dict[str, str | None]] = {}
    for quality, (before, after) in delta.items():
        before_desc, after_desc = describe_quality_pair(quality, before, after, mission)
        if before_desc is None and after_desc is None:
            continue
        if before_desc == after_desc:
            continue
        out[quality] = {
            "before": before_desc,
            "after": after_desc,
            "agent_id": _agent_id_from_cover_quality(quality),
        }
    return out


def _what_moved_block(
    last_cycle_delta: DeltaMap | None, mission: Mission
) -> str:
    """Render the SoP user-prompt block summarising what changed in the
    previous cycle. Empty when no prior cycle exists, or when nothing in
    the delta produced a narrative-level change.
    """
    if not last_cycle_delta:
        return ""
    descriptors = describe_delta(last_cycle_delta, mission)
    if not descriptors:
        return ""
    # Order: cover changes first (most player-visible), then extraction
    # progress, then operation clock. Within each, mission-authored order.
    cover_lines, mission_lines, clock_lines = [], [], []
    for q, d in descriptors.items():
        if d["agent_id"] is not None:
            cover_lines.append(f"{d['agent_id']}: {d['before']} → {d['after']}")
        elif q == "mission.cycles_remaining":
            clock_lines.append(str(d["after"]))
        else:
            mission_lines.append(str(d["after"]))
    return _bullet_block(
        "WHAT MOVED LAST CYCLE (open the memo by naming this pivot in one "
        "sentence, before the dramatic question):",
        [*cover_lines, *mission_lines, *clock_lines],
        leading_blank=False,
        trailing_blank=True,
    )


def _narrative_state_block(state: State, mission: Mission) -> str:
    """In-fiction state summary for the SoP and renderer. No raw numerics, no
    raw quality ids, no fired-event ledger noise."""
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
    cycle_index: int,
    state: State,
    mission: Mission,
    dispatches: list[DispatchRecord],
    *,
    last_cycle_delta: DeltaMap | None = None,
) -> str:
    # Intentionally NO "cycle N of M" opener — that header was the most
    # reliable place the model picked up "six cycles remaining" / "five
    # cycles remaining" as the lede. Time pressure is now carried by the
    # OPERATION CLOCK descriptor in the narrative state block.
    act_phase = derive_act_phase(state, mission)
    dramatic_question = select_dramatic_question(state, mission)
    dq_block = (
        f"DRAMATIC QUESTION (lead the memo with this — in prose):\n  {dramatic_question}\n\n"
        if dramatic_question is not None
        else ""
    )
    what_moved = _what_moved_block(last_cycle_delta, mission)
    return f"""\
State of Play memo. Field reports follow.

ACT PHASE: {act_phase}

{what_moved}{dq_block}CURRENT STATE:
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


def _forced_event_block(event: Event | None) -> str:
    """Render the FORCED EVENT block for the drafter prompt — the authored
    `Event` with `force_into_drafter=True` whose `when:` predicate fires this
    cycle.
    """
    if event is None:
        return "(no forced event this cycle.)"
    return f"""\
A FORCED EVENT must appear among this cycle's options:
  event_id: {event.id}
  intent: {event.intent.strip()}
  must_communicate:
{chr(10).join(f"    - {x}" for x in event.must_communicate)}
  must_not:
{chr(10).join(f"    - {x}" for x in event.must_not)}
  tone: {event.tone}
EXACTLY ONE option this cycle must set `satisfies_event` = "{event.id}"
and write prose that honors the rules above. The mechanical effect of the
event is engine-applied; you only write the prose situated in current state.
"""


def _locked_block(
    state: State, mission: Mission
) -> str:
    """Per-cycle list of archetypes that are NOT in the catalog right now,
    each annotated with which precondition leaf failed. Helps the drafter
    reason about *why* an option is unavailable instead of seeing only the
    trimmed catalog (4i)."""
    lines = [
        f"{arch_id}({agent_id}) — {reason}" if agent_id else f"{arch_id} — {reason}"
        for arch_id, agent_id, reason in invalid_archetype_combinations(state, mission)
    ]
    return _bullet_block(
        "LOCKED THIS CYCLE (use as context for option framing — do NOT propose these):",
        lines,
        leading_blank=False,
    )


def _recently_tried_block(
    recently_tried: Sequence[tuple[int, str, str | None]] | None,
) -> str:
    """Render the RECENTLY TRIED block: the last 1-2 cycles' authorized
    (cycle_index, archetype, agent_id_or_None) tuples. Empty for cycle 1.
    Drafter is told to span a different tonal range from these picks."""
    lines = [
        f"C{cycle_idx}: {archetype}({agent_id})" if agent_id else f"C{cycle_idx}: {archetype}"
        for cycle_idx, archetype, agent_id in (recently_tried or [])
    ]
    return _bullet_block(
        "RECENTLY TRIED (vary tonal range from these — different agent, "
        "different cost shape, or different point on the safe-↔-risky "
        "axis; do not propose the same (archetype, agent) pair the player "
        "just authorized):",
        lines,
    )


def _drafter_user_prompt(
    cycle_index: int,
    state: State,
    mission: Mission,
    sop_text: str,
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_event: Event | None,
    dispatches: list[DispatchRecord],
    *,
    retry_feedback: str | None = None,
    recently_tried: Sequence[tuple[int, str, str | None]] | None = None,
) -> str:
    archetype_intents = "\n".join(
        f"  - {a.id}: {a.intent}" for a in mission.archetypes
    )
    retry = ""
    if retry_feedback:
        retry = f"\nPREVIOUS ATTEMPT WAS REJECTED:\n{retry_feedback}\n"
    act_phase = derive_act_phase(state, mission)
    return f"""\
Draft 3 or 4 options now.

ACT PHASE: {act_phase}

CURRENT STATE:
{_narrative_state_block(state, mission)}

RECENT DISPATCHES (last 2):
{_dispatches_block(dispatches, limit=2)}

STATE OF PLAY MEMO (this cycle, just rendered):
{sop_text.strip()}

VALID CATALOG (engine-computed; you MUST pick from this exact set):
{_catalog_block(catalog)}

{_locked_block(state, mission)}ARCHETYPE INTENTS (for prose framing — these are NOT the player-facing prose):
{archetype_intents}
{_recently_tried_block(recently_tried)}
{_forced_event_block(forced_event)}{retry}
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
    *,
    prior_state: State | None = None,
    leans: list[str] = (),
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

    act_phase = derive_act_phase(state, mission)
    tone_band = _TONE_BAND[act_phase]
    tension_band = derive_tension_band(prior_state, state)

    return f"""\
Cycle {cycle_index} of {mission.max_cycles}. Render the FIELD DISPATCH.

ACT PHASE: {act_phase}
TONE BAND: {tone_band}
TENSION (this cycle's cover swing): {tension_band}

{voice_note}
{_leans_block(leans)}
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
    forced_event_id: str | None,
) -> None:
    """Raise `DrafterValidationError` with a corrective message if the option
    set doesn't satisfy the constraints. Used by `draft_with_retries` to
    decide whether to retry. `forced_event_id` (if set) is the id of the
    forced event that must appear as `Option.satisfies_event` on exactly one
    option.
    """
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
    if forced_event_id is not None:
        satisfiers = [o for o in options if o.satisfies_event == forced_event_id]
        if len(satisfiers) != 1:
            problems.append(
                f"forced event {forced_event_id!r}: expected exactly 1 option "
                f"with satisfies_event={forced_event_id!r}, found {len(satisfiers)}"
            )
    if problems:
        raise DrafterValidationError("; ".join(problems))


def engine_fallback_options(
    catalog: list[tuple[str, str | None, dict[str, str]]],
    mission: Mission,
    forced_event: Event | None,
) -> list[Option]:
    """Engine-authored fallback when the drafter exhausts retries. Picks
    deterministic options from the valid catalog with terse generic prose.

    Strategy:
    - If a forced event is required, the first option satisfies it (using
      any agent-bound catalog entry as carrier — forced events apply their
      effects on top of the carrier archetype's).
    - HOLD is included next if available.
    - Then one (archetype, agent) per remaining agent.
    - Fill from leftover catalog entries until 3–4 options exist.

    The validator requires 3–4 options and (when a forced event is active)
    exactly one option with `satisfies_event` set. The function preserves
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

    # Forced event FIRST so it's never overwritten by later fill.
    if forced_event is not None:
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
                    title=forced_event.id,
                    prose=forced_event.intent.strip()[:240] or f"Honor the {forced_event.id} event.",
                    satisfies_event=forced_event.id,
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
    forced_event: Event | None,
    dispatches: list[DispatchRecord],
    max_retries: int = 3,
    recently_tried: Sequence[tuple[int, str, str | None]] | None = None,
) -> tuple[list[Option], bool]:
    """Call the drafter up to `max_retries + 1` times. Each failure feeds the
    validator's complaints back into the next prompt as `retry_feedback`. If
    all attempts fail, return the engine-authored fallback option set and the
    flag `engine_fallback=True`."""
    forced_id = forced_event.id if forced_event is not None else None
    last_error: str | None = None
    attempts = 0
    while attempts <= max_retries:
        prompt = _drafter_user_prompt(
            cycle_index, state, mission, sop_text, catalog, forced_event, dispatches,
            retry_feedback=last_error,
            recently_tried=recently_tried,
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
    return engine_fallback_options(catalog, mission, forced_event), True
