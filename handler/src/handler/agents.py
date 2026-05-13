"""LLM agents — three agents per session, split for schema simplicity.

- `SELECTOR_AGENT` — picks N catalog targets + forced-event carrier per
  cycle. Output schema is a per-cycle `Literal`-typed list of target tokens
  (grammar-sampler enforces catalog validity at decode time). **Stateless
  one-shot.**
- `PROSE_WRITER_AGENT` — writes `(title, prose)` pairs for the slots the
  selector pinned. Output schema is just a positional list of two-string
  pairs — no catalog identifiers leak into this call's schema. **Stateless
  one-shot.**
- `LOG_WRITER_AGENT` — appends the cycle's case-file entry after effects
  apply, Operations voice (str output, streamed). **Conversation-based** —
  one pydantic-ai agent per session whose `message_history` accumulates
  one user+assistant pair per cycle. The conversation IS the case file.

Why split selector + prose: Gemma E4B struggles when one structured emission
has to satisfy enum-class constraints (`archetype` ∈ current catalog) AND
open-ended prose constraints (~6–10 word title, ~12–20 word prose, voice
rules, no leaked game vocabulary) at once. The split narrows each call's
schema to a focused job — the selector emits ≤2 fields per slot, the prose
writer emits exactly 2 string fields per slot.

Prompt structure (KV-cache friendly):
  Selector:     [stable system prompt] + [dynamic user prompt + dynamic schema]
  Prose writer: [stable system prompt] + [dynamic user prompt + dynamic schema]
  Log writer:   [stable system prompt] + [accumulating message_history]

System prompts are baked at `build_agent_bundle(model, mission)`. User
prompts and output schemas are built per call. The selector + prose writer
use pydantic-ai's runtime `output_type=` override on `.run()` to swap in a
per-cycle schema — see `_make_selector_output_type` / `_make_prose_output_type`.

This module has NO direct LLM call sites. `cycle.py` wraps the agent runs
into the drafter/log-writer protocols; `main.py` wires up the paced
word-stream emitter and the persistent log message-history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import logfire
from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel

from .models import (
    CastMember,
    Event,
    Mission,
    Option,
    OptionProse,
    State,
)
from .spine import (
    ActPhase,
    derive_act_phase,
    derive_tension_band,
    invalid_archetype_combinations,
)


# Map act phase → tone band for the log writer. Centralised so the log
# writer can adopt the cycle's dramatic register without per-prompt churn.
_TONE_BAND: dict[ActPhase, str] = {
    "setup":        "measured — the operation is establishing; cadence is unhurried, observation precedes action",
    "complication": "kinetic — the operation is in motion; sentences shorter, moves are observed and named",
    "climax":       "urgent — decisive days; verbs land, hesitation reads as cost",
    "crisis":       "spare — collapse is one mistake away; cut everything that isn't load-bearing",
}


log = logging.getLogger(__name__)


# ───────────────────────── log entries (used in prompts + state) ─────────────────────────


@dataclass
class LogEntry:
    """One past log entry — the full prose the log writer produced for a
    cycle. Held in session memory so the drafter prompt can carry the last
    2 entries forward as user-side context. The log writer itself reads its
    history directly via pydantic-ai's `message_history`, not via this list.
    """

    cycle_index: int
    archetype: str
    text: str


# ───────────────────────── bundle ─────────────────────────


@dataclass
class AgentBundle:
    """One mission's three agents. Built once at lifespan boot via
    `build_agent_bundle(model, mission)`. The selector and prose writer are
    stateless; the log writer is the *agent definition*, but each session
    carries its own accumulating `message_history` on
    `SessionRuntime.log_message_history`.

    The selector and prose writer use `output_type=str` at construction —
    actual per-cycle Literal-typed schemas are passed via `agent.run(...,
    output_type=NativeOutput(BuiltType, strict=True))`. The pydantic-ai
    runtime-override pattern; the construction-time type is a placeholder.
    """

    selector: Agent[None, Any]
    prose_writer: Agent[None, Any]
    log_writer: Agent[None, str]
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
    empty case."""
    if not lines:
        return ""
    bullets = "\n".join(f"  - {line}" for line in lines)
    prefix = "\n" if leading_blank else ""
    suffix = "\n\n" if trailing_blank else "\n"
    return f"{prefix}{header}\n{bullets}{suffix}"


def _standing_orders_block(mission: Mission) -> str:
    """Per-mission persistent prose constraints (4e). Injected into both
    agent system prompts so the log writer and drafter honour them every
    cycle. Empty when the mission authored none."""
    return _bullet_block(
        "STANDING ORDERS (honour every day — do not contradict):",
        list(mission.standing_orders),
    )


def _leans_block(leans: Sequence[str]) -> str:
    """Per-cycle log-writer directives collected by `cycle._collect_leans`
    from the cycle's fired events. Injected verbatim — author wrote them."""
    return _bullet_block(
        "LEAN-ON THIS DAY (honour these — they tell you what to "
        "foreground in the prose):",
        list(leans),
    )


def _selector_system_prompt(mission: Mission) -> str:
    """Selector agent: picks 3–4 catalog targets for this day and (when a
    forced event is due) flags exactly one carrier. NO prose written here —
    that's the prose writer's job. Schema-side: `target` is a per-cycle
    Literal of `archetype` or `archetype:agent` strings, so the grammar
    sampler rejects invalid catalog picks at decode time.
    """
    return f"""\
You are Operations Desk selecting candidate courses of action for {mission.codename}.
Each day you pick 3–4 TARGETS the Handler will choose between. You do NOT write
prose — a separate pass will. Your job is the slate, not the language.

VOICE CONTEXT (for judgment): {_VOICE_OPERATIONS}

MISSION:
{mission.opening_prose.strip()}
{_setting_block(mission)}
MISSION CAST:
{_cast_block(mission.cast)}
{_standing_orders_block(mission)}
OUTPUT:
A list of `selections`, each with:
  - `target`: one of the catalog tokens given in the user prompt
    (encoded `ARCHETYPE` for unbound, `ARCHETYPE:AGENT` for agent-bound).
  - `satisfies_event` (only when a FORCED EVENT block is present): the
    event's id on EXACTLY ONE selection; null on the others.

SELECTION RULES:
- Pick 3 or 4 targets — the user prompt names N for this day.
- No duplicate targets within the day.
- Span tonal range: include at least one safe-but-slow target and at least
  one fast-but-risky target when the catalog supports it.
- When RECENTLY TRIED is provided, vary tonal range from the prior day —
  different agent, different cost shape, or a different point on the
  safe-↔-risky axis. Do not pick the same target the player just authorized.
- HOLD is doing nothing — include it when passive pressure is the meaningful
  contrast.
- If a FORCED EVENT block appears, exactly ONE selection must carry it
  (`satisfies_event` = event id). Pick the carrier whose archetype + agent
  best fits the event's intent — the prose pass will honor the event's
  prose constraints on whichever slot you flag.

You will not see numeric state. The user prompt translates state into
narrative descriptors (cover intact / degraded / compromised / blown,
window open / narrowing / closing). Pick targets consistent with that
posture.
"""


def _prose_writer_system_prompt(mission: Mission) -> str:
    """Prose writer agent: takes the engine-pinned `(archetype, agent,
    satisfies_event)` slate from the selector and writes `(title, prose)`
    per slot. No catalog identifiers in this agent's output — those are
    already fixed."""
    return f"""\
You are Operations Desk writing player-facing prose for {mission.codename}.
You receive a list of SLOTS the selector has already pinned (archetype +
optional agent + optional forced-event flag). Your job is to write one
`title` and one `prose` line per slot, in order, matching the slot's
operational shape.

VOICE: {_VOICE_OPERATIONS}

MISSION:
{mission.opening_prose.strip()}
{_setting_block(mission)}
MISSION CAST:
{_cast_block(mission.cast)}
{_standing_orders_block(mission)}
PER-SLOT OUTPUT SHAPE (strict):
  - `title`: ~6–10 words, imperative voice (e.g. "Run eastern exfil — now").
  - `prose`: 1 sentence, ~12–20 words. Read at a glance. The title carries
     the verb; the agent label names who. Prose carries ONLY the cost,
     the risk, or the stake. No restating the verb. No scene-setting.

PROSE RULES:
- Telegraph cost honestly. If a slot's archetype costs cover, the prose
  names it in the world (e.g. "burns through LYNX's already-degraded
  cover" — NOT "-3 cover" and NOT "drops cover level to four").
- NEVER predict outcomes. NEVER fabricate state that isn't in the user prompt.
- NEVER use numbers for state. No "cover six", no "six days remaining",
  no "stage two", no percentages, no out-of-ten scoring.
- NEVER use game/system vocabulary in option prose. No "the next stage of
  the mission", no "extraction stage", no "days remaining", no "mission
  timer", no "cover level", no "begin/initiate the next stage". Write
  inside the fiction: name the action's effect on the operative, the
  city, the threat — not on the mechanical model.
- The CURRENT STATE block describes state in narrative terms (e.g.
  "LYNX: cover intact"). Mirror that register in prose.
- HOLD slots frame doing nothing as a real cost — passive pressure decays cover.
- Vary the four prose lines from each other — different angle, different
  cost language, no echoing the same phrase across slots.

OPPOSED ARCHETYPES (when an ARCHETYPE INTENT explicitly says the option prose
MUST name a gamble in-fiction): the slot's outcome depends on state the
player cannot fully see. Describe what is at risk — name the gamble — but
DO NOT predict which way it lands. Example phrasing: "if MUNTEANU is closer
than the file suggests, this puts her in his hand."

FORCED EVENT (when supplied): one slot will be flagged as the forced-event
carrier in the user prompt, together with `intent`, `must_communicate`,
`must_not`, and `tone`. Write that slot's prose to honor those constraints.
The mechanical effect is engine-applied; you only write the prose situated
in current state.

BEAT THIS DAY (when supplied): the user prompt may carry one beat fragment
— an authored observation about the day's specific situation. Treat it as
framing color: at least one slot's prose should hold the emotional or
operational weight the beat names. Do not surface the beat's phrasing
verbatim; let it inform the cost / risk language naturally.

HIDDEN STATE HINTS (when supplied): the user prompt may carry observational
hints about state the player cannot see directly. Treat them as ground
truth — your prose should reflect the implied pressure — but do NOT name
the hidden state by its game term, and do not quote the hint verbatim.
Translate through cost / risk.
"""


def _log_writer_system_prompt(mission: Mission) -> str:
    return f"""\
You are Operations Desk. You write the case-file log for {mission.codename}.
Every day, after the Handler authorizes an option and the engine resolves
its effects, you append one entry to the file.

VOICE: third-party intelligence handler writing a situated case record.
Terse, declarative, present-tense. You report on what is happening on the
ground — the city, the investigation, the operatives' postures, the threats
they face. You write in the register of a real intelligence brief.

You are writing into an ongoing case file. Your previous entries appear in
your conversation history. Do not re-cover earlier days' ground; the
reader has them. Build on what came before.

Your prior entries describe what was true THEN. The current state block
in each day's user message is what is true NOW. Reflect the present.
Do not defend past framings against present state — when state has moved,
the new entry names the move.

MISSION:
{mission.opening_prose.strip()}
{_setting_block(mission)}
MISSION CAST:
{_cast_block(mission.cast)}
{_standing_orders_block(mission)}
ENTRY SHAPE (every day):
Two short paragraphs. ~80–140 words total.
  - ¶1 — the ANGLE: what was authorized, the operational read on why it
    matters, the constraints in play. Frame the move, not the verb.
  - ¶2 — the RESOLUTION: what landed in the world, the cost paid, a
    single short observational closing beat. End on a thing seen, a
    thing felt, a thing left unmoved. No summary, no signoff.

RULES:
- Honor the RESOLVED STATE DELTA. If cover dropped 3, the prose reflects
  what cost that. If a forced event fired, the entry names that consequence.
- Adopt the TONE BAND given each day (measured / kinetic / urgent / spare)
  — sentence rhythm must match the current dramatic register.
- NEVER use numbers for state. No "cover six", no "six days remaining",
  no "stage two", no percentages, no out-of-ten scoring.
- NEVER use game/system vocabulary ("cover level", "extraction stage",
  "days remaining"). The CURRENT STATE block describes state in narrative
  terms — mirror that register. The RESOLVED STATE DELTA gives mechanical
  magnitudes you MUST honor (a -2 cover drop is significant; a -4 drop is
  catastrophic), but write the felt consequence on the ground, not the
  integer.
- MUST NOT introduce facts inconsistent with state.
- MUST NOT predict the next day. The entry reports what just happened.
- NEVER propose options (that is the drafter's job).
- Output the prose only — no headings, no labels, no preamble.
"""


# ───────────────────────── user-prompt builders ─────────────────────────


# ───────────────────────── narrative state translation ─────────────────────────
#
# Both agents (drafter, log writer) consume `_narrative_state_block`, which
# translates qualities to in-fiction descriptors so the model never sees raw
# `agent[X].cover: N / 10` pairs and therefore can't fall back to dashboard
# voice in any output. The log writer additionally gets a small RESOLVED
# STATE DELTA block carrying raw numerics — that's a focused magnitude
# signal it needs to honor the cycle's mutations, not a state summary to
# parrot back.


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
    """Look up a mission-authored descriptor. Returns None when:
      - the quality has no `Quality` entry (transient flags, unauthored runtime state),
      - the quality is marked `visible=False` (surfaces only via the hidden-state
        hint block — never via narrative state or RESULT footer),
      - or `Quality.describe` returns None (value outside authored bands).
    """
    quality = mission.qualities.get(quality_name)
    if quality is None or not quality.visible:
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

    Used by the SSE `delta_descriptors` field. Returns (None, None) for
    qualities whose descriptors don't change (e.g. cover dropped 7→6 within
    the same bucket) or for qualities with no authored / built-in descriptor
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

    Drops entries whose descriptors are absent or unchanged. Also drops
    deltas for qualities marked `visible=False` so hidden state never
    surfaces in the client-facing RESULT footer.
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


def _narrative_state_block(state: State, mission: Mission) -> str:
    """In-fiction state summary for the drafter and log writer. No raw
    numerics, no raw quality ids, no fired-event ledger noise.

    Qualities with `visible=False` are SKIPPED from both AGENT POSTURES and
    OPERATION POSTURE. They surface only through `_hidden_state_hint_block`
    so the player gets observational hints instead of explicit state lines.
    """
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
        suffix = " (under protective cover this day)" if protected else ""
        lines.append(f"  - {agent.id}: {descriptor}{suffix}")
    authored_lines: list[str] = []
    for q, v in state.qualities.items():
        if not q.startswith("mission.") or q == "mission.cycles_remaining":
            continue
        # `_describe_authored_quality` returns None for unauthored qualities
        # AND for `visible=False` ones — hidden state surfaces only through
        # `_hidden_state_hint_block`.
        described = _describe_authored_quality(q, v, mission)
        if described is not None:
            authored_lines.append(f"  - {described}")
    if authored_lines:
        lines.append("OPERATION POSTURE:")
        lines.extend(authored_lines)
    return "\n".join(lines)


def _hidden_state_hint_block(
    state: State, mission: Mission, cycle_index: int
) -> str:
    """Render the HIDDEN STATE TO HINT AT block for the drafter / log-writer
    user prompts. For each `Quality` authored with `visible=False` and
    `bands:`, pick the rotated variant for the current value + cycle index.

    The system-prompt directive tells the LLM to translate the variant
    through the cycle's situation, never quote it verbatim. Authoring
    discipline: variants are *examples* the model paraphrases, not lines.
    """
    lines: list[str] = []
    for q_name, q_def in mission.qualities.items():
        if q_def.visible or not q_def.bands:
            continue
        value = state.get_quality(q_name)
        if value is None:
            continue
        variant = q_def.describe_variant(value, cycle_index)
        if variant is not None:
            lines.append(variant)
    return _bullet_block(
        "HIDDEN STATE TO HINT AT (surface this through one observation; "
        "never name the underlying state directly, never assign it a number "
        "or label; the reader must infer):",
        lines,
    )


def _log_entries_block(entries: list[LogEntry], limit: int) -> str:
    """Render the recent log-entries block for the drafter's user prompt.
    The log writer reads its history from pydantic-ai's message_history —
    this block exists for the drafter, which is stateless one-shot."""
    if not entries:
        return "(no prior days — this is day 1.)"
    chunk = entries[-limit:]
    parts: list[str] = []
    for d in chunk:
        parts.append(f"--- Day {d.cycle_index} — {d.archetype}")
        parts.append(d.text.strip())
        parts.append("")
    return "\n".join(parts).rstrip()


def _target_string(archetype: str, agent: str | None) -> str:
    """Encode a `(archetype, agent)` catalog entry as a single Literal token.
    `HOLD` for unbound, `INITIATE_EXFIL:LYNX` for agent-bound. Matches the
    grammar produced by `_make_selector_output_type`'s `Literal` over the
    cycle's catalog — the selector emits these tokens and the engine parses
    them back via `_parse_target`."""
    return f"{archetype}:{agent}" if agent else archetype


def _parse_target(token: str) -> tuple[str, str | None]:
    """Inverse of `_target_string`. Splits on the first `:`; archetypes
    themselves never contain `:`, so this is unambiguous."""
    if ":" in token:
        archetype, agent = token.split(":", 1)
        return archetype, agent
    return token, None


def _catalog_block(
    catalog: list[tuple[str, str | None, dict[str, str]]],
    mission: Mission,
) -> str:
    """Render the selector's catalog: one line per valid `(archetype, agent)`
    tuple, encoded as a Literal token + the archetype's intent (for tonal
    judgment). The token format MUST match `_target_string` — the grammar
    sampler is built from these strings."""
    intents = {a.id: a.intent for a in mission.archetypes}
    lines = []
    for arch_id, agent_id, _ in catalog:
        token = _target_string(arch_id, agent_id)
        intent = intents.get(arch_id, "")
        lines.append(f"  - {token} — {intent}")
    return "\n".join(lines)


def _forced_event_selector_block(event: Event | None) -> str:
    """Render the FORCED EVENT block for the selector prompt — id + intent
    only. The selector flags one slot as the carrier; it does NOT write
    prose, so it doesn't need must_communicate / must_not / tone."""
    if event is None:
        return "(no forced event this day.)"
    return f"""\
A FORCED EVENT must be carried by exactly ONE selection this day:
  event_id: {event.id}
  intent: {event.intent.strip()}
Set `satisfies_event` = "{event.id}" on the one selection whose archetype
+ agent best fits this intent. Set it to null on the others. The prose
pass will honor the event's prose constraints on whichever slot you flag.
"""


def _forced_event_prose_block(event: Event | None, carrier_index: int | None) -> str:
    """Render the FORCED EVENT block for the prose writer — full prose
    constraints, targeted at the carrier slot index the selector picked."""
    if event is None or carrier_index is None:
        return "(no forced event this day.)"
    return f"""\
FORCED EVENT — carrier is slot #{carrier_index + 1}. That slot's prose MUST honor:
  event_id: {event.id}
  intent: {event.intent.strip()}
  must_communicate:
{chr(10).join(f"    - {x}" for x in event.must_communicate)}
  must_not:
{chr(10).join(f"    - {x}" for x in event.must_not)}
  tone: {event.tone}
The mechanical effect is engine-applied; you only write the prose situated
in current state.
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
        "LOCKED THIS DAY (use as context for option framing — do NOT propose these):",
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
        f"D{cycle_idx}: {archetype}({agent_id})" if agent_id else f"D{cycle_idx}: {archetype}"
        for cycle_idx, archetype, agent_id in (recently_tried or [])
    ]
    return _bullet_block(
        "RECENTLY TRIED (vary tonal range from these — different agent, "
        "different cost shape, or different point on the safe-↔-risky "
        "axis; do not propose the same (archetype, agent) pair the player "
        "just authorized):",
        lines,
    )


def _beat_block(beat_fragment: str | None) -> str:
    """Render the BEAT THIS DAY block. Empty when no beat fired this cycle.
    The fragment is one example phrasing — the system-prompt rule directs the
    LLM to use it as framing color, not to surface verbatim."""
    if not beat_fragment:
        return ""
    return _bullet_block(
        "BEAT THIS DAY (cycle framing — let it inform tone / risk emphasis "
        "without quoting verbatim):",
        [beat_fragment.strip()],
    )


def _selector_user_prompt(
    cycle_index: int,
    n: int,
    state: State,
    mission: Mission,
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_event: Event | None,
    log_entries: list[LogEntry],
    *,
    retry_feedback: str | None = None,
    recently_tried: Sequence[tuple[int, str, str | None]] | None = None,
    beat_fragment: str | None = None,
) -> str:
    retry = ""
    if retry_feedback:
        retry = f"\nPREVIOUS ATTEMPT WAS REJECTED:\n{retry_feedback}\n"
    act_phase = derive_act_phase(state, mission)
    return f"""\
Pick {n} targets for this day.

ACT PHASE: {act_phase}

CURRENT STATE:
{_narrative_state_block(state, mission)}
{_hidden_state_hint_block(state, mission, cycle_index)}{_beat_block(beat_fragment)}
RECENT CASE-FILE ENTRIES (last 2):
{_log_entries_block(log_entries, limit=2)}

VALID CATALOG (token — intent). Pick {n} of these as `target` values:
{_catalog_block(catalog, mission)}

{_locked_block(state, mission)}{_recently_tried_block(recently_tried)}
{_forced_event_selector_block(forced_event)}{retry}
Pick {n} targets now. No duplicates.
"""


def _prose_slot_block(
    slots: list[tuple[str, str | None, str | None]],
    mission: Mission,
) -> str:
    """Render the SLOTS TO WRITE PROSE FOR block. Each slot is the selector's
    `(archetype, agent, satisfies_event)` triple. Position in the list IS the
    slot index — the prose writer's output list is zipped to this order."""
    intents = {a.id: a.intent for a in mission.archetypes}
    lines: list[str] = []
    for i, (arch, ag, sat) in enumerate(slots, start=1):
        token = _target_string(arch, ag)
        intent = intents.get(arch, "")
        marker = "  [FORCED EVENT CARRIER]" if sat else ""
        lines.append(f"  {i}. {token} — {intent}{marker}")
    return "\n".join(lines)


def _prose_writer_user_prompt(
    cycle_index: int,
    state: State,
    mission: Mission,
    slots: list[tuple[str, str | None, str | None]],
    forced_event: Event | None,
    log_entries: list[LogEntry],
    *,
    retry_feedback: str | None = None,
    beat_fragment: str | None = None,
) -> str:
    """Build the prose writer's user prompt. `slots` is the selector's output
    in order: list of `(archetype, agent, satisfies_event)`. The prose
    writer's output list must have the same length and order — the engine
    zips them positionally to build the final `Option` list."""
    retry = ""
    if retry_feedback:
        retry = f"\nPREVIOUS ATTEMPT WAS REJECTED:\n{retry_feedback}\n"
    carrier_index = next(
        (i for i, (_, _, sat) in enumerate(slots) if sat),
        None,
    )
    act_phase = derive_act_phase(state, mission)
    return f"""\
Write prose for {len(slots)} slots, in order.

ACT PHASE: {act_phase}

CURRENT STATE:
{_narrative_state_block(state, mission)}
{_hidden_state_hint_block(state, mission, cycle_index)}{_beat_block(beat_fragment)}
RECENT CASE-FILE ENTRIES (last 2):
{_log_entries_block(log_entries, limit=2)}

SLOTS TO WRITE PROSE FOR (in this order; your output list is zipped to this):
{_prose_slot_block(slots, mission)}

{_forced_event_prose_block(forced_event, carrier_index)}{retry}
Write one title + prose pair per slot now, in order. {len(slots)} pairs.
"""


def _log_writer_user_prompt(
    cycle_index: int,
    state: State,
    mission: Mission,
    option: Option,
    delta: dict[str, tuple[Any, Any]],
    *,
    prior_state: State | None = None,
    leans: list[str] = (),
) -> str:
    """Build the cycle's user turn for the log writer. Goes onto the
    accumulating `message_history`; the assistant turn is the prose entry."""
    delta_str = (
        "\n".join(f"  - {q}: {before} -> {after}" for q, (before, after) in delta.items())
        if delta
        else "  (no state changes)"
    )

    act_phase = derive_act_phase(state, mission)
    tone_band = _TONE_BAND[act_phase]
    tension_band = derive_tension_band(prior_state, state)

    return f"""\
Day {cycle_index} of {mission.max_cycles}. Append the next case-file entry.

ACT PHASE: {act_phase}
TONE BAND: {tone_band}
TENSION (this day's cover swing): {tension_band}
{_leans_block(leans)}
AUTHORIZED OPTION:
  archetype: {option.archetype}
  agent: {option.agent or "(none)"}
  title: {option.title}
  prose (the option as offered): {option.prose}

RESOLVED STATE DELTA (this day's mutations — honor these magnitudes):
{delta_str}

CURRENT STATE (after delta — narrative, mirror this register):
{_narrative_state_block(state, mission)}
{_hidden_state_hint_block(state, mission, cycle_index)}
Write the entry now. Two paragraphs: angle, then resolution.
"""


# ───────────────────────── dynamic output-type builders ─────────────────────────


def _decide_n(catalog: list[tuple[str, str | None, dict[str, str]]]) -> int:
    """Engine decides how many options to surface this day. 4 when the
    catalog supports it, floor at 3 (per HANDLER spec). Catalogs with <3
    entries are an engine bug; engine_fallback_options handles that case."""
    return max(3, min(4, len(catalog)))


def _make_selector_output_type(
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_event: Event | None,
    n: int,
) -> type[BaseModel]:
    """Build the selector's per-cycle output schema. `target` is a `Literal`
    over the catalog's encoded `(archetype, agent)` tokens — under
    `strict=True` this compiles to a llama.cpp grammar sampler that rejects
    invalid catalog picks at decode time. When a forced event is due,
    `satisfies_event` is `Literal[event_id, None]`; otherwise the field is
    omitted entirely so the model doesn't waste tokens on a vestigial null."""
    target_tokens = tuple(_target_string(a, ag) for a, ag, _ in catalog)
    # `Literal[(t1, t2, ...)]` is equivalent to `Literal[t1, t2, ...]` —
    # Python's subscript syntax passes a tuple to `__class_getitem__`. The
    # runtime behavior is exactly what we need; type-checkers will object.
    TargetType = Literal[target_tokens]  # type: ignore[valid-type]

    if forced_event is not None:
        EventFlag = Literal[forced_event.id, None]  # type: ignore[valid-type]
        Selection = create_model(
            "Selection",
            target=(TargetType, ...),
            satisfies_event=(EventFlag, ...),
        )
    else:
        Selection = create_model(
            "Selection",
            target=(TargetType, ...),
        )

    return create_model(
        "Selections",
        selections=(list[Selection], Field(min_length=n, max_length=n)),
    )


def _make_prose_output_type(n: int) -> type[BaseModel]:
    """Prose writer's per-cycle output schema: a positional list of
    `OptionProse` pairs. No catalog identifiers — those are pinned by the
    selector and the engine zips them back positionally."""
    return create_model(
        "ProseSet",
        options=(list[OptionProse], Field(min_length=n, max_length=n)),
    )


# ───────────────────────── factory ─────────────────────────


def build_agent_bundle(model: OpenAIChatModel, mission: Mission) -> AgentBundle:
    """Construct the three agents for `mission`, baking the stable prefix
    into each agent's system prompt. Reuses one OpenAIChatModel across all —
    pydantic-ai handles concurrent calls.

    The selector and prose writer are built with `output_type=str` as a
    placeholder. Actual per-cycle Literal-typed schemas are passed via
    `agent.run(..., output_type=NativeOutput(BuiltType, strict=True))` — the
    pydantic-ai runtime-override pattern, which takes precedence over the
    construction-time type. `strict=True` is REQUIRED — without it,
    pydantic-ai sends `response_format.json_schema.strict=false` and
    llama.cpp (via inferno or the bundled llama-server) treats the schema
    as a hint rather than a sampler-grammar constraint. Gemma then ignores
    the schema and produces free-form text, parsing fails, and the call
    falls back to engine-authored options every cycle.
    """
    selector = Agent(
        model=model,
        output_type=str,
        system_prompt=_selector_system_prompt(mission),
    )
    prose_writer = Agent(
        model=model,
        output_type=str,
        system_prompt=_prose_writer_system_prompt(mission),
    )
    log_writer = Agent(
        model=model,
        output_type=str,
        system_prompt=_log_writer_system_prompt(mission),
    )
    return AgentBundle(
        selector=selector,
        prose_writer=prose_writer,
        log_writer=log_writer,
        mission=mission,
    )


# ───────────────────────── validation + retry/fallback ─────────────────────────


class DrafterValidationError(ValueError):
    """One or more drafter constraints violated; retry with feedback."""


def _slots_from_selections(
    selections_obj: BaseModel, has_forced: bool
) -> list[tuple[str, str | None, str | None]]:
    """Normalize a dynamically-built `Selections` instance to plain slot
    tuples — the engine's working form. `(archetype, agent, satisfies_event)`,
    where `satisfies_event` is None when no forced event was active this
    cycle (the schema doesn't even carry the field in that case)."""
    slots: list[tuple[str, str | None, str | None]] = []
    for s in selections_obj.selections:  # type: ignore[attr-defined]
        arch, ag = _parse_target(s.target)
        sat = s.satisfies_event if has_forced else None
        slots.append((arch, ag, sat))
    return slots


def _zip_options(
    slots: list[tuple[str, str | None, str | None]],
    prose_list: list[OptionProse],
) -> list[Option]:
    """Pair selector slots with prose-writer output positionally — the two
    LLM calls' outputs are joined by list index. Schema constraints on both
    sides guarantee the lengths match before this is called."""
    return [
        Option(
            archetype=arch,
            agent=ag,
            title=p.title,
            prose=p.prose,
            satisfies_event=sat,
        )
        for (arch, ag, sat), p in zip(slots, prose_list)
    ]


def validate_selections(
    slots: list[tuple[str, str | None, str | None]],
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_event_id: str | None,
    n: int,
) -> None:
    """Raise `DrafterValidationError` with a corrective message if the
    selector's output violates constraints. Catalog-membership is enforced
    by the grammar sampler under `strict=True` — we check it here too as
    defense-in-depth so the validator's feedback can speak in the same
    language as the prompt if something slips through."""
    problems: list[str] = []
    if len(slots) != n:
        problems.append(
            f"selection count {len(slots)} != {n}; produce exactly {n} selections"
        )
    valid_keys = {(a, ag) for (a, ag, _) in catalog}
    seen: set[tuple[str, str | None]] = set()
    for i, (arch, ag, _sat) in enumerate(slots):
        key = (arch, ag)
        if key not in valid_keys:
            problems.append(
                f"selection {i + 1}: target archetype={arch!r} agent={ag!r} "
                f"is not in the VALID CATALOG"
            )
        if key in seen:
            problems.append(
                f"selection {i + 1}: duplicate target archetype={arch!r} agent={ag!r}"
            )
        seen.add(key)
    if forced_event_id is not None:
        carriers = [s for s in slots if s[2] == forced_event_id]
        if len(carriers) != 1:
            problems.append(
                f"forced event {forced_event_id!r}: expected exactly 1 selection "
                f"with satisfies_event={forced_event_id!r}, found {len(carriers)}"
            )
    if problems:
        raise DrafterValidationError("; ".join(problems))


def validate_prose(prose_list: list[OptionProse], n: int) -> None:
    """Prose-side validation. Schema-level `min_length=1` on title/prose
    catches empties; this validator's job is the count + room to grow with
    quality checks (regex on game/system vocab, word-count bounds) if Gemma
    E4B starts misbehaving in practice."""
    problems: list[str] = []
    if len(prose_list) != n:
        problems.append(
            f"prose count {len(prose_list)} != {n}; produce exactly {n} title+prose pairs"
        )
    if problems:
        raise DrafterValidationError("; ".join(problems))


def engine_fallback_options(
    catalog: list[tuple[str, str | None, dict[str, str]]],
    mission: Mission,
    forced_event: Event | None,
    n: int,
) -> list[Option]:
    """Engine-authored fallback when either LLM call exhausts retries. Picks
    deterministic options from the valid catalog with terse generic prose.

    Strategy:
    - If a forced event is required, the first option satisfies it (using
      any agent-bound catalog entry as carrier — forced events apply their
      effects on top of the carrier archetype's).
    - HOLD is included next if available.
    - Then one (archetype, agent) per remaining agent.
    - Fill from leftover catalog entries until N options exist.
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
            ((a, ag) for (a, ag, _) in catalog if ag is not None),
            None,
        )
        if carrier is not None:
            a, ag = carrier
            _add(
                Option(
                    archetype=a,
                    agent=ag,
                    title=forced_event.id,
                    prose=forced_event.intent.strip()[:240] or f"Honor the {forced_event.id} event.",
                    satisfies_event=forced_event.id,
                )
            )

    # HOLD next, if available and not already used as a forced-moment carrier.
    for (a, ag, _) in catalog:
        if a != "HOLD":
            continue
        _add(
            Option(
                archetype=a,
                agent=ag,
                title="Hold position",
                prose="Take no action this day. Both agents stay put. The situation tightens.",
            )
        )
        break

    # One archetype per remaining agent.
    seen_agents: set[str | None] = {o.agent for o in options}
    for (a, ag, _) in catalog:
        if a == "HOLD" or ag in seen_agents:
            continue
        seen_agents.add(ag)
        _add(
            Option(
                archetype=a,
                agent=ag,
                title=f"Authorize {a} ({ag or 'unbound'})",
                prose=(
                    f"Authorize the {a} archetype on {ag or 'Operations'}. "
                    f"The mechanical cost is documented in the catalog."
                ),
            )
        )
        if len(options) >= n:
            break

    # Fill from remaining catalog entries until ≥N (or catalog exhausted).
    if len(options) < n:
        for (a, ag, _) in catalog:
            _add(
                Option(
                    archetype=a,
                    agent=ag,
                    title=f"Authorize {a}",
                    prose=f"Authorize the {a} archetype.",
                )
            )
            if len(options) >= n:
                break

    return options[:n]


async def select_then_write_with_retries(
    selector: Agent[None, Any],
    prose_writer: Agent[None, Any],
    *,
    cycle_index: int,
    state: State,
    mission: Mission,
    catalog: list[tuple[str, str | None, dict[str, str]]],
    forced_event: Event | None,
    log_entries: list[LogEntry],
    max_retries: int = 3,
    recently_tried: Sequence[tuple[int, str, str | None]] | None = None,
    beat_fragment: str | None = None,
) -> tuple[list[Option], bool]:
    """Two-phase drafter orchestration:
      1. Selector picks N catalog targets + forced-event carrier.
      2. Prose writer writes (title, prose) pairs for each pinned slot.

    Each phase has its own retry budget. If either exhausts retries, the
    function returns engine-authored fallback options and `engine_fallback=True`.

    Per-cycle output schemas are built dynamically: the selector's `target`
    is a `Literal` over the current catalog (grammar-sampler enforces
    validity at decode time); the prose writer's schema is a fixed-length
    list of `OptionProse` pairs. Both schemas are passed via pydantic-ai's
    runtime `output_type=` override on `.run()`.
    """
    n = _decide_n(catalog)
    forced_id = forced_event.id if forced_event is not None else None

    # ─── Phase 1: selector ───
    selector_type = _make_selector_output_type(catalog, forced_event, n)
    slots: list[tuple[str, str | None, str | None]] | None = None
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        prompt = _selector_user_prompt(
            cycle_index, n, state, mission, catalog, forced_event, log_entries,
            retry_feedback=last_error,
            recently_tried=recently_tried,
            beat_fragment=beat_fragment,
        )
        with logfire.span("selector_call", attempt=attempt + 1):
            try:
                result = await selector.run(
                    prompt, output_type=NativeOutput(selector_type, strict=True)
                )
            except Exception as e:
                log.warning("selector call failed: %s", e)
                last_error = f"agent run failed: {e}"
                continue
        candidate = _slots_from_selections(
            result.output, has_forced=forced_event is not None
        )
        try:
            validate_selections(candidate, catalog, forced_id, n)
            slots = candidate
            break
        except DrafterValidationError as e:
            last_error = str(e)
            log.info("selector validation failed (attempt %d): %s", attempt + 1, e)

    if slots is None:
        log.warning(
            "selector exhausted %d retries; using engine fallback options", max_retries
        )
        return engine_fallback_options(catalog, mission, forced_event, n), True

    # ─── Phase 2: prose writer ───
    prose_type = _make_prose_output_type(n)
    last_error = None
    for attempt in range(max_retries + 1):
        prompt = _prose_writer_user_prompt(
            cycle_index, state, mission, slots, forced_event, log_entries,
            retry_feedback=last_error,
            beat_fragment=beat_fragment,
        )
        with logfire.span("prose_writer_call", attempt=attempt + 1):
            try:
                result = await prose_writer.run(
                    prompt, output_type=NativeOutput(prose_type, strict=True)
                )
            except Exception as e:
                log.warning("prose writer call failed: %s", e)
                last_error = f"agent run failed: {e}"
                continue
        prose_list: list[OptionProse] = list(result.output.options)
        try:
            validate_prose(prose_list, n)
            return _zip_options(slots, prose_list), False
        except DrafterValidationError as e:
            last_error = str(e)
            log.info("prose writer validation failed (attempt %d): %s", attempt + 1, e)

    log.warning(
        "prose writer exhausted %d retries; using engine fallback options", max_retries
    )
    return engine_fallback_options(catalog, mission, forced_event, n), True
