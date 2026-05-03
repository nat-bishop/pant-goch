"""The two LLM agents: EMAIL_AGENT and TEXTURE_AGENT. Both use ToolOutput.
Neither writes state directly — they return structured outputs that the spine
applies after validation.

Construction is deferred to `build_agents()` (called from `lifespan`) so the
shared OpenAI-compatible model handle can be passed in once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_ai import Agent, ToolOutput

from .models import (
    EmailRender,
    InboxState,
    NPCDossier,
    RenderedBeat,
    TextureScore,
    VoiceBand,
)
from .spine import select_voice_band


log = logging.getLogger("npc_service.agents")


@dataclass
class AgentBundle:
    """Both agents wired up against one shared model handle."""

    email_agent: Agent
    texture_agent: Agent


def build_agents(model) -> AgentBundle:
    """Construct the two agents. System prompts are static (no per-NPC
    customization) so prompt caching stays warm across calls; the per-call
    NPC dossier and beat spec go in the user prompt.
    """
    email_agent = Agent(
        model,
        name="email_agent",
        output_type=ToolOutput(EmailRender),
        instructions=_EMAIL_SYSTEM_PROMPT,
        model_settings={"max_tokens": 800, "temperature": 0.7},
    )

    texture_agent = Agent(
        model,
        name="texture_agent",
        output_type=ToolOutput(TextureScore),
        instructions=_TEXTURE_SYSTEM_PROMPT,
        model_settings={"max_tokens": 256, "temperature": 0.1},
    )

    return AgentBundle(
        email_agent=email_agent,
        texture_agent=texture_agent,
    )


# ───────────────────────── system prompts ─────────────────────────


_EMAIL_SYSTEM_PROMPT = """You write a single email in the voice of a specific NPC, given a dossier and a beat spec.

You will receive in the user prompt:
- The NPC's full dossier (voice register, will-say / will-not-say, voice band examples).
- The selected voice band for THIS render (warm / neutral / reserved).
- The thread transcript so far (prior emails in this exchange, plain text).
- The beat spec (intent, must-include facts, must-not constraints, length window).

Your job: write the email body. ONE email. No subject line, no headers, no signature unless the dossier says so. Just the body text.

Hard rules:
- Stay inside the voice band. If the band says lower-case, lower-case. If it forbids 'lol', no 'lol'.
- Do not invent named characters or places not in the dossier or transcript.
- Do not mention game mechanics, scoring, or the meta-fiction.
- Do not address the player directly as a player. Address the recipient as the email's `to:` field.
- Length: stay within the spec's word range.

Return: {"prose": "<the email body>"}.
"""


_TEXTURE_SYSTEM_PROMPT = """You score the warmth of a player-written email reply on a -3..3 scale.

You will receive:
- The recipient's persona blurb (who the player is writing to).
- The thread transcript (what came before).
- The player's just-typed reply (the new send).

Score warmth on a -3..3 integer scale:

  +3  intimate, specific, attentive — names the recipient's feelings or specifics
  +1  friendly, present
   0  neutral, transactional
  -1  curt, distant
  -3  cold, dismissive

A reply naming a specific concern shared in the recipient's prior emails scores high warmth.

Also write a one-sentence rationale (kept for telemetry, not shown to player).

Return: {"warmth": <int>, "rationale": "<one sentence>"}.
"""


# ───────────────────────── prompt builders ─────────────────────────


def build_email_user_prompt(
    beat: RenderedBeat,
    npc: NPCDossier,
    state: InboxState,
    transcript: str,
) -> str:
    band = select_voice_band(npc, state)
    band_block = _format_voice_band(band) if band else "(NPC has no voice bands; use base voice register only.)"

    must_include = "\n".join(f"  - {f}" for f in beat.spec.must_include_facts) or "  (none specified)"
    must_not = "\n".join(f"  - {f}" for f in beat.spec.must_not) or "  (none specified)"

    length = ""
    if beat.spec.length_words:
        length = f"\nLength: {beat.spec.length_words[0]}-{beat.spec.length_words[1]} words."

    return f"""DOSSIER for {npc.display_name} ({npc.id}):
Relationship: {npc.relationship_to_owner}
Base voice: {npc.base_voice_register}
{_format_dossier_voice(npc)}

VOICE BAND for this render:
{band_block}

THREAD TRANSCRIPT (prior emails in this exchange):
{transcript}

BEAT SPEC:
Intent: {beat.spec.intent.strip()}{length}

Must include:
{must_include}

Must not:
{must_not}

Now write the email body."""


def _format_dossier_voice(npc: NPCDossier) -> str:
    return (
        f"Forbidden:\n{_bullets(npc.base_voice_forbidden)}\n"
        f"Will say:\n{_bullets(npc.will_say)}\n"
        f"Will not say:\n{_bullets(npc.will_not_say)}"
    )


def _format_voice_band(band: VoiceBand) -> str:
    vocab = ", ".join(f"{v!r}" for v in band.vocabulary_examples) or "(none)"
    examples = "\n".join(f"  - {ex!r}" for ex in band.tone_examples) or "  (none)"
    return (
        f"Band: {band.name}\n"
        f"Selected when: {band.selected_when}\n"
        f"Sentence register: {band.sentence_register}\n"
        f"Vocabulary cues: {vocab}\n"
        f"Tone examples:\n{examples}"
    )


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items) or "  (none)"


def build_texture_user_prompt(
    recipient_blurb: str,
    transcript: str,
    reply_body: str,
) -> str:
    return f"""RECIPIENT (who the player is writing to):
{recipient_blurb}

THREAD TRANSCRIPT (what came before):
{transcript}

PLAYER'S REPLY (just typed):
{reply_body}

Now score this reply's warmth (-3..3 integer) and write a one-sentence rationale."""
