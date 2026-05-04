"""The two LLM agents: CHAT_AGENT and TEXTURE_AGENT.

CHAT_AGENT renders one email body per turn. It uses Pydantic AI's native
`message_history` for the per-thread conversation, plain-text `output_type=str`
so the assistant turn lands as a `TextPart` (clean for subsequent turns), and
dynamic `@instructions` keyed on `ChatDeps` to layer dossier / voice-band /
optional plot-directive context per run.

TEXTURE_AGENT is unchanged in shape: one-shot, ToolOutput, scores warmth on
the player's just-typed reply. It never sees `message_history`.

Construction is deferred to `build_agents()` (called from `lifespan`) so the
shared OpenAI-compatible model handle can be passed in once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, ToolOutput

from .models import (
    BeatSpec,
    NPCDossier,
    TextureScore,
    VoiceBand,
)


log = logging.getLogger("npc_service.agents")


# ───────────────────────── chat-agent deps ─────────────────────────


@dataclass
class ChatDeps:
    """Per-run injection for the chat agent.

    `plot_directive` is None for chat-turn fallbacks (free conversation in
    character) and is a `BeatSpec` when a PlotBeat is firing on this turn —
    in which case the directive is layered into the prompt as authored
    intent / must-include / must-not / length window.
    """

    npc: NPCDossier
    voice_band: VoiceBand | None
    plot_directive: BeatSpec | None


# ───────────────────────── system prompts ─────────────────────────


_BASE_CHAT_INSTRUCTIONS = """You are writing a single email body in the voice of the NPC described below.

OUTPUT FORMAT (strict):
- Body text only. No subject line. No "From:" / "To:" / "Date:" headers.
- No sign-off / signature unless the dossier explicitly specifies one.
- No quoted prior message. No "Re:" prefix in the output.
- No JSON, no markdown code fences, no meta-commentary.

VOICE RULES:
- Stay inside the selected voice band. If the band says lower-case, lower-case.
  If it forbids 'lol', no 'lol'.
- Do not invent named characters or places not in the dossier or transcript.
- Do not mention game mechanics, scoring, or the meta-fiction.
- Address the recipient by name or as "you", never as "the player".
- Stay within the length window if a plot directive specifies one.
"""


_TEXTURE_SYSTEM_PROMPT = """You score the warmth of a player-written email reply on a -3..3 scale.

You will receive:
- The recipient's persona blurb (who the player is writing to).
- The most recent prior email in the thread (what the player is replying to).
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


# ───────────────────────── agent construction ─────────────────────────


@dataclass
class AgentBundle:
    """Both agents wired up against one shared model handle."""

    chat_agent: Agent
    texture_agent: Agent


def build_agents(model) -> AgentBundle:
    """Construct the two agents. The chat agent's per-run customization is
    in `@instructions` decorators below — they read from `ctx.deps` (ChatDeps)
    so a single shared agent handles every NPC and every turn type.
    """
    chat_agent: Agent[ChatDeps, str] = Agent(
        model,
        name="chat_agent",
        deps_type=ChatDeps,
        output_type=str,
        instructions=_BASE_CHAT_INSTRUCTIONS,
        model_settings={"max_tokens": 800, "temperature": 0.7},
    )

    @chat_agent.instructions
    def _dossier_block(ctx: RunContext[ChatDeps]) -> str:
        return _format_dossier(ctx.deps.npc)

    @chat_agent.instructions
    def _voice_band_block(ctx: RunContext[ChatDeps]) -> str:
        band = ctx.deps.voice_band
        if band is None:
            return ""
        return "VOICE BAND for this render:\n" + _format_voice_band(band)

    @chat_agent.instructions
    def _plot_directive_block(ctx: RunContext[ChatDeps]) -> str:
        spec = ctx.deps.plot_directive
        if spec is None:
            return ""
        must_include = "\n".join(f"  - {f}" for f in spec.must_include_facts) or "  (none specified)"
        must_not = "\n".join(f"  - {f}" for f in spec.must_not) or "  (none specified)"
        length = ""
        if spec.length_words:
            length = f"\nLength: {spec.length_words[0]}-{spec.length_words[1]} words."
        return (
            f"PLOT DIRECTIVE (this turn only):\n"
            f"Intent: {spec.intent.strip()}{length}\n"
            f"Must include:\n{must_include}\n"
            f"Must not:\n{must_not}"
        )

    texture_agent = Agent(
        model,
        name="texture_agent",
        output_type=ToolOutput(TextureScore),
        instructions=_TEXTURE_SYSTEM_PROMPT,
        model_settings={"max_tokens": 256, "temperature": 0.1},
    )

    return AgentBundle(
        chat_agent=chat_agent,
        texture_agent=texture_agent,
    )


# ───────────────────────── prompt formatting helpers ─────────────────────────


def _format_dossier(npc: NPCDossier) -> str:
    return (
        f"DOSSIER for {npc.display_name} ({npc.id}):\n"
        f"Relationship: {npc.relationship_to_owner}\n"
        f"Base voice: {npc.base_voice_register}\n"
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
    prior_email_body: str,
    reply_body: str,
) -> str:
    """One-shot input for TEXTURE_AGENT. Single prior email is sufficient
    context for warmth scoring; we don't need the full thread."""
    return f"""RECIPIENT (who the player is writing to):
{recipient_blurb}

MOST RECENT PRIOR EMAIL (what the player is replying to):
{prior_email_body.strip() or '(no prior email)'}

PLAYER'S REPLY (just typed):
{reply_body}

Now score this reply's warmth (-3..3 integer) and write a one-sentence rationale."""
