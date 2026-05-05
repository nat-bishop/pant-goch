"""The two LLM agents: CHAT_AGENT and CLASSIFIER_AGENT.

CHAT_AGENT renders one email body per turn. It uses Pydantic AI's native
`message_history` for the per-thread conversation, plain-text `output_type=str`
so the assistant turn lands as a `TextPart` (clean for subsequent turns), and
dynamic `@instructions` keyed on `ChatDeps` to layer dossier and optional
plot-directive context per run.

CLASSIFIER_AGENT is one-shot, ToolOutput, returns a `value: str` chosen from
a beat-authored `allowed_values` list. The list is enforced via the user
prompt (not the schema) since allowed values vary per beat. The caller
validates post-hoc and skips the quality write if the returned value is
off-spec. The classifier runs only when the player is replying to a beat
that authored a `classify_reply` block — never on every send.

Construction is deferred to `build_agents()` (called from `lifespan`) so the
shared OpenAI-compatible model handle can be passed in once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, ToolOutput

from .models import (
    BeatSpec,
    ClassificationResult,
    NPCDossier,
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
    plot_directive: BeatSpec | None


# ───────────────────────── system prompts ─────────────────────────


_BASE_CHAT_INSTRUCTIONS = """You are writing a single email body in the voice of the NPC described below.

OUTPUT FORMAT (strict):
- Body text only. No subject line. No "From:" / "To:" / "Date:" headers.
- No sign-off / signature unless the dossier explicitly specifies one.
- No quoted prior message. No "Re:" prefix in the output.
- No JSON, no markdown code fences, no meta-commentary.

VOICE RULES:
- Stay inside the dossier's base voice register.
- Do not invent named characters or places not in the dossier or transcript.
- Do not mention game mechanics, scoring, or the meta-fiction.
- Address the recipient by name or as "you", never as "the player".
"""


_CLASSIFIER_SYSTEM_PROMPT = """You classify a player-written email reply against a beat-authored question.

You will receive:
- The recipient's persona blurb (who the player is writing to).
- The most recent prior email in the thread (what the player is replying to).
- The player's just-typed reply (the new send).
- The classification question (what the author wants to know).
- The allowed values (you MUST pick exactly one, verbatim).

Pick the single allowed value that best matches the player's reply. Do not
return values outside the allowed list. Also write a one-sentence rationale
(kept for telemetry, not shown to the player).

Return: {"value": "<one of allowed_values>", "rationale": "<one sentence>"}.
"""


# ───────────────────────── agent construction ─────────────────────────


@dataclass
class AgentBundle:
    """Both agents wired up against one shared model handle."""

    chat_agent: Agent
    classifier_agent: Agent


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

    classifier_agent = Agent(
        model,
        name="classifier_agent",
        output_type=ToolOutput(ClassificationResult),
        instructions=_CLASSIFIER_SYSTEM_PROMPT,
        # Headroom: Gemma 4 E4B sometimes preambles before emitting the
        # tool call. At max_tokens=256 it regularly hit `finish_reasons=
        # ["length"]` on the first attempt, forcing Pydantic AI to retry
        # and ~2x the latency. 2048 leaves ample budget for the call plus
        # any preamble the model wants to write.
        model_settings={"max_tokens": 2048, "temperature": 0.1},
    )

    return AgentBundle(
        chat_agent=chat_agent,
        classifier_agent=classifier_agent,
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


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items) or "  (none)"


def build_classifier_user_prompt(
    recipient_blurb: str,
    prior_email_body: str,
    reply_body: str,
    question: str,
    allowed_values: list[str],
) -> str:
    """The schema only enforces that *some* string is returned, so the LLM
    must respect the allowed_values list given in the prompt."""
    values = ", ".join(f'"{v}"' for v in allowed_values)
    return f"""RECIPIENT (who the player is writing to):
{recipient_blurb}

MOST RECENT PRIOR EMAIL (what the player is replying to):
{prior_email_body.strip() or '(no prior email)'}

PLAYER'S REPLY (just typed):
{reply_body}

CLASSIFICATION QUESTION:
{question.strip()}

ALLOWED VALUES (pick exactly one, verbatim):
{values}

Return one of the allowed values and a one-sentence rationale."""
