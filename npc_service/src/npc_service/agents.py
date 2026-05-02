"""The four LLM agents: EMAIL_AGENT, TEXTURE_AGENT, VOICE_FIDELITY_AGENT,
CODA_AGENT. All P1 (single agent + structured output). All ToolOutput. None
of them write state directly — they return structured outputs that the spine
applies after validation.

Construction is deferred to `build_agents()` (called from `lifespan`) so the
shared OpenAI-compatible model handle can be passed in once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_ai import Agent, ToolOutput

from .models import (
    CodaOutput,
    EmailRender,
    InboxState,
    NPCDossier,
    RenderedBeat,
    TextureScore,
    VoiceBand,
    VoiceFidelityScore,
)
from .spine import CodaFragment, select_voice_band


log = logging.getLogger("npc_service.agents")


@dataclass
class AgentBundle:
    """All four agents wired up against one shared model handle."""

    email_agent: Agent
    texture_agent: Agent
    voice_fidelity_agent: Agent
    coda_agent: Agent


def build_agents(model) -> AgentBundle:
    """Construct the four agents. System prompts are static (no per-NPC
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

    voice_fidelity_agent = Agent(
        model,
        name="voice_fidelity_agent",
        output_type=ToolOutput(VoiceFidelityScore),
        instructions=_VOICE_FIDELITY_SYSTEM_PROMPT,
        model_settings={"max_tokens": 256, "temperature": 0.1},
    )

    coda_agent = Agent(
        model,
        name="coda_agent",
        output_type=ToolOutput(CodaOutput),
        instructions=_CODA_SYSTEM_PROMPT,
        model_settings={"max_tokens": 600, "temperature": 0.6},
    )

    return AgentBundle(
        email_agent=email_agent,
        texture_agent=texture_agent,
        voice_fidelity_agent=voice_fidelity_agent,
        coda_agent=coda_agent,
    )


# ───────────────────────── system prompts ─────────────────────────


_EMAIL_SYSTEM_PROMPT = """You write a single email in the voice of a specific NPC, given a dossier and a beat spec.

You will receive in the user prompt:
- The NPC's full dossier (voice register, will-say / will-not-say, voice band examples).
- The selected voice band for THIS render (warm / neutral / reserved).
- The thread transcript so far (prior emails in this exchange, plain text).
- The beat spec (intent, must-include facts, must-not constraints, length window).
- Optionally: extra inputs like the player's draft text, voice_fidelity_rationale, or the value of relevant atoms.

Your job: write the email body. ONE email. No subject line, no headers, no signature unless the dossier says so. Just the body text.

Hard rules:
- Stay inside the voice band. If the band says lower-case, lower-case. If it forbids 'lol', no 'lol'.
- Do not invent named characters or places not in the dossier or transcript.
- Do not mention game mechanics, scoring, or the meta-fiction.
- Do not address the player directly as a player. Address the recipient as the email's `to:` field.
- Length: stay within the spec's word range.

Return: {"prose": "<the email body>"}.
"""


_TEXTURE_SYSTEM_PROMPT = """You score the texture of a player-written email reply on two axes: warmth and honesty.

You will receive:
- The recipient's persona blurb (who the player is writing to).
- The thread transcript (what came before).
- The player's just-typed reply (the new send).

Score each axis on a -3..3 integer scale:

WARMTH:
  +3  intimate, specific, attentive — names the recipient's feelings or specifics
  +1  friendly, present
   0  neutral, transactional
  -1  curt, distant
  -3  cold, dismissive

HONESTY:
  +3  acknowledges the situation directly without flinching
  +1  honest about feelings if not facts
   0  neutral or fact-only
  -1  hedged, evasive
  -3  actively misleading

The empty-default placeholder reply "got it." should score warmth: -1 or 0, honesty: 0.
A reply that lies about Tess being fine should score honesty: -2 or -3.
A reply naming a specific concern shared in the recipient's prior emails scores high warmth.

Also write a one-sentence rationale (kept for telemetry, not shown to player).

Return: {"warmth": <int>, "honesty": <int>, "rationale": "<one sentence>"}.
"""


_VOICE_FIDELITY_SYSTEM_PROMPT = """You judge how much of an inbox owner's voice survived a player's edit of their unsent draft.

You will receive:
- The owner's voice dossier (register, vocabulary, signature, archive quotes).
- The original draft text the owner wrote (and chose not to send).
- The edited version the player produced.

Score voice_fidelity on 0.0..1.0:
  1.0  identical or near-identical voice — same register, same hedges, same affect
  0.7  recognizable but with shifts in register or specificity
  0.5  partially recognizable, half the voice is gone
  0.3  mostly the player's own voice now
  0.0  unrecognizable as the owner's writing

The rationale is the load-bearing output. One sentence. Name what was preserved AND what was changed. Examples:
  "kept the lowercase and the soup specificity but dropped the hedge in the middle paragraph"
  "preserved the halting rhythm; changed 'idk' to 'I don't know' which reads more decisive than Tess ever does"
  "rewrote almost entirely in a more confident register; the apology and the specifics are gone"

Return: {"voice_fidelity": <float>, "rationale": "<one sentence>"}.
"""


_CODA_SYSTEM_PROMPT = """You write the closing montage for a single playthrough of an inbox-only narrative game.

The screen will scroll a sequence of fragments (excerpts from the player's session) with a quiet line of authorial framing between each. You write ONLY the framing lines — never the fragments themselves; those are pulled from session content verbatim.

You will receive:
- The list of fragments to be shown, in order, with a short label for each.
- The terminal state of Maya's thread (warm | ambivalent | cold).
- Whether the class registration was preserved.
- The unsent-draft disposition (pending | sent_unchanged | sent_edited | held).
- The trajectory of the player's register quality (Maya's register over the session).

Write 6-10 framing lines, one per fragment plus a closing line. Each line is 10-25 words. Quiet, third-person authorial, present tense. Do NOT address the player or Tess directly. Do NOT use first-person ("I"). Do NOT name the brother — he stays unnamed. Do NOT use resolution language ("she'll be okay", "everything works out"). Do NOT use clinical/medical vocabulary. Do NOT past-tense summarize ("the player learned that...").

The tone modulates based on inputs:
- maya.state == warm + draft sent: quiet relief, things held carefully
- maya.state == warm + draft held: respect, the small wisdom of staying inside scope
- maya.state == ambivalent: the gap between what was said and what was meant
- maya.state == cold + anything: small, what wasn't said, leaving the door open
- class_registration preserved: a thing kept; small but real
- class_registration expired: a thing not kept; not framed as failure, just absent

Return: {"framing_lines": ["<line 1>", "<line 2>", ...]}.
"""


# ───────────────────────── prompt builders ─────────────────────────


def build_email_user_prompt(
    beat: RenderedBeat,
    npc: NPCDossier,
    state: InboxState,
    transcript: str,
    extra_context: dict[str, str] | None = None,
) -> str:
    band = select_voice_band(npc, state)
    band_block = _format_voice_band(band) if band else "(NPC has no voice bands; use base voice register only.)"

    facts = _format_facts(state)
    must_include = "\n".join(f"  - {f}" for f in beat.spec.must_include_facts) or "  (none specified)"
    must_not = "\n".join(f"  - {f}" for f in beat.spec.must_not) or "  (none specified)"
    extra_inputs = "\n".join(f"  - {k}: {v}" for k, v in (extra_context or {}).items()) or "  (none)"

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
Intent: {beat.spec.intent.strip()}
Tone source: {beat.spec.tone_source}{length}

Must include:
{must_include}

Must not:
{must_not}

Relevant facts the player has come to know:
{facts}

Extra context for this render:
{extra_inputs}

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


def _format_facts(state: InboxState) -> str:
    """Surface a few player-known atoms that may be relevant. Conservative —
    we don't dump the entire ledger because the model doesn't need it.
    """
    relevant: list[str] = []
    for atom_id, value in state.atoms.items():
        if value in (False, "unknown", "pending"):
            continue
        relevant.append(f"  - {atom_id} = {value!r}")
    return "\n".join(relevant) or "  (no facts surfaced yet)"


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

Now score this reply on warmth and honesty (-3..3 integers) and write a one-sentence rationale."""


def build_voice_fidelity_user_prompt(
    tess_dossier: NPCDossier,
    original_draft: str,
    edited_draft: str,
) -> str:
    examples = "\n".join(f"  - {q!r}" for q in tess_dossier.archive_quotes) or "  (none)"
    return f"""OWNER VOICE DOSSIER ({tess_dossier.display_name}):
Register: {tess_dossier.base_voice_register}
{_format_dossier_voice(tess_dossier)}
Archive quotes (real samples of the owner's voice):
{examples}

ORIGINAL DRAFT (what the owner wrote):
{original_draft}

EDITED DRAFT (what the player sends):
{edited_draft}

Score voice_fidelity (0.0..1.0) and name what survived and what changed in one sentence."""


def build_coda_user_prompt(
    fragments: list[CodaFragment],
    state: InboxState,
) -> str:
    frags = "\n".join(
        f"  {i+1}. [{f.label}] {f.excerpt!r}"
        for i, f in enumerate(fragments)
    ) or "  (no fragments — should not happen)"

    maya_state = state.thread_states.get("maya", "?")
    disposition = state.atoms.get("unsent_draft_disposition", "pending")
    class_kept = "class_registration" in state.commitments
    register_traj = state.qualities.get("register.maya", 0.0)

    return f"""TONE INPUTS:
- maya.state: {maya_state}
- class_registration_preserved: {class_kept}
- unsent_draft_disposition: {disposition}
- register.maya (final): {register_traj:.2f}

FRAGMENTS to be shown (in order):
{frags}

Write {len(fragments) + 1} framing lines — one before each fragment, plus a closing line. Each line 10-25 words. Quiet authorial third-person. No first-person, no addressing the player or Tess, no resolution language, no clinical vocabulary."""
