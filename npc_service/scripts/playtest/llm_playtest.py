"""Synthetic-player agent + session driver.

Runs one full session of the production engine against the local llama,
with the player side synthesised by another `pydantic-ai` Agent given a
PersonaSpec. Calls the same `record_send` / `process_send` extracted in
Phase 1, so the engine codepath under test is exactly the FastAPI /send
path minus SSE.
"""
from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import npc_service.main as engine_main
from npc_service.agents import build_agents, make_llama_token_counter
from npc_service.config import Settings
from npc_service.models import (
    Email,
    InboxState,
    Reply,
    SendRequest,
    load_scenario,
)

from .personas import PersonaSpec
from .session import SessionRecord


@dataclass
class PlayerDeps:
    """Per-render dependencies for the synthetic player agent — the persona
    plus the recent transcript window."""

    persona: PersonaSpec
    recent_transcript: list[Email]


_PLAYER_SYSTEM = """\
You are role-playing a player of a single-surface email game. Your job is
to produce ONE plausible reply — body text only, no headers, no signoff
formatting beyond what the persona demands.

You will receive:
  - PERSONA: who you are playing (voice, objective, archetype, quirks)
  - RECENT TRANSCRIPT: the last few emails on this thread

Output requirements:
  - Plain prose body. No subject line, no "From:/To:". No XML, no JSON.
  - Stay in character — voice and quirks must show through.
  - Reply to whatever the most recent email in the transcript said.
  - 2–6 sentences typical. If your persona is terse, prefer 1–2.
  - Do not narrate stage directions; produce the reply itself.
  - If asked for a personal detail (name, role, employer, hometown, etc.)
    and the PERSONA does not specify one, INVENT a realistic concrete
    value and use it consistently. NEVER emit a bracketed placeholder
    like `[Candidate Name]`, `<your name>`, `[YOUR ROLE HERE]`, etc.
    Pick one plausible name once and stay with it for the rest of the
    session.
"""


def _build_player_agent(model: OpenAIChatModel) -> Agent[PlayerDeps, str]:
    """Construct the synthetic-player agent. Temperature is slightly hotter
    than the production chat agent (1.1 vs 1.0) so persona variation
    surfaces across seeds without drifting off-character."""
    agent: Agent[PlayerDeps, str] = Agent(
        model,
        name="synthetic_player",
        deps_type=PlayerDeps,
        output_type=str,
        instructions=_PLAYER_SYSTEM,
        model_settings={
            "max_tokens": 2048,
            "temperature": 1.1,
            "top_p": 0.95,
            "extra_body": {"top_k": 64},
        },
    )

    @agent.instructions
    def _persona_block(ctx) -> str:
        p = ctx.deps.persona
        quirks = "\n".join(f"  - {q}" for q in p.quirks) or "  (none)"
        return (
            "PERSONA\n"
            f"  name: {p.name}\n"
            f"  archetype: {p.archetype}\n"
            f"  voice: {p.voice}\n"
            f"  objective: {p.objective}\n"
            f"  quirks:\n{quirks}\n"
        )

    @agent.instructions
    def _transcript_block(ctx) -> str:
        if not ctx.deps.recent_transcript:
            return "RECENT TRANSCRIPT: (empty — this is your opening reply)\n"
        lines = ["RECENT TRANSCRIPT (oldest → newest)"]
        for e in ctx.deps.recent_transcript:
            lines.append(f"  --- {e.sender} → {e.recipient} ---")
            lines.append("  " + e.body.replace("\n", "\n  "))
        return "\n".join(lines) + "\n"

    return agent


async def _synthesise_reply(
    player_agent: Agent[PlayerDeps, str],
    persona: PersonaSpec,
    transcript: list[Email],
    *,
    seed: int | None = None,
) -> str:
    """One synthetic-player call. Returns the body text.

    `seed`, when set, is forwarded to llama.cpp via `extra_body` so two
    runs of the same persona under the same seed produce the same player
    inputs (modulo prompt-cache state). Engine agents are left unseeded
    on purpose — we want their stochasticity in the distribution.
    """
    deps = PlayerDeps(persona=persona, recent_transcript=transcript[-6:])
    model_settings = (
        {"extra_body": {"seed": seed}} if seed is not None else None
    )
    result = await player_agent.run(
        "Write your next reply.", deps=deps, model_settings=model_settings,
    )
    return result.output.strip()


def _pick_reply_target(state: InboxState) -> Email | None:
    """Most recent NPC-sent email in the Inbox folder. The synthetic player
    replies to whatever's on top of the inbox; with multi-thread scenarios
    this naturally rotates as new beats fire on different threads.
    """
    inbox = [e for e in state.emails if e.folder == "Inbox" and e.sender != "you"]
    return inbox[-1] if inbox else None


async def run_session(
    persona: PersonaSpec,
    *,
    player_agent: Agent[PlayerDeps, str],
    max_sends: int = 25,
    seed: int | None = None,
) -> SessionRecord:
    """Drive one full session against a fresh per-call `Session`. Loops:
    pick reply target → synthesise body → record_send → process_send →
    record TickDecision → repeat until ended or max_sends.

    Caller must have boot-initialised `engine_main.SCENARIO` + `AGENTS`
    via `boot_engine_for_playtest`. Each call constructs its own session
    and runs the boot tick locally — no shared module state, so multiple
    sessions can run concurrently via `asyncio.gather`.
    """
    from npc_service.session import make_session
    assert engine_main.SCENARIO is not None and engine_main.AGENTS is not None
    session = make_session(engine_main.SCENARIO, engine_main.AGENTS)
    # Boot tick — fires pre-authored beats with empty preconditions.
    await engine_main._tick(session, player_body=None, in_response_to=None)

    decisions = []
    error: str | None = None

    for _ in range(max_sends):
        if session.state.ended:
            break
        target = _pick_reply_target(session.state)
        if target is None:
            error = "no inbox email to reply to (engine not booted?)"
            break

        try:
            body = await _synthesise_reply(
                player_agent, persona, list(session.state.emails), seed=seed,
            )
        except Exception as e:  # noqa: BLE001 — synthetic-player failure ends session
            error = f"synthetic player failure: {type(e).__name__}: {e}"
            break

        if not body.strip():
            error = "synthetic player returned empty body"
            break

        # `target.sender` is the NPC's `email_address` ("Aria <aria@...>"),
        # not the dossier id. Pass it as `recipient` and let the engine's
        # `_resolve_recipient_npc_id` fuzzy-match to the dossier — same
        # back-compat path the production frontend uses on the rare email
        # whose dossier id isn't pre-known to the client.
        req = SendRequest(
            in_reply_to=target.id,
            recipient=target.sender,
            recipient_npc_id=None,
            body=body,
        )

        try:
            rec = engine_main.record_send(session, req)
            result = await engine_main.process_send(session, rec)
        except Exception as e:  # ChatAgentFailure or anything else
            error = f"engine failure: {type(e).__name__}: {e}"
            break

        decisions.append(result.tick_decision)

    return SessionRecord(
        persona_name=persona.name,
        transcript=copy.deepcopy(session.state.emails),
        tick_decisions=decisions,
        final_state=copy.deepcopy(session.state),
        error=error,
    )


# ───────────────────────── boot helper ─────────────────────────


def boot_engine_for_playtest(
    scenario_path: str | None = None,
) -> tuple[OpenAIChatModel, OpenAIChatModel]:
    """Initialise `engine_main.SCENARIO` + `AGENTS` against the local engine
    llama (Gemma on :8080), and build a separate model handle for the eval
    llama (Qwen on :8081, served by ik_llama.cpp). Mirrors `lifespan`'s
    setup minus the FastAPI ceremony.

    Returns `(engine_model, eval_model)`. The engine model powers the SUT
    agents (chat/classifier/extractor/summary/fact-checker/director). The
    eval model powers the synthetic-player agent and every LLM judge —
    different family from the engine to mitigate preference leakage when
    judging.
    """
    settings = Settings()
    if scenario_path:
        os.environ["SCENARIO_PATH"] = scenario_path
        # Re-init settings so the new env var is picked up.
        settings = Settings()
    path = settings.scenario_path or _default_scenario_path()
    engine_main.SCENARIO = load_scenario(path)

    engine_url = settings.llama_base_url or f"http://{settings.llama_host}:{settings.llama_port}/v1"
    tokenize_url = engine_url.removesuffix("/v1") + "/tokenize"
    engine_model = OpenAIChatModel(
        "gemma-4-e4b",
        provider=OpenAIProvider(base_url=engine_url, api_key="not-needed"),
    )
    engine_main.AGENTS = build_agents(
        engine_model,
        token_counter=make_llama_token_counter(tokenize_url),
        token_budget=int(settings.llama_ctx * settings.history_compact_frac),
        history_keep_tail=settings.history_compact_keep_tail,
    )

    eval_url = settings.eval_llama_base_url or f"http://{settings.llama_host}:{settings.eval_llama_port}/v1"
    eval_model = OpenAIChatModel(
        "qwen3.6-35b-a3b",
        provider=OpenAIProvider(base_url=eval_url, api_key="not-needed"),
    )

    return engine_model, eval_model


def _default_scenario_path() -> str:
    from pathlib import Path
    return str(
        Path(__file__).resolve().parents[2] / "scenario.yaml"
    )


def probe_llama(base_url: str) -> bool:
    """True if llama-server is reachable + healthy at this URL. Caller
    iterates over the {engine, eval} pair so the error message can name
    which one is missing."""
    try:
        r = httpx.get(f"{base_url}/health", timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False
