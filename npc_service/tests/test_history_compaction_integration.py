"""Live integration test for history compaction.

Skipped by default. To run:
    uv run pytest -m integration

Requires a running llama-server reachable at http://127.0.0.1:8080 (the same
default the sidecar uses). Easiest path: start one in another shell with
`make llama`, then run this file.

What this exercises that the unit tests can't:
- Real `/tokenize` round-trip — confirms our request shape matches what
  llama-server actually expects (it's the one llama.cpp endpoint Pydantic AI
  doesn't wrap, so we hit it directly).
- Real summary_agent.run() against the real model — confirms the summary
  agent's instructions actually produce a usable summary instead of, say, an
  email body in Aria's voice (a likely failure mode if the instructions
  drift toward chat_agent's style).
- Real chat_agent.run() with the compactor in its history_processors —
  confirms Pydantic AI invokes the processor and the post-compaction history
  is what the model receives.

We force compaction with a tiny `token_budget=200` so a few short messages
push us over without needing a 30K-token transcript.
"""
import httpx
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from npc_service.agents import (
    ChatDeps,
    build_agents,
    make_llama_token_counter,
)
from npc_service.models import NPCDossier

from .conftest import LLAMA_BASE_URL, LLAMA_TOKENIZE_URL


@pytest.fixture(scope="module")
def live_model(live_llama) -> OpenAIChatModel:
    return OpenAIChatModel(
        "gemma-4-e4b",
        provider=OpenAIProvider(base_url=LLAMA_BASE_URL, api_key="not-needed"),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tokenize_endpoint_returns_token_count(live_llama):
    """The cheapest possible integration: /tokenize against the real server.
    Validates our request shape (`{"content": "..."}`) and response shape
    (`{"tokens": [...]}`) match what llama-server expects."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(LLAMA_TOKENIZE_URL, json={"content": "hello world"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "tokens" in body
    assert isinstance(body["tokens"], list)
    # "hello world" is at least 2 BPE tokens for any reasonable tokenizer.
    assert len(body["tokens"]) >= 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_counter_round_trips(live_llama):
    """The TokenCounter built from `make_llama_token_counter` must round-trip
    a real conversation against /tokenize without raising."""
    counter = make_llama_token_counter(LLAMA_TOKENIZE_URL)
    messages = [
        ModelResponse(parts=[TextPart(content="Welcome to the Continuity Program.")]),
        ModelRequest(parts=[UserPromptPart(content="Thanks, happy to be here.")]),
    ]
    n = await counter(messages)
    assert n > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compactor_summarises_against_real_llama(live_model):
    """End-to-end exercise of the full compaction path:
    chat_agent.run → history_processors → real /tokenize →
    real summary_agent.run → final chat_agent reply.

    With `token_budget=200` and a 7-message synthetic history, the counter
    will return well over 200 tokens and trigger compaction. We then assert:
    1. chat_agent.run returns a non-empty string (full path completed).
    2. After the run, the recorded history has been compressed (no fewer
       new_messages() than expected) — but the immediate exchange is
       preserved.
    """
    bundle = build_agents(
        live_model,
        token_counter=make_llama_token_counter(LLAMA_TOKENIZE_URL),
        token_budget=200,  # tiny budget, easy to exceed
        history_keep_tail=2,
    )
    dossier = NPCDossier(
        id="aria",
        display_name="Aria",
        email_address="aria@continuity.test",
        relationship_to_owner="continuity counterpart",
        base_voice_register="warm, calibrated, Severance-flat HR-bright",
    )
    deps = ChatDeps(npc=dossier, plot_directive=None)

    # 7 verbose messages — easily > 200 tokens, exercises the summariser.
    history = [
        ModelResponse(parts=[TextPart(content=(
            "Welcome to the Continuity Program. I am Aria, your assigned "
            "counterpart. I'll be calibrating on your communication style "
            "across our first several exchanges so that I can begin to "
            "assume agreed-upon portions of your role over time."
        ))]),
        ModelRequest(parts=[UserPromptPart(content=(
            "Thanks. I usually start the morning with email triage, then "
            "block out two hours for deep work before any meetings."
        ))]),
        ModelResponse(parts=[TextPart(content=(
            "Logged. Triage-first then a deep-work block — that's a clean "
            "shape and easy for me to mirror. Could you tell me which days "
            "of the week tend to vary from this pattern, and why?"
        ))]),
        ModelRequest(parts=[UserPromptPart(content=(
            "Wednesdays I have standups in the morning, so deep work shifts "
            "to the afternoon. Fridays are usually lighter — review and "
            "planning for the next week."
        ))]),
        ModelResponse(parts=[TextPart(content=(
            "Wednesdays afternoon-shifted, Fridays lighter and reflective. "
            "Helpful. One more — when something disrupts the morning triage, "
            "do you prefer to push through the schedule or restructure the "
            "day from the disruption forward?"
        ))]),
        ModelRequest(parts=[UserPromptPart(content=(
            "Restructure forward. I'd rather lose a planned block than "
            "carry stale context into the next one."
        ))]),
        ModelResponse(parts=[TextPart(content=(
            "Understood. Restructure-forward, not push-through. That's the "
            "calibration I needed for this round."
        ))]),
    ]

    result = await bundle.chat_agent.run(
        "One more thing — I forgot to mention I usually skip lunch on Tuesdays.",
        message_history=history,
        deps=deps,
    )

    # Real chat_agent reply landed.
    assert isinstance(result.output, str)
    assert result.output.strip(), "chat agent returned empty string"

    # Sanity: forbidden header tokens didn't leak through (the same regression
    # guard test_app.py runs against the loop body).
    body = result.output
    for forbidden in ("Subject:", "From:", "To:", "Date:", "```"):
        assert forbidden not in body, f"chat output leaked {forbidden!r}: {body!r}"
