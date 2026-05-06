"""Live end-to-end integration tests for /send and the classifier.

Skipped by default. To run:
    uv run pytest -m integration

Requires a running llama-server on http://127.0.0.1:8080. Easiest path:
`make llama` in another shell.

These tests boot the full FastAPI app (with `LAUNCH_LLAMA=false` so we use
the externally-managed llama), then hit `/send` through TestClient and let
the real classifier_agent + chat_agent run. They're slow (~5-15s per send
on Gemma 4 E4B) but they're the only thing that catches:
- prompt drift (classifier returning off-spec values, chat agent leaking
  email headers)
- llama-server endpoint contract changes (/v1/chat/completions, /tokenize)
- Pydantic AI tool-call format regressions

Unit tests with TestModel can't catch any of these.
"""
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import npc_service.main as main
from npc_service.agents import build_agents, build_classifier_user_prompt

from .conftest import LLAMA_BASE_URL, consume_sse


@pytest.fixture
def live_client(monkeypatch, live_llama):
    """Boot the app pointing at a real llama-server (no agent overrides).
    Lifespan still skips spawning a llama process — we expect one running
    externally — but every agent.run() goes against the real model."""
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 5.0)
    with TestClient(main.app) as c:
        assert main.AGENTS is not None
        yield c


@pytest.fixture(scope="module")
def live_classifier(live_llama):
    """A classifier_agent backed by the real model, built without going
    through main.AGENTS so the lifespan fixture isn't required."""
    model = OpenAIChatModel(
        "gemma-4-e4b",
        provider=OpenAIProvider(base_url=LLAMA_BASE_URL, api_key="not-needed"),
    )
    return build_agents(model).classifier_agent  # no compactor needed for one classifier run


@pytest.mark.integration
def test_live_first_send_fires_loop(live_client):
    """Reply to the welcome email with a real /send. The loop beat must fire,
    Aria's body must be non-empty and free of forbidden header tokens."""
    welcome = live_client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(live_client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": (
            "Happy to be here. I usually start the morning with email "
            "triage, then block out two hours for deep work."
        ),
    })))

    # Player send recorded.
    assert events[0]["type"] == "send_recorded"

    # Aria's loop reply landed.
    new_emails = [e for e in events if e["type"] == "new_email"]
    loop_emails = [
        e["email"] for e in new_emails
        if e["email"]["beat_id"] == "aria.loop"
    ]
    assert len(loop_emails) == 1, f"expected exactly one loop email, got events: {events}"

    body = loop_emails[0]["body"]
    assert body.strip(), "loop email body is empty"
    for forbidden in ("Subject:", "From:", "To:", "Date:", "```"):
        assert forbidden not in body, f"loop body leaked {forbidden!r}: {body!r}"

    # Stream terminated cleanly.
    assert events[-1]["type"] == "done"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply_body, expected",
    [
        # Substantive answer — must classify 'engaged'. If Gemma flips this
        # the prompt has drifted and the warm-arc gate will mis-trigger.
        (
            "Tuesday mornings I do email triage from 9 to 10, then deep work "
            "until lunch. Afternoons are usually meetings and code review, "
            "and I try to wrap up by 5 with a short planning block.",
            "engaged",
        ),
        # One-word dismissive — must classify 'deflected'. Catches the
        # Gemma-on-Gemma sycophancy hazard where everything trends 'engaged'.
        ("sure", "deflected"),
    ],
    ids=["substantive→engaged", "dismissive→deflected"],
)
async def test_live_classifier_picks_correct_value(live_classifier, reply_body, expected):
    from npc_service.models import ClassifyReply

    axes = [
        ClassifyReply(
            quality="engagement",
            question="Is the player engaging substantively or deflecting?",
            allowed_values=["engaged", "deflected"],
            map={"engaged": 1.0, "deflected": -1.0},
        )
    ]
    prompt = build_classifier_user_prompt(
        recipient_blurb="Aria — continuity counterpart",
        prior_email_body="Could you tell me how you typically structure a Tuesday?",
        reply_body=reply_body,
        axes=axes,
    )

    result = await live_classifier.run(prompt)
    assert len(result.output) == 1, f"expected 1 axis result, got {result.output!r}"
    axis = result.output[0]
    assert axis.value in ("engaged", "deflected"), (
        f"classifier returned off-spec: {axis.value!r}"
    )
    assert axis.value == expected, (
        f"reply {reply_body!r} classified as {axis.value!r}, "
        f"rationale: {axis.rationale!r}"
    )
