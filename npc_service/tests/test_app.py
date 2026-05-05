"""FastAPI smoke tests with Pydantic AI's TestModel — no llama-server, no
real LLM. Verifies wiring: lifespan boots, the welcome anchored beat lands,
/open is read-only, /send rejects empty bodies and runs the classifier on
beats that author classify_reply, the multi-turn loop re-fires across sends,
engagement accumulates, and ends_session terminals flip state.ended.
"""
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

import npc_service.main as main

from .conftest import consume_sse


CHAT_TURN_PROSE = (
    "Thank you for your reply. I have logged it as a calibration sample and look "
    "forward to our continued partnership.\n\n"
    "Warmly,\nAria\nContinuity Counterpart | ARIA-7"
)


@pytest.fixture
def client(monkeypatch):
    """Boot the app with no llama-server and override both agents with a
    TestModel that returns canned outputs.

    Default classifier value is 'deflected' so engagement starts moving
    backwards by default; individual tests override per-call when they need
    'engaged' to push past the warm-ending threshold.
    """
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    with TestClient(main.app) as c:
        assert main.AGENTS is not None
        with ExitStack() as stack:
            stack.enter_context(main.AGENTS.chat_agent.override(model=_test_model_for_chat()))
            stack.enter_context(main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("deflected")))
            yield c


def _test_model_for_chat(text: str = CHAT_TURN_PROSE) -> TestModel:
    return TestModel(custom_output_text=text)


def _test_model_for_classifier(value: str) -> TestModel:
    return TestModel(custom_output_args={"value": value, "rationale": "(test)"})


# ───────────────────────── boot + read-only endpoints ─────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["scenario_loaded"] is True


def test_scenario(client):
    r = client.get("/scenario")
    assert r.status_code == 200
    j = r.json()
    assert j["title"] == "Continuity Program"
    assert "premise" in j


def test_state_starts_clean(client):
    r = client.get("/state").json()
    assert r["ended"] is False
    assert r["end_reason"] is None


def test_boot_inbox_has_one_email(client):
    r = client.get("/inbox")
    assert r.status_code == 200
    j = r.json()
    inbox = j["folders"]["Inbox"]
    assert len(inbox) == 1
    assert any("Aria" in e["from"] for e in inbox), [e["from"] for e in inbox]


def test_boot_seeds_thread_messages(client):
    """The anchored aria.welcome fires at boot. Its body lands in
    thread_messages["aria"] as a ModelResponse(TextPart) so the chat agent's
    next run sees Aria's welcome as her prior turn."""
    history = main.STATE.thread_messages.get("aria", [])
    assert len(history) == 1
    assert isinstance(history[0], ModelResponse)
    assert any(isinstance(p, TextPart) for p in history[0].parts)


# ───────────────────────── /open ─────────────────────────


def test_open_is_read_only(client):
    """/open marks the email read but mutates no qualities — engagement
    stays at default after /open."""
    inbox = client.get("/inbox").json()
    welcome = inbox["folders"]["Inbox"][0]
    initial_engagement = main.STATE.qualities["engagement"]
    r = client.post("/open", json={"email_id": welcome["id"]})
    assert r.status_code == 200
    assert main.STATE.qualities["engagement"] == initial_engagement


def test_open_marks_email_read(client):
    inbox = client.get("/inbox").json()
    welcome = inbox["folders"]["Inbox"][0]
    client.post("/open", json={"email_id": welcome["id"]})
    inbox_after = client.get("/inbox").json()
    welcome_after = inbox_after["folders"]["Inbox"][0]
    assert welcome_after["is_read"] is True


# ───────────────────────── /send ─────────────────────────


def test_send_rejects_empty_body(client):
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    r = client.post("/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "   ",
    })
    assert r.status_code == 400


def test_first_send_fires_loop(client):
    """After the first reply (replies≥1 satisfied), aria.loop becomes
    eligible and renders. Welcome has no classify_reply so the classifier
    doesn't run on this send."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to be here",
    })))
    new_email_events = [e for e in events if e["type"] == "new_email"]
    loop_emails = [
        e["email"] for e in new_email_events
        if e["email"]["beat_id"] == "aria.loop"
    ]
    assert len(loop_emails) == 1
    # Multi-turn beat: never enters fired_beat_ids on its own.
    assert "aria.loop" not in main.STATE.fired_beat_ids
    # Welcome has no classify_reply → no classification on this send.
    assert not any(e["type"] == "classified" for e in events)


def test_loop_re_fires_across_sends(client):
    """Two /sends to Aria. The loop is multi_turn=True, so it re-fires every
    tick. thread_messages accumulates: welcome (ModelResponse) + two
    (ModelRequest, ModelResponse) pairs = 5 entries."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello there",
    })))
    # Reply to the loop's email this time so the classifier runs.
    loop_email = next(
        e for e in main.STATE.emails if e.beat_id == "aria.loop"
    )
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": loop_email.id, "recipient": "Aria", "body": "another note",
    })))

    # Two loop emails should now exist.
    loop_emails = [e for e in main.STATE.emails if e.beat_id == "aria.loop"]
    assert len(loop_emails) == 2

    history = main.STATE.thread_messages["aria"]
    assert len(history) == 5
    assert isinstance(history[0], ModelResponse)
    assert isinstance(history[1], ModelRequest)
    assert isinstance(history[2], ModelResponse)
    assert isinstance(history[3], ModelRequest)
    assert isinstance(history[4], ModelResponse)


def test_loop_body_does_not_leak_email_headers(client):
    """Regression guard: with output_type=str (no JSON schema enforcement)
    the chat agent could in principle drift into emitting `Subject:` or
    `From:` headers. The base instructions forbid this."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello",
    })))
    loop_emails = [
        e["email"] for e in events
        if e["type"] == "new_email" and e["email"]["beat_id"] == "aria.loop"
    ]
    assert loop_emails
    body = loop_emails[0]["body"]
    for forbidden in ("Subject:", "From:", "To:", "Date:", "```"):
        assert forbidden not in body, f"loop body leaked {forbidden!r}: {body!r}"


def test_classifier_runs_on_reply_to_loop_and_accumulates(client):
    """Replying to a loop email triggers classify_reply. The classifier
    writes to `engagement` with `accumulate=True`, so each engaged reply
    pushes the quality up by 1.0."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    # First send: replies to welcome → loop fires; welcome has no classifier.
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    assert main.STATE.qualities["engagement"] == 0.0
    loop_email = next(e for e in main.STATE.emails if e.beat_id == "aria.loop")

    # Override classifier to 'engaged' for the next two sends; engagement
    # should accumulate to 2.0.
    with main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("engaged")):
        events = list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": loop_email.id, "recipient": "Aria",
            "body": "Tuesday afternoons I usually triage email, then deep work.",
        })))
        classified = [e for e in events if e["type"] == "classified"]
        assert len(classified) == 1
        assert classified[0]["quality"] == "engagement"
        assert classified[0]["value"] == "engaged"

        next_loop_email = next(
            e for e in reversed(main.STATE.emails) if e.beat_id == "aria.loop"
        )
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": next_loop_email.id, "recipient": "Aria",
            "body": "and Wednesdays are deep work mornings.",
        })))
    assert main.STATE.qualities["engagement"] == pytest.approx(2.0)


def test_complete_warm_fires_and_ends_session(client):
    """Engineer the warm ending: 5 replies on the aria thread, classifier
    returning 'engaged' so engagement accumulates past the 4.0 threshold.
    On the 5th send (replies≥5 AND engagement≥4.0), complete_warm fires
    and state.ended flips."""
    with main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("engaged")):
        _play_five_replies(client)

    assert main.STATE.ended is True
    assert main.STATE.end_reason == "aria.complete_warm"
    warm_emails = [e for e in main.STATE.emails if e.beat_id == "aria.complete_warm"]
    assert len(warm_emails) == 1


def test_complete_standard_fires_when_engagement_low(client):
    """5 replies but classifier always 'deflected' → engagement floors at 0.0,
    complete_warm precondition fails, complete_standard wins via mutex."""
    # Default fixture classifier is 'deflected' — no override needed.
    _play_five_replies(client)

    assert main.STATE.ended is True
    assert main.STATE.end_reason == "aria.complete_standard"
    standard_emails = [
        e for e in main.STATE.emails if e.beat_id == "aria.complete_standard"
    ]
    assert len(standard_emails) == 1
    assert main.STATE.qualities["engagement"] == 0.0  # floor-clamped


def test_send_emits_structured_error_when_chat_agent_fails(client, monkeypatch):
    """When chat_agent.run raises, _send_stream must emit a structured SSE
    `error` event (matching `web/src/api.ts` SseEvent.error) followed by a
    final state + done — never a torn HTTP stream. The player's email is
    already persisted so the client sees their send accepted before the
    error.

    Replaces the real chat_agent.run with one that raises RuntimeError so
    the test exercises the ChatAgentFailure → SSE error path without
    needing to break the TestModel internals.
    """
    async def _raise(*args, **kwargs):
        raise RuntimeError("simulated llama failure")

    monkeypatch.setattr(main.AGENTS.chat_agent, "run", _raise)
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))

    # Player's send is recorded first.
    assert events[0]["type"] == "send_recorded"

    # Exactly one error event with a player-friendly message.
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert "Aria" in errors[0]["message"]

    # Stream still terminates cleanly with state + done.
    assert events[-2]["type"] == "state"
    assert events[-1]["type"] == "done"

    # No NPC reply was emitted (rendering failed before any new_email event).
    assert not any(e["type"] == "new_email" for e in events)

    # Player email IS in state — the send was accepted before the failure.
    assert any(e.body == "hi" and e.sender == "you" for e in main.STATE.emails)


def test_send_returns_409_after_session_ends(client):
    """Once a terminal beat fires, /send rejects further activity."""
    _play_five_replies(client)

    assert main.STATE.ended is True
    last_npc = next(e for e in reversed(main.STATE.emails) if e.sender != "you")
    r = client.post("/send", json={
        "in_reply_to": last_npc.id, "recipient": "Aria", "body": "more?",
    })
    assert r.status_code == 409


# ───────────────────────── helpers ─────────────────────────


def _play_five_replies(client, body: str = "k") -> None:
    """Send five replies on the aria thread, chaining each to the most
    recent NPC email so the classifier always lands on the loop beat (or
    its terminal). Caller controls the classifier via Agent.override."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    last_in_reply_to = welcome["id"]
    for _ in range(5):
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": last_in_reply_to, "recipient": "Aria", "body": body,
        })))
        last_npc = next(
            (e for e in reversed(main.STATE.emails) if e.sender != "you"),
            None,
        )
        if last_npc is not None:
            last_in_reply_to = last_npc.id
