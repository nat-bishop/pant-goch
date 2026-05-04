"""FastAPI smoke tests with Pydantic AI's TestModel — no llama-server, no
real LLM. Verifies wiring: lifespan boots, boot inbox emits, /open no longer
advances calibration, /send rejects empty bodies, runs the texture scorer
end-to-end, fires an immediate chat-turn fallback for the first send, and
threshold-delays PlotBeats so they land on a later /send.
"""
import json
from contextlib import ExitStack
from typing import Iterable

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

import npc_service.main as main


CHAT_TURN_PROSE = (
    "Thank you for your reply. I have logged it as a calibration sample and look "
    "forward to our continued partnership.\n\n"
    "Warmly,\nAria\nContinuity Counterpart | ARIA-7"
)


@pytest.fixture
def client(monkeypatch):
    """Boot the app with no llama-server and override every agent with a
    TestModel that returns canned outputs.

    `Agent.override(model=...)` is a context manager — push both onto an
    ExitStack so they live for the whole test.
    """
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    with TestClient(main.app) as c:
        assert main.AGENTS is not None
        with ExitStack() as stack:
            stack.enter_context(main.AGENTS.chat_agent.override(model=_test_model_for_chat()))
            stack.enter_context(main.AGENTS.texture_agent.override(model=_test_model_for_texture(1.0)))
            yield c


def _test_model_for_chat(text: str = CHAT_TURN_PROSE) -> TestModel:
    return TestModel(custom_output_text=text)


def _test_model_for_texture(warmth: float) -> TestModel:
    return TestModel(custom_output_args={"warmth": warmth, "rationale": "(test)"})


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


def test_state_starts_at_zero(client):
    r = client.get("/state").json()
    assert r["calibration"] == 0.0
    assert r["ended"] is False


def test_boot_inbox_has_one_email(client):
    r = client.get("/inbox")
    assert r.status_code == 200
    j = r.json()
    inbox = j["folders"]["Inbox"]
    assert len(inbox) == 1
    assert any("Aria" in e["from"] for e in inbox), [e["from"] for e in inbox]


def test_boot_seeds_thread_messages(client):
    """The anchored aria.welcome fires at boot. Its body is appended to
    thread_messages["aria"] as a ModelResponse(TextPart) so the chat agent's
    next run sees Aria's welcome as her prior turn."""
    history = main.STATE.thread_messages.get("aria", [])
    assert len(history) == 1
    assert isinstance(history[0], ModelResponse)
    assert any(isinstance(p, TextPart) for p in history[0].parts)


# ───────────────────────── /open ─────────────────────────


def test_open_does_not_advance_calibration(client):
    """/open is no longer a mutating endpoint for calibration. Reading isn't
    a communication act — the meter is exclusively driven by texture-scored
    sends."""
    inbox = client.get("/inbox").json()
    welcome = inbox["folders"]["Inbox"][0]
    initial = client.get("/state").json()["calibration"]
    r = client.post("/open", json={"email_id": welcome["id"]})
    assert r.status_code == 200
    cal_after = r.json()["state"]["calibration"]
    assert cal_after == initial


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


def test_send_advances_calibration_by_warmth_derived_delta(client):
    """TestModel returns warmth=1.0; calibration_base=2.0, coef=1.5
    → delta = 2.0 + 1.5·1.0 = 3.5."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    initial = client.get("/state").json()["calibration"]
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to start - looking forward to our partnership.",
    })))

    types = [e["type"] for e in events]
    assert "send_recorded" in types
    assert "texture" in types
    assert "done" in types

    cal_after = client.get("/state").json()["calibration"]
    assert cal_after == pytest.approx(initial + 3.5)


def test_first_send_to_aria_returns_chat_turn_immediately(client):
    """No PlotBeat is eligible after one player reply (midprogram needs ≥3,
    terminals need ≥4 + cal ≥ 80). The chat-turn fallback fires inside the
    same /send response — no `wait for second send` for the first reply.
    """
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to be here",
    })))
    new_email_events = [e for e in events if e["type"] == "new_email"]
    assert len(new_email_events) >= 1
    # Chat-turn email: beat_id is None, sender is Aria's dossier address.
    chat_emails = [e["email"] for e in new_email_events if e["email"]["beat_id"] is None]
    assert len(chat_emails) == 1
    assert "aria" in chat_emails[0]["from"].lower()


def test_thread_messages_accumulates_across_sends(client):
    """Two /sends to Aria should leave thread_messages["aria"] holding:
    welcome (ModelResponse) + two (ModelRequest, ModelResponse) pairs."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello there",
    })))
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "another note",
    })))

    history = main.STATE.thread_messages["aria"]
    # 1 ModelResponse (welcome) + 2 × (ModelRequest, ModelResponse) = 5 entries.
    assert len(history) == 5
    assert isinstance(history[0], ModelResponse)
    assert isinstance(history[1], ModelRequest)
    assert isinstance(history[2], ModelResponse)
    assert isinstance(history[3], ModelRequest)
    assert isinstance(history[4], ModelResponse)


def test_chat_turn_body_does_not_leak_email_headers(client):
    """Regression guard: with output_type=str (no JSON schema enforcement)
    the chat agent could in principle drift into emitting `Subject:` or
    `From:` headers. The base instructions forbid this. TestModel echoes
    the canned prose verbatim so this would only fail if the canned text
    itself contains forbidden tokens — guard the canned prose too."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello",
    })))
    chat_emails = [
        e["email"] for e in events
        if e["type"] == "new_email" and e["email"]["beat_id"] is None
    ]
    assert chat_emails
    body = chat_emails[0]["body"]
    for forbidden in ("Subject:", "From:", "To:", "Date:", "```"):
        assert forbidden not in body, f"chat-turn body leaked {forbidden!r}: {body!r}"


def test_midprogram_plotbeat_lands_after_threshold_send(client):
    """midprogram requires ≥3 player replies in aria thread. After the
    third send, find_eligible_plot_beat returns midprogram and
    schedule_thread_reply queues it with a calibration-threshold runtime
    predicate. The threshold sits above the post-send calibration value,
    so midprogram does NOT fire on its scheduling tick. A subsequent /send
    crosses the threshold and the beat lands.
    """
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    # Three sends to satisfy midprogram's ≥3-reply gate.
    for body in ["one", "two", "three"]:
        list(_consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": welcome["id"], "recipient": "Aria", "body": body,
        })))
    # After the third send: midprogram should be queued but not fired.
    pending_ids = {pb.beat_id for pb in main.STATE.pending_beats}
    assert "aria.midprogram" in pending_ids
    assert "aria.midprogram" not in main.STATE.fired_beat_ids

    # A fourth /send pushes calibration past the threshold; midprogram fires
    # on this tick as the chat-agent reply with author directive injected.
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "fourth",
    })))
    new_beat_ids = {
        e["email"]["beat_id"] for e in events if e["type"] == "new_email"
    }
    assert "aria.midprogram" in new_beat_ids
    assert "aria.midprogram" in main.STATE.fired_beat_ids


def test_send_threshold_anchored_beat_fires_via_calibration_cross(client):
    """system.midpoint_report is gated on calibration ≥ 40. Crank calibration
    just below the threshold, /send once (delta 3.5) → AnchoredBeat fires
    on the post-send tick."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    main.STATE.qualities["calibration"] = 39.0
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "ok",
    })))
    new_beat_ids = {
        e["email"]["beat_id"] for e in events if e["type"] == "new_email"
    }
    assert "system.midpoint_report" in new_beat_ids


# ───────────────────────── helpers ─────────────────────────


def _consume_sse(response_cm) -> Iterable[dict]:
    with response_cm as r:
        assert r.status_code == 200, r.text
        buffer = ""
        for chunk in r.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, _, buffer = buffer.partition("\n\n")
                for line in block.split("\n"):
                    if line.startswith("data: "):
                        try:
                            yield json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
