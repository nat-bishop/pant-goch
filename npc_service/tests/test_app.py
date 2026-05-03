"""FastAPI smoke tests with Pydantic AI's TestModel — no llama-server, no
real LLM. Verifies wiring: lifespan boots, boot inbox emits, /open advances
calibration, /send rejects empty bodies and runs the texture scorer
end-to-end, and consecutive verbs surface scheduled NPC replies.
"""
import json
from contextlib import ExitStack
from typing import Iterable

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

import npc_service.main as main


@pytest.fixture
def client(monkeypatch):
    """Boot the app with no llama-server and override every agent with a
    TestModel that returns canned structured outputs.

    `Agent.override(model=...)` is a context manager — the override only
    holds while the context is active. We push both onto an ExitStack
    so they live for the whole test.
    """
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    with TestClient(main.app) as c:
        assert main.AGENTS is not None
        with ExitStack() as stack:
            stack.enter_context(main.AGENTS.email_agent.override(model=_test_model_for_email()))
            stack.enter_context(main.AGENTS.texture_agent.override(model=_test_model_for_texture(1.0)))
            yield c


def _test_model_for_email() -> TestModel:
    return TestModel(custom_output_args={"prose": "(test rendered email body)"})


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
    assert "max_hours" not in j  # time mechanic dropped


def test_state_starts_at_zero(client):
    r = client.get("/state").json()
    assert r["calibration"] == 0.0
    assert r["ended"] is False
    assert "game_time_hours" not in r  # time mechanic dropped


def test_boot_inbox_has_one_email(client):
    r = client.get("/inbox")
    assert r.status_code == 200
    j = r.json()
    inbox = j["folders"]["Inbox"]
    assert len(inbox) == 1
    assert any("Aria" in e["from"] for e in inbox), [e["from"] for e in inbox]
    assert "Archive" not in j["folders"]


# ───────────────────────── /open advances calibration ─────────────────────────


def test_open_advances_calibration(client):
    inbox = client.get("/inbox").json()
    welcome = inbox["folders"]["Inbox"][0]
    initial = client.get("/state").json()["calibration"]
    r = client.post("/open", json={"email_id": welcome["id"]})
    assert r.status_code == 200
    cal_after = r.json()["state"]["calibration"]
    assert cal_after == pytest.approx(initial + 1.0)  # OPEN_CALIBRATION_DELTA default


def test_open_marks_email_read(client):
    inbox = client.get("/inbox").json()
    welcome = inbox["folders"]["Inbox"][0]
    client.post("/open", json={"email_id": welcome["id"]})
    inbox_after = client.get("/inbox").json()
    welcome_after = inbox_after["folders"]["Inbox"][0]
    assert welcome_after["is_read"] is True


# ───────────────────────── /send ─────────────────────────


def test_send_rejects_empty_body(client):
    r = client.post("/send", json={
        "in_reply_to": "aria.welcome",
        "recipient": "Aria",
        "body": "   ",
    })
    assert r.status_code == 400


def test_send_advances_calibration_and_runs_texture(client):
    initial = client.get("/state").json()["calibration"]
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": "aria.welcome",
        "recipient": "Aria",
        "body": "happy to start - looking forward to our partnership.",
    })))

    types = [e["type"] for e in events]
    assert "send_recorded" in types
    assert "texture" in types
    assert "done" in types

    cal_after = client.get("/state").json()["calibration"]
    # Send should advance by SEND_CALIBRATION_DELTA = 5.0.
    assert cal_after == pytest.approx(initial + 5.0)


def test_send_to_aria_schedules_a_reply(client):
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": "aria.welcome",
        "recipient": "Aria",
        "body": "happy to be here",
    })))
    pending_ids = {pb.beat_id for pb in main.STATE.pending_beats}
    assert "aria.first_response" in pending_ids


def test_aria_reply_lands_after_second_send(client):
    """Single /send queues Aria's reply with calibration threshold above the
    post-send value (delay > SEND_DELTA-side); a second /send ticks
    calibration past the threshold and the reply lands."""
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": "aria.welcome",
        "recipient": "Aria",
        "body": "happy to be here",
    })))
    # First send: pending should still hold aria.first_response.
    assert any(pb.beat_id == "aria.first_response" for pb in main.STATE.pending_beats)

    # Second send: calibration ticks past the threshold; reply fires.
    events2 = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": "aria.welcome",
        "recipient": "Aria",
        "body": "another note for you",
    })))
    new_email_events = [e for e in events2 if e["type"] == "new_email"]
    fired_beat_ids = {e["email"]["beat_id"] for e in new_email_events}
    assert "aria.first_response" in fired_beat_ids


def test_open_surfaces_calibration_gated_anchored_beat(client):
    """system.midpoint_report is gated on calibration >= 40. A series of
    /opens (or /sends) ticks calibration past the threshold; the next
    scheduler tick fires the beat."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    # Crank calibration directly to just below the threshold so the test
    # doesn't have to click 39 times.
    main.STATE.qualities["calibration"] = 39.5
    r = client.post("/open", json={"email_id": welcome["id"]})
    new_emails = r.json()["new_emails"]
    fired_beat_ids = {e["beat_id"] for e in new_emails}
    assert "system.midpoint_report" in fired_beat_ids


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
