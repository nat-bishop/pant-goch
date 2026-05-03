"""FastAPI smoke tests with Pydantic AI's TestModel — no llama-server, no
real LLM. Verifies wiring: lifespan boots, boot inbox emits, /open is free,
/send advances the clock and runs the texture scorer end-to-end, /wait and
/sleep advance the clock and surface scheduled beats.
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
    assert j["title"] == "Threads"
    assert j["max_hours"] == 96.0
    assert "premise" in j


def test_state_starts_at_zero(client):
    r = client.get("/state").json()
    assert r["game_time_hours"] == 0.0
    assert r["max_hours"] == 96.0
    assert r["ended"] is False


def test_boot_inbox_has_two_emails(client):
    r = client.get("/inbox")
    assert r.status_code == 200
    j = r.json()
    inbox = j["folders"]["Inbox"]
    senders = [e["from"] for e in inbox]
    # The inciting bounce + Sam's opening message.
    assert any("mailer-daemon" in s for s in senders), senders
    assert any("Sam" in s for s in senders), senders
    assert len(inbox) == 2


# ───────────────────────── free verbs (no clock advance) ─────────────────────────


def test_open_is_free(client):
    inbox = client.get("/inbox").json()
    sam_email = next(e for e in inbox["folders"]["Inbox"] if "Sam" in e["from"])
    r = client.post("/open", json={"email_id": sam_email["id"]})
    assert r.status_code == 200
    state = r.json()["state"]
    assert state["game_time_hours"] == 0.0


def test_archive_is_free(client):
    inbox = client.get("/inbox").json()
    bounce_email = next(e for e in inbox["folders"]["Inbox"] if "mailer-daemon" in e["from"])
    initial = client.get("/state").json()["game_time_hours"]
    r = client.post("/archive", json={"email_id": bounce_email["id"]})
    assert r.status_code == 200
    assert r.json()["state"]["game_time_hours"] == initial


# ───────────────────────── send + texture scoring ─────────────────────────


def test_send_advances_clock_and_runs_texture(client):
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": "sam.opening",
        "recipient": "Sam Park",
        "body": "yeah no idea who that is - can you check?",
    })))

    types = [e["type"] for e in events]
    assert "send_recorded" in types
    assert "texture" in types
    assert "done" in types

    state_after = client.get("/state").json()
    # Clock should have advanced by SEND_HOURS = 0.25.
    assert state_after["game_time_hours"] == pytest.approx(0.25)


def test_send_to_sam_schedules_a_reply(client):
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": "sam.opening",
        "recipient": "Sam Park",
        "body": "yeah no idea who that is",
    })))
    # Pending now includes sam.first_response, scheduled at send_time + 1.25h.
    pending_ids = {pb.beat_id for pb in main.STATE.pending_beats}
    assert "sam.first_response" in pending_ids


# ───────────────────────── wait + sleep ─────────────────────────


def test_wait_advances_clock(client):
    initial = client.get("/state").json()["game_time_hours"]
    r = client.post("/wait")
    assert r.status_code == 200
    assert r.json()["state"]["game_time_hours"] == pytest.approx(initial + 2.0)


def test_sleep_advances_clock(client):
    initial = client.get("/state").json()["game_time_hours"]
    r = client.post("/sleep")
    assert r.status_code == 200
    assert r.json()["state"]["game_time_hours"] == pytest.approx(initial + 8.0)


def test_wait_surfaces_scheduled_npc_reply(client):
    """Send to Sam, then wait long enough — Sam's reply lands."""
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": "sam.opening",
        "recipient": "Sam Park",
        "body": "what's going on",
    })))
    # Sam's latency midpoint is 1.25h; one /wait (2h) is enough.
    r = client.post("/wait")
    new_emails = r.json()["new_emails"]
    sam_reply = next((e for e in new_emails if e.get("beat_id") == "sam.first_response"), None)
    assert sam_reply is not None, f"expected sam.first_response in new_emails; got {new_emails}"


def test_wait_surfaces_time_gated_anchored_beat(client):
    """Alex's checking_in beat is time-gated at 2h; one /wait should fire it."""
    r = client.post("/wait")
    new_emails = r.json()["new_emails"]
    alex_email = next((e for e in new_emails if e.get("beat_id") == "alex.checking_in"), None)
    assert alex_email is not None, f"expected alex.checking_in in new_emails; got {new_emails}"


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
