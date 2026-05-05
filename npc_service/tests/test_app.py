"""FastAPI smoke tests with Pydantic AI's TestModel — no llama-server, no
real LLM. Verifies wiring: lifespan boots, boot inbox emits, /open does not
advance calibration, /send rejects empty bodies, advances calibration by a
flat per-send delta, fires aria.first_calibration on the first send, and
re-fires it across multiple ticks until the classifier writes engaged_initial
across the gate (multi-turn beat).
"""
import json
from contextlib import ExitStack
from typing import Iterable

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

import npc_service.main as main
from npc_service.models import Email


CHAT_TURN_PROSE = (
    "Thank you for your reply. I have logged it as a calibration sample and look "
    "forward to our continued partnership.\n\n"
    "Warmly,\nAria\nContinuity Counterpart | ARIA-7"
)


@pytest.fixture
def client(monkeypatch):
    """Boot the app with no llama-server and override both agents with a
    TestModel that returns canned outputs.

    `Agent.override(model=...)` is a context manager — push both onto an
    ExitStack so they live for the whole test. Default classifier value is
    'deflected' so the multi-turn first_calibration re-renders by default;
    individual tests can override per-call.
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
    a communication act — the meter is exclusively driven by /send."""
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


def test_send_advances_calibration_by_flat_delta(client):
    """Calibration advances by a flat config.calibration_base per /send,
    independent of player input. Default base is 5.0."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    initial = client.get("/state").json()["calibration"]
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to start - looking forward to our partnership.",
    })))

    types = [e["type"] for e in events]
    assert "send_recorded" in types
    assert "calibration" in types
    assert "done" in types

    cal_event = next(e for e in events if e["type"] == "calibration")
    assert cal_event["calibration_delta"] == pytest.approx(5.0)
    cal_after = client.get("/state").json()["calibration"]
    assert cal_after == pytest.approx(initial + 5.0)


def test_first_send_to_aria_fires_first_calibration(client):
    """After the player's first reply (replies≥1 satisfied),
    aria.first_calibration becomes eligible and renders. Welcome itself
    has no classify_reply, so on this send the classifier doesn't run —
    engaged_initial stays at default 0.5, gate (≥0.7) is not met, and the
    multi-turn beat renders without marking complete.
    """
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to be here",
    })))
    new_email_events = [e for e in events if e["type"] == "new_email"]
    plot_emails = [
        e["email"] for e in new_email_events
        if e["email"]["beat_id"] == "aria.first_calibration"
    ]
    assert len(plot_emails) == 1
    # Multi-turn beat: gate not met → not added to fired_beat_ids yet.
    assert "aria.first_calibration" not in main.STATE.fired_beat_ids
    # Pending: still re-fireable on the next tick.
    assert any(pb.beat_id == "aria.first_calibration" for pb in main.STATE.pending_beats)


def test_thread_messages_accumulates_across_sends(client):
    """Two /sends to Aria should leave thread_messages["aria"] holding:
    welcome (ModelResponse) + two (ModelRequest, ModelResponse) pairs.
    Each send fires aria.first_calibration (multi-turn beat re-rendering
    while the classifier returns 'deflected' from the fixture)."""
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


def test_plot_beat_body_does_not_leak_email_headers(client):
    """Regression guard: with output_type=str (no JSON schema enforcement)
    the chat agent could in principle drift into emitting `Subject:` or
    `From:` headers. The base instructions forbid this. TestModel echoes
    the canned prose verbatim so this would only fail if the canned text
    itself contains forbidden tokens — guard the canned prose too. We hit
    the same code path through aria.first_calibration's plot directive
    render."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello",
    })))
    plot_emails = [
        e["email"] for e in events
        if e["type"] == "new_email" and e["email"]["beat_id"] == "aria.first_calibration"
    ]
    assert plot_emails
    body = plot_emails[0]["body"]
    for forbidden in ("Subject:", "From:", "To:", "Date:", "```"):
        assert forbidden not in body, f"plot beat body leaked {forbidden!r}: {body!r}"


def test_classifier_runs_on_reply_to_first_calibration(client):
    """When the player replies to an email that originated from a beat with
    `classify_reply` (aria.first_calibration), the classifier runs and emits
    a `classified` SSE event with the mapped score."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    # First send: replies to welcome, fires first_calibration. Welcome has no
    # classify_reply so the classifier doesn't run on this send.
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    assert not any(e["type"] == "classified" for e in events)

    # Find the first_calibration email Aria just rendered, then reply to IT.
    inbox = client.get("/inbox").json()
    fc_emails = [
        e for e in inbox["folders"]["Inbox"]
        if any(em.id == e["id"] and em.beat_id == "aria.first_calibration"
               for em in main.STATE.emails)
    ]
    assert fc_emails
    fc_email_id = fc_emails[0]["id"]

    # Second send: replies to first_calibration's email. Classifier runs.
    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": fc_email_id, "recipient": "Aria", "body": "I dont want to",
    })))
    classified = [e for e in events if e["type"] == "classified"]
    assert len(classified) == 1
    assert classified[0]["quality"] == "engaged_initial"
    assert classified[0]["value"] == "deflected"  # fixture default
    # Mapped score: deflected → 0.0.
    assert main.STATE.qualities["engaged_initial"] == 0.0


def test_first_calibration_completes_when_classifier_engaged(client, monkeypatch):
    """When the classifier returns 'engaged' (mapped to 1.0 ≥ 0.7), the
    multi-turn beat completes. _fire_beat skips the render that would have
    happened and adds first_calibration to fired_beat_ids."""
    # Override the classifier to return 'engaged' for the upcoming send.
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "starting",
    })))
    fc_email = next(
        e for e in main.STATE.emails if e.beat_id == "aria.first_calibration"
    )

    with main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("engaged")):
        list(_consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": fc_email.id, "recipient": "Aria",
            "body": "Tuesday afternoons I usually triage email, then deep work.",
        })))

    assert main.STATE.qualities["engaged_initial"] == 1.0
    # Beat completed: gate met before the next render → marked done, no
    # further first_calibration emails fired this tick.
    assert "aria.first_calibration" in main.STATE.fired_beat_ids


def test_midprogram_reply_classifies_partnership_warmth(client):
    """When the player replies to aria.midprogram, classify_reply is now
    authored on that beat — the classifier runs, writes partnership_warmth,
    and emits a `classified` SSE event with the mapped score.

    We seed a synthetic midprogram email rather than walking the chain
    end-to-end (faster and stays focused on the new wiring)."""
    # Force a midprogram email into the inbox and history.
    main.STATE.fired_beat_ids.update({"aria.first_calibration", "aria.midprogram"})
    midprogram_email = Email(
        id="email_midprog",
        beat_id="aria.midprogram",
        thread_id="aria",
        folder="Inbox",
        sender="Aria | Continuity Program <aria.7@continuity.halcyon.com>",
        recipient="n.beckett@halcyon.com",
        subject="Calibration Milestone — Counterpart Fidelity Update",
        body="Counterpart fidelity now at approximately 50%.",
        sent_at_label="just now",
    )
    main.STATE.emails.append(midprogram_email)

    # Override classifier to return 'warm' for this send.
    with main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("warm")):
        events = list(_consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": midprogram_email.id, "recipient": "Aria",
            "body": "Tuesday afternoons I usually triage email — the slow shift after lunch.",
        })))

    classified = [e for e in events if e["type"] == "classified"]
    assert len(classified) == 1
    assert classified[0]["quality"] == "partnership_warmth"
    assert classified[0]["value"] == "warm"
    assert main.STATE.qualities["partnership_warmth"] == 1.0


def test_system_midpoint_fires_via_beat_fired_chain(client):
    """system.midpoint_report is gated on `beat_fired(aria.midprogram)`.
    Force the chain by marking midprogram fired, then /send to advance the
    scheduler — system.midpoint_report should land on the same tick the
    chain reaches it."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    # Pre-seed: simulate the chain having reached midprogram.
    main.STATE.fired_beat_ids.update({
        "aria.first_calibration", "aria.midprogram",
    })
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
