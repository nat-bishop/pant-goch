"""FastAPI smoke tests with Pydantic AI's TestModel — no llama-server, no
real LLM. Verifies wiring: lifespan boots, boot inbox emits, /open advances
the action counter, /send runs the texture scorer end-to-end, /draft branches
the three paths, and /coda emits an inbox montage.
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
    holds while the context is active. We push all four onto an ExitStack
    so they live for the whole test.
    """
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    with TestClient(main.app) as c:
        assert main.AGENTS is not None
        with ExitStack() as stack:
            stack.enter_context(main.AGENTS.email_agent.override(model=_test_model_for_email()))
            stack.enter_context(main.AGENTS.texture_agent.override(model=_test_model_for_texture(1.0, 1.0)))
            stack.enter_context(main.AGENTS.voice_fidelity_agent.override(model=_test_model_for_voice_fidelity()))
            stack.enter_context(main.AGENTS.coda_agent.override(model=_test_model_for_coda()))
            yield c


def _test_model_for_email() -> TestModel:
    return TestModel(custom_output_args={"prose": "(test rendered email body — Maya is responding warmly)"})


def _test_model_for_texture(warmth: float, honesty: float) -> TestModel:
    return TestModel(custom_output_args={"warmth": warmth, "honesty": honesty, "rationale": "(test)"})


def _test_model_for_voice_fidelity() -> TestModel:
    return TestModel(custom_output_args={
        "voice_fidelity": 0.85,
        "rationale": "kept the lowercase but tightened the ending",
    })


def _test_model_for_coda() -> TestModel:
    return TestModel(custom_output_args={
        "framing_lines": [
            "There are emails she did not write.",
            "There is a thing left in the drafts folder.",
            "Outside, it is Thursday.",
        ],
    })


# ───────────────────────── boot + endpoints ─────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["scenario_loaded"] is True


def test_scenario(client):
    r = client.get("/scenario")
    assert r.status_code == 200
    j = r.json()
    assert j["title"] == "Inbox"
    assert "premise" in j


def test_boot_inbox_has_three_emails(client):
    r = client.get("/inbox")
    assert r.status_code == 200
    j = r.json()
    inbox = j["folders"]["Inbox"]
    senders = [e["from"] for e in inbox]
    # Janet (intake), Maya (old), and the class registration auto-reminder.
    assert any("Janet" in s for s in senders)
    assert any("Maya" in s for s in senders)
    assert any("Northbridge" in s for s in senders)


def test_state_starts_at_zero(client):
    r = client.get("/state").json()
    assert r["action_count"] == 0
    assert r["max_actions"] == 18
    assert r["ended"] is False


def test_open_advances_action_counter(client):
    inbox = client.get("/inbox").json()
    janet_email = next(e for e in inbox["folders"]["Inbox"] if "Janet" in e["from"])
    r = client.post("/open", json={"email_id": janet_email["id"]})
    assert r.status_code == 200
    state = r.json()["state"]
    assert state["action_count"] == 1


def test_open_idempotent_after_first_read(client):
    inbox = client.get("/inbox").json()
    daniel_email = next((e for e in inbox["folders"]["Inbox"] if "Daniel" in e["from"]), None)
    if daniel_email is None:
        pytest.skip("daniel hasn't arrived yet at boot")
    client.post("/open", json={"email_id": daniel_email["id"]})
    state_after_first = client.get("/state").json()
    client.post("/open", json={"email_id": daniel_email["id"]})
    state_after_second = client.get("/state").json()
    assert state_after_first["action_count"] == state_after_second["action_count"]


def test_archive_advances_counter(client):
    inbox = client.get("/inbox").json()
    e = inbox["folders"]["Inbox"][0]
    initial = client.get("/state").json()["action_count"]
    r = client.post("/archive", json={"email_id": e["id"]})
    assert r.status_code == 200
    assert r.json()["state"]["action_count"] == initial + 1


def test_search_does_not_advance_counter(client):
    initial = client.get("/state").json()["action_count"]
    r = client.get("/search", params={"q": "writing group"})
    assert r.status_code == 200
    assert client.get("/state").json()["action_count"] == initial


def test_search_finds_results(client):
    r = client.get("/search", params={"q": "writing group"}).json()
    assert len(r["hits"]) >= 1
    assert any("writing group" in h["body"].lower() or "writing group" in h["subject"].lower() for h in r["hits"])


# ───────────────────────── send + texture scoring ─────────────────────────


def test_send_runs_texture_and_advances_counter(client):
    inbox = client.get("/inbox").json()
    maya_email = next(e for e in inbox["folders"]["Inbox"] if "Maya" in e["from"])
    initial = client.get("/state").json()["action_count"]

    events = list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": maya_email["beat_id"] if "beat_id" in maya_email else None,
        "recipient": "Maya Chen",
        "body": "thinking of u, im here",
    })))

    types = [e["type"] for e in events]
    assert "send_recorded" in types
    assert "texture" in types
    assert "done" in types

    state_after = client.get("/state").json()
    assert state_after["action_count"] == initial + 1


# ───────────────────────── draft mechanic — three branches ─────────────────────────


def test_draft_does_not_appear_until_player_replies_to_maya(client):
    """The draft is gated on player_replied_to: maya.first_email."""
    inbox = client.get("/inbox").json()
    drafts = inbox["folders"]["Drafts"]
    assert drafts == [], f"expected no drafts at boot, got {drafts}"


def test_draft_held_branch(client):
    _reply_to_maya_to_unlock_draft(client, "thanks for writing.")
    drafts = _get_drafts(client)
    assert len(drafts) == 1, f"expected the draft to surface; got {drafts}"
    draft = drafts[0]

    events = list(_consume_sse(
        client.stream("POST", f"/draft/{draft['id']}/decision", json={"action": "hold"})
    ))
    types = [e["type"] for e in events]
    assert "draft_held" in types
    assert "done" in types

    # Atom set, no maya.response_to_draft beat fired.
    state = client.get("/state").json()
    assert "maya" in state["thread_states"]
    # maya.response_to_draft only fires on send paths.
    inbox = client.get("/inbox").json()
    maya_emails = [e for e in inbox["folders"]["Inbox"] if "Maya" in e["from"]]
    rendered_response_to_draft = [e for e in maya_emails if e.get("beat_id") == "maya.response_to_draft"]
    assert rendered_response_to_draft == []


def test_draft_send_unchanged_branch(client):
    _reply_to_maya_to_unlock_draft(client, "hey, i'm taking care of things")
    drafts = _get_drafts(client)
    draft = drafts[0]

    events = list(_consume_sse(
        client.stream("POST", f"/draft/{draft['id']}/decision", json={"action": "send_unchanged"})
    ))
    types = [e["type"] for e in events]
    assert "draft_sent" in types

    # The send appears in Sent folder.
    inbox = client.get("/inbox").json()
    sent = inbox["folders"]["Sent"]
    assert any("on behalf of Tess" in e["from"] for e in sent)


def test_draft_edit_and_send_runs_voice_fidelity(client):
    _reply_to_maya_to_unlock_draft(client, "thanks for the message")
    drafts = _get_drafts(client)
    draft = drafts[0]

    events = list(_consume_sse(client.stream(
        "POST", f"/draft/{draft['id']}/decision",
        json={"action": "edit_and_send", "body": "hey - im handling some things for tess. she's safe."},
    )))
    types = [e["type"] for e in events]
    assert "voice_fidelity" in types, f"expected voice_fidelity event; got {types}"
    vf_event = next(e for e in events if e["type"] == "voice_fidelity")
    assert 0.0 <= vf_event["voice_fidelity"] <= 1.0


# ───────────────────────── coda ─────────────────────────


def test_coda_emits_fragments_and_framing(client):
    # Drive minimal state so the coda has something to montage.
    _reply_to_maya_to_unlock_draft(client, "im here")
    # /coda is gated on STATE.ended — flip it directly. End-condition triggering
    # itself is covered by test_spine.py's check_end_conditions tests.
    main.STATE.ended = True
    main.STATE.end_reason = "test"
    events = list(_consume_sse(client.stream("GET", "/coda")))
    types = [e["type"] for e in events]
    assert "fragment" in types
    assert "framing" in types
    assert "done" in types


def test_coda_refused_before_end_conditions(client):
    """Stray clients can't force the coda mid-game."""
    _reply_to_maya_to_unlock_draft(client, "hi")
    assert main.STATE.ended is False
    events = list(_consume_sse(client.stream("GET", "/coda")))
    types = [e["type"] for e in events]
    assert "error" in types
    assert "fragment" not in types


# ───────────────────────── helpers ─────────────────────────


def _reply_to_maya_to_unlock_draft(client, body: str) -> None:
    """Send one reply to Maya's boot email so the draft becomes visible."""
    inbox = client.get("/inbox").json()
    maya_email = next(e for e in inbox["folders"]["Inbox"] if "Maya" in e["from"])
    list(_consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": maya_email.get("beat_id") or "maya.first_email",
        "recipient": "Maya Chen",
        "body": body,
    })))


def _get_drafts(client) -> list[dict]:
    return client.get("/inbox").json()["folders"]["Drafts"]


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
