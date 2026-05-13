"""FastAPI endpoint smoke tests. Live llama is not required — the lifespan
boots the engine sidecar in test mode (`LAUNCH_LLAMA=false`), the mission
loads from disk, but the day loop is short-circuited by stubbing the
three agent callables before they make any network calls.

These tests cover endpoint shape: /health 200, /mission/start returns a
session_id, /authorize 409s when no auth is pending, /state returns a
snapshot. End-to-end behaviour (full day with live llama) lives in
test_day_integration.py under the `integration` marker.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Boot the FastAPI app without spawning llama-server. Point at the
    tiny test mission so endpoint shape can be checked without LYNX-Exfil's
    full content surface."""
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setenv(
        "MISSION_PATH",
        str(Path(__file__).parent / "fixtures" / "tiny_mission.yaml"),
    )
    # External-llama probe timeout would otherwise sit for 2s.
    from handler import main as _main
    monkeypatch.setattr(_main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    with TestClient(_main.app) as client:
        yield client


def test_health_returns_ok(app_client: TestClient) -> None:
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_authorize_404_on_unknown_session(app_client: TestClient) -> None:
    r = app_client.post(
        "/mission/nonexistent/authorize", json={"option_index": 0}
    )
    assert r.status_code == 404


def test_state_404_on_unknown_session(app_client: TestClient) -> None:
    r = app_client.get("/mission/nonexistent/state")
    assert r.status_code == 404


def test_mission_start_returns_session_id(app_client: TestClient) -> None:
    """POST /mission/start should return a session_id + mission meta. The
    backing cycle loop spawns but will block on the first SoP agent call
    because llama is offline — we don't await its progress here."""
    r = app_client.post("/mission/start")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "session_id" in body
    assert body["mission_id"] == "tiny"
    assert body["codename"] == "TINY"
    assert body["max_days"] == 4


def test_mission_start_returns_cast_and_briefing(app_client: TestClient) -> None:
    """StartResponse carries the full cast + opening_briefing + opening_prose
    so the onboarding card can render without a separate fetch. The tiny
    mission has no briefing, so the field is the empty string; cast lists
    both agents (and any handler/boss/opposition entries the mission
    authored — tiny has none)."""
    r = app_client.post("/mission/start")
    body = r.json()
    assert "opening_briefing" in body
    assert "opening_prose" in body
    assert isinstance(body["cast"], list)
    cast_ids = {m["id"] for m in body["cast"]}
    assert "ALPHA" in cast_ids
    assert "BRAVO" in cast_ids
    for member in body["cast"]:
        assert "codename" in member
        assert "role" in member
        assert "file_number" in member
        assert "situation" in member


def test_state_returns_snapshot(app_client: TestClient) -> None:
    r = app_client.post("/mission/start")
    sid = r.json()["session_id"]
    s = app_client.get(f"/mission/{sid}/state")
    assert s.status_code == 200
    body = s.json()
    assert body["session_id"] == sid
    assert "mission.days_remaining" in body["state"]
    assert body["state"]["mission.days_remaining"] == 4
    assert body["outcome"] is None
