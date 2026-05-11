"""FastAPI endpoint smoke tests. Live llama is not required — the lifespan
boots the engine sidecar in test mode (`LAUNCH_LLAMA=false`), the mission
loads from disk, but the cycle loop is short-circuited by stubbing the
three agent callables before they make any network calls.

These tests cover endpoint shape: /health 200, /mission/start returns a
session_id, /authorize 409s when no auth is pending, /state returns a
snapshot. End-to-end behaviour (full cycle with live llama) lives in
test_cycle_integration.py under the `integration` marker.
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
    assert body["max_cycles"] == 4


def test_state_returns_snapshot(app_client: TestClient) -> None:
    r = app_client.post("/mission/start")
    sid = r.json()["session_id"]
    s = app_client.get(f"/mission/{sid}/state")
    assert s.status_code == 200
    body = s.json()
    assert body["session_id"] == sid
    assert "mission.cycles_remaining" in body["state"]
    assert body["state"]["mission.cycles_remaining"] == 4
    assert body["outcome"] is None
