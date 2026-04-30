"""FastAPI smoke tests + end-to-end captor agent test.

`LAUNCH_LLAMA=false` skips spawning llama-server; the model is overridden with
TestModel so /action and /agent.run() flow without any external dependencies.
"""
import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

import npc_service.main as main
from npc_service.models import CaptorResponse


@pytest.fixture
def app(monkeypatch):
    """Boot the app with no llama-server. Yields the live module so tests can
    reach into globals (CAPTOR_AGENT, GAME_STATE) for overrides + assertions."""
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    # Probe is a 2s wait_healthy when llama isn't there — cut it short for tests.
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    with TestClient(main.app) as client:
        assert main.CAPTOR_AGENT is not None
        assert main.GAME_STATE is not None
        yield client


def test_health_reports_ready(app):
    r = app.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["scenario_loaded"] is True
    assert j["agent_ready"] is True


def test_scenario_omits_internal_fields(app):
    r = app.get("/scenario")
    assert r.status_code == 200
    j = r.json()
    assert "title" in j
    assert "captor" in j
    # public view: never expose authoring-only fields
    assert "known_facts" not in j["captor"]
    assert "seed_exchanges" not in j["captor"]
    assert "starting_meters" not in j["captor"]


def test_state_omits_meters(app):
    r = app.get("/state")
    assert r.status_code == 200
    j = r.json()
    assert "stress" not in j
    assert "trust" not in j
    assert "desperation" not in j
    assert "hostages" in j


def test_action_rejects_empty_line(app):
    r = app.post("/action", json={"line": "   "})
    assert r.status_code == 422


async def test_captor_agent_emits_validated_response():
    """Direct agent.run() under TestModel — no FastAPI involved.

    Confirms the wiring `ToolOutput(CaptorResponse)` produces a validated
    CaptorResponse whose fields flow through unchanged.
    """
    from pydantic_ai import Agent, ToolOutput

    fake = CaptorResponse(
        response_text="I'm tired of this.",
        stress_delta=-1,
        trust_delta=2,
        desperation_delta=0,
    )
    agent = Agent(
        TestModel(custom_output_args=fake.model_dump()),
        output_type=ToolOutput(CaptorResponse),
        instructions="be the captor",
    )
    result = await agent.run("How are you holding up?")

    assert isinstance(result.output, CaptorResponse)
    assert result.output.response_text == "I'm tired of this."
    assert result.output.stress_delta == -1
    assert result.output.trust_delta == 2
