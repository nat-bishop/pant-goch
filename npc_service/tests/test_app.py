"""FastAPI smoke tests + end-to-end captor/options agent tests.

`LAUNCH_LLAMA=false` skips spawning llama-server; the model is overridden with
TestModel so /action and /agent.run() flow without any external dependencies.
"""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

import npc_service.main as main
from npc_service.models import CaptorResponse, DialogueOption, OptionsResponse


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


def _stub_options(payload):
    """Override OPTIONS_AGENT to return `payload` (an OptionsResponse or raw
    dict that may fail validation). Use as a context manager."""
    args = payload.model_dump() if hasattr(payload, "model_dump") else payload
    return main.OPTIONS_AGENT.override(model=TestModel(custom_output_args=args))


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


async def test_options_agent_emits_validated_response():
    """Direct agent.run() under TestModel — confirms ToolOutput(OptionsResponse)
    parses a faked tool-call payload back into the typed model unchanged.
    """
    from pydantic_ai import Agent, ToolOutput

    fake = OptionsResponse(options=[
        DialogueOption(tag="Empathize", hidden_line="Talk to me."),
        DialogueOption(tag="Probe", hidden_line="Anyone hurt in there?"),
        DialogueOption(tag="Stall", hidden_line="Give me a minute on that."),
    ])
    agent = Agent(
        TestModel(custom_output_args=fake.model_dump()),
        output_type=ToolOutput(OptionsResponse),
        instructions="be the options coach",
    )
    result = await agent.run("Generate options.")

    assert isinstance(result.output, OptionsResponse)
    assert [o.tag for o in result.output.options] == ["Empathize", "Probe", "Stall"]
    assert result.output.options[1].hidden_line == "Anyone hurt in there?"


def test_dialogue_option_tag_uniqueness_rejects_duplicates():
    """OptionsResponse._tags_distinct dedups case-insensitively. Locks in the
    invariant the system prompt advertises and the validator enforces."""
    with pytest.raises(ValidationError):
        OptionsResponse(options=[
            DialogueOption(tag="Empathize", hidden_line="A"),
            DialogueOption(tag="empathize", hidden_line="B"),
        ])


def test_options_endpoint_returns_options(app):
    """/options returns 2-5 picks from a fixture-backed OPTIONS_AGENT."""
    fake = OptionsResponse(options=[
        DialogueOption(tag="Empathize", hidden_line="Talk to me."),
        DialogueOption(tag="Press", hidden_line="You know what no costs you."),
    ])
    with _stub_options(fake):
        r = app.get("/options")
    assert r.status_code == 200
    j = r.json()
    assert len(j["options"]) == 2
    tags = [o["tag"] for o in j["options"]]
    assert tags == ["Empathize", "Press"]
    assert all("hidden_line" in o for o in j["options"])


def test_options_endpoint_returns_empty_when_ended(app):
    """When the game has ended, /options short-circuits without invoking the
    agent — keeps the GameOver overlay path simple."""
    main.GAME_STATE.ended = True
    main.GAME_STATE.end_reason = "captor_surrendered"
    try:
        r = app.get("/options")
    finally:
        main.GAME_STATE.ended = False
        main.GAME_STATE.end_reason = None
    assert r.status_code == 200
    assert r.json() == {"options": []}


def test_options_endpoint_does_not_mutate_history(app):
    """CLAUDE.md L53 invariant: /options is read-only over HISTORY. Locks it in
    so a future refactor that switches to result.all_messages() trips a test."""
    fake = OptionsResponse(options=[
        DialogueOption(tag="Empathize", hidden_line="Talk to me."),
        DialogueOption(tag="Probe", hidden_line="Anyone hurt?"),
    ])
    before = list(main.HISTORY)
    with _stub_options(fake):
        r = app.get("/options")
    assert r.status_code == 200
    assert main.HISTORY == before


def test_options_endpoint_returns_empty_on_validation_error(app):
    """If the model produces something that fails OptionsResponse validation,
    the endpoint degrades to {"options": []} rather than 5xx — Godot then falls
    into its existing 'No options available.' path."""
    # Single option violates min_length=2 → ValidationError on the agent run.
    bad = {"options": [{"tag": "Empathize", "hidden_line": "Talk to me."}]}
    with _stub_options(bad):
        r = app.get("/options")
    assert r.status_code == 200
    assert r.json() == {"options": []}
