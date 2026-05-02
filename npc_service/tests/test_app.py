"""FastAPI smoke tests + end-to-end agent/options agent tests.

`LAUNCH_LLAMA=false` skips spawning llama-server; the model is overridden with
TestModel so /action and /agent.run() flow without any external dependencies.
"""
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

import npc_service.main as main
from npc_service.models import AgentReport, DialogueOption, OptionsResponse


@pytest.fixture
def app(monkeypatch):
    """Boot the app with no llama-server. Yields the live module so tests can
    reach into globals (AGENTS, HANDLER_STATE) for overrides + assertions."""
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    with TestClient(main.app) as client:
        assert main.AGENTS  # at least one agent built
        assert main.HANDLER_STATE is not None
        yield client


def _stub_options(payload):
    """Override OPTIONS_AGENT to return `payload` (an OptionsResponse or raw
    dict that may fail validation). Use as a context manager.

    OPTIONS_AGENT uses PromptedOutput, so the fake response is JSON text that
    PromptedOutput parses client-side into the typed model.
    """
    args = payload.model_dump() if hasattr(payload, "model_dump") else payload
    return main.OPTIONS_AGENT.override(model=TestModel(custom_output_text=json.dumps(args)))


def _stub_agent(agent_id: str, payload: AgentReport | dict):
    args = payload.model_dump() if hasattr(payload, "model_dump") else payload
    return main.AGENTS[agent_id].override(model=TestModel(custom_output_args=args))


# ───────────────────────── /health, /scenario, /state ─────────────────────────


def test_health_reports_ready(app):
    r = app.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["scenario_loaded"] is True
    assert sorted(j["agents_ready"]) == ["cicada", "dove", "magpie"]


def test_scenario_omits_internal_fields(app):
    r = app.get("/scenario")
    assert r.status_code == 200
    j = r.json()
    assert "title" in j
    assert "operation" in j
    # public view: never expose authoring-only fields
    assert "world_state" not in j["operation"]
    for a in j["agents"]:
        assert "known_facts" not in a
        assert "seed_exchanges" not in a
        assert "deflection_style" not in a
        # opening_transmission IS exposed — the client renders it on boot
        assert "opening_transmission" in a


def test_state_response_shape(app):
    r = app.get("/state")
    assert r.status_code == 200
    j = r.json()
    assert "agents" in j
    for a in j["agents"]:
        assert "id" in a
        assert "last_report_action_count" in a


# ───────────────────────── /action validation ─────────────────────────


def test_action_rejects_empty_line(app):
    r = app.post("/action", json={"agent_id": "magpie", "line": "   "})
    assert r.status_code == 422


def test_action_rejects_unknown_agent(app):
    r = app.post("/action", json={"agent_id": "ghost", "line": "hello"})
    assert r.status_code == 404


def test_action_rejects_missing_agent_id(app):
    r = app.post("/action", json={"line": "hello"})
    assert r.status_code == 422


# ───────────────────────── agent persona structured output ─────────────────────────


async def test_agent_persona_emits_validated_report():
    """Direct agent.run() under TestModel — no FastAPI involved.

    Confirms the wiring `ToolOutput(AgentReport)` produces a validated
    AgentReport whose fields flow through unchanged.
    """
    from pydantic_ai import Agent, ToolOutput

    fake = AgentReport(report_text="Confidence MED, control. Standing by.")
    agent = Agent(
        TestModel(custom_output_args=fake.model_dump()),
        output_type=ToolOutput(AgentReport),
        instructions="be MAGPIE",
    )
    result = await agent.run("Status?")

    assert isinstance(result.output, AgentReport)
    assert result.output.report_text == "Confidence MED, control. Standing by."


async def test_options_agent_emits_validated_response():
    """Direct agent.run() under TestModel — confirms ToolOutput(OptionsResponse)
    parses faked tool-call args back into the typed model unchanged.
    """
    from pydantic_ai import Agent, ToolOutput

    fake = OptionsResponse(options=[
        DialogueOption(tag="Reassure", hidden_line="Hold position."),
        DialogueOption(tag="Probe", hidden_line="Walk me through the timing."),
        DialogueOption(tag="Hedge", hidden_line="Stand down for the next exchange."),
    ])
    agent = Agent(
        TestModel(custom_output_args=fake.model_dump()),
        output_type=ToolOutput(OptionsResponse),
        instructions="be the dialogue coach",
    )
    result = await agent.run("Generate options for MAGPIE.")

    assert isinstance(result.output, OptionsResponse)
    assert [o.tag for o in result.output.options] == ["Reassure", "Probe", "Hedge"]
    assert result.output.options[1].hidden_line == "Walk me through the timing."


def test_dialogue_option_tag_uniqueness_rejects_duplicates():
    with pytest.raises(ValidationError):
        OptionsResponse(options=[
            DialogueOption(tag="Probe", hidden_line="A"),
            DialogueOption(tag="probe", hidden_line="B"),
        ])


# ───────────────────────── /options ─────────────────────────


def test_options_endpoint_returns_options(app):
    """/options returns 2-5 picks from a fixture-backed OPTIONS_AGENT."""
    fake = OptionsResponse(options=[
        DialogueOption(tag="Reassure", hidden_line="Hold position."),
        DialogueOption(tag="Press", hidden_line="Are you sure that's not a tail?"),
    ])
    with _stub_options(fake):
        r = app.get("/options", params={"agent_id": "magpie"})
    assert r.status_code == 200
    j = r.json()
    assert len(j["options"]) == 2
    assert [o["tag"] for o in j["options"]] == ["Reassure", "Press"]


def test_options_endpoint_rejects_unknown_agent(app):
    r = app.get("/options", params={"agent_id": "ghost"})
    assert r.status_code == 404


def test_options_endpoint_requires_agent_id(app):
    r = app.get("/options")
    assert r.status_code == 422


def test_options_endpoint_returns_empty_when_ended(app):
    main.HANDLER_STATE.ended = True
    main.HANDLER_STATE.end_reason = "budget_exhausted"
    try:
        r = app.get("/options", params={"agent_id": "magpie"})
    finally:
        main.HANDLER_STATE.ended = False
        main.HANDLER_STATE.end_reason = None
    assert r.status_code == 200
    assert r.json() == {"options": []}


def test_options_endpoint_does_not_mutate_history(app):
    """CLAUDE.md invariant: /options is read-only over HISTORIES[agent_id]."""
    fake = OptionsResponse(options=[
        DialogueOption(tag="Reassure", hidden_line="Hold position."),
        DialogueOption(tag="Probe", hidden_line="Walk me through it."),
    ])
    before = list(main.HISTORIES["magpie"])
    with _stub_options(fake):
        r = app.get("/options", params={"agent_id": "magpie"})
    assert r.status_code == 200
    assert main.HISTORIES["magpie"] == before


def test_options_endpoint_returns_empty_on_validation_error(app):
    bad = {"options": [{"tag": "Probe", "hidden_line": "Talk to me."}]}  # min_length=2
    with _stub_options(bad):
        r = app.get("/options", params={"agent_id": "magpie"})
    assert r.status_code == 200
    assert r.json() == {"options": []}


# ───────────────────────── /action SSE channel isolation ─────────────────────────


def _parse_sse(body: str) -> list[dict]:
    """Parse an SSE response body into a list of decoded JSON payloads."""
    out = []
    for line in body.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


def test_action_streams_text_and_state_for_active_agent(app):
    fake = AgentReport(report_text="Confidence MED.")
    with _stub_agent("magpie", fake):
        r = app.post("/action", json={"agent_id": "magpie", "line": "Status?"})
    assert r.status_code == 200
    payloads = _parse_sse(r.text)
    types = [p["type"] for p in payloads]
    assert "state" in types
    assert types[-1] == "done"
    # all SSE events scoped to magpie
    for p in payloads:
        if "agent_id" in p:
            assert p["agent_id"] == "magpie"


def test_action_isolates_histories_per_agent(app):
    """A turn against MAGPIE must not append messages to CICADA's or DOVE's HISTORY."""
    seed_cicada = list(main.HISTORIES["cicada"])
    seed_dove = list(main.HISTORIES["dove"])
    seed_magpie_len = len(main.HISTORIES["magpie"])

    fake = AgentReport(report_text="copy.")
    with _stub_agent("magpie", fake):
        r = app.post("/action", json={"agent_id": "magpie", "line": "x"})
    assert r.status_code == 200

    assert main.HISTORIES["cicada"] == seed_cicada
    assert main.HISTORIES["dove"] == seed_dove
    # magpie's history grew (user prompt + tool-call response)
    assert len(main.HISTORIES["magpie"]) > seed_magpie_len


def test_action_increments_global_action_count(app):
    """One action against any channel increments the global counter by exactly 1."""
    before = main.HANDLER_STATE.action_count
    fake = AgentReport(report_text="copy.")
    with _stub_agent("cicada", fake):
        r = app.post("/action", json={"agent_id": "cicada", "line": "x"})
    assert r.status_code == 200
    assert main.HANDLER_STATE.action_count == before + 1


def test_action_updates_last_report_action_count_for_correct_agent(app):
    """A turn against CICADA records last_report_action_count for CICADA only."""
    before = main.HANDLER_STATE.agents["cicada"].last_report_action_count
    dove_before = main.HANDLER_STATE.agents["dove"].last_report_action_count
    fake = AgentReport(report_text="copy.")
    with _stub_agent("cicada", fake):
        r = app.post("/action", json={"agent_id": "cicada", "line": "x"})
    assert r.status_code == 200
    assert main.HANDLER_STATE.agents["cicada"].last_report_action_count == before + 1
    assert main.HANDLER_STATE.agents["dove"].last_report_action_count == dove_before


# ───────────────────────── /reset ─────────────────────────


def test_reset_clears_all_histories_and_state(app):
    fake = AgentReport(report_text="x")
    with _stub_agent("magpie", fake):
        app.post("/action", json={"agent_id": "magpie", "line": "x"})
    assert main.HANDLER_STATE.action_count == 1

    r = app.post("/reset")
    assert r.status_code == 200
    assert main.HANDLER_STATE.action_count == 0
    # all three histories should be back to seed length (1 scene-update each)
    for hist in main.HISTORIES.values():
        assert len(hist) == 1
