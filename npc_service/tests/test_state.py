"""Unit tests for end-condition logic, timeline injection routing, and history seeding.

Pure-Python — no LLM, no FastAPI, no llama-server.
"""
import npc_service.main as main
from npc_service.main import (
    check_end_conditions,
    fire_due_timeline_events,
    seed_history,
)
from npc_service.models import (
    AgentDef,
    AgentState,
    HandlerState,
    Operation,
    Scenario,
    TimelineEvent,
)


def _make_agent_def(
    id: str = "magpie",
    callsign: str = "MAGPIE",
    city: str = "Riga",
) -> AgentDef:
    return AgentDef(
        id=id,
        callsign=callsign,
        name=f"{callsign}-name",
        city=city,
        role="test role",
        persona="test persona",
        deflection_style="dry",
        known_facts=["fact one"],
        opening_transmission="opening line",
    )


def _make_state(
    action_count: int = 0,
    max_actions: int = 18,
    agent_ids: list[str] | None = None,
) -> HandlerState:
    if agent_ids is None:
        agent_ids = ["magpie", "cicada", "dove"]
    return HandlerState(
        max_actions=max_actions,
        action_count=action_count,
        agents={id_: AgentState(id=id_) for id_ in agent_ids},
    )


def _make_scenario(timeline: list[TimelineEvent], max_actions: int = 18) -> Scenario:
    return Scenario(
        title="Test Op",
        setup="x",
        max_actions=max_actions,
        operation=Operation(codename="TEST", objective="x"),
        agents=[
            _make_agent_def("magpie", "MAGPIE", "Riga"),
            _make_agent_def("cicada", "CICADA", "Vienna"),
            _make_agent_def("dove", "DOVE", "Lyon"),
        ],
        timeline=timeline,
    )


# ───────────────────────── end conditions ─────────────────────────


def test_action_budget_exhaustion_ends_game():
    state = _make_state(action_count=18, max_actions=18)
    check_end_conditions(state)
    assert state.ended
    assert state.end_reason == "budget_exhausted"


def test_under_budget_does_not_end():
    state = _make_state(action_count=17, max_actions=18)
    check_end_conditions(state)
    assert not state.ended


def test_already_ended_state_is_not_overwritten():
    state = _make_state(action_count=5, max_actions=18)
    state.ended = True
    state.end_reason = "budget_exhausted"
    check_end_conditions(state)
    # re-running end checks must not flip a settled outcome
    assert state.end_reason == "budget_exhausted"


# ───────────────────────── timeline injection routing ─────────────────────────


def test_fire_due_timeline_routes_to_addressed_agents(monkeypatch):
    state = _make_state(action_count=3)
    scenario = _make_scenario([
        TimelineEvent(
            id="evt1",
            after_actions=3,
            kind="world_event",
            notification_text="x",
            injections={
                "magpie": "MAGPIE-only update",
                "cicada": "CICADA-only update",
            },
        ),
        TimelineEvent(
            id="evt2",
            after_actions=10,
            kind="flash_traffic",
            notification_text="x",
            injections={"dove": "DOVE-only update"},
        ),
        TimelineEvent(
            id="evt3",
            after_actions=15,
            kind="directive",
            notification_text="x",
            injections={"dove": "later"},
        ),
    ])
    monkeypatch.setattr(main, "HISTORIES", {a.id: [] for a in scenario.agents})

    fired = fire_due_timeline_events(state, scenario)

    assert [e.id for e in fired] == ["evt1"]
    # magpie + cicada got injections, dove did not
    assert len(main.HISTORIES["magpie"]) == 1
    assert len(main.HISTORIES["cicada"]) == 1
    assert len(main.HISTORIES["dove"]) == 0
    assert "MAGPIE-only update" in str(main.HISTORIES["magpie"][0])
    assert "evt1" in state.fired_timeline_event_ids


def test_fire_due_timeline_idempotent(monkeypatch):
    """Re-running with the same action_count must not double-fire an event."""
    state = _make_state(action_count=3)
    scenario = _make_scenario([
        TimelineEvent(id="evt1", after_actions=3, kind="world_event",
                      notification_text="x", injections={"magpie": "u"}),
        TimelineEvent(id="evt2", after_actions=10, kind="flash_traffic",
                      notification_text="x", injections={"cicada": "u"}),
        TimelineEvent(id="evt3", after_actions=15, kind="directive",
                      notification_text="x", injections={"dove": "u"}),
    ])
    monkeypatch.setattr(main, "HISTORIES", {a.id: [] for a in scenario.agents})

    fire_due_timeline_events(state, scenario)
    fire_due_timeline_events(state, scenario)

    assert len(main.HISTORIES["magpie"]) == 1  # not 2


def test_fire_due_timeline_skips_future_events(monkeypatch):
    state = _make_state(action_count=2)
    scenario = _make_scenario([
        TimelineEvent(id="evt1", after_actions=3, kind="world_event",
                      notification_text="x", injections={"magpie": "u"}),
        TimelineEvent(id="evt2", after_actions=10, kind="flash_traffic",
                      notification_text="x", injections={"cicada": "u"}),
        TimelineEvent(id="evt3", after_actions=15, kind="directive",
                      notification_text="x", injections={"dove": "u"}),
    ])
    monkeypatch.setattr(main, "HISTORIES", {a.id: [] for a in scenario.agents})

    fired = fire_due_timeline_events(state, scenario)

    assert fired == []
    assert all(len(h) == 0 for h in main.HISTORIES.values())


# ───────────────────────── seeding ─────────────────────────


def test_seed_history_returns_one_scene_update():
    agent = _make_agent_def()
    h = seed_history(agent)
    assert len(h) == 1
    # rendered as a UserPromptPart scene-update, NOT a ModelResponse
    msg_str = str(h[0])
    assert "opening line" in msg_str or "transmission" in msg_str
