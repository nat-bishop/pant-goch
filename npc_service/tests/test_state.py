"""Unit tests for state mutation, end-condition logic, and timeline injection routing.

Pure-Python — no LLM, no FastAPI, no llama-server.
"""
import npc_service.main as main
from npc_service.main import (
    apply_report,
    check_end_conditions,
    fire_due_timeline_events,
    seed_history,
)
from npc_service.models import (
    AgentDef,
    AgentMeters,
    AgentReport,
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
    stress: int = 25,
    trust: int = 75,
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
        starting_meters=AgentMeters(stress=stress, trust=trust),
        opening_transmission="opening line",
    )


def _make_state(
    action_count: int = 0,
    max_actions: int = 18,
    agents: list[tuple[str, int, int]] | None = None,
) -> HandlerState:
    if agents is None:
        agents = [("magpie", 25, 75), ("cicada", 35, 70), ("dove", 50, 65)]
    return HandlerState(
        max_actions=max_actions,
        action_count=action_count,
        agents={
            id_: AgentState(id=id_, callsign=id_.upper(), city=id_, stress=s, trust=t)
            for id_, s, t in agents
        },
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


# ───────────────────────── apply_report ─────────────────────────


def test_apply_report_clamps_meters_to_0_100():
    state = _make_state(agents=[("magpie", 10, 95), ("cicada", 35, 70), ("dove", 50, 65)])
    apply_report(state, "magpie", AgentReport(stress_delta=-20, trust_delta=20))
    assert state.agents["magpie"].stress == 0
    assert state.agents["magpie"].trust == 100
    # other agents untouched
    assert state.agents["cicada"].stress == 35
    assert state.agents["dove"].trust == 65


def test_apply_report_updates_last_report_action_count():
    state = _make_state(action_count=5)
    apply_report(state, "cicada", AgentReport(stress_delta=2))
    assert state.agents["cicada"].last_report_action_count == 5
    # the other agents' last_report_action_count stays at default 0
    assert state.agents["magpie"].last_report_action_count == 0


def test_apply_report_isolates_per_agent_meters():
    """Mutating MAGPIE must not bleed into CICADA or DOVE."""
    state = _make_state()
    apply_report(state, "magpie", AgentReport(stress_delta=10, trust_delta=-5))
    assert state.agents["magpie"].stress == 35  # 25 + 10
    assert state.agents["magpie"].trust == 70   # 75 - 5
    assert state.agents["cicada"].stress == 35  # unchanged
    assert state.agents["dove"].trust == 65     # unchanged


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


def test_all_agents_lost_ends_game():
    state = _make_state()
    for a in state.agents.values():
        a.lost = True
    check_end_conditions(state)
    assert state.ended
    assert state.end_reason == "all_lost"


def test_partial_agents_lost_does_not_end():
    state = _make_state()
    state.agents["magpie"].lost = True
    state.agents["cicada"].lost = True
    check_end_conditions(state)
    assert not state.ended


def test_already_ended_state_is_not_overwritten():
    state = _make_state(action_count=18, max_actions=18)
    state.ended = True
    state.end_reason = "all_lost"
    check_end_conditions(state)
    # re-running end checks must not flip a settled outcome
    assert state.end_reason == "all_lost"


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
