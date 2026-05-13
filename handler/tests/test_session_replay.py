"""Replay path coverage: `session.state_at(i)`, JSONL persistor → load → fold
reproduces the live state. The narrower fold/live convergence is pinned in
test_events; this suite drives the disk round-trip end-to-end.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from handler.events import (
    CycleClosed,
    CycleOpened,
    EffectsApplied,
    EventsFired,
    LogEntryAppended,
    MissionStarted,
    OptionAuthorized,
    OptionsDrafted,
)
from handler.mission_loader import load_mission
from handler.models import Option, State, make_initial_state
from handler.session import (
    load_event_log,
    make_jsonl_persistor,
    make_session,
    replay_into_session,
)
from handler.spine import apply_archetype_effects, tick_cycle_close


_TINY = Path(__file__).parent / "fixtures" / "tiny_mission.yaml"


@pytest.fixture
def mission():
    return load_mission(_TINY)


@pytest.fixture
def agents_stub(mission):
    # `Session` carries an AgentBundle but replay never invokes the agents.
    # A SimpleNamespace stand-in keeps the dataclass happy without booting the
    # OpenAIChatModel + pydantic-ai stack.
    from types import SimpleNamespace

    return SimpleNamespace(drafter=None, log_writer=None, mission=mission)


def _emit_cycle_one(session, mission) -> None:
    """Drive a single cycle's worth of events into `session` via the live path.

    Mirrors what `cycle.run_cycle` would emit, minus the SoP / drafter / renderer
    streaming. Each event carries the delta produced by the spine functions, so
    fold() can reproduce state from the log alone.
    """
    # Snapshot the initial state. model_copy() is shallow — qualities is a
    # dict and would alias with session.state.qualities, polluting replay.
    # `cycle.start_mission` solves this by constructing a fresh State; here a
    # deep copy is equivalent and clearer at the test boundary.
    initial = session.state.model_copy(deep=True)
    session.apply(MissionStarted(mission_id=mission.id, initial_state=initial))
    session.apply(CycleOpened(cycle_index=1))
    opt = Option(
        archetype="SACRIFICE_BY_AGENT",
        agent="ALPHA",
        title="t",
        prose="p",
    )
    session.apply(OptionsDrafted(cycle_index=1, options=[opt]))
    session.apply(
        OptionAuthorized(
            cycle_index=1,
            option=opt,
            by_handler_at=datetime.now(timezone.utc),
        )
    )
    delta, _branches = apply_archetype_effects(session.state, opt, mission)
    session.apply(EffectsApplied(cycle_index=1, delta=delta))
    close_delta = tick_cycle_close(session.state)
    session.apply(CycleClosed(cycle_index=1, delta=close_delta))


# ───────────────────────── state_at ─────────────────────────


def test_state_at_reproduces_intermediate_snapshots(mission, agents_stub) -> None:
    session = make_session(mission, agents_stub)
    _emit_cycle_one(session, mission)

    # state_at(0) is the initial state (no events folded).
    s0 = session.state_at(0)
    assert s0.qualities == make_initial_state(mission).qualities

    # state_at after MissionStarted + CycleOpened — same as initial.
    s2 = session.state_at(2)
    assert s2.qualities == make_initial_state(mission).qualities

    # state_at(len) matches the live state.
    s_end = session.state_at(len(session.event_log))
    assert dict(s_end.qualities) == dict(session.state.qualities)


def test_state_at_rejects_out_of_range(mission, agents_stub) -> None:
    session = make_session(mission, agents_stub)
    _emit_cycle_one(session, mission)
    with pytest.raises(IndexError):
        session.state_at(-1)
    with pytest.raises(IndexError):
        session.state_at(len(session.event_log) + 1)


# ───────────────────────── JSONL round-trip ─────────────────────────


def test_jsonl_persist_then_replay_matches_live(
    tmp_path: Path, mission, agents_stub
) -> None:
    """Drive cycle 1 through a session with a JSONL persistor, then load the
    file into a fresh session via `replay_into_session` and assert state
    equivalence. The disk round-trip is the canonical save/load path."""
    log_path = tmp_path / "session.jsonl"
    live = make_session(mission, agents_stub, persist=make_jsonl_persistor(log_path))
    _emit_cycle_one(live, mission)

    # JSONL file should now have one line per emitted event.
    raw = log_path.read_text(encoding="utf-8").splitlines()
    assert len(raw) == len(live.event_log)

    # Load + replay into a fresh session.
    events = load_event_log(log_path)
    assert len(events) == len(live.event_log)

    replayed = make_session(mission, agents_stub)
    replay_into_session(replayed, events)

    assert dict(replayed.state.qualities) == dict(live.state.qualities)
    assert replayed.state.fired == live.state.fired
    assert replayed.state.fired_per_agent == live.state.fired_per_agent


def test_replay_into_session_skips_persist_hook(
    tmp_path: Path, mission, agents_stub
) -> None:
    """`replay_into_session` must not re-fire the persistor — otherwise the
    JSONL doubles on every boot. We assert by reusing the same path for the
    replay target and confirming file size unchanged."""
    log_path = tmp_path / "session.jsonl"
    live = make_session(mission, agents_stub, persist=make_jsonl_persistor(log_path))
    _emit_cycle_one(live, mission)

    before = log_path.read_text(encoding="utf-8")

    fresh = make_session(mission, agents_stub, persist=make_jsonl_persistor(log_path))
    replay_into_session(fresh, load_event_log(log_path))

    after = log_path.read_text(encoding="utf-8")
    assert before == after, "replay_into_session leaked into the persistor"
