"""Event round-trip + `fold()` equivalence with the live spine mutation path.

Critical invariant per HANDLER_SPEC_V2 §8.5: the live path (spine mutates +
event records delta) and the replay path (fold replays delta) must converge
on the same state. These tests pin that.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from handler.events import (
    CycleClosed,
    CycleOpened,
    EffectsApplied,
    Event,
    EventsFired,
    MissionStarted,
    OptionAuthorized,
    OptionsDrafted,
    fold,
)
from handler.mission_loader import load_mission
from handler.models import MissionResolution, Option, State, make_initial_state
from handler.spine import (
    apply_archetype_effects,
    fire_due_passive_events,
    tick_cycle_close,
)


_TINY = Path(__file__).parent / "fixtures" / "tiny_mission.yaml"


@pytest.fixture
def mission():
    return load_mission(_TINY)


def test_event_union_roundtrip(mission) -> None:
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    initial = make_initial_state(mission)
    samples: list[Event] = [
        MissionStarted(mission_id="tiny", initial_state=initial),
        CycleOpened(cycle_index=1),
        EffectsApplied(cycle_index=1, delta={"mission.progress": (0, 1)}),
        EventsFired(cycle_index=1, fires=[("passive_decay", None, {"agent[ALPHA].cover": (5, 4)})]),
        CycleClosed(cycle_index=1, delta={"mission.cycles_remaining": (4, 3)}),
    ]
    for ev in samples:
        dumped = adapter.dump_python(ev)
        restored = adapter.validate_python(dumped)
        # The validator picks the right branch via `kind` discriminator.
        assert type(restored).__name__ == type(ev).__name__


def test_fold_replay_matches_live_mutation(mission) -> None:
    """Apply effects via spine (live), record the delta as an event, then
    re-fold a fresh state with the recorded event and assert equality."""
    live = make_initial_state(mission)
    opt = Option(archetype="PUSH", title="t", prose="p")
    delta = apply_archetype_effects(live, opt, mission)

    replay = make_initial_state(mission)
    fold(replay, EffectsApplied(cycle_index=1, delta=delta))

    assert dict(replay.qualities) == dict(live.qualities)


def test_fold_observation_events_dont_mutate(mission) -> None:
    s = make_initial_state(mission)
    snapshot = dict(s.qualities)
    fold(s, CycleOpened(cycle_index=1))
    fold(s, OptionsDrafted(cycle_index=1, options=[]))
    fold(s, OptionAuthorized(
        cycle_index=1,
        option=Option(archetype="PUSH", title="t", prose="p"),
        by_handler_at=datetime.now(timezone.utc),
    ))
    assert dict(s.qualities) == snapshot
    assert s.outcome is None


def test_fold_events_fired_records_fire_ledger(mission) -> None:
    s = make_initial_state(mission)
    ev = EventsFired(
        cycle_index=1,
        fires=[
            ("agent_lost", "ALPHA", {"agent[ALPHA].lost": (None, True)}),
        ],
    )
    fold(s, ev)
    assert s.get_quality("agent[ALPHA].lost") is True
    assert "agent_lost" in s.fired
    assert "agent_lost:ALPHA" in s.fired_per_agent


def test_fold_mission_started_reseeds_state(mission) -> None:
    """Replay path: a session starts with a `MissionStarted` event carrying
    the initial snapshot. fold() must reseed state to exactly that snapshot
    (clearing any prior content)."""
    seeded = make_initial_state(mission)
    fresh = State()
    fresh.qualities["spurious"] = 999
    fold(fresh, MissionStarted(mission_id=mission.id, initial_state=seeded))
    assert dict(fresh.qualities) == dict(seeded.qualities)
    assert "spurious" not in fresh.qualities
