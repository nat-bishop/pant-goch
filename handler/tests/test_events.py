"""Event round-trip + `fold()` equivalence with the live spine mutation path.

Critical invariant: the live path (spine mutates + event records delta) and
the replay path (fold replays delta) must converge on the same state. These
tests pin that.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from handler.events import (
    DayClosed,
    DayOpened,
    EffectsApplied,
    Event,
    MissionStarted,
    OptionAuthorized,
    OptionsDrafted,
    ReactionsFired,
    fold,
)
from handler.mission_loader import load_mission
from handler.models import Option, State, make_initial_state
from handler.spine import apply_action_effects


_TINY = Path(__file__).parent / "fixtures" / "tiny_mission.yaml"


@pytest.fixture
def mission():
    return load_mission(_TINY)


def test_event_union_roundtrip(mission) -> None:
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    initial = make_initial_state(mission)
    samples: list[Event] = [
        MissionStarted(mission_id="tiny", initial_state=initial),
        DayOpened(day_index=1),
        EffectsApplied(day_index=1, delta={"mission.progress": (0, 1)}, action_id="PUSH", action_forced=False),
        ReactionsFired(day_index=1, fires=[("passive_decay", None, {"agent[ALPHA].cover": (5, 4)})]),
        DayClosed(day_index=1, delta={"mission.days_remaining": (4, 3)}),
    ]
    for ev in samples:
        dumped = adapter.dump_python(ev)
        restored = adapter.validate_python(dumped)
        assert type(restored).__name__ == type(ev).__name__


def test_fold_replay_matches_live_mutation(mission) -> None:
    """Apply effects via spine (live), record the delta as an event, then
    re-fold a fresh state with the recorded event and assert equality."""
    live = make_initial_state(mission)
    opt = Option(action="PUSH", title="t", prose="p")
    delta, _ = apply_action_effects(live, opt, mission)

    replay = make_initial_state(mission)
    fold(replay, EffectsApplied(day_index=1, delta=delta, action_id="PUSH", action_forced=False))

    assert dict(replay.qualities) == dict(live.qualities)


def test_fold_observation_events_dont_mutate(mission) -> None:
    s = make_initial_state(mission)
    snapshot = dict(s.qualities)
    fold(s, DayOpened(day_index=1))
    fold(s, OptionsDrafted(day_index=1, options=[]))
    fold(s, OptionAuthorized(
        day_index=1,
        option=Option(action="PUSH", title="t", prose="p"),
        by_handler_at=datetime.now(timezone.utc),
    ))
    assert dict(s.qualities) == snapshot
    assert s.outcome is None


def test_fold_reactions_fired_records_fire_ledger(mission) -> None:
    s = make_initial_state(mission)
    ev = ReactionsFired(
        day_index=1,
        fires=[("agent_lost", "ALPHA", {"agent[ALPHA].lost": (None, True)})],
    )
    fold(s, ev)
    assert s.get_quality("agent[ALPHA].lost") is True
    assert "agent_lost" in s.fired
    assert "agent_lost:ALPHA" in s.fired_per_agent


def test_fold_effects_applied_records_forced_action_in_fired(mission) -> None:
    """Forced Actions append their id to state.fired on authorization. Non-forced
    actions do not."""
    s = make_initial_state(mission)
    fold(s, EffectsApplied(day_index=1, delta={}, action_id="turning_point", action_forced=True))
    assert "turning_point" in s.fired

    s2 = make_initial_state(mission)
    fold(s2, EffectsApplied(day_index=1, delta={}, action_id="PUSH", action_forced=False))
    assert "PUSH" not in s2.fired


def test_fold_mission_started_reseeds_state(mission) -> None:
    seeded = make_initial_state(mission)
    fresh = State()
    fresh.qualities["spurious"] = 999
    fold(fresh, MissionStarted(mission_id=mission.id, initial_state=seeded))
    assert dict(fresh.qualities) == dict(seeded.qualities)
    assert "spurious" not in fresh.qualities
