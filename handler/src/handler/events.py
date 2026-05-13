"""Event-sourced session log: every state change in the engine is a log
entry (different concept from `models.Event`, the mission-authored event).

Two flavors of log entry:
- **Observation events** — recorded LLM outputs (SoP memo, drafted options,
  rendered dispatches, mission start). `fold` records them but does not
  mutate state.
- **Mutation events** — the actual state writes (effects applied, passive
  events fired, cycle closed, mission resolved). `fold` performs the mutation
  by replaying the recorded delta onto state.

The live path: `spine.py` functions mutate state in place AND return a
delta. `cycle.py` builds an event from that delta and calls `session.apply`,
which folds the (same) delta onto state. During replay (`session.state_at`,
`replay_into_session`), the live mutation never happens — only the fold
path runs. Tests assert both paths converge to the same state.

Authoring discipline: NEVER remove a `kind` value, only deprecate. Adding
a new event type is `+1 class, +1 union member, +1 fold case` — old saved
logs still deserialize.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union, assert_never

from pydantic import BaseModel, Field

from .models import MissionResolution, Option, State
from .spine import per_agent_key


# A delta is a map of quality_name -> (before, after).
Delta = dict[str, tuple[Any, Any]]


# ───────────────────────── observation events ─────────────────────────


class MissionStarted(BaseModel):
    """Recorded at session boot. The initial state snapshot is stored so
    replay can reconstruct the universal-quality injection without
    re-instantiating from the mission YAML."""

    kind: Literal["mission_started"] = "mission_started"
    mission_id: str
    initial_state: State


class CycleOpened(BaseModel):
    kind: Literal["cycle_opened"] = "cycle_opened"
    cycle_index: int  # 1-based; CYC 01 / 06 maps to cycle_index=1


class OptionsDrafted(BaseModel):
    kind: Literal["options_drafted"] = "options_drafted"
    cycle_index: int
    options: list[Option]
    # Id of the forced event surfaced to the drafter this cycle, if any.
    forced_event_id: str | None = None
    engine_fallback: bool = False  # True if engine emitted fallback options


class OptionAuthorized(BaseModel):
    kind: Literal["option_authorized"] = "option_authorized"
    cycle_index: int
    option: Option
    by_handler_at: datetime


class LogEntryAppended(BaseModel):
    """The cycle's case-file entry, written by the log writer in Operations
    voice after effects apply. Always Ops voice — voice multiplexing is gone
    per HANDLER_SPEC_V2 §8.2."""

    kind: Literal["log_entry_appended"] = "log_entry_appended"
    cycle_index: int
    archetype: str
    text: str


# ───────────────────────── mutation events ─────────────────────────


class EffectsApplied(BaseModel):
    """Carries the delta from `apply_archetype_effects` (and any forced-event
    effects merged in)."""

    kind: Literal["effects_applied"] = "effects_applied"
    cycle_index: int
    delta: Delta
    # Id of the forced event the chosen option satisfied this cycle, if any.
    forced_event_id: str | None = None


class EventsFired(BaseModel):
    """Records one or more passive events that fired this cycle. Each entry
    carries its own delta plus the (event_id, agent_id_or_None) pair.
    Terminal events are included; the next log entry will be `MissionResolved`.
    """

    kind: Literal["events_fired"] = "events_fired"
    cycle_index: int
    fires: list[tuple[str, str | None, Delta]]  # (event_id, agent_id|None, delta)


class CycleClosed(BaseModel):
    """End-of-cycle housekeeping: cycles_remaining decrement + one-cycle
    flag clear. Carries the close delta."""

    kind: Literal["cycle_closed"] = "cycle_closed"
    cycle_index: int
    delta: Delta


class MissionResolved(BaseModel):
    kind: Literal["mission_resolved"] = "mission_resolved"
    cycle_index: int
    resolution: MissionResolution


# ───────────────────────── union + fold ─────────────────────────


Event = Annotated[
    Union[
        MissionStarted,
        CycleOpened,
        OptionsDrafted,
        OptionAuthorized,
        LogEntryAppended,
        EffectsApplied,
        EventsFired,
        CycleClosed,
        MissionResolved,
    ],
    Field(discriminator="kind"),
]


def _apply_delta(state: State, delta: Delta) -> None:
    """Write each `(before, after)` entry's `after` into state qualities."""
    for quality, (_before, after) in delta.items():
        state.set_quality(quality, after)


def fold(state: State, event: Event) -> None:
    """Apply `event` to `state` IN PLACE. Observation events are no-ops on
    state; mutation events replay their delta. The discriminator ensures
    every branch is checked statically — adding a new variant without
    updating fold raises at runtime.
    """
    if isinstance(event, MissionStarted):
        # Replay path: reseed state from the recorded initial snapshot.
        state.qualities.clear()
        state.qualities.update(event.initial_state.qualities)
        state.fired.clear()
        state.fired.extend(event.initial_state.fired)
        state.fired_per_agent.clear()
        state.fired_per_agent.extend(event.initial_state.fired_per_agent)
        state.outcome = event.initial_state.outcome
        state.one_cycle_flags = dict(event.initial_state.one_cycle_flags)
        return
    if isinstance(event, (CycleOpened, OptionsDrafted, OptionAuthorized, LogEntryAppended)):
        return  # observation only
    if isinstance(event, EffectsApplied):
        _apply_delta(state, event.delta)
        if event.forced_event_id is not None:
            state.fired.append(event.forced_event_id)
        return
    if isinstance(event, EventsFired):
        for event_id, agent_id, delta in event.fires:
            _apply_delta(state, delta)
            state.fired.append(event_id)
            if agent_id is not None:
                state.fired_per_agent.append(per_agent_key(event_id, agent_id))
        return
    if isinstance(event, CycleClosed):
        _apply_delta(state, event.delta)
        state.one_cycle_flags.clear()
        return
    if isinstance(event, MissionResolved):
        state.outcome = event.resolution.outcome_key
        return
    assert_never(event)
