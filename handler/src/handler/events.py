"""Event-sourced session log: every state change in the engine is an Event.

Two flavors:
- **Observation events** — recorded LLM outputs (SoP memo, drafted options,
  rendered dispatches, mission start). `fold` records them but does not
  mutate state.
- **Mutation events** — the actual state writes (effects applied, beats
  fired, cycle closed, mission resolved). `fold` performs the mutation
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
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from .models import MissionResolution, Option, State


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


class SopRendered(BaseModel):
    kind: Literal["sop_rendered"] = "sop_rendered"
    cycle_index: int
    text: str


class OptionsDrafted(BaseModel):
    kind: Literal["options_drafted"] = "options_drafted"
    cycle_index: int
    options: list[Option]
    moment_id: str | None = None  # set if drafter included a forced moment
    engine_fallback: bool = False  # True if engine emitted fallback options


class OptionAuthorized(BaseModel):
    kind: Literal["option_authorized"] = "option_authorized"
    cycle_index: int
    option: Option
    by_handler_at: datetime


class DispatchRendered(BaseModel):
    """The acting agent's field report (or Operations for HOLD / terminal-on-timeout)."""

    kind: Literal["dispatch_rendered"] = "dispatch_rendered"
    cycle_index: int
    acting: str  # agent id, or "operations" for HOLD / Ops-voiced dispatches
    archetype: str
    text: str


# ───────────────────────── mutation events ─────────────────────────


class EffectsApplied(BaseModel):
    """Carries the delta from `apply_archetype_effects` (and any moment
    effects merged in)."""

    kind: Literal["effects_applied"] = "effects_applied"
    cycle_index: int
    delta: Delta
    moment_id: str | None = None  # if the option satisfied a moment


class BeatsFired(BaseModel):
    """Records one or more beats that fired this cycle. Each entry carries
    its own delta plus the (beat_id, agent_id_or_None) pair. Terminal beats
    are included; the next event will be `MissionResolved`."""

    kind: Literal["beats_fired"] = "beats_fired"
    cycle_index: int
    fires: list[tuple[str, str | None, Delta]]  # (beat_id, agent_id|None, delta)


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
        SopRendered,
        OptionsDrafted,
        OptionAuthorized,
        DispatchRendered,
        EffectsApplied,
        BeatsFired,
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
        state.fired_beats.clear()
        state.fired_beats.extend(event.initial_state.fired_beats)
        state.fired_moments.clear()
        state.fired_moments.extend(event.initial_state.fired_moments)
        state.fired_per_agent.clear()
        state.fired_per_agent.extend(event.initial_state.fired_per_agent)
        state.outcome = event.initial_state.outcome
        state.one_cycle_flags = dict(event.initial_state.one_cycle_flags)
        return
    if isinstance(event, (CycleOpened, SopRendered, OptionsDrafted, OptionAuthorized, DispatchRendered)):
        return  # observation only
    if isinstance(event, EffectsApplied):
        _apply_delta(state, event.delta)
        if event.moment_id is not None and event.moment_id not in state.fired_moments:
            state.fired_moments.append(event.moment_id)
        return
    if isinstance(event, BeatsFired):
        for beat_id, agent_id, delta in event.fires:
            _apply_delta(state, delta)
            state.fired_beats.append(beat_id)
            if agent_id is not None:
                state.fired_per_agent.append(f"{beat_id}:{agent_id}")
        return
    if isinstance(event, CycleClosed):
        _apply_delta(state, event.delta)
        state.one_cycle_flags.clear()
        return
    if isinstance(event, MissionResolved):
        state.outcome = event.resolution.outcome_key
        return
    raise AssertionError(f"unhandled event kind: {type(event).__name__}")
