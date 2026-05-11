"""Event-sourced session log: every state change in the engine is an Event.

Two flavors:
- **Observation events** — recorded LLM outputs (SoP memo, drafted options,
  rendered dispatches). `fold` records them but does not mutate state.
- **Mutation events** — the actual state writes (effects applied, beats
  fired, cycle counter decremented). `fold` performs the mutation.

Authoring discipline: NEVER remove a `kind` value, only deprecate. Adding a
new event type is `+1 class, +1 union member, +1 fold case` — old saved logs
still deserialize because their kinds remain in the union.

Phase 1 populates this module with MissionStarted, CycleOpened, SopRendered,
OptionsDrafted, OptionAuthorized, EffectsApplied, BeatsFired,
DispatchRendered, CycleClosed, MissionResolved.
"""
from __future__ import annotations

from typing import Any

from .models import State


Event = Any


def fold(state: State, event: Event) -> None:
    """Apply `event` to `state`. Phase 1 fills the dispatch table."""
    raise NotImplementedError("Handler events authored in Phase 1")
