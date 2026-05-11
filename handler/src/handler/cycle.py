"""The cycle orchestration. One function (`run_cycle`) ties together the
deterministic spine and the three LLM agents per HANDLER_SPEC_V2 §3:

  i.   SoP synthesizer       (agents.sop)
  ii.  Drafter               (agents.drafter)
       ── player picks ──    (await session.await_authorization)
  iii. Effects + beats       (spine.apply_* + spine.fire_due_beats)
  iv.  Dispatch renderer     (agents.renderer)
       ── cycle close ──     (spine.tick_cycle_close)

This module is the ONLY orchestration code. `spine.py` is pure; `cycle.py`
is the only place where LLM calls and state mutation interleave.

State mutation always flows through `session.apply(event)`. The spine
functions mutate session.state in place — that's the live path — but
the emitted event carries the same delta so `fold()` reproduces the
state during replay.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol

import logfire

from .events import (
    BeatsFired,
    CycleClosed,
    CycleOpened,
    DispatchRendered,
    EffectsApplied,
    MissionResolved,
    MissionStarted,
    OptionAuthorized,
    OptionsDrafted,
    SopRendered,
)
from .models import (
    Mission,
    MissionResolution,
    Option,
    State,
    make_initial_state,
)
from .session import Session
from .spine import (
    apply_archetype_effects,
    apply_moment_effects,
    due_moments,
    fire_due_beats,
    option_progresses_mission,
    tick_cycle_close,
    valid_archetype_combinations,
)


log = logging.getLogger(__name__)


# ───────────────────────── agent + emitter protocols ─────────────────────────


class SopFn(Protocol):
    async def __call__(self, session: Session, cycle_index: int) -> str:
        """Render the State-of-Play memo. Returns the full text after streaming."""


class DrafterFn(Protocol):
    async def __call__(
        self,
        session: Session,
        cycle_index: int,
        sop_text: str,
        valid_catalog: list[tuple[str, str | None, dict[str, str]]],
        forced_moment_id: str | None,
    ) -> tuple[list[Option], bool]:
        """Return (options, engine_fallback). On engine_fallback=True the
        engine emitted a deterministic fallback set after retries exhausted."""


class RendererFn(Protocol):
    async def __call__(
        self,
        session: Session,
        cycle_index: int,
        option: Option,
        delta: dict,
        acting: str,
    ) -> str:
        """Render the field dispatch in the acting agent's voice. Returns
        the full text after streaming."""


class Emitter(Protocol):
    """Sink for SSE-side events (separate from session.apply, which writes
    the canonical log). Provided by main.py so cycle.py can stream paced
    word events to the connected client without knowing about HTTP."""

    async def __call__(self, payload: dict) -> None: ...


# ───────────────────────── start / await primitives ─────────────────────────


async def start_mission(
    session: Session,
    mission: Mission,
) -> None:
    """Record `MissionStarted` (the first event in every session log).
    Captures the initial state as a standalone snapshot (not a model_copy
    of session.state) so fold() can re-seed during replay without aliasing
    the live qualities dict.
    """
    session.state = make_initial_state(mission)
    session.apply(MissionStarted(mission_id=mission.id, initial_state=make_initial_state(mission)))


# ───────────────────────── cycle orchestration ─────────────────────────


async def run_cycle(
    session: Session,
    cycle_index: int,
    *,
    sop: SopFn,
    drafter: DrafterFn,
    renderer: RendererFn,
    emit: Emitter,
    await_authorization: Callable[[], Awaitable[Option]],
) -> bool:
    """Run one cycle. Returns True if the mission is still active, False on
    terminal (caller stops iterating).

    `await_authorization` is supplied by main.py: it waits for the player's
    POST /authorize, validates the picked option against `options`, and
    returns the resolved `Option` (which carries the prose from drafter).
    """
    mission = session.mission
    state = session.state
    if state.outcome is not None:
        return False  # mission already resolved

    with logfire.span(
        "run_cycle",
        cycle_index=cycle_index,
        mission=mission.id,
    ):
        # ── cycle opened ──
        session.apply(CycleOpened(cycle_index=cycle_index))
        await emit({"event": "cycle_opened", "cycle": cycle_index, "state": state.qualities})

        # ── i. SoP synthesizer ──
        sop_text = await sop(session, cycle_index)
        session.apply(SopRendered(cycle_index=cycle_index, text=sop_text))

        # ── ii. Drafter (with optional forced moment) ──
        catalog = valid_archetype_combinations(state, mission)
        forced = due_moments(state, mission)
        forced_id = forced[0].id if forced else None

        options, engine_fallback = await drafter(
            session, cycle_index, sop_text, catalog, forced_id,
        )
        session.apply(
            OptionsDrafted(
                cycle_index=cycle_index,
                options=options,
                moment_id=forced_id,
                engine_fallback=engine_fallback,
            )
        )
        await emit({"event": "awaiting_auth"})

        # ── player picks ──
        chosen = await await_authorization()
        session.apply(
            OptionAuthorized(
                cycle_index=cycle_index,
                option=chosen,
                by_handler_at=datetime.now(timezone.utc),
            )
        )

        # ── iii. Effects + beats ──
        delta = apply_archetype_effects(state, chosen, mission)
        moment_id: str | None = None
        if chosen.satisfies_moment is not None and forced is not None:
            for m in forced:
                if m.id == chosen.satisfies_moment:
                    mdelta = apply_moment_effects(state, m, chosen)
                    delta = _merge_delta(delta, mdelta)
                    moment_id = m.id
                    break
        session.apply(EffectsApplied(cycle_index=cycle_index, delta=delta, moment_id=moment_id))
        await emit({"event": "effects_applied", "delta": _serialize_delta(delta)})

        made_progress = option_progresses_mission(delta, mission)
        fires, agg = fire_due_beats(state, mission, made_progress)
        if fires:
            session.apply(
                BeatsFired(
                    cycle_index=cycle_index,
                    fires=[(b.id, agent_id, d) for (b, agent_id, d) in fires],
                )
            )
            await emit({
                "event": "beats_fired",
                "beats": [b.id for (b, _, _) in fires],
                "delta": _serialize_delta(agg),
            })

        terminal = next(((b, ag, d) for (b, ag, d) in fires if b.terminal), None)
        if terminal is not None:
            t_beat, _, _ = terminal
            # ── terminal dispatch (still rendered, in appropriate voice) ──
            acting = _acting_for(chosen, state, mission)
            text = await renderer(session, cycle_index, chosen, delta, acting)
            session.apply(
                DispatchRendered(
                    cycle_index=cycle_index, acting=acting, archetype=chosen.archetype, text=text,
                )
            )
            resolution = _build_resolution(mission, state, t_beat.id, t_beat.outcome_key or "")
            session.apply(MissionResolved(cycle_index=cycle_index, resolution=resolution))
            await emit(_resolved_payload(resolution))
            return False

        # ── iv. Dispatch ──
        acting = _acting_for(chosen, state, mission)
        text = await renderer(session, cycle_index, chosen, delta, acting)
        session.apply(
            DispatchRendered(
                cycle_index=cycle_index, acting=acting, archetype=chosen.archetype, text=text,
            )
        )

        # ── cycle close ──
        close_delta = tick_cycle_close(state)
        session.apply(CycleClosed(cycle_index=cycle_index, delta=close_delta))
        await emit({"event": "cycle_closed", "cycle": cycle_index, "state": state.qualities})

        # Timeout terminal — check after cycles_remaining decrement.
        # Re-run beat phase so `timeout` (a normal beat) can fire. Pass
        # `made_progress=True` to suppress passive-pressure beats: those
        # already had their chance on the first pass this cycle; we don't
        # want unlimited-fire passive beats double-firing across passes.
        if state.outcome is None:
            fires2, agg2 = fire_due_beats(state, mission, made_progress=True)
            if fires2:
                session.apply(
                    BeatsFired(
                        cycle_index=cycle_index,
                        fires=[(b.id, agent_id, d) for (b, agent_id, d) in fires2],
                    )
                )
                t2 = next(((b, ag, d) for (b, ag, d) in fires2 if b.terminal), None)
                if t2 is not None:
                    t_beat, _, _ = t2
                    resolution = _build_resolution(mission, state, t_beat.id, t_beat.outcome_key or "")
                    session.apply(MissionResolved(cycle_index=cycle_index, resolution=resolution))
                    await emit(_resolved_payload(resolution))
                    return False

    return state.outcome is None


# ───────────────────────── helpers ─────────────────────────


def _merge_delta(a: dict, b: dict) -> dict:
    """Merge two deltas — preserve first-seen `before`, latest `after`."""
    out = dict(a)
    for k, (b_before, b_after) in b.items():
        if k in out:
            out[k] = (out[k][0], b_after)
        else:
            out[k] = (b_before, b_after)
    return out


def _serialize_delta(delta: dict) -> dict:
    """For SSE payloads: collapse (before, after) tuples to a JSON-friendly shape."""
    return {q: {"before": b, "after": a} for q, (b, a) in delta.items()}


def _resolved_payload(resolution: MissionResolution) -> dict:
    """Build the `mission_resolved` SSE payload from the engine's `MissionResolution`."""
    return {
        "event": "mission_resolved",
        "outcome": resolution.outcome_key,
        "summary": resolution.summary,
        "terminal_beat": resolution.terminal_beat_id,
        "cycles_used": resolution.cycles_used,
        "cycles_max": resolution.cycles_max,
        "agents_lost": resolution.agents_lost,
        "agents_alive": resolution.agents_alive,
    }


def _acting_for(option: Option, state: State, mission: Mission) -> str:
    """Decide whose voice renders the dispatch. Bound agents → that agent;
    HOLD and unbound archetypes → 'operations'. See HANDLER_SPEC_V2 §8.3."""
    if option.agent is not None:
        return option.agent
    return "operations"


def _build_resolution(
    mission: Mission, state: State, terminal_beat_id: str, outcome_key: str
) -> MissionResolution:
    cycles_used = mission.max_cycles - int(state.get_quality("mission.cycles_remaining") or 0)
    lost: list[str] = []
    alive: list[str] = []
    for agent in mission.agents:
        cover = state.get_quality(f"agent[{agent.id}].cover")
        if cover is not None and int(cover) > 0:
            alive.append(agent.id)
        else:
            lost.append(agent.id)
    summary = mission.outcomes.get(outcome_key, outcome_key)
    return MissionResolution(
        outcome_key=outcome_key,
        terminal_beat_id=terminal_beat_id,
        cycles_used=cycles_used,
        cycles_max=mission.max_cycles,
        agents_lost=lost,
        agents_alive=alive,
        summary=summary,
    )
