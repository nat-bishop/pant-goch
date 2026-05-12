"""The cycle orchestration. One function (`run_cycle`) ties together the
deterministic spine and the three LLM agents per HANDLER_SPEC_V2 §3:

  i.   SoP synthesizer       (agents.sop)
  ii.  Drafter               (agents.drafter)
       ── player picks ──    (await session.await_authorization)
  iii. Effects + events      (spine.apply_* + spine.fire_due_passive_events)
  iv.  Dispatch renderer     (agents.renderer)
       ── cycle close ──     (spine.tick_cycle_close)

This module is the ONLY orchestration code. `spine.py` is pure; `cycle.py`
is the only place where LLM calls and state mutation interleave.

State mutation always flows through `session.apply(event)`. The spine
functions mutate session.state in place — that's the live path — but
the emitted event carries the same delta so `fold()` reproduces the
state during replay.

Wall-clock optimisations (HANDLER_SPEC_V2 §11):
  - The SoP callable returns `(text, pacing_task)` so the drafter LLM can
    start the moment SoP's LLM stream is exhausted, while paced sop_word
    SSE events continue to the client in the background.
  - The renderer callable does the same for `dispatch_word` pacing, and
    fires `on_renderer_text_ready` so `main.py` can schedule cycle N+1's
    prefetch while cycle N's dispatch is still pacing to the modal.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol

import logfire

from .events import (
    CycleClosed,
    CycleOpened,
    DispatchRendered,
    EffectsApplied,
    EventsFired,
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
    apply_forced_event_effects,
    due_forced_events,
    fire_due_passive_events,
    merge_delta_into,
    option_progresses_mission,
    snapshot,
    tick_cycle_close,
    valid_archetype_combinations,
)


log = logging.getLogger(__name__)


# ───────────────────────── agent + emitter protocols ─────────────────────────


class SopFn(Protocol):
    async def __call__(
        self, session: Session, cycle_index: int
    ) -> tuple[str, asyncio.Task]:
        """Render the State-of-Play memo. Returns (full text, pacing_task).
        Text resolves as soon as the LLM stream is exhausted; pacing_task
        continues emitting paced `sop_word` SSE events in the background."""


class DrafterLLMFn(Protocol):
    async def __call__(
        self,
        session: Session,
        cycle_index: int,
        sop_text: str,
        valid_catalog: list[tuple[str, str | None, dict[str, str]]],
        forced_event_id: str | None,
    ) -> tuple[list[Option], bool]:
        """LLM-only phase of the drafter. No SSE emission. Returns
        (options, engine_fallback). On engine_fallback=True the drafter
        exhausted retries and the engine synthesised a fallback set."""


class DrafterEmitFn(Protocol):
    async def __call__(
        self,
        cycle_index: int,
        options: list[Option],
        engine_fallback: bool,
        forced_event_id: str | None,
    ) -> None:
        """Pace each option's prose to the client as option_started +
        option_word ×N events, terminated by options_complete."""


class RendererFn(Protocol):
    async def __call__(
        self,
        session: Session,
        cycle_index: int,
        option: Option,
        delta: dict,
        acting: str,
        prior_state: State,
    ) -> tuple[str, asyncio.Task]:
        """Render the field dispatch. Returns (full text, pacing_task).
        Same shape as SopFn — text resolves on LLM-done so the caller can
        kick off cross-cycle prefetch while pacing continues.

        `prior_state` is the snapshot taken at cycle open, before any of
        this cycle's mutations applied. Used by the renderer prompt to
        compute the tension band (4c) — magnitude of the cycle's cover swing."""


class OnRendererTextReadyFn(Protocol):
    async def __call__(self, cycle_index: int) -> None:
        """Fires once cycle N's renderer LLM has produced its text AND state
        is final (tick_cycle_close + timeout re-check done). main.py uses
        this to schedule cycle N+1's prefetch."""


class Emitter(Protocol):
    """Sink for SSE-side events (separate from session.apply, which writes
    the canonical log). Provided by main.py so cycle.py can stream paced
    word events to the connected client without knowing about HTTP."""

    async def __call__(self, payload: dict) -> None: ...


async def _noop_renderer_text_ready(_cycle_index: int) -> None:
    return


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
    drafter_llm: DrafterLLMFn,
    drafter_emit: DrafterEmitFn,
    renderer: RendererFn,
    emit: Emitter,
    await_authorization: Callable[[], Awaitable[Option]],
    on_renderer_text_ready: OnRendererTextReadyFn = _noop_renderer_text_ready,
) -> bool:
    """Run one cycle. Returns True if the mission is still active, False on
    terminal (caller stops iterating).

    `await_authorization` is supplied by main.py: it waits for the player's
    POST /authorize, validates the picked option against `options`, and
    returns the resolved `Option` (which carries the prose from drafter).

    `on_renderer_text_ready` is invoked after the renderer LLM produces its
    text AND state is final (tick_cycle_close + any timeout re-check done).
    Main.py uses this to schedule cycle N+1's prefetch.
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
        # Snapshot state at cycle open. Renderer reads this back to compute
        # the tension band (4c) — magnitude of the cycle's cover swing.
        prior_state = snapshot(state)

        # ── cycle opened ──
        session.apply(CycleOpened(cycle_index=cycle_index))
        await emit({"event": "cycle_opened", "cycle": cycle_index, "state": state.qualities})

        # ── i. SoP synthesizer ──
        # Returns text the moment the LLM stream is exhausted; pacing_task
        # continues emitting paced sop_word events to the client in the bg.
        sop_text, sop_pacing = await sop(session, cycle_index)
        session.apply(SopRendered(cycle_index=cycle_index, text=sop_text))

        # ── ii. Drafter (with optional forced event) ──
        catalog = valid_archetype_combinations(state, mission)
        forced = due_forced_events(state, mission)
        forced_event_id = forced[0].id if forced else None

        # Run drafter LLM in parallel with SoP pacing. The drafter only needs
        # sop_text (the LLM output), not the paced emission, so it can start
        # as soon as the LLM finishes generating.
        drafter_task = asyncio.create_task(
            drafter_llm(session, cycle_index, sop_text, catalog, forced_event_id)
        )

        # Wait for SoP pacing to finish before showing options — keeps the
        # reading order (SoP, then options) intact on the client.
        await sop_pacing

        options, engine_fallback = await drafter_task
        session.apply(
            OptionsDrafted(
                cycle_index=cycle_index,
                options=options,
                forced_event_id=forced_event_id,
                engine_fallback=engine_fallback,
            )
        )
        await drafter_emit(cycle_index, options, engine_fallback, forced_event_id)
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

        # ── iii. Effects + passive events ──
        delta = apply_archetype_effects(state, chosen, mission)
        satisfied_event_id: str | None = None
        if chosen.satisfies_event is not None:
            for e in forced:
                if e.id == chosen.satisfies_event:
                    edelta = apply_forced_event_effects(state, e, chosen)
                    merge_delta_into(delta, edelta)
                    satisfied_event_id = e.id
                    break
        session.apply(EffectsApplied(
            cycle_index=cycle_index, delta=delta, forced_event_id=satisfied_event_id,
        ))
        await emit({"event": "effects_applied", "delta": _serialize_delta(delta)})

        made_progress = option_progresses_mission(delta, mission)
        fires, agg = fire_due_passive_events(state, mission, made_progress)
        if fires:
            session.apply(
                EventsFired(
                    cycle_index=cycle_index,
                    fires=[(ev.id, agent_id, d) for (ev, agent_id, d) in fires],
                )
            )
            await emit({
                "event": "events_fired",
                "events": [ev.id for (ev, _, _) in fires],
                "delta": _serialize_delta(agg),
            })

        terminal = next(((ev, ag, d) for (ev, ag, d) in fires if ev.terminal), None)
        if terminal is not None:
            t_event, _, _ = terminal
            # ── terminal dispatch (still rendered, in appropriate voice) ──
            acting = _acting_for(chosen)
            text, render_pacing = await renderer(
                session, cycle_index, chosen, delta, acting, prior_state
            )
            session.apply(
                DispatchRendered(
                    cycle_index=cycle_index, acting=acting, archetype=chosen.archetype, text=text,
                )
            )
            resolution = _build_resolution(mission, state, t_event.id, t_event.outcome_key or "")
            session.apply(MissionResolved(cycle_index=cycle_index, resolution=resolution))
            # Wait for dispatch pacing so the client sees the full final
            # dispatch before mission_resolved hits.
            await render_pacing
            await emit(_resolved_payload(resolution))
            return False

        # ── iv. Dispatch ──
        acting = _acting_for(chosen)
        text, render_pacing = await renderer(
            session, cycle_index, chosen, delta, acting, prior_state
        )
        session.apply(
            DispatchRendered(
                cycle_index=cycle_index, acting=acting, archetype=chosen.archetype, text=text,
            )
        )

        # ── cycle close (deterministic; runs while pacing continues in bg) ──
        close_delta = tick_cycle_close(state)
        session.apply(CycleClosed(cycle_index=cycle_index, delta=close_delta))

        # Timeout terminal — check after cycles_remaining decrement. Re-run
        # the passive-event phase so `timeout` (a normal passive event) can
        # fire. Pass `made_progress=True` to suppress passive-pressure events:
        # those already had their chance on the first pass this cycle.
        #
        # State mutations happen here synchronously; the matching SSE emit
        # (mission_resolved) is deferred below until dispatch pacing finishes.
        terminal_resolved: dict | None = None
        if state.outcome is None:
            fires2, agg2 = fire_due_passive_events(state, mission, made_progress=True)
            if fires2:
                session.apply(
                    EventsFired(
                        cycle_index=cycle_index,
                        fires=[(ev.id, agent_id, d) for (ev, agent_id, d) in fires2],
                    )
                )
                t2 = next(((ev, ag, d) for (ev, ag, d) in fires2 if ev.terminal), None)
                if t2 is not None:
                    t_event, _, _ = t2
                    resolution = _build_resolution(
                        mission, state, t_event.id, t_event.outcome_key or ""
                    )
                    session.apply(MissionResolved(cycle_index=cycle_index, resolution=resolution))
                    terminal_resolved = _resolved_payload(resolution)

        # Fire prefetch hook only if mission is still active. State is final
        # at this point (cycle_close + timeout re-check both done), so the
        # prefetched SoP/drafter will see the correct cycle N+1 starting state.
        if terminal_resolved is None:
            await on_renderer_text_ready(cycle_index)

        # Wait for dispatch pacing to finish before emitting any cycle-close
        # signals — otherwise the client would see cycle_closed (or
        # mission_resolved) while still receiving dispatch_word events.
        await render_pacing
        await emit({"event": "cycle_closed", "cycle": cycle_index, "state": state.qualities})
        if terminal_resolved is not None:
            await emit(terminal_resolved)
            return False

    return state.outcome is None


# ───────────────────────── helpers ─────────────────────────


def _serialize_delta(delta: dict) -> dict:
    """For SSE payloads: collapse (before, after) tuples to a JSON-friendly shape."""
    return {q: {"before": b, "after": a} for q, (b, a) in delta.items()}


def _resolved_payload(resolution: MissionResolution) -> dict:
    """Build the `mission_resolved` SSE payload from the engine's `MissionResolution`."""
    return {
        "event": "mission_resolved",
        "outcome": resolution.outcome_key,
        "summary": resolution.summary,
        "terminal_event": resolution.terminal_event_id,
        "cycles_used": resolution.cycles_used,
        "cycles_max": resolution.cycles_max,
        "agents_lost": resolution.agents_lost,
        "agents_alive": resolution.agents_alive,
    }


def _acting_for(option: Option) -> str:
    """Decide whose voice renders the dispatch. Bound agents → that agent;
    HOLD and unbound archetypes → 'operations'. See HANDLER_SPEC_V2 §8.3."""
    if option.agent is not None:
        return option.agent
    return "operations"


def _build_resolution(
    mission: Mission, state: State, terminal_event_id: str, outcome_key: str
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
        terminal_event_id=terminal_event_id,
        cycles_used=cycles_used,
        cycles_max=mission.max_cycles,
        agents_lost=lost,
        agents_alive=alive,
        summary=summary,
    )
