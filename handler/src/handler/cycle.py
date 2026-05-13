"""The cycle orchestration. One function (`run_cycle`) ties together the
deterministic spine and the two LLM agents:

  i.   Drafter               (agents.drafter — stateless one-shot)
       ── player picks ──    (await session.await_authorization)
  ii.  Effects + events      (spine.apply_* + spine.fire_due_passive_events)
  iii. Log writer            (agents.log_writer — conversation-based)
       ── cycle close ──     (spine.tick_cycle_close)

This module is the ONLY orchestration code. `spine.py` is pure; `cycle.py`
is the only place where LLM calls and state mutation interleave.

State mutation always flows through `session.apply(event)`. The spine
functions mutate session.state in place — that's the live path — but
the emitted event carries the same delta so `fold()` reproduces the
state during replay.

Wall-clock optimisations:
  - The log writer's callable returns `(text, pacing_task)` so the cycle
    close can run while paced `entry_word` SSE events continue to the
    client in the background, and fires `on_log_text_ready` so `main.py`
    can schedule cycle N+1's prefetch.
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
    EffectsApplied,
    EventsFired,
    LogEntryAppended,
    MissionResolved,
    MissionStarted,
    OptionAuthorized,
    OptionsDrafted,
)
from .models import (
    Event,
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
    select_beat,
    snapshot,
    tick_cycle_close,
    valid_archetype_combinations,
)


log = logging.getLogger(__name__)


# ───────────────────────── agent + emitter protocols ─────────────────────────


class DrafterLLMFn(Protocol):
    async def __call__(
        self,
        session: Session,
        cycle_index: int,
        valid_catalog: list[tuple[str, str | None, dict[str, str]]],
        forced_event_id: str | None,
        beat_fragment: str | None,
    ) -> tuple[list[Option], bool]:
        """LLM-only phase of the drafter. No SSE emission. Returns
        (options, engine_fallback). On engine_fallback=True the drafter
        exhausted retries and the engine synthesised a fallback set.

        `beat_fragment` is the cycle's selected beat fragment (if any). The
        drafter renders it as a BEAT THIS DAY block — cycle framing color
        that informs option authoring without prescribing outcomes."""


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


class LogWriterFn(Protocol):
    async def __call__(
        self,
        session: Session,
        cycle_index: int,
        option: Option,
        delta: dict,
        prior_state: State,
        *,
        leans: list[str] = (),
    ) -> tuple[str, asyncio.Task]:
        """Append the cycle's case-file entry. Returns (full text,
        pacing_task). Text resolves the moment the LLM stream is exhausted;
        pacing_task continues emitting paced `entry_word` SSE events in
        the background.

        Internally the callable also updates the session's accumulating
        `log_message_history` with the just-rendered user+assistant pair.

        `prior_state` is the snapshot taken at cycle open, before any of
        this cycle's mutations applied. Used by the log-writer prompt to
        compute the tension band.

        `leans` is the (already prioritized + capped) list of free-form
        directives collected from this cycle's fired events."""


class OnLogTextReadyFn(Protocol):
    async def __call__(self, cycle_index: int) -> None:
        """Fires once cycle N's log-writer LLM has produced its text AND
        state is final (tick_cycle_close + timeout re-check done). main.py
        uses this to schedule cycle N+1's prefetch."""


class Emitter(Protocol):
    """Sink for SSE-side events (separate from session.apply, which writes
    the canonical log). Provided by main.py so cycle.py can stream paced
    word events to the connected client without knowing about HTTP."""

    async def __call__(self, payload: dict) -> None: ...


async def _noop_log_text_ready(_cycle_index: int) -> None:
    return


def fire_beat(state: State, mission: Mission):
    """Select and fire one beat at cycle-open: appends the beat's id to
    `state.fired` when `fires_at_most="once_per_mission"` so the same
    observation never surfaces twice in a run. Returns the Beat (or None).

    Called from `run_cycle` at cycle-open AND mirrored read-only from the
    prefetch path via `spine.select_beat` (same result given the same state,
    so the prefetched options and the live drafter both see the same beat
    fragment). The single fire-site is here.
    """
    beat = select_beat(state, mission)
    if beat is None:
        return None
    if beat.fires_at_most == "once_per_mission":
        state.fired.append(beat.id)
    return beat


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
    drafter_llm: DrafterLLMFn,
    drafter_emit: DrafterEmitFn,
    log_writer: LogWriterFn,
    emit: Emitter,
    await_authorization: Callable[[], Awaitable[Option]],
    on_log_text_ready: OnLogTextReadyFn = _noop_log_text_ready,
) -> bool:
    """Run one cycle. Returns True if the mission is still active, False on
    terminal (caller stops iterating).

    `await_authorization` is supplied by main.py: it waits for the player's
    POST /authorize, validates the picked option against `options`, and
    returns the resolved `Option` (which carries the prose from drafter).

    `on_log_text_ready` is invoked after the log-writer LLM produces its
    text AND state is final (tick_cycle_close + any timeout re-check done).
    Main.py uses this to schedule cycle N+1's prefetch.
    """
    # Local import — agents imports spine; cycle imports agents would cycle.
    from .agents import describe_delta

    mission = session.mission
    state = session.state
    if state.outcome is not None:
        return False  # mission already resolved

    with logfire.span(
        "run_cycle",
        cycle_index=cycle_index,
        mission=mission.id,
    ):
        # Snapshot state at cycle open. Log writer reads this back to compute
        # the tension band (4c) — magnitude of the cycle's cover swing.
        prior_state = snapshot(state)

        # ── cycle opened ──
        # Fire the cycle's beat BEFORE the drafter so its fragment can frame
        # the option set. Fire-once semantics live in `fire_beat` — the
        # prefetch path called `select_beat` (read-only) against the same
        # state, so the prefetched options were drafted with the same fragment.
        beat = fire_beat(state, mission)
        beat_fragment = beat.fragment if beat is not None else None
        session.apply(CycleOpened(cycle_index=cycle_index))
        await emit({"event": "cycle_opened", "cycle": cycle_index, "state": state.qualities})

        # ── i. Drafter (with optional forced event) ──
        catalog = valid_archetype_combinations(state, mission)
        forced = due_forced_events(state, mission)
        forced_event_id = forced[0].id if forced else None

        options, engine_fallback = await drafter_llm(
            session, cycle_index, catalog, forced_event_id, beat_fragment
        )
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

        # ── ii. Effects + passive events ──
        delta, branches_taken = apply_archetype_effects(state, chosen, mission)
        satisfied_event_id: str | None = None
        satisfied_forced: Event | None = None
        if chosen.satisfies_event is not None:
            for e in forced:
                if e.id == chosen.satisfies_event:
                    edelta = apply_forced_event_effects(state, e, chosen)
                    merge_delta_into(delta, edelta)
                    satisfied_event_id = e.id
                    satisfied_forced = e
                    break
        session.apply(EffectsApplied(
            cycle_index=cycle_index, delta=delta, forced_event_id=satisfied_event_id,
        ))
        await emit({
            "event": "effects_applied",
            "delta": _serialize_delta(delta),
            "delta_descriptors": describe_delta(delta, mission),
        })

        # If the authorized archetype fired one or more author-tagged
        # branches (opposed-check shape — clean / partial / failure),
        # collect the matching `branch_leans` for the log-writer prompt.
        # `branches_taken` is typically empty or a single-element set.
        branch_leans: list[str] = []
        if branches_taken:
            arch = mission.archetype(chosen.archetype)
            for label in branches_taken:
                branch_leans.extend(arch.branch_leans.get(label, []))

        made_progress = option_progresses_mission(delta, mission)
        fires, agg = fire_due_passive_events(state, mission, made_progress)
        leans = _collect_leans(
            satisfied_forced,
            fires,
            branch_leans=branch_leans,
            beat_fragment=beat_fragment,
        )
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
                "delta_descriptors": describe_delta(agg, mission),
            })

        terminal = next(((ev, ag, d) for (ev, ag, d) in fires if ev.terminal), None)
        if terminal is not None:
            t_event, _, _ = terminal
            # ── terminal log entry (still rendered, in Operations voice) ──
            text, log_pacing = await log_writer(
                session, cycle_index, chosen, delta, prior_state, leans=leans,
            )
            session.apply(
                LogEntryAppended(
                    cycle_index=cycle_index, archetype=chosen.archetype, text=text,
                )
            )
            resolution = _build_resolution(mission, state, t_event.id, t_event.outcome_key or "")
            session.apply(MissionResolved(cycle_index=cycle_index, resolution=resolution))
            # Wait for pacing so the client sees the full final entry before
            # mission_resolved hits.
            await log_pacing
            await emit(_resolved_payload(resolution))
            return False

        # ── iii. Log writer ──
        text, log_pacing = await log_writer(
            session, cycle_index, chosen, delta, prior_state, leans=leans,
        )
        session.apply(
            LogEntryAppended(
                cycle_index=cycle_index, archetype=chosen.archetype, text=text,
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
        # (mission_resolved) is deferred below until log pacing finishes.
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
        # prefetched drafter will see the correct cycle N+1 starting state.
        if terminal_resolved is None:
            await on_log_text_ready(cycle_index)

        # Wait for log pacing to finish before emitting any cycle-close
        # signals — otherwise the client would see cycle_closed (or
        # mission_resolved) while still receiving entry_word events.
        await log_pacing
        # `delta_descriptors` on cycle_closed carries the close-tick decrement
        # (cycles_remaining). The current-cycle panel's RESULT footer
        # accumulates effects_applied + events_fired + cycle_closed descriptors
        # for the full picture; 2nd-pass passive deltas only fire on timeout
        # and are handled via the mission_resolved card.
        await emit({
            "event": "cycle_closed",
            "cycle": cycle_index,
            "state": state.qualities,
            "delta_descriptors": describe_delta(close_delta, mission),
        })
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


_LEAN_CAP = 6


def _collect_leans(
    forced_event: Event | None,
    passive_fires: list,
    *,
    branch_leans: list[str] = (),
    beat_fragment: str | None = None,
) -> list[str]:
    """Prioritize and cap the cycle's lean directives for the log writer.

    Order: satisfied forced event → branch leans (opposed-check resolution
    on the authorized archetype) → terminal passive events → non-terminal
    passive events → beat fragment (cycle-open framing color).
    Capped at `_LEAN_CAP` to keep the prompt focused.

    Branch leans land early because they describe *the immediate outcome of
    the just-authorized option* — the log writer needs them to write a reveal
    on a failure branch vs a confirmation on a clean branch. The beat
    fragment lands last as ambient framing — the writer can pull it in if
    space allows.
    """
    out: list[str] = []
    if forced_event is not None:
        out.extend(forced_event.lean_on)
    out.extend(branch_leans)
    terminal_leans: list[str] = []
    nonterminal_leans: list[str] = []
    for (ev, _, _) in passive_fires:
        (terminal_leans if ev.terminal else nonterminal_leans).extend(ev.lean_on)
    out.extend(terminal_leans)
    out.extend(nonterminal_leans)
    if beat_fragment:
        out.append(beat_fragment)
    return out[:_LEAN_CAP]


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
