"""Day orchestration. One function (`run_day`) ties together the
deterministic spine and the two LLM agents:

  i.   Drafter               (agents.drafter — stateless one-shot)
       ── player picks ──    (await session.await_authorization)
  ii.  Effects + reactions   (spine.apply_action_effects + spine.fire_due_reactions)
  iii. Log writer            (agents.log_writer — conversation-based)
       ── day close ──       (spine.tick_day_close)

This module is the ONLY orchestration code. `spine.py` is pure; `day.py`
is the only place where LLM calls and state mutation interleave.

State mutation always flows through `session.apply(event)`. The spine
functions mutate session.state in place — that's the live path — but
the emitted event carries the same delta so `fold()` reproduces the
state during replay.

Wall-clock optimisations:
  - The log writer's callable returns `(text, pacing_task)` so the day
    close can run while paced `entry_word` SSE events continue to the
    client in the background, and fires `on_log_text_ready` so `main.py`
    can schedule day N+1's prefetch.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol

import logfire

from .events import (
    DayClosed,
    DayOpened,
    EffectsApplied,
    LogEntryAppended,
    MissionResolved,
    MissionStarted,
    OptionAuthorized,
    OptionsDrafted,
    ReactionsFired,
)
from .models import (
    Action,
    Mission,
    MissionResolution,
    Option,
    State,
    make_initial_state,
)
from .session import Session
from .spine import (
    apply_action_effects,
    due_forced_actions,
    fire_due_reactions,
    option_progresses_mission,
    select_beat,
    snapshot,
    tick_day_close,
    valid_action_catalog,
)


log = logging.getLogger(__name__)


# ───────────────────────── agent + emitter protocols ─────────────────────────


class DrafterLLMFn(Protocol):
    async def __call__(
        self,
        session: Session,
        day_index: int,
        valid_catalog: list[tuple[str, str | None, dict[str, str]]],
        forced_action_id: str | None,
        beat_fragment: str | None,
    ) -> tuple[list[Option], bool]:
        """LLM-only phase of the drafter. No SSE emission. Returns
        (options, engine_fallback). On engine_fallback=True the drafter
        exhausted retries and the engine synthesised a fallback set.

        `beat_fragment` is the day's selected beat fragment (if any). The
        drafter renders it as a BEAT THIS DAY block — framing color that
        informs option authoring without prescribing outcomes."""


class DrafterEmitFn(Protocol):
    async def __call__(
        self,
        day_index: int,
        options: list[Option],
        engine_fallback: bool,
        forced_action_id: str | None,
    ) -> None:
        """Pace each option's prose to the client as option_started +
        option_word ×N events, terminated by options_complete."""


class LogWriterFn(Protocol):
    async def __call__(
        self,
        session: Session,
        day_index: int,
        option: Option,
        delta: dict,
        prior_state: State,
        *,
        leans: list[str] = (),
    ) -> tuple[str, asyncio.Task]:
        """Append the day's case-file entry. Returns (full text,
        pacing_task). Text resolves the moment the LLM stream is exhausted;
        pacing_task continues emitting paced `entry_word` SSE events in
        the background.

        Internally the callable also updates the session's accumulating
        `log_message_history` with the just-rendered user+assistant pair.

        `prior_state` is the snapshot taken at day open, before any of
        this day's mutations applied. Used by the log-writer prompt to
        compute the tension band.

        `leans` is the (already prioritized + capped) list of free-form
        directives collected from this day's resolved Outcome + Reaction fires."""


class OnLogTextReadyFn(Protocol):
    async def __call__(self, day_index: int) -> None:
        """Fires once day N's log-writer LLM has produced its text AND
        state is final (tick_day_close + timeout re-check done). main.py
        uses this to schedule day N+1's prefetch."""


class Emitter(Protocol):
    """Sink for SSE-side events (separate from session.apply, which writes
    the canonical log). Provided by main.py so day.py can stream paced
    word events to the connected client without knowing about HTTP."""

    async def __call__(self, payload: dict) -> None: ...


async def _noop_log_text_ready(_day_index: int) -> None:
    return


def fire_beat(state: State, mission: Mission):
    """Select and fire one beat at day-open: appends the beat's id to
    `state.fired` when `fires_at_most="once_per_mission"` so the same
    observation never surfaces twice in a run. Returns the Beat (or None).

    Called from `run_day` at day-open AND mirrored read-only from the
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


# ───────────────────────── day orchestration ─────────────────────────


async def run_day(
    session: Session,
    day_index: int,
    *,
    drafter_llm: DrafterLLMFn,
    drafter_emit: DrafterEmitFn,
    log_writer: LogWriterFn,
    emit: Emitter,
    await_authorization: Callable[[], Awaitable[Option]],
    on_log_text_ready: OnLogTextReadyFn = _noop_log_text_ready,
) -> bool:
    """Run one day. Returns True if the mission is still active, False on
    terminal (caller stops iterating).

    `await_authorization` is supplied by main.py: it waits for the player's
    POST /authorize, validates the picked option against `options`, and
    returns the resolved `Option`.
    """
    # Local import — agents imports spine; day imports agents would cycle.
    from .agents import describe_delta

    mission = session.mission
    state = session.state
    if state.outcome is not None:
        return False

    with logfire.span(
        "run_day",
        day_index=day_index,
        mission=mission.id,
    ):
        # Snapshot state at day open. Log writer reads this back to compute
        # the tension band — magnitude of the day's cover swing.
        prior_state = snapshot(state)

        # ── day opened ──
        # Fire the day's beat BEFORE the drafter so its fragment can frame
        # the option set.
        beat = fire_beat(state, mission)
        beat_fragment = beat.fragment if beat is not None else None
        session.apply(DayOpened(day_index=day_index))
        await emit({"event": "day_opened", "day": day_index, "state": state.qualities})

        # ── i. Drafter (with optional forced Action) ──
        catalog = valid_action_catalog(state, mission)
        forced = due_forced_actions(state, mission)
        forced_action_id = forced[0].id if forced else None

        options, engine_fallback = await drafter_llm(
            session, day_index, catalog, forced_action_id, beat_fragment
        )
        session.apply(
            OptionsDrafted(
                day_index=day_index,
                options=options,
                forced_action_id=forced_action_id,
                engine_fallback=engine_fallback,
            )
        )
        await drafter_emit(day_index, options, engine_fallback, forced_action_id)
        await emit({"event": "awaiting_auth"})

        # ── player picks ──
        chosen = await await_authorization()
        session.apply(
            OptionAuthorized(
                day_index=day_index,
                option=chosen,
                by_handler_at=datetime.now(timezone.utc),
            )
        )

        # ── ii. Action effects + Reaction pass ──
        delta, action_leans = apply_action_effects(state, chosen, mission)
        chosen_action = mission.action(chosen.action)
        session.apply(EffectsApplied(
            day_index=day_index,
            delta=delta,
            action_id=chosen.action,
            action_forced=chosen_action.forced,
        ))
        await emit({
            "event": "effects_applied",
            "delta": _serialize_delta(delta),
            "delta_descriptors": describe_delta(delta, mission),
        })

        made_progress = option_progresses_mission(delta, mission)
        fires, agg = fire_due_reactions(state, mission, made_progress)
        leans = _collect_leans(
            action_leans=action_leans,
            reaction_fires=fires,
            beat_fragment=beat_fragment,
        )
        if fires:
            session.apply(
                ReactionsFired(
                    day_index=day_index,
                    fires=[(r.id, agent_id, d) for (r, agent_id, d) in fires],
                )
            )
            await emit({
                "event": "reactions_fired",
                "reactions": [r.id for (r, _, _) in fires],
                "delta": _serialize_delta(agg),
                "delta_descriptors": describe_delta(agg, mission),
            })

        terminal = next(((r, ag, d) for (r, ag, d) in fires if r.terminal), None)
        if terminal is not None:
            t_reaction, _, _ = terminal
            # ── terminal log entry (still rendered, in Operations voice) ──
            text, log_pacing = await log_writer(
                session, day_index, chosen, delta, prior_state, leans=leans,
            )
            session.apply(
                LogEntryAppended(
                    day_index=day_index, action=chosen.action, text=text,
                )
            )
            resolution = _build_resolution(mission, state, t_reaction.id, t_reaction.outcome_key or "")
            session.apply(MissionResolved(day_index=day_index, resolution=resolution))
            await log_pacing
            await emit(_resolved_payload(resolution))
            return False

        # ── iii. Log writer ──
        text, log_pacing = await log_writer(
            session, day_index, chosen, delta, prior_state, leans=leans,
        )
        session.apply(
            LogEntryAppended(
                day_index=day_index, action=chosen.action, text=text,
            )
        )

        # ── day close (deterministic; runs while pacing continues in bg) ──
        close_delta = tick_day_close(state)
        session.apply(DayClosed(day_index=day_index, delta=close_delta))

        # Timeout terminal — check after days_remaining decrement. Re-run
        # the reaction phase so `timeout` (a normal Reaction) can fire.
        # Pass `made_progress=True` to suppress passive-pressure reactions:
        # those already had their chance on the first pass this day.
        terminal_resolved: dict | None = None
        if state.outcome is None:
            fires2, _ = fire_due_reactions(state, mission, made_progress=True)
            if fires2:
                session.apply(
                    ReactionsFired(
                        day_index=day_index,
                        fires=[(r.id, agent_id, d) for (r, agent_id, d) in fires2],
                    )
                )
                t2 = next(((r, ag, d) for (r, ag, d) in fires2 if r.terminal), None)
                if t2 is not None:
                    t_reaction, _, _ = t2
                    resolution = _build_resolution(
                        mission, state, t_reaction.id, t_reaction.outcome_key or ""
                    )
                    session.apply(MissionResolved(day_index=day_index, resolution=resolution))
                    terminal_resolved = _resolved_payload(resolution)

        # Fire prefetch hook only if mission is still active.
        if terminal_resolved is None:
            await on_log_text_ready(day_index)

        await log_pacing
        await emit({
            "event": "day_closed",
            "day": day_index,
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
        "terminal_reaction": resolution.terminal_reaction_id,
        "days_used": resolution.days_used,
        "days_max": resolution.days_max,
        "agents_lost": resolution.agents_lost,
        "agents_alive": resolution.agents_alive,
    }


_LEAN_CAP = 6


def _collect_leans(
    *,
    action_leans: list[str],
    reaction_fires: list,
    beat_fragment: str | None = None,
) -> list[str]:
    """Prioritize and cap the day's lean directives for the log writer.

    Order: matched-outcome leans (the just-resolved action's outcome) →
    terminal Reaction leans → non-terminal Reaction leans → beat fragment
    (day-open framing color). Capped at `_LEAN_CAP` to keep the prompt focused.

    Outcome leans land first because they describe *the immediate outcome of
    the just-authorized option* — the log writer needs them to write a reveal
    on a failure branch vs a confirmation on a clean branch. The beat
    fragment lands last as ambient framing — the writer pulls it in if space
    allows.
    """
    out: list[str] = list(action_leans)
    terminal_leans: list[str] = []
    nonterminal_leans: list[str] = []
    for (reaction, _, _) in reaction_fires:
        (terminal_leans if reaction.terminal else nonterminal_leans).extend(reaction.lean_on)
    out.extend(terminal_leans)
    out.extend(nonterminal_leans)
    if beat_fragment:
        out.append(beat_fragment)
    return out[:_LEAN_CAP]


def _build_resolution(
    mission: Mission, state: State, terminal_reaction_id: str, outcome_key: str
) -> MissionResolution:
    days_used = mission.max_days - int(state.get_quality("mission.days_remaining") or 0)
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
        terminal_reaction_id=terminal_reaction_id,
        days_used=days_used,
        days_max=mission.max_days,
        agents_lost=lost,
        agents_alive=alive,
        summary=summary,
    )
