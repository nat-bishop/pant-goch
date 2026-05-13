"""FastAPI surface for HANDLER.

Endpoints:
  POST   /mission/start                 → boot a new session, return session_id
  GET    /mission/{session_id}/stream   → SSE event stream for that session
  POST   /mission/{session_id}/authorize → record the player's option pick
  GET    /mission/{session_id}/state    → snapshot for reconnect

The cycle loop runs as a background asyncio.Task per session, pushing events
into the session's `out_queue`. The SSE endpoint consumes from the queue.
`/authorize` resolves the cycle's pending `auth_future`, unblocking the loop
to compute effects + render the log entry + advance to the next cycle.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import logging
import os
import re
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
import logfire
from logfire import ConsoleOptions
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider
from dataclasses import replace

from .agents import (
    AgentBundle,
    LogEntry,
    _log_writer_user_prompt,
    build_agent_bundle,
    select_then_write_with_retries,
)
from .config import Settings
from .cycle import run_cycle, start_mission
from .events import (
    Event,
    LogEntryAppended,
    MissionResolved,
    OptionAuthorized,
    OptionsDrafted,
)
from .launcher import (
    LlamaStartupError,
    resolve_config,
    start_llama,
    stop_llama,
    wait_healthy,
)
from .mission_loader import MissionLoadError, load_mission
from .models import Mission, Option, State
from .session import Session, make_jsonl_persistor, make_session
from .spine import due_forced_events, select_beat, valid_archetype_combinations


_BOOT_SETTINGS = Settings()
logging.getLogger("httpx").setLevel(logging.WARNING)

_logfire_creds = Path(__file__).resolve().parents[2] / ".logfire" / "logfire_credentials.json"
_logfire_token = os.getenv("LOGFIRE_TOKEN")
if not _logfire_token and _logfire_creds.is_file():
    with contextlib.suppress(Exception):
        _logfire_token = json.loads(_logfire_creds.read_text()).get("token")
try:
    _service_version = importlib.metadata.version("handler")
except importlib.metadata.PackageNotFoundError:
    _service_version = "unknown"
logfire.configure(
    service_name="handler",
    service_version=_service_version,
    send_to_logfire="if-token-present",
    token=_logfire_token,
    console=ConsoleOptions() if _BOOT_SETTINGS.handler_logfire_console else False,
    scrubbing=False,
)
logfire.instrument_pydantic_ai()
# record='failure' keeps the failure-only safety net (mission YAML, drafter
# schema violations) without spamming a span for every field validation —
# default 'all' floods stdout during streaming chat completions and was the
# observed cause of a logfire-buffer blow-up that killed the sidecar mid-cycle.
logfire.instrument_pydantic(record="failure")
logfire.instrument_httpx(capture_all=True)
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())

log = logging.getLogger(__name__)
LLM_LOGFIRE = logfire.with_tags("llm")


# ───────────────────────── globals (set in lifespan) ─────────────────────────


SETTINGS: Settings | None = None
LLAMA_PROC: subprocess.Popen | None = None
EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS = 2.0
INFERNO_PROBE_TIMEOUT_SECONDS = 5.0
MISSION: Mission | None = None
AGENTS: AgentBundle | None = None


async def _probe_inferno_health(url: str, timeout: float = INFERNO_PROBE_TIMEOUT_SECONDS) -> None:
    """One-shot probe of inferno's /health. Fail loud — when use_inferno is
    set there is no silent fallback to the bundled llama-server (rollback is
    `unset SMALLVILLE_USE_INFERNO; restart`)."""
    health_url = f"{url.rstrip('/')}/health"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(health_url)
        except httpx.HTTPError as e:
            raise LlamaStartupError(
                f"inferno at {health_url} unreachable: {e}. "
                f"Start inferno (uvicorn inferno.server:app --port 7770) or "
                f"unset SMALLVILLE_USE_INFERNO."
            ) from e
        if resp.status_code != 200:
            raise LlamaStartupError(
                f"inferno at {health_url} returned {resp.status_code}: {resp.text[:200]}"
            )


@dataclass
class SessionRuntime:
    """Per-session live state: the engine session, the SSE queue, the two
    futures the cycle loop alternates between (auth = pause INSIDE a cycle
    waiting for the player's pick; continue = pause BETWEEN cycles waiting
    for the player to click through), the cycle task itself, and the
    accumulating case-file artifacts (log_entries, log_message_history) for
    the log writer.

    Cross-cycle prefetch (wall-clock optimisation): when cycle N's log writer
    text is ready, a background task pre-runs cycle N+1's drafter LLM call
    so options are ready by the time the player clicks Continue. The drafter
    is stateless, so prefetching it is trivial. Prefetched results are
    tagged with `prefetched_cycle_index`.

    `last_authorized` carries one-cycle context forward so the next cycle's
    drafter prompt can lead with "vary from what was recently tried".
    """

    session: Session
    out_queue: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)
    auth_future: asyncio.Future[Option] | None = None
    continue_future: asyncio.Future[None] | None = None
    current_options: list[Option] | None = None
    cycle_task: asyncio.Task | None = None
    # Accumulating case-file artifacts.
    # `log_entries` — full prose per cycle. Drafter prompt pulls the last 2
    # for cross-cycle context (drafter is stateless one-shot). Frontend
    # caches the full text for the click-to-expand history UI.
    log_entries: list[LogEntry] = field(default_factory=list)
    # `log_message_history` — the log-writer agent's accumulating pydantic-ai
    # `message_history`. One user+assistant pair per cycle. The conversation
    # IS the case file. Initialized empty; grown by `_build_log_writer_callable`
    # after each cycle's stream completes.
    log_message_history: list[ModelMessage] = field(default_factory=list)
    # Cross-cycle drafter prefetch — populated by `prefetch_next_cycle`,
    # consumed by the drafter LLM callable on the next run_cycle iteration.
    prefetched_cycle_index: int | None = None
    prefetched_options: list[Option] | None = None
    prefetched_engine_fallback: bool = False
    prefetch_task: asyncio.Task | None = None
    # Last 2 authorized picks as (cycle_index, archetype, agent_id|None).
    # Cycle N+1's drafter prompt reads this to vary tonal range. Updated by
    # `_run_session_loop` after each /authorize lands.
    last_authorized: list[tuple[int, str, str | None]] = field(default_factory=list)

    async def await_auth(self) -> Option:
        """Block the cycle loop until POST /authorize resolves auth_future."""
        self.auth_future = asyncio.get_event_loop().create_future()
        try:
            return await self.auth_future
        finally:
            self.auth_future = None

    async def await_continue(self) -> None:
        """Block the cycle loop until POST /continue resolves continue_future."""
        self.continue_future = asyncio.get_event_loop().create_future()
        try:
            await self.continue_future
        finally:
            self.continue_future = None

    def _clear_prefetch(self) -> None:
        """Reset prefetch fields (after consumption, cancellation, or a
        stale-result discard)."""
        self.prefetched_cycle_index = None
        self.prefetched_options = None
        self.prefetched_engine_fallback = False


SESSIONS: dict[str, SessionRuntime] = {}


def _get_runtime(session_id: str) -> SessionRuntime:
    rt = SESSIONS.get(session_id)
    if rt is None:
        raise HTTPException(404, f"session {session_id!r} not found")
    return rt


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot llama-server, build the model, load the mission, build agents.
    Mission and agent bundle are process-scope singletons — MVP serves one
    mission. Per-session state lives in `SESSIONS`."""
    global SETTINGS, LLAMA_PROC, MISSION, AGENTS

    settings = Settings()
    SETTINGS = settings

    cfg = resolve_config(settings)
    with logfire.span(
        "llama_lifecycle",
        launch=settings.launch_llama,
        use_inferno=settings.use_inferno,
        port=cfg.port,
        ctx=cfg.ctx,
        model_path=str(cfg.model_path) if cfg.model_path else None,
        hf_repo=cfg.hf_repo,
    ) as llama_span:
        if settings.use_inferno:
            logfire.info(
                "inferno mode — skipping bundled llama-server",
                url=settings.inferno_url,
            )
            await _probe_inferno_health(settings.inferno_url)
            llama_span.set_attribute("inferno", True)
            llama_span.set_attribute("healthy", True)
        elif settings.launch_llama:
            logfire.info("starting llama-server (managed)")
            try:
                LLAMA_PROC = start_llama(cfg)
                llama_span.set_attribute("pid", LLAMA_PROC.pid)
                await wait_healthy(host=cfg.host, port=cfg.port)
                llama_span.set_attribute("healthy", True)
            except LlamaStartupError:
                logfire.exception("llama-server failed to come up")
                stop_llama(LLAMA_PROC)
                raise
        else:
            logfire.info("llama-server externally managed (LAUNCH_LLAMA=false)")
            try:
                await wait_healthy(
                    host=cfg.host, port=cfg.port,
                    timeout=EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS,
                )
            except LlamaStartupError as e:
                logfire.warn(
                    "llama-server not reachable yet — sidecar will boot anyway",
                    error=str(e),
                )

        if settings.use_inferno:
            api_url = f"{settings.inferno_url.rstrip('/')}/v1"
        else:
            api_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
        model_name = "gemma-4-e4b"
        logfire.info("building model", model=model_name, base_url=api_url)
        base_profile = openai_model_profile(model_name)
        # `openai_chat_send_back_thinking_parts=False` — required for Gemma 4
        # multi-turn (see CLAUDE.md). The log writer's accumulating
        # message_history makes this load-bearing: without it, pydantic-ai's
        # default 'auto' round-trips ThinkingParts back into assistant
        # content as <think>...</think> blocks and drift compounds every cycle.
        profile = replace(base_profile, openai_chat_send_back_thinking_parts=False)
        provider = OpenAIProvider(base_url=api_url, api_key="local")
        model = OpenAIChatModel(model_name, provider=provider, profile=profile)

        mission_path = settings.mission_path
        if mission_path is None:
            mission_path = Path(__file__).resolve().parents[2] / "missions" / "lynx_exfil.yaml"
        try:
            MISSION = load_mission(mission_path)
        except MissionLoadError as e:
            logfire.exception("failed to load mission", mission_path=str(mission_path))
            raise
        logfire.info("mission loaded", mission=MISSION.id, agents=[a.id for a in MISSION.agents])

        AGENTS = build_agent_bundle(model, MISSION)
        app.state.model = model
        app.state.mission = MISSION
        app.state.agents = AGENTS

        try:
            yield
        finally:
            logfire.info("stopping llama-server")
            for sid, rt in list(SESSIONS.items()):
                if rt.cycle_task is not None and not rt.cycle_task.done():
                    rt.cycle_task.cancel()
            stop_llama(LLAMA_PROC)


# ───────────────────────── paced word-stream emitter ─────────────────────────


WORD_PACE_S = 0.08  # min seconds between paced word emissions


_WORD_RE = re.compile(r"\S+\s*")


async def _words_from_text(text: str) -> AsyncIterator[str]:
    """Split a complete text into words, preserving trailing whitespace per word."""
    for match in _WORD_RE.finditer(text):
        yield match.group(0)


async def _words_from_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Buffer streaming text chunks; yield each complete word as it crosses a
    whitespace boundary. The final word (no trailing whitespace) is yielded on
    stream end. Each yielded word includes its trailing whitespace, if any."""
    buf = ""
    async for chunk in chunks:
        buf += chunk
        # Repeatedly extract leading complete words.
        while True:
            match = _WORD_RE.match(buf)
            if not match:
                break
            # Word is "complete" iff there's whitespace after it OR we know
            # more content is coming. Trailing-whitespace match means complete.
            word = match.group(0)
            # If the buffer is exactly the match and the match has no trailing
            # whitespace, the word might still be growing. Hold it.
            if match.end() == len(buf) and not re.search(r"\s$", word):
                break
            yield word
            buf = buf[match.end():]
    if buf.strip():
        yield buf


async def paced_word_emitter(
    word_source: AsyncIterator[str],
    emit: Callable[[dict], Awaitable[None]],
    *,
    event_kind: str,
    min_interval: float = WORD_PACE_S,
    extra: dict | None = None,
) -> str:
    """Emit each word from `word_source` as an SSE event, throttled to at
    least `min_interval` seconds between emissions. Returns the accumulated
    full text.

    Designed for two call sites (HANDLER_SPEC_V2 §11 "Streaming architecture"):
    1. Log entry: `word_source = _words_from_stream(llm_stream)` — words
       emerge as the LLM produces them; the emitter throttles when the LLM
       outpaces the cadence and passes through when the LLM is slower.
    2. Drafter options: `word_source = _words_from_text(prose)` — full prose
       chunked into words after the structured-output call completes.
    """
    next_slot = time.monotonic()
    idx = 0
    full = ""
    async for word in word_source:
        now = time.monotonic()
        if now < next_slot:
            await asyncio.sleep(next_slot - now)
        payload = {"event": event_kind, "idx": idx, "word": word}
        if extra:
            payload.update(extra)
        await emit(payload)
        full += word
        idx += 1
        next_slot = time.monotonic() + min_interval
    sentinel = {"event": event_kind, "idx": idx, "word": "", "is_last": True}
    if extra:
        sentinel.update(extra)
    await emit(sentinel)
    return full


# ───────────────────────── agent call wrappers (the DrafterFn / LogWriterFn shapes) ─────────────────────────


def _build_drafter_llm_callable(rt: SessionRuntime) -> Any:
    """LLM-only phase of the drafter — runs `select_then_write_with_retries`
    (or consumes prefetched options). NO SSE emission; that's
    `_build_drafter_emit_callable`.

    Two stateless one-shot LLM calls per cycle: the selector picks N catalog
    targets + forced-event carrier, then the prose writer writes title/prose
    for each pinned slot. Both take `rt.log_entries` (last 2) as user-side
    context, not a message_history.
    """

    async def _drafter_llm(
        session: Session,
        cycle_index: int,
        catalog: list[tuple[str, str | None, dict[str, str]]],
        forced_event_id: str | None,
        beat_fragment: str | None,
    ) -> tuple[list[Option], bool]:
        assert AGENTS is not None

        # Consume prefetched options if tagged for this cycle.
        if (
            rt.prefetched_cycle_index == cycle_index
            and rt.prefetched_options is not None
        ):
            options = rt.prefetched_options
            engine_fallback = rt.prefetched_engine_fallback
            rt.prefetched_options = None
            rt.prefetched_engine_fallback = False
            return options, engine_fallback

        forced_event = (
            session.mission.event(forced_event_id) if forced_event_id else None
        )
        return await select_then_write_with_retries(
            AGENTS.selector,
            AGENTS.prose_writer,
            cycle_index=cycle_index,
            state=session.state,
            mission=session.mission,
            catalog=catalog,
            forced_event=forced_event,
            log_entries=rt.log_entries,
            recently_tried=list(rt.last_authorized),
            beat_fragment=beat_fragment,
        )

    return _drafter_llm


def _build_drafter_emit_callable(rt: SessionRuntime) -> Any:
    """Emit phase: paces each option's prose to the client as
    `option_started` + `option_word` ×N events, then `options_complete`.
    Also stores `options` on `rt.current_options` so `/authorize` can index it.
    """

    async def _drafter_emit(
        cycle_index: int,
        options: list[Option],
        engine_fallback: bool,
        forced_event_id: str | None,
    ) -> None:
        rt.current_options = options
        for option_idx, option in enumerate(options):
            await rt.out_queue.put(
                {
                    "event": "option_started",
                    "cycle": cycle_index,
                    "option_idx": option_idx,
                    "archetype": option.archetype,
                    "agent": option.agent,
                    "title": option.title,
                    "satisfies_event": option.satisfies_event,
                }
            )
            await paced_word_emitter(
                _words_from_text(option.prose),
                rt.out_queue.put,
                event_kind="option_word",
                extra={"cycle": cycle_index, "option_idx": option_idx},
            )
        await rt.out_queue.put(
            {
                "event": "options_complete",
                "cycle": cycle_index,
                "count": len(options),
                "has_forced_event": forced_event_id is not None,
                "event_id": forced_event_id,
                "engine_fallback": engine_fallback,
            }
        )

    return _drafter_emit


def _build_log_writer_callable(rt: SessionRuntime) -> Any:
    """Conversation-based log writer. Streams the cycle's case-file entry
    against `rt.log_message_history`, updates the persistent history with
    the new user+assistant pair, and appends a `LogEntry` to
    `rt.log_entries`.

    Returns `(text, pacing_task)` — the same shape as the older `RendererFn`,
    so cycle.py can continue scheduling the cycle-close + cross-cycle prefetch
    while paced `entry_word` events stream to the client.

    Wall-clock structure:
    - Text resolves the moment the LLM stream is exhausted (via `text_ready`),
      letting cycle.py advance past the await.
    - History updates happen synchronously inside the streaming generator —
      so prefetch (which fires before `await pacing_task`) sees the just-
      appended log entry in `rt.log_entries`.
    """

    async def _log_writer(
        session: Session,
        cycle_index: int,
        option: Option,
        delta: dict,
        prior_state: State,
        *,
        leans: list[str] = (),
    ) -> tuple[str, asyncio.Task]:
        assert AGENTS is not None
        await rt.out_queue.put(
            {
                "event": "entry_started",
                "cycle": cycle_index,
                "archetype": option.archetype,
            }
        )
        prompt = _log_writer_user_prompt(
            cycle_index, session.state, session.mission, option, delta,
            prior_state=prior_state,
            leans=list(leans),
        )

        # Snapshot the persistent history at the start of this call. Used
        # as `message_history` input to the stream; after the stream
        # completes, `stream_result.all_messages()` gives us the updated
        # full history (system + every cycle's entry, including this one).
        history_in = list(rt.log_message_history)
        text_ready: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        buf: list[str] = []

        async def _stream_and_pace() -> None:
            try:
                async with AGENTS.log_writer.run_stream(
                    prompt, message_history=history_in
                ) as stream_result:

                    async def teed() -> AsyncIterator[str]:
                        async for chunk in stream_result.stream_text(delta=True):
                            buf.append(chunk)
                            yield chunk
                        # Stream exhausted — text is complete. Append the
                        # LogEntry and update message_history synchronously
                        # so the prefetch hook (which fires before pacing
                        # completes) sees the just-rendered entry. Then
                        # resolve the future so cycle.py can advance.
                        full_text = "".join(buf)
                        rt.log_entries.append(
                            LogEntry(
                                cycle_index=cycle_index,
                                archetype=option.archetype,
                                text=full_text,
                            )
                        )
                        rt.log_message_history = stream_result.all_messages()
                        if not text_ready.done():
                            text_ready.set_result(full_text)

                    await paced_word_emitter(
                        _words_from_stream(teed()),
                        rt.out_queue.put,
                        event_kind="entry_word",
                        extra={"cycle": cycle_index},
                    )
            except BaseException as e:
                if not text_ready.done():
                    text_ready.set_exception(e)
                raise

        pacing_task = asyncio.create_task(_stream_and_pace())
        text = await text_ready

        return text, pacing_task

    return _log_writer


# ───────────────────────── cross-cycle prefetch ─────────────────────────


async def prefetch_next_cycle(rt: SessionRuntime, next_cycle_index: int) -> None:
    """Run cycle N+1's drafter LLM call in the background while the player
    reads cycle N's log entry. State is final after cycle N's
    `tick_cycle_close` + timeout re-check; the drafter is stateless; the only
    thing gating its call is the /continue UI gate.

    The drafter is stateless one-shot, so prefetching it is trivial. The log
    writer is conversation-based and cannot be prefetched ahead of the
    player's pick — we don't know the authorized option until /authorize.

    Results are stashed on `rt` (tagged with `next_cycle_index`) and consumed
    by the drafter LLM callable on the next `run_cycle` iteration.
    """
    assert AGENTS is not None
    state = rt.session.state
    if state.outcome is not None:
        return  # mission resolved; no next cycle
    mission = rt.session.mission

    with logfire.span("prefetch_next_cycle", cycle_index=next_cycle_index):
        catalog = valid_archetype_combinations(state, mission)
        forced = due_forced_events(state, mission)
        forced_event = forced[0] if forced else None
        # Read-only peek at the beat the live cycle-open will fire. State is
        # frozen between now and the player's /continue click — `select_beat`
        # returns the same beat `fire_beat` will commit, so the prefetched
        # options carry the same framing the live drafter would produce.
        beat = select_beat(state, mission)
        beat_fragment = beat.fragment if beat is not None else None
        options, fallback = await select_then_write_with_retries(
            AGENTS.selector,
            AGENTS.prose_writer,
            cycle_index=next_cycle_index,
            state=state,
            mission=mission,
            catalog=catalog,
            forced_event=forced_event,
            log_entries=rt.log_entries,
            recently_tried=list(rt.last_authorized),
            beat_fragment=beat_fragment,
        )
        rt.prefetched_options = options
        rt.prefetched_engine_fallback = fallback


def _build_log_text_ready_hook(rt: SessionRuntime) -> Any:
    """Fires when cycle N's log-writer LLM completes (state is final by the
    time cycle.py invokes it: tick_cycle_close + timeout re-check have run).
    Cancels any stale prefetch left over from earlier cycles, then schedules
    cycle N+1's prefetch in the background."""

    async def _hook(cycle_index: int) -> None:
        # Defensive: cancel any in-flight prefetch task from an earlier cycle.
        if rt.prefetch_task is not None and not rt.prefetch_task.done():
            rt.prefetch_task.cancel()
            with contextlib.suppress(BaseException):
                await rt.prefetch_task
        rt._clear_prefetch()
        next_idx = cycle_index + 1
        rt.prefetched_cycle_index = next_idx
        rt.prefetch_task = asyncio.create_task(prefetch_next_cycle(rt, next_idx))

    return _hook


# ───────────────────────── cycle loop ─────────────────────────


async def _run_session_loop(rt: SessionRuntime) -> None:
    """Drive cycles back-to-back until terminal. Run as an asyncio.Task per session."""
    assert AGENTS is not None
    mission = rt.session.mission
    await start_mission(rt.session, mission)
    await rt.out_queue.put(_mission_started_payload(mission, rt.session.state.qualities))
    cycle_index = 1
    # Per-session callable bundle. Built once: the drafter LLM phase / drafter
    # emit phase / log writer / prefetch-hook closures all capture `rt`, so
    # we don't need to rebuild them every iteration.
    drafter_llm_fn = _build_drafter_llm_callable(rt)
    drafter_emit_fn = _build_drafter_emit_callable(rt)
    log_writer_fn = _build_log_writer_callable(rt)
    on_log_text_ready_fn = _build_log_text_ready_hook(rt)
    try:
        while True:
            active = await run_cycle(
                rt.session,
                cycle_index,
                drafter_llm=drafter_llm_fn,
                drafter_emit=drafter_emit_fn,
                log_writer=log_writer_fn,
                emit=rt.out_queue.put,
                await_authorization=rt.await_auth,
                on_log_text_ready=on_log_text_ready_fn,
            )
            # After the cycle's chosen option (if any) lands on the event log,
            # extract its (archetype, agent) for the next cycle's RECENTLY
            # TRIED drafter prompt. Keep the last 2 picks only.
            last_auth = _last_authorized_pick(rt.session)
            if last_auth is not None and (
                not rt.last_authorized or rt.last_authorized[-1] != last_auth
            ):
                rt.last_authorized.append(last_auth)
                rt.last_authorized = rt.last_authorized[-2:]
            if not active:
                break
            # Gate the next cycle on the player clicking "Continue" — symmetric
            # with the in-cycle auth gate. Cycle N+1's LLM work is already in
            # flight via the prefetch task scheduled by `on_log_text_ready`,
            # so this gate now only waits on the player, not on LLM latency.
            await rt.out_queue.put({"event": "awaiting_continue", "cycle": cycle_index})
            await rt.await_continue()
            cycle_index += 1
    except asyncio.CancelledError:
        logfire.info("session loop cancelled", session_id=rt.session.id)
        raise
    except Exception:
        logfire.exception("session loop crashed", session_id=rt.session.id)
        await rt.out_queue.put({"event": "error", "code": "session_crashed"})
    finally:
        # Cancel any in-flight prefetch so it doesn't outlive the session.
        if rt.prefetch_task is not None and not rt.prefetch_task.done():
            rt.prefetch_task.cancel()
            with contextlib.suppress(BaseException):
                await rt.prefetch_task
        await rt.out_queue.put({"event": "session_closed"})


# ───────────────────────── endpoints ─────────────────────────


class CastEntry(BaseModel):
    """Player-facing cast descriptor surfaced to the frontend onboarding card.
    Agents, the handler (player), the boss, and named opposition all ship —
    the onboarding modal renders a roster so REDFIELD / MUNTEANU /
    MERIWETHER are known names before they appear in option titles or log
    entries."""

    id: str
    codename: str
    role: str
    file_number: str
    situation: str


class StartResponse(BaseModel):
    session_id: str
    mission_id: str
    mission_name: str
    codename: str
    max_cycles: int
    opening_briefing: str
    opening_prose: str
    cast: list[CastEntry]


def _cast_payload(mission: Mission) -> list[CastEntry]:
    """Build the full-cast payload for the onboarding card. Pydantic models
    serialise via `_json_default`'s `model_dump` branch — keeping these as
    CastEntry (not raw dicts) lets `StartResponse.cast` reuse the same list
    instead of rebuilding it."""
    return [
        CastEntry(
            id=m.id,
            codename=m.codename,
            role=m.role,
            file_number=m.file_number,
            situation=m.situation,
        )
        for m in mission.cast
    ]


def _mission_started_payload(mission: Mission, state_qualities: dict) -> dict:
    """Build the `mission_started` SSE payload. Includes the full cast and
    the boss's opening briefing so the client's onboarding card can render
    without a separate fetch. `agents` (agent-role only) stays for the
    sidebar's roster which only cares about operatives with cover."""
    return {
        "event": "mission_started",
        "mission_id": mission.id,
        "mission_name": mission.name,
        "codename": mission.codename,
        "max_cycles": mission.max_cycles,
        "agents": [
            {"id": a.id, "codename": a.codename, "file_number": a.file_number}
            for a in mission.agents
        ],
        "cast": _cast_payload(mission),
        "opening_briefing": mission.opening_briefing,
        "opening_prose": mission.opening_prose,
        "state": state_qualities,
    }


def _last_authorized_pick(session: Session) -> tuple[int, str, str | None] | None:
    """Pull the most recent (cycle_index, archetype, agent_id) from the
    session event log. Used to populate `rt.last_authorized` for the next
    cycle's RECENTLY TRIED drafter prompt. Returns None if no option has
    been authorized yet."""
    for ev in reversed(session.event_log):
        if isinstance(ev, OptionAuthorized):
            return (ev.cycle_index, ev.option.archetype, ev.option.agent)
    return None


class AuthorizeRequest(BaseModel):
    option_index: int


class StateResponse(BaseModel):
    session_id: str
    cycle_index: int | None
    state: dict[str, Any]
    outcome: str | None


app = FastAPI(title="HANDLER", version=_service_version, lifespan=lifespan)
# Health probes (Docker/k8s/proxy) hit these constantly and 404 — exclude them
# from tracing so the cloud isn't flooded with WARN spans for missing routes.
logfire.instrument_fastapi(
    app,
    capture_headers=True,
    excluded_urls=r"/healthz?$",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/mission/start", response_model=StartResponse)
async def mission_start() -> StartResponse:
    assert MISSION is not None and AGENTS is not None and SETTINGS is not None, "lifespan not run"
    persist = None
    if SETTINGS.session_save_path is not None:
        SETTINGS.session_save_path.parent.mkdir(parents=True, exist_ok=True)
        persist = make_jsonl_persistor(SETTINGS.session_save_path)
    session = make_session(MISSION, AGENTS, persist=persist)
    rt = SessionRuntime(session=session)
    SESSIONS[session.id] = rt
    rt.cycle_task = asyncio.create_task(_run_session_loop(rt))
    return StartResponse(
        session_id=session.id,
        mission_id=MISSION.id,
        mission_name=MISSION.name,
        codename=MISSION.codename,
        max_cycles=MISSION.max_cycles,
        opening_briefing=MISSION.opening_briefing,
        opening_prose=MISSION.opening_prose,
        cast=_cast_payload(MISSION),
    )


@app.get("/mission/{session_id}/stream")
async def mission_stream(session_id: str, request: Request) -> StreamingResponse:
    rt = _get_runtime(session_id)

    async def event_source() -> AsyncIterator[bytes]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(rt.out_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Heartbeat to keep the connection live and let us notice
                    # client disconnects between events.
                    yield b": heartbeat\n\n"
                    continue
                line = json.dumps(payload, default=_json_default).encode("utf-8")
                yield b"data: " + line + b"\n\n"
                if payload.get("event") == "session_closed":
                    break
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/mission/{session_id}/authorize")
async def mission_authorize(session_id: str, req: AuthorizeRequest) -> dict:
    rt = _get_runtime(session_id)
    if rt.auth_future is None or rt.auth_future.done():
        raise HTTPException(409, "no authorization pending")
    if rt.current_options is None:
        raise HTTPException(409, "no options currently drafted")
    if not (0 <= req.option_index < len(rt.current_options)):
        raise HTTPException(400, f"option_index out of range [0, {len(rt.current_options)})")
    chosen = rt.current_options[req.option_index]
    rt.auth_future.set_result(chosen)
    return {"accepted": True, "option_index": req.option_index, "archetype": chosen.archetype}


@app.post("/mission/{session_id}/continue")
async def mission_continue(session_id: str) -> dict:
    rt = _get_runtime(session_id)
    if rt.continue_future is None or rt.continue_future.done():
        raise HTTPException(409, "no continuation pending")
    rt.continue_future.set_result(None)
    return {"accepted": True}


@app.get("/mission/{session_id}/state", response_model=StateResponse)
async def mission_state(session_id: str) -> StateResponse:
    rt = _get_runtime(session_id)
    cycle_index: int | None = None
    for ev in reversed(rt.session.event_log):
        if hasattr(ev, "cycle_index"):
            cycle_index = ev.cycle_index  # type: ignore[attr-defined]
            break
    return StateResponse(
        session_id=session_id,
        cycle_index=cycle_index,
        state=rt.session.state.qualities,
        outcome=rt.session.state.outcome,
    )


# ───────────────────────── JSON helpers ─────────────────────────


def _json_default(o: Any) -> Any:
    """SSE payloads sometimes carry tuples (`delta` values), pydantic models
    (option payloads), or datetimes (authorize timestamps). Coerce to plain
    JSON-friendly shapes."""
    if isinstance(o, tuple):
        return list(o)
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")
