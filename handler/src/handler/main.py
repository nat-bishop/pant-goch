"""FastAPI surface for HANDLER.

Endpoints:
  POST   /mission/start                 → boot a new session, return session_id
  GET    /mission/{session_id}/stream   → SSE event stream for that session
  POST   /mission/{session_id}/authorize → record the player's option pick
  GET    /mission/{session_id}/state    → snapshot for reconnect

The cycle loop runs as a background asyncio.Task per session, pushing events
into the session's `out_queue`. The SSE endpoint consumes from the queue.
`/authorize` resolves the cycle's pending `auth_future`, unblocking the loop
to compute effects + render dispatch + advance to the next cycle.
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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider
from dataclasses import replace

from .agents import (
    AgentBundle,
    DispatchRecord,
    _drafter_user_prompt,
    _renderer_user_prompt,
    _sop_user_prompt,
    build_agent_bundle,
    draft_with_retries,
)
from .config import Settings
from .cycle import run_cycle, start_mission
from .events import (
    DispatchRendered,
    Event,
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
from .spine import due_forced_events, valid_archetype_combinations


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
    for the player to click through), the cycle task itself, and a rolling
    buffer of past dispatches (for renderer voice anchoring).

    Cross-cycle prefetch (HANDLER_SPEC_V2 §11 wall-clock optimisation): when
    cycle N's renderer LLM finishes, a background task pre-runs cycle N+1's
    SoP + drafter LLM calls so they're ready by the time the player clicks
    Continue. Results are tagged with `prefetched_cycle_index` so the SoP /
    drafter callables only consume data meant for the cycle they're running.
    """

    session: Session
    out_queue: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)
    auth_future: asyncio.Future[Option] | None = None
    continue_future: asyncio.Future[None] | None = None
    current_options: list[Option] | None = None
    cycle_task: asyncio.Task | None = None
    dispatches: list[DispatchRecord] = field(default_factory=list)
    # Cross-cycle prefetch fields — populated by `prefetch_next_cycle`,
    # consumed by the SoP / drafter callables on the next run_cycle iteration.
    prefetched_cycle_index: int | None = None
    prefetched_sop_text: str | None = None
    prefetched_options: list[Option] | None = None
    prefetched_engine_fallback: bool = False
    prefetch_task: asyncio.Task | None = None

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
        """Reset all prefetch fields (after consumption, cancellation, or a
        stale-result discard)."""
        self.prefetched_cycle_index = None
        self.prefetched_sop_text = None
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
    1. SoP / Dispatch: `word_source = _words_from_stream(llm_stream)` — words
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


# ───────────────────────── agent call wrappers (the SopFn/DrafterFn/RendererFn shapes) ─────────────────────────


async def _run_paced_stream(
    agent: Any,
    prompt: str,
    emit: Callable[[dict], Awaitable[None]],
    *,
    event_kind: str,
    extra: dict,
) -> tuple[str, asyncio.Task]:
    """Stream `agent.run_stream(prompt)` to the client as paced word events,
    but resolve the full text the moment the LLM stream is exhausted — not
    when pacing finishes. Returns `(text, pacing_task)` so the caller can
    kick off downstream work (drafter LLM or cross-cycle prefetch) while
    paced word emission continues in the background.
    """
    text_ready: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    buf: list[str] = []

    async def _stream_and_pace() -> None:
        try:
            async with agent.run_stream(prompt) as result:

                async def teed() -> AsyncIterator[str]:
                    async for chunk in result.stream_text(delta=True):
                        buf.append(chunk)
                        yield chunk
                    if not text_ready.done():
                        text_ready.set_result("".join(buf))

                await paced_word_emitter(
                    _words_from_stream(teed()),
                    emit,
                    event_kind=event_kind,
                    extra=extra,
                )
        except BaseException as e:
            if not text_ready.done():
                text_ready.set_exception(e)
            raise

    pacing_task = asyncio.create_task(_stream_and_pace())
    text = await text_ready
    return text, pacing_task


def _build_sop_callable(rt: SessionRuntime) -> Any:
    """Returns the SoP text as soon as the LLM finishes generating, and a
    pacing_task that continues emitting paced `sop_word` SSE events in the
    background. The caller can start the drafter LLM (which needs sop_text)
    while pacing finishes — net wall-clock win equal to ~drafter LLM time.

    If `rt.prefetched_sop_text` was populated by `prefetch_next_cycle` for
    this cycle, skip the LLM call entirely and just pace the buffered text.
    """

    async def _sop(session: Session, cycle_index: int) -> tuple[str, asyncio.Task]:
        assert AGENTS is not None
        extra = {"cycle": cycle_index}

        # Consume prefetched data if it's tagged for this cycle.
        if (
            rt.prefetched_cycle_index == cycle_index
            and rt.prefetched_sop_text is not None
        ):
            text = rt.prefetched_sop_text
            rt.prefetched_sop_text = None  # drafter still needs prefetched_options
            pacing_task = asyncio.create_task(
                paced_word_emitter(
                    _words_from_text(text),
                    rt.out_queue.put,
                    event_kind="sop_word",
                    extra=extra,
                )
            )
            return text, pacing_task

        prompt = _sop_user_prompt(cycle_index, session.state, session.mission, rt.dispatches)
        return await _run_paced_stream(
            AGENTS.sop, prompt, rt.out_queue.put,
            event_kind="sop_word", extra=extra,
        )

    return _sop


def _build_drafter_llm_callable(rt: SessionRuntime) -> Any:
    """LLM-only phase of the drafter — runs `draft_with_retries` (or consumes
    prefetched options). NO SSE emission; that's `_build_drafter_emit_callable`.

    Splitting LLM from emission lets `cycle.py` schedule this as a background
    task while SoP pacing finishes, so options arrive at the client with no
    idle gap after SoP.
    """

    async def _drafter_llm(
        session: Session,
        cycle_index: int,
        sop_text: str,
        catalog: list[tuple[str, str | None, dict[str, str]]],
        forced_event_id: str | None,
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

        forced_event = None
        if forced_event_id is not None:
            forced_event = next(
                (e for e in session.mission.events if e.id == forced_event_id), None
            )
        return await draft_with_retries(
            AGENTS.drafter,
            cycle_index=cycle_index,
            state=session.state,
            mission=session.mission,
            sop_text=sop_text,
            catalog=catalog,
            forced_event=forced_event,
            dispatches=rt.dispatches,
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
                    "params": option.params,
                    "title": option.title,
                    "satisfies_moment": option.satisfies_moment,
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
                "has_forced_moment": forced_event_id is not None,
                "moment_id": forced_event_id,
                "engine_fallback": engine_fallback,
            }
        )

    return _drafter_emit


def _build_renderer_callable(rt: SessionRuntime) -> Any:
    """Returns the dispatch text as soon as the LLM finishes, and a
    pacing_task that continues emitting paced `dispatch_word` SSE events.
    Cycle.py uses the LLM-done signal to fire `on_renderer_text_ready`,
    which schedules cycle N+1's prefetch — overlapping cycle N+1's LLM
    work with cycle N's modal-display wall time.

    The dispatch is appended to `rt.dispatches` synchronously once text is
    ready (before pacing finishes), so prefetch sees the just-rendered
    dispatch when building cycle N+1's user prompts.
    """

    async def _render(
        session: Session,
        cycle_index: int,
        option: Option,
        delta: dict,
        acting: str,
        prior_state,
    ) -> tuple[str, asyncio.Task]:
        assert AGENTS is not None
        await rt.out_queue.put(
            {
                "event": "dispatch_started",
                "cycle": cycle_index,
                "acting": acting,
                "archetype": option.archetype,
            }
        )
        prompt = _renderer_user_prompt(
            cycle_index, session.state, session.mission, option, delta, acting, rt.dispatches,
            prior_state=prior_state,
        )
        text, pacing_task = await _run_paced_stream(
            AGENTS.renderer, prompt, rt.out_queue.put,
            event_kind="dispatch_word",
            extra={"cycle": cycle_index, "acting": acting},
        )
        # Append synchronously so any prefetch kicked off by the renderer-done
        # hook sees this dispatch in `rt.dispatches`.
        rt.dispatches.append(
            DispatchRecord(
                cycle_index=cycle_index,
                acting=acting,
                archetype=option.archetype,
                text=text,
            )
        )
        return text, pacing_task

    return _render


# ───────────────────────── cross-cycle prefetch ─────────────────────────


async def prefetch_next_cycle(rt: SessionRuntime, next_cycle_index: int) -> None:
    """Run cycle N+1's SoP + drafter LLM calls in the background while the
    player reads cycle N's dispatch modal. State is final after cycle N's
    `tick_cycle_close` + timeout re-check; agents are stateless; the only
    thing gating these calls is the /continue UI gate.

    Results are stashed on `rt` (tagged with `next_cycle_index`) and consumed
    by the SoP / drafter callables on the next `run_cycle` iteration.
    """
    assert AGENTS is not None
    state = rt.session.state
    if state.outcome is not None:
        return  # mission resolved; no next cycle
    mission = rt.session.mission

    with logfire.span("prefetch_next_cycle", cycle_index=next_cycle_index):
        sop_prompt = _sop_user_prompt(next_cycle_index, state, mission, rt.dispatches)
        sop_result = await AGENTS.sop.run(sop_prompt)
        rt.prefetched_sop_text = sop_result.output

        catalog = valid_archetype_combinations(state, mission)
        forced = due_forced_events(state, mission)
        forced_event = forced[0] if forced else None
        options, fallback = await draft_with_retries(
            AGENTS.drafter,
            cycle_index=next_cycle_index,
            state=state,
            mission=mission,
            sop_text=rt.prefetched_sop_text,
            catalog=catalog,
            forced_event=forced_event,
            dispatches=rt.dispatches,
        )
        rt.prefetched_options = options
        rt.prefetched_engine_fallback = fallback


def _build_renderer_text_ready_hook(rt: SessionRuntime) -> Any:
    """Fires when cycle N's renderer LLM completes (state is final by the
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
    await rt.out_queue.put(
        {
            "event": "mission_started",
            "mission_id": mission.id,
            "mission_name": mission.name,
            "codename": mission.codename,
            "max_cycles": mission.max_cycles,
            "agents": [{"id": a.id, "codename": a.codename, "file_number": a.file_number} for a in mission.agents],
            "state": rt.session.state.qualities,
        }
    )
    cycle_index = 1
    # Per-session callable bundle. Built once: the SoP / drafter LLM phase /
    # drafter emit phase / renderer / prefetch-hook closures all capture `rt`,
    # so we don't need to rebuild them every iteration.
    sop_fn = _build_sop_callable(rt)
    drafter_llm_fn = _build_drafter_llm_callable(rt)
    drafter_emit_fn = _build_drafter_emit_callable(rt)
    renderer_fn = _build_renderer_callable(rt)
    on_renderer_text_ready_fn = _build_renderer_text_ready_hook(rt)
    try:
        while True:
            active = await run_cycle(
                rt.session,
                cycle_index,
                sop=sop_fn,
                drafter_llm=drafter_llm_fn,
                drafter_emit=drafter_emit_fn,
                renderer=renderer_fn,
                emit=rt.out_queue.put,
                await_authorization=rt.await_auth,
                on_renderer_text_ready=on_renderer_text_ready_fn,
            )
            if not active:
                break
            # Gate the next cycle on the player clicking "Continue" — symmetric
            # with the in-cycle auth gate. Cycle N+1's LLM work is already in
            # flight via the prefetch task scheduled by `on_renderer_text_ready`,
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


class StartResponse(BaseModel):
    session_id: str
    mission_id: str
    mission_name: str
    codename: str
    max_cycles: int


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
