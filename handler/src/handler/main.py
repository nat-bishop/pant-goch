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
MISSION: Mission | None = None
AGENTS: AgentBundle | None = None


@dataclass
class SessionRuntime:
    """Per-session live state: the engine session, the SSE queue, the two
    futures the cycle loop alternates between (auth = pause INSIDE a cycle
    waiting for the player's pick; continue = pause BETWEEN cycles waiting
    for the player to click through), the cycle task itself, and a rolling
    buffer of past dispatches (for renderer voice anchoring)."""

    session: Session
    out_queue: asyncio.Queue[dict] = field(default_factory=asyncio.Queue)
    auth_future: asyncio.Future[Option] | None = None
    continue_future: asyncio.Future[None] | None = None
    current_options: list[Option] | None = None
    cycle_task: asyncio.Task | None = None
    dispatches: list[DispatchRecord] = field(default_factory=list)

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
        port=cfg.port,
        ctx=cfg.ctx,
        model_path=str(cfg.model_path) if cfg.model_path else None,
        hf_repo=cfg.hf_repo,
    ) as llama_span:
        if settings.launch_llama:
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


def _build_sop_callable(rt: SessionRuntime) -> Any:
    async def _sop(session: Session, cycle_index: int) -> str:
        assert AGENTS is not None
        prompt = _sop_user_prompt(cycle_index, session.state, session.mission, rt.dispatches)
        async with AGENTS.sop.run_stream(prompt) as result:
            text = await paced_word_emitter(
                _words_from_stream(result.stream_text(delta=True)),
                rt.out_queue.put,
                event_kind="sop_word",
                extra={"cycle": cycle_index},
            )
        return text
    return _sop


def _build_drafter_callable(rt: SessionRuntime) -> Any:
    async def _draft(
        session: Session,
        cycle_index: int,
        sop_text: str,
        catalog: list[tuple[str, str | None, dict[str, str]]],
        forced_moment_id: str | None,
    ) -> tuple[list[Option], bool]:
        assert AGENTS is not None
        forced_moment = None
        if forced_moment_id is not None:
            forced_moment = next(
                (m for m in session.mission.moments if m.id == forced_moment_id), None
            )
        options, engine_fallback = await draft_with_retries(
            AGENTS.drafter,
            cycle_index=cycle_index,
            state=session.state,
            mission=session.mission,
            sop_text=sop_text,
            catalog=catalog,
            forced_moment=forced_moment,
            dispatches=rt.dispatches,
        )
        # Reveal each option's prose word-by-word over SSE.
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
                "has_forced_moment": forced_moment_id is not None,
                "moment_id": forced_moment_id,
                "engine_fallback": engine_fallback,
            }
        )
        rt.current_options = options
        return options, engine_fallback
    return _draft


def _build_renderer_callable(rt: SessionRuntime) -> Any:
    async def _render(
        session: Session,
        cycle_index: int,
        option: Option,
        delta: dict,
        acting: str,
    ) -> str:
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
        )
        async with AGENTS.renderer.run_stream(prompt) as result:
            text = await paced_word_emitter(
                _words_from_stream(result.stream_text(delta=True)),
                rt.out_queue.put,
                event_kind="dispatch_word",
                extra={"cycle": cycle_index, "acting": acting},
            )
        rt.dispatches.append(
            DispatchRecord(
                cycle_index=cycle_index,
                acting=acting,
                archetype=option.archetype,
                text=text,
            )
        )
        return text
    return _render


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
    try:
        while True:
            active = await run_cycle(
                rt.session,
                cycle_index,
                sop=_build_sop_callable(rt),
                drafter=_build_drafter_callable(rt),
                renderer=_build_renderer_callable(rt),
                emit=rt.out_queue.put,
                await_authorization=rt.await_auth,
            )
            if not active:
                break
            # Gate the next cycle on the player clicking "Continue" — symmetric
            # with the in-cycle auth gate. Without this, the engine would loop
            # straight into cycle N+1's SoP while the player is still reading
            # the dispatch they just authorized.
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
