"""FastAPI surface for HANDLER v2.

M0 — Skeleton. Lifespan boots the llama subsystem (bundled `llama-server`
OR inferno gateway probe) and exposes GET /health. World loading lands in
M1; scene-runner agent lands in M3; the rest of the SSE / scene API lands
in M4-M6.

Lifted from handler/src/handler/main.py: logfire setup (lines 84-99),
_probe_inferno_health (lines 116-133), lifespan span branching (lines
219-274). The externally-managed-llama branch (LAUNCH_LLAMA=false) is NOT
lifted — v2 has no `make sidecar` workflow.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import AsyncIterator

import httpx
import logfire
from logfire import ConsoleOptions
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

from .agents import (
    AgentBundle,
    LogWriterRuntime,
    SceneRunnerRuntime,
    build_scene_runner_bundle,
    make_log_writer_runtime,
    make_scene_runner_runtime,
)
from .config import Settings
from .launcher import (
    LlamaStartupError,
    resolve_config,
    start_llama,
    stop_llama,
    wait_healthy,
)
from .scene_loop import play_session
from .state import State, make_initial_state
from .world import World, load_world


_BOOT_SETTINGS = Settings()
logging.getLogger("httpx").setLevel(logging.WARNING)

# Lift: handler/src/handler/main.py:75-99 — Logfire creds + setup. Same
# pattern; just renamed service.
_logfire_creds = Path(__file__).resolve().parents[3] / ".logfire" / "logfire_credentials.json"
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
# `record="failure"` is critical — `record="all"` floods stdout during
# streaming chat completions and was the observed cause of a logfire-buffer
# blow-up that killed v1's sidecar mid-day. Don't change.
logfire.instrument_pydantic(record="failure")
logfire.instrument_httpx(capture_all=True)
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())

log = logging.getLogger(__name__)


# ───────────────────────── globals (set in lifespan) ─────────────────────────


SETTINGS: Settings | None = None
LLAMA_PROC: subprocess.Popen | None = None
INFERNO_PROBE_TIMEOUT_SECONDS = 5.0


# Single-session play per SPEC §7 (close-tab resets). One module-global
# SessionRuntime holds the live state. Subsequent POST /start cancels the
# previous task before replacing.
SESSION: "SessionRuntime | None" = None


@dataclass
class SessionRuntime:
    """Per-session live state for the SSE backend.

    The `event_queue` is single-consumer (per the lift note on
    `useStream`: React StrictMode dev-mode double-mounts; the
    `setTimeout(0)` deferred-fetch trick in the frontend prevents two
    consumers racing). The `pick_future` / `walk_future` are resolved by
    POST /pick / POST /walk respectively; the `play_task` awaits them
    from inside the pick_option/pick_walk callbacks passed to
    play_session.
    """

    bundle: AgentBundle
    scene_runner: SceneRunnerRuntime
    log_writer: LogWriterRuntime
    state: State
    world: World
    case_file_entries: list[str] = field(default_factory=list)
    event_queue: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)
    pick_future: "asyncio.Future[int] | None" = None
    walk_future: "asyncio.Future[str] | None" = None
    play_task: "asyncio.Task | None" = None


def build_model(api_url: str, model_name: str = "gemma-4-e4b") -> OpenAIChatModel:
    """Construct the OpenAIChatModel for both lifespan (FastAPI) and CLI
    (`__main__.py`). The OpenAIModelProfile flag is load-bearing for the
    conversation-based log writer — without
    `openai_chat_send_back_thinking_parts=False`, pydantic-ai's default
    'auto' round-trips ThinkingParts back into prior assistant content as
    <think>...</think> blocks, which Gemma 4 cannot tolerate in
    multi-turn (cf. v1 CLAUDE.md). The log writer's accumulating
    message_history makes this load-bearing — without the profile, drift
    compounds every scene.
    """
    base_profile = openai_model_profile(model_name)
    profile = dc_replace(base_profile, openai_chat_send_back_thinking_parts=False)
    provider = OpenAIProvider(base_url=api_url, api_key="local")
    return OpenAIChatModel(model_name, provider=provider, profile=profile)


async def _probe_inferno_health(url: str, timeout: float = INFERNO_PROBE_TIMEOUT_SECONDS) -> None:
    """One-shot probe of inferno's /health. Fail loud — when use_inferno is
    set there is no silent fallback to the bundled llama-server (rollback is
    `unset SMALLVILLE_USE_INFERNO; restart`).

    Lifted from handler/src/handler/main.py:116-133.
    """
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


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot llama subsystem (bundled llama-server OR inferno probe).

    M0 scope: llama lifecycle only. Model construction, world loading, and
    agent bundle wiring land in M3 — this lifespan is extended there.
    """
    global SETTINGS, LLAMA_PROC

    # Reuse the boot-time Settings instance — re-instantiating here would
    # re-read .env and risk divergence if env vars mutate between import
    # and lifespan startup.
    settings = _BOOT_SETTINGS
    SETTINGS = settings

    cfg = resolve_config(settings)
    with logfire.span(
        "llama_lifecycle",
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
        else:
            logfire.info("starting llama-server (bundled)")
            try:
                LLAMA_PROC = start_llama(cfg)
                llama_span.set_attribute("pid", LLAMA_PROC.pid)
                await wait_healthy(host=cfg.host, port=cfg.port)
                llama_span.set_attribute("healthy", True)
            except LlamaStartupError:
                logfire.exception("llama-server failed to come up")
                stop_llama(LLAMA_PROC)
                raise

        # api_url is the OpenAI-compatible endpoint the model construction
        # points at.
        if settings.use_inferno:
            api_url = f"{settings.inferno_url.rstrip('/')}/v1"
        else:
            api_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
        logfire.info("llama subsystem ready", api_url=api_url)
        app.state.api_url = api_url

        # Load world, build model + agent bundle once. The bundle is
        # world-scoped (one phase-2 agent per location) and reused
        # across sessions. The log-writer runtime is built per-session
        # in POST /start so message_history starts fresh.
        world_path = settings.world_path or Path(__file__).resolve().parents[2] / "world"
        logfire.info("loading world", path=str(world_path))
        world = load_world(world_path)
        logfire.info(
            "world loaded",
            locations=len(world.locations),
            npcs=len(world.npcs),
            beats=len(world.beats),
            endings=len(world.endings),
        )
        model = build_model(api_url)
        bundle = build_scene_runner_bundle(model, world)
        app.state.world = world
        app.state.model = model
        app.state.bundle = bundle
        # Pre-render the /world response once — the data is static for
        # the session and the frontend fetches it on every page load.
        app.state.world_meta = {
            "player_starts_at": world.brief.player_starts_at,
            "locations": [
                {
                    "id": loc.id,
                    "name": loc.name,
                    "adjacent_to": list(loc.adjacent_to),
                }
                for loc in world.locations.values()
            ],
        }

        try:
            yield
        finally:
            logfire.info("stopping llama-server")
            global SESSION
            if SESSION is not None and SESSION.play_task is not None:
                SESSION.play_task.cancel()
                with contextlib.suppress(BaseException):
                    await SESSION.play_task
            stop_llama(LLAMA_PROC)


# ───────────────────────── app ─────────────────────────


app = FastAPI(title="HANDLER v2", version=_service_version, lifespan=lifespan)
# Health probes hit /health constantly and 404 — exclude from tracing so the
# cloud isn't flooded with WARN spans for missing routes.
logfire.instrument_fastapi(
    app,
    capture_headers=True,
    excluded_urls=r"/healthz?$",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/world")
def get_world() -> dict:
    """Static world metadata the frontend needs to render location names
    and walk targets. Pre-rendered in lifespan; safe to fetch before
    starting a session."""
    return app.state.world_meta


# ───────────────────────── session API ─────────────────────────


class PickBody(BaseModel):
    option_idx: int


class WalkBody(BaseModel):
    to_location: str


# Sentinel placed on the event_queue when play_session terminates. The
# /events generator stops yielding when it sees this.
_DONE_SENTINEL: dict = {"event": "_session_done"}


@app.post("/start")
async def start_session() -> dict[str, str]:
    """Initialize a fresh play session. Cancels any previous session's
    play_task. Returns 202 with status; the actual prose stream comes
    over GET /events.
    """
    global SESSION
    if SESSION is not None and SESSION.play_task is not None and not SESSION.play_task.done():
        SESSION.play_task.cancel()
        with contextlib.suppress(BaseException):
            await SESSION.play_task

    world: World = app.state.world
    model: OpenAIChatModel = app.state.model
    bundle: AgentBundle = app.state.bundle
    scene_runner = make_scene_runner_runtime()
    log_writer = make_log_writer_runtime(model, world)
    state = make_initial_state(world)

    rt = SessionRuntime(
        bundle=bundle,
        scene_runner=scene_runner,
        log_writer=log_writer,
        state=state,
        world=world,
    )
    SESSION = rt

    async def emit(event: dict) -> None:
        await rt.event_queue.put(event)

    async def pick(_prose_part) -> int:
        rt.pick_future = asyncio.get_event_loop().create_future()
        try:
            return await rt.pick_future
        finally:
            rt.pick_future = None

    async def walk(_state, _world) -> str:
        rt.walk_future = asyncio.get_event_loop().create_future()
        try:
            return await rt.walk_future
        finally:
            rt.walk_future = None

    async def run() -> None:
        try:
            await play_session(
                bundle, scene_runner, log_writer, state, world,
                pick_option=pick,
                pick_walk=walk,
                opening_seed=world.brief.opening_seed,
                case_file_entries=rt.case_file_entries,
                emit_event=emit,
            )
        except asyncio.CancelledError:
            pass
        except BaseException as e:
            logfire.exception("play_session crashed", error=str(e))
            await rt.event_queue.put({"event": "error", "message": str(e)})
        finally:
            await rt.event_queue.put(_DONE_SENTINEL)

    rt.play_task = asyncio.create_task(run())
    return {"status": "started"}


@app.post("/pick")
async def pick(body: PickBody) -> dict[str, str]:
    """Resolve the in-flight pick_option future with the chosen option index."""
    if SESSION is None:
        raise HTTPException(409, "no active session — call POST /start first")
    if SESSION.pick_future is None or SESSION.pick_future.done():
        raise HTTPException(409, "no pick awaiting")
    SESSION.pick_future.set_result(body.option_idx)
    return {"status": "ok"}


@app.post("/walk")
async def walk(body: WalkBody) -> dict[str, str]:
    """Resolve the in-flight pick_walk future with the chosen destination
    location id. Validates adjacency defensively (the frontend already
    disables non-adjacent map clicks)."""
    if SESSION is None:
        raise HTTPException(409, "no active session — call POST /start first")
    if SESSION.walk_future is None or SESSION.walk_future.done():
        raise HTTPException(409, "no walk awaiting — scene is active")
    loc = SESSION.world.locations[SESSION.state.player_location]
    if body.to_location not in loc.adjacent_to:
        raise HTTPException(
            400, f"{body.to_location!r} is not adjacent to {loc.id!r}"
        )
    SESSION.walk_future.set_result(body.to_location)
    return {"status": "ok"}


@app.get("/events")
async def events() -> StreamingResponse:
    """SSE stream of session events. Single-consumer per session — the
    frontend's `useStream` hook uses the `setTimeout(0)` deferred-fetch
    trick to prevent React StrictMode dev-mode double-mount races."""
    if SESSION is None:
        raise HTTPException(409, "no active session — call POST /start first")

    rt = SESSION

    async def gen() -> AsyncIterator[bytes]:
        # Initial comment line — keeps proxies from buffering the
        # connection close; also gives `readSseStream` something to
        # discard cleanly on connect.
        yield b": connected\n\n"
        while True:
            event = await rt.event_queue.get()
            if event is _DONE_SENTINEL:
                yield b"data: " + json.dumps({"event": "session_done"}).encode() + b"\n\n"
                break
            yield b"data: " + json.dumps(event).encode() + b"\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
