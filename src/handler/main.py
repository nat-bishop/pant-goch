"""FastAPI surface for HANDLER v2.

Lifespan boots the llama subsystem (bundled `llama-server` OR inferno
gateway probe), loads the world, builds the agent bundle, and exposes the
SSE scene API + static asset routes.
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
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .agents import (
    AgentBundle,
    ChroniclerRuntime,
    SceneRunnerRuntime,
    build_scene_runner_bundle,
    make_chronicler_runtime,
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

# Logfire creds live at <project-root>/.logfire/logfire_credentials.json.
# src/handler/main.py → parents[2] = project root.
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
    """Per-session async glue for the SSE backend.

    Holds only what binds the FastAPI side to the scene loop: the live
    state, the three runtimes, the event queue, and the pick/walk
    futures resolved by POST /pick and POST /walk. Session-persistent
    data (case_file_entries, npc_memory, dialogue_history, ...) lives
    on `state` so the save-game shape is unambiguous.

    The `event_queue` is single-consumer (per the lift note on
    `useStream`: React StrictMode dev-mode double-mounts; the
    `setTimeout(0)` deferred-fetch trick in the frontend prevents two
    consumers racing).
    """

    bundle: AgentBundle
    scene_runner: SceneRunnerRuntime
    chronicler: ChroniclerRuntime
    state: State
    world: World
    event_queue: "asyncio.Queue[dict]" = field(default_factory=asyncio.Queue)
    pick_future: "asyncio.Future[int] | None" = None
    walk_future: "asyncio.Future[str] | None" = None
    free_text_future: "asyncio.Future[str] | None" = None
    play_task: "asyncio.Task | None" = None


def build_model(api_url: str, model_name: str = "gemma-4-e4b") -> OpenAIChatModel:
    """Construct the OpenAIChatModel for both lifespan (FastAPI) and CLI
    (`__main__.py`).

    Two thinking-suppression mechanisms are stacked here, because the
    failure modes are different and we hit both:

    1. `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` —
       stops pydantic-ai from echoing past ThinkingParts back into prior
       assistant content as <think>...</think> blocks on subsequent
       turns. Without it, the chronicler's accumulating message_history
       drifts every scene (cf. v1 CLAUDE.md).

    2. `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` —
       per-request body param forwarded by the OpenAI SDK to
       llama-server's `/v1/chat/completions`. llama.cpp passes
       `chat_template_kwargs` into the Jinja chat template; the Gemma 4
       template (and Qwen3 / GLM and other recent reasoning-capable
       models) gates its reasoning preamble on `enable_thinking`. This
       is the only lever that works when the model is served by inferno
       (or any external gateway) — the launcher's `--reasoning off`
       flag is bypassed in that mode. Without this, ~100 of every Phase 2
       call's output tokens are burned on a hidden "Plan / Constraint
       Checklist" preamble that never reaches the player.
    """
    base_profile = openai_model_profile(model_name)
    profile = dc_replace(base_profile, openai_chat_send_back_thinking_parts=False)
    provider = OpenAIProvider(base_url=api_url, api_key="local")
    settings = ModelSettings(
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return OpenAIChatModel(
        model_name, provider=provider, profile=profile, settings=settings,
    )


async def _probe_inferno_health(url: str, timeout: float = INFERNO_PROBE_TIMEOUT_SECONDS) -> None:
    """One-shot probe of inferno's /health. Fail loud — when use_inferno is
    set there is no silent fallback to the bundled llama-server (rollback is
    `unset SMALLVILLE_USE_INFERNO; restart`).

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
        # world-scoped and reused across sessions. The session-scoped
        # runtimes (scene_runner, chronicler) are built per-session in
        # POST /start so message_history starts fresh.
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
                    "map_xy": (
                        list(world.map_coords[loc.id])
                        if loc.id in world.map_coords
                        else None
                    ),
                }
                for loc in world.locations.values()
            ],
            "has_map": bool(world.map_coords),
            "npcs": [
                {"id": n.id, "name": n.name, "default_location": n.default_location}
                for n in world.npcs.values()
            ],
            "glosses": [g.model_dump() for g in world.glosses],
        }
        # Resolve the map asset path once at lifespan. Stays None if the
        # world has no drawn map; the frontend's /world response will set
        # has_map=false and fall back to the text walk panel.
        map_png_path = world_path / "map.png"
        app.state.map_png = map_png_path if map_png_path.is_file() else None
        # Backdrops directory — one PNG per location, named <location_id>.png.
        # Optional and partial: the route 404s for locations without an asset
        # and the frontend falls back to a neutral dark vignette.
        backdrops_dir = world_path / "backdrops"
        app.state.backdrops_dir = backdrops_dir if backdrops_dir.is_dir() else None

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


@app.get("/world/map.png")
def get_world_map_png() -> FileResponse:
    """Serve the drawn village map. The frontend overlays clickable
    hotspots over this background using the fractional coordinates
    returned by GET /world. 404 if the world has no map authored."""
    map_path = app.state.map_png
    if map_path is None:
        raise HTTPException(404, "no map authored for this world")
    return FileResponse(map_path, media_type="image/png")


@app.get("/world/cover.png")
def get_world_cover() -> FileResponse:
    """Frontispiece backdrop shown in the left pane while the title page
    is up. Served from `backdrops/cover.png`. 404 when no asset is
    authored — frontend falls back to the neutral dark vignette."""
    backdrops_dir = app.state.backdrops_dir
    if backdrops_dir is None:
        raise HTTPException(404, "no backdrops directory for this world")
    cover_path = backdrops_dir / "cover.png"
    if not cover_path.is_file():
        raise HTTPException(404, "no cover authored for this world")
    return FileResponse(cover_path, media_type="image/png")


@app.get("/world/backdrops/{location_id}.png")
def get_world_backdrop(location_id: str) -> FileResponse:
    """Per-location backdrop image shown in the left pane. 404 when no
    asset is authored for the given location — frontend falls back to a
    neutral vignette. `location_id` is grammar-checked against the
    loaded world to keep this route from doubling as a generic file
    server."""
    backdrops_dir = app.state.backdrops_dir
    if backdrops_dir is None:
        raise HTTPException(404, "no backdrops directory for this world")
    if location_id not in app.state.world.locations:
        raise HTTPException(404, f"unknown location {location_id!r}")
    backdrop_path = backdrops_dir / f"{location_id}.png"
    if not backdrop_path.is_file():
        raise HTTPException(404, f"no backdrop authored for {location_id!r}")
    return FileResponse(backdrop_path, media_type="image/png")


# ───────────────────────── session API ─────────────────────────


class PickBody(BaseModel):
    option_idx: int


class WalkBody(BaseModel):
    to_location: str


class FreeTextBody(BaseModel):
    text: str


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
    bundle: AgentBundle = app.state.bundle
    scene_runner = make_scene_runner_runtime()
    chronicler = make_chronicler_runtime()
    state = make_initial_state(world)

    rt = SessionRuntime(
        bundle=bundle,
        scene_runner=scene_runner,
        chronicler=chronicler,
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

    async def free_text(_req) -> str:
        rt.free_text_future = asyncio.get_event_loop().create_future()
        try:
            return await rt.free_text_future
        finally:
            rt.free_text_future = None

    async def run() -> None:
        try:
            await play_session(
                bundle, scene_runner, chronicler, state, world,
                pick_option=pick,
                pick_walk=walk,
                pick_free_text=free_text,
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


@app.post("/free_text")
async def free_text(body: FreeTextBody) -> dict[str, str]:
    """Resolve the in-flight pick_free_text future with the player's
    typed line. The next turn's prose writer sees the typed string as
    the player's pick."""
    if SESSION is None:
        raise HTTPException(409, "no active session — call POST /start first")
    if SESSION.free_text_future is None or SESSION.free_text_future.done():
        raise HTTPException(409, "no free-text awaiting")
    if not body.text.strip():
        raise HTTPException(400, "typed text must not be empty")
    SESSION.free_text_future.set_result(body.text.strip())
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
