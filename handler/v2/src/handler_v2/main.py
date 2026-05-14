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

import contextlib
import importlib.metadata
import json
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import logfire
from logfire import ConsoleOptions
from fastapi import FastAPI

from .config import Settings
from .launcher import (
    LlamaStartupError,
    resolve_config,
    start_llama,
    stop_llama,
    wait_healthy,
)


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
    _service_version = importlib.metadata.version("handler-v2")
except importlib.metadata.PackageNotFoundError:
    _service_version = "unknown"
logfire.configure(
    service_name="handler-v2",
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
        # in M3 will point at. Computed here so the lifespan logs it once.
        if settings.use_inferno:
            api_url = f"{settings.inferno_url.rstrip('/')}/v1"
        else:
            api_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
        logfire.info("llama subsystem ready", api_url=api_url)
        app.state.api_url = api_url

        try:
            yield
        finally:
            logfire.info("stopping llama-server")
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
