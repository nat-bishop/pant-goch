"""FastAPI surface for HANDLER. Phase 0 leaves the app skeleton + llama
lifespan + load-bearing model-profile flag wired up. Phase 3 adds the
endpoints (`/mission/start`, `/mission/{id}/stream` SSE, `/mission/{id}/authorize`,
`/mission/{id}/state`) and the `paced_word_emitter` SSE helper.
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

import logfire
from logfire import ConsoleOptions
from fastapi import FastAPI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider
from dataclasses import replace

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

# Logfire credentials live next to the handler project root.
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
    # Single-player local game with no PII; default scrubbing would redact
    # the prose we want to inspect.
    scrubbing=False,
)
logfire.instrument_pydantic_ai()
logfire.instrument_pydantic()
logfire.instrument_httpx(capture_all=True)
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())

# A pre-tagged Logfire instance for spans/events that wrap LLM work — filter
# by the `llm` tag in the Logfire UI to surface every agent call across traces.
LLM_LOGFIRE = logfire.with_tags("llm")


# ───────────────────────── globals (set in lifespan) ─────────────────────────


SETTINGS: Settings | None = None
LLAMA_PROC: subprocess.Popen | None = None
EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS = 2.0  # tests set this to 0.1 for fast boot


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boots llama-server (or probes externally-managed), builds the shared
    OpenAIChatModel, then hands control to the app. Phase 2 attaches the
    three pydantic-ai agents to this model; Phase 3 wires endpoints to them.
    """
    global SETTINGS, LLAMA_PROC

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
        # Local-Gemma via llama-server: don't round-trip prior-turn thoughts.
        # Default 'auto' re-serialises ThinkingParts as <think>…</think> in
        # historical assistant content, violating Google's Gemma 4 multi-turn
        # rule (https://ai.google.dev/gemma/docs/capabilities/thinking) and
        # inflating input tokens by hundreds per turn. Replace on the default
        # profile (which carries supports_json_schema_output=True, etc.).
        base_profile = openai_model_profile(model_name)
        profile = replace(base_profile, openai_chat_send_back_thinking_parts=False)
        provider = OpenAIProvider(base_url=api_url, api_key="local")
        model = OpenAIChatModel(model_name, provider=provider, profile=profile)
        app.state.model = model

        # Phase 2 attaches the three pydantic-ai agents; Phase 3 wires
        # endpoints to call them through the shared model.

        try:
            yield
        finally:
            logfire.info("stopping llama-server")
            stop_llama(LLAMA_PROC)


# ───────────────────────── app ─────────────────────────


app = FastAPI(title="HANDLER", version=_service_version, lifespan=lifespan)
logfire.instrument_fastapi(app, capture_headers=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Phase 3 endpoints (POST /mission/start, GET /mission/{id}/stream,
# POST /mission/{id}/authorize, GET /mission/{id}/state) and the
# `paced_word_emitter` SSE helper go here.
