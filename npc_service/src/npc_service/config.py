"""Runtime configuration loaded from env vars and an optional `.env` file.

Pydantic-settings reads env vars case-insensitively, so snake_case field names
here match uppercase env names (`launch_llama` ↔ `LAUNCH_LLAMA`). The .env file
is anchored at the repo root rather than CWD because the sidecar is launched
with mixed CWDs (Godot launches it with cwd=repo-root; `make sidecar` cd's into
npc_service/).

Most of these fields are tuned once and forgotten — context size, slot count,
ports, history-compaction tunings. Only a handful are actually toggled at
runtime: `launch_llama` (the `make sidecar` workflow + tests), `scenario_path`
(test fixtures, playtester), `npc_logfire_console` (conftest sets false), and
`session_save_path` (opt-in JSONL persistence). Pydantic-settings still owns
the lot so they all share one typed parsing path and re-bind cleanly inside
FastAPI's `lifespan` (which is what makes `monkeypatch.setenv(...)` work in
tests). The README documents only the toggle-able ones.

Instantiate `Settings()` fresh at each entry point (lifespan, uvicorn bootstrap)
rather than caching at module scope — tests rely on `monkeypatch.setenv(...)`
inside fixtures, which only takes effect for instantiations that happen after
the fixture runs.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# src layout: src/npc_service/config.py → workspace root is parents[3].
# parents[0]=src/npc_service, [1]=src, [2]=npc_service, [3]=<workspace>.
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ───────── derived constants ─────────
# Convenience exports for call sites that don't want to instantiate `Settings`
# (e.g. argparse `default=...` in the playtest CLI; the integration-test
# fixture URL in tests/conftest.py). These are the always-computed defaults;
# `Settings` below owns the env-override surface.
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"
EVAL_LLAMA_PORT = 8081
EVAL_LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{EVAL_LLAMA_PORT}/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ───────── service ─────────
    npc_service_port: int = 8000
    npc_log_level: str = "info"
    # True (default) prints Logfire spans + log records to stderr for fast local
    # feedback; tests set this to false to keep pytest output uncluttered.
    npc_logfire_console: bool = True

    # ───────── llama lifecycle ─────────
    launch_llama: bool = True
    llama_server_bin: Path | None = None
    llama_model_path: Path | None = None
    llama_hf_repo: str = "ggml-org/gemma-4-E4B-it-GGUF:Q8_0"
    llama_host: str = LLAMA_HOST
    llama_port: int = LLAMA_PORT
    llama_ctx: int = 32768
    llama_ngl: int = 99
    # Number of parallel decoding slots in llama-server (`-np N`). Each slot
    # can serve one in-flight request; past N, requests queue server-side.
    # On a 96 GB Blackwell with `--kv-unified` we comfortably go to 128.
    # See benchmark in plan/migration-day notes.
    llama_n_parallel: int = 128
    # Unified KV cache (`--kv-unified`): one ctx-sized KV buffer shared across
    # all slots via paging. Without this, each slot gets its own ctx-sized KV
    # and VRAM scales `N × ctx × hidden` — fast OOM at high `-np`. The
    # paged path is well-tested in current llama.cpp; default ON.
    llama_kv_unified: bool = True
    # Escape hatch for any flag we haven't promoted to a typed Settings field.
    # Whitespace-split, appended verbatim to the llama-server command.
    llama_extra_args: str = ""

    # ───────── pydantic-ai wiring (override only when pointing at remote llama) ─────────
    llama_base_url: str | None = None

    # ───────── eval llama (playtester second server, Qwen on ik_llama.cpp) ─────────
    # Powers the synthetic-player agent and the LLM judges. Different family from
    # the engine SUT (Gemma) to mitigate same-family preference leakage in judging
    # — see arxiv 2502.01534. Default points at port 8081 so the engine llama on
    # 8080 stays unaffected. Lifecycle is user-managed (`make llama-eval`); the
    # playtester just probes /health on this URL before starting.
    eval_llama_port: int = EVAL_LLAMA_PORT
    eval_llama_base_url: str | None = None

    scenario_path: Path | None = None

    # ───────── history compaction ─────────
    # When the chat-agent's message_history would exceed this share of llama_ctx,
    # the oldest messages get summarised by a sibling agent and replaced with a
    # single recap. Token budget is computed at boot from llama_ctx.
    history_compact_frac: float = 0.8
    # Most recent N messages always kept verbatim past the recap so the model
    # sees the immediate exchange without paraphrase loss.
    history_compact_keep_tail: int = 4

    # ───────── session persistence (off by default) ─────────
    # When set, every committed Event on the default session is appended as
    # one JSON object per line to <session_save_path>. On boot, lifespan
    # replays the JSONL into a fresh session before yielding. Unset → no
    # disk I/O (in-memory only).
    session_save_path: Path | None = None
