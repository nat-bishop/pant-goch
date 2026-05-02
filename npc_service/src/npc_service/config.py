"""Runtime configuration loaded from env vars and an optional `.env` file.

Pydantic-settings reads env vars case-insensitively, so snake_case field names
here match uppercase env names (`launch_llama` ↔ `LAUNCH_LLAMA`). The .env file
is anchored at the repo root rather than CWD because the sidecar is launched
with mixed CWDs (Godot launches it with cwd=repo-root; `make sidecar` cd's into
npc_service/).

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

    # ───────── llama lifecycle ─────────
    launch_llama: bool = True
    llama_server_bin: Path | None = None
    llama_model_path: Path | None = None
    llama_hf_repo: str = "unsloth/gemma-4-E4B-it-GGUF:Q8_0"
    llama_port: int = 8080
    llama_ctx: int = 32768
    llama_ngl: int = 99
    llama_extra_args: str = ""

    # ───────── pydantic-ai wiring (override only when pointing at remote llama) ─────────
    llama_base_url: str | None = None

    scenario_path: Path | None = None

    # ───────── history compaction (rarely tuned at runtime) ─────────
    # summarize_when_large fires when input would exceed this share of LLAMA_CTX.
    summary_trigger_frac: float = 0.8
    # Absolute trigger override (tokens). 0 → compute from summary_trigger_frac * llama_ctx.
    summary_trigger_tokens: int = 0
    # Trailing messages kept verbatim past the recap.
    summary_keep_tail: int = 4
    # Whole-stream timeout for /action; surface error rather than hang the SSE.
    action_stream_timeout_seconds: float = 120.0
    # Whole-call timeout for /options; bounds how long ACTION_LOCK can be held by
    # an options call so a stalled model can't lock out /action indefinitely.
    options_timeout_seconds: float = 30.0
