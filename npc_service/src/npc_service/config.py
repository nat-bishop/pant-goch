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
    # True (default) prints Logfire spans + log records to stderr for fast local
    # feedback; tests set this to false to keep pytest output uncluttered.
    npc_logfire_console: bool = True

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

    # ───────── history compaction ─────────
    # When the chat-agent's message_history would exceed this share of llama_ctx,
    # the oldest messages get summarised by a sibling agent and replaced with a
    # single recap. Token budget is computed at boot from llama_ctx.
    history_compact_frac: float = 0.8
    # Most recent N messages always kept verbatim past the recap so the model
    # sees the immediate exchange without paraphrase loss.
    history_compact_keep_tail: int = 4
