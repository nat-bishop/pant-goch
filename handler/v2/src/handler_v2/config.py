"""Runtime configuration loaded from env vars and an optional `.env` file.

Pared from v1's Settings (handler/src/handler/config.py): drops the eval-llama
fields (no playtester in v2), drops session_save_path (no JSONL persistence),
drops the externally-managed llama path (no `make sidecar` workflow), drops
mission_path (replaced by world-directory loading in M1).

Pydantic-settings reads env vars case-insensitively, so snake_case field names
match uppercase env names (`use_inferno` ↔ `USE_INFERNO`, etc.).

Anchored at handler/v2/.env, NOT the smallville root — v2 is a self-contained
subproject and shouldn't share env state with v1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# src layout: src/handler_v2/config.py → handler/v2/ is parents[2].
# parents[0]=src/handler_v2, [1]=src, [2]=handler/v2.
_V2_ROOT = Path(__file__).resolve().parents[2]


# Convenience exports for call sites that don't want to instantiate `Settings`.
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_V2_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ───────── service ─────────
    handler_port: int = 8000
    handler_log_level: str = "info"
    handler_logfire_console: bool = True

    # ───────── llama lifecycle ─────────
    llama_server_bin: Path | None = None
    llama_model_path: Path | None = None
    llama_hf_repo: str = "ggml-org/gemma-4-E4B-it-GGUF:Q8_0"
    llama_host: str = LLAMA_HOST
    llama_port: int = LLAMA_PORT
    # Per-call context budget. Each agent's prompt (system + user) must fit
    # within this. 32k absorbs the world bible (system prefix) + the dynamic
    # suffix (case file + reachable beats + this scene so far) comfortably;
    # SPEC §4.2 estimates ~5k tokens of dynamic suffix at end of game.
    llama_ctx: int = 32768
    llama_ngl: int = 99
    # Single-user game; one in-flight LLM call at a time. With per-slot KV
    # allocation (default), `-np 4` reserves 4× llama_ctx; bump if running
    # parallel sessions (not v1's use case).
    llama_n_parallel: int = 4
    # Gemma 4 reasoning mode. `off` (default): no `<think>` blocks; v2's
    # scene runner returns prose directly. The conversation-based log writer
    # ALSO depends on no thinking parts in assistant messages — see
    # `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` in
    # main.py's lifespan.
    llama_reasoning_mode: Literal["on", "off", "auto"] = "off"
    # Per-slot KV allocation. v1's measurement: --no-kv-unified is ~2.7×
    # faster than --kv-unified at 64-concurrent on Gemma 4 E4B Q8 (no shared
    # prefix to amortise). v2 is single-session so the win is even clearer.
    # Always emit explicit flag — recent llama.cpp builds default to unified
    # when -np is auto.
    llama_kv_unified: bool = False
    # Total KV pool size passed to llama-server's `-c` flag. With unified KV
    # off, llama-server divides this by `-np` per slot. Default 0 → fall back
    # to `n_parallel × llama_ctx` automatically.
    llama_total_ctx: int = 0
    # Escape hatch for any flag we haven't promoted to a typed Settings field.
    llama_extra_args: str = ""

    # ───────── pydantic-ai wiring (override only when pointing at remote llama) ─────────
    llama_base_url: str | None = None

    # ───────── inferno gateway (dev-mode opt-in) ─────────
    # When `use_inferno` is true, the sidecar skips spawning its bundled
    # `llama-server` subprocess and points pydantic-ai at inferno's `/v1`
    # endpoint instead. The shipped game ignores this — env var unset in
    # distribution, so the bundled subprocess boots. Dev sessions on the
    # Blackwell set `SMALLVILLE_USE_INFERNO=1` to share VRAM with FLUX /
    # TRELLIS / Qwen-Image via inferno's budgeted co-residency.
    # Namespaced `SMALLVILLE_USE_INFERNO` so it doesn't collide with other
    # inferno-aware projects on the same box.
    use_inferno: bool = Field(default=False, validation_alias="SMALLVILLE_USE_INFERNO")
    inferno_url: str = "http://127.0.0.1:7770"

    # ───────── world directory ─────────
    # Path to the world bible (locations + cast + truth + brief). None →
    # fall back to handler/v2/world/ relative to this file. Tests point at
    # fixture worlds.
    world_path: Path | None = None
