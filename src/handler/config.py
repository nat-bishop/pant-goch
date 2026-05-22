"""Runtime configuration loaded from env vars and an optional `.env` file.

Pydantic-settings reads env vars case-insensitively, so snake_case field names
match uppercase env names (`use_inferno` ↔ `USE_INFERNO`, etc.).

Anchored at the project root `.env`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# src layout: src/handler/config.py → project root is parents[2].
# parents[0]=src/handler, [1]=src, [2]=project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Convenience exports for call sites that don't want to instantiate `Settings`.
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8080
LLAMA_BASE_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1"


# Per-request body forwarded by the OpenAI SDK to llama-server/inferno. Gemma 4's
# Jinja template gates its reasoning preamble on `enable_thinking`; setting it
# False is the only thinking-suppression lever that survives the inferno gateway
# (where `--reasoning off` is bypassed). Lives here so both `main.py`'s global
# model settings and `agents.py`'s per-agent decision settings reference ONE
# payload — a per-agent ModelSettings.extra_body REPLACES the model's (pydantic-ai
# merges settings shallowly), so the decision agents must re-include this exact
# dict or they silently lose suppression. See main.py build_model.
THINKING_OFF_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
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
    # Gemma 4 reasoning mode, passed verbatim as `--reasoning <mode>` to
    # llama-server. `off` (default): server does not extract thinking, and
    # Gemma 4's Jinja template emits content tokens directly. DO NOT
    # interpret `off` as "skip the flag" — llama-server's CLI default is
    # `auto`, which on Gemma 4 with `--jinja` activates the reasoning
    # channel and wastes ~125 output tokens per call on hidden planning.
    # The conversation-based chronicler ALSO depends on no thinking parts
    # in assistant messages — see
    # `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` in
    # main.py's lifespan.
    llama_reasoning_mode: Literal["on", "off", "auto"] = "off"
    # Output ceiling for the two empty-list "decision" agents (populator,
    # phase1). A clean direct answer is <200 tokens; runaways floor at ~17.8k.
    # Gemma 4 occasionally opens its (ungrammared) reasoning channel on these
    # "decide whether to act" agents and loops to the provider ceiling without
    # ever emitting the answer JSON. 500 forces a direct answer: long reasoning
    # / runaways are aborted (length finish → UnexpectedModelBehavior) and
    # re-rolled by the agents.py retry layer. Deliberately tighter than the
    # ~2.3k max reasoned-then-succeeded output, since a reasoned answer is a
    # degraded answer for these agents. Does NOT apply to the content agents
    # (prose/chronicle legitimately need long output).
    decision_agent_max_tokens: int = 500
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

    # ───────── playtest harness (eval tooling) ─────────
    # Gemini model the simulated player uses (evals/). Engine model is
    # unaffected — the player is a separate cloud agent.
    playtest_player_model: str = "gemini-3.1-flash-lite"
    # Max concurrent sessions the harness runs against the engine. Should match
    # the engine's KV-slot count (inferno serves Gemma at np=16) so demand fills
    # the slots without oversubscribing. DECOUPLED from runs_per_persona on
    # purpose: total sessions (personas × runs) may exceed this, and
    # pydantic-evals queues the surplus, backfilling a fresh session each time one
    # finishes — so the engine stays saturated instead of decaying as short
    # sessions drop out. Override with --parallel or SMALLVILLE_PLAYTEST_PARALLEL.
    playtest_parallel: int = Field(
        default=16, validation_alias="SMALLVILLE_PLAYTEST_PARALLEL"
    )
    # API key for the Gemini player. Sourced from the project .env (which
    # `uv run` does NOT export to os.environ), so the harness reads it here
    # and passes it explicitly to GoogleProvider rather than relying on the
    # provider's env lookup. Accepts either GEMINI_API_KEY or GOOGLE_API_KEY.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    # ───────── eval LLM judges ─────────
    # The judges (evals/judges/) score recorded artifacts. Default to local Qwen
    # served by inferno on its OpenAI-compatible endpoint — no API key needed,
    # independent of the Gemma-under-test, and inferno's continuous batching
    # absorbs the whole batch fired at once. Override with a `gemini:`/`claude:`
    # prefix to A/B against a cloud judge. `judge_base_url` defaults to inferno's
    # /v1; only consulted for the local (OpenAI-compatible) path.
    judge_model: str = "qwen-3.6-35b-a3b"
    judge_base_url: str | None = None  # None → f"{inferno_url}/v1"

    @property
    def resolved_judge_base_url(self) -> str:
        return self.judge_base_url or f"{self.inferno_url.rstrip('/')}/v1"

    # ───────── world directory ─────────
    # Path to the world bible (locations + cast + truth + brief). None →
    # fall back to world/ relative to the project root. Tests point at
    # fixture worlds.
    world_path: Path | None = None
