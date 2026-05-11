"""Runtime configuration loaded from env vars and an optional `.env` file.

Pydantic-settings reads env vars case-insensitively, so snake_case field names
here match uppercase env names (`launch_llama` ↔ `LAUNCH_LLAMA`). The .env file
is anchored at the repo root rather than CWD because the sidecar is launched
with mixed CWDs (Godot launches it with cwd=repo-root; `make sidecar` cd's into
handler/).

Most of these fields are tuned once and forgotten — context size, slot count,
ports, history-compaction tunings. Only a handful are actually toggled at
runtime: `launch_llama` (the `make sidecar` workflow + tests), `mission_path`
(test fixtures, playtester), `handler_logfire_console` (conftest sets false), and
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
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# src layout: src/handler/config.py → workspace root is parents[3].
# parents[0]=src/handler, [1]=src, [2]=handler, [3]=<workspace>.
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
    handler_port: int = 8000
    handler_log_level: str = "info"
    # True (default) prints Logfire spans + log records to stderr for fast local
    # feedback; tests set this to false to keep pytest output uncluttered.
    handler_logfire_console: bool = True

    # ───────── llama lifecycle ─────────
    launch_llama: bool = True
    llama_server_bin: Path | None = None
    llama_model_path: Path | None = None
    llama_hf_repo: str = "ggml-org/gemma-4-E4B-it-GGUF:Q8_0"
    llama_host: str = LLAMA_HOST
    llama_port: int = LLAMA_PORT
    # Per-session conversation budget. Drives the chat agent's history-compaction
    # threshold (`history_compact_frac × llama_ctx`). The summary agent pre-empts
    # the model's hard ctx limit by recapping oldest messages once the running
    # history would exceed this. Default sized for one in-flight session (single
    # player); the playtester's per-session budget is the same — what scales for
    # high concurrency is `llama_total_ctx` (the server's `-c` flag), NOT this.
    llama_ctx: int = 32768
    llama_ngl: int = 99
    # Number of parallel decoding slots in llama-server (`-np N`). Each slot
    # can serve one in-flight request; past N, requests queue server-side.
    # With `--no-kv-unified` (our default), each slot pre-allocates
    # `llama_total_ctx / n_parallel` tokens of KV — so `-np` directly drives
    # VRAM cost. Default 4 is sized for single-user `make run` (one /send
    # peaks at 2–3 concurrent agent calls: extractor in parallel with chat,
    # plus fact-check on retries). Past benchmarks showed 128 slots
    # fragmented the KV pool and degraded effective batch for the active
    # slot — interactive throughput improved ~2× moving from 128 to 4 on
    # a 96 GB Blackwell. The playtester overrides via `LLAMA_N_PARALLEL=64`
    # to match its `--parallel 64` driver (Makefile:llama-playtest).
    llama_n_parallel: int = 4
    # Gemma 4 reasoning mode. `off` (default): no `<think>` blocks; the chat
    # and director agents return prose directly. `on`: restore Gemma 4
    # thinking on chat + director (classifier/extractor/fact-checker/summary
    # keep their per-agent `_NO_THINK_KWARGS` opt-out so they stay no-think
    # regardless). `auto`: llama-server detects from the chat template.
    # Wired through to `--reasoning <mode>` and (only when mode != "off")
    # `--reasoning-format deepseek` on the llama-server command. Set
    # `LLAMA_REASONING_MODE=on` to flip back without code changes.
    llama_reasoning_mode: Literal["on", "off", "auto"] = "off"
    # Unified KV cache (`--kv-unified`): one shared KV buffer of total size
    # `llama_total_ctx` paged across all slots, with prefix bytes shared across
    # sequences. Optimised for many-users-one-system-prompt fleets, where the
    # shared prefix amortises prefill. Penalises workloads where each session
    # has a distinct prefix (agentic/stateful), because the page-management
    # overhead applies on every batch step but there's nothing to share —
    # measured ~2.7× slower at 64-concurrent on Gemma 4 E4B Q8 vs per-slot
    # allocation. Smallville is the latter (each session has its own persona,
    # history, canon ledger), so default OFF. Recent llama.cpp builds enable
    # unified by default when `-np` is auto, so we always emit an EXPLICIT
    # flag (--kv-unified or --no-kv-unified) to defend against drift.
    llama_kv_unified: bool = False
    # Total KV pool size passed to llama-server's `-c` flag. With unified KV
    # off (the default), llama-server divides this by `-np` per slot, so each
    # slot gets `llama_total_ctx / n_parallel` tokens of context. Sizing rule:
    # `llama_total_ctx >= n_parallel × llama_ctx` so each slot has the full
    # per-session budget. Default 0 → fall back to `n_parallel × llama_ctx`
    # automatically. The playtester sets `LLAMA_TOTAL_CTX=2097152` (2M = 64 ×
    # 32k) explicitly so the value stays self-documenting at the call site.
    llama_total_ctx: int = 0
    # Escape hatch for any flag we haven't promoted to a typed Settings field.
    # Whitespace-split, appended verbatim to the llama-server command.
    llama_extra_args: str = ""

    # ───────── pydantic-ai wiring (override only when pointing at remote llama) ─────────
    llama_base_url: str | None = None

    # ───────── eval llama (playtester second server, Qwen on mainline llama.cpp) ─────────
    # Powers the synthetic-player agent and the LLM judges. Different family from
    # the engine SUT (Gemma) to mitigate same-family preference leakage in judging
    # — see arxiv 2502.01534. Default points at port 8081 so the engine llama on
    # 8080 stays unaffected. Lifecycle is user-managed (`make llama-eval`); the
    # playtester just probes /health on this URL before starting.
    eval_llama_port: int = EVAL_LLAMA_PORT
    eval_llama_base_url: str | None = None
    # Boot-time fields (mirror engine field-for-field; routed through
    # `launcher.resolve_eval_config` → `build_command`). Defaults sized for the
    # playtester at `--parallel 64`.
    eval_llama_server_bin: Path | None = None
    eval_llama_model_path: Path | None = None
    eval_llama_hf_repo: str = "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"
    eval_llama_host: str = LLAMA_HOST
    # Per-session conversation budget. Same default as the engine so the
    # synthetic player + judges have the same headroom as the production chat
    # agent. Total pool sizing is driven by `eval_llama_total_ctx`.
    eval_llama_ctx: int = 32768
    eval_llama_ngl: int = 99
    # Slots = 16, smaller than the playtester's `--parallel 64` default. Why:
    # `eval_llama_n_parallel × eval_llama_ctx` is the KV pool reservation
    # (with `--kv-unified`, llama.cpp pre-allocates the full pool at boot),
    # and at 64 × 32k = 2M tokens the Qwen 35B-A3B GQA KV pool is ~40 GB —
    # which combined with weights (~37 GB) + compute (~10 GB) + the engine
    # llama (~15 GB) overshoots a 96 GB Blackwell, pushing VRAM to 99% used
    # and triggering driver paging that collapsed observed throughput from
    # 185 tok/s solo to ~6 tok/s under playtest load (May 2026 measurement).
    # 16 slots × 32k = 512k pool → ~10 GB KV → ~78 GB total, ~18 GB margin.
    # Cap the playtester at `--parallel 16` to match. Raise these together
    # only if you have headroom (multi-GPU, smaller quant, or ik_llama.cpp).
    eval_llama_n_parallel: int = 16
    # Default `True`. The engine's measurement of `--no-kv-unified` being
    # ~2.7× faster (see `llama_kv_unified` above) was on dense Gemma 4 E4B;
    # Qwen 35B-A3B is MoE, where unified pages help expert-routing cache
    # locality. The original raw `make llama-eval` command (deleted) used
    # `--kv-unified`. Re-measure on Qwen before flipping. With unified, `-c`
    # is the total pool dynamically shared across slots; the auto-sized
    # `n_parallel × ctx = 2M` pool gives every slot up to 32k headroom.
    eval_llama_kv_unified: bool = True
    eval_llama_total_ctx: int = 0
    eval_llama_extra_args: str = ""

    # Path to the mission YAML to load at lifespan boot. None → fall back to
    # handler/missions/lynx_exfil.yaml relative to this file. Tests + the
    # playtester set this to point at fixtures.
    mission_path: Path | None = None

    # ───────── session persistence (off by default) ─────────
    # When set, every committed Event on the default session is appended as
    # one JSON object per line to <session_save_path>. On boot, lifespan
    # replays the JSONL into a fresh session before yielding. Unset → no
    # disk I/O (in-memory only).
    session_save_path: Path | None = None
