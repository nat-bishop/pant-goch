"""Unit tests pinning the family-specific argv shape for both llama roles.

These run in <100 ms (no subprocess); they exist to catch drift between the
two resolvers and the conditional `--reasoning` flag block in `build_command`.
If you change Gemma's reasoning flags or add a new family-specific axis,
update both tests in lockstep.
"""
from __future__ import annotations

import pytest

from handler.config import Settings
from handler.launcher import build_command, resolve_config, resolve_eval_config


def test_engine_argv_default_has_no_reasoning_flags() -> None:
    """Engine default is `llama_reasoning_mode='off'` (config.py) — neither
    `--reasoning` nor `--reasoning-format` should appear in the argv. Flip
    via env (`LLAMA_REASONING_MODE=on`) restores them; covered below.
    """
    cmd = build_command(resolve_config())
    assert "--jinja" in cmd
    assert "--reasoning" not in cmd
    assert "--reasoning-format" not in cmd


def test_engine_argv_includes_perf_flags() -> None:
    """The performance flags applied to every llama-server boot. KV cache is
    fp16 (NOT q8_0) on the engine — bench_kv_compare showed q8_0 is ~2.5×
    slower for Qwen short outputs (Gated Delta Net dequant cost); the
    Gemma engine win only mattered at playtest concurrency.
    """
    cmd = build_command(resolve_config())
    assert "--flash-attn" in cmd
    fa_idx = cmd.index("--flash-attn")
    assert cmd[fa_idx + 1] == "on"
    assert "--batch-size" in cmd and "--ubatch-size" in cmd
    # KV cache stays fp16 on the engine — no --cache-type-{k,v} flags.
    assert "--cache-type-k" not in cmd
    assert "--cache-type-v" not in cmd


def test_engine_argv_reasoning_on_restores_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping `LLAMA_REASONING_MODE=on` should restore both `--reasoning on`
    and `--reasoning-format deepseek` — no code change needed."""
    monkeypatch.setenv("LLAMA_REASONING_MODE", "on")
    cmd = build_command(resolve_config(Settings()))
    assert "--reasoning" in cmd
    r_idx = cmd.index("--reasoning")
    assert cmd[r_idx + 1] == "on"
    assert "--reasoning-format" in cmd
    f_idx = cmd.index("--reasoning-format")
    assert cmd[f_idx + 1] == "deepseek"


def test_eval_argv_enables_thinking_with_deepseek_format() -> None:
    cmd = build_command(resolve_eval_config())
    # Both families need --jinja for their bundled chat templates.
    assert "--jinja" in cmd
    # Qwen thinks (lifts judge/player quality) then writes the reply; the
    # deepseek format routes the `<think>` trace to `reasoning_content` so
    # the body stays clean.
    assert "--reasoning" in cmd
    r_idx = cmd.index("--reasoning")
    assert cmd[r_idx + 1] == "on"
    assert "--reasoning-format" in cmd
    f_idx = cmd.index("--reasoning-format")
    assert cmd[f_idx + 1] == "deepseek"


def test_eval_uses_eval_port_and_unified_kv() -> None:
    """Sanity-check the eval-specific defaults: port 8081 and --kv-unified
    (per the comment block on `Settings.eval_llama_kv_unified`)."""
    cmd = build_command(resolve_eval_config())
    assert "8081" in cmd
    assert "--kv-unified" in cmd
    assert "--no-kv-unified" not in cmd
