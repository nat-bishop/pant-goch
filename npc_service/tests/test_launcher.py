"""Unit tests pinning the family-specific argv shape for both llama roles.

These run in <100 ms (no subprocess); they exist to catch drift between the
two resolvers and the conditional `--reasoning` flag block in `build_command`.
If you change Gemma's reasoning flags or add a new family-specific axis,
update both tests in lockstep.
"""
from __future__ import annotations

from npc_service.launcher import build_command, resolve_config, resolve_eval_config


def test_engine_argv_includes_reasoning_flags() -> None:
    cmd = build_command(resolve_config())
    assert "--jinja" in cmd
    # Gemma 4 best practice: reasoning routed into a separate field, not body.
    assert "--reasoning" in cmd
    assert "on" in cmd
    assert "--reasoning-format" in cmd
    assert "deepseek" in cmd


def test_eval_argv_enables_thinking_with_deepseek_format() -> None:
    cmd = build_command(resolve_eval_config())
    # Both families need --jinja for their bundled chat templates.
    assert "--jinja" in cmd
    # Qwen thinks (lifts judge/player quality) then writes the reply; the
    # deepseek format routes the `<think>` trace to `reasoning_content` so
    # the body stays clean.
    assert "--reasoning" in cmd
    assert "on" in cmd
    assert "--reasoning-format" in cmd
    assert "deepseek" in cmd


def test_eval_uses_eval_port_and_unified_kv() -> None:
    """Sanity-check the eval-specific defaults: port 8081 and --kv-unified
    (per the comment block on `Settings.eval_llama_kv_unified`)."""
    cmd = build_command(resolve_eval_config())
    assert "8081" in cmd
    assert "--kv-unified" in cmd
    assert "--no-kv-unified" not in cmd
