"""Tests for history-compaction processors.

`hard_tail_cap` is a pure list operation. `summarize_when_large` calls the
SUMMARIZER_AGENT global, which we patch with a TestModel-backed stand-in so the
test runs without an LLM provider.
"""
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

import npc_service.main as main
from npc_service.main import hard_tail_cap, summarize_when_large


def _user_msg(content: str = "hello") -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _model_msg(content: str = "reply") -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


# ───────────────────────── hard_tail_cap ─────────────────────────


def test_hard_tail_cap_under_limit_passthrough():
    msgs = [_user_msg() for _ in range(5)]
    assert hard_tail_cap(msgs) == msgs


def test_hard_tail_cap_at_limit_passthrough(monkeypatch):
    monkeypatch.setattr(main, "MAX_HISTORY_MSGS", 4)
    msgs = [_user_msg(f"m{i}") for i in range(4)]
    assert hard_tail_cap(msgs) == msgs


def test_hard_tail_cap_drops_oldest(monkeypatch):
    monkeypatch.setattr(main, "MAX_HISTORY_MSGS", 4)
    msgs = [_user_msg(f"m{i}") for i in range(9)]
    out = hard_tail_cap(msgs)
    assert len(out) == 4
    assert out == msgs[-4:]


# ───────────────────────── summarize_when_large ─────────────────────────


async def test_summarize_no_summarizer_agent_passthrough(monkeypatch):
    monkeypatch.setattr(main, "SUMMARIZER_AGENT", None)
    msgs = [_user_msg("x" * 1000) for _ in range(20)]
    assert await summarize_when_large(msgs) == msgs


async def test_summarize_below_token_threshold_passthrough(monkeypatch):
    monkeypatch.setattr(main, "SUMMARIZER_AGENT", Agent(TestModel(), instructions="x"))
    monkeypatch.setattr(main, "SUMMARY_TRIGGER_TOKENS", 100_000)
    msgs = [_user_msg("short") for _ in range(3)]
    assert await summarize_when_large(msgs) == msgs


async def test_summarize_below_keep_tail_passthrough(monkeypatch):
    monkeypatch.setattr(main, "SUMMARIZER_AGENT", Agent(TestModel(), instructions="x"))
    monkeypatch.setattr(main, "SUMMARY_TRIGGER_TOKENS", 1)
    monkeypatch.setattr(main, "SUMMARY_KEEP_TAIL", 10)
    msgs = [_user_msg("hi") for _ in range(3)]
    assert await summarize_when_large(msgs) == msgs


async def test_summarize_folds_head_into_recap(monkeypatch):
    summarizer = Agent(TestModel(), instructions="summarize")
    monkeypatch.setattr(main, "SUMMARIZER_AGENT", summarizer)
    monkeypatch.setattr(main, "SUMMARY_TRIGGER_TOKENS", 100)
    monkeypatch.setattr(main, "SUMMARY_KEEP_TAIL", 4)

    # 10 × 200-char messages → ~500 approx tokens, well above the 100 threshold
    msgs = [_user_msg("x" * 200) for _ in range(10)]

    out = await summarize_when_large(msgs)

    # head (6) collapsed → 1 recap; tail (4) kept verbatim
    assert len(out) == 5
    first = out[0]
    assert isinstance(first, ModelRequest)
    assert any(
        isinstance(p, UserPromptPart) and "[recap of earlier negotiation]" in p.content
        for p in first.parts
    )
    assert out[1:] == msgs[-4:]


async def test_summarize_skips_when_head_too_small(monkeypatch):
    """Head must be ≥ 6 messages to be worth summarizing."""
    summarizer = Agent(TestModel(), instructions="summarize")
    monkeypatch.setattr(main, "SUMMARIZER_AGENT", summarizer)
    monkeypatch.setattr(main, "SUMMARY_TRIGGER_TOKENS", 1)
    monkeypatch.setattr(main, "SUMMARY_KEEP_TAIL", 4)

    # 8 messages → tail=4, head=4, head < 6 → skip
    msgs = [_user_msg("x" * 200) for _ in range(8)]
    out = await summarize_when_large(msgs)
    assert out == msgs
