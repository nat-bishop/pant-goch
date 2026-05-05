"""Unit tests for the history-compaction history_processor in agents.py.

We test the compactor in isolation rather than through the FastAPI surface:
- Token counting is stubbed (production hits llama-server `/tokenize`).
- The summary agent is overridden with a Pydantic AI TestModel so no LLM
  call goes out.
- Inputs are synthetic ModelMessage lists; we assert shape, not content.

Three behaviours covered:
1. Under-budget → messages returned unchanged.
2. Over-budget but len <= keep_tail → returned unchanged (nothing safe to drop).
3. Over-budget and long enough → summary replaces oldest, last keep_tail kept verbatim.
"""
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from npc_service.agents import make_history_compactor


SUMMARY_TEXT = "Aria and the player exchanged calibration notes about scheduling."


def _build_summary_agent() -> Agent:
    """A summary agent backed by TestModel so no real LLM call is made."""
    return Agent(
        TestModel(custom_output_text=SUMMARY_TEXT),
        output_type=str,
        instructions="(test summary instructions)",
    )


def _player_msg(text: str) -> ModelMessage:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _aria_msg(text: str) -> ModelMessage:
    return ModelResponse(parts=[TextPart(content=text)])


def _make_history(n_pairs: int) -> list[ModelMessage]:
    """One Aria welcome + n_pairs of (player, Aria) — matches the runtime
    shape of `STATE.thread_messages["aria"]` after n_pairs sends."""
    history: list[ModelMessage] = [_aria_msg("Welcome.")]
    for i in range(n_pairs):
        history.append(_player_msg(f"player-turn-{i}"))
        history.append(_aria_msg(f"aria-turn-{i}"))
    return history


@pytest.mark.asyncio
async def test_under_budget_returns_unchanged():
    async def counter(_messages):
        return 100  # well under budget

    compact = make_history_compactor(
        summary_agent=_build_summary_agent(),
        token_counter=counter,
        token_budget=1000,
        keep_tail=4,
    )

    messages = _make_history(n_pairs=5)
    result = await compact(messages)

    # Same list contents (identity not asserted — implementation may copy).
    assert result == messages


@pytest.mark.asyncio
async def test_short_history_returns_unchanged_even_when_over_budget():
    """If the history has only `keep_tail` messages or fewer, the compactor
    has nothing safe to summarise — leaves messages alone, even at over-budget
    token counts. Prevents pathological 'summary replaces everything' cases."""
    async def counter(_messages):
        return 99_999  # comically over budget

    compact = make_history_compactor(
        summary_agent=_build_summary_agent(),
        token_counter=counter,
        token_budget=1000,
        keep_tail=4,
    )

    messages = _make_history(n_pairs=1)  # welcome + 1 pair = 3 messages
    assert len(messages) == 3
    assert len(messages) <= 4  # ≤ keep_tail

    result = await compact(messages)
    assert result == messages


@pytest.mark.asyncio
async def test_over_budget_replaces_oldest_with_summary():
    """The interesting case. With budget=1000 and counter returning 5000,
    the compactor must call the summary agent on `messages[:-keep_tail]`
    and stitch the summary in front of the kept tail."""
    async def counter(_messages):
        return 5000

    compact = make_history_compactor(
        summary_agent=_build_summary_agent(),
        token_counter=counter,
        token_budget=1000,
        keep_tail=4,
    )

    # Welcome + 5 pairs = 11 messages; oldest 7 will be summarised, last 4 kept.
    messages = _make_history(n_pairs=5)
    assert len(messages) == 11

    result = await compact(messages)

    # Last 4 messages preserved verbatim.
    assert result[-4:] == messages[-4:]

    # Everything else is the summary's new_messages() output. With TestModel
    # output_type=str, that's a ModelRequest (the synthesised user prompt that
    # primed the summary run, since message_history was used) plus a
    # ModelResponse(TextPart=SUMMARY_TEXT). We don't pin the exact count — we
    # just assert the summary text shows up somewhere in the head.
    head = result[:-4]
    assert any(
        any(isinstance(p, TextPart) and SUMMARY_TEXT in p.content for p in m.parts)
        for m in head
    ), f"summary text not found in head: {head!r}"

    # And the result is shorter than the original (otherwise compaction was a
    # no-op — would defeat the purpose).
    assert len(result) < len(messages)
