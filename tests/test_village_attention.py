"""Village-attention agent + emit_village_attention + apply_village_attention.

The village-attention classifier is stateless per scene close: it reads the
just-ended scene's transcript (plus the prior heat level) and returns one of four
levels; the engine ratchets `village_attention` up to it. Split out of Phase 1
because, bundled with the per-NPC deltas, the model left attention at its default.
Tests stub the agent's `run` directly to return structured output without a live
llama / TestModel's native-output check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from handler.agents import (
    AgentBundle,
    build_scene_runner_bundle,
    build_village_attention_agent,
    emit_village_attention,
)
from handler.engine import apply_village_attention
from handler.prompts import build_village_attention_user_prompt
from handler.schemas import Option, VillageAttentionVerdict
from handler.state import State, TurnRecord
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


@dataclass
class _FakeRunResult:
    """Stand-in for pydantic_ai's RunResult — emit_village_attention reads
    `.output.level` only."""

    output: VillageAttentionVerdict


def _bundle_with_canned_level(world, level: str) -> AgentBundle:
    """Real AgentBundle whose village-attention agent's `run` is stubbed to
    return the canned level (stateless, no message_history threaded)."""
    bundle = build_scene_runner_bundle(TestModel(), world)

    async def fake_run(user_prompt, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeRunResult(output=VillageAttentionVerdict(level=level))

    bundle.village_attention.run = fake_run  # type: ignore[assignment]
    return bundle


def _state_with_one_scene(prior: int = 0) -> State:
    """A completed scene at the pub with Tom present and the player leaning hard."""
    state = State(player_location="the_slaughtered_lamb")
    state.scene_id = 1
    state.village_attention = prior
    state.dialogue_history = [
        TurnRecord(
            prose="Tom doesn't look up from his pint.",
            applied_state_changes=[],
            player_pick=Option(text="Demand he tell you what happened to Megan."),
            player_pick_typed=True,
        ),
        TurnRecord(
            prose='"Not my business," he says, going stiff.',
            applied_state_changes=[],
            player_pick=Option(text="Threaten to print that he covered it up."),
            player_pick_typed=True,
        ),
    ]
    return state


# ───────────────────────── prompts ─────────────────────────


def test_user_prompt_carries_transcript_and_prior_level():
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene(prior=1)  # village already "noticed"
    prompt = build_village_attention_user_prompt(state, world)
    assert "## Prior Village Attention" in prompt
    assert "noticed" in prompt  # the prior level word, reverse-mapped from 1
    assert "## This Scene" in prompt
    assert "the_slaughtered_lamb" in prompt
    # The player's verbatim moves surface so the agent can judge conduct.
    assert "Threaten to print" in prompt


def test_build_agent_constructs_cleanly_and_carries_levels():
    """No schema-construction error; the system prompt carries cast + the four
    level words (so the classifier knows its output vocabulary)."""
    world = load_world(FIXTURE_VALID)
    agent = build_village_attention_agent(TestModel(), world)
    sys_text = "\n".join(agent._system_prompts)
    assert "Old Tom Bevan" in sys_text  # voice-only cast present
    for word in ("none", "noticed", "talked_about", "hostile"):
        assert word in sys_text


# ───────────────────────── emit ─────────────────────────


@pytest.mark.asyncio
async def test_emit_returns_level_word():
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    bundle = _bundle_with_canned_level(world, "talked_about")
    level = await emit_village_attention(bundle, world, state)
    assert level == "talked_about"


# ───────────────────────── ratchet ─────────────────────────


def test_apply_ratchets_up_only():
    state = State(player_location="the_slaughtered_lamb")
    assert state.village_attention == 0
    assert apply_village_attention(state, "talked_about") == 2
    assert apply_village_attention(state, "none") == 2     # never lowers
    assert apply_village_attention(state, "noticed") == 2  # nor a lower verdict
    assert apply_village_attention(state, "hostile") == 3  # but climbs further


def test_level_vocabulary_is_consistent_across_modules():
    """The level words live in three places — the VillageAttentionVerdict Literal,
    engine.ATTENTION_LEVEL_VALUES (int map), and prompts._ATTENTION_WORDS (derived).
    A typo/drift in any one would KeyError at apply time or mismatch the prompt;
    pin them together here so the agent's output vocabulary stays in lockstep."""
    from typing import get_args

    from handler.engine import ATTENTION_LEVEL_VALUES
    from handler.prompts import _ATTENTION_WORDS

    literal_levels = set(get_args(VillageAttentionVerdict.model_fields["level"].annotation))
    assert set(ATTENTION_LEVEL_VALUES) == literal_levels
    # _ATTENTION_WORDS is the value-ordered reverse map — index i ↔ level value i.
    assert _ATTENTION_WORDS == tuple(
        w for w, _ in sorted(ATTENTION_LEVEL_VALUES.items(), key=lambda kv: kv[1])
    )
    assert {ATTENTION_LEVEL_VALUES[w] for w in _ATTENTION_WORDS} == set(range(len(_ATTENTION_WORDS)))
