"""M6 — log writer agent + emit_log_entry.

The log writer is conversation-based: one user+assistant pair per scene
end, accumulating in `runtime.message_history`. Tests stub the agent with
TestModel so message history mechanics + LogEntry assembly can be
verified without a live llama.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from handler.agents import (
    LogWriterRuntime,
    build_log_writer_agent,
    emit_log_entry,
    make_log_writer_runtime,
)
from handler.prompts import build_log_writer_user_prompt
from handler.schemas import ContinueScene, EndScene, Option, UnlockBeat
from handler.state import State, TurnRecord
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def _state_with_one_scene() -> State:
    """A minimal state with a single completed scene's worth of
    dialogue_history. The scene was at the pub with Tom present, three
    turns, with tom_named_a_vehicle fired in the middle turn."""
    state = State(player_location="the_slaughtered_lamb")
    state.scene_id = 1
    state.npc_scenes_co_present = {"tom": 1, "maud": 0}
    state.unlocked_beats = {"tom_named_a_vehicle"}
    state.dialogue_history = [
        TurnRecord(
            scene_id=1,
            location_id="the_slaughtered_lamb",
            npcs_present=["tom"],
            prose="The peat fire was going. Tom glanced up.",
            applied_state_changes=[],
            next_kind="continue",
            options=[Option(text="Sit a few stools down"), Option(text="Take the stool next to him")],
            player_pick=Option(text="Take the stool next to him"),
        ),
        TurnRecord(
            scene_id=1,
            location_id="the_slaughtered_lamb",
            npcs_present=["tom"],
            prose="Tom set the pint down very slowly. 'Saw a Land Rover. Eluned Vaughn's.'",
            applied_state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")],
            next_kind="continue",
            options=[Option(text="'Why didn't you tell them?'"), Option(text="Stay quiet")],
            player_pick=Option(text="Stay quiet"),
        ),
        TurnRecord(
            scene_id=1,
            location_id="the_slaughtered_lamb",
            npcs_present=["tom"],
            prose="He finished the pint. 'I'd best be off.'",
            applied_state_changes=[],
            next_kind="end_scene",
            options=[],
            player_pick=None,
        ),
    ]
    return state


# ───────────────────────── prompt builders ─────────────────────────


def test_log_writer_user_prompt_includes_scene_anchors():
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    prompt = build_log_writer_user_prompt(state, world)
    # Location anchor surfaces
    assert "the_slaughtered_lamb" in prompt
    # NPC anchor surfaces (rendered by name, not id)
    assert "Old Tom Bevan" in prompt
    # Beats unlocked surfaces
    assert "tom_named_a_vehicle" in prompt
    # Turn-by-turn appears
    assert "Turn 1" in prompt and "Turn 2" in prompt and "Turn 3" in prompt
    # Player picks surface
    assert "Take the stool next to him" in prompt
    # Closing instruction is there
    assert "First person" in prompt or "first person" in prompt


def test_log_writer_user_prompt_includes_leave_pick_marker():
    """When a turn's pick has leaves_for, the prompt makes the exit visible
    so the log writer can mention "I walked back to..." naturally."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    state.dialogue_history[-1].player_pick = Option(
        text="Walk to the post office", leaves_for="the_post_office"
    )
    prompt = build_log_writer_user_prompt(state, world)
    assert "the_post_office" in prompt
    assert "left for" in prompt or "leaves_for" in prompt


# ───────────────────────── runtime construction ─────────────────────────


def test_make_log_writer_runtime_starts_with_empty_history():
    world = load_world(FIXTURE_VALID)
    runtime = make_log_writer_runtime(TestModel(), world)
    assert isinstance(runtime, LogWriterRuntime)
    assert runtime.message_history == []


def test_build_log_writer_agent_constructs_cleanly():
    """No schema-construction errors; the agent is freeform str output."""
    world = load_world(FIXTURE_VALID)
    agent = build_log_writer_agent(TestModel(), world)
    # System prompt should carry the bible — cast names surface.
    sys_text = "\n".join(agent._system_prompts)
    assert "Old Tom Bevan" in sys_text
    assert "Maud Pryce" in sys_text
    # Voice rules surface so the agent writes in first person.
    assert "First person" in sys_text or "first person" in sys_text


# ───────────────────────── emit_log_entry ─────────────────────────


@pytest.mark.asyncio
async def test_emit_log_entry_returns_log_entry_with_correct_anchors():
    """The LogEntry's scene_id / location / npcs_present come from the
    just-completed scene's first dialogue_history turn."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    custom_text = (
        "I went into the Slaughtered Lamb that afternoon. Tom was at the "
        "second stool, like Mrs Hughes had said. He named the vehicle, "
        "eventually, and looked relieved when he did. I did not buy him a "
        "drink."
    )
    runtime = make_log_writer_runtime(
        TestModel(custom_output_text=custom_text), world
    )
    entry = await emit_log_entry(runtime, world, state)

    assert entry.scene_id == 1
    assert entry.location_id == "the_slaughtered_lamb"
    assert entry.npcs_present == ["tom"]
    assert entry.text == custom_text.strip()


@pytest.mark.asyncio
async def test_emit_log_entry_grows_message_history():
    """After the first call, message_history holds the user+assistant
    pair (plus the system message). After the second call, it holds
    two pairs — the conversation IS the case file."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    runtime = make_log_writer_runtime(
        TestModel(custom_output_text="An entry."), world
    )

    assert len(runtime.message_history) == 0
    await emit_log_entry(runtime, world, state)
    after_one = len(runtime.message_history)
    assert after_one > 0

    # Second scene-end — history grows but the existing entries stay.
    await emit_log_entry(runtime, world, state)
    after_two = len(runtime.message_history)
    assert after_two > after_one


