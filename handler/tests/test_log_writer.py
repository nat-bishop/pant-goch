"""M6 — log writer agent + emit_log_entry.

The log writer is conversation-based: one user+assistant pair per scene
end, accumulating in `runtime.message_history`. Tests stub the agent's
`run` method directly so the structured output (LogWriterOutput) can be
returned without a live llama and without TestModel's native-output
profile check kicking in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

from handler.agents import (
    LogWriterRuntime,
    build_log_writer_agent,
    emit_log_entry,
    make_log_writer_runtime,
)
from handler.prompts import build_log_writer_user_prompt
from handler.schemas import (
    ContinueScene,
    EndScene,
    LogWriterOutput,
    Option,
    UnlockBeat,
)
from handler.state import State, TurnRecord
from handler.world import load_world


# ───────────────────────── fake agent helpers ─────────────────────────


@dataclass
class _FakeRunResult:
    """Minimal stand-in for pydantic_ai's RunResult. `emit_log_entry`
    accesses `.output` and `.all_messages()` only."""

    output: LogWriterOutput
    messages: list[Any] = field(default_factory=list)

    def all_messages(self) -> list[Any]:
        return list(self.messages)


def _runtime_with_canned_outputs(
    world, outputs: list[LogWriterOutput]
) -> LogWriterRuntime:
    """Build a real LogWriterRuntime and monkeypatch its agent.run with
    a sequence of canned `LogWriterOutput` responses. Each call appends
    a single placeholder ModelMessage to `messages` so the runtime's
    `message_history` grows by one per call (the production code
    receives all_messages() and overwrites with the result)."""
    runtime = make_log_writer_runtime(TestModel(), world)
    iter_outputs = iter(outputs)
    history: list[Any] = []

    async def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        output = next(iter_outputs)
        # Grow the simulated history by a marker message per call so the
        # length-grows test can observe accumulation.
        history.append(("call", output))
        return _FakeRunResult(output=output, messages=list(history))

    runtime.agent.run = fake_run  # type: ignore[assignment]
    return runtime


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
    just-completed scene's first dialogue_history turn; case_file_entry
    and npc_memory_updates come from the log writer's structured output."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    custom_text = (
        "I went into the Slaughtered Lamb that afternoon. Tom was at the "
        "second stool, like Mrs Hughes had said. He named the vehicle, "
        "eventually, and looked relieved when he did. I did not buy him a "
        "drink."
    )
    runtime = _runtime_with_canned_outputs(world, [LogWriterOutput(
        case_file_entry=custom_text,
        npc_memory_updates={
            "tom": "Tom unburdened. Named Eluned's Land Rover. Walked himself home.",
        },
    )])
    entry = await emit_log_entry(runtime, world, state)

    assert entry.scene_id == 1
    assert entry.location_id == "the_slaughtered_lamb"
    assert entry.npcs_present == ["tom"]
    assert entry.text == custom_text.strip()
    assert "tom" in entry.npc_memory_updates
    assert "Land Rover" in entry.npc_memory_updates["tom"]


@pytest.mark.asyncio
async def test_emit_log_entry_filters_memory_updates_to_present_npcs():
    """The log writer must not emit memory for NPCs who weren't in the
    scene. Anchor's `npcs_present` is the source of truth; emit_log_entry
    drops keys outside it and keys not in the world."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()  # tom present, maud not
    runtime = _runtime_with_canned_outputs(world, [LogWriterOutput(
        case_file_entry="The scene.",
        npc_memory_updates={
            "tom": "Tom note.",
            "maud": "Maud was NOT in this scene; should be filtered.",
            "ghost_npc": "Not a real NPC; should be filtered.",
        },
    )])
    entry = await emit_log_entry(runtime, world, state)
    assert set(entry.npc_memory_updates.keys()) == {"tom"}


@pytest.mark.asyncio
async def test_emit_log_entry_drops_empty_string_memory_updates():
    """If the log writer emits an empty/whitespace note for a present
    NPC, drop it rather than carrying noise forward into npc_memory."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    runtime = _runtime_with_canned_outputs(world, [LogWriterOutput(
        case_file_entry="Scene.",
        npc_memory_updates={"tom": "   "},
    )])
    entry = await emit_log_entry(runtime, world, state)
    assert entry.npc_memory_updates == {}


@pytest.mark.asyncio
async def test_emit_log_entry_grows_message_history():
    """After the first call, message_history holds the runtime-captured
    messages. After the second call, it holds more — the conversation IS
    the case file."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    runtime = _runtime_with_canned_outputs(
        world,
        [
            LogWriterOutput(case_file_entry="An entry.", npc_memory_updates={}),
            LogWriterOutput(case_file_entry="Another entry.", npc_memory_updates={}),
        ],
    )

    assert len(runtime.message_history) == 0
    await emit_log_entry(runtime, world, state)
    after_one = len(runtime.message_history)
    assert after_one > 0

    # Second scene-end — history grows but the existing entries stay.
    await emit_log_entry(runtime, world, state)
    after_two = len(runtime.message_history)
    assert after_two > after_one


