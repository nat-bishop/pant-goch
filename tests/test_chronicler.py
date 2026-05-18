"""M6 — chronicler agent + emit_chronicle_entry.

The chronicler is conversation-based: one user+assistant pair per scene
end, accumulating in `runtime.message_history`. Tests stub the agent's
`run` method directly so the structured output (ChronicleEntry) can be
returned without a live llama and without TestModel's native-output
profile check kicking in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

from handler.agents import (
    AgentBundle,
    ChroniclerRuntime,
    _build_chronicle_entry_type,
    build_chronicler_agent,
    build_scene_runner_bundle,
    emit_chronicle_entry,
    make_chronicler_runtime,
)
from handler.prompts import build_chronicler_user_prompt
from handler.schemas import (
    ChronicleEntry,
    NpcMemoryUpdate,
    Option,
    UnlockBeat,
)
from handler.state import State, TurnRecord
from handler.world import load_world


# ───────────────────────── fake agent helpers ─────────────────────────


@dataclass
class _FakeRunResult:
    """Minimal stand-in for pydantic_ai's RunResult. `emit_chronicle_entry`
    accesses `.output` and `.all_messages()` only."""

    output: ChronicleEntry
    messages: list[Any] = field(default_factory=list)

    def all_messages(self) -> list[Any]:
        return list(self.messages)


def _bundle_and_runtime_with_canned_outputs(
    world, outputs: list[ChronicleEntry]
) -> tuple[AgentBundle, ChroniclerRuntime]:
    """Build a real AgentBundle (whose chronicler agent we monkeypatch)
    and a fresh ChroniclerRuntime. Each `bundle.chronicler.run` call
    returns the next canned `ChronicleEntry` and appends a single
    placeholder ModelMessage to `messages` so the runtime's
    `message_history` grows by one per call (the production code
    receives all_messages() and overwrites with the result)."""
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_chronicler_runtime()
    iter_outputs = iter(outputs)
    history: list[Any] = []

    async def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        output = next(iter_outputs)
        history.append(("call", output))
        return _FakeRunResult(output=output, messages=list(history))

    bundle.chronicler.run = fake_run  # type: ignore[assignment]
    return bundle, runtime


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def _state_with_one_scene() -> State:
    """A minimal state with a single completed scene's worth of
    dialogue_history. The scene was at the pub with Tom present, three
    turns, with tom_named_a_vehicle fired in the middle turn."""
    state = State(player_location="the_slaughtered_lamb")
    state.scene_id = 1
    state.unlocked_beats = {"tom_named_a_vehicle"}
    state.dialogue_history = [
        TurnRecord(
            prose="The peat fire was going. Tom glanced up.",
            applied_state_changes=[],
            player_pick=Option(text="Take the stool next to him"),
        ),
        TurnRecord(
            prose="Tom set the pint down very slowly. 'Saw a Land Rover. Eluned Vaughn's.'",
            applied_state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")],
            player_pick=Option(text="Stay quiet"),
        ),
        TurnRecord(
            prose="He finished the pint. 'I'd best be off.'",
            applied_state_changes=[],
            player_pick=None,
        ),
    ]
    return state


# ───────────────────────── prompt builders ─────────────────────────


def test_chronicler_user_prompt_includes_scene_anchors():
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    prompt = build_chronicler_user_prompt(state, world)
    # Location anchor surfaces
    assert "the_slaughtered_lamb" in prompt
    # NPC anchor surfaces (rendered by name, not id)
    assert "Old Tom Bevan" in prompt
    # Beats unlocked surfaces — rendered as the LLM-facing name.
    assert world.beats["tom_named_a_vehicle"].name in prompt
    # Turn-by-turn appears
    assert "Turn 1" in prompt and "Turn 2" in prompt and "Turn 3" in prompt
    # Player picks surface
    assert "Take the stool next to him" in prompt
    # Closing instruction is there
    assert "First person" in prompt or "first person" in prompt


# ───────────────────────── runtime construction ─────────────────────────


def test_make_chronicler_runtime_starts_with_empty_history():
    runtime = make_chronicler_runtime()
    assert isinstance(runtime, ChroniclerRuntime)
    assert runtime.message_history == []


def test_build_chronicler_agent_constructs_cleanly():
    """No schema-construction errors; the agent is freeform str output."""
    world = load_world(FIXTURE_VALID)
    agent = build_chronicler_agent(TestModel(), world)
    # System prompt should carry the bible — cast names surface.
    sys_text = "\n".join(agent._system_prompts)
    assert "Old Tom Bevan" in sys_text
    assert "Maud Pryce" in sys_text
    # Voice rules surface so the agent writes in first person.
    assert "First person" in sys_text or "first person" in sys_text


# ───────────────────────── emit_chronicle_entry ─────────────────────────


@pytest.mark.asyncio
async def test_emit_chronicle_entry_returns_chronicle_entry_with_case_file_and_memory():
    """`emit_chronicle_entry` returns the canonical `ChronicleEntry`; the
    caller (scene_loop.play_session) decomposes it into
    `state.case_file_entries` and `state.npc_memory`. The scene anchor
    lives on state — no denormalization on the output."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    custom_text = (
        "I went into the Slaughtered Lamb that afternoon. Tom was at the "
        "second stool, like Mrs Hughes had said. He named the vehicle, "
        "eventually, and looked relieved when he did. I did not buy him a "
        "drink."
    )
    bundle, runtime = _bundle_and_runtime_with_canned_outputs(world, [ChronicleEntry(
        case_file_entry=custom_text,
        npc_memory_updates=[
            NpcMemoryUpdate(
                npc_id="tom",
                note="Tom unburdened. Named Eluned's Land Rover. Walked himself home.",
            ),
        ],
    )])
    output = await emit_chronicle_entry(bundle, runtime, world, state)

    assert isinstance(output, ChronicleEntry)
    assert output.case_file_entry == custom_text.strip()
    assert len(output.npc_memory_updates) == 1
    assert output.npc_memory_updates[0].npc_id == "tom"
    assert "Land Rover" in output.npc_memory_updates[0].note


@pytest.mark.asyncio
async def test_emit_chronicle_entry_drops_empty_string_memory_updates():
    """If the chronicler emits an empty/whitespace note for a present
    NPC, drop it rather than carrying noise forward into npc_memory.
    String non-emptiness is the one rule the grammar sampler can't
    enforce; the engine-side strip is belt-and-braces."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    bundle, runtime = _bundle_and_runtime_with_canned_outputs(world, [ChronicleEntry(
        case_file_entry="Scene.",
        npc_memory_updates=[NpcMemoryUpdate(npc_id="tom", note="   ")],
    )])
    output = await emit_chronicle_entry(bundle, runtime, world, state)
    assert output.npc_memory_updates == []


@pytest.mark.asyncio
async def test_emit_chronicle_entry_grows_message_history():
    """After the first call, message_history holds the runtime-captured
    messages. After the second call, it holds more — the conversation IS
    the case file."""
    world = load_world(FIXTURE_VALID)
    state = _state_with_one_scene()
    bundle, runtime = _bundle_and_runtime_with_canned_outputs(
        world,
        [
            ChronicleEntry(
                case_file_entry="An entry.",
                npc_memory_updates=[
                    NpcMemoryUpdate(npc_id="tom", note="Tom note one."),
                ],
            ),
            ChronicleEntry(
                case_file_entry="Another entry.",
                npc_memory_updates=[
                    NpcMemoryUpdate(npc_id="tom", note="Tom note two."),
                ],
            ),
        ],
    )

    assert len(runtime.message_history) == 0
    await emit_chronicle_entry(bundle, runtime, world, state)
    after_one = len(runtime.message_history)
    assert after_one > 0

    # Second scene-end — history grows but the existing entries stay.
    await emit_chronicle_entry(bundle, runtime, world, state)
    after_two = len(runtime.message_history)
    assert after_two > after_one


# ───────────────────────── dynamic schema ─────────────────────────


def test_chronicle_entry_rejects_duplicate_npc_ids():
    """Canonical _no_duplicate_npc_ids validator — mirrors
    StateChangeBundle._no_duplicate_beat_ids. Grammar sampling can't
    enforce uniqueness across items, so this validator is the rule
    that fires (and triggers retry under output_retries=3)."""
    with pytest.raises(ValidationError, match="appears twice"):
        ChronicleEntry(
            case_file_entry="Scene.",
            npc_memory_updates=[
                NpcMemoryUpdate(npc_id="tom", note="first"),
                NpcMemoryUpdate(npc_id="tom", note="second"),
            ],
        )


def test_build_chronicle_entry_type_constrains_to_present_npcs():
    """Dynamic schema narrows `npc_id` to Literal[*present_npc_ids] and
    constrains the list length to exactly N — grammar-enforced at decode
    time, model-validated here."""
    DynamicType = _build_chronicle_entry_type(("tom", "maud"))

    # Empty list is rejected (min_length=2).
    with pytest.raises(ValidationError):
        DynamicType.model_validate({
            "case_file_entry": "Scene.",
            "npc_memory_updates": [],
        })

    # Out-of-cast id is rejected (Literal narrowing).
    with pytest.raises(ValidationError):
        DynamicType.model_validate({
            "case_file_entry": "Scene.",
            "npc_memory_updates": [
                {"npc_id": "tom", "note": "ok"},
                {"npc_id": "ghost", "note": "not a present npc"},
            ],
        })

    # Too many records is rejected (max_length=2).
    with pytest.raises(ValidationError):
        DynamicType.model_validate({
            "case_file_entry": "Scene.",
            "npc_memory_updates": [
                {"npc_id": "tom", "note": "a"},
                {"npc_id": "maud", "note": "b"},
                {"npc_id": "tom", "note": "c"},  # also dup but caught by length
            ],
        })

    # Correct shape passes.
    ok = DynamicType.model_validate({
        "case_file_entry": "Scene.",
        "npc_memory_updates": [
            {"npc_id": "tom", "note": "a"},
            {"npc_id": "maud", "note": "b"},
        ],
    })
    assert len(ok.npc_memory_updates) == 2


def test_build_chronicle_entry_type_empty_present_caps_length_zero():
    """When no NPCs are present, the dynamic schema must accept an
    empty list and reject any non-empty one — mirrors the empty-
    reachable path in _build_state_change_bundle_type."""
    DynamicType = _build_chronicle_entry_type(())

    ok = DynamicType.model_validate({
        "case_file_entry": "Alone at the desk.",
        "npc_memory_updates": [],
    })
    assert ok.npc_memory_updates == []

    with pytest.raises(ValidationError):
        DynamicType.model_validate({
            "case_file_entry": "Alone at the desk.",
            "npc_memory_updates": [{"npc_id": "tom", "note": "nope"}],
        })
