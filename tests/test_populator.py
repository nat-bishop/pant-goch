"""Populator agent + emit_placements.

The populator runs once at scene entry, deciding which NPCs are at which
locations now. Empty bundle is the no-op default (most scenes). Tests
stub the agent's run method to verify the engine integration (schema
narrowing, placement application, state mutation) without invoking a
live model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from handler.agents import (
    _build_npc_placement_bundle_type,
    build_scene_runner_bundle,
    emit_placements,
)
from handler.schemas import NpcPlacement, NpcPlacementBundle
from handler.state import make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


@dataclass
class _FakeRunResult:
    output: Any
    messages: list[Any] = field(default_factory=list)

    def all_messages(self) -> list[Any]:
        return list(self.messages)


def test_npc_placement_bundle_type_narrows_to_world_ids():
    """The dynamic schema is a fixed-shape object: one field per NPC, value a
    Literal over the world's location ids plus the "stay" sentinel. Unknown NPC
    keys and invalid location values are both rejected."""
    world = load_world(FIXTURE_VALID)
    BundleType = _build_npc_placement_bundle_type(world)

    # Valid: tom -> a defined location.
    valid = BundleType.model_validate({"tom": "the_slaughtered_lamb"})
    assert valid.tom == "the_slaughtered_lamb"
    # "stay" is the keep-current sentinel and is always valid.
    assert BundleType.model_validate({"tom": "stay"}).tom == "stay"

    # Unknown npc key rejected (extra="forbid").
    with pytest.raises(Exception):
        BundleType.model_validate({"ghost": "the_slaughtered_lamb"})

    # Invalid location value rejected (Literal).
    with pytest.raises(Exception):
        BundleType.model_validate({"tom": "atlantis"})


def test_npc_placement_bundle_type_empty_bundle_is_default():
    """The object's default is every NPC at "stay" (keep current location).
    Most scene entries should produce this — no NPCs moved."""
    world = load_world(FIXTURE_VALID)
    BundleType = _build_npc_placement_bundle_type(world)
    bundle = BundleType.model_validate({})
    assert world.npcs  # sanity: the fixture has a cast
    for npc_id in world.npcs:
        assert getattr(bundle, npc_id) == "stay"


def test_npc_placement_bundle_type_has_no_array_fields():
    """The runaway fix, locked in: the populator wire schema must contain no
    list field. An array is exactly what the grammar can't bound and what the
    model loops on, so a future refactor must not reintroduce one."""
    import typing

    world = load_world(FIXTURE_VALID)
    BundleType = _build_npc_placement_bundle_type(world)
    for name, f in BundleType.model_fields.items():
        origins = [typing.get_origin(f.annotation)] + [
            typing.get_origin(a) for a in typing.get_args(f.annotation)
        ]
        assert list not in origins, (
            f"NpcPlacementBundle.{name} is/contains a list — array runaway risk"
        )


@pytest.mark.asyncio
async def test_emit_placements_returns_canonical_bundle(monkeypatch):
    """emit_placements reads the fixed-shape wire object (one field per NPC),
    skips "stay", and returns the canonical list-shaped bundle."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    # Wire object: tom moved, every other NPC defaults to "stay".
    WireType = _build_npc_placement_bundle_type(world)
    canned = WireType(tom="the_post_office")

    async def fake_run(*args, **kwargs):
        return _FakeRunResult(output=canned)

    monkeypatch.setattr(bundle.populator, "run", fake_run)

    result = await emit_placements(bundle, state)
    assert isinstance(result, NpcPlacementBundle)
    assert len(result.placements) == 1
    assert result.placements[0].npc_id == "tom"
    assert result.placements[0].location_id == "the_post_office"


@pytest.mark.asyncio
async def test_emit_placements_empty_bundle_passes_through(monkeypatch):
    """No-op populator response (every NPC "stay"): emit_placements returns an
    empty canonical bundle."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)

    WireType = _build_npc_placement_bundle_type(world)

    async def fake_run(*args, **kwargs):
        return _FakeRunResult(output=WireType())  # all defaults = "stay"

    monkeypatch.setattr(bundle.populator, "run", fake_run)
    result = await emit_placements(bundle, state)
    assert result.placements == []


@pytest.mark.asyncio
async def test_play_scene_applies_populator_placements(monkeypatch):
    """When the populator emits a placement, play_scene mutates
    state.npc_locations and emits an npc_moved event."""
    from handler.agents import (
        make_scene_runner_runtime,
    )
    from handler.schemas import (
        ChoiceBundle,
        Leave,
        StateChangeBundle,
    )
    from handler.scene_loop import WalkTo, play_scene

    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    # Tom defaults to the pub; we'll have the populator move him to the post
    # office for this scene.
    assert state.npc_locations["tom"] == "the_slaughtered_lamb"

    async def fake_placements(*args, **kwargs):
        return NpcPlacementBundle(placements=[
            NpcPlacement(npc_id="tom", location_id="the_post_office")
        ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_prose(*args, **kwargs):
        yield "A quiet pub."

    async def fake_choices(*args, **kwargs):
        return ChoiceBundle(next=Leave())

    async def fake_gate(*args, **kwargs):
        return False

    async def fake_closing(*args, **kwargs):
        return "You step back out."

    monkeypatch.setattr("handler.scene_loop.emit_placements", fake_placements)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.stream_prose_text", fake_prose)
    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices)
    monkeypatch.setattr("handler.scene_loop.emit_free_text_gate", fake_gate)
    monkeypatch.setattr("handler.scene_loop.emit_closing_prose", fake_closing)

    events: list[dict] = []

    async def emit(e):
        events.append(e)

    # Speed up pacing.
    monkeypatch.setattr("handler.sse.WORD_PACE_S", 0.001)

    # The scene soft-closes (Leave, no ending); the player walks out.
    await play_scene(
        bundle, runtime, state,
        request_player_input=lambda pp, s, w: WalkTo(
            location_id=w.locations[s.player_location].adjacent_to[0]
        ),
        emit_event=emit,
    )

    # State mutation: tom moved.
    assert state.npc_locations["tom"] == "the_post_office"
    # Event emitted.
    npc_moves = [e for e in events if e["event"] == "npc_moved"]
    assert len(npc_moves) == 1
    assert npc_moves[0]["npc_id"] == "tom"
    assert npc_moves[0]["from"] == "the_slaughtered_lamb"
    assert npc_moves[0]["to"] == "the_post_office"
