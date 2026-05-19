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
    """The dynamic schema narrows npc_id / location_id to Literals of
    the world's ids. Validating a NpcPlacement with a non-existent id
    should fail; valid ids should pass."""
    world = load_world(FIXTURE_VALID)
    BundleType = _build_npc_placement_bundle_type(world)

    # Valid placement: tom is a defined NPC; the_slaughtered_lamb is defined.
    valid = BundleType.model_validate({
        "placements": [{"npc_id": "tom", "location_id": "the_slaughtered_lamb"}]
    })
    assert len(valid.placements) == 1
    assert valid.placements[0].npc_id == "tom"

    # Invalid npc_id rejected at validation time.
    with pytest.raises(Exception):
        BundleType.model_validate({
            "placements": [{"npc_id": "ghost", "location_id": "the_slaughtered_lamb"}]
        })

    # Invalid location_id rejected.
    with pytest.raises(Exception):
        BundleType.model_validate({
            "placements": [{"npc_id": "tom", "location_id": "atlantis"}]
        })


def test_npc_placement_bundle_type_empty_bundle_is_default():
    """The grammar-narrowed schema's default is an empty placements list.
    Most scene entries should produce this — no NPCs moved."""
    world = load_world(FIXTURE_VALID)
    BundleType = _build_npc_placement_bundle_type(world)
    bundle = BundleType.model_validate({})
    assert bundle.placements == []


@pytest.mark.asyncio
async def test_emit_placements_returns_canonical_bundle(monkeypatch):
    """emit_placements builds a per-call dynamic schema, runs the agent,
    and returns the result round-tripped through the canonical bundle."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    canned = NpcPlacementBundle(placements=[
        NpcPlacement(npc_id="tom", location_id="the_post_office")
    ])

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
    """No-op populator response: emit_placements returns an empty bundle."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)

    async def fake_run(*args, **kwargs):
        return _FakeRunResult(output=NpcPlacementBundle())

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
    from handler.scene_loop import play_scene

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
        return "A quiet pub."

    async def fake_choices(*args, **kwargs):
        return ChoiceBundle(next=Leave())

    monkeypatch.setattr("handler.scene_loop.emit_placements", fake_placements)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose)
    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices)

    events: list[dict] = []

    async def emit(e):
        events.append(e)

    # Speed up pacing.
    monkeypatch.setattr("handler.sse.WORD_PACE_S", 0.001)

    await play_scene(
        bundle, runtime, state,
        pick_option=lambda _pp: 0,
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
