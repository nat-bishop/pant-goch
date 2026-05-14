"""M3 — Scene runner agent + dynamic schema builders.

Tests build the bundle and verify the dynamic types narrow id fields to
per-world Literal enums. Live-LLM round-trips are marked `integration`
and skipped by default.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from handler_v2.agents import (
    _build_prose_part_type,
    _build_state_change_bundle_type,
    build_scene_runner_bundle,
)
from handler_v2.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def test_state_change_bundle_type_accepts_authored_beat_ids():
    world = load_world(FIXTURE_VALID)
    BundleType = _build_state_change_bundle_type(world)
    # All three authored beat ids are accepted.
    for beat_id in world.beats:
        bundle = BundleType.model_validate({
            "state_changes": [
                {"type": "unlock_beat", "beat_id": beat_id}
            ]
        })
        assert bundle.state_changes[0].beat_id == beat_id  # type: ignore[union-attr]


def test_state_change_bundle_type_rejects_unknown_beat_id():
    world = load_world(FIXTURE_VALID)
    BundleType = _build_state_change_bundle_type(world)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [
                {"type": "unlock_beat", "beat_id": "ghost_beat"}
            ]
        })


def test_state_change_bundle_type_accepts_empty():
    world = load_world(FIXTURE_VALID)
    BundleType = _build_state_change_bundle_type(world)
    bundle = BundleType.model_validate({"state_changes": []})
    assert bundle.state_changes == []
    bundle = BundleType.model_validate({})
    assert bundle.state_changes == []


def test_prose_part_type_accepts_authored_adjacent_for_leaves_for():
    """At the post office (adjacent to pub + vicarage), leaves_for must
    accept those two and reject anything else."""
    world = load_world(FIXTURE_VALID)
    ProseType = _build_prose_part_type(world, "the_post_office")
    # Valid leave to pub
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {
            "type": "continue",
            "options": [
                {"text": "stay"},
                {"text": "leave for pub", "leaves_for": "the_slaughtered_lamb"},
            ],
        },
    })
    assert pp.next.options[1].leaves_for == "the_slaughtered_lamb"  # type: ignore[union-attr]


def test_prose_part_type_rejects_non_adjacent_leaves_for():
    """At the post office, knighton is NOT adjacent (and not even in the
    fixture) — must be rejected by the Literal."""
    world = load_world(FIXTURE_VALID)
    ProseType = _build_prose_part_type(world, "the_post_office")
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {
                "type": "continue",
                "options": [
                    {"text": "leave for knighton", "leaves_for": "knighton"},
                ],
            },
        })


def test_prose_part_type_accepts_authored_ending_id():
    world = load_world(FIXTURE_VALID)
    ProseType = _build_prose_part_type(world, "the_post_office")
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {"type": "end_game", "ending_id": "published_story"},
    })
    assert pp.next.ending_id == "published_story"  # type: ignore[union-attr]


def test_prose_part_type_rejects_unknown_ending_id():
    world = load_world(FIXTURE_VALID)
    ProseType = _build_prose_part_type(world, "the_post_office")
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {"type": "end_game", "ending_id": "imaginary_ending"},
        })


def test_prose_part_type_end_scene_passes():
    world = load_world(FIXTURE_VALID)
    ProseType = _build_prose_part_type(world, "the_post_office")
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {"type": "end_scene"},
    })
    assert pp.next.type == "end_scene"  # type: ignore[union-attr]


def test_build_scene_runner_bundle_constructs_cleanly():
    """End-to-end bundle construction with TestModel as the LLM
    placeholder. Verifies no schema-construction errors and that the
    bundle holds one phase-2 agent per location."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    assert bundle.world is world
    # One phase-2 agent per location (output_type has location-specific
    # adjacents Literal baked in).
    assert set(bundle.phase2_by_location) == set(world.locations)


def test_all_agents_share_system_prompt():
    """Phase 1 + every phase 2 agent must have byte-identical system
    prompts so they share KV cache server-side regardless of which agent
    issues the HTTP request (cf. plan §"Context construction")."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    p1_sys = str(bundle.phase1._instructions)
    for loc_id, p2 in bundle.phase2_by_location.items():
        assert str(p2._instructions) == p1_sys, (
            f"phase2[{loc_id}] system prompt diverges from phase1"
        )
