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

from handler.agents import (
    _build_prose_part_type,
    _build_state_change_bundle_type,
    build_scene_runner_bundle,
)
from handler.world import load_world


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


def _agent_system_prompt(agent) -> str:
    """Pull the static system prompt off a pydantic-ai Agent. Static
    prompts (passed at construction) live in `_system_prompts` as a
    tuple of strings."""
    return "\n".join(agent._system_prompts)


def test_phase2_agents_share_system_prompt():
    """All phase 2 agents (one per location) must have byte-identical
    system prompts. Phase 2's system prompt is the narrator's voice
    prompt — same across every location so KV cache hits regardless of
    which per-location agent runs."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    sys_prompts = {
        loc_id: _agent_system_prompt(p2)
        for loc_id, p2 in bundle.phase2_by_location.items()
    }
    distinct = set(sys_prompts.values())
    assert len(distinct) == 1, (
        f"phase 2 agents have divergent system prompts: "
        f"{sorted(sys_prompts.keys())} → {len(distinct)} distinct prompts"
    )


def test_phase1_and_phase2_system_prompts_diverge():
    """Phase 1 and Phase 2 carry DIFFERENT system prompts post-split. The
    classifier has the full bible; the narrator has voice samples and
    crime overview only. Asserting divergence locks in the split."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    p1_sys = _agent_system_prompt(bundle.phase1)
    any_p2 = next(iter(bundle.phase2_by_location.values()))
    p2_sys = _agent_system_prompt(any_p2)
    assert p1_sys != p2_sys, "phase 1 and phase 2 must have different system prompts"
    # Phase 1 carries beat structure; Phase 2 does not.
    assert "[THE TRUTH]" in p1_sys
    assert "[THE TRUTH]" not in p2_sys
    assert "[WHAT HAPPENED]" in p2_sys
    assert "[STYLE EXAMPLES]" in p2_sys
