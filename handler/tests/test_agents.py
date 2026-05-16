"""M3 — Scene runner agents + dynamic schema builders.

Live-LLM round-trips are marked `integration` and skipped by default.
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
from handler.schemas import Option
from handler.state import State, make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def test_state_change_bundle_type_accepts_reachable_beat_ids():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    BundleType = _build_state_change_bundle_type(world, state)
    # Every initially-reachable beat id is accepted by the dynamic schema.
    from handler.engine import reachable_beat_ids
    for beat_id in reachable_beat_ids(world, state):
        bundle = BundleType.model_validate({
            "state_changes": [
                {"type": "unlock_beat", "beat_id": beat_id}
            ]
        })
        assert bundle.state_changes[0].beat_id == beat_id  # type: ignore[union-attr]


def test_state_change_bundle_type_rejects_unknown_beat_id():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [
                {"type": "unlock_beat", "beat_id": "ghost_beat"}
            ]
        })


def test_state_change_bundle_type_accepts_empty():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    BundleType = _build_state_change_bundle_type(world, state)
    bundle = BundleType.model_validate({"state_changes": []})
    assert bundle.state_changes == []
    bundle = BundleType.model_validate({})
    assert bundle.state_changes == []


def test_prose_part_type_accepts_authored_adjacent_for_leaves_for():
    """At the post office (adjacent to pub + vicarage), leaves_for must
    accept those two and reject anything else."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    ProseType = _build_prose_part_type(world, state)
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
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    ProseType = _build_prose_part_type(world, state)
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


def test_prose_part_type_accepts_satisfied_ending_id():
    """End-game ending_id is narrowed to currently-satisfied endings only.
    Satisfy the prereqs in state, then assert the ending is acceptable."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    # Satisfy the prereqs of published_story.
    ending = world.endings["published_story"]
    state.unlocked_beats = set(ending.requires_beats)
    ProseType = _build_prose_part_type(world, state)
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {"type": "end_game", "ending_id": "published_story"},
    })
    assert pp.next.ending_id == "published_story"  # type: ignore[union-attr]


def test_prose_part_type_rejects_unknown_ending_id():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    ProseType = _build_prose_part_type(world, state)
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {"type": "end_game", "ending_id": "imaginary_ending"},
        })


def test_prose_part_type_end_scene_passes():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    ProseType = _build_prose_part_type(world, state)
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {"type": "end_scene"},
    })
    assert pp.next.type == "end_scene"  # type: ignore[union-attr]


def test_prose_part_type_leave_pick_forces_end_scene():
    """The regression that originally motivated `__base__=`. When leave_pick
    is set, the dynamic ProsePart's `next` is `EndScene` (no union) — so
    Continue/EndGame are rejected at parse time, before any application
    validation can run. This test would have been the canary the original
    bug needed."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    leave_pick = Option(text="leave", leaves_for="the_slaughtered_lamb")
    ProseType = _build_prose_part_type(world, state, leave_pick=leave_pick)

    # ContinueScene rejected.
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {
                "type": "continue",
                "options": [
                    {"text": "stay"},
                    {"text": "leave", "leaves_for": "the_slaughtered_lamb"},
                ],
            },
        })

    # EndGame rejected.
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {"type": "end_game", "ending_id": "published_story"},
        })

    # EndScene accepted.
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {"type": "end_scene"},
    })
    assert pp.next.type == "end_scene"


def test_build_scene_runner_bundle_constructs_cleanly():
    """End-to-end bundle construction with TestModel as the LLM
    placeholder. Verifies the bundle holds one phase-1 and one phase-2
    agent (no N+1 per-location dict)."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    assert bundle.world is world
    assert bundle.phase1 is not None
    assert bundle.phase2 is not None


def _agent_system_prompt(agent) -> str:
    """Pull the static system prompt off a pydantic-ai Agent. Static
    prompts (passed at construction) live in `_system_prompts` as a
    tuple of strings."""
    return "\n".join(agent._system_prompts)


def test_phase1_and_phase2_system_prompts_diverge():
    """Phase 1 and Phase 2 carry DIFFERENT system prompts post-split. The
    classifier has the full bible; the narrator has voice samples and
    crime overview only. Asserting divergence locks in the split."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    p1_sys = _agent_system_prompt(bundle.phase1)
    p2_sys = _agent_system_prompt(bundle.phase2)
    assert p1_sys != p2_sys, "phase 1 and phase 2 must have different system prompts"
    # Phase 1 carries beat structure; Phase 2 does not.
    assert "[THE TRUTH]" in p1_sys
    assert "[THE TRUTH]" not in p2_sys
    assert "[WHAT HAPPENED]" in p2_sys
    assert "[STYLE EXAMPLES]" in p2_sys
