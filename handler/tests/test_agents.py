"""M3 — Scene runner agents + dynamic schema builders.

Live-LLM round-trips are marked `integration` and skipped by default.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from handler.agents import (
    _build_choice_bundle_type,
    _build_state_change_bundle_type,
    build_scene_runner_bundle,
)
from handler.state import State, make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def test_state_change_bundle_type_accepts_reachable_beat_names():
    """Phase 1 emits beat NAMES, not ids. Every initially-reachable beat's
    name must round-trip through the dynamic schema."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    BundleType = _build_state_change_bundle_type(world, state)
    from handler.engine import reachable_beat_ids
    for beat_id in reachable_beat_ids(world, state):
        beat_name = world.beats[beat_id].name
        bundle = BundleType.model_validate({
            "state_changes": [
                {"type": "unlock_beat", "beat_name": beat_name}
            ]
        })
        assert bundle.state_changes[0].beat_name == beat_name  # type: ignore[union-attr]
        # Boundary translation: name → id is the engine's job in emit_state_changes.
        assert world.beat_id_for_name(beat_name) == beat_id


def test_state_change_bundle_type_rejects_unknown_beat_name():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [
                {"type": "unlock_beat", "beat_name": "Ghost beat"}
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


def test_choice_bundle_type_accepts_authored_adjacent_for_leaves_for():
    """At the post office (adjacent to pub + vicarage), leaves_for must
    accept those two and reject anything else."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    ChoiceType = _build_choice_bundle_type(world, state)
    bundle = ChoiceType.model_validate({
        "options": [
            {"text": "stay"},
            {"text": "leave for pub", "leaves_for": "the_slaughtered_lamb"},
        ],
    })
    assert bundle.options[1].leaves_for == "the_slaughtered_lamb"


def test_choice_bundle_type_rejects_non_adjacent_leaves_for():
    """At the post office, knighton is not adjacent — must be rejected
    by the Literal."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    ChoiceType = _build_choice_bundle_type(world, state)
    with pytest.raises(ValidationError):
        ChoiceType.model_validate({
            "options": [
                {"text": "leave for knighton", "leaves_for": "knighton"},
            ],
        })


def test_choice_bundle_requires_at_least_one_leave_option():
    """Canonical validator: ≥1 option must have leaves_for set."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    ChoiceType = _build_choice_bundle_type(world, state)
    with pytest.raises(ValidationError, match="leaves_for"):
        ChoiceType.model_validate({
            "options": [
                {"text": "wait"},
                {"text": "look around"},
            ],
        })


def test_build_scene_runner_bundle_constructs_cleanly():
    """Bundle holds the five agents post-split."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    assert bundle.world is world
    assert bundle.phase1 is not None
    assert bundle.prose_writer is not None
    assert bundle.choice_author is not None
    assert bundle.populator is not None
    assert bundle.chronicler is not None


def _agent_system_prompt(agent) -> str:
    """Pull the static system prompt off a pydantic-ai Agent."""
    return "\n".join(agent._system_prompts)


def test_phase1_prose_writer_choice_author_prompts_diverge():
    """Each agent carries its own system prompt. Phase 1 has the full
    bible; prose writer has voice + crime + endings; choice author has
    the smallest prompt of the three."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    p1 = _agent_system_prompt(bundle.phase1)
    pw = _agent_system_prompt(bundle.prose_writer)
    ca = _agent_system_prompt(bundle.choice_author)

    assert p1 != pw != ca
    # Phase 1 has the full beat catalog.
    assert "[THE TRUTH]" in p1
    # Prose writer has voice + crime + endings — but no beat catalog.
    assert "[THE TRUTH]" not in pw
    assert "[WHAT HAPPENED]" in pw
    assert "[STYLE EXAMPLES]" in pw
    assert "[ENDINGS]" in pw
    assert "[TOOLS]" in pw
    # Choice author is the smallest.
    assert len(ca) < len(pw)
    assert len(ca) < len(p1)
    assert "[STYLE EXAMPLES]" not in ca
    assert "[THE TRUTH]" not in ca
