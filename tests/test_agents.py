"""M3 — Scene runner agents + dynamic schema builders.

Live-LLM round-trips are marked `integration` and skipped by default.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from handler.agents import (
    _build_state_change_bundle_type,
    build_scene_runner_bundle,
)
from handler.schemas import ChoiceBundle, FreeTextCaption, FreeTextGate
from handler.state import make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def test_state_change_bundle_type_accepts_fireable_beat_names():
    """Phase 1 emits beat NAMES, not ids. Every currently-FIREABLE beat's name
    must round-trip through the dynamic schema (the schema is built from
    `fireable_beat_ids`, the fire-now gate). Use a state at the pub so Tom's
    presence-gated beats are actually fireable and the round-trip is
    non-vacuous."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # Tom is here
    BundleType = _build_state_change_bundle_type(world, state)
    from handler.engine import fireable_beat_ids
    fireable = fireable_beat_ids(world, state)
    assert fireable, "expected at least one fireable beat at the pub"
    for beat_id in fireable:
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


def test_choice_bundle_continue_passes():
    """A 2-option Continue must validate against the canonical ChoiceBundle.
    The choice author now uses the static `ChoiceBundle` directly — no
    dynamic narrowing (FreeText was moved to the engine + gate)."""
    bundle = ChoiceBundle.model_validate({
        "next": {
            "type": "continue",
            "options": [{"text": "stay"}, {"text": "ask"}],
        }
    })
    assert bundle.next.type == "continue"  # type: ignore[union-attr]


def test_choice_bundle_accepts_leave():
    """Leave is a marker variant — no payload required."""
    bundle = ChoiceBundle.model_validate({"next": {"type": "leave"}})
    assert bundle.next.type == "leave"  # type: ignore[union-attr]


def test_choice_bundle_rejects_free_text():
    """FreeText is no longer a choice-author arm — the gate + engine own
    it. The choice author cannot emit it."""
    with pytest.raises(ValidationError):
        ChoiceBundle.model_validate({
            "next": {"type": "free_text", "prompt": "go on", "npc_id": None},
        })


def test_choice_bundle_continue_min_2_options_enforced():
    with pytest.raises(ValidationError):
        ChoiceBundle.model_validate({
            "next": {"type": "continue", "options": [{"text": "only"}]},
        })


def test_free_text_gate_schema():
    """The gate emits a single bool."""
    assert FreeTextGate.model_validate({"offer": True}).offer is True
    assert FreeTextGate.model_validate({"offer": False}).offer is False
    with pytest.raises(ValidationError):
        FreeTextGate.model_validate({})  # offer is required


def test_free_text_caption_schema():
    """The caption author emits a caption + suggested move, both required
    and length-capped."""
    cap = FreeTextCaption.model_validate(
        {"caption": "What do you say?", "suggested": "I ask about the keys."}
    )
    assert cap.caption == "What do you say?"
    assert cap.suggested == "I ask about the keys."
    with pytest.raises(ValidationError):
        FreeTextCaption.model_validate({"caption": "x"})  # suggested required
    with pytest.raises(ValidationError):
        FreeTextCaption.model_validate({"caption": "", "suggested": "y"})  # min_length
    with pytest.raises(ValidationError):
        FreeTextCaption.model_validate({"caption": "x" * 81, "suggested": "y"})  # max_length


def test_build_scene_runner_bundle_constructs_cleanly():
    """End-to-end bundle construction with TestModel as the LLM
    placeholder. Verifies the bundle holds one phase-1, one prose-writer,
    and one choice-author agent (plus populator + chronicler)."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    assert bundle.world is world
    assert bundle.phase1 is not None
    assert bundle.prose_writer is not None
    assert bundle.choice_author is not None
    assert bundle.free_text_gate is not None
    assert bundle.caption_author is not None
    assert bundle.populator is not None
    assert bundle.chronicler is not None


def _agent_system_prompt(agent) -> str:
    """Pull the static system prompt off a pydantic-ai Agent. Static
    prompts (passed at construction) live in `_system_prompts` as a
    tuple of strings."""
    return "\n".join(agent._system_prompts)


def test_phase1_prose_writer_choice_author_system_prompts_diverge():
    """All three Phase-2-relevant system prompts must differ from each other.
    The classifier carries cast knowledge + crime narrative (no beat catalog
    — beats live in the per-turn user prompt); the prose writer has voice
    samples + crime overview only; the choice author has neither bible nor
    voice."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    p1 = _agent_system_prompt(bundle.phase1)
    pw = _agent_system_prompt(bundle.prose_writer)
    ca = _agent_system_prompt(bundle.choice_author)
    assert p1 != pw, "phase 1 and prose writer must diverge"
    assert p1 != ca, "phase 1 and choice author must diverge"
    assert pw != ca, "prose writer and choice author must diverge"
    # Phase 1 carries cast knowledge; the others do not.
    tom = world.npcs["tom"]
    assert tom.knowledge.strip() in p1
    assert tom.knowledge.strip() not in pw
    assert tom.knowledge.strip() not in ca
    # Phase 1 + prose writer carry crime context via ## What Happened.
    assert "## What Happened" in p1
    assert "## What Happened" in pw
    assert "## What Happened" not in ca
    # Style examples remain Phase-2-only.
    assert "## Style Examples" in pw
    assert "## Style Examples" not in p1
    assert "## Style Examples" not in ca
