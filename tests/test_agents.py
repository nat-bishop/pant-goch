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


def test_choice_bundle_type_continue_passes():
    """A 2-option Continue must validate against the dynamic ChoiceBundle."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    ChoiceType = _build_choice_bundle_type(world, state)
    bundle = ChoiceType.model_validate({
        "next": {
            "type": "continue",
            "options": [{"text": "stay"}, {"text": "ask"}],
        }
    })
    assert bundle.next.type == "continue"  # type: ignore[union-attr]


def test_choice_bundle_type_rejects_leave():
    """Leave was removed from the schema. The grammar must NOT accept it
    — scene closure is engine + player only now."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    ChoiceType = _build_choice_bundle_type(world, state)
    with pytest.raises(ValidationError):
        ChoiceType.model_validate({"next": {"type": "leave"}})


def test_choice_bundle_type_free_text_narrows_npc_id_to_present():
    """Tom is at the pub by default. FreeText addressed to tom passes;
    addressed to maud (at the post office) is rejected."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    ChoiceType = _build_choice_bundle_type(world, state)

    bundle = ChoiceType.model_validate({
        "next": {"type": "free_text", "prompt": "go on", "npc_id": "tom"},
    })
    assert bundle.next.npc_id == "tom"  # type: ignore[union-attr]

    with pytest.raises(ValidationError):
        ChoiceType.model_validate({
            "next": {"type": "free_text", "prompt": "go on", "npc_id": "maud"},
        })


def test_choice_bundle_type_continue_min_2_options_enforced():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    ChoiceType = _build_choice_bundle_type(world, state)
    with pytest.raises(ValidationError):
        ChoiceType.model_validate({
            "next": {"type": "continue", "options": [{"text": "only"}]},
        })


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
    assert bundle.populator is not None
    assert bundle.chronicler is not None


def _agent_system_prompt(agent) -> str:
    """Pull the static system prompt off a pydantic-ai Agent. Static
    prompts (passed at construction) live in `_system_prompts` as a
    tuple of strings."""
    return "\n".join(agent._system_prompts)


def test_phase1_prose_writer_choice_author_system_prompts_diverge():
    """All three Phase-2-relevant system prompts must differ from each
    other. The classifier has the full bible; the prose writer has voice
    samples + crime overview only; the choice author has neither bible
    nor voice."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    p1 = _agent_system_prompt(bundle.phase1)
    pw = _agent_system_prompt(bundle.prose_writer)
    ca = _agent_system_prompt(bundle.choice_author)
    assert p1 != pw, "phase 1 and prose writer must diverge"
    assert p1 != ca, "phase 1 and choice author must diverge"
    assert pw != ca, "prose writer and choice author must diverge"
    # Phase 1 carries beat structure; the others do not.
    assert "[THE TRUTH]" in p1
    assert "[THE TRUTH]" not in pw
    assert "[THE TRUTH]" not in ca
    # Prose writer carries voice + crime overview.
    assert "[WHAT HAPPENED]" in pw
    assert "[STYLE EXAMPLES]" in pw
    # Choice author has neither.
    assert "[WHAT HAPPENED]" not in ca
    assert "[STYLE EXAMPLES]" not in ca
