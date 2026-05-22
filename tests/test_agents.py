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


def test_bundle_accepts_present_npc_delta_and_emittable_commitment():
    """Phase 1 reports quality deltas for present NPCs (npc_id narrowed) and
    may emit one emittable commitment by NAME (the engine translates name→id).
    """
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # Tom is here
    BundleType = _build_state_change_bundle_type(world, state)
    b = BundleType.model_validate({
        "npc_deltas": [{"npc_id": "tom", "trust": "+1"}],
    })
    assert b.npc_deltas[0].npc_id == "tom"  # type: ignore[union-attr]
    assert b.attention == "0"  # scalar, defaults to no-change

    # At the post office, the observation milestone is an emittable commitment.
    state.player_location = "the_post_office"
    BundleType2 = _build_state_change_bundle_type(world, state)
    name = world.beats["read_the_letter"].name
    b2 = BundleType2.model_validate({
        "commitments": [{"type": "unlock_beat", "beat_name": name}],
    })
    assert b2.commitments[0].beat_name == name  # type: ignore[union-attr]
    assert world.beat_id_for_name(name) == "read_the_letter"


def test_bundle_rejects_unknown_commitment_name():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_post_office"
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "commitments": [{"type": "unlock_beat", "beat_name": "Ghost beat"}]
        })


def test_bundle_accepts_empty():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    BundleType = _build_state_change_bundle_type(world, state)
    bundle = BundleType.model_validate(
        {"npc_deltas": [], "attention": "0", "commitments": []}
    )
    assert bundle.npc_deltas == [] and bundle.attention == "0" and bundle.commitments == []
    bundle = BundleType.model_validate({})
    assert bundle.npc_deltas == [] and bundle.attention == "0" and bundle.commitments == []


def test_state_change_bundle_rejects_duplicate_delta_npc():
    """Structural cap: one quality delta per NPC per turn, so a single bundle
    can't move standing by ±2 and skip a confession band."""
    from handler.schemas import StateChangeBundle
    with pytest.raises(ValidationError):
        StateChangeBundle.model_validate({
            "deltas": [
                {"type": "quality_delta", "npc_id": "tom", "trust": "+1"},
                {"type": "quality_delta", "npc_id": "tom", "trust": "-1"},
            ],
        })
    # Distinct NPCs + an attention delta (no npc_id) validate fine.
    ok = StateChangeBundle.model_validate({
        "deltas": [
            {"type": "quality_delta", "npc_id": "tom", "trust": "+1"},
            {"type": "quality_delta", "npc_id": "helen", "trust": "+1"},
            {"type": "attention_delta", "attention": "+1"},
        ],
    })
    assert len(ok.deltas) == 3


async def test_emit_state_changes_drops_noop_and_dedups_npc(monkeypatch):
    """emit_state_changes filters no-op deltas (trust=guard="0", attention="0")
    and keeps the first delta per NPC — the grammar narrows ids/values but not
    repeats, so this is what holds the ±1/turn and filler-metric contracts."""
    from types import SimpleNamespace
    import handler.agents as agents_mod
    from handler.agents import build_scene_runner_bundle, emit_state_changes

    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # Tom present

    raw = SimpleNamespace(
        npc_deltas=[
            SimpleNamespace(npc_id="tom", trust="0", guard="0"),   # no-op
            SimpleNamespace(npc_id="tom", trust="+1", guard="0"),  # kept
            SimpleNamespace(npc_id="tom", trust="+1", guard="0"),  # dup npc
        ],
        attention="+1",   # kept
        commitments=[],
    )

    async def fake_run(agent, prompt, output_type):
        return SimpleNamespace(output=raw)

    monkeypatch.setattr(agents_mod, "_run_decision_agent", fake_run)

    bundle = build_scene_runner_bundle(TestModel(), world)
    out = await emit_state_changes(bundle, state)

    quality = [d for d in out.deltas if getattr(d, "npc_id", None) is not None]
    attention = [d for d in out.deltas if getattr(d, "npc_id", None) is None]
    assert len(quality) == 1
    assert quality[0].npc_id == "tom" and quality[0].trust == "+1"
    assert len(attention) == 1 and attention[0].attention == "+1"


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


# ───────────────────────── decision-agent cap + retry ─────────────────────────


def test_decision_agents_carry_max_tokens_cap():
    """The two empty-list decision agents (phase1, populator) get a per-agent
    output cap so a runaway reasoning loop fails fast; the content agents keep
    the model default (no cap) — prose/chronicle legitimately need long output."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world, decision_max_tokens=500)

    for agent in (bundle.phase1, bundle.populator):
        assert agent.model_settings is not None
        assert agent.model_settings.get("max_tokens") == 500

    for agent in (
        bundle.prose_writer,
        bundle.choice_author,
        bundle.free_text_gate,
        bundle.caption_author,
        bundle.chronicler,
    ):
        settings = agent.model_settings or {}
        assert "max_tokens" not in settings


def test_decision_agents_keep_thinking_suppression():
    """Regression guard: a per-agent extra_body REPLACES the model-level one
    (pydantic-ai merges ModelSettings shallowly), so the capped agents must
    re-include the enable_thinking=False payload or they silently re-enable
    the reasoning channel — the very leak the cap exists to contain."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world, decision_max_tokens=500)

    for agent in (bundle.phase1, bundle.populator):
        extra_body = (agent.model_settings or {}).get("extra_body", {})
        assert extra_body["chat_template_kwargs"]["enable_thinking"] is False


async def test_run_decision_agent_retries_then_succeeds():
    """_run_decision_agent re-rolls on UnexpectedModelBehavior (a capped
    runaway) and returns the first clean result."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from handler.agents import _run_decision_agent

    class _FlakyAgent:
        name = "populator"

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, prompt, *, output_type):
            self.calls += 1
            if self.calls < 3:
                raise UnexpectedModelBehavior("token limit exceeded")
            return "ok"

    agent = _FlakyAgent()
    result = await _run_decision_agent(agent, "prompt", object(), attempts=3)
    assert result == "ok"
    assert agent.calls == 3


async def test_run_decision_agent_reraises_after_exhaustion():
    """Once attempts are spent the last UnexpectedModelBehavior propagates, so
    the scene_loop's empty-bundle fallback still fires."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from handler.agents import _run_decision_agent

    class _AlwaysFails:
        name = "phase1"

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, prompt, *, output_type):
            self.calls += 1
            raise UnexpectedModelBehavior("token limit exceeded")

    agent = _AlwaysFails()
    with pytest.raises(UnexpectedModelBehavior):
        await _run_decision_agent(agent, "prompt", object(), attempts=2)
    assert agent.calls == 2
