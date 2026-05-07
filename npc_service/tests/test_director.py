"""Unit-test coverage for `_select_beat` — the Director / scene-continuity
seam.

These tests exercise `_select_beat` directly with stubbed Director runs:

1. When ≥2 beats are due (and no scene-continuity short-circuit fires),
   the Director is called with the full eligible list reflected in the
   dynamic Pydantic output-type Literal.
2. When exactly 1 beat is due, the Director is NOT called (the 0/1
   short-circuits in `_select_beat` keep cost down).
3. When the player's reply lands on a thread where a scene with prior
   fires is still eligible, `_select_beat` short-circuits to "continue"
   without consulting the Director.

We test against the production scenario.yaml because the Director's
prompt summarises beat intents — the synthetic spine fixture has
empty intents and would not exercise the same code path.
"""
from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from npc_service import main
from npc_service.agents import build_agents
from npc_service.models import (
    AxisResult,
    CanonCategory,
    Email,
    InboxState,
    PlayerCanonEntry,
    Reply,
    Scenario,
    load_scenario,
)
from npc_service.spine import find_due_beats


_NATIVE_OUTPUT_PROFILE = ModelProfile(
    supports_json_schema_output=True, default_structured_output_mode="native"
)


@pytest.fixture
def production_scenario() -> Scenario:
    return load_scenario(Path(__file__).resolve().parents[1] / "scenario.yaml")


def _seed_canon_name(state: InboxState, name: str = "Marcus") -> None:
    """Close calibration by dropping a name into player_canon — every
    Phase-2 body beat gates on `*calibration_done` and these fixtures
    target the body-beat phase."""
    state.player_canon["name"] = PlayerCanonEntry(
        summary=name,
        literal_text_quote=f"I'm {name}",
        salience=0.9,
        category=CanonCategory.NAME,
        turn_of_origin=0,
    )


@pytest.fixture
def state_with_two_eligible_scenes(production_scenario: Scenario) -> InboxState:
    """Engineer state where `aria.bargain_scene` AND `aria.recognition_scene`
    are both eligible on the same tick.

    bargain_scene requires: calibration_done, replies(aria)>=4.
    recognition_scene requires: replies(aria)>=5, creature_grievance>=2.0,
    BeatFired(aria.calibration_scene). With replies=5 + canon name +
    grievance 2.0 + calibration_scene marked fired, both gates pass.
    """
    state = InboxState.from_scenario(production_scenario)
    # Drain boot — fires aria.welcome.
    for beat in find_due_beats(state, production_scenario)[0]:
        state.fired_beat_ids.add(beat.id)
    _seed_canon_name(state)
    # Mark the calibration scene completed so the recognition scene's
    # BeatFired(aria.calibration_scene) precondition passes.
    state.fired_beat_ids.add("aria.calibration_scene")
    state.replies.extend(
        Reply(
            id=f"r{i}", thread_id="aria", beat_responded_to=None,
            recipient="Aria", body="(t)",
        )
        for i in range(5)
    )
    state.qualities["creature_grievance"] = 2.0
    return state


@pytest.fixture
def state_with_only_calibration_due(production_scenario: Scenario) -> InboxState:
    """1 reply, no canon name → only aria.calibration_scene is eligible.
    With this setup the bargain / recognition scenes (post-calibration)
    haven't unlocked yet."""
    state = InboxState.from_scenario(production_scenario)
    for beat in find_due_beats(state, production_scenario)[0]:
        state.fired_beat_ids.add(beat.id)
    state.replies.append(Reply(
        id="r0", thread_id="aria", beat_responded_to=None,
        recipient="Aria", body="(t)",
    ))
    return state


def _swap_globals(monkeypatch, scenario: Scenario, state: InboxState):
    """`_select_beat` reads STATE / AGENTS / SCENARIO from module globals.
    Tests inject fixtures by overriding those slots."""
    placeholder_model = TestModel(custom_output_text="placeholder")
    agents = build_agents(placeholder_model)
    monkeypatch.setattr(main, "STATE", state)
    monkeypatch.setattr(main, "SCENARIO", scenario)
    monkeypatch.setattr(main, "AGENTS", agents)
    return agents


@pytest.mark.asyncio
async def test_director_receives_both_ids_when_two_scenes_eligible(
    production_scenario, state_with_two_eligible_scenes, monkeypatch,
):
    """Both `aria.bargain_scene` and `aria.recognition_scene` are due →
    `_select_beat` builds the dynamic output type with both ids in its
    closed Literal, invokes the Director, and honors its choice. We pass
    `reply_thread_id=None` so the scene-continuity short-circuit doesn't
    fire (it would otherwise pre-empt the Director when there's a prior
    fire on a same-thread scene).
    """
    agents = _swap_globals(
        monkeypatch, production_scenario, state_with_two_eligible_scenes,
    )

    captured: dict = {}

    class _FakeOutput:
        chosen_beat_id = "aria.bargain_scene"
        rationale = "advance the arc"

    class _FakeResult:
        output = _FakeOutput()

    async def _stub_run(user_prompt, *, output_type, **kwargs):
        captured["user_prompt"] = user_prompt
        captured["output_type"] = output_type
        return _FakeResult()

    monkeypatch.setattr(agents.director_agent, "run", _stub_run)

    eligible, _report = find_due_beats(
        state_with_two_eligible_scenes, production_scenario,
    )
    eligible_ids = {b.id for b in eligible}
    # Sanity: both contested ids are actually in the due list.
    assert "aria.bargain_scene" in eligible_ids
    assert "aria.recognition_scene" in eligible_ids
    assert len(eligible) >= 2  # confirms we hit the Director branch

    chosen, selection, rationale = await main._select_beat(
        eligible, reply_thread_id=None,
    )

    assert selection == "director"
    assert chosen is not None
    assert chosen.id == "aria.bargain_scene"
    assert rationale == "advance the arc"

    # The dynamic Pydantic output_type's chosen_beat_id Literal must offer
    # every eligible id plus "none" — the schema is what stops the LLM from
    # returning an off-list id.
    output_model = captured["output_type"].outputs  # NativeOutput wraps the model
    field = output_model.model_fields["chosen_beat_id"]
    literal_values = set(get_args(field.annotation))
    for due_id in eligible_ids:
        assert due_id in literal_values, (
            f"{due_id!r} not offered to Director; literal was {literal_values}"
        )
    assert "none" in literal_values


@pytest.mark.asyncio
async def test_director_shortcircuits_when_only_one_beat_due(
    production_scenario, state_with_only_calibration_due, monkeypatch,
):
    """One eligible beat → `_select_beat` returns it as `selection="sole"`
    without calling the Director. Tests the cost-saving short-circuit."""
    agents = _swap_globals(
        monkeypatch, production_scenario, state_with_only_calibration_due,
    )

    call_count = {"n": 0}

    async def _boom(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError("Director was called for a single-eligible tick")

    monkeypatch.setattr(agents.director_agent, "run", _boom)

    eligible, _report = find_due_beats(
        state_with_only_calibration_due, production_scenario,
    )
    assert len(eligible) == 1, (
        f"fixture should make exactly one beat eligible; got "
        f"{[b.id for b in eligible]}"
    )

    chosen, selection, rationale = await main._select_beat(eligible)

    assert selection == "sole"
    assert chosen is not None
    assert chosen.id == "aria.calibration_scene"
    assert rationale is None
    assert call_count["n"] == 0


@pytest.mark.asyncio
async def test_director_none_means_skip_tick(
    production_scenario, state_with_two_eligible_scenes, monkeypatch,
):
    """Director may return `chosen_beat_id="none"` to skip the tick — the
    contract that lets the Director punt when no eligible beat fits the
    moment."""
    agents = _swap_globals(
        monkeypatch, production_scenario, state_with_two_eligible_scenes,
    )

    class _FakeOutput:
        chosen_beat_id = "none"
        rationale = "no beat fits the moment"

    class _FakeResult:
        output = _FakeOutput()

    async def _stub_run(user_prompt, *, output_type, **kwargs):
        return _FakeResult()

    monkeypatch.setattr(agents.director_agent, "run", _stub_run)

    eligible, _ = find_due_beats(
        state_with_two_eligible_scenes, production_scenario,
    )
    chosen, selection, rationale = await main._select_beat(
        eligible, reply_thread_id=None,
    )

    assert chosen is None
    assert selection == "none"
    assert rationale == "no beat fits the moment"


@pytest.mark.asyncio
async def test_select_beat_continues_active_scene_skips_director(
    production_scenario, state_with_two_eligible_scenes, monkeypatch,
):
    """The new scene-continuity short-circuit: when the player replies on
    a thread where one of the eligible beats is a scene that has already
    fired at least once, `_select_beat` returns it with selection="continue"
    WITHOUT consulting the Director.
    """
    state = state_with_two_eligible_scenes
    # Simulate one prior fire of bargain_scene by appending an Email with
    # its beat_id. (Engine bookkeeping that `_fire_beat` would have done.)
    state.emails.append(Email(
        id="email_prior",
        beat_id="aria.bargain_scene",
        thread_id="aria",
        folder="Inbox",
        sender="Aria",
        recipient="player@x",
        body="(prior bargain fire)",
        sent_at_label="just now",
    ))

    agents = _swap_globals(monkeypatch, production_scenario, state)

    call_count = {"n": 0}

    async def _boom(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError(
            "Director was called despite scene-continuity short-circuit"
        )

    monkeypatch.setattr(agents.director_agent, "run", _boom)

    eligible, _ = find_due_beats(state, production_scenario)
    eligible_ids = {b.id for b in eligible}
    assert "aria.bargain_scene" in eligible_ids
    # Confirm we're hitting a multi-eligible case (otherwise it's just
    # the "sole" path, which already works).
    assert len(eligible) >= 2

    chosen, selection, rationale = await main._select_beat(
        eligible, reply_thread_id="aria",
    )

    assert selection == "continue"
    assert chosen is not None
    assert chosen.id == "aria.bargain_scene"
    assert rationale is None
    assert call_count["n"] == 0
