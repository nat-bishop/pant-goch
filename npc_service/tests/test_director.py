"""Unit-test coverage for `_select_beat` — the Director-vs-loop seam.

CLAUDE.md flags `aria.loop` as load-bearing: it's eligible from reply 1
onward forever (no `completes_when`), so the Director agent is the ONLY
mechanism keeping it from monopolizing once a fire-once plot beat also
becomes due. The smoke fixture in `test_app.py` stubs `_select_beat`
to "first eligible" for determinism — meaning the live Director-vs-loop
selection logic is otherwise exercised only under `make integration`
(live llama). These tests close that gap by exercising `_select_beat`
directly with a stubbed Director run, asserting:

1. When ≥2 beats are due, the Director is called with the full eligible
   list reflected in the dynamic Pydantic output-type Literal.
2. When exactly 1 beat is due, the Director is NOT called (the 0/1
   short-circuits in `_select_beat` keep cost down).

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
def state_with_loop_and_bargain_due(production_scenario: Scenario) -> InboxState:
    """Engineer state where `aria.loop` AND `aria.bargain_offer` are both
    eligible on the same tick.

    aria.bargain_offer requires: `*calibration_done`,
    `player_replies_to_thread_gte: aria, 4`, AND `creature_grievance >= 1.5`.
    aria.loop only requires `*calibration_done` AND replies>=1. With 4
    replies + canon name + grievance 2.0, both gates pass.
    """
    state = InboxState.from_scenario(production_scenario)
    # Drain boot — fires aria.welcome, no-ops the rest.
    for beat in find_due_beats(state, production_scenario)[0]:
        state.fired_beat_ids.add(beat.id)
    _seed_canon_name(state)
    state.replies.extend(
        Reply(
            id=f"r{i}", thread_id="aria", beat_responded_to=None,
            recipient="Aria", body="(t)",
        )
        for i in range(4)
    )
    state.qualities["creature_grievance"] = 2.0
    return state


@pytest.fixture
def state_with_only_loop_due(production_scenario: Scenario) -> InboxState:
    """3 replies + canon name set so calibration_done is satisfied,
    no grievance: only aria.loop is eligible. With this setup:
      - aria.calibration:     Not_(*calibration_done) → blocked (canon set).
      - aria.early_emergence: gated by `Not(replies>=3)` → blocked.
      - aria.naming_demand:   needs replies>=3 (✓) AND grievance>=2.0 (fails).
      - aria.bargain_offer:   needs replies>=4 (fails).
      - aria.loop:            *calibration_done (✓) AND replies>=1 (✓) → only eligible beat.
    """
    state = InboxState.from_scenario(production_scenario)
    for beat in find_due_beats(state, production_scenario)[0]:
        state.fired_beat_ids.add(beat.id)
    _seed_canon_name(state)
    state.replies.extend(
        Reply(
            id=f"r{i}", thread_id="aria", beat_responded_to=None,
            recipient="Aria", body="(t)",
        )
        for i in range(3)
    )
    # Grievance default 0.0 — naming_demand and bargain_offer's grievance
    # gates fail.
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
async def test_director_receives_both_ids_when_loop_and_plot_beat_eligible(
    production_scenario, state_with_loop_and_bargain_due, monkeypatch,
):
    """Both `aria.loop` and `aria.bargain_offer` are due → `_select_beat`
    builds the dynamic output type with both ids in its closed Literal,
    invokes the Director, and honors its choice."""
    agents = _swap_globals(
        monkeypatch, production_scenario, state_with_loop_and_bargain_due,
    )

    captured: dict = {}

    class _FakeOutput:
        chosen_beat_id = "aria.bargain_offer"
        rationale = "advance the arc"

    class _FakeResult:
        output = _FakeOutput()

    async def _stub_run(user_prompt, *, output_type, **kwargs):
        captured["user_prompt"] = user_prompt
        captured["output_type"] = output_type
        return _FakeResult()

    monkeypatch.setattr(agents.director_agent, "run", _stub_run)

    eligible, _report = find_due_beats(
        state_with_loop_and_bargain_due, production_scenario,
    )
    eligible_ids = {b.id for b in eligible}
    # Sanity: both contested ids are actually in the due list.
    assert "aria.loop" in eligible_ids
    assert "aria.bargain_offer" in eligible_ids
    assert len(eligible) >= 2  # confirms we hit the Director branch

    chosen, selection, rationale = await main._select_beat(eligible)

    assert selection == "director"
    assert chosen is not None
    assert chosen.id == "aria.bargain_offer"
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
    production_scenario, state_with_only_loop_due, monkeypatch,
):
    """One eligible beat → `_select_beat` returns it as `selection="sole"`
    without calling the Director. Tests the cost-saving short-circuit."""
    agents = _swap_globals(
        monkeypatch, production_scenario, state_with_only_loop_due,
    )

    call_count = {"n": 0}

    async def _boom(*args, **kwargs):
        call_count["n"] += 1
        raise AssertionError("Director was called for a single-eligible tick")

    monkeypatch.setattr(agents.director_agent, "run", _boom)

    eligible, _report = find_due_beats(
        state_with_only_loop_due, production_scenario,
    )
    assert len(eligible) == 1, (
        f"fixture should make exactly one beat eligible; got "
        f"{[b.id for b in eligible]}"
    )

    chosen, selection, rationale = await main._select_beat(eligible)

    assert selection == "sole"
    assert chosen is not None
    assert chosen.id == "aria.loop"
    assert rationale is None
    assert call_count["n"] == 0


@pytest.mark.asyncio
async def test_director_none_means_skip_tick(
    production_scenario, state_with_loop_and_bargain_due, monkeypatch,
):
    """Director may return `chosen_beat_id="none"` to skip the tick — the
    contract that lets the Director punt when no eligible beat fits the
    moment."""
    agents = _swap_globals(
        monkeypatch, production_scenario, state_with_loop_and_bargain_due,
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
        state_with_loop_and_bargain_due, production_scenario,
    )
    chosen, selection, rationale = await main._select_beat(eligible)

    assert chosen is None
    assert selection == "none"
    assert rationale == "no beat fits the moment"
