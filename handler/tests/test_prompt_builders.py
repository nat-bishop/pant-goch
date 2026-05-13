"""Unit tests for the drafter / log-writer prompt builders.

Goal: prove the model never sees raw digits in the narrative state block,
prove built-in cover/days descriptors behave, prove the system prompts
forbid the numeric/system-vocabulary voice, prove the catalog block stays
intact.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from handler.agents import (
    LogEntry,
    _describe_authored_quality,
    _describe_cover,
    _describe_days,
    _forced_action_prose_block,
    _forced_action_selector_block,
    _log_entries_block,
    _log_writer_system_prompt,
    _log_writer_user_prompt,
    _narrative_state_block,
    _prose_writer_system_prompt,
    _prose_writer_user_prompt,
    _recently_tried_block,
    _selector_system_prompt,
    _selector_user_prompt,
    describe_delta,
)
from handler.mission_loader import load_mission
from handler.models import (
    Action,
    Band,
    CastMember,
    Mission,
    Option,
    Quality,
    QualityGte,
    Reaction,
    State,
    make_initial_state,
)


_FIXTURES = Path(__file__).parent / "fixtures"
_TINY = _FIXTURES / "tiny_mission.yaml"
_LYNX = Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml"


# ───────────────────────── built-in cover descriptors ─────────────────────────


@pytest.mark.parametrize(
    "cover,expected",
    [
        (0, "cover blown"),
        (1, "cover compromised"),
        (2, "cover compromised"),
        (3, "cover degraded"),
        (5, "cover degraded"),
        (6, "cover intact"),
        (8, "cover intact"),
        (9, "cover solid"),
        (10, "cover solid"),
    ],
)
def test_describe_cover_built_in_buckets(cover: int, expected: str) -> None:
    mission = load_mission(_TINY)
    assert _describe_cover(cover, mission, "ALPHA") == expected


def test_describe_cover_mission_override_wins() -> None:
    """A mission may override the universal cover descriptor via the wildcard
    key. The override beats the built-in."""
    mission = Mission(
        id="m",
        name="m",
        max_days=4,
        cast=[CastMember(id="A", codename="A", starting_cover=5)],
        actions=[Action(id="X", when=QualityGte(quality="a", value=0))],
        reactions=[
            Reaction(
                id="end",
                terminal=True,
                outcome_key="w",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"w": ""},
        qualities={
            "agent[*].cover": Quality(
                bands=[
                    Band(threshold=0.0, variants=["ALPHA DOWN"]),
                    Band(threshold=10.0, variants=["ALPHA UP"]),
                ]
            )
        },
    )
    assert _describe_cover(0, mission, "A") == "ALPHA DOWN"
    assert _describe_cover(5, mission, "A") == "ALPHA UP"


# ───────────────────────── built-in days descriptors ─────────────────────────


@pytest.mark.parametrize(
    "remaining,total,expected",
    [
        (6, 6, "the window is open"),
        (4, 6, "the window is open"),
        (3, 6, "the window is narrowing"),
        (2, 6, "the window is tightening"),
        (1, 6, "the window is closing — final approach"),
        (0, 6, "the window has closed"),
    ],
)
def test_describe_days_built_in_buckets(
    remaining: int, total: int, expected: str
) -> None:
    mission = load_mission(_TINY)
    assert _describe_days(remaining, total, mission) == expected


# ───────────────────────── narrative state block ─────────────────────────


def test_narrative_state_block_emits_no_digits() -> None:
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    block = _narrative_state_block(state, mission)
    assert re.search(r"\d", block) is None, (
        f"narrative state block leaked a digit:\n{block}"
    )


def test_narrative_state_block_includes_operation_clock_and_postures() -> None:
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    block = _narrative_state_block(state, mission)
    assert "OPERATION CLOCK:" in block
    assert "AGENT POSTURES:" in block
    assert "ALPHA:" in block
    assert "BRAVO:" in block


def test_narrative_state_block_uses_authored_descriptor() -> None:
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    state.set_quality("mission.progress", 1)
    block = _narrative_state_block(state, mission)
    assert "the operation has advanced one step" in block


def test_narrative_state_block_skips_qualities_without_descriptor() -> None:
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    state.set_quality("mission.unauthored_runtime_flag", 42)
    block = _narrative_state_block(state, mission)
    assert "unauthored_runtime_flag" not in block
    assert "42" not in block


def test_narrative_state_block_emits_protected_cover_suffix() -> None:
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    state.set_quality("agent[LYNX].cover_protected", True)
    block = _narrative_state_block(state, mission)
    assert "under protective cover this day" in block


def test_narrative_state_block_omits_fired_events() -> None:
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    state.fired.append("some_event")
    block = _narrative_state_block(state, mission)
    assert "some_event" not in block


def test_narrative_state_block_lynx_initial_reads_in_fiction() -> None:
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    block = _narrative_state_block(state, mission)
    assert "the extraction documents have not yet been surfaced" in block
    assert "LYNX: cover intact" in block
    assert "KESTREL: cover solid" in block
    assert "the window is open" in block
    assert "days_remaining" not in block
    assert "/" not in block


# ───────────────────────── selector system prompt ─────────────────────────


def test_selector_system_prompt_mentions_recently_tried_rule() -> None:
    mission = load_mission(_LYNX)
    prompt = _selector_system_prompt(mission)
    assert "RECENTLY TRIED" in prompt


def test_selector_system_prompt_describes_target_token_format() -> None:
    mission = load_mission(_LYNX)
    prompt = _selector_system_prompt(mission)
    # Token format encoded as `ACTION` or `ACTION:AGENT`.
    assert "ACTION" in prompt or "action" in prompt
    assert "AGENT" in prompt or "agent" in prompt


def test_selector_system_prompt_includes_standing_orders() -> None:
    mission = load_mission(_LYNX)
    prompt = _selector_system_prompt(mission)
    assert "STANDING ORDERS" in prompt
    assert "Vienna network is offline" in prompt


# ───────────────────────── prose writer system prompt ─────────────────────────


def test_prose_writer_system_prompt_forbids_numbers() -> None:
    mission = load_mission(_LYNX)
    prompt = _prose_writer_system_prompt(mission)
    assert "NEVER use numbers" in prompt


def test_prose_writer_system_prompt_mentions_forced_action() -> None:
    mission = load_mission(_LYNX)
    prompt = _prose_writer_system_prompt(mission)
    assert "FORCED ACTION" in prompt


# ───────────────────────── log writer system prompt ─────────────────────────


def test_log_writer_system_prompt_describes_two_paragraph_shape() -> None:
    mission = load_mission(_LYNX)
    prompt = _log_writer_system_prompt(mission)
    assert "ANGLE" in prompt or "angle" in prompt
    assert "RESOLUTION" in prompt or "resolution" in prompt


# ───────────────────────── user prompts ─────────────────────────


def test_selector_user_prompt_contains_catalog_block() -> None:
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    catalog = [("PUSH", None, {}), ("HOLD", None, {})]
    prompt = _selector_user_prompt(
        day_index=1, n=3, state=state, mission=mission,
        catalog=catalog, forced_action=None, log_entries=[],
    )
    assert "VALID CATALOG" in prompt
    assert "PUSH" in prompt


def test_forced_action_selector_block_renders_action() -> None:
    mission = load_mission(_LYNX)
    action = next(a for a in mission.actions if a.id == "kestrel_sacrifice")
    block = _forced_action_selector_block(action)
    assert "kestrel_sacrifice" in block
    assert "FORCED ACTION" in block


def test_forced_action_prose_block_renders_constraints() -> None:
    mission = load_mission(_LYNX)
    action = next(a for a in mission.actions if a.id == "kestrel_sacrifice")
    block = _forced_action_prose_block(action, carrier_index=1)
    assert "must_communicate" in block
    assert "must_not" in block
    assert "tone:" in block


def test_log_writer_user_prompt_includes_authorized_action() -> None:
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    opt = Option(action="PUSH", title="Push forward", prose="forward we go")
    prompt = _log_writer_user_prompt(
        day_index=1, state=state, mission=mission, option=opt,
        delta={"mission.progress": (0, 1)}, prior_state=state, leans=[],
    )
    assert "action: PUSH" in prompt
    assert "Push forward" in prompt


def test_log_entries_block_handles_empty() -> None:
    assert _log_entries_block([], limit=2) == "(no prior days — this is day 1.)"


def test_log_entries_block_renders_recent() -> None:
    entries = [
        LogEntry(day_index=1, action="PUSH", text="first day"),
        LogEntry(day_index=2, action="HOLD", text="second day"),
    ]
    block = _log_entries_block(entries, limit=2)
    assert "Day 1" in block
    assert "Day 2" in block
    assert "PUSH" in block
    assert "HOLD" in block


def test_recently_tried_block_empty_is_silent() -> None:
    assert _recently_tried_block(None) == ""
    assert _recently_tried_block([]) == ""


def test_recently_tried_block_renders_picks() -> None:
    picks = [(1, "PUSH", None), (2, "SACRIFICE_BY_AGENT", "ALPHA")]
    block = _recently_tried_block(picks)
    assert "D1: PUSH" in block
    assert "D2: SACRIFICE_BY_AGENT(ALPHA)" in block


# ───────────────────────── describe_delta ─────────────────────────


def test_describe_delta_renders_cover_change() -> None:
    mission = load_mission(_TINY)
    out = describe_delta({"agent[ALPHA].cover": (5, 2)}, mission)
    assert "agent[ALPHA].cover" in out
    assert out["agent[ALPHA].cover"]["before"] == "cover degraded"
    assert out["agent[ALPHA].cover"]["after"] == "cover compromised"


def test_describe_delta_skips_unchanged_descriptors() -> None:
    mission = load_mission(_TINY)
    # 7→6 stays in "intact" bucket — descriptors unchanged, entry dropped.
    out = describe_delta({"agent[ALPHA].cover": (7, 6)}, mission)
    assert out == {}


def test_describe_delta_skips_invisible_qualities() -> None:
    mission = load_mission(_LYNX)
    out = describe_delta({"mission.opposition.naming_progress": (2, 5)}, mission)
    assert "mission.opposition.naming_progress" not in out


# ───────────────────────── authored quality descriptors ─────────────────────────


def test_describe_authored_quality_returns_none_for_invisible() -> None:
    mission = load_mission(_LYNX)
    out = _describe_authored_quality("mission.opposition.naming_progress", 3, mission)
    assert out is None


def test_describe_authored_quality_returns_visible_descriptor() -> None:
    mission = load_mission(_LYNX)
    out = _describe_authored_quality("mission.extraction_stage", 0, mission)
    assert out == "the extraction documents have not yet been surfaced"
