"""Unit tests for the SoP/renderer prompt builders — specifically the
narrative-state translation that replaces raw `id: value` dumps.

The goal: prove the model NEVER sees a digit in the narrative state block,
prove built-in cover/cycles buckets behave as authored, prove the system
prompts now forbid the old numeric voice, and prove the drafter's mechanical
state block is untouched (it needs the numerics)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from handler.agents import (
    _describe_authored_quality,
    _describe_cover,
    _describe_cycles,
    _drafter_system_prompt,
    _drafter_user_prompt,
    _narrative_state_block,
    _renderer_system_prompt,
    _sop_system_prompt,
)
from handler.mission_loader import load_mission
from handler.models import (
    CastMember,
    Archetype,
    Event,
    Mission,
    Quality,
    QualityGte,
    State,
    make_initial_state,
)


_FIXTURES = Path(__file__).parent / "fixtures"
_TINY = _FIXTURES / "tiny_mission.yaml"
_LYNX = Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml"


# ───────────────────────── built-in cover buckets ─────────────────────────


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
        max_cycles=4,
        cast=[CastMember(id="A", codename="A", starting_cover=5)],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        events=[
            Event(
                id="end",
                terminal=True,
                outcome_key="w",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"w": ""},
        qualities={
            "agent[*].cover": Quality(
                buckets=[(0.0, "ALPHA DOWN"), (10.0, "ALPHA UP")]
            )
        },
    )
    # 0 hits the first bucket; 5 falls into the second.
    assert _describe_cover(0, mission, "A") == "ALPHA DOWN"
    assert _describe_cover(5, mission, "A") == "ALPHA UP"


# ───────────────────────── built-in cycles buckets ─────────────────────────


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
def test_describe_cycles_built_in_buckets(
    remaining: int, total: int, expected: str
) -> None:
    mission = load_mission(_TINY)
    # _describe_cycles uses the ratio remaining/total, not the mission's
    # max_cycles, so passing `total` is enough.
    assert _describe_cycles(remaining, total, mission) == expected


# ───────────────────────── narrative state block ─────────────────────────


def test_narrative_state_block_emits_no_digits() -> None:
    """The core invariant: a player-facing memo can't say 'six cycles' or
    'cover six' if the prompt never carries a digit."""
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
    """`mission.progress=1` is authored as 'the operation has advanced one step'."""
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    state.set_quality("mission.progress", 1)
    block = _narrative_state_block(state, mission)
    assert "the operation has advanced one step" in block


def test_narrative_state_block_skips_qualities_without_descriptor() -> None:
    """If state acquires a `mission.*` quality at runtime that wasn't authored
    with a descriptor (e.g. an effect added it), it MUST NOT leak as
    `mission.foo: 0` — the block silently omits it."""
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
    assert "under protective cover this cycle" in block


def test_narrative_state_block_omits_fired_events() -> None:
    """The fired-event ledger reads like a log line — strip it from the SoP
    context entirely (the recent-dispatch block carries continuity)."""
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    state.fired.append("some_event")
    block = _narrative_state_block(state, mission)
    assert "some_event" not in block
    assert "fired_" not in block


def test_narrative_state_block_lynx_initial_reads_in_fiction() -> None:
    """Smoke test on the real mission: confirm the cycle-1 LYNX block contains
    the in-fiction grounding we authored and none of the meta language the
    user reported."""
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    block = _narrative_state_block(state, mission)
    # The keystone descriptor for extraction_stage=0:
    assert "the extraction papers have not yet been surfaced" in block
    # Cover descriptors for both agents:
    assert "LYNX: cover intact" in block
    assert "KESTREL: cover solid" in block
    # Operation clock:
    assert "the window is open" in block
    # And the meta language MUST NOT appear:
    assert "cycles_remaining" not in block
    assert "/" not in block
    assert "out of" not in block


# ───────────────────────── SoP system prompt voice rules ─────────────────────────


def test_sop_system_prompt_forbids_numerics_rule_present() -> None:
    mission = load_mission(_LYNX)
    prompt = _sop_system_prompt(mission)
    assert "NEVER use numbers for state" in prompt
    assert "NEVER use game/system vocabulary" in prompt


def test_sop_system_prompt_drops_old_numeric_permission() -> None:
    """Regression guard against re-introducing the line that explicitly
    licensed dashboard voice."""
    mission = load_mission(_LYNX)
    prompt = _sop_system_prompt(mission)
    assert "May reference state numerically" not in prompt


def test_sop_system_prompt_includes_authored_setting() -> None:
    mission = load_mission(_LYNX)
    prompt = _sop_system_prompt(mission)
    assert "SETTING:" in prompt
    assert "Bucharest" in prompt  # from lynx_exfil.yaml setting


def test_sop_system_prompt_omits_setting_when_unauthored() -> None:
    """tiny_mission.yaml has no `setting:` field — the SETTING section
    should not appear in the prompt."""
    mission = load_mission(_TINY)
    prompt = _sop_system_prompt(mission)
    assert "SETTING:" not in prompt


def test_sop_system_prompt_includes_agent_situation_when_authored() -> None:
    mission = load_mission(_LYNX)
    prompt = _sop_system_prompt(mission)
    assert "Situation:" in prompt
    assert "Forty-eight hours since the" in prompt  # from LYNX situation


# ───────────────────────── renderer system prompt voice rules ─────────────────────────


def test_renderer_system_prompt_forbids_numerics_rule_present() -> None:
    mission = load_mission(_LYNX)
    prompt = _renderer_system_prompt(mission)
    assert "NEVER use numbers for state" in prompt
    assert "NEVER use game/system vocabulary" in prompt


# ───────────────────────── drafter voice rules ─────────────────────────


def test_drafter_system_prompt_forbids_numerics_and_game_vocab() -> None:
    """Drafter option prose was leaking phrases like 'This begins the next
    stage of the mission' — explicit rules now forbid that register."""
    mission = load_mission(_LYNX)
    prompt = _drafter_system_prompt(mission)
    assert "NEVER use numbers for state" in prompt
    assert "NEVER use game/system vocabulary in option prose" in prompt
    # The specific phrase the user flagged is called out by example so the
    # model sees it as forbidden:
    assert "begin/initiate the next stage" in prompt


def test_drafter_user_prompt_uses_narrative_state_block() -> None:
    """The drafter no longer sees raw `agent[X].cover: N / 10`. It sees the
    same narrative posture lines SoP/renderer see."""
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    prompt = _drafter_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        sop_text="(test SoP body)",
        catalog=[],
        forced_event=None,
        dispatches=[],
    )
    # narrative descriptors present:
    assert "LYNX: cover intact" in prompt
    assert "the extraction papers have not yet been surfaced" in prompt
    # raw quality dump absent:
    assert "agent[LYNX].cover: 6 / 10" not in prompt
    assert "cycles_remaining: 6 / 6" not in prompt


# ───────────────────────── authored descriptor helper ─────────────────────────


def test_describe_authored_quality_returns_none_when_unauthored() -> None:
    mission = load_mission(_TINY)
    assert _describe_authored_quality("mission.totally_unknown", 0, mission) is None


# ───────────────────────── prompt enrichment signals ─────────────────────────


def _mission_with_signals() -> Mission:
    """Mission carrying every signal the enrichment plumbing reads."""
    from handler.models import DramaticQuestion, QualityLte
    return Mission(
        id="m",
        name="m",
        max_cycles=6,
        cast=[
            CastMember(id="A", codename="A", starting_cover=8),
            CastMember(id="B", codename="B", starting_cover=8),
        ],
        archetypes=[
            Archetype(id="X", when=QualityGte(quality="a", value=0)),
            # Locked for the 'cycle 1' state — no agent has cover 0 etc.
            Archetype(id="ENDGAME", when=QualityGte(quality="mission.stage", value=2)),
        ],
        events=[
            Event(id="end", terminal=True, outcome_key="w", when=QualityGte(quality="x", value=0))
        ],
        outcomes={"w": ""},
        qualities={"mission.stage": Quality(initial=0, values={"0": "untested", "1": "live", "2": "done"})},
        standing_orders=["LYNX must not be photographed again."],
        dramatic_questions=[
            DramaticQuestion(
                when=QualityGte(quality="mission.cycles_remaining", value=1),
                question="What now?",
            ),
        ],
    )


def test_sop_system_prompt_includes_standing_orders_block() -> None:
    mission = _mission_with_signals()
    prompt = _sop_system_prompt(mission)
    assert "STANDING ORDERS" in prompt
    assert "LYNX must not be photographed again" in prompt


def test_drafter_system_prompt_includes_standing_orders_block() -> None:
    mission = _mission_with_signals()
    prompt = _drafter_system_prompt(mission)
    assert "STANDING ORDERS" in prompt


def test_renderer_system_prompt_includes_end_on_a_beat_rule() -> None:
    mission = _mission_with_signals()
    prompt = _renderer_system_prompt(mission)
    assert "End on a single short observational sentence" in prompt
    assert "TONE BAND" in prompt


def test_renderer_system_prompt_includes_standing_orders_block() -> None:
    mission = _mission_with_signals()
    prompt = _renderer_system_prompt(mission)
    assert "STANDING ORDERS" in prompt


def test_system_prompts_omit_standing_orders_block_when_empty() -> None:
    """Authors with no standing orders shouldn't see the heading at all."""
    mission = load_mission(_TINY)
    assert "STANDING ORDERS" not in _sop_system_prompt(mission)
    assert "STANDING ORDERS" not in _drafter_system_prompt(mission)
    assert "STANDING ORDERS" not in _renderer_system_prompt(mission)


def test_sop_user_prompt_includes_act_phase_and_dramatic_question() -> None:
    mission = _mission_with_signals()
    state = make_initial_state(mission)
    from handler.agents import _sop_user_prompt
    prompt = _sop_user_prompt(cycle_index=1, state=state, mission=mission, dispatches=[])
    assert "ACT PHASE: setup" in prompt
    assert "DRAMATIC QUESTION" in prompt
    assert "What now?" in prompt


def test_sop_user_prompt_omits_dramatic_question_when_no_match() -> None:
    """When no dramatic question matches (or none are authored), the DQ block
    is omitted entirely. The act phase still appears unconditionally."""
    mission = load_mission(_TINY)  # tiny mission authors no dramatic_questions
    state = make_initial_state(mission)
    from handler.agents import _sop_user_prompt
    prompt = _sop_user_prompt(cycle_index=1, state=state, mission=mission, dispatches=[])
    assert "DRAMATIC QUESTION" not in prompt
    assert "ACT PHASE:" in prompt


def test_drafter_user_prompt_includes_locked_block() -> None:
    """ENDGAME requires stage>=2 which isn't met — must appear in LOCKED block."""
    mission = _mission_with_signals()
    state = make_initial_state(mission)
    prompt = _drafter_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        sop_text="(test sop)",
        catalog=[],
        forced_event=None,
        dispatches=[],
    )
    assert "LOCKED THIS CYCLE" in prompt
    assert "ENDGAME" in prompt
    assert "ACT PHASE:" in prompt


def test_drafter_user_prompt_omits_locked_block_when_empty() -> None:
    """tiny mission has only 3 archetypes, all available at cycle 1 — no LOCKED block."""
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    prompt = _drafter_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        sop_text="(test sop)",
        catalog=[],
        forced_event=None,
        dispatches=[],
    )
    assert "LOCKED THIS CYCLE" not in prompt


def test_renderer_user_prompt_includes_act_phase_tone_and_tension() -> None:
    """Renderer prompt surfaces all three per-cycle tonal signals."""
    from handler.agents import _renderer_user_prompt
    from handler.models import Option
    mission = load_mission(_LYNX)
    prior = make_initial_state(mission)
    state = make_initial_state(mission)
    state.set_quality("agent[LYNX].cover", 4)  # -2 swing → escalating
    option = Option(
        archetype="HOLD",
        agent=None,
        title="Hold position",
        prose="Take no action this cycle.",
    )
    prompt = _renderer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={"agent[LYNX].cover": (6, 4)},
        acting="operations",
        dispatches=[],
        prior_state=prior,
    )
    assert "ACT PHASE:" in prompt
    assert "TONE BAND:" in prompt
    assert "TENSION" in prompt
    # 2-point swing → escalating (per spine logic).
    assert "escalating" in prompt


def test_renderer_user_prompt_tension_steady_when_no_prior() -> None:
    """If prior_state is None the band defaults to steady."""
    from handler.agents import _renderer_user_prompt
    from handler.models import Option
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _renderer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={},
        acting="operations",
        dispatches=[],
        prior_state=None,
    )
    assert "steady" in prompt


# ───────────────────────── cast block + briefing + leans ─────────────────────────


def _basic_mission_kwargs() -> dict:
    return dict(
        id="t",
        name="t",
        max_cycles=2,
        cast=[CastMember(id="A", codename="A", starting_cover=5)],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        events=[
            Event(
                id="end",
                terminal=True,
                outcome_key="w",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"w": ""},
    )


def _mission_with_handler() -> Mission:
    return Mission(
        id="m",
        name="m",
        codename="TEST-CODE",
        max_cycles=4,
        cast=[
            CastMember(
                id="ALPHA",
                codename="Alpha",
                role="agent",
                voice_register="agent register",
                starting_cover=6,
            ),
            CastMember(
                id="RAVEN",
                codename="Raven",
                role="handler",
                voice_register="handler register",
                voice_details="codename only; trusted by Control",
            ),
            CastMember(
                id="CONTROL",
                codename="Control",
                role="boss",
                voice_register="boss register",
            ),
            CastMember(
                id="VOLKOV",
                codename="Volkov",
                role="opposition",
                voice_register="opposition register",
            ),
        ],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        events=[
            Event(
                id="end",
                terminal=True,
                outcome_key="w",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"w": ""},
    )


def test_cast_block_renders_role_label_for_non_agents() -> None:
    mission = _mission_with_handler()
    prompt = _renderer_system_prompt(mission)
    assert "RAVEN (handler" in prompt
    assert "CONTROL (boss" in prompt
    assert "VOLKOV (opposition" in prompt


def test_cast_block_omits_role_label_for_agents() -> None:
    """Agent entries keep the legacy `(file N, codename 'X')` shape so the
    KV-cache prefix is stable for agent-only missions."""
    mission = _mission_with_handler()
    prompt = _renderer_system_prompt(mission)
    # Agent role does NOT prefix "(agent;" before the file/codename parens.
    assert "ALPHA (agent" not in prompt
    # But the agent IS in the cast block:
    assert "ALPHA" in prompt


def test_cast_block_renders_voice_details_when_present() -> None:
    mission = _mission_with_handler()
    prompt = _renderer_system_prompt(mission)
    assert "codename only; trusted by Control" in prompt


def test_sop_user_prompt_prepends_briefing_on_cycle_1() -> None:
    from handler.agents import _sop_user_prompt
    mission = Mission(
        **_basic_mission_kwargs(),
        opening_briefing="OPS: clean exfil only. Burn nothing.",
    )
    state = make_initial_state(mission)
    prompt = _sop_user_prompt(cycle_index=1, state=state, mission=mission, dispatches=[])
    assert "OPENING BRIEFING" in prompt
    assert "clean exfil only" in prompt


def test_sop_user_prompt_omits_briefing_after_cycle_1() -> None:
    from handler.agents import _sop_user_prompt
    mission = Mission(
        **_basic_mission_kwargs(),
        opening_briefing="OPS: clean exfil only. Burn nothing.",
    )
    state = make_initial_state(mission)
    prompt = _sop_user_prompt(cycle_index=2, state=state, mission=mission, dispatches=[])
    assert "OPENING BRIEFING" not in prompt
    assert "clean exfil only" not in prompt


def test_sop_user_prompt_omits_briefing_when_empty() -> None:
    """Cycle 1 with no authored opening_briefing must not produce the
    block (no empty heading)."""
    from handler.agents import _sop_user_prompt
    mission = Mission(**_basic_mission_kwargs())  # no opening_briefing
    state = make_initial_state(mission)
    prompt = _sop_user_prompt(cycle_index=1, state=state, mission=mission, dispatches=[])
    assert "OPENING BRIEFING" not in prompt


def test_renderer_user_prompt_includes_leans_when_provided() -> None:
    from handler.agents import _renderer_user_prompt
    from handler.models import Option
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _renderer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={},
        acting="operations",
        dispatches=[],
        prior_state=None,
        leans=[
            "foreground KESTREL's grief for her daughter",
            "name LYNX's exhaustion at the window",
        ],
    )
    assert "LEAN-ON THIS CYCLE" in prompt
    assert "foreground KESTREL" in prompt
    assert "exhaustion at the window" in prompt


def test_renderer_user_prompt_omits_lean_block_when_empty() -> None:
    """Default — no leans, no block at all (no empty heading)."""
    from handler.agents import _renderer_user_prompt
    from handler.models import Option
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _renderer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={},
        acting="operations",
        dispatches=[],
        prior_state=None,
    )
    assert "LEAN-ON THIS CYCLE" not in prompt


def test_describe_authored_quality_supports_label_prefix() -> None:
    mission = Mission(
        id="m",
        name="m",
        max_cycles=4,
        cast=[CastMember(id="A", codename="A", starting_cover=5)],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        events=[
            Event(
                id="end",
                terminal=True,
                outcome_key="w",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"w": ""},
        qualities={
            "mission.stage": Quality(
                initial=0,
                label="extraction",
                values={"0": "papers not yet surfaced"},
            )
        },
    )
    assert (
        _describe_authored_quality("mission.stage", 0, mission)
        == "extraction: papers not yet surfaced"
    )
