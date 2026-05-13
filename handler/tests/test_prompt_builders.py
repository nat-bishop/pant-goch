"""Unit tests for the drafter / log-writer / shorthand prompt builders.

The goal: prove the model NEVER sees a digit in the narrative state block,
prove built-in cover/cycles buckets behave as authored, prove the system
prompts forbid the numeric/system-vocabulary voice, and prove the drafter's
mechanical catalog block stays untouched (it needs the keys).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from handler.agents import (
    LogEntry,
    _describe_authored_quality,
    _describe_cover,
    _describe_cycles,
    _drafter_system_prompt,
    _drafter_user_prompt,
    _forced_event_block,
    _log_entries_block,
    _log_writer_system_prompt,
    _log_writer_user_prompt,
    _narrative_state_block,
    _recently_tried_block,
    _shorthand_user_prompt,
    describe_delta,
)
from handler.mission_loader import load_mission
from handler.models import (
    Archetype,
    CastMember,
    Event,
    EventFired,
    Mission,
    Option,
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
    assert _describe_cycles(remaining, total, mission) == expected


# ───────────────────────── narrative state block ─────────────────────────


def test_narrative_state_block_emits_no_digits() -> None:
    """The core invariant: a player-facing entry can't say 'six cycles' or
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
    """The fired-event ledger reads like a log line — strip it from the
    prompt context entirely."""
    mission = load_mission(_TINY)
    state = make_initial_state(mission)
    state.fired.append("some_event")
    block = _narrative_state_block(state, mission)
    assert "some_event" not in block
    assert "fired_" not in block


def test_narrative_state_block_lynx_initial_reads_in_fiction() -> None:
    """Smoke test on the real mission: confirm the cycle-1 LYNX block contains
    the in-fiction grounding we authored and none of the meta language."""
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


# ───────────────────────── drafter system prompt voice rules ─────────────────────────


def test_drafter_system_prompt_forbids_numerics_and_game_vocab() -> None:
    """Drafter option prose was leaking phrases like 'This begins the next
    stage of the mission' — explicit rules forbid that register."""
    mission = load_mission(_LYNX)
    prompt = _drafter_system_prompt(mission)
    assert "NEVER use numbers for state" in prompt
    assert "NEVER use game/system vocabulary in option prose" in prompt
    assert "begin/initiate the next stage" in prompt


def test_drafter_system_prompt_mentions_recently_tried_rule() -> None:
    """The drafter must be told to vary tonal range when RECENTLY TRIED is
    provided — otherwise we get the same option set every cycle when state
    is static."""
    mission = load_mission(_LYNX)
    prompt = _drafter_system_prompt(mission)
    assert "RECENTLY TRIED" in prompt


def test_drafter_system_prompt_no_longer_references_sop() -> None:
    """Regression guard: the drafter prompt must not mention the SoP memo —
    that surface was dropped when the log writer became the single
    post-resolution prose surface (HANDLER_SPEC_V2 §8.1)."""
    mission = load_mission(_LYNX)
    prompt = _drafter_system_prompt(mission)
    assert "State of Play" not in prompt
    assert "SoP memo" not in prompt


def test_drafter_user_prompt_uses_narrative_state_block() -> None:
    """The drafter no longer sees raw `agent[X].cover: N / 10`. It sees the
    same narrative posture lines the log writer sees."""
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    prompt = _drafter_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        catalog=[],
        forced_event=None,
        log_entries=[],
    )
    assert "LYNX: cover intact" in prompt
    assert "the extraction papers have not yet been surfaced" in prompt
    assert "agent[LYNX].cover: 6 / 10" not in prompt


def test_drafter_user_prompt_threads_recently_tried() -> None:
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    prompt = _drafter_user_prompt(
        cycle_index=2,
        state=state,
        mission=mission,
        catalog=[],
        forced_event=None,
        log_entries=[],
        recently_tried=[(1, "DIVERSION_BY_AGENT", "LYNX")],
    )
    assert "RECENTLY TRIED" in prompt
    assert "C1: DIVERSION_BY_AGENT(LYNX)" in prompt


def test_drafter_user_prompt_carries_recent_log_entries() -> None:
    """The drafter is stateless one-shot, so it relies on the last 2 log
    entries (passed as user-side context) for cross-cycle continuity."""
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    entries = [
        LogEntry(cycle_index=1, archetype="HOLD", text="No move; the window holds."),
        LogEntry(cycle_index=2, archetype="ACTIVATE_SLEEPER_FOR_PAPERS",
                 text="REDFIELD goes live."),
    ]
    prompt = _drafter_user_prompt(
        cycle_index=3,
        state=state,
        mission=mission,
        catalog=[],
        forced_event=None,
        log_entries=entries,
    )
    assert "REDFIELD goes live" in prompt
    assert "the window holds" in prompt
    assert "RECENT CASE-FILE ENTRIES" in prompt


# ───────────────────────── log entries block ─────────────────────────


def test_log_entries_block_cycle_1_message() -> None:
    """No prior entries — the block reads as the explicit cycle-1 marker."""
    block = _log_entries_block([], limit=2)
    assert "no prior cycles" in block
    assert "cycle 1" in block


def test_log_entries_block_limits_to_last_n() -> None:
    """With more entries than the limit, only the tail is rendered."""
    entries = [
        LogEntry(cycle_index=1, archetype="A", text="First."),
        LogEntry(cycle_index=2, archetype="B", text="Second."),
        LogEntry(cycle_index=3, archetype="C", text="Third."),
    ]
    block = _log_entries_block(entries, limit=2)
    assert "First" not in block
    assert "Second" in block
    assert "Third" in block


# ───────────────────────── log writer system prompt ─────────────────────────


def test_log_writer_system_prompt_voice_and_rules() -> None:
    mission = load_mission(_LYNX)
    prompt = _log_writer_system_prompt(mission)
    # Voice register
    assert "Operations Desk" in prompt
    assert "Terse, declarative" in prompt
    # Numeric / game-vocab guard
    assert "NEVER use numbers for state" in prompt
    assert "NEVER use game/system vocabulary" in prompt


def test_log_writer_system_prompt_describes_two_beat_structure() -> None:
    mission = load_mission(_LYNX)
    prompt = _log_writer_system_prompt(mission)
    assert "Two short paragraphs" in prompt
    assert "ANGLE" in prompt
    assert "RESOLUTION" in prompt


def test_log_writer_system_prompt_frames_conversation_continuity() -> None:
    """The system prompt must instruct the model NOT to defend prior framings
    when state has moved — the key mitigation against multi-turn drift."""
    mission = load_mission(_LYNX)
    prompt = _log_writer_system_prompt(mission)
    assert "ongoing case file" in prompt
    assert "Do not defend past framings" in prompt


def test_log_writer_system_prompt_includes_authored_setting() -> None:
    mission = load_mission(_LYNX)
    prompt = _log_writer_system_prompt(mission)
    assert "SETTING:" in prompt
    assert "Bucharest" in prompt  # lynx_exfil.yaml setting


def test_log_writer_system_prompt_omits_setting_when_unauthored() -> None:
    """tiny_mission.yaml has no `setting:` field."""
    mission = load_mission(_TINY)
    prompt = _log_writer_system_prompt(mission)
    assert "SETTING:" not in prompt


def test_log_writer_system_prompt_no_first_person_or_acting_voice() -> None:
    """Voice multiplexing is gone (HANDLER_SPEC_V2 §8.2). The log-writer
    system prompt must not instruct the model to switch voices per cycle.
    (The agent dossiers may mention first-person as their authored voice
    register — that's data about the cast, not an instruction to the model.)"""
    mission = load_mission(_LYNX)
    prompt = _log_writer_system_prompt(mission)
    # The voice instruction at the top of the prompt establishes
    # "third-party" Operations Desk voice — the inverse of first-person.
    assert "third-party intelligence handler" in prompt
    # The old `ACTING is X — write FIRST-PERSON in...` instruction must be gone.
    assert "ACTING is" not in prompt
    assert "Write FIRST-PERSON" not in prompt
    assert "first-person" not in prompt[: prompt.index("MISSION CAST")].lower()


# ───────────────────────── log writer user prompt ─────────────────────────


def test_log_writer_user_prompt_includes_act_phase_tone_and_tension() -> None:
    """Log-writer user prompt surfaces all three per-cycle tonal signals."""
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
    prompt = _log_writer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={"agent[LYNX].cover": (6, 4)},
        prior_state=prior,
    )
    assert "ACT PHASE:" in prompt
    assert "TONE BAND:" in prompt
    assert "TENSION" in prompt
    assert "escalating" in prompt


def test_log_writer_user_prompt_tension_steady_when_no_prior() -> None:
    """If prior_state is None the band defaults to steady."""
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _log_writer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={},
        prior_state=None,
    )
    assert "steady" in prompt


def test_log_writer_user_prompt_includes_leans_when_provided() -> None:
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _log_writer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={},
        prior_state=None,
        leans=[
            "foreground KESTREL's grief for her daughter",
            "name LYNX's exhaustion at the window",
        ],
    )
    assert "LEAN-ON THIS CYCLE" in prompt
    assert "foreground KESTREL" in prompt
    assert "exhaustion at the window" in prompt


def test_log_writer_user_prompt_omits_lean_block_when_empty() -> None:
    """Default — no leans, no block at all (no empty heading)."""
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _log_writer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={},
        prior_state=None,
    )
    assert "LEAN-ON THIS CYCLE" not in prompt


def test_log_writer_user_prompt_carries_resolved_delta() -> None:
    """The model needs the raw delta to honour magnitudes — a -3 cover drop
    is significant; -4 is catastrophic."""
    mission = load_mission(_LYNX)
    state = make_initial_state(mission)
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _log_writer_user_prompt(
        cycle_index=1,
        state=state,
        mission=mission,
        option=option,
        delta={"agent[LYNX].cover": (6, 3)},
        prior_state=None,
    )
    assert "RESOLVED STATE DELTA" in prompt
    assert "agent[LYNX].cover" in prompt
    assert "6 -> 3" in prompt


# ───────────────────────── shorthand prompt ─────────────────────────


def test_shorthand_user_prompt_includes_cycle_and_archetype() -> None:
    option = Option(
        archetype="DIVERSION_BY_AGENT",
        agent="KESTREL",
        title="K diverts",
        prose="x",
    )
    prompt = _shorthand_user_prompt(3, option)
    assert "C3" in prompt
    assert "DIVERSION_BY_AGENT(KESTREL)" in prompt
    assert "case-file index" in prompt


def test_shorthand_user_prompt_unbound_archetype() -> None:
    """HOLD-style archetypes (no bound agent) render without parens."""
    option = Option(archetype="HOLD", agent=None, title="x", prose="y")
    prompt = _shorthand_user_prompt(5, option)
    assert "HOLD" in prompt
    assert "HOLD(" not in prompt


# ───────────────────────── recently tried ─────────────────────────


def test_recently_tried_block_renders_authorized_picks() -> None:
    block = _recently_tried_block(
        [(1, "DIVERSION_BY_AGENT", "LYNX"), (2, "DIVERSION_BY_AGENT", "KESTREL")]
    )
    assert "RECENTLY TRIED" in block
    assert "C1: DIVERSION_BY_AGENT(LYNX)" in block
    assert "C2: DIVERSION_BY_AGENT(KESTREL)" in block


def test_recently_tried_block_unbound_archetype() -> None:
    block = _recently_tried_block([(1, "HOLD", None)])
    assert "C1: HOLD" in block
    assert "HOLD(" not in block


def test_recently_tried_block_empty_when_none() -> None:
    assert _recently_tried_block(None) == ""
    assert _recently_tried_block([]) == ""


# ───────────────────────── forced event drafter block ─────────────────────────


def test_forced_event_block_carries_must_communicate_and_must_not() -> None:
    """The drafter's forced-event block surfaces intent + the prose rules so
    one option can be authored to satisfy the event."""
    event = Event(
        id="burn_redfield",
        force_into_drafter=True,
        when=EventFired(event="never_fires"),
        intent="Burn the sleeper REDFIELD to surface the extraction papers.",
        must_communicate=[
            "REDFIELD is named, and named as dead by the dispatch's close",
        ],
        must_not=["Frame REDFIELD as surviving"],
        tone="solemn, terse, present-tense",
    )
    block = _forced_event_block(event)
    assert "burn_redfield" in block
    assert "Burn the sleeper REDFIELD" in block
    assert "named as dead" in block
    assert "Frame REDFIELD as surviving" in block
    assert "solemn, terse" in block


def test_forced_event_block_empty_when_no_event() -> None:
    block = _forced_event_block(None)
    assert "no forced event" in block


# ───────────────────────── describe_delta ─────────────────────────


def test_describe_delta_filters_unchanged_descriptors() -> None:
    """If the descriptor for `before` matches the descriptor for `after`
    (a small swing within one bucket), the entry must be dropped — the
    RESULT footer would otherwise show vacuous transitions."""
    mission = load_mission(_LYNX)
    result = describe_delta({"agent[LYNX].cover": (8, 7)}, mission)
    assert "agent[LYNX].cover" not in result


def test_describe_delta_includes_meaningful_change() -> None:
    mission = load_mission(_LYNX)
    result = describe_delta({"agent[LYNX].cover": (6, 4)}, mission)
    assert result["agent[LYNX].cover"]["before"] == "cover intact"
    assert result["agent[LYNX].cover"]["after"] == "cover degraded"


def test_describe_delta_skips_transient_flags() -> None:
    """`cover_protected` and other one-cycle flags have no authored
    descriptor — they should not surface in the delta block."""
    mission = load_mission(_LYNX)
    result = describe_delta(
        {"agent[LYNX].cover_protected": (None, True)}, mission
    )
    assert result == {}


# ───────────────────────── authored descriptor helper ─────────────────────────


def test_describe_authored_quality_returns_none_when_unauthored() -> None:
    mission = load_mission(_TINY)
    assert _describe_authored_quality("mission.totally_unknown", 0, mission) is None


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


# ───────────────────────── validate_options ─────────────────────────


def test_validate_options_accepts_valid_set() -> None:
    """Happy path: 3 options with distinct catalog keys all pass."""
    from handler.agents import validate_options
    catalog = [
        ("DIVERSION_BY_AGENT", "KESTREL", {"agent": "KESTREL"}),
        ("DIVERSION_BY_AGENT", "LYNX", {"agent": "LYNX"}),
        ("HOLD", None, {}),
    ]
    options = [
        Option(
            archetype="DIVERSION_BY_AGENT",
            agent="KESTREL",
            params={"agent": "KESTREL"},
            title="K",
            prose="kp",
        ),
        Option(
            archetype="DIVERSION_BY_AGENT",
            agent="LYNX",
            params={"agent": "LYNX"},
            title="L",
            prose="lp",
        ),
        Option(archetype="HOLD", agent=None, params={}, title="H", prose="hp"),
    ]
    validate_options(options, catalog, forced_event_id=None)  # no exception


def test_validate_options_rejects_archetype_not_in_catalog() -> None:
    from handler.agents import DrafterValidationError, validate_options
    catalog = [("HOLD", None, {})]
    options = [
        Option(archetype="GHOST", agent=None, params={}, title="g", prose="g"),
        Option(archetype="HOLD", agent=None, params={}, title="h", prose="h"),
        Option(archetype="HOLD", agent=None, params={}, title="h2", prose="h2"),
    ]
    with pytest.raises(DrafterValidationError, match="VALID CATALOG"):
        validate_options(options, catalog, forced_event_id=None)


def test_validate_options_requires_forced_event_satisfier() -> None:
    """When a forced event is active, exactly one option must carry
    `satisfies_event = <event_id>`."""
    from handler.agents import DrafterValidationError, validate_options
    catalog = [
        ("HOLD", None, {}),
        ("PUSH", "LYNX", {"agent": "LYNX"}),
    ]
    options = [
        Option(archetype="HOLD", agent=None, params={}, title="h", prose="h"),
        Option(archetype="PUSH", agent="LYNX", params={"agent": "LYNX"}, title="p", prose="p"),
        Option(archetype="HOLD", agent=None, params={}, title="h2", prose="h2"),
    ]
    with pytest.raises(DrafterValidationError, match="expected exactly 1 option"):
        validate_options(options, catalog, forced_event_id="must_fire")
