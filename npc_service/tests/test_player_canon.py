"""Unit tests for the player-canon mechanic: predicate evaluation, canon
extraction, email resolution, and the snapshot ranking used by the chat
agent's `_player_canon_block` decorator."""
from __future__ import annotations

import pytest

from npc_service.agents import (
    CanonCandidate,
    PLAYER_CANON_RECENCY_DECAY,
    PLAYER_CANON_SURFACE_LIMIT,
    format_player_canon,
    player_canon_snapshot,
)
from npc_service.main import _slugify_name, resolve_player_email
from npc_service.models import (
    BeatFired,
    CanonCategory,
    InboxState,
    Not_,
    Or_,
    PlayerCanonEntry,
    PlayerCanonExists,
    PlayerRepliesToThreadGte,
    QualityGte,
    Reply,
    Scenario,
)
from npc_service.spine import apply_canon_extraction, evaluate_predicate


# ───────────────────────── Or_ predicate ─────────────────────────


def test_or_requires_non_empty_of():
    with pytest.raises(Exception):  # pydantic ValidationError
        Or_(of=[])


def test_or_any_true_returns_true():
    state = InboxState()
    state.qualities["x"] = 5.0
    pred = Or_(of=[
        QualityGte(quality="x", value=3.0),  # True
        PlayerRepliesToThreadGte(thread="aria", value=1),  # False (no replies)
    ])
    assert evaluate_predicate(pred, state) is True


def test_or_all_false_returns_false():
    state = InboxState()
    state.qualities["x"] = 1.0
    pred = Or_(of=[
        QualityGte(quality="x", value=3.0),
        PlayerRepliesToThreadGte(thread="aria", value=1),
    ])
    assert evaluate_predicate(pred, state) is False


def test_or_nested_under_not():
    """Calibration's pattern: Not_(Or_(canon_exists, turn_cap))."""
    state = InboxState()
    pred = Not_(of=Or_(of=[
        PlayerCanonExists(name="name"),
        PlayerRepliesToThreadGte(thread="aria", value=4),
    ]))
    # No canon, no replies — Or_ is False, Not_ flips to True.
    assert evaluate_predicate(pred, state) is True
    # Add canon name → Or_ becomes True, Not_ flips to False.
    state.player_canon["name"] = _entry("Marcus", "I'm Marcus")
    assert evaluate_predicate(pred, state) is False


# ───────────────────────── PlayerCanonExists ─────────────────────────


def test_player_canon_exists_present():
    state = InboxState()
    state.player_canon["gretchen"] = _entry(
        "Gretchen, the cousin in marketing", "my cousin Gretchen"
    )
    assert evaluate_predicate(PlayerCanonExists(name="gretchen"), state) is True


def test_player_canon_exists_absent():
    state = InboxState()
    assert evaluate_predicate(PlayerCanonExists(name="gretchen"), state) is False


# ───────────────────────── apply_canon_extraction ─────────────────────────


def test_extraction_above_threshold_promotes():
    state = InboxState()
    candidates = [
        _candidate("name", "Marcus", "I'm Marcus", 0.9, CanonCategory.NAME),
    ]
    promoted = apply_canon_extraction(state, candidates, turn_of_origin=1)
    assert promoted == ["name"]
    assert state.player_canon["name"].summary == "Marcus"
    assert state.player_canon["name"].turn_of_origin == 1


def test_extraction_below_threshold_dropped():
    state = InboxState()
    candidates = [
        _candidate("vague", "Maybe a hobby?", "kind of", 0.2, CanonCategory.CLAIM_ABOUT_SELF),
    ]
    promoted = apply_canon_extraction(state, candidates, turn_of_origin=1)
    assert promoted == []
    assert state.player_canon == {}


def test_extraction_replace_semantics_on_collision():
    """A later mention of the same key overwrites the earlier summary —
    'actually call me Robert' replaces 'Bob'."""
    state = InboxState()
    apply_canon_extraction(state, [
        _candidate("name", "Bob", "I'm Bob", 0.9, CanonCategory.NAME),
    ], turn_of_origin=1)
    apply_canon_extraction(state, [
        _candidate("name", "Robert", "actually call me Robert", 0.9, CanonCategory.NAME),
    ], turn_of_origin=4)
    assert state.player_canon["name"].summary == "Robert"
    assert state.player_canon["name"].turn_of_origin == 4


def test_extraction_custom_threshold():
    state = InboxState()
    candidates = [
        _candidate("low", "marginal", "ish", 0.3, CanonCategory.CLAIM_ABOUT_SELF),
    ]
    promoted = apply_canon_extraction(state, candidates, turn_of_origin=1, salience_threshold=0.2)
    assert promoted == ["low"]


# ───────────────────────── _slugify_name ─────────────────────────


def test_slugify_simple():
    assert _slugify_name("Marcus") == "marcus"


def test_slugify_two_words():
    assert _slugify_name("Marcus Chen") == "marcus.chen"


def test_slugify_apostrophe():
    assert _slugify_name("O'Brien") == "obrien"


def test_slugify_middle_initial():
    assert _slugify_name("Marcus L. Chen") == "marcus.l.chen"


def test_slugify_strips_trailing_dot():
    assert _slugify_name("Marcus.") == "marcus"


def test_slugify_empty_returns_empty():
    assert _slugify_name("") == ""
    assert _slugify_name("   ") == ""


# ───────────────────────── resolve_player_email ─────────────────────────


def _make_scenario(**overrides) -> Scenario:
    base = {
        "title": "T",
        "premise": "p",
        "email_domain": "halcyon.com",
        "qualities": [],
        "npcs": [{"id": "aria", "display_name": "Aria", "email_address": "aria@h"}],
        "threads": [{"id": "aria", "npc": "aria"}],
        "beats": [],
    }
    base.update(overrides)
    return Scenario.model_validate(base)


def test_resolve_uses_canonized_name_when_present():
    scenario = _make_scenario()
    state = InboxState()
    state.player_canon["name"] = _entry("Marcus Chen", "I'm Marcus Chen")
    assert resolve_player_email(scenario, state) == "marcus.chen@halcyon.com"


def test_resolve_falls_back_when_no_canon():
    scenario = _make_scenario(fallback_player_local="candidate")
    state = InboxState()
    assert resolve_player_email(scenario, state) == "candidate@halcyon.com"


def test_resolve_player_email_override_wins_over_fallback():
    scenario = _make_scenario(player_email="legacy@example.com")
    state = InboxState()
    assert resolve_player_email(scenario, state) == "legacy@example.com"


def test_resolve_canon_name_wins_over_player_email():
    scenario = _make_scenario(player_email="legacy@example.com")
    state = InboxState()
    state.player_canon["name"] = _entry("Marcus", "I'm Marcus")
    assert resolve_player_email(scenario, state) == "marcus@halcyon.com"


def test_resolve_empty_summary_falls_back():
    """A canon entry whose summary slugifies to empty (e.g. all punctuation)
    should drop back to fallback rather than emit '@halcyon.com'."""
    scenario = _make_scenario()
    state = InboxState()
    state.player_canon["name"] = _entry("---", "they wrote ---")
    assert resolve_player_email(scenario, state) == "candidate@halcyon.com"


# ───────────────────────── player_canon_snapshot ─────────────────────────


def test_snapshot_empty_returns_empty():
    assert player_canon_snapshot({}, current_turn=0) == {}


def test_snapshot_under_limit_returns_all():
    canon = {
        "name": PlayerCanonEntry(summary="Marcus", literal_text_quote="I'm Marcus", salience=0.9, category="name", turn_of_origin=1),
        "gretchen": PlayerCanonEntry(summary="cousin", literal_text_quote="my cousin Gretchen", salience=0.7, category="person", turn_of_origin=2),
    }
    snap = player_canon_snapshot(canon, current_turn=2)
    assert set(snap.keys()) == {"name", "gretchen"}


def test_snapshot_caps_at_limit():
    canon = {
        f"k{i}": PlayerCanonEntry(
            summary=f"s{i}", literal_text_quote=f"q{i}",
            salience=0.5, category="claim_about_self", turn_of_origin=1,
        )
        for i in range(PLAYER_CANON_SURFACE_LIMIT + 5)
    }
    snap = player_canon_snapshot(canon, current_turn=1)
    assert len(snap) == PLAYER_CANON_SURFACE_LIMIT


def test_snapshot_recency_beats_age_at_equal_salience():
    """Two entries with the same salience but different turn_of_origin —
    the more recent should rank above the older."""
    canon = {
        "old": PlayerCanonEntry(summary="o", literal_text_quote="o", salience=0.7, category="person", turn_of_origin=1),
        "new": PlayerCanonEntry(summary="n", literal_text_quote="n", salience=0.7, category="person", turn_of_origin=10),
    }
    snap = player_canon_snapshot(canon, current_turn=10, limit=1)
    assert "new" in snap and "old" not in snap


def test_snapshot_high_salience_old_can_beat_low_salience_new():
    """recency × salience: a high-salience old entry can outrank a barely-
    above-zero recent entry, depending on the decay constant."""
    canon = {
        "old_strong": PlayerCanonEntry(summary="o", literal_text_quote="o", salience=1.0, category="name", turn_of_origin=1),
        "new_weak": PlayerCanonEntry(summary="n", literal_text_quote="n", salience=0.05, category="claim_about_self", turn_of_origin=10),
    }
    # At current_turn=10, old age = 9, decay**9 ~= 0.85**9 ~= 0.232.
    # Old score = 0.232 * 1.0 = 0.232. New score = 1.0 * 0.05 = 0.05. Old wins.
    snap = player_canon_snapshot(canon, current_turn=10, limit=1)
    assert "old_strong" in snap


# ───────────────────────── format_player_canon ─────────────────────────


def test_format_includes_summary_and_quote():
    canon = {
        "name": _entry("Marcus", "I'm Marcus"),
    }
    lines = format_player_canon(canon)
    assert len(lines) == 1
    assert "Marcus" in lines[0]
    assert "I'm Marcus" in lines[0]
    assert lines[0].startswith("name:")


def test_format_empty_returns_empty_list():
    assert format_player_canon({}) == []


# ───────────────────────── helpers ─────────────────────────


def _entry(
    summary: str = "x",
    quote: str = "x",
    salience: float = 0.8,
    category: str = "name",
    turn: int = 1,
) -> PlayerCanonEntry:
    return PlayerCanonEntry(
        summary=summary,
        literal_text_quote=quote,
        salience=salience,
        category=category,
        turn_of_origin=turn,
    )


def _candidate(
    key: str, summary: str, quote: str, salience: float, category: CanonCategory,
) -> CanonCandidate:
    return CanonCandidate(
        key=key,
        summary=summary,
        literal_text_quote=quote,
        salience=salience,
        category=category,
    )
