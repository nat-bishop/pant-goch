"""Corpus distributional metrics (Tier 1) — pure-function unit tests."""
from __future__ import annotations

from pathlib import Path

from evals.metrics import (
    CorpusMetrics,
    _ending_entropy_normalized,
    _filler_turn_ratio,
    _ladder_compression,
    _longest_common_beat_prefix,
    _mean_pairwise_beat_jaccard_distance,
    _persona_outcome_separation,
    _post_exhaustion_turn_ratio,
    corpus_metrics,
)
from evals.session import SessionArtifact
from handler.world import load_world

FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def _art(**overrides) -> SessionArtifact:
    base = dict(
        persona="p", session_id="s", model_engine="e", model_player="m",
        started_at="t", duration_s=1.0, scene_count=2, truncated=False,
        ending_id="published_story", unlocked_beats=[], case_file_entries=[],
        npc_memory={}, transcript=[],
    )
    base.update(overrides)
    return SessionArtifact(**base)


def _turn(scene_id, location, applied) -> dict:
    return {
        "kind": "turn", "scene_id": scene_id, "location": location,
        "prose": "", "options": [], "pick_index": 0, "pick_text": "",
        "applied_beats": applied, "fireable": [],
    }


def test_ending_entropy_single_vs_even():
    same = [_art(ending_id="published_story") for _ in range(4)]
    assert _ending_entropy_normalized(same, endings_total=4) == 0.0
    spread = [_art(ending_id=e) for e in ("a", "b", "c", "d")]
    # four distinct endings out of four authored → maximal variety → 1.0
    assert _ending_entropy_normalized(spread, endings_total=4) == 1.0
    assert _ending_entropy_normalized([], endings_total=1) == 0.0


def test_beat_jaccard_distance_extremes():
    disjoint = [_art(unlocked_beats=["a"]), _art(unlocked_beats=["b"])]
    assert _mean_pairwise_beat_jaccard_distance(disjoint) == 1.0
    identical = [_art(unlocked_beats=["a", "b"]), _art(unlocked_beats=["a", "b"])]
    assert _mean_pairwise_beat_jaccard_distance(identical) == 0.0
    assert _mean_pairwise_beat_jaccard_distance([_art()]) == 0.0


def test_filler_turn_ratio():
    all_filler = _art(transcript=[_turn(1, "x", []), _turn(1, "x", [])])
    assert _filler_turn_ratio([all_filler]) == 1.0
    none_filler = _art(transcript=[_turn(1, "x", ["beat_a"]), _turn(1, "x", ["beat_b"])])
    assert _filler_turn_ratio([none_filler]) == 0.0
    half = _art(transcript=[_turn(1, "x", []), _turn(1, "x", ["beat_a"])])
    assert _filler_turn_ratio([half]) == 0.5
    # legacy artifact (no applied_beats key) contributes nothing
    legacy = _art(transcript=[{"kind": "turn", "scene_id": 1, "location": "x"}])
    assert _filler_turn_ratio([legacy]) == 0.0


def test_post_exhaustion_turn_ratio():
    # last new beat at turn 1 (index 0); 3 dead turns trail it → 3/4 = 0.75 (the grind)
    grind = _art(transcript=[
        _turn(1, "x", ["beat_a"]), _turn(1, "x", []), _turn(1, "x", []), _turn(1, "x", []),
    ])
    assert _post_exhaustion_turn_ratio([grind]) == 0.75
    # tight session: each turn advances, last turn fires a beat → no dead tail
    tight = _art(transcript=[_turn(1, "x", ["beat_a"]), _turn(2, "y", ["beat_b"])])
    assert _post_exhaustion_turn_ratio([tight]) == 0.0
    # a re-fired (not new) beat does not count as progress → still a dead tail
    refire = _art(transcript=[_turn(1, "x", ["beat_a"]), _turn(1, "x", ["beat_a"])])
    assert _post_exhaustion_turn_ratio([refire]) == 0.5
    # never advanced at all → whole session is post-exhaustion
    dead = _art(transcript=[_turn(1, "x", []), _turn(1, "x", [])])
    assert _post_exhaustion_turn_ratio([dead]) == 1.0
    # legacy artifact (no applied_beats) contributes nothing
    legacy = _art(transcript=[{"kind": "turn", "scene_id": 1, "location": "x"}])
    assert _post_exhaustion_turn_ratio([legacy]) == 0.0


def test_longest_common_beat_prefix_majority():
    # 3 of 5 open with [a, b]; the other 2 diverge immediately.
    shared = [
        _art(transcript=[_turn(1, "x", ["a"]), _turn(1, "x", ["b"]), _turn(2, "y", ["c"])])
        for _ in range(3)
    ]
    other = [_art(transcript=[_turn(1, "x", ["z"])]) for _ in range(2)]
    assert _longest_common_beat_prefix(shared + other) == ["a", "b", "c"]


def test_persona_outcome_separation():
    converged = [_art(persona=p, ending_id="same") for p in ("p1", "p2")]
    assert _persona_outcome_separation(converged) == 0.0
    diverged = [_art(persona="p1", ending_id="x"), _art(persona="p2", ending_id="y")]
    assert _persona_outcome_separation(diverged) > 0.0
    assert _persona_outcome_separation([_art()]) == 0.0  # single persona


def test_ladder_compression():
    # Two Tom rungs fired in one scene → compression 2.0 for that ladder.
    compressed = _art(transcript=[
        _turn(1, "pub", ["tom_speaks_of_megan_as_a_person", "tom_alludes_to_being_on_the_road"]),
    ])
    out = _ladder_compression([compressed])
    assert out["tom"] == 2.0
    assert out["dylan"] == 0.0  # never touched


def test_corpus_metrics_smoke():
    world = load_world(FIXTURE_VALID)
    arts = [
        _art(unlocked_beats=["tom_named_a_vehicle"], ending_id="published_story",
             transcript=[_turn(1, "the_old_vicarage", ["tom_named_a_vehicle"])]),
        _art(unlocked_beats=[], ending_id=None, truncated=True, transcript=[]),
    ]
    m = corpus_metrics(arts, world)
    assert isinstance(m, CorpusMetrics)
    assert m.sessions == 2
    assert m.beats_total == len(world.beats)
    assert "published_story" in m.endings_reached
    assert "tom_named_a_vehicle" not in m.beats_never_fired
    assert 0.0 <= m.location_coverage <= 1.0
    # serializes to a flat dict for the report diff
    assert {"ending_entropy_normalized", "filler_turn_ratio",
            "post_exhaustion_turn_ratio"}.issubset(m.to_dict())
