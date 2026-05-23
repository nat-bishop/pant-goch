"""Static topology lint (the no-engine eval lane) — pure-function unit tests."""
from __future__ import annotations

from pathlib import Path

from evals.static import analyze, beat_fire_locations
from handler.world import load_world

FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"
# Fixture map is a 3-node chain: the_old_vicarage — the_post_office — the_slaughtered_lamb


def test_reachability_distances():
    world = load_world(FIXTURE_VALID)
    f = analyze(world)
    assert f["start"] == "the_old_vicarage"
    assert f["distances"] == {
        "the_old_vicarage": 0,
        "the_post_office": 1,
        "the_slaughtered_lamb": 2,
    }
    assert f["unreachable_locations"] == []


def test_chokepoint_detection():
    world = load_world(FIXTURE_VALID)
    cps = analyze(world)["chokepoints"]
    # The post office is the sole path to the pub, so it gates Tom (and his beats).
    assert "the_post_office" in cps
    assert any("the_slaughtered_lamb" in g for g in cps["the_post_office"])
    # A leaf node gates nothing.
    assert "the_slaughtered_lamb" not in cps


def test_beat_fire_locations():
    world = load_world(FIXTURE_VALID)
    # location-gated beat → its required location
    assert beat_fire_locations(world, "read_the_letter") == {"the_post_office"}
    # trust beat → where the NPC is (Tom at the pub)
    assert beat_fire_locations(world, "tom_named_a_vehicle") == {"the_slaughtered_lamb"}
    # a decision/commitment beat has no location anchor
    assert beat_fire_locations(world, "decided_to_file") == set()


def test_beat_reach_hops_and_flags():
    world = load_world(FIXTURE_VALID)
    reach = {b["beat"]: b for b in analyze(world)["beat_reach"]}
    assert reach["read_the_letter"]["hops"] == 1
    assert reach["tom_named_a_vehicle"]["hops"] == 2
    # the pub beats sit behind the post-office chokepoint
    assert "the_post_office" in reach["tom_named_a_vehicle"]["note"]
    # the no-anchor decision beat reports no hops
    assert reach["decided_to_file"]["hops"] is None
