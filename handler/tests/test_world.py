"""M1 — World loader tests.

Happy path + one test per SPEC §2.6 invariant. Each invariant test starts
from `fixtures/world_valid/`, copies it to tmp_path, and mutates the copy
to violate the invariant. The loader must raise WorldLoadError with a
message that names the violation.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from handler.world import World, WorldLoadError, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def _copy_valid(tmp_path: Path) -> Path:
    target = tmp_path / "world"
    shutil.copytree(FIXTURE_VALID, target)
    return target


def _write_yaml(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ───────────────────────── happy path ─────────────────────────


def test_load_world_succeeds_on_valid_fixture():
    world = load_world(FIXTURE_VALID)
    assert isinstance(world, World)
    assert set(world.locations) == {
        "the_slaughtered_lamb", "the_post_office", "the_old_vicarage"
    }


def test_load_world_tolerates_missing_few_shots_yaml():
    """few_shots.yaml is optional — older fixtures load with []."""
    world = load_world(FIXTURE_VALID)
    assert world.few_shots == []


def test_load_world_loads_few_shots_when_present(tmp_path):
    """When few_shots.yaml exists, entries parse into FewShot records and
    are accessible via world.few_shots."""
    target = _copy_valid(tmp_path)
    _write_yaml(target / "few_shots.yaml", [
        {
            "setup": "A pub. A man at the bar.",
            "player": '"Mind if I sit?"',
            "narrator": "He turned the glass once. Once more. Said nothing for a while.",
        },
    ])
    world = load_world(target)
    assert len(world.few_shots) == 1
    assert world.few_shots[0].player == '"Mind if I sit?"'
    assert "turned the glass" in world.few_shots[0].narrator


def test_load_world_rejects_few_shots_yaml_non_list(tmp_path):
    target = _copy_valid(tmp_path)
    _write_yaml(target / "few_shots.yaml", {"not": "a list"})
    with pytest.raises(WorldLoadError, match="expected a list"):
        load_world(target)


def test_npc_voice_sample_defaults_to_empty_for_legacy_fixture():
    """Fixture cast.yaml doesn't author voice_sample → it loads as []
    (Field default_factory). Production cast.yaml authors voice samples
    per NPC."""
    world = load_world(FIXTURE_VALID)
    for npc in world.npcs.values():
        assert npc.voice_sample == []


def test_load_world_tolerates_missing_map_yaml():
    """map.yaml is optional; fixtures without one load with empty
    map_coords and the frontend falls back to the text walk panel."""
    world = load_world(FIXTURE_VALID)
    assert world.map_coords == {}


def test_load_world_loads_map_coords_when_present(tmp_path):
    target = _copy_valid(tmp_path)
    _write_yaml(target / "map.yaml", {
        "the_old_vicarage": {"x": 0.87, "y": 0.83},
        "the_post_office": {"x": 0.60, "y": 0.83},
        "the_slaughtered_lamb": {"x": 0.66, "y": 0.60},
    })
    world = load_world(target)
    assert world.map_coords["the_old_vicarage"] == (0.87, 0.83)
    assert world.map_coords["the_post_office"] == (0.60, 0.83)


def test_load_world_rejects_map_yaml_unknown_location(tmp_path):
    target = _copy_valid(tmp_path)
    _write_yaml(target / "map.yaml", {"ghost_location": {"x": 0.5, "y": 0.5}})
    with pytest.raises(WorldLoadError, match="not a defined location"):
        load_world(target)


def test_load_world_rejects_map_yaml_out_of_range(tmp_path):
    target = _copy_valid(tmp_path)
    _write_yaml(target / "map.yaml", {
        "the_old_vicarage": {"x": 1.5, "y": 0.83},
    })
    with pytest.raises(WorldLoadError, match="out of"):
        load_world(target)


def test_npc_voice_sample_loaded_when_authored(tmp_path):
    target = _copy_valid(tmp_path)
    cast = yaml.safe_load((target / "cast.yaml").read_text(encoding="utf-8"))
    cast[0]["voice_sample"] = ["Aye.", "Not my business, was it."]
    _write_yaml(target / "cast.yaml", cast)
    world = load_world(target)
    npc = world.npcs[cast[0]["id"]]
    assert npc.voice_sample == ["Aye.", "Not my business, was it."]
    assert set(world.npcs) == {"tom", "maud"}
    assert set(world.beats) == {
        "tom_named_a_vehicle", "tom_admits_more", "tom_closes_off"
    }
    assert set(world.endings) == {"published_story"}
    assert world.brief.player_starts_at == "the_old_vicarage"
    # npcs_at returns the right occupants
    assert [n.id for n in world.npcs_at("the_slaughtered_lamb")] == ["tom"]
    assert [n.id for n in world.npcs_at("the_old_vicarage")] == []


def test_npcs_at_state_aware_uses_npc_locations_override():
    """When `state` is passed, npcs_at reads `state.npc_locations` as the
    override (populator's output). NPCs missing from the dict fall back
    to default_location. None / no state == default-only view."""
    from handler.state import make_initial_state
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    # Defaults: tom at pub, maud at post office.
    assert [n.id for n in world.npcs_at("the_slaughtered_lamb", state=state)] == ["tom"]
    # Override: move tom to the_post_office; npcs_at follows.
    state.npc_locations["tom"] = "the_post_office"
    assert [n.id for n in world.npcs_at("the_slaughtered_lamb", state=state)] == []
    assert {n.id for n in world.npcs_at("the_post_office", state=state)} == {"tom", "maud"}
    # No state → defaults (overrides ignored).
    assert [n.id for n in world.npcs_at("the_slaughtered_lamb")] == ["tom"]


# ───────────────────────── invariant 1: location id uniqueness ─────────────────────────


def test_invariant_1_location_id_uniqueness(tmp_path):
    target = _copy_valid(tmp_path)
    # Duplicate the_slaughtered_lamb's id under a different filename.
    _write_yaml(target / "locations" / "extra.yaml", {
        "id": "the_slaughtered_lamb",  # collision
        "name": "Some Other Place",
        "adjacent_to": [],
        "description": "x",
        "notes": [],
    })
    with pytest.raises(WorldLoadError, match="duplicate location id"):
        load_world(target)


# ───────────────────────── invariant 2: symmetric adjacency ─────────────────────────


def test_invariant_2_asymmetric_adjacency(tmp_path):
    target = _copy_valid(tmp_path)
    # vicarage already has [the_post_office]; remove from post_office's side.
    post_office_path = target / "locations" / "the_post_office.yaml"
    data = yaml.safe_load(post_office_path.read_text(encoding="utf-8"))
    data["adjacent_to"] = ["the_slaughtered_lamb"]  # drop the_old_vicarage
    _write_yaml(post_office_path, data)
    with pytest.raises(WorldLoadError, match="asymmetric adjacency"):
        load_world(target)


def test_invariant_2_adjacency_to_undefined_location(tmp_path):
    target = _copy_valid(tmp_path)
    pub_path = target / "locations" / "the_slaughtered_lamb.yaml"
    data = yaml.safe_load(pub_path.read_text(encoding="utf-8"))
    data["adjacent_to"].append("knighton")  # not defined
    _write_yaml(pub_path, data)
    with pytest.raises(WorldLoadError, match="undefined location"):
        load_world(target)


# ───────────────────────── invariant 3: NPC id uniqueness ─────────────────────────


def test_invariant_3_npc_id_uniqueness(tmp_path):
    target = _copy_valid(tmp_path)
    cast_path = target / "cast.yaml"
    cast = yaml.safe_load(cast_path.read_text(encoding="utf-8"))
    cast.append({
        "id": "tom",  # collision
        "name": "Other Tom",
        "default_location": "the_post_office",
        "themes": [],
        "knowledge": "x",
        "attitude": "y",
    })
    _write_yaml(cast_path, cast)
    with pytest.raises(WorldLoadError, match="duplicate npc id"):
        load_world(target)


# ───────────────────────── invariant 4: NPC default_location ─────────────────────────


def test_invariant_4_npc_default_location_undefined(tmp_path):
    target = _copy_valid(tmp_path)
    cast_path = target / "cast.yaml"
    cast = yaml.safe_load(cast_path.read_text(encoding="utf-8"))
    cast[0]["default_location"] = "atlantis"  # nope
    _write_yaml(cast_path, cast)
    with pytest.raises(WorldLoadError, match="not a defined location"):
        load_world(target)


# ───────────────────────── invariant 5: beat id uniqueness ─────────────────────────


def test_invariant_5_beat_id_uniqueness(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    truth["beats"].append({
        "id": "tom_named_a_vehicle",  # collision
        "description": "x",
        "notes_to_llm": "y",
    })
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="duplicate beat id"):
        load_world(target)


# ───────────────────────── invariant 6: ending id uniqueness + namespace ─────────────────────────


def test_invariant_6_ending_id_uniqueness(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    truth["endings"].append({
        "id": "published_story",  # collision
        "requires_beats": [],
        "notes_to_llm": "x",
        "epilogue": "y",
    })
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="duplicate ending id"):
        load_world(target)


def test_invariant_6_beat_ending_namespace_collision(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    # Add an ending whose id matches an existing beat.
    truth["endings"].append({
        "id": "tom_named_a_vehicle",  # collides with a beat
        "requires_beats": [],
        "notes_to_llm": "x",
        "epilogue": "y",
    })
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="beat/ending id collision"):
        load_world(target)


# ───────────────────────── invariant 7: brief player_starts_at ─────────────────────────


def test_invariant_7_brief_player_starts_at_undefined(tmp_path):
    target = _copy_valid(tmp_path)
    brief_path = target / "brief.yaml"
    brief = yaml.safe_load(brief_path.read_text(encoding="utf-8"))
    brief["player_starts_at"] = "nowhere_in_particular"
    _write_yaml(brief_path, brief)
    with pytest.raises(WorldLoadError, match="player_starts_at"):
        load_world(target)


# ───────────────────────── invariant 8: beat refs (requires/blocked_by + no self) ─────────────────────────


def test_invariant_8_requires_undefined_beat(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    # tom_admits_more requires_beats: append a missing id
    for beat in truth["beats"]:
        if beat["id"] == "tom_admits_more":
            beat["requires_beats"].append("ghost_beat")
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="requires_beats references undefined"):
        load_world(target)


def test_invariant_8_blocked_by_undefined_beat(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    for beat in truth["beats"]:
        if beat["id"] == "tom_named_a_vehicle":
            beat["blocked_by_beats"].append("phantom_beat")
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="blocked_by_beats references undefined"):
        load_world(target)


def test_invariant_8_self_reference_in_requires(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    for beat in truth["beats"]:
        if beat["id"] == "tom_named_a_vehicle":
            beat["requires_beats"] = ["tom_named_a_vehicle"]
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="contains itself"):
        load_world(target)


def test_invariant_8_self_reference_in_blocked_by(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    for beat in truth["beats"]:
        if beat["id"] == "tom_closes_off":
            beat["blocked_by_beats"] = ["tom_closes_off"]
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="contains itself"):
        load_world(target)


# ───────────────────────── invariant 9: no requires_beats cycles ─────────────────────────


def test_invariant_9_simple_cycle(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    # Create A -> B -> A by making tom_named_a_vehicle require tom_admits_more,
    # which already requires tom_named_a_vehicle.
    for beat in truth["beats"]:
        if beat["id"] == "tom_named_a_vehicle":
            beat["requires_beats"] = ["tom_admits_more"]
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="cycle"):
        load_world(target)


# ───────────────────────── invariant 10: ending requires_beats ─────────────────────────


def test_invariant_10_ending_requires_undefined_beat(tmp_path):
    target = _copy_valid(tmp_path)
    truth_path = target / "truth.yaml"
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    truth["endings"][0]["requires_beats"].append("imaginary_beat")
    _write_yaml(truth_path, truth)
    with pytest.raises(WorldLoadError, match="ending .* references"):
        load_world(target)


# ───────────────────────── missing files ─────────────────────────


def test_missing_locations_directory(tmp_path):
    target = _copy_valid(tmp_path)
    shutil.rmtree(target / "locations")
    with pytest.raises(WorldLoadError, match="missing locations directory"):
        load_world(target)


def test_missing_cast_yaml(tmp_path):
    target = _copy_valid(tmp_path)
    (target / "cast.yaml").unlink()
    with pytest.raises(WorldLoadError, match="missing cast.yaml"):
        load_world(target)


def test_missing_truth_yaml(tmp_path):
    target = _copy_valid(tmp_path)
    (target / "truth.yaml").unlink()
    with pytest.raises(WorldLoadError, match="missing truth.yaml"):
        load_world(target)


def test_missing_brief_yaml(tmp_path):
    target = _copy_valid(tmp_path)
    (target / "brief.yaml").unlink()
    with pytest.raises(WorldLoadError, match="missing brief.yaml"):
        load_world(target)
