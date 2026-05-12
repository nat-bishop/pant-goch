"""YAML → Mission loader tests. The schema validators in `models.py` do the
heavy lifting; this suite mostly ensures the wrapper reports errors with the
file path attached."""
from __future__ import annotations

from pathlib import Path

import pytest

from handler.mission_loader import MissionLoadError, load_mission


_FIXTURES = Path(__file__).parent / "fixtures"
_TINY = _FIXTURES / "tiny_mission.yaml"
_LYNX = Path(__file__).resolve().parents[1] / "missions" / "lynx_exfil.yaml"


def test_tiny_mission_loads() -> None:
    m = load_mission(_TINY)
    assert m.id == "tiny"
    assert {a.id for a in m.agents} == {"ALPHA", "BRAVO"}
    assert {a.id for a in m.archetypes} == {"PUSH", "SACRIFICE_BY_AGENT", "HOLD"}
    assert {e.id for e in m.events} >= {"win", "timeout", "passive_decay"}


def test_lynx_exfil_loads() -> None:
    m = load_mission(_LYNX)
    assert m.codename == "LYNX-EXFIL"
    assert m.max_cycles == 6
    assert len(m.archetypes) >= 5
    assert any(e.id == "kestrel_sacrifice" and e.force_into_drafter for e in m.events)


def test_missing_file_raises_with_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(MissionLoadError) as exc:
        load_mission(missing)
    assert "nope.yaml" in str(exc.value)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: valid: yaml: at: all", encoding="utf-8")
    with pytest.raises(MissionLoadError):
        load_mission(bad)


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(MissionLoadError) as exc:
        load_mission(bad)
    assert "mapping" in str(exc.value)


def test_load_fails_on_quality_without_descriptor(tmp_path: Path) -> None:
    """A `qualities:` entry with `initial:` but no `buckets:`/`values:` is
    rejected by the per-quality validator. Co-locating initial + descriptor
    means the author can't write one without the other."""
    bad = tmp_path / "missing_descriptor.yaml"
    bad.write_text(
        """\
id: bad
name: "Bad mission — undescribed quality"
max_cycles: 3
agents:
  - id: A
    codename: A
    starting_cover: 5
qualities:
  mission.undescribed:
    initial: 0
    # no buckets, no values — the Quality model rejects this
archetypes:
  - id: X
    when:
      kind: quality_gte
      quality: a
      value: 0
events:
  - id: end
    terminal: true
    outcome_key: w
    priority: 0
    when:
      kind: quality_gte
      quality: a
      value: 99
outcomes:
  w: ""
""",
        encoding="utf-8",
    )
    with pytest.raises(MissionLoadError) as exc:
        load_mission(bad)
    msg = str(exc.value)
    # The Quality model's _has_descriptor validator names buckets/values.
    assert "buckets" in msg or "values" in msg
