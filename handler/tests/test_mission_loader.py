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
    assert {c.id for c in m.cast} == {"ALPHA", "BRAVO", "HANDLER"}
    assert m.member("HANDLER").role == "handler"
    # Normal actions + the forced turning_point action.
    assert {a.id for a in m.actions} >= {"PUSH", "SACRIFICE_BY_AGENT", "HOLD", "turning_point"}
    assert {r.id for r in m.reactions} >= {"win", "timeout", "passive_decay"}


def test_tiny_mission_loads_opening_briefing() -> None:
    """The tiny fixture authors a day-1 briefing."""
    m = load_mission(_TINY)
    assert "Test briefing" in m.opening_briefing


def test_lynx_exfil_loads() -> None:
    m = load_mission(_LYNX)
    assert m.codename == "LYNX-EXFIL"
    assert m.max_days == 6
    assert len(m.actions) >= 5
    # kestrel_sacrifice is now a forced Action (no more force_into_drafter on Event).
    assert any(a.id == "kestrel_sacrifice" and a.forced for a in m.actions)


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


def test_load_fails_on_quality_without_bands(tmp_path: Path) -> None:
    """A `qualities:` entry with `initial:` but no `bands:` is rejected by
    the per-quality validator. Co-locating initial + descriptor means the
    author can't write one without the other."""
    bad = tmp_path / "missing_descriptor.yaml"
    bad.write_text(
        """\
id: bad
name: "Bad mission — undescribed quality"
max_days: 3
cast:
  - id: A
    codename: A
    starting_cover: 5
qualities:
  mission.undescribed:
    initial: 0
    # no bands — the Quality model rejects this
actions:
  - id: X
    when:
      kind: quality_gte
      quality: a
      value: 0
reactions:
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
    assert "bands" in str(exc.value)
