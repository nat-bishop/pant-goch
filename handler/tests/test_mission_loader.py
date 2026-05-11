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
    assert {b.id for b in m.beats} >= {"win", "timeout", "passive_decay"}


def test_lynx_exfil_loads() -> None:
    m = load_mission(_LYNX)
    assert m.codename == "LYNX-EXFIL"
    assert m.max_cycles == 6
    assert len(m.archetypes) >= 5
    assert any(mm.id == "kestrel_sacrifice" for mm in m.moments)


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
