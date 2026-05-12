"""Player persona spec for HANDLER's option-picking playtest.

A persona is a weighted preference profile over archetypes plus a few
behavioural flags. The option picker (`option_picker.py`) scores each drafted
option against the persona and picks the highest-scoring one.

YAML schema:

  name: cold_calculator
  archetype: "Cold Calculator"
  description: |
    Sacrifices when needed. Pushes the mission forward.
  expected_outcome: win_lynx_extracted   # optional — evaluator anchor

  priorities:
    RUN_EXFIL: 100
    RETRIEVE_PAPERS_VIA_AGENT: 80
    ACTIVATE_SLEEPER_FOR_PAPERS: 70
    LYNX_MOVES_TO_SAFEHOUSE: 50
    DEAD_DROP_PAPERS: 40
    DIVERSION_BY_AGENT: 30
    HOLD: -50

  # When a forced event appears in the option set, always pick the option
  # that satisfies it (overrides priorities). Default true.
  always_satisfy_forced_events: true
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PersonaSpec:
    name: str
    archetype: str
    description: str
    priorities: dict[str, int]
    expected_outcome: str | None = None
    always_satisfy_forced_events: bool = True
    quirks: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "PersonaSpec":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            archetype=data["archetype"],
            description=data["description"],
            priorities=dict(data.get("priorities", {})),
            expected_outcome=data.get("expected_outcome"),
            always_satisfy_forced_events=data.get("always_satisfy_forced_events", True),
            quirks=list(data.get("quirks", [])),
        )


def load_personas(personas_dir: Path | None = None) -> list[PersonaSpec]:
    """Load every `*.yaml` in `personas_dir`, sorted by persona name."""
    if personas_dir is None:
        personas_dir = Path(__file__).resolve().parent / "personas"
    return sorted(
        (PersonaSpec.from_yaml(p) for p in personas_dir.glob("*.yaml")),
        key=lambda s: s.name,
    )
