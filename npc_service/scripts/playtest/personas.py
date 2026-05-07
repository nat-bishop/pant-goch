"""Player persona spec.

Each persona is a YAML file in `personas/` that defines a synthetic player's
voice, objective, and (optionally) the terminal we expect them to reach. The
synthetic-player agent reads this when generating each reply to keep the
persona consistent across the session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PersonaSpec:
    """Single player persona consumed by the synthetic-player agent."""

    name: str
    voice: str
    objective: str
    archetype: str
    quirks: list[str] = field(default_factory=list)
    expected_terminal: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "PersonaSpec":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            voice=data["voice"],
            objective=data["objective"],
            archetype=data["archetype"],
            quirks=data.get("quirks", []),
            expected_terminal=data.get("expected_terminal"),
        )


def load_personas(personas_dir: Path | None = None) -> list[PersonaSpec]:
    """Load every `*.yaml` file in `personas_dir`. Defaults to the sibling
    `personas/` directory next to this module."""
    if personas_dir is None:
        personas_dir = Path(__file__).resolve().parent / "personas"
    return sorted(
        (PersonaSpec.from_yaml(p) for p in personas_dir.glob("*.yaml")),
        key=lambda s: s.name,
    )
