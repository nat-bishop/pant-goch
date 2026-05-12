"""YAML → `Mission` loader. Single entry point: `load_mission(path)`.

Two-pass: (1) `dsl.normalise_mission_dsl` rewrites any string-form predicates
or effects into their tagged-union dict form; (2) pydantic validates the
result against `Mission`. The schema in `models.py` does the heavy lifting —
this loader just deserialises and surfaces errors with the file path.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .dsl import DSLParseError, normalise_mission_dsl
from .models import Mission


class MissionLoadError(Exception):
    """Wraps any YAML / pydantic validation failure with the file path."""


def load_mission(path: Path) -> Mission:
    """Load and validate a mission from a YAML file. Raises `MissionLoadError`
    with the offending path on any failure."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise MissionLoadError(f"failed to read {path}: {e}") from e
    if not isinstance(raw, dict):
        raise MissionLoadError(
            f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}"
        )
    try:
        normalised = normalise_mission_dsl(raw)
    except DSLParseError as e:
        raise MissionLoadError(f"{path}: DSL parse error — {e}") from e
    try:
        return Mission.model_validate(normalised)
    except ValidationError as e:
        raise MissionLoadError(f"{path}: {e}") from e
