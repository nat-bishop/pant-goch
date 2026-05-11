"""YAML → `Mission` loader. Single entry point: `load_mission(path)`.

Pydantic does the heavy lifting — the schema in `models.py` validates ids,
predicate shapes, terminal-beat outcome references, and id uniqueness via
the `Mission._id_uniqueness` model validator. This loader just deserialises
and surfaces errors with the file path.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

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
        return Mission.model_validate(raw)
    except ValidationError as e:
        raise MissionLoadError(f"{path}: {e}") from e
