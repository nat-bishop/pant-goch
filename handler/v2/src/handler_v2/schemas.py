"""Scene-runner LLM output schemas.

Two-stage emission per turn (SPEC §0.4 / §3.3 + plan):
  Phase 1 — `StateChangeBundle`: 0-N state changes (UnlockBeat only in v1;
            see plan "MoveNPC cut from v1"). Cap: ≤2 UnlockBeat per bundle.
  Phase 2 — `ProsePart`: prose + `next` (ContinueScene | EndScene | EndGame).

The schemas here use `str` for id fields. At call time, `agents.py` builds
per-call dynamic types (`Literal[*beat_ids]`, `Literal[*adjacent_location_ids]`,
`Literal[*ending_ids]`) and passes them via pydantic-ai's runtime `output_type=`
override on `agent.run()` — that's what the llama.cpp grammar sampler uses
to reject invalid ids at decode time. The str-typed schemas here serve as
the structural spec + the M2 unit-test surface.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Forbid extras — drift in LLM output (extra fields) should fail loud,
    not silently parse."""

    model_config = ConfigDict(extra="forbid")


# ───────────────────────── Phase 1: state changes ─────────────────────────


class UnlockBeat(_Strict):
    type: Literal["unlock_beat"] = "unlock_beat"
    beat_id: str


# Discriminated union for state changes. v1 has only UnlockBeat; widen to
# `UnlockBeat | MoveNPC | ...` if v1.1+ adds more kinds.
StateChange = Annotated[UnlockBeat, Field(discriminator="type")]


class StateChangeBundle(_Strict):
    state_changes: list[StateChange] = Field(default_factory=list)


# ───────────────────────── Phase 2: prose + next ─────────────────────────


class Option(_Strict):
    """One player-facing choice within a ContinueScene.

    `leaves_for` non-None marks this option as a leave option. The engine
    schema-validates that every ContinueScene includes ≥1 leave option, so
    the player always has a walk-away choice. When the player picks a leave
    option, phase 2 of the *next* turn must emit EndScene (engine output_validator
    enforces).
    """

    text: str
    leaves_for: str | None = None


class ContinueScene(_Strict):
    type: Literal["continue"] = "continue"
    options: list[Option]


class EndScene(_Strict):
    type: Literal["end_scene"] = "end_scene"


class EndGame(_Strict):
    type: Literal["end_game"] = "end_game"
    ending_id: str


SceneNext = Annotated[
    Union[ContinueScene, EndScene, EndGame],
    Field(discriminator="type"),
]


class ProsePart(_Strict):
    prose: str  # 80-200 words; streamed
    next: SceneNext
