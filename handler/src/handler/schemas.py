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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    state_changes: list[StateChange] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _no_duplicate_beat_ids(self) -> StateChangeBundle:
        seen: set[str] = set()
        for change in self.state_changes:
            if isinstance(change, UnlockBeat):
                if change.beat_id in seen:
                    raise ValueError(
                        f"beat_id {change.beat_id!r} appears twice in bundle"
                    )
                seen.add(change.beat_id)
        return self


# ───────────────────────── Phase 2: prose + next ─────────────────────────


class Option(_Strict):
    """One player-facing choice within a ContinueScene.

    `leaves_for` non-None marks this option as a leave option.
    `ContinueScene._has_at_least_one_leave_option` enforces ≥1 leave
    option per ContinueScene (player always has a walk-away choice). When
    the player picks a leave option, the next turn's dynamic ProsePart
    schema narrows `next` to `EndScene` — grammar-enforced.
    """

    text: str
    leaves_for: str | None = None


class ContinueScene(_Strict):
    type: Literal["continue"] = "continue"
    options: list[Option]

    @model_validator(mode="after")
    def _has_at_least_one_leave_option(self) -> ContinueScene:
        if not any(o.leaves_for is not None for o in self.options):
            raise ValueError(
                "ContinueScene must include at least one option with "
                "leaves_for set (player must always have a walk-away choice)"
            )
        return self


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
