"""World-bible Pydantic types.

The author writes ~15 YAML files in `world/`. These types parse them.
Per-record shape validation lives here (Pydantic field validators);
cross-file invariants (id uniqueness, symmetric adjacency, beat-graph
acyclicity, etc.) live in `world.py`'s loader. SPEC §2.6 lists the 10
invariants enforced.

v2 has no flags, no qualities, no effects, no predicates, no DSL — beats
are unlock units identified by id. Schema mirrors that simplicity.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base for world-bible types — frozen + forbid-extra so YAML typos
    surface loudly as a load error rather than silently dropping the field."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Location(_Frozen):
    id: str
    name: str
    adjacent_to: list[str] = Field(default_factory=list)
    description: str
    notes: list[str] = Field(default_factory=list)


class NPC(_Frozen):
    id: str
    name: str
    default_location: str
    themes: list[str] = Field(default_factory=list)
    knowledge: str
    attitude: str
    # Why: 1-2 short representative lines used by Phase 2's narrator-voice
    # system prompt. For a 4B model, "Aye. Not my business, was it." carries
    # more dialogue weight than three lines of attitude prose. Defaults to []
    # so older fixtures load; production cast should fill it.
    voice_sample: list[str] = Field(default_factory=list)


class Beat(_Frozen):
    """A unit of investigation progress. Fired by the LLM via UnlockBeat
    state-change. Once fired, stays in `state.unlocked_beats` permanently.

    Negative beats (`tom_closes_off`, `dylan_refuses_to_speak_again`,
    `eluned_throws_you_out`) use `blocked_by_beats` on the corresponding
    positive beats to make failure paths permanent.

    Decision beats encode irrevocable player choices that gate endings.
    The ending's `requires_beats` carries only the decision beat (which
    in turn carries the evidence prereqs); endings never end the game on
    inferred intent.
    """

    id: str
    description: str
    requires_beats: list[str] = Field(default_factory=list)
    blocked_by_beats: list[str] = Field(default_factory=list)
    prerequisites: str = ""  # soft prereqs the LLM judges, beyond requires/blocks
    notes_to_llm: str


class Ending(_Frozen):
    id: str
    requires_beats: list[str] = Field(default_factory=list)
    notes_to_llm: str
    epilogue: str


class Truth(_Frozen):
    """Container for the truth file: the crime + beats + endings."""

    crime: str
    beats: list[Beat]
    endings: list[Ending]


class Brief(_Frozen):
    player_starts_at: str
    opening_seed: str


class FewShot(_Frozen):
    """One hand-authored example exchange surfaced in Phase 2's system
    prompt as a [STYLE EXAMPLES] block. `setup` is a one-line scene-set;
    `player` is the option text the player picked; `narrator` is the
    prose the narrator wrote back. The biggest single lever for prose
    quality on a 4B model."""

    setup: str
    player: str
    narrator: str
