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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Frozen(BaseModel):
    """Base for world-bible types — frozen + forbid-extra so YAML typos
    surface loudly as a load error rather than silently dropping the field."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Location(_Frozen):
    id: str
    name: str
    # Short, prose-friendly form for closing-line narration ("heading
    # toward the church"). Falls back to `name` via `display_short` when
    # unset. The full `name` still drives slates, labels, the case file.
    short_name: str | None = None
    adjacent_to: list[str] = Field(default_factory=list)
    description: str
    notes: list[str] = Field(default_factory=list)

    @property
    def display_short(self) -> str:
        """Concise name for inline prose; the full name when no
        `short_name` is authored."""
        return self.short_name or self.name


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
    # LLM-facing short name. Phase 1 emits this (not the id) in
    # UnlockBeat.beat_name; the engine maps name→id at the boundary. Names
    # also lead every LLM-facing render (reachable beats, applied changes,
    # chronicler turn list). Hard caps below are dual: max_length caps the
    # grammar-sample token count; the word check catches "barely-50-chars
    # sentence" cases. Whitespace check guards the name→id lookup.
    name: str = Field(max_length=50)
    description: str
    requires_beats: list[str] = Field(default_factory=list)
    # OR-clause: when non-empty, at least one id here must be unlocked (or
    # reachable) — in addition to all of `requires_beats`. Empty = no OR.
    requires_any_beats: list[str] = Field(default_factory=list)
    blocked_by_beats: list[str] = Field(default_factory=list)
    # State-dependent fireability: when non-empty, the beat is only emittable
    # while every listed NPC is present at the player's location (enforced via
    # engine.fireable_beat_ids → per-call grammar narrowing). Leave empty for
    # object beats (reading a letter) and decision beats (player commitments)
    # that fire with no NPC in the room. Validated against the cast at load.
    requires_npcs_present: list[str] = Field(default_factory=list)
    # State-dependent fireability: when non-empty, the beat is only emittable
    # while the player is physically at one of these locations (enforced via
    # engine.fireable_beat_ids → per-call grammar narrowing). The hard gate
    # for place-bound beats like `examined_the_crash_site` (the Black Dog
    # Bend) that `notes_to_llm` alone could not enforce against a 4B model.
    # Validated against the locations at load.
    requires_location: list[str] = Field(default_factory=list)
    # Explicit opt-out of Invariant 14 (every beat carries ≥1 hard gate).
    # Set only on a beat that is *intentionally* ungated — the player's
    # always-available "I'm walking away" commitment. Self-documenting so the
    # ungated beat is a deliberate, greppable exception rather than an
    # oversight the loader would otherwise reject.
    always_available: bool = False
    prerequisites: str = ""  # soft prereqs the LLM judges, beyond requires/blocks
    notes_to_llm: str

    @field_validator("name")
    @classmethod
    def _name_terse(cls, v: str) -> str:
        if v != v.strip():
            raise ValueError(
                f"Beat.name cannot have leading/trailing whitespace: {v!r}"
            )
        if not v:
            raise ValueError("Beat.name cannot be blank")
        words = v.split()
        if len(words) > 8:
            raise ValueError(
                f"Beat.name must be ≤8 words (got {len(words)}): {v!r}"
            )
        return v


class Ending(_Frozen):
    id: str
    requires_beats: list[str] = Field(default_factory=list)
    # OR-clause mirror of Beat.requires_any_beats — see that docstring.
    # An ending fires iff ALL of `requires_beats` are unlocked AND (if
    # `requires_any_beats` is non-empty) AT LEAST ONE of those is unlocked.
    requires_any_beats: list[str] = Field(default_factory=list)
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


class Gloss(_Frozen):
    """Hover-popup entry for a phrase that may appear in prose. Frontend
    only — Phase 2 never sees these. `name` matches in prose along with
    every string in `aliases`; the matched phrase opens a popup whose
    heading is `name` and body is `body`."""

    id: str
    kind: Literal["npc", "location", "object"]
    name: str
    aliases: list[str] = Field(default_factory=list)
    body: str
    cross_ref: str | None = None
