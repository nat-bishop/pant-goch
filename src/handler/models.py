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


class Disposition(_Frozen):
    """Per-NPC quality control surface (the thin spine). Only authored on the
    handful of NPCs whose opening-up is load-bearing (Tom, Dylan, Helen,
    Eluned, Iwan); auxiliary characters omit it and stay pure prose.

    - `warms` / `spooks` set how many raw trust/guard points make one band
      (slow = widest = slowest to open). Replaces the old `min_scenes_with_npc`.
    - `breach` lists the prose triggers that latch this NPC's guard shut
      permanently (replaces the per-NPC negative beats). The LLM emits a guard
      BREACH only for these.
    - `voice_by_band` is the EXISTING `voice_sample` registers, keyed by band
      (cold/thawing/open/at_peace + the guard-high `closed`). The engine renders
      the current band's lines into the prose writer's ## Present card.
    - `impression_by_band` is the journalist's own read on this person, keyed by
      band — surfaced on the dossier "People" card (frontend), one short
      third-person line per band. Diegetic, never a number; parallel to
      `voice_by_band` but in the reporter's voice. Resolved server-side into the
      `npc_standing` SSE event; an unauthored band shows no impression line.
    """

    warms: Literal["slow", "steady", "fast"] = "steady"
    spooks: Literal["slow", "steady", "fast"] = "steady"
    breach: list[str] = Field(default_factory=list)
    voice_by_band: dict[str, list[str]] = Field(default_factory=dict)
    impression_by_band: dict[str, str] = Field(default_factory=dict)
    # Locations where this NPC will WARM (positive trust accrues) — the
    # mechanical home of "Dylan only opens up somewhere private". Empty = no
    # restriction. Guard/BREACH still apply anywhere (he can shut down at the
    # farm); only positive trust is location-pinned.
    opens_up_only_at: list[str] = Field(default_factory=list)


class When(_Frozen):
    """Quality/milestone gate, evaluated by `engine.when_satisfied`. Replaces
    the boolean `requires_*` proxy family. All set conditions must hold (AND).

      - `npc` + `trust_at_least`: the named NPC has reached this confession band.
      - `any_evidence_of`: at least one of these beats is unlocked (the evidence-OR).
      - `requires_beats`: all of these beats are unlocked (e.g. a decision beat).
      - `npcs_thawed_at_least`: at least N distinct NPCs have reached `thawing`
        (the walk-away traction barrier — replaces the old requires_count:3).
      - `village_attention_at_least` / `_below`: the global heat band.
    """

    npc: str | None = None
    trust_at_least: Literal["thawing", "open", "at_peace"] | None = None
    any_evidence_of: list[str] = Field(default_factory=list)
    requires_beats: list[str] = Field(default_factory=list)
    npcs_thawed_at_least: int | None = None
    village_attention_at_least: Literal["noticed", "talked_about", "hostile"] | None = None
    village_attention_below: Literal["noticed", "talked_about", "hostile"] | None = None


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
    # Quality control surface — see Disposition. None for auxiliary NPCs
    # (Gareth, Owain, Wendy, Maud) who stay pure prose and carry no tracked
    # quality.
    disposition: Disposition | None = None


class Beat(_Frozen):
    """A milestone in `state.unlocked_beats` (once unlocked, permanent). Two
    kinds in the quality engine:

      - engine-derived (`commitment=False`, carries `when`): an evidence /
        outcome milestone the engine auto-unlocks when its `when` gate is met
        (e.g. trust reaching a band). The LLM never fires these.
      - LLM-emitted (`commitment=True`): a player decision (`decided_*`,
        mutually exclusive) or an observation milestone (gated on
        `requires_location`). The classifier emits these explicitly.
    """

    id: str
    # LLM-facing short name. Phase 1 emits this (not the id) in
    # UnlockBeat.beat_name for commitments; the engine maps name→id at the
    # boundary. Names also lead every LLM-facing render (choices available,
    # applied changes, chronicler turn list). max_length caps the
    # grammar-sample token count; the word check catches over-long names;
    # the whitespace check guards the name→id lookup.
    name: str = Field(max_length=50)
    description: str
    # Observation milestones (reading the letter, standing at the crash site)
    # gate on the player's LOCATION rather than on a person opening up. Empty
    # for trust-derived milestones and decisions. Validated against locations
    # at load (engine.emittable_beat_ids consumes it).
    requires_location: list[str] = Field(default_factory=list)
    notes_to_llm: str
    # ── quality engine gate ──
    # `when` is the quality/milestone gate (see When). For a trust-derived
    # evidence milestone it carries {npc, trust_at_least}; for a commitment it
    # gates availability (e.g. any_evidence_of). `commitment=True` marks the
    # LLM-emitted beats (player decisions + observation milestones); everything
    # else with a `when` is engine-derived (auto-unlocked when the gate is met).
    when: When | None = None
    commitment: bool = False

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
    # The decision/milestone beat(s) that must ALL be unlocked.
    requires_beats: list[str] = Field(default_factory=list)
    # Quality gate on top of requires_beats — the de-redundification surface.
    # e.g. iwan_finds_peace carries when={npc: iwan, trust_at_least: open};
    # run_out_of_the_village carries when={village_attention_at_least: hostile}
    # and no requires_beats.
    when: When | None = None
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
