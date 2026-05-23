"""Scene-runner LLM output schemas.

Two-stage emission per turn, with Phase 2 itself now split into prose-writer
and choice-author calls (SPEC §0.4 / §3.3 + the Phase-2-split refactor):

  Phase 1 — `StateChangeBundle`: 0-N state changes (UnlockBeat only in v1).
            Cap: ≤2 UnlockBeat per bundle.
  Phase 2a — prose writer: plain `str` output (no schema below — the agent
            uses `output_type=str`).
  Phase 2b — choice author: `ChoiceBundle` with a Continue | Leave tagged
            union. Stateless per call. Runs concurrently with —
  Phase 2c — free-text gate: `FreeTextGate` (a single `offer: bool`).
            Decides whether the player writes their own move this turn
            instead of picking a canned option. Stateless, pure-prose.

The engine assembles a composite `ProsePart` from the Phase 2 calls so
downstream consumers (scene_loop, CLI, SSE) keep dispatching on the same
shape. `ProsePart.next` is `Continue | EndScene | EndGame | FreeText` —
`Continue` passes through from the choice author verbatim; `FreeText` is
built by the engine (fixed caption) when the gate offers it; `EndScene`
and `EndGame` are engine-only (the LLM never emits an `ending_id`).

The schemas here use `str` for id fields. At call time, `agents.py` builds
per-call dynamic types (`Literal[*beat_names]`, `Literal[*present_npc_ids]`)
and passes them via pydantic-ai's runtime `output_type=` override on
`agent.run()` — that's what the llama.cpp grammar sampler uses to reject
invalid ids at decode time. The str-typed schemas here serve as the
structural spec + the M2 unit-test surface.
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


# ── bounded ordinal quality deltas (the quality-engine Phase-1 output) ──
# String literals so the same decode-time grammar narrowing that constrains
# beat names / npc ids applies verbatim; the engine parses "+1"→+1 and
# "BREACH"→latch after decode (mirrors beat_name→beat_id). Bounding to ±1 (and
# the deliberate, separate BREACH member for guard) is what keeps a runaway
# emission to one recoverable notch instead of a permanent mis-fire.
TrustDelta = Literal["-1", "0", "+1"]
GuardDelta = Literal["-1", "0", "+1", "BREACH"]

# Parsers: turn the on-the-wire string into an engine effect.
TRUST_DELTA_VALUES = {"-1": -1, "0": 0, "+1": 1}


class QualityDelta(_Strict):
    """One present-NPC's trust/guard nudge this turn. `guard="BREACH"` is the
    latch signal (not a magnitude); the engine adds the npc to
    `state.guard_breached` and leaves the numeric guard alone."""

    type: Literal["quality_delta"] = "quality_delta"
    npc_id: str
    trust: TrustDelta = "0"
    guard: GuardDelta = "0"


class StateChangeBundle(_Strict):
    """Phase-1 output (quality engine). Per turn the classifier reports how the
    player's move shifted standing — bounded ordinal `deltas` for present NPCs —
    and at most one explicit `commitments` beat (an observation milestone or a
    player decision). Evidence/outcome milestones are NOT here: the engine
    derives them from trust (`engine._derive_milestones`). Global
    `village_attention` is NOT here either — it is owned by a dedicated
    scene-close agent (`agents.emit_village_attention`)."""

    deltas: list[QualityDelta] = Field(default_factory=list, max_length=4)
    commitments: list[UnlockBeat] = Field(default_factory=list, max_length=1)

    @model_validator(mode="after")
    def _no_duplicate_delta_npcs(self) -> StateChangeBundle:
        """One quality delta per NPC per turn. trust/guard accrue ±1 a turn by
        design (engine band width); two deltas for the same NPC in one bundle
        would let a single turn move standing by ±2 and cross a confession band
        early. The grammar sampler narrows npc_id but does not forbid repeats,
        so the structural cap lives here (agents.emit_state_changes dedups the
        raw model output before constructing the bundle, so this never trips in
        the live path — it guards every other constructor)."""
        seen: set[str] = set()
        for d in self.deltas:
            npc_id = d.npc_id
            if npc_id in seen:
                raise ValueError(
                    f"npc_id {npc_id!r} appears in more than one quality delta"
                )
            seen.add(npc_id)
        return self


# ───────────────────────── Phase 2b: choice author output ─────────────────────────


class Option(_Strict):
    """One player-facing choice within a Continue. Just text now —
    `leaves_for` was removed (the map is the navigation surface per SPEC §1)."""

    text: str


class Continue(_Strict):
    """Choice-author variant: keep the scene going with 2-4 canned options."""

    type: Literal["continue"] = "continue"
    options: list[Option] = Field(min_length=2, max_length=4)


class Leave(_Strict):
    """Choice-author variant: the scene has reached a natural close. The
    engine treats the just-written prose as the closing beat and resolves
    which ending (if any) fires from `state.unlocked_beats`."""

    type: Literal["leave"] = "leave"


class FreeText(_Strict):
    """Engine-built ProsePart variant: the natural next move is the player
    writing their own move (speech *or* action) rather than picking a
    canned option. Surfaced when the free-text gate (`FreeTextGate`) offers
    AND the engine cadence allows; the engine then builds this per turn
    from the caption author's output (`prompt`/`suggested`) and feeds the
    typed string into the next turn as the player's pick.

    The LLM never emits this directly — the gate emits a bool and the
    caption author emits `FreeTextCaption`; the engine constructs the
    `FreeText`. `npc_id` is always None (general agency: the typed move is
    not necessarily addressed to anyone).

    `frozen` so the engine's shared `FREE_TEXT_FALLBACK` singleton can't be
    mutated by a downstream consumer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["free_text"] = "free_text"
    prompt: str
    npc_id: str | None = None
    # Caption-author suggestion (`FreeTextCaption.suggested`): a first-person
    # move pre-loaded into the box as a muted, editable backup. None when
    # the caption author failed and the engine fell back to a fixed caption.
    suggested: str | None = None


ChoiceNext = Annotated[
    Union[Continue, Leave],
    Field(discriminator="type"),
]


class ChoiceBundle(_Strict):
    """Choice-author output. The prose writer's text + this bundle's `next`
    are composed by the engine into a `ProsePart` for downstream dispatch."""

    next: ChoiceNext


class FreeTextGate(_Strict):
    """Free-text gate output (Phase 2c). A single yes/no: should the player
    write their own move this turn instead of picking a canned option?

    Pure-prose judgment, runs concurrently with the choice author. When
    `offer` is True the engine surfaces a free-text box (and Leave still
    takes precedence — see `scene_loop._assemble_prose_part`)."""

    offer: bool


class FreeTextCaption(_Strict):
    """Caption-author output (Phase 2d). Runs only when the gate offered
    AND the engine cadence allows (see `scene_loop.run_turn`). Pure-prose
    judgment like the gate — sees only the just-written prose.

    Two short strings:
      - `caption` — the eyebrow shown above the box, an invitation fitted
        to *this* moment (e.g. "What do you say to her?", "Your move.").
      - `suggested` — one suggested first-person move, pre-loaded into the
        textarea as a muted, editable backup the player can submit, tweak,
        or clear.

    Lengths are capped to keep both terse — a caption is a line, not a
    paragraph; the suggestion is a single move."""

    caption: str = Field(min_length=1, max_length=80)
    suggested: str = Field(min_length=1, max_length=160)


# ───────────────────────── engine-composite: ProsePart for downstream dispatch ─────────────────────────
#
# `Continue` and `FreeText` are reused verbatim in the ProsePart union;
# only Leave is rewritten as EndScene / EndGame, since the engine resolves
# which one fires from satisfied prereqs.


class EndScene(_Strict):
    """Engine-only variant. The LLM never emits this. Two roles, disambiguated
    by `to_location`:

    - `to_location=None` — a **soft-close request**: the choice author emitted
      Leave with no ending satisfied, so the scene is spent and the engine
      awaits the player's destination (`scene_loop.play_scene` surfaces the
      map and waits for a `WalkTo`).
    - `to_location` set — the **finalized scene end**: the player has walked,
      the closing prose has streamed, and this names where they went. Returned
      by `play_scene` for `play_session` to drive `location_change` + the
      player_location update.
    """

    type: Literal["end_scene"] = "end_scene"
    to_location: str | None = None


class EndGame(_Strict):
    """Engine-only variant: from a choice-author Leave when an ending IS
    satisfied. `ending_id` is engine-resolved via
    `engine.resolved_ending_id_at_close`; the LLM never emits it."""

    type: Literal["end_game"] = "end_game"
    ending_id: str


SceneNext = Annotated[
    Union[Continue, EndScene, EndGame, FreeText],
    Field(discriminator="type"),
]


class ProsePart(_Strict):
    prose: str  # 30–100 words; streamed
    next: SceneNext


# ───────────────────────── Chronicler dual output ─────────────────────────


class NpcMemoryUpdate(_Strict):
    """One per-NPC memory note emitted by the chronicler at scene close.

    Per call, `npc_id` is narrowed to `Literal[*present_npc_ids]` by
    `agents._build_chronicle_entry_type`, so the grammar sampler
    physically cannot emit an out-of-cast or out-of-scene id. `note` is
    1-2 sentences in the journalist's voice — "what I now know / feel
    about this person", distinct from the case file's "what happened"
    register.
    """

    npc_id: str
    note: str


class ChronicleEntry(_Strict):
    """Scene-end chronicler emission. One LLM call emits two artifacts
    under one set of voice rules — `case_file_entry` (player-facing
    sidebar prose) and `npc_memory_updates` (structured per-NPC memory
    channel surfaced to Phase 2 on subsequent scenes). Single call so
    the journalist's interpretation is coherent across both.

    `npc_memory_updates` shape and constraints — see `NpcMemoryUpdate`
    and `agents._build_chronicle_entry_type`. The engine merges each
    note into `state.npc_memory[npc_id]` (appending one entry per
    scene the NPC was present).
    """

    case_file_entry: str
    npc_memory_updates: list[NpcMemoryUpdate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_npc_ids(self) -> ChronicleEntry:
        seen: set[str] = set()
        for upd in self.npc_memory_updates:
            if upd.npc_id in seen:
                raise ValueError(
                    f"npc_id {upd.npc_id!r} appears twice in npc_memory_updates"
                )
            seen.add(upd.npc_id)
        return self


# ───────────────────────── Village attention (scene-close heat) ─────────────────────────


class VillageAttentionVerdict(_Strict):
    """Scene-close village-attention classifier output. A dedicated agent reads
    the just-closed scene's transcript (plus the prior heat level) and judges
    how much the village has COLLECTIVELY closed ranks against the journalist.

    Split out of the Phase-1 `StateChangeBundle` because, bundled with the
    salient per-NPC trust/guard deltas, the model anchored on those and left
    attention at its default — so private aggression never raised it. Asked in
    isolation the same model classifies it reliably. The level maps to
    `engine.ATTENTION_LEVEL_VALUES`; the engine ratchets `village_attention` up
    to it (max), preserving the up-only invariant."""

    level: Literal["none", "noticed", "talked_about", "hostile"]


# ───────────────────────── Populator (scene-entry NPC placement) ─────────────────────────


class NpcPlacement(_Strict):
    """One NPC-at-a-location override emitted by the populator agent.

    The populator runs once at scene entry. It decides which NPCs are
    where in light of the case file, beats so far, and dramatic logic
    (e.g., Dylan retreats to the wood after a tense Tan-y-coed scene).
    Defaults remain in effect for NPCs the populator doesn't move.
    """

    npc_id: str
    location_id: str


class NpcPlacementBundle(_Strict):
    """Populator output. `placements` is a (possibly empty) list of
    NPC overrides. Empty bundle = default placements stand for this scene
    — the common case (most scene entries don't need movement)."""

    placements: list[NpcPlacement] = Field(default_factory=list)
