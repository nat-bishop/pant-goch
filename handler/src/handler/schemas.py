"""Scene-runner LLM output schemas.

Two-stage emission per turn, with Phase 2 itself now split into prose-writer
and choice-author calls (SPEC §0.4 / §3.3 + the Phase-2-split refactor):

  Phase 1 — `StateChangeBundle`: 0-N state changes (UnlockBeat only in v1).
            Cap: ≤2 UnlockBeat per bundle.
  Phase 2a — prose writer: plain `str` output (no schema below — the agent
            uses `output_type=str`).
  Phase 2b — choice author: `ChoiceBundle` with a Continue | Leave | FreeText
            tagged union. Stateless per call.

The engine assembles a composite `ProsePart` from the two Phase 2 calls so
downstream consumers (scene_loop, CLI, SSE) keep dispatching on the same
shape. `ProsePart.next` is `Continue | EndScene | EndGame | FreeText` —
`Continue` and `FreeText` pass through from the choice author verbatim;
`EndScene` and `EndGame` are engine-only (the LLM never emits an
`ending_id`).

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
    """Choice-author variant: the natural next move is the player typing
    their own line — interrogation, a delicate moment where canned options
    would cheapen it. The engine surfaces the prompt to the player, takes
    the typed string, and feeds it into the next turn as the player's pick.

    `npc_id` (optional) is narrowed per-call to `Literal[*present_npc_ids]`
    so the model cannot direct typed input at an NPC who is not in the
    scene.
    """

    type: Literal["free_text"] = "free_text"
    prompt: str
    npc_id: str | None = None


ChoiceNext = Annotated[
    Union[Continue, Leave, FreeText],
    Field(discriminator="type"),
]


class ChoiceBundle(_Strict):
    """Choice-author output. The prose writer's text + this bundle's `next`
    are composed by the engine into a `ProsePart` for downstream dispatch."""

    next: ChoiceNext


# ───────────────────────── engine-composite: ProsePart for downstream dispatch ─────────────────────────
#
# `Continue` and `FreeText` are reused verbatim in the ProsePart union;
# only Leave is rewritten as EndScene / EndGame, since the engine resolves
# which one fires from satisfied prereqs.


class EndScene(_Strict):
    """Engine-only variant: from a choice-author Leave when no ending is
    satisfied. The LLM never emits this."""

    type: Literal["end_scene"] = "end_scene"


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
    prose: str  # 80-200 words; streamed
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
