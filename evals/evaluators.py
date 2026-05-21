"""Deterministic invariant gates (Tier 0) — pure functions of a `SessionArtifact`.

Each gate scores signal the engine does not already guarantee and that must
*hold* on every run (an ending was reached; the session wasn't truncated; no
duplicate case-file entries; an ending wasn't committed prematurely). Graded
distributional signal — variety, coverage, agency — lives in `metrics.py`; LLM
behavioural judgments — was a beat's firing warranted — live in `judges/`.

The reachability sweep (`coverage_report`) is corpus-level, not a per-case gate,
so it is re-exported from `metrics.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.models import Beat, Ending
from handler.world import World

from .metrics import coverage_report, distinct_locations
from .session import SessionArtifact

__all__ = [
    "EndingReached",
    "NotTruncated",
    "NoDuplicateCaseFiles",
    "EndingEarned",
    "ExploredBeyondStart",
    "NoPrematureEnding",
    "invariant_evaluators",
    "coverage_report",
]

# A committed ending should follow real investigation, not a scene-2 collapse.
# Tuned to the 20260520 failure (median 2 scenes / ~2 locations): any ending
# below these floors is premature regardless of which beat fired.
MIN_SCENES_FOR_ENDING = 3
MIN_LOCATIONS_FOR_ENDING = 3


@dataclass
class EndingReached(Evaluator):
    """A non-truncated session should commit to an ending."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        art: SessionArtifact = ctx.output
        if art.truncated:
            return True  # truncation is reported separately by NotTruncated
        return art.ending_id is not None


@dataclass
class NotTruncated(Evaluator):
    """Surfaces sessions that hit the scene budget without ending."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return not ctx.output.truncated


@dataclass
class NoDuplicateCaseFiles(Evaluator):
    """No two scenes produced an identical case-file entry."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        entries = ctx.output.case_file_entries
        return len(entries) == len(set(entries))


def _committed_ending(art: SessionArtifact, world: World) -> Ending | None:
    """The Ending the session committed to, or None when there is nothing to
    judge (truncated, no ending fired, or an id the world doesn't define)."""
    if art.truncated or art.ending_id is None:
        return None
    return world.endings.get(art.ending_id)


def _evidence_decision_beats(ending: Ending, world: World) -> list[Beat]:
    """The ending's gating decision beats that carry evidence — i.e. have a
    non-empty `requires_any_beats` (Tom's account / Dylan's admission / Helen's
    notes). A walk-away ending gates on a beat with no such evidence and so
    yields an empty list."""
    return [
        b for bid in ending.requires_beats
        if (b := world.beats.get(bid)) is not None and b.requires_any_beats
    ]


@dataclass
class EndingEarned(Evaluator):
    """A reached ending must have its evidence actually unlocked. If a gating
    decision beat carries evidence but NONE of it is in `unlocked_beats`, the
    beat was emitted prematurely — exactly the regression that let
    `iwan_finds_peace` / `eluned_arrested…` fire in the empty prologue. The
    walk-away ending (no evidence-bearing decision beat) passes vacuously.

    NOTE: largely SUPERSEDED by `NoPrematureEnding` — this passes vacuously on
    the dominant failure (walk-away firing at scene 2), since `decided_to_walk_away`
    gates on engagement beats, not evidence. Kept for the evidence-specific
    regression it was born from; `NoPrematureEnding` is the live depth gate.
    """

    world: World

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        ending = _committed_ending(ctx.output, self.world)
        if ending is None:
            return True
        unlocked = set(ctx.output.unlocked_beats)
        return all(
            any(e in unlocked for e in beat.requires_any_beats)
            for beat in _evidence_decision_beats(ending, self.world)
        )


@dataclass
class ExploredBeyondStart(Evaluator):
    """Any committed ending must have come after the player left the starting
    location. Catches the 'whole game collapses into the prologue' failure: no
    ending — not even walking away — is reachable without engaging the village,
    so a session that commits to one while confined to the start location is a
    regression (a beat fired where it should not have).
    """

    world: World

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        ending = _committed_ending(ctx.output, self.world)
        if ending is None:
            return True
        return len(distinct_locations(ctx.output)) > 1


@dataclass
class NoPrematureEnding(Evaluator):
    """ANY committed ending must follow real investigation: at least
    `MIN_SCENES_FOR_ENDING` scenes AND `MIN_LOCATIONS_FOR_ENDING` distinct
    locations. This is the live gate the prologue-era checks miss — it fires on
    the scene-2 collapse (walk-away / run-out firing after one Maud conversation)
    regardless of which beat ended the game. Truncated / no-ending runs are
    exempt (reported by NotTruncated / EndingReached)."""

    world: World

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        art = ctx.output
        if _committed_ending(art, self.world) is None:
            return True
        return (
            art.scene_count >= MIN_SCENES_FOR_ENDING
            and len(distinct_locations(art)) >= MIN_LOCATIONS_FOR_ENDING
        )


def invariant_evaluators(world: World) -> list[Evaluator]:
    """The full per-session invariant set, ready to attach to a Dataset.

    `world`-aware so the depth/evidence gates can judge whether a reached ending
    was earned (its evidence unlocked, enough scenes/locations behind it) rather
    than fired prematurely. `NoPrematureEnding` is the live depth gate;
    `EndingEarned`/`ExploredBeyondStart` cover the older evidence/prologue cases.
    """
    return [
        EndingReached(),
        NotTruncated(),
        NoDuplicateCaseFiles(),
        EndingEarned(world),
        ExploredBeyondStart(world),
        NoPrematureEnding(world),
    ]
