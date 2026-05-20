"""Deterministic invariant evaluators (Bucket 1) + the corpus reachability
aggregate.

Each evaluator is a pure function of the `SessionArtifact`, scoring signal the
engine does not already guarantee (an ending was reached; the session wasn't
truncated; the chronicler's unconstrained prose didn't repeat a case-file
entry). The reachability sweep is a corpus-level function (not a per-case
evaluator) because dead-content detection is only meaningful across the whole
run.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.models import Beat, Ending
from handler.world import World

from .session import SessionArtifact


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


def _distinct_locations(art: SessionArtifact) -> set[str]:
    """Every location the session touched, read from the transcript (turn /
    free_text / scene_close carry `location`; walk carries `from`/`to`)."""
    locs: set[str] = set()
    for e in art.transcript:
        if "location" in e:
            locs.add(e["location"])
        if e.get("kind") == "walk":
            locs.add(e["from"])
            locs.add(e["to"])
    return locs


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
    """A session that reached an evidence-bearing ending must have left the
    starting location. Catches the 'whole game collapses into the prologue'
    failure: no investigation can be earned without visiting the village. A
    deliberate walk-away (no evidence-bearing decision beat) is exempt.
    """

    world: World

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        ending = _committed_ending(ctx.output, self.world)
        if ending is None:
            return True
        if not _evidence_decision_beats(ending, self.world):
            return True  # walk-away / trivial ending — quitting at the start is legal
        return len(_distinct_locations(ctx.output)) > 1


def invariant_evaluators(world: World) -> list[Evaluator]:
    """The full per-session invariant set, ready to attach to a Dataset.

    `world`-aware so `EndingEarned` / `ExploredBeyondStart` can judge whether a
    reached ending was actually earned (its evidence unlocked, the village
    visited) rather than fired prematurely in the prologue.
    """
    return [
        EndingReached(),
        NotTruncated(),
        NoDuplicateCaseFiles(),
        EndingEarned(world),
        ExploredBeyondStart(world),
    ]


def coverage_report(artifacts: list[SessionArtifact], world: World) -> dict:
    """Corpus-level reachability sweep: which authored beats / endings never
    fired across the whole run. The dead-content detector."""
    fired_beats: set[str] = set()
    reached_endings: set[str] = set()
    for art in artifacts:
        fired_beats.update(art.unlocked_beats)
        if art.ending_id:
            reached_endings.add(art.ending_id)
    return {
        "sessions": len(artifacts),
        "beats_total": len(world.beats),
        "beats_fired": len(fired_beats & set(world.beats)),
        "beats_never_fired": sorted(set(world.beats) - fired_beats),
        "endings_total": len(world.endings),
        "endings_reached": sorted(reached_endings),
        "endings_never_reached": sorted(set(world.endings) - reached_endings),
    }
