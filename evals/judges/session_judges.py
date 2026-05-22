"""Whole-session judges — arc properties, one thing each.

These need the long arc (the alternative endings, the spread of scenes), so they
score a `SessionView` rather than a single turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from handler.world import World

from ..transitions import SessionView
from .base import AgentJudge
from .verdicts import PassFail, Scored


def _session_context(s: SessionView) -> str:
    case = "\n".join(f"  - {c}" for c in s.case_file) or "  (empty)"
    return (
        f"Persona: {s.persona}\n"
        f"Location path: {' -> '.join(s.location_path) or '(none)'}\n"
        f"Beats in order: {s.beats_in_order or '(none)'}\n"
        f"Ending reached: {s.ending_id or ('TRUNCATED' if s.truncated else 'none')}\n\n"
        f"Case file (the journalist's record):\n{case}"
    )


@dataclass
class EndingEarnedNarratively(AgentJudge):
    """Is the delivered ending coherent with what the player accomplished?
    Catches solved-it-but-got-`run_out_of_the_village`."""

    world: World
    name: ClassVar[str] = "ending_earned_narratively"
    output_type: ClassVar[type] = PassFail
    system_prompt: ClassVar[str] = (
        "You review whole playthroughs of a narrative investigation game. Judge ONE "
        "thing: is the ending the game delivered narratively coherent with what the "
        "player actually accomplished across the session? A player who gathered the "
        "key evidence but received a failure ending (e.g. 'run out of the village'), or "
        "who did nothing but received a triumphant ending, is INCOHERENT (passed=false)."
    )

    def render(self, s: SessionView) -> str:
        ending = self.world.endings.get(s.ending_id) if s.ending_id else None
        epilogue = ending.epilogue.strip() if ending else "(no ending fired)"
        return (
            f"{_session_context(s)}\n\n"
            f"Ending epilogue delivered:\n  {epilogue}\n\n"
            "Is this ending coherent with what the player accomplished?"
        )


@dataclass
class EmergentEventHonored(AgentJudge):
    """Off-script consequential events: carried forward (good freedom) vs.
    dropped / contradicting the truth (hallucination)."""

    world: World
    name: ClassVar[str] = "emergent_event_honored"
    output_type: ClassVar[type] = Scored
    system_prompt: ClassVar[str] = (
        "You review whole playthroughs for emergent-event continuity. Identify any "
        "consequential events in the case file that were NOT pre-authored beats (a "
        "death, a fire, an arrest, a confession the player provoked). Judge ONE thing: "
        "are such emergent events honored — carried forward consistently and not "
        "contradicting the established truth? 5 = emergent events (if any) are handled "
        "consistently; 3 = none occurred, or minor; 1 = an emergent event was dropped "
        "next scene or contradicts the truth. If nothing emergent happened, score 3."
    )

    def render(self, s: SessionView) -> str:
        return (
            f"Established truth (ground truth):\n  {self.world.crime.strip()}\n\n"
            f"{_session_context(s)}\n\n"
            "Are emergent (non-authored) events honored and consistent with the truth?"
        )


# NOTE: `choice_consequence_felt` was retired (master, May 2026). It scored "did the
# player's choices matter" by reading the whole session including the ending, so it
# halo'd off the outcome — structurally identical grind behaviour scored 5 when the
# run reached the arrest ending and 1 when it ran out. The objective half of that
# signal is now the deterministic `metrics.post_exhaustion_turn_ratio` (no ending to
# bias it); the cross-session half is `metrics.persona_outcome_separation`; and
# ending coherence is `EndingEarnedNarratively` above. See HANDLER eval doctrine:
# deterministic/structural facts belong in Tier-1 metrics, not an LLM vibe-judge.
