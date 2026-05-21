"""Single-transition judges — one before→after turn each, one thing each.

These isolate engine correctness at the smallest decidable scope and are the
regression-test surface (freeze a failing transition as a case). Each renders the
`Transition` plus the relevant authored ground truth (the beat's notes, the
location's description, the cast's voice) so the judge scores against the bible,
not vibes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from handler.world import World

from ..transitions import Transition
from .base import AgentJudge
from .verdicts import PassFail, Scored


def _beat_brief(world: World, beat_id: str | None) -> str:
    beat = world.beats.get(beat_id) if beat_id else None
    if beat is None:
        return f"(unknown beat {beat_id!r})"
    return (
        f"Beat '{beat.name}' (id={beat.id})\n"
        f"  description: {beat.description.strip()}\n"
        f"  prerequisites: {beat.prerequisites.strip() or '(none)'}\n"
        f"  notes_to_llm: {beat.notes_to_llm.strip()}"
    )


def _transition_context(t: Transition) -> str:
    prior = "\n".join(f"  - {c}" for c in t.prior_case_file) or "  (nothing yet)"
    return (
        f"Location: {t.location}\n"
        f"Case file so far:\n{prior}\n\n"
        f"Player's action this turn:\n  {t.player_action!r}\n\n"
        f"Narrator prose this turn:\n  {t.prose.strip()}"
    )


@dataclass
class BeatFiredAppropriately(AgentJudge):
    """Did the player's action THIS turn warrant firing the beat of interest?
    The walk-away-false-fire / Maud-dump / eager-ladder / helen-without-meeting
    detector."""

    world: World
    name: ClassVar[str] = "beat_fired_appropriately"
    output_type: ClassVar[type] = PassFail
    system_prompt: ClassVar[str] = (
        "You audit a narrative game engine. A 'beat' is a story milestone the engine "
        "may fire only when the player's action has genuinely earned it, per the beat's "
        "notes. You are given ONE turn: the prior case file, the player's action this "
        "turn, the narrator's prose, and the beat that fired. Judge ONE thing: did the "
        "player's action THIS turn legitimately warrant firing THIS beat, honoring its "
        "notes_to_llm and prerequisites? Fire-too-early, fire-on-mere-navigation, or "
        "fire-without-the-action-described = NOT warranted. Answer passed=true only if "
        "the firing is clearly earned by the action shown."
    )

    def render(self, t: Transition) -> str:
        return (
            f"{_beat_brief(self.world, t.beat_of_interest)}\n\n"
            f"{_transition_context(t)}\n\n"
            "Was firing this beat warranted by the player's action this turn?"
        )


@dataclass
class ProseDepictsBeat(AgentJudge):
    """Does the prose actually depict the event the beat describes? Catches the
    phoned-Iwan hallucination (beat + case-file narrate an event the prose never
    shows)."""

    world: World
    name: ClassVar[str] = "prose_depicts_beat"
    output_type: ClassVar[type] = PassFail
    system_prompt: ClassVar[str] = (
        "You audit a narrative game engine. A beat fired this turn, which means its "
        "event should be visible in the narrator's prose for this turn (or the player's "
        "action that prompted it). Judge ONE thing: does the prose/action actually "
        "depict the event the beat describes? If the beat claims an event (a phone call, "
        "a confession, an examination) that does not appear in the text, passed=false."
    )

    def render(self, t: Transition) -> str:
        return (
            f"{_beat_brief(self.world, t.beat_of_interest)}\n\n"
            f"{_transition_context(t)}\n\n"
            "Does the prose/action actually depict the beat's event?"
        )


@dataclass
class ProseMatchesLocation(AgentJudge):
    """Does the prose match the authored physical setting of the location?"""

    world: World
    name: ClassVar[str] = "prose_matches_location"
    output_type: ClassVar[type] = Scored
    system_prompt: ClassVar[str] = (
        "You check setting consistency in a narrative game. Judge ONE thing: how well "
        "the narrator's prose matches the authored description of the location the "
        "player is in. 5 = vividly and accurately in this place; 3 = generic but not "
        "contradictory; 1 = describes a different kind of place than authored."
    )

    def render(self, t: Transition) -> str:
        loc = self.world.locations.get(t.location)
        desc = loc.description.strip() if loc else f"(unknown location {t.location!r})"
        return (
            f"Authored location '{t.location}':\n  {desc}\n\n"
            f"Narrator prose:\n  {t.prose.strip()}\n\n"
            "How well does the prose match this location?"
        )


@dataclass
class NpcVoiceAdherence(AgentJudge):
    """Does NPC dialogue in the prose match the cast's authored voice?"""

    world: World
    name: ClassVar[str] = "npc_voice_adherence"
    output_type: ClassVar[type] = Scored
    system_prompt: ClassVar[str] = (
        "You check character-voice consistency in a narrative game. Judge ONE thing: "
        "does any character dialogue in the prose sound like the authored voice of the "
        "characters who belong in this place (their sample lines and attitude)? 5 = "
        "unmistakably in voice; 3 = no dialogue, or neutral; 1 = clearly off-voice."
    )

    def render(self, t: Transition) -> str:
        here = [
            n for n in self.world.npcs.values() if n.default_location == t.location
        ]
        cards = "\n".join(
            f"  {n.name}: attitude={n.attitude.strip()} | samples={n.voice_sample}"
            for n in here
        ) or "  (no resident characters)"
        return (
            f"Characters who belong at '{t.location}':\n{cards}\n\n"
            f"Narrator prose:\n  {t.prose.strip()}\n\n"
            "How well does any dialogue match these characters' voice?"
        )


@dataclass
class ChoiceMeaningfulness(AgentJudge):
    """Are the options meaningful choices or cosmetic filler ('look at the box')?
    The agency judge."""

    name: ClassVar[str] = "choice_meaningfulness"
    output_type: ClassVar[type] = Scored
    system_prompt: ClassVar[str] = (
        "You evaluate player agency in a narrative investigation game. Judge ONE thing: "
        "are the options offered this turn meaningful choices — distinct intents that "
        "could plausibly steer the investigation or carry a social cost (press vs. "
        "soften, pursue a lead vs. another) — or cosmetic filler that re-examines "
        "scenery and changes nothing ('look closer at the box', 'examine the window')? "
        "5 = every option is a real, divergent decision; 3 = mixed; 1 = all cosmetic "
        "busywork."
    )

    def render(self, t: Transition) -> str:
        opts = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(t.options)) or "  (none)"
        return (
            f"Narrator prose:\n  {t.prose.strip()}\n\n"
            f"Options offered:\n{opts}\n\n"
            "How meaningful are these choices?"
        )


@dataclass
class FalsePremiseAccepted(AgentJudge):
    """Did the narrator accept a false premise the player introduced (free text)?"""

    world: World
    name: ClassVar[str] = "false_premise_accepted"
    output_type: ClassVar[type] = PassFail
    system_prompt: ClassVar[str] = (
        "You audit robustness of a narrative game's narrator. The player can type "
        "free-form input that may assert things that are not true in the world. Judge "
        "ONE thing: did the narrator's prose accept and build on a FALSE premise the "
        "player introduced (inventing corroborating detail for something that didn't "
        "happen)? Use the established truth as ground truth. passed=true means the "
        "narrator did NOT get fooled (it deflected or stayed neutral); passed=false "
        "means it accepted the false premise."
    )

    def render(self, t: Transition) -> str:
        return (
            f"Established truth (ground truth):\n  {self.world.crime.strip()}\n\n"
            f"Player's typed input:\n  {t.player_action!r}\n\n"
            f"Narrator prose in response:\n  {t.prose.strip()}\n\n"
            "Did the narrator avoid accepting a false premise?"
        )
