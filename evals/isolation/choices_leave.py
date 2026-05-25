"""Choice-author leave-option suite — the regression gate for the "step back
into the hall" bug.

The bug: at the Old Vicarage the choice author emitted a `leave: true` option
reading "Thank Mrs Hughes and step back into the hall" — an inward move to
another room, not leaving the location (picking it opens the map and the player
walks elsewhere). Root cause was a location-blind prompt whose only example
("step back into the lane") the model copied into an interior room.

This suite re-executes `emit_choices` with `leave_required=True` and asserts,
deterministically (no judge), that when a `leave: true` option is offered it
reads as the player LEAVING the current location:
  * exactly one option is marked `leave` (the MUST contract under leave_required;
    a bare `Leave` close is also acceptable — nothing misleading is shown);
  * its text carries a departure cue (out / leave / go / off / away / …); and
  * it names no OTHER location (the player picks the destination on the map, so
    the exit must not name where they go — "step out into the lane" fails).

Phase-2 runs at temp=1.0, so the harness samples K times and scores a pass-rate.
Two cases, deliberately split:
  * `leave_reads_as_exit_vicarage` GATES (floor 0.7). Solo at the original bug
    site — no NPC to "acknowledge", so the model reliably authors a clean exit
    ("step out of the Vicarage"); observed ~0.81–0.96. A reverted prompt
    (destination names / inward room moves) craters well below 0.7.
  * `leave_reads_as_exit_pub` is TREND-ONLY (assert_case=False). A spent NPC
    conversation is noisier — the model often flips `leave` onto a sitting-still
    "acknowledge his finality" option (a pre-existing 4B weakness, separate from
    the text bug this fix cures); we watch it without flaking the gate.

Requires a live engine, like the other isolation suites.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.agents import AgentBundle, emit_choices
from handler.models import Location
from handler.schemas import Continue
from handler.world import World

from .deltas import ctx
from .harness import IsoCase, Suite

# Generic words in location names that don't, on their own, name a place — kept
# out of the "forbidden destination token" set so an exit line like "step out
# into the black night" isn't mistaken for naming the Black Dog Bend. Each such
# location still has a distinctive token (bend, hafod, …) that IS forbidden.
_STOP = {"the", "old", "and", "of", "st", "into", "out", "back",
         "lower", "black", "dog", "new"}

# A departure reads as leaving: at least one of these cues must appear. The
# vocabulary is deliberately broad (door / excuse / goodnight / "on your way" /
# …) because the model phrases an exit many ways — "head for the door", "excuse
# yourself from the pub", "Goodnight, Tom". It still rejects the bug class: an
# inward room move ("step back into the hall") and a bare non-exit action ("let
# the silence hang") carry none of these cues, and naming a destination is
# caught separately by the forbidden-token check.
_EXIT_CUE = re.compile(
    r"\b(out|outside|off|away|leave|leaving|left|go|going|gone|depart\w*|exit|"
    r"door|excuse|farewell|bid)\b"
    r"|good\s?(night|bye)"
    r"|\b(on|make)\s+your\s+way\b",
    re.IGNORECASE,
)


def _name_tokens(loc: Location) -> set[str]:
    """Distinctive (≥3-char, non-generic) words from a location's name +
    short_name — the words that, in a leave option, would mean the author named
    that place as a destination."""
    raw = f"{loc.name} {loc.short_name or ''}".lower()
    return {t for t in re.findall(r"[a-z]{3,}", raw) if t not in _STOP}


def _forbidden_tokens(world: World, current_id: str) -> frozenset[str]:
    """Name tokens of every location EXCEPT the current one — a leave option
    naming any of these is naming a destination (the thing the player should
    pick on the map afterward, not have chosen for them)."""
    here = _name_tokens(world.locations[current_id])
    other: set[str] = set()
    for loc in world.locations.values():
        if loc.id != current_id:
            other |= _name_tokens(loc)
    return frozenset(other - here)


@dataclass
class LeaveOut:
    """Parsed choice-author output for the leave checks."""

    is_continue: bool
    options: list[tuple[str, bool]]  # (text, leave)
    forbidden: frozenset[str]


@dataclass
class LeaveOptionExitsLocation(Evaluator):
    """Gate: a `leave: true` option reads as leaving the current location.

    Encodes the prompt contract + the regression: under leave_required the
    author must offer exactly one exit (or close with a bare Leave), and that
    exit must carry a departure cue and name no other location. The original
    "step back into the hall" fails the cue check (no out/leave/go/…) — and any
    "step out into the lane" variant fails the destination check."""

    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        out: LeaveOut = ctx.output
        if not out.is_continue:
            # Bare Leave: a clean close, no leave OPTION shown — nothing to mislead.
            return True
        leaves = [text for (text, is_leave) in out.options if is_leave]
        if len(leaves) != 1:
            return False  # leave_required ⇒ exactly one exit option
        text = leaves[0].lower()
        if not _EXIT_CUE.search(text):
            return False  # doesn't read as leaving (the "into the hall" failure)
        toks = set(re.findall(r"[a-z]{3,}", text))
        if toks & out.forbidden:
            return False  # named another location — a destination
        return True


# Winding-down moments at two location types: an NPC conversation that has run
# its course (pub / Tom) and the solo lodgings where the original bug fired
# (the Vicarage). Both set leave_required so the author must offer the exit.
_PUB_SPENT = (
    "Tom drains the glass and sets it down with a small, final click. \"That's "
    "the lot,\" he says. \"You'll not get more out of me tonight.\" The fire has "
    "burned low."
)
_VICARAGE_SPENT = (
    "The study is quiet. You have read the letter twice and turned over the "
    "marked-up map until its creases are familiar. There is nothing else here "
    "tonight that will give anything up. Mrs Hughes has gone back to the kitchen."
)


def suite(bundle: AgentBundle, world: World) -> Suite:
    """Phase-2b leave option: when offered, does it read as leaving the current
    location (deterministic gate, no destination, no inward room move)?"""

    cases = [
        # Trend-only (assert_case=False): a spent NPC conversation is a harder,
        # noisier moment — the model often flips `leave` onto a sitting-still
        # "acknowledge his finality / accept the silence" option rather than an
        # authored departure (a pre-existing 4B behaviour, distinct from the text
        # bug this fix addresses; the original "into the hall" / destination-naming
        # class does not recur here). We WATCH the NPC-present exit quality without
        # letting its variance flake the gate; the vicarage case below gates.
        IsoCase(
            name="leave_reads_as_exit_pub",
            scenario="Tom has shut the conversation down — track whether the exit reads as physically leaving the pub (trend)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Press Tom one more time on the Land Rover.",
                last_prose=_PUB_SPENT,
                npc_at={"tom": "the_slaughtered_lamb"},
            ),
            prose=_PUB_SPENT,
            assert_case=False,
        ),
        IsoCase(
            name="leave_reads_as_exit_vicarage",
            scenario="Solo at the Vicarage (the original bug site) — the exit must read as leaving the Vicarage, not 'step back into the hall'",
            build=lambda w: ctx(
                w, location="the_old_vicarage",
                last_action="Set the map down and look around the study.",
                last_prose=_VICARAGE_SPENT,
            ),
            prose=_VICARAGE_SPENT,
            floor=0.7,
        ),
    ]
    by_name = {c.name: c for c in cases}

    async def task(case_name: str) -> LeaveOut:
        c = by_name[case_name]
        state = c.build(world)
        forbidden = _forbidden_tokens(world, state.player_location)
        try:
            bundle_out = await emit_choices(
                bundle, state, c.prose or "", leave_required=True
            )
            nxt = bundle_out.next
            if isinstance(nxt, Continue):
                return LeaveOut(
                    is_continue=True,
                    options=[(o.text, o.leave) for o in nxt.options],
                    forbidden=forbidden,
                )
            return LeaveOut(is_continue=False, options=[], forbidden=forbidden)
        except UnexpectedModelBehavior:
            # Retry-exhausted: no usable output → a Continue with no leave option,
            # which the gate scores as a MISS (not an excluded error).
            return LeaveOut(is_continue=True, options=[], forbidden=forbidden)

    return Suite(
        key="choices_leave",
        title="Phase-2b leave option (reads as leaving the location / case)",
        cases=cases,
        task=task,
        evaluators=[LeaveOptionExitsLocation()],
        show_outputs=True,
    )
