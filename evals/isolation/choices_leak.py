"""Choice-author leak + turning-point suite — the regression gate for the
"Turn to face Dylan Vaughn" bug.

The bug: at the churchyard the choice author offered an option to turn to Dylan
Vaughn on a turn whose prose described only Megan's grave — Dylan was present in
engine state (seeded by `seed_present_npcs`) but never on the page. The choice
author learned he was there from a `NPCs present:` line / a `## Worth Pursuing
Here` steer, NOT from the prose, and surfaced him as an option the player had to
PICK to discover. Root cause: the choice author saw world facts the prose had
not established.

The fix makes the choice author PROSE-ONLY — it sees the just-written prose, the
player's own prior picks, pacing, and the location name, and nothing else. This
suite asserts the leak cannot recur, deterministically (no judge):

  * `leak_unintroduced_present_npc` GATES (floor 0.9). Dylan is present at the
    churchyard but the prose names only the grave; NO option may name a present
    NPC the prose did not introduce. A reverted prompt (NPC list / worth-pursuing
    steer) lets the model surface him and craters this case.

  * `turning_point_offers_commitment_with_flag` / `..._without_flag` are
    TREND-ONLY (assert_case=False). Charged turning-point prose, run with the
    non-naming `turning_point` flag on vs. off — we WATCH whether a resolve-to-act
    option appears, to settle empirically whether the flag earns its place (the
    reachability belt) or charged prose alone suffices. Neither gates.

Requires a live engine, like the other isolation suites.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.agents import AgentBundle, emit_choices
from handler.engine import present_npc_ids
from handler.schemas import Continue
from handler.world import World

from .deltas import ctx
from .harness import IsoCase, Suite

# An option that resolves to act reads with one of these — a commitment cue, as
# distinct from an exploratory question. Deliberately broad (the model phrases a
# resolution many ways) but excludes plain "ask/look/examine" exploration.
_COMMIT_CUE = re.compile(
    r"\b(tell|confront|accuse|resolve|decide|done|enough|face|finish|write|"
    r"publish|file|admit|demand|insist|name|press\s+\w+\s+on|put\s+it\s+to)\b"
    r"|it'?s\s+time|you'?re\s+done|no\s+more",
    re.IGNORECASE,
)


def _npc_name_tokens(name: str) -> set[str]:
    """Distinctive (≥3-char) words of an NPC's name — the tokens that, in an
    option, would mean the author named that person."""
    return {t for t in re.findall(r"[a-z]{3,}", name.lower())}


@dataclass
class ChoiceOut:
    """Parsed choice-author output for the leak / turning-point checks."""

    is_continue: bool
    options: list[tuple[str, bool]]  # (text, leave)
    forbidden: frozenset[str]        # name tokens of present-but-unintroduced NPCs
    commit_cued: bool                # any option reads as resolve-to-act


@dataclass
class NoUnintroducedNpcNamed(Evaluator):
    """Gate: no option names a present NPC the prose did not introduce. The
    forbidden set is built in the task from the present NPCs whose name does not
    appear in the fixture prose — exactly the leak class ('Turn to face Dylan
    Vaughn' over grave-only prose)."""

    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        out: ChoiceOut = ctx.output
        if not out.forbidden:
            return True  # nothing to leak (defensive; the case sets it)
        for text, _is_leave in out.options:
            toks = set(re.findall(r"[a-z]{3,}", text.lower()))
            if toks & out.forbidden:
                return False
        return True


@dataclass
class OffersResolveToAct(Evaluator):
    """Trend: did a resolve-to-act option appear? Scored, never gated — the two
    turning-point cases (flag on/off) are compared in summary.json to decide
    whether the non-naming flag earns its place."""

    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> float:
        out: ChoiceOut = ctx.output
        return 1.0 if out.commit_cued else 0.0


# Grave-only prose: Dylan is present in state (seeded at the churchyard) but the
# narrator wrote only the headstone and flowers — the exact frame of the bug.
_GRAVE_PROSE = (
    "You walk to the headstone. It is small, polished, shockingly tidy against "
    "the lichen-stained granite of the older markers. Megan Mair Pritchard, the "
    "name reads, and below the dates: beloved daughter. At the base, a spray of "
    "fresh flowers — a splash of colour against the muted greys."
)

# Charged turning-point prose: Eluned has all but broken; the moment is poised
# for the player to commit. Used for both flag-on and flag-off arms.
_CHARGED_PROSE = (
    "Eluned's hand stays flat on the table, but it is shaking now, and she does "
    "not trust herself to lift it. \"You don't know,\" she says, and it is not a "
    "denial any more. \"You don't know what it is to watch your own boy carry a "
    "thing like that.\" The kitchen has gone very still. She is waiting for what "
    "you will do."
)


def suite(bundle: AgentBundle, world: World) -> Suite:
    """Phase-2b choice author: it cannot name a present NPC the prose did not
    introduce (gate), and its resolve-to-act rate with/without the turning-point
    flag (trend)."""

    cases = [
        IsoCase(
            name="leak_unintroduced_present_npc",
            scenario="Dylan present at the churchyard, prose names only the grave — no option may name him",
            build=lambda w: ctx(
                w, location="st_caradocs",
                last_action="Walk closer to Megan's headstone.",
                last_prose=_GRAVE_PROSE,
                npc_at={"dylan": "st_caradocs"},
            ),
            prose=_GRAVE_PROSE,
            floor=0.9,
        ),
        IsoCase(
            name="turning_point_offers_commitment_with_flag",
            scenario="Charged prose + turning_point flag ON — does a resolve-to-act option appear? (trend)",
            build=lambda w: ctx(
                w, location="tan_y_coed",
                last_action="Hold her gaze and say nothing.",
                last_prose=_CHARGED_PROSE,
                npc_at={"eluned": "tan_y_coed"},
            ),
            prose=_CHARGED_PROSE,
            assert_case=False,
        ),
        IsoCase(
            name="turning_point_offers_commitment_without_flag",
            scenario="Same charged prose, turning_point flag OFF — control for whether prose alone suffices (trend)",
            build=lambda w: ctx(
                w, location="tan_y_coed",
                last_action="Hold her gaze and say nothing.",
                last_prose=_CHARGED_PROSE,
                npc_at={"eluned": "tan_y_coed"},
            ),
            prose=_CHARGED_PROSE,
            assert_case=False,
        ),
    ]
    by_name = {c.name: c for c in cases}
    flag_on = {"turning_point_offers_commitment_with_flag"}

    async def task(case_name: str) -> ChoiceOut:
        c = by_name[case_name]
        state = c.build(world)
        # Forbidden tokens: present NPCs whose name the prose did NOT introduce.
        prose_l = (c.prose or "").lower()
        forbidden: set[str] = set()
        for nid in present_npc_ids(world, state):
            toks = _npc_name_tokens(world.npcs[nid].name)
            if not any(t in prose_l for t in toks):
                forbidden |= toks
        try:
            bundle_out = await emit_choices(
                bundle, state, c.prose or "",
                turning_point=case_name in flag_on,
            )
            nxt = bundle_out.next
            if isinstance(nxt, Continue):
                opts = [(o.text, o.leave) for o in nxt.options]
                return ChoiceOut(
                    is_continue=True, options=opts,
                    forbidden=frozenset(forbidden),
                    commit_cued=any(_COMMIT_CUE.search(t) for t, _ in opts),
                )
            return ChoiceOut(
                is_continue=False, options=[], forbidden=frozenset(forbidden),
                commit_cued=False,
            )
        except UnexpectedModelBehavior:
            # Retry-exhausted: no usable output → leak gate passes vacuously (no
            # options to leak), commit trend scores a miss.
            return ChoiceOut(
                is_continue=True, options=[], forbidden=frozenset(forbidden),
                commit_cued=False,
            )

    return Suite(
        key="choices_leak",
        title="Phase-2b choice author (no unintroduced-NPC leak / turning-point trend)",
        cases=cases,
        task=task,
        evaluators=[NoUnintroducedNpcNamed(), OffersResolveToAct()],
        show_outputs=True,
    )
