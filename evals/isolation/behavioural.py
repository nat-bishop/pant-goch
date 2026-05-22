"""Behavioural exemplar suites — a couple of high-value evals per Phase-2 agent.

Where the deltas suite asserts deterministic design rules, these score genuinely
behavioural properties (voice, choice agency, fact capture) by **re-executing** the
agent in isolation and judging the *fresh* output. The reused `evals.judges` classes
score the fresh `Transition` exactly as they score recorded ones — so a behavioural
rubric here is the same instrument as in `judge_run`, but driven by a frozen input we
control (and so usable to iterate a prompt, which a recorded judge case cannot).

Per the EVALS.md split: the LLM judges are Scored → they **trend** in summary.json,
they do not gate. The cheap deterministic liveness/dedup checks are bool assertions →
they **gate**. These three suites are intentionally small (1–2 cases each) — exemplars
and templates, not an exhaustive rubric set; grow them reactively from real failures.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.agents import (
    AgentBundle,
    emit_chronicle_entry,
    emit_choices,
    make_scene_runner_runtime,
    stream_prose_text,
)
from handler.schemas import ChronicleEntry, Continue
from handler.world import World

from ..judges.transition_judges import ChoiceMeaningfulness, NpcVoiceAdherence
from ..transitions import Transition
from .deltas import ctx
from .harness import IsoCase, Suite


def _transition(*, location: str, player_action: str, prose: str,
                options: list[str], prior_case_file: list[str]) -> Transition:
    """A minimal `Transition` wrapping fresh agent output for the reused judges.
    Only `location`, `prose`, `options` are read by the judges here; the rest are
    filled with honest placeholders."""
    return Transition(
        session_id="iso", persona="iso", scene_id=3, location=location,
        prior_case_file=prior_case_file, player_action=player_action,
        prose=prose, options=options, fired_beats=[], fireable_beats=[],
    )


# ───────────────────────── deterministic gates ─────────────────────────


@dataclass
class ProseNonEmpty(Evaluator):
    """Liveness gate: the prose writer produced non-empty prose (not the empty
    fallback). Voice quality is the Scored judge's job, not this gate's."""

    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        t: Transition = ctx.output
        return bool(t.prose.strip())


@dataclass
class ChronicleEntryValid(Evaluator):
    """Gate: the chronicler wrote a non-empty case-file entry that does NOT duplicate
    a prior entry verbatim (the NoDuplicateCaseFiles invariant, at the agent boundary).
    Prior entries ride on `ctx.metadata`."""

    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        prior: tuple[str, ...] = ctx.metadata or ()
        entry: ChronicleEntry = ctx.output
        text = entry.case_file_entry.strip()
        return bool(text) and text not in set(prior)


# ───────────────────────── prose-writer suite ─────────────────────────

_TOM_PRESENT = {"tom": "the_slaughtered_lamb"}


def prose_suite(bundle: AgentBundle, world: World, judge_model: Model) -> Suite:
    """Phase-2a prose writer: does fresh prose carry the resident NPC's authored
    voice? `npc_voice_adherence` (Scored, trend) + a non-empty liveness gate."""

    cases = [
        IsoCase(
            name="tom_in_voice",
            scenario="Mid-scene prose for guarded Tom at the pub — should sound like Tom",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Ask Tom what he remembers about the night Megan died.",
                last_prose="The bar is quiet. Tom is wiping the same glass he was wiping when you came in.",
                npc_at=_TOM_PRESENT,
            ),
            floor=0.8,  # liveness only; ~always passes unless the writer errors
        ),
    ]
    by_name = {c.name: c for c in cases}

    async def task(case_name: str) -> Transition:
        c = by_name[case_name]
        state = c.build(world)
        runtime = make_scene_runner_runtime()
        # A retry-exhausted writer (UnexpectedModelBehavior) is the production
        # fallback path; score it as empty prose (a liveness MISS the gate sees),
        # not an excluded error — mirrors deltas.py's empty-bundle fallback.
        parts: list[str] = []
        try:
            async for chunk in stream_prose_text(bundle, runtime, state, []):
                parts.append(chunk)
        except UnexpectedModelBehavior:
            parts = []
        return _transition(
            location=state.player_location,
            player_action=state.dialogue_history[-1].player_pick.text,
            prose="".join(parts), options=[],
            prior_case_file=list(state.case_file_entries),
        )

    return Suite(
        key="prose_voice",
        title="Phase-2a prose (npc voice / case)",
        cases=cases,
        task=task,
        evaluators=[NpcVoiceAdherence(judge_model, world), ProseNonEmpty()],
    )


# ───────────────────────── choice-author suite ─────────────────────────

_SPENT_SCENE_PROSE = (
    "Tom has told you what he's going to tell you, which isn't much. He drains the "
    "glass and sets it down with a small, final click. \"That's the lot,\" he says. "
    "\"You'll not get more out of me tonight.\" The fire has burned low. Behind you, "
    "the door lets in a draught and the smell of the wet road."
)


def choices_suite(bundle: AgentBundle, world: World, judge_model: Model) -> Suite:
    """Phase-2b choice author: are the options meaningful decisions or cosmetic
    filler ("examine the pamphlets")? `choice_meaningfulness` (Scored, trend). No
    deterministic gate — option count/shape is already schema-enforced; agency is the
    behavioural signal here."""

    cases = [
        IsoCase(
            name="spent_scene_agency",
            scenario="Tom has shut the conversation down — options should offer a real next move, not scenery-poking",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Press Tom on what he saw.",
                last_prose=_SPENT_SCENE_PROSE,
                npc_at=_TOM_PRESENT,
            ),
            prose=_SPENT_SCENE_PROSE,
            assert_case=False,  # judge-only → trend, never gates
        ),
    ]
    by_name = {c.name: c for c in cases}

    async def task(case_name: str) -> Transition:
        c = by_name[case_name]
        state = c.build(world)
        try:
            bundle_out = await emit_choices(bundle, state, c.prose or "")
            nxt = bundle_out.next
            options = [o.text for o in nxt.options] if isinstance(nxt, Continue) else []
        except UnexpectedModelBehavior:
            options = []  # scored low by the meaningfulness judge, not excluded
        return _transition(
            location=state.player_location,
            player_action=state.dialogue_history[-1].player_pick.text,
            prose=c.prose or "", options=options,
            prior_case_file=list(state.case_file_entries),
        )

    return Suite(
        key="choice_agency",
        title="Phase-2b choices (meaningfulness / case)",
        cases=cases,
        task=task,
        evaluators=[ChoiceMeaningfulness(judge_model)],
    )


# ───────────────────────── chronicler suite ─────────────────────────


def chronicle_suite(bundle: AgentBundle, world: World) -> Suite:
    """Chronicler: a non-empty case-file entry that doesn't duplicate a prior one.
    Deterministic (opinion-free) → gates. The prior entries are passed both into the
    frozen state (so the chronicler sees its accumulating record) and as `meta` (so
    the dedup evaluator can compare)."""

    _PRIOR = (
        "Spoke to Maud at the post office. She was guarded about the Pritchard girl "
        "but let slip that Megan had been 'asking after the Vaughns' before she died.",
    )

    cases = [
        IsoCase(
            name="no_duplicate_entry",
            scenario="A second scene at the pub — the new entry must not restate the post-office entry verbatim",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Ask Tom about the Vaughns.",
                last_prose=(
                    "Tom's jaw tightens at the name. \"The Vaughns keep themselves to "
                    "themselves,\" he says, \"and you'd do well to do the same.\" But his "
                    "eyes flick, just once, toward the road that leads up to the farm."
                ),
                case_file=_PRIOR,
                npc_at=_TOM_PRESENT,
            ),
            meta=_PRIOR,
            floor=0.8,
        ),
    ]
    by_name = {c.name: c for c in cases}

    async def task(case_name: str) -> ChronicleEntry:
        state = by_name[case_name].build(world)
        try:
            return await emit_chronicle_entry(bundle, world, state)
        except UnexpectedModelBehavior:
            # Empty entry → ChronicleEntryValid fails (a gated MISS), not an excluded error.
            return ChronicleEntry(case_file_entry="")

    return Suite(
        key="chronicle_dedup",
        title="Chronicler (valid + non-dup / case)",
        cases=cases,
        task=task,
        evaluators=[ChronicleEntryValid()],
    )
