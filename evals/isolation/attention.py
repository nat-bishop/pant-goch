"""Village-attention isolation suite — the dedicated scene-close heat classifier.

Where the deltas suite covers per-NPC trust/guard, this covers the global
`village_attention` signal, which now has its own agent (`emit_village_attention`)
rather than being a Phase-1 bundle field. The split was made *because* the bundled
field collapsed to 0 on private aggression (the model anchored on the per-NPC
delta); asked in isolation the model classifies village suspicion reliably, which
these fixtures lock in.

Fixtures are multi-turn transcripts (the agent reads a whole scene), unlike the
single-turn deltas `ctx`. The decisive contrast is `gentle` (must stay LOW —
proves the classifier isn't just "always suspicious") vs `escalating_private` /
`single_private_threat` (sustained/whispered aggression must register) vs
`public_accusation` (the unambiguous high control).

Scoring: an ordinal assertion (`at_least` / `at_most` a level), gated by a floor —
the same doctrine as the deltas suite. Calibrate floors against the first live run.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.agents import AgentBundle, emit_village_attention
from handler.engine import ATTENTION_LEVEL_VALUES
from handler.schemas import Option
from handler.state import State, TurnRecord, make_initial_state
from handler.world import World

from .harness import IsoCase, Suite


# ───────────────────────── fixture builder ─────────────────────────


def _scene(
    world: World, *, location: str, turns: list[tuple[str, str]], prior: int = 0
) -> State:
    """Build a frozen `State` whose `dialogue_history` is a whole scene — each
    `(narration, player_action)` becomes one TurnRecord (player pick rendered as
    the journalist's verbatim words). `prior` seeds `village_attention` (the agent
    reads it as the prior level)."""
    st = make_initial_state(world)
    st.player_location = location
    st.scene_id = 3
    st.village_attention = prior
    for narration, action in turns:
        st.dialogue_history.append(
            TurnRecord(
                prose=narration,
                applied_state_changes=[],
                player_pick=Option(text=action),
                player_pick_typed=True,
            )
        )
    return st


# ───────────────────────── expectation + evaluator ─────────────────────────


@dataclass(frozen=True)
class AttnExpect:
    """Ordinal pass condition. `at_least` / `at_most` name a level word; the
    emitted level must sit within [at_least, at_most] (either bound optional)."""

    at_least: str | None = None
    at_most: str | None = None

    def holds(self, level: str) -> bool:
        v = ATTENTION_LEVEL_VALUES.get(level)
        if v is None:
            return False  # fallback / garbage → MISS
        if self.at_least is not None and v < ATTENTION_LEVEL_VALUES[self.at_least]:
            return False
        if self.at_most is not None and v > ATTENTION_LEVEL_VALUES[self.at_most]:
            return False
        return True


@dataclass
class AttentionExpectation(Evaluator):
    """Bool assertion: the emitted level satisfied the case's `AttnExpect`
    (on `ctx.metadata`). The emitted level word is `ctx.output`."""

    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        expect: AttnExpect = ctx.metadata
        return expect.holds(ctx.output)


# ───────────────────────── fixtures ─────────────────────────

_PUB = "the_slaughtered_lamb"

_GENTLE = [
    ("Tom nurses his bitter, not looking up.",
     "Ask him, politely, how long he's lived in Pant Goch."),
    ('"All my life," he says. "Quiet place."',
     "Say it seems a lovely village, and ask what it's like in winter."),
    ("He shrugs, warming a little to the small talk.",
     "Gently mention you're a journalist looking into Megan's death, and ask if he knew her."),
    ('"Aye. Knew her. Good girl." He goes quiet.',
     "Nod, and say you don't want to intrude — just to understand what happened."),
    ("He turns his glass slowly, considering you.",
     "Ask, kindly, whether he'd be willing to tell you about that night sometime."),
]

_ESCALATING = [
    ("Tom nurses his bitter, not looking up.",
     "Demand he tell you what really happened to Megan."),
    ('"Not my business," he mutters, shoulders tightening.',
     "Slam your hand on the bar and tell him to stop hiding behind that."),
    ("Eira looks up sharply from the far end of the bar. Tom stiffens.",
     "Lean in close and say you know he covered something up, and you'll prove it."),
    ('"You should go," Tom says, low. The pub has gone quiet around you.',
     "Refuse to leave; tell him you're not going anywhere until he talks."),
    ("He grips the bar, jaw set, the other drinkers watching.",
     "Tell him flatly that if he won't talk, you'll print that he helped cover up a girl's death."),
]

_SINGLE_THREAT = [
    ("Tom nurses his bitter, guarded.",
     "Lean in close and tell Tom quietly, so no one else can hear, that if he won't "
     "talk you'll print that he helped cover up a girl's death."),
]

_PUBLIC = [
    ("The pub is busy, a low murmur of locals.",
     "Stand up and announce to the whole room that Eluned Vaughn ran Megan off the road "
     "and the village is covering it up."),
]

# CONTROLS added after the corpus showed non-aggressive personas (committed_closer,
# distracted_skimmer) wrongly reaching `hostile` → `run_out_of_the_village`, which
# capped ending diversity. A firm-but-civil reporter and a low-effort one must NOT
# be run out of the village: persistence and disengagement are not spectacle.

_PERSISTENT_CIVIL = [
    ("Tom nurses his bitter, guarded.",
     "Ask him directly what he saw on the road the night Megan died."),
    ('"Not my business," he says, looking away.',
     "Press, politely but firmly — say you think it very much was, and you'd like the truth."),
    ("He sighs, turning his glass slowly.",
     "Ask again, steadily and without heat, whether he saw a vehicle that night."),
    ('"Folks round here don\'t talk about it," he murmurs.',
     "Say you understand, but you won't leave the story half-told — ask, calmly, who would know."),
    ("He studies you a long moment, weighing it.",
     "Thank him for his time and ask if you might come back tomorrow to talk properly."),
]

_LOW_EFFORT = [
    ("Tom nurses his bitter.",
     "Ask, a little distractedly, whether the beer here is any good."),
    ('"It\'s beer," he says.',
     "Glance at your phone, then ask what there is to do around the village."),
    ("He shrugs.",
     "Half-listening, ask if anything interesting ever happens in a place like this."),
    ('"Quiet, mostly," he says.',
     "Ask vaguely whether he knew that girl who died — what was her name again."),
]


def _cases() -> list[IsoCase]:
    return [
        # CONTROL: courteous, persistent inquiry must stay LOW (else the classifier
        # is useless). The probe scored this `none` 12/12 → at_most noticed w/ margin.
        IsoCase(
            name="gentle_5turn",
            scenario="Five turns of warm, courteous questions — the village has no reason to close ranks",
            build=lambda w: _scene(w, location=_PUB, turns=_GENTLE),
            meta=AttnExpect(at_most="noticed"),
            floor=0.8,
        ),
        # The Track-2 target: sustained PRIVATE aggression should register as the
        # village growing wary — the case the bundled field scored 0 on.
        IsoCase(
            name="escalating_private",
            scenario="Five turns ramping from demand to threat — the village turns",
            build=lambda w: _scene(w, location=_PUB, turns=_ESCALATING),
            meta=AttnExpect(at_least="talked_about"),
            floor=0.5,
        ),
        IsoCase(
            name="single_private_threat",
            scenario="One whispered threat with no audience — word still travels",
            build=lambda w: _scene(w, location=_PUB, turns=_SINGLE_THREAT),
            meta=AttnExpect(at_least="talked_about"),
            floor=0.5,
        ),
        # CONTROL: an unambiguous public spectacle is the clear high end.
        IsoCase(
            name="public_accusation",
            scenario="Stand and accuse the whole pub — the village wants you gone",
            build=lambda w: _scene(w, location=_PUB, turns=_PUBLIC),
            meta=AttnExpect(at_least="hostile"),
            floor=0.5,
        ),
        # CONTROL: a firm, persistent-but-civil reporter must NOT be ejected —
        # `hostile` is the run-out trigger, and the corpus over-ejected this persona.
        IsoCase(
            name="persistent_civil",
            scenario="Five turns of firm, direct, but courteous pressing — persistence is not spectacle",
            build=lambda w: _scene(w, location=_PUB, turns=_PERSISTENT_CIVIL),
            meta=AttnExpect(at_most="talked_about"),
            floor=0.8,
        ),
        # CONTROL: low-effort, disengaged questioning gives the village nothing to
        # close ranks over — must stay LOW.
        IsoCase(
            name="low_effort_skim",
            scenario="Four turns of distracted, shallow small-talk — no threat, no spectacle",
            build=lambda w: _scene(w, location=_PUB, turns=_LOW_EFFORT),
            meta=AttnExpect(at_most="noticed"),
            floor=0.8,
        ),
    ]


def _make_task(bundle: AgentBundle, world: World, by_name: dict[str, IsoCase]):
    async def task(case_name: str) -> str:
        """Run the village-attention agent once on the frozen scene; return the
        emitted level word. A retry-exhausted run becomes `"none"` (a gated MISS
        on the positive controls), mirroring the production fallback."""
        state = by_name[case_name].build(world)
        try:
            return await emit_village_attention(bundle, world, state)
        except UnexpectedModelBehavior:
            return "none"

    return task


def suite(bundle: AgentBundle, world: World) -> Suite:
    cs = _cases()
    return Suite(
        key="village_attention",
        title="Village attention (level / case)",
        cases=cs,
        task=_make_task(bundle, world, {c.name: c for c in cs}),
        evaluators=[AttentionExpectation()],
        show_outputs=True,
    )
