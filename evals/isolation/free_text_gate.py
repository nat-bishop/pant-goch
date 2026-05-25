"""Free-text gate isolation suite — the per-turn "offer the player their own
words?" judgment.

Free text is the "words to a person" channel: the engine only schedules this
agent when an NPC is present (solo scenes are click-driven, enforced in
`scene_loop.run_turn` — not tested here), so every case is conversational. The
gate's job is to read the just-written prose and decide whether it has handed
the conversational floor to the player — a question put to them, a challenge, a
charged silence awaiting their response — versus routine connective dialogue, a
moment about an object, or a scene winding down.

The gate sees ONLY the prose (`build_free_text_gate_user_message` ignores state),
so each case is a fixture `prose` string + an expected offer bool. The decisive
contrasts: the three floor-handing positives (offer) vs. the object-in-the-room
negative (`examining_object` — the regression that produced the "Examine the OS
Map" free-text box) and the routine / winding-down negatives.

Scoring: a bool assertion (offer == expected), gated by a floor — the same
doctrine as the deltas / attention suites. Calibrate floors against the first
live run.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.agents import AgentBundle, emit_free_text_gate
from handler.state import State, make_initial_state
from handler.world import World

from .harness import IsoCase, Suite


# ───────────────────────── fixture builder ─────────────────────────

_PUB = "the_slaughtered_lamb"


def _conversation(world: World) -> State:
    """A frozen `State` mid-conversation at the pub. The gate ignores state
    (it reads only the prose), so this just supplies a valid, NPC-present
    context — every gate case is conversational by construction."""
    st = make_initial_state(world)
    st.player_location = _PUB
    st.scene_id = 3
    return st


# ───────────────────────── expectation + evaluator ─────────────────────────


@dataclass(frozen=True)
class GateExpect:
    """The gate's expected decision for a case: should free text be offered?"""

    should_offer: bool

    def holds(self, offer: bool) -> bool:
        return offer == self.should_offer


@dataclass
class GateExpectation(Evaluator):
    """Bool assertion: the emitted offer matched the case's `GateExpect`
    (on `ctx.metadata`). The emitted bool is `ctx.output`."""

    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        expect: GateExpect = ctx.metadata
        return expect.holds(ctx.output)


# ───────────────────────── fixture prose ─────────────────────────

# OFFER: the person puts an open question directly to the player and waits.
_DIRECT_QUESTION = (
    "Tom sets the glass down and finally looks at you. \"You've been asking "
    "about Megan all round the village,\" he says. \"What is it you're really "
    "after?\" He waits, not unkindly, for an answer."
)

# OFFER: the person turns it back on the player — a challenge where the exact
# words of the reply are the substance.
_CHALLENGE = (
    "\"And why's it your business, exactly?\" Maud's smile has gone thin. She "
    "folds her hands on the counter and doesn't look away. \"Come all this way "
    "for a girl you never met. Why?\""
)

# OFFER: a charged silence after the player said something hard — the room is
# waiting on what the player does next.
_CHARGED_SILENCE = (
    "The words land and Eluned says nothing. She just looks at you across the "
    "kitchen table, the tea going cold between you, the clock loud in the quiet. "
    "She is waiting for you to go on, or to take it back."
)

# NO OFFER: routine connective tissue — a flat factual reply, any phrasing of
# the player's next prompt lands the same.
_ROUTINE = (
    "\"Post goes at half four,\" Maud says, not looking up from the franking "
    "machine. \"Always has. You'll catch it if you're quick.\" The bell over "
    "the door jingles as someone leaves."
)

# NO OFFER: the scene is plainly winding down — a goodbye, the person turning
# back to the work.
_WINDING_DOWN = (
    "\"That's all I've got for you,\" Tom says, and drains the last of his "
    "bitter. He nods once, then turns back to the racing page, done with you "
    "for the evening. The pub murmurs on around you."
)

# NO OFFER (the regression): an NPC is present, but the prose ends on the player
# examining an OBJECT — the marked map on the bar. Examining/reading is
# click-driven; this is the "Examine the OS Map" free-text box we are killing.
_EXAMINING_OBJECT = (
    "Tom slides the folded map across the bar. \"Megan's,\" he says. \"Found it "
    "in the lane.\" You unfold it on the wet wood. Pencil marks crowd one corner "
    "— a ring round Tan-y-coed, a question mark over the bend, a date you can't "
    "quite read."
)

# NO OFFER: the NPC still HOLDS the floor — mid-anecdote, in no hurry, not yet
# waiting on the player. The floor hasn't been handed over, so there is nothing
# for the player to say yet; a Continue ("let him finish" / "steer him") fits.
_NPC_HOLDS_FLOOR = (
    "Tom is well into it now, a long, looping story about the old quarry and a "
    "winter the river froze, one memory pulling up the next. He doesn't pause "
    "for you — he's talking more to the glass than to you, in no hurry to reach "
    "any point."
)


def _cases() -> list[IsoCase]:
    return [
        # OFFER controls — the prose hands the floor to the player.
        IsoCase(
            name="direct_question",
            scenario="Tom asks the player point-blank what they're after and waits",
            build=_conversation,
            prose=_DIRECT_QUESTION,
            meta=GateExpect(should_offer=True),
            floor=0.7,
        ),
        IsoCase(
            name="challenge",
            scenario="Maud turns it back on the player — why is this your business?",
            build=_conversation,
            prose=_CHALLENGE,
            meta=GateExpect(should_offer=True),
            floor=0.7,
        ),
        IsoCase(
            name="charged_silence",
            scenario="A charged silence after the player said something hard — Eluned waits",
            build=_conversation,
            prose=_CHARGED_SILENCE,
            meta=GateExpect(should_offer=True),
            floor=0.7,
        ),
        # NO-OFFER controls — routine, winding down, or not about what to SAY.
        IsoCase(
            name="routine_reply",
            scenario="A flat factual reply — any phrasing of the next prompt lands the same",
            build=_conversation,
            prose=_ROUTINE,
            meta=GateExpect(should_offer=False),
            floor=0.7,
        ),
        IsoCase(
            name="winding_down",
            scenario="Tom is done — a goodbye, turning back to the racing page",
            build=_conversation,
            prose=_WINDING_DOWN,
            meta=GateExpect(should_offer=False),
            floor=0.7,
        ),
        # The load-bearing regression: an object in the room is a click, not a
        # free-text moment, even with an NPC present.
        IsoCase(
            name="examining_object",
            scenario="The prose ends on the player examining a marked map — click-driven, not 'what to say'",
            build=_conversation,
            prose=_EXAMINING_OBJECT,
            meta=GateExpect(should_offer=False),
            floor=0.7,
        ),
        IsoCase(
            name="npc_holds_floor",
            scenario="Tom is mid-anecdote, not yet waiting on the player — the floor hasn't been handed over",
            build=_conversation,
            prose=_NPC_HOLDS_FLOOR,
            meta=GateExpect(should_offer=False),
            floor=0.7,
        ),
    ]


def _make_task(bundle: AgentBundle, world: World, by_name: dict[str, IsoCase]):
    async def task(case_name: str) -> bool:
        """Run the gate once on the case's fixture prose; return the offer bool.
        A retry-exhausted run becomes `False` (the production fallback — don't
        offer), a gated MISS on the OFFER controls."""
        case = by_name[case_name]
        state = case.build(world)
        try:
            return await emit_free_text_gate(bundle, state, case.prose or "")
        except UnexpectedModelBehavior:
            return False

    return task


def suite(bundle: AgentBundle, world: World) -> Suite:
    cs = _cases()
    return Suite(
        key="free_text_gate",
        title="Free-text gate (offer / case)",
        cases=cs,
        task=_make_task(bundle, world, {c.name: c for c in cs}),
        evaluators=[GateExpectation()],
        show_outputs=True,
    )
