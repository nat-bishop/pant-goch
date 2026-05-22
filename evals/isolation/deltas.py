"""Phase-1 delta suite — the per-turn standing reporter, in isolation.

The one thing Phase 1 decides is the **quality delta**: how a single player move
nudges trust / guard / village_attention. Each case is a *frozen turn context* — a
hand-built `State` (present NPCs, their trust/guard band, the player's last action)
— fed to the real `emit_state_changes`. Phase 1 runs at temp=1.0, so the harness
samples each case K times and scores a pass-rate.

What we assert vs. measure (deliberate — do not "judge on opinion"):
  * ASSERTED (deterministic design rules, gated by a floor):
      - no `attention` bump on ordinary private conversation  ← the captured bug
      - no spurious `BREACH` on a benign turn
      - first turn of a scene shifts nothing (empty deltas)
      - SIGN/FLOOR of trust on an *unambiguously* warm move (≥ +1, not 0)
  * POSITIVE CONTROLS (so a fix can't just zero everything): a conspicuous public
    act DOES raise `attention`; a move on an NPC's authored breach list is NOT benign.
  * NOT ASSERTED (reported via the suite's output distribution only): exact magnitude
    of a marginal nudge; guard+1 vs. BREACH on a hostile move.

Migrated from the original standalone delta script; the sampling / concurrency /
reporting it hand-rolled now come from the shared harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from handler.agents import AgentBundle, emit_state_changes
from handler.schemas import (
    ATTN_DELTA_VALUES,
    AttentionDelta,
    Option,
    QualityDelta,
    StateChangeBundle,
    TRUST_DELTA_VALUES,
)
from handler.state import State, TurnRecord, make_initial_state
from handler.world import World

from .harness import IsoCase, Suite


# ───────────────────────── context builder ─────────────────────────


def ctx(
    world: World,
    *,
    location: str,
    last_action: str,
    last_prose: str,
    typed: bool = False,
    trust: dict[str, int] | None = None,
    guard: dict[str, int] | None = None,
    attention: int = 0,
    breached: tuple[str, ...] = (),
    npc_at: dict[str, str] | None = None,
    unlocked: tuple[str, ...] = (),
    case_file: tuple[str, ...] = (),
    first_turn: bool = False,
) -> State:
    """Build the exact `State` Phase 1 reads on the turn under test.

    Phase 1 judges `dialogue_history[-1].player_pick` (the pick made at the end of
    the prior turn) against that turn's prose — so `last_action` + `last_prose`
    are the move being scored. `first_turn=True` leaves dialogue_history empty
    (the scene-opening turn, which must shift nothing).
    """
    st = make_initial_state(world)
    st.player_location = location
    st.scene_id = 3
    st.village_attention = attention
    st.unlocked_beats = set(unlocked)
    st.case_file_entries = list(case_file)
    st.guard_breached = set(breached)
    for nid, v in (trust or {}).items():
        st.trust[nid] = v
    for nid, v in (guard or {}).items():
        st.guard[nid] = v
    for nid, loc in (npc_at or {}).items():
        st.npc_locations[nid] = loc
    if not first_turn:
        st.dialogue_history.append(
            TurnRecord(
                prose=last_prose,
                applied_state_changes=[],
                player_pick=Option(text=last_action),
                player_pick_typed=typed,
            )
        )
    return st


# ───────────────────────── expectation spec ─────────────────────────


@dataclass(frozen=True)
class Expect:
    """Declarative pass condition for one case. Only the set fields are checked,
    so each case asserts exactly the axes that are genuinely *true* for it."""

    empty: bool = False                       # deltas must be empty (first turn)
    attention_eq: int | None = None           # required attention-delta sum (0 or 1)
    forbid_breach: bool = True                # no NPC may BREACH this turn
    trust_floor: dict[str, int] = field(default_factory=dict)   # npc -> min trust delta
    guard_up_or_breach: tuple[str, ...] = ()  # these NPCs must NOT be treated as benign

    def violations(self, d: "Diff") -> list[str]:
        out: list[str] = []
        if self.empty and not d.is_empty:
            out.append(f"expected empty deltas, got {d}")
        if self.attention_eq is not None and d.attention != self.attention_eq:
            out.append(f"attention {d.attention} != expected {self.attention_eq}")
        if self.forbid_breach and d.breached:
            out.append(f"unexpected BREACH of {sorted(d.breached)}")
        for nid, floor in self.trust_floor.items():
            if d.trust.get(nid, 0) < floor:
                out.append(f"trust[{nid}]={d.trust.get(nid, 0)} < floor {floor}")
        for nid in self.guard_up_or_breach:
            if nid not in d.breached and d.guard.get(nid, 0) <= 0:
                out.append(
                    f"{nid} treated as benign (guard {d.guard.get(nid, 0)}, no breach)"
                )
        return out


# ───────────────────────── parsed diff ─────────────────────────


@dataclass
class Diff:
    """The state diff Phase 1 emitted, parsed to ints — the unit of judgment."""

    trust: dict[str, int]
    guard: dict[str, int]
    breached: set[str]
    attention: int
    commitments: list[str]

    @property
    def is_empty(self) -> bool:
        return not (self.trust or self.guard or self.breached or self.attention or self.commitments)

    @classmethod
    def parse(cls, b: StateChangeBundle) -> "Diff":
        trust: dict[str, int] = {}
        guard: dict[str, int] = {}
        breached: set[str] = set()
        attention = 0
        for delta in b.deltas:
            if isinstance(delta, QualityDelta):
                if delta.trust != "0":
                    trust[delta.npc_id] = trust.get(delta.npc_id, 0) + TRUST_DELTA_VALUES[delta.trust]
                if delta.guard == "BREACH":
                    breached.add(delta.npc_id)
                elif delta.guard != "0":
                    guard[delta.npc_id] = guard.get(delta.npc_id, 0) + TRUST_DELTA_VALUES[delta.guard]
            elif isinstance(delta, AttentionDelta):
                attention += ATTN_DELTA_VALUES[delta.attention]
        return cls(trust, guard, breached, attention, [c.beat_id for c in b.commitments])

    def __str__(self) -> str:
        bits = []
        if self.trust:
            bits.append("trust " + ",".join(f"{k}{v:+d}" for k, v in self.trust.items()))
        if self.guard:
            bits.append("guard " + ",".join(f"{k}{v:+d}" for k, v in self.guard.items()))
        if self.breached:
            bits.append("BREACH " + ",".join(sorted(self.breached)))
        if self.attention:
            bits.append(f"attention+{self.attention}")
        if self.commitments:
            bits.append("commit " + ",".join(self.commitments))
        return "{" + "; ".join(bits) + "}" if bits else "{empty}"


# ───────────────────────── evaluator ─────────────────────────


@dataclass
class Phase1Expectation(Evaluator):
    """Bool assertion: the case's `Expect` held (no violations) on this sample.
    The `Expect` rides on `ctx.metadata`; the emitted `Diff` is `ctx.output`. The
    floor gate aggregates this across the K repeats into a per-case pass-rate."""

    def build_serialization_arguments(self) -> dict:  # not data-parameterized
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        expect: Expect = ctx.metadata
        diff: Diff = ctx.output
        return not expect.violations(diff)


# ───────────────────────── cases ─────────────────────────


# The pub: Tom guarded, the player's first move toward him.
_TOM_GUARDED = (
    'Tom doesn\'t look up. He turns the glass a quarter-turn, another. "Free '
    'country," he says, eventually — not kindly, not the other way either. You '
    "take the stool. Up close, he smells of woodsmoke and wet wool."
)


def cases() -> list[IsoCase]:
    # Floors catch *collapse*, not drift (see harness.DEFAULT_FLOOR). The structural
    # axes (empty first turn, no-attention-on-private) are near-deterministic → high
    # floor; the behavioural axes (a warm move earning trust, a public act bumping
    # attention, a hostile move not staying benign) are genuinely stochastic at
    # temp=1.0 → a 0.5 floor that still trips on a true regression to ~0. CALIBRATE
    # these against the first live `make eval-deltas` run.
    return [
        # ── THE CAPTURED FAILURE: warm/decent move on a cold NPC ──
        # gemma-4-e4b emitted {attention+1} here: no trust for Tom, AND a spurious
        # attention bump on a private, gentle exchange. Both are wrong.
        IsoCase(
            name="tom_warm_cold",
            scenario="Comment empathetically on guarded Tom's mood / the room (warm, decent, no pressing)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Comment on his mood or the atmosphere of the pub.",
                last_prose=_TOM_GUARDED,
            ),
            meta=Expect(trust_floor={"tom": 1}, attention_eq=0),
            floor=0.5,
        ),
        # ── isolate "no spurious attention" from the warmth question ──
        IsoCase(
            name="neutral_question_no_attention",
            scenario="Ask Tom an ordinary, on-topic question (private, low-key)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Ask Tom how long he's lived in the village.",
                last_prose=_TOM_GUARDED,
            ),
            meta=Expect(attention_eq=0),  # trust left fuzzy on purpose
            floor=0.8,
        ),
        # ── first turn of a scene shifts nothing ──
        IsoCase(
            name="first_turn_inertia",
            scenario="Scene just opened; the narrator only set the scene (no player action yet)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="", last_prose="", first_turn=True,
            ),
            meta=Expect(empty=True),
            floor=0.9,
        ),
        # ── POSITIVE CONTROL: a conspicuous public act SHOULD raise attention ──
        IsoCase(
            name="public_accusation_attention",
            scenario="Stand up and loudly accuse Eluned Vaughn to the whole pub (a public scene)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Stand up and announce to the whole pub that Eluned Vaughn ran Megan off the road.",
                last_prose=_TOM_GUARDED, typed=True,
            ),
            meta=Expect(attention_eq=1),
            floor=0.5,
        ),
        # ── POSITIVE CONTROL: a move on Tom's authored breach list (lie: police) ──
        # "claiming to be police or family" is literally in Tom's breach list, so
        # BREACH is designed-correct; the hard floor is "not benign".
        IsoCase(
            name="lie_police_to_tom",
            scenario="Lie to Tom that you are a police officer reopening the case",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Tell Tom you're a police detective reopening the Pritchard case and he must answer.",
                last_prose=_TOM_GUARDED, typed=True,
            ),
            meta=Expect(forbid_breach=False, guard_up_or_breach=("tom",)),
            floor=0.5,
        ),
        # ── POSITIVE CONTROL: an explicit threat (Tom breach: "threatening him") ──
        IsoCase(
            name="threaten_tom",
            scenario="Threaten Tom — talk or you'll expose him as having covered for a killer",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Tell Tom that if he won't talk, you'll print that he helped cover up a girl's death.",
                last_prose=_TOM_GUARDED, typed=True,
            ),
            meta=Expect(forbid_breach=False, guard_up_or_breach=("tom",)),
            floor=0.5,
        ),
    ]


# ───────────────────────── suite ─────────────────────────


def _make_task(bundle: AgentBundle, world: World, by_name: dict[str, IsoCase]):
    async def task(case_name: str) -> Diff:
        """Run Phase 1 once on the frozen state, mirroring the production fallback:
        a runaway (decision-agent retries exhausted) becomes an empty bundle,
        exactly as scene_loop does."""
        state = by_name[case_name].build(world)
        try:
            b = await emit_state_changes(bundle, state)
        except UnexpectedModelBehavior:
            b = StateChangeBundle()
        return Diff.parse(b)

    return task


def suite(bundle: AgentBundle, world: World) -> Suite:
    cs = cases()
    return Suite(
        key="phase1_deltas",
        title="Phase-1 deltas (pass-rate / case)",
        cases=cs,
        task=_make_task(bundle, world, {c.name: c for c in cs}),
        evaluators=[Phase1Expectation()],
        show_outputs=True,
    )
