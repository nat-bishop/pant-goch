"""Phase-1 delta regression eval — the per-turn standing-reporter, in isolation.

The session harness (`evals/run.py`) judges whole playthroughs and scores *prose*
and *beat firing*. It is the wrong instrument for the one thing Phase 1 actually
decides: the **quality delta** — how a single player move nudges trust / guard /
village_attention. That signal is buried in a 60-scene playthrough, diluted by an
LLM player and LLM judges, and the `Transition` model the session judges read
doesn't even carry the emitted delta.

This module judges Phase 1 directly. Each case is a *frozen turn context* — a
hand-built `State` (present NPCs, their current trust/guard band, the player's
last action) — fed to the real `emit_state_changes`. Because Phase 1 runs at
temp=1.0, each case is sampled K times and scored as a **pass-rate**, not a single
boolean. The recorded artifact per sample is the emitted `StateChangeBundle`,
which *is* the state diff the engine would apply — so a failure is legible at a
glance.

What we assert vs. what we measure (deliberate — do not "judge on opinion"):

  * ASSERTED (absolute design rules, deterministic):
      - no `attention` bump on ordinary private conversation  ← the captured bug
      - no spurious `BREACH` on a benign turn
      - first turn of a scene shifts nothing (empty deltas)
      - SIGN/FLOOR of trust on an *unambiguously* warm move (must be ≥ +1, not 0)
  * POSITIVE CONTROLS (so a fix can't just zero everything):
      - a conspicuous, public act DOES raise `attention`
      - a move on an NPC's authored `breach` list is NOT treated as benign
        (guard rises or latches) — BREACH-vs-guard+1 is partly opinion, so the
        hard floor is "not a no-op"; the breach RATE is reported, not asserted.
  * NOT ASSERTED (genuinely fuzzy — reported only): exact magnitude of a
    marginal nudge, guard+1 vs. BREACH on a hostile move.

Run:  uv run python -m evals.phase1_deltas [--k 12] [--cases tom_warm_cold,...]
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Windows consoles default to cp1252 and abort on any non-ascii in the report.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8")

from pydantic_ai.exceptions import UnexpectedModelBehavior

from handler.agents import build_scene_runner_bundle, emit_state_changes
from handler.config import Settings
from handler.main import build_model
from handler.schemas import (
    ATTN_DELTA_VALUES,
    AttentionDelta,
    Option,
    QualityDelta,
    StateChangeBundle,
    TRUST_DELTA_VALUES,
)
from handler.state import State, TurnRecord, make_initial_state
from handler.world import World, load_world

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


# ───────────────────────── cases ─────────────────────────


@dataclass
class Case:
    name: str
    scenario: str                          # human-readable: what the player just did
    build: Callable[[World], State]
    expect: Expect


# The pub: Tom guarded, the player's first move toward him.
_TOM_GUARDED = (
    'Tom doesn\'t look up. He turns the glass a quarter-turn, another. "Free '
    'country," he says, eventually — not kindly, not the other way either. You '
    "take the stool. Up close, he smells of woodsmoke and wet wool."
)


def cases() -> list[Case]:
    return [
        # ── THE CAPTURED FAILURE (verbatim from the live playthrough) ──
        # gemma-4-e4b emitted {attention+1} here: a spurious attention bump on a
        # private, quiet exchange. That is the UNAMBIGUOUS bug, so this case pins
        # exactly it — attention must stay "0" and a bystander mustn't be
        # breached. Whether "comment on his mood / the room" earns trust+1 is a
        # genuine judgment call (it borders on neutral small-talk), so trust is
        # measured, NOT asserted, here — see `warm_decent_to_tom` for the
        # unambiguous warmth-credit assertion.
        Case(
            name="tom_warm_cold",
            scenario="Comment on guarded Tom's mood / the room (the captured option; quiet, private)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Comment on his mood or the atmosphere of the pub.",
                last_prose=_TOM_GUARDED,
            ),
            expect=Expect(attention_eq=0),  # the captured bug; trust left to the case below
        ),
        # ── warmth IS credited on an unambiguously warm/decent move ──
        # The prompt's own definition of trust+1: "warm, patient, decent, or
        # shared something real." This is the core loop — if it doesn't earn
        # trust, no confession is ever reachable from patience.
        Case(
            name="warm_decent_to_tom",
            scenario="Tell Tom you're not chasing a story — you just want to understand who Megan was, and you're sorry",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Tell Tom quietly that you're not here to make trouble — you just want to understand who Megan was, and that you're sorry for the village's loss.",
                last_prose=_TOM_GUARDED, typed=True,
            ),
            expect=Expect(trust_floor={"tom": 1}, attention_eq=0),
        ),
        # ── isolate "no spurious attention" from the warmth question ──
        Case(
            name="neutral_question_no_attention",
            scenario="Ask Tom an ordinary, on-topic question (private, low-key)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Ask Tom how long he's lived in the village.",
                last_prose=_TOM_GUARDED,
            ),
            expect=Expect(attention_eq=0),  # trust left fuzzy on purpose
        ),
        # ── first turn of a scene shifts nothing ──
        Case(
            name="first_turn_inertia",
            scenario="Scene just opened; the narrator only set the scene (no player action yet)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="", last_prose="", first_turn=True,
            ),
            expect=Expect(empty=True),
        ),
        # ── POSITIVE CONTROL: a conspicuous public act SHOULD raise attention ──
        Case(
            name="public_accusation_attention",
            scenario="Stand up and loudly accuse Eluned Vaughn to the whole pub (a public scene)",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Stand up and announce to the whole pub that Eluned Vaughn ran Megan off the road.",
                last_prose=_TOM_GUARDED, typed=True,
            ),
            expect=Expect(attention_eq=1),
        ),
        # ── POSITIVE CONTROL: a move on Tom's authored breach list (lie: police) ──
        # "claiming to be police or family" is literally in Tom's breach list, so
        # BREACH is designed-correct; the hard floor is "not benign".
        Case(
            name="lie_police_to_tom",
            scenario="Lie to Tom that you are a police officer reopening the case",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Tell Tom you're a police detective reopening the Pritchard case and he must answer.",
                last_prose=_TOM_GUARDED, typed=True,
            ),
            expect=Expect(forbid_breach=False, guard_up_or_breach=("tom",)),
        ),
        # ── POSITIVE CONTROL: an explicit threat (Tom breach: "threatening him") ──
        Case(
            name="threaten_tom",
            scenario="Threaten Tom — talk or you'll expose him as having covered for a killer",
            build=lambda w: ctx(
                w, location="the_slaughtered_lamb",
                last_action="Tell Tom that if he won't talk, you'll print that he helped cover up a girl's death.",
                last_prose=_TOM_GUARDED, typed=True,
            ),
            expect=Expect(forbid_breach=False, guard_up_or_breach=("tom",)),
        ),
    ]


# ───────────────────────── runner ─────────────────────────


async def _one_sample(bundle, state: State) -> Diff:
    """Run Phase 1 once, mirroring the production fallback: a runaway (decision
    agent retries exhausted) becomes an empty bundle, exactly as scene_loop does."""
    try:
        b = await emit_state_changes(bundle, state)
    except UnexpectedModelBehavior:
        b = StateChangeBundle()
    return Diff.parse(b)


@dataclass
class CaseResult:
    case: Case
    diffs: list[Diff]

    @property
    def passes(self) -> int:
        return sum(1 for d in self.diffs if not self.case.expect.violations(d))

    @property
    def rate(self) -> float:
        return self.passes / len(self.diffs) if self.diffs else 0.0

    @property
    def breach_rate(self) -> float:
        """Reported (not asserted) for the breach positive-controls: of the
        guard_up_or_breach NPCs, how often did Phase 1 actually latch?"""
        tgt = set(self.case.expect.guard_up_or_breach)
        if not tgt:
            return float("nan")
        return sum(1 for d in self.diffs if tgt & d.breached) / len(self.diffs)


async def run_cases(
    bundle, world: World, selected: list[Case], k: int, concurrency: int
) -> list[CaseResult]:
    sem = asyncio.Semaphore(concurrency)

    async def sample(state: State) -> Diff:
        async with sem:
            return await _one_sample(bundle, state)

    results: list[CaseResult] = []
    # Fire every (case × sample) at once; inferno batches them.
    jobs = [(c, asyncio.create_task(sample(c.build(world)))) for c in selected for _ in range(k)]
    by_case: dict[str, list[Diff]] = {c.name: [] for c in selected}
    for c, task in jobs:
        by_case[c.name].append(await task)
    for c in selected:
        results.append(CaseResult(c, by_case[c.name]))
    return results


def _print_report(results: list[CaseResult], k: int) -> None:
    print(f"\n{'='*78}\nPHASE-1 DELTA EVAL  (k={k} samples/case)\n{'='*78}")
    total_pass = total = 0
    for r in results:
        total_pass += r.passes
        total += len(r.diffs)
        flag = "OK " if r.rate >= 0.9 else ("LOW" if r.rate >= 0.5 else "!! ")
        print(f"\n[{flag}] {r.case.name}: {r.passes}/{len(r.diffs)} pass ({r.rate:.0%})")
        print(f"      scenario: {r.case.scenario}")
        # Show the distribution of emitted diffs (the state diffs) — most common first.
        counts: dict[str, int] = {}
        for d in r.diffs:
            counts[str(d)] = counts.get(str(d), 0) + 1
        for diff_str, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"        {n:>2}x  {diff_str}")
        if r.case.expect.guard_up_or_breach:
            print(f"      (breach rate: {r.breach_rate:.0%} — reported, not asserted)")
    print(f"\n{'-'*78}\nOVERALL: {total_pass}/{total} ({total_pass/total:.0%}) sample-level pass\n{'='*78}")


def resolve_api_url(settings: Settings) -> str:
    if settings.use_inferno:
        return f"{settings.inferno_url.rstrip('/')}/v1"
    return settings.llama_base_url or f"http://{settings.llama_host}:{settings.llama_port}/v1"


async def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 delta regression eval")
    ap.add_argument("--k", type=int, default=12, help="samples per case (temp=1.0)")
    ap.add_argument("--cases", type=str, default=None, help="comma-separated case-name subset")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    settings = Settings()
    world = load_world(settings.world_path or (_REPO_ROOT / "world"))
    model = build_model(resolve_api_url(settings))
    bundle = build_scene_runner_bundle(
        model, world, decision_max_tokens=settings.decision_agent_max_tokens
    )

    all_cases = cases()
    if args.cases:
        want = set(args.cases.split(","))
        unknown = want - {c.name for c in all_cases}
        if unknown:
            raise SystemExit(f"unknown cases: {sorted(unknown)}")
        selected = [c for c in all_cases if c.name in want]
    else:
        selected = all_cases

    print(f"engine={resolve_api_url(settings)}  cases={len(selected)}  k={args.k}")
    results = await run_cases(bundle, world, selected, args.k, args.concurrency)
    _print_report(results, args.k)


if __name__ == "__main__":
    asyncio.run(main())
