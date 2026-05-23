"""Pairwise preference judge + theme extraction — the A/B half of the judges.

The single-item `AgentJudge` scores one transcript in isolation; A/B needs to
compare two. `run_pairwise` shows the judge two whole-session transcripts
(blinded, order-randomized, arm hidden), asks which is the better scene
experience plus ONE concrete reason, and un-blinds the verdict back to the
arm that won. `cluster_themes` then groups every reason into recurring themes
tagged by the arm they favor — the "common themes" deliverable.

Reuses `build_judge_agent` (same NativeOutput + temp=0 wiring as the absolute
judges) and `_session_context` (same session-as-text framing the session judges
show), so the pairwise judge sees a transcript exactly as the absolute ones do —
only doubled and blinded.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic_ai.models import Model

from ..transitions import SessionView
from .base import CASE_TIMEOUT_S, build_judge_agent
from .session_judges import _session_context


class Preference(BaseModel):
    # reason first: the model commits to a justification before naming a winner
    # (same ordering doctrine as the PassFail/Scored verdicts).
    reason: str
    winner: Literal["A", "B", "tie"]


class Theme(BaseModel):
    label: str
    favors_arm: Literal["control", "test", "mixed"]
    frequency: int
    summary: str


class Themes(BaseModel):
    themes: list[Theme]


_PAIRWISE_SYSTEM = (
    "You compare two playthroughs of a narrative investigation game set in a "
    "grounded Welsh-village murder mystery — the player is a journalist working "
    "sources. You are shown Transcript A and Transcript B (the SAME player "
    "persona, two different runs). Judge ONE thing: which is the better scene "
    "experience overall — more engaging and coherent, with real momentum (the "
    "player's choices matter, scenes advance the investigation rather than "
    "stalling or circling), while staying true to the grounded fiction. Pick A "
    "or B; answer 'tie' ONLY when they are genuinely indistinguishable in "
    "quality. Give exactly ONE concrete reason naming the specific thing that "
    "decided it (e.g. 'B kept circling the same question without new information' "
    "or 'A's scenes each surfaced a new lead'). Do not mention transcript length "
    "or letter; judge the experience."
)

_THEMES_SYSTEM = (
    "You are given a list of head-to-head judgements, each one rationale for why "
    "one playthrough beat another, tagged with which arm won (control or test). "
    "Group the rationales into recurring THEMES — the distinct, repeated reasons "
    "one arm tends to win or lose. For each theme give: a short label; which arm "
    "it favors (control, test, or mixed if it cuts both ways); the frequency "
    "(how many rationales fall under it); and a one-sentence summary. Prefer a "
    "few strong themes over many singletons. The frequencies need not sum to the "
    "total (a rationale may fit one theme or none)."
)


@dataclass
class PairOutcome:
    pair_id: str
    persona: str
    winner_arm: str | None   # "control" / "test", or None on a tie
    reason: str


def _render_pair(a: SessionView, b: SessionView) -> str:
    return (
        "Transcript A:\n"
        f"{_session_context(a)}\n\n"
        "Transcript B:\n"
        f"{_session_context(b)}\n\n"
        "Which transcript is the better scene experience, and why?"
    )


async def run_pairwise(
    model: Model,
    pairs: list[tuple[str, str, SessionView, SessionView]],
    *,
    concurrency: int = 16,
    seed: int = 0,
) -> list[PairOutcome]:
    """Judge each (pair_id, persona, control_view, test_view) pair.

    Blinding is decided sequentially from a seeded RNG BEFORE the concurrent
    judging (so reproducibility doesn't depend on task completion order): for
    each pair we flip whether control is shown as "A" or "B", hide the arm, then
    map the judge's A/B verdict back to the true arm.
    """
    rng = random.Random(seed)
    # (pair_id, persona, control, test, control_is_a)
    plans = [(pid, persona, c, t, rng.random() < 0.5) for pid, persona, c, t in pairs]

    agent = build_judge_agent(model, Preference, _PAIRWISE_SYSTEM)
    sem = asyncio.Semaphore(concurrency)

    async def one(pid, persona, control, test, control_is_a) -> PairOutcome:
        side_a, side_b = (control, test) if control_is_a else (test, control)
        async with sem:
            try:
                res = await asyncio.wait_for(
                    agent.run(_render_pair(side_a, side_b)), timeout=CASE_TIMEOUT_S
                )
            except (asyncio.TimeoutError, TimeoutError):
                return PairOutcome(pid, persona, None, "(judge timed out)")
            except Exception as e:  # noqa: BLE001
                # A single malformed verdict (e.g. NativeOutput ValidationError)
                # or a transient judge error must not sink every other pair's
                # completed comparison — record it as an undecided pair instead.
                return PairOutcome(pid, persona, None, f"(judge error: {type(e).__name__})")
            pref: Preference = res.output
        if pref.winner == "tie":
            winner_arm = None
        else:
            winner_is_a = pref.winner == "A"
            winner_arm = "control" if (winner_is_a == control_is_a) else "test"
        return PairOutcome(pid, persona, winner_arm, pref.reason.strip())

    return await asyncio.gather(*(one(*p) for p in plans))


def tally(outcomes: list[PairOutcome]) -> dict:
    """Win/tie counts + test win-rate (over decided pairs). Mirrors the
    isolation lane's habit of reporting the rate, not a single verdict."""
    wins = {"control": 0, "test": 0}
    ties = 0
    for o in outcomes:
        if o.winner_arm is None:
            ties += 1
        else:
            wins[o.winner_arm] += 1
    decided = wins["control"] + wins["test"]
    return {
        "n_pairs": len(outcomes),
        "wins": wins,
        "ties": ties,
        "win_rate_test": round(wins["test"] / decided, 3) if decided else None,
    }


async def cluster_themes(model: Model, outcomes: list[PairOutcome]) -> Themes:
    """Group the pairwise rationales into recurring themes. Best-effort — the
    caller should tolerate an empty result if the judge errors."""
    decided = [o for o in outcomes if o.winner_arm is not None]
    if not decided:
        return Themes(themes=[])
    listing = "\n".join(
        f"{i+1}. [{o.winner_arm} won] {o.reason}" for i, o in enumerate(decided)
    )
    agent = build_judge_agent(model, Themes, _THEMES_SYSTEM)
    res = await asyncio.wait_for(
        agent.run(f"Rationales ({len(decided)}):\n{listing}"), timeout=CASE_TIMEOUT_S
    )
    return res.output
