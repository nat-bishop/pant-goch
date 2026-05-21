"""Run judges over a corpus of items and persist the results.

Replaces the pydantic-evals `Dataset.evaluate(_identity, ...)` idiom: judges score
pre-extracted turns/sessions, which aren't a "task to run", so a plain
gather→aggregate→persist is both more honest and durable. Writes `judges.json`
(structured: every case's value + reason) and `judges.md` (readable: per-judge
headline + the lowest-scoring cases with their reasons) into the run dir, next to
`summary.json`.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .base import AgentJudge
from .verdicts import PassFail

# Per-case wall-clock ceiling. Without it, one stuck response (a vLLM request that
# never returns its body) hangs the whole asyncio.gather indefinitely — exactly the
# stall seen judging 20260520-194311 at high concurrency. On timeout the case is
# recorded as an error (visible in the report) and the run still terminates.
CASE_TIMEOUT_S = 180.0


@dataclass
class CaseScore:
    case_id: str
    value: bool | int | None  # passed (bool) or score (int); None on judge error
    reason: str


@dataclass
class JudgeReport:
    judge: str
    kind: str  # "passfail" | "scored"
    n: int
    errors: int
    pass_rate: float | None  # set for passfail judges
    mean_score: float | None  # set for scored judges
    cases: list[CaseScore]


async def _score_case(
    judge: AgentJudge, case_id: str, item: object, sem: asyncio.Semaphore
) -> CaseScore:
    async with sem:
        try:
            v = await asyncio.wait_for(judge.judge(item), timeout=CASE_TIMEOUT_S)
        except TimeoutError:
            return CaseScore(case_id, None, f"ERROR: timeout after {CASE_TIMEOUT_S:.0f}s")
        except Exception as e:  # a judge failure must not sink the whole run
            return CaseScore(case_id, None, f"ERROR: {type(e).__name__}: {e}")
    return CaseScore(case_id, v.passed if isinstance(v, PassFail) else v.score, v.reason)


def _aggregate(judge: AgentJudge, scores: list[CaseScore]) -> JudgeReport:
    is_passfail = judge.output_type is PassFail
    ok = [s for s in scores if s.value is not None]
    return JudgeReport(
        judge=judge.name,
        kind="passfail" if is_passfail else "scored",
        n=len(scores),
        errors=len(scores) - len(ok),
        pass_rate=round(sum(1 for s in ok if s.value) / len(ok), 3) if is_passfail and ok else None,
        mean_score=round(sum(s.value for s in ok) / len(ok), 3) if not is_passfail and ok else None,
        cases=scores,
    )


async def run_judge(
    judge: AgentJudge, items: list[tuple[str, object]], *, concurrency: int = 256
) -> JudgeReport:
    """Score every item with one judge, all cases fired concurrently."""
    sem = asyncio.Semaphore(concurrency)
    scores = await asyncio.gather(*(_score_case(judge, cid, it, sem) for cid, it in items))
    return _aggregate(judge, scores)


async def score_plan(
    plan: list[tuple[AgentJudge, list[tuple[str, object]]]], *, concurrency: int = 256
) -> list[JudgeReport]:
    """Fire EVERY judge × case across all judges concurrently under ONE shared
    semaphore, then group back per judge. This saturates inferno's continuous
    batching (~64–256 in-flight) instead of draining the pipe between judges;
    the cap only bounds client-side sockets, it is not a rate limit. Reports are
    returned in `plan` order."""
    sem = asyncio.Semaphore(concurrency)
    owners: list[AgentJudge] = []
    coros = []
    for judge, items in plan:
        for case_id, item in items:
            owners.append(judge)
            coros.append(_score_case(judge, case_id, item, sem))
    results = await asyncio.gather(*coros)

    grouped: dict[int, list[CaseScore]] = {}
    for judge, score in zip(owners, results):
        grouped.setdefault(id(judge), []).append(score)
    return [_aggregate(judge, grouped.get(id(judge), [])) for judge, _ in plan]


def _headline(r: JudgeReport) -> str:
    if r.kind == "passfail":
        return f"pass-rate {r.pass_rate}"
    return f"mean score {r.mean_score}/5"


def _worst(r: JudgeReport, k: int = 5) -> list[CaseScore]:
    """The lowest-scoring cases (failures first / lowest scores) — the ones whose
    reasons are worth reading and freezing as regression cases."""
    graded = [c for c in r.cases if c.value is not None]
    return sorted(graded, key=lambda c: (c.value is True, c.value))[:k]


def persist(out_dir: Path, reports: list[JudgeReport]) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    json_path = out_dir / "judges.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8"
    )

    lines = ["# Judge results", ""]
    for r in reports:
        flag = ""
        if r.kind == "passfail" and r.pass_rate is not None and r.pass_rate < 1.0:
            flag = "  ⚠️"
        lines.append(f"## {r.judge} — {_headline(r)} (n={r.n}, errors={r.errors}){flag}")
        for c in _worst(r):
            lines.append(f"- `{c.case_id}` → {c.value}: {c.reason.strip()}")
        lines.append("")
    md_path = out_dir / "judges.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def render_console(reports: list[JudgeReport]) -> str:
    return "\n".join(
        f"  {r.judge:28s} {_headline(r):20s} (n={r.n}, errors={r.errors})"
        for r in reports
    )
