"""Judge run — score a recorded playtest with the LLM judges, and persist it.

    uv run python -m evals.judge_run --run evals/artifacts/<timestamp>

Decoupled from generation (`evals.run`): reads dumped artifacts only, so it needs
no llama session and can re-judge a frozen corpus any time. Defaults to local Qwen
via inferno; warms it once, then scores each scope as a pydantic-evals `Dataset`
over an identity task so judge results land in the same Logfire experiment surface
(and run-over-run diff) as the deterministic invariants. Merges a `judges` block
into the run's `summary.json` and writes a readable `judges.md` (worst cases).

Judges are grouped by the scope they score, one `Dataset` each:
  - handler_judge_warrants      — every (turn × fired beat): firing warranted / depicted?
  - handler_judge_transitions   — every turn: prose↔location, choice meaningfulness, voice
  - handler_judge_free_text     — typed turns: did the narrator accept a false premise?
  - handler_judge_sessions      — whole playthroughs: earned ending, emergent continuity

The high-volume transition scope (the three Scored quality-gauges) is sampled to
`--max-cases-per-scope` with a seeded RNG; the small PassFail bug-detector scopes
(warrants / free-text / sessions) fall under the cap and run in full. Sampling is
for wall-clock only — the local Qwen judge has no API cost.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HANDLER_ENV", "eval")
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8")

# Importing handler.main configures Logfire + instruments pydantic-ai (same setup
# evals.run gets), so judge model calls are traced and tagged env=eval.
import handler.main  # noqa: F401

from pydantic_evals import Case, Dataset

from handler.config import Settings
from handler.world import load_world

from . import report as run_report
from .judges.model import build_judge_model, warm_judge
from .judges.verdicts import PassFail
from .judges.session_judges import (
    EmergentEventHonored,
    EndingEarnedNarratively,
)
from .judges.transition_judges import (
    BeatFiredAppropriately,
    ChoiceMeaningfulness,
    FalsePremiseAccepted,
    NpcVoiceAdherence,
    ProseDepictsBeat,
    ProseMatchesLocation,
)
from .session import load_run
from .transitions import extract_transitions, session_view

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HANDLER judge run")
    ap.add_argument("--run", type=Path, required=True, help="artifact dir to judge")
    ap.add_argument("--world", type=Path, default=_REPO_ROOT / "world")
    ap.add_argument("--judge-model", type=str, default=None,
                    help="override Settings.judge_model (e.g. gemini:gemini-3.1-pro)")
    ap.add_argument("--max-cases-per-scope", type=int, default=1000,
                    help="cap cases per scope via a seeded random sample; 0 = no cap. "
                         "Only the transition scope exceeds the default; the PassFail "
                         "bug-detector scopes run in full.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for sampling, so re-judging a run is reproducible")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="max judge calls in flight per scope. 64 drains reliably on a "
                         "single vLLM instance; higher saturated the queue and stalled.")
    return ap.parse_args()


async def _identity(item):
    return item


def _sample(items: list, k: int, rng: random.Random) -> list:
    """Seeded random sample down to `k`; full list when under the cap or k<=0.
    Random (not a head slice) so the sample isn't biased toward early-session turns."""
    if k and k > 0 and len(items) > k:
        return rng.sample(items, k)
    return items


def _worst_cases_md(results: list[tuple[str, list, object]], k: int = 5) -> str:
    """Per-judge headline + floor + the genuinely failing cases with their reasons —
    the ones worth reading and freezing as regression cases. Only real lows are shown
    (PassFail failures, Scored cases ≤3); when a judge has fewer than `k` lows the list
    is short rather than padded with passing 5-score cases. Built from the reports."""
    rows: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    kinds: dict[str, str] = {}
    for _scope, judges, rep in results:
        names = {j.name for j in judges}
        for j in judges:
            kinds[j.name] = "passfail" if j.output_type is PassFail else "scored"
        for case in rep.cases:
            present = {**case.assertions, **case.scores}
            for jname, r in present.items():
                rows[jname].append((float(r.value), case.name, (r.reason or "").strip()))
        # A judge that errored on a case (e.g. timeout) leaves no result for it.
        scored = defaultdict(int)
        for case in rep.cases:
            for jname in {**case.assertions, **case.scores}:
                scored[jname] += 1
        for jname in names:
            errors[jname] += len(rep.cases) - scored.get(jname, 0)

    def _is_low(kind: str, v: float) -> bool:
        return v < 1.0 if kind == "passfail" else v <= 3

    lines = ["# Judge results", ""]
    for jname in sorted(rows):
        kind = kinds.get(jname, "scored")
        vals = [v for v, _, _ in rows[jname]]
        head = round(sum(vals) / len(vals), 3) if vals else None
        if kind == "scored":
            n_below = sum(1 for v in vals if v <= 3)
            floor = f", min={min(vals)}, ≤3×{n_below}/{len(vals)}" if vals else ""
            flag = "  ⚠️" if n_below else ""
        else:
            floor = ""
            flag = "  ⚠️" if head is not None and head < 1.0 else ""
        lines.append(
            f"## {jname} — {head} (n={len(vals)}, errors={errors.get(jname, 0)}{floor}){flag}"
        )
        lows = [row for row in sorted(rows[jname], key=lambda t: t[0]) if _is_low(kind, row[0])]
        if lows:
            for v, cid, reason in lows[:k]:
                lines.append(f"- `{cid}` → {v}: {reason}")
        else:
            lines.append("- (no failures)")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    args = _parse_args()
    settings = Settings()
    world = load_world(args.world)
    model = build_judge_model(settings, override=args.judge_model)
    rng = random.Random(args.seed)

    print(f"judge_run: scoring {args.run} with "
          f"{args.judge_model or settings.judge_model}", flush=True)
    warm_judge(settings, override=args.judge_model)

    artifacts = load_run(args.run)
    transitions = [t for art in artifacts for t in extract_transitions(art)]
    sessions = [session_view(art) for art in artifacts]
    print(f"extracted {len(transitions)} transitions, {len(sessions)} sessions", flush=True)

    # Item lists keyed (case_id, item) per scope. free-text and warrant items derive
    # from the FULL transition list (they stay under the cap and run in full); only
    # the transition scope is sampled, and once — so all three transition judges
    # score the same sampled turns.
    tx_items = [(f"{t.session_id}-s{t.scene_id}-{i}", t) for i, t in enumerate(transitions)]
    warrant_items = [
        (f"{t.session_id}-s{t.scene_id}-{b}-{i}", t.model_copy(update={"beat_of_interest": b}))
        for i, t in enumerate(transitions) for b in t.fired_beats
    ]
    free_text_items = [it for it in tx_items if it[1].is_free_text]
    session_items = [(s.session_id, s) for s in sessions]

    K = args.max_cases_per_scope
    scopes: list[tuple[str, list, list]] = [
        ("handler_judge_warrants", _sample(warrant_items, K, rng),
         [BeatFiredAppropriately(model, world), ProseDepictsBeat(model, world)]),
        ("handler_judge_transitions", _sample(tx_items, K, rng),
         [ProseMatchesLocation(model, world), ChoiceMeaningfulness(model),
          NpcVoiceAdherence(model, world)]),
        ("handler_judge_free_text", _sample(free_text_items, K, rng),
         [FalsePremiseAccepted(model, world)]),
        ("handler_judge_sessions", _sample(session_items, K, rng),
         [EndingEarnedNarratively(model, world), EmergentEventHonored(model, world)]),
    ]
    scopes = [(name, items, judges) for name, items, judges in scopes if items]
    total = sum(len(items) * len(judges) for _, items, judges in scopes)
    print(f"scoring {total} cases across {sum(len(j) for _, _, j in scopes)} judges "
          f"(≤{args.concurrency} in flight per scope) ...", flush=True)

    # One Dataset per scope, run sequentially: each evaluates all cases × its judges
    # concurrently up to max_concurrency; running scopes in series keeps the in-flight
    # cap at ~64 (concurrent scopes would stack and stall the vLLM queue).
    results: list[tuple[str, list, object]] = []
    for name, items, judges in scopes:
        ds = Dataset(
            name=name,
            cases=[Case(name=cid, inputs=item) for cid, item in items],
            evaluators=judges,
        )
        rep = await ds.evaluate(_identity, max_concurrency=args.concurrency)
        results.append((name, judges, rep))

    # Headline numbers → merge into summary.json, diffed against the previous run.
    merged: dict[str, dict] = {}
    for _name, _judges, rep in results:
        merged.update(run_report.judge_results(rep))
    summary_path = run_report.write_judges(args.run, merged)
    prev = run_report.previous_summary(Path(args.run).parent, args.run)
    prev_judges = (prev or {}).get("judges")

    md = _worst_cases_md(results)
    md_path = Path(args.run) / "judges.md"
    md_path.write_text(md, encoding="utf-8")

    print("\n=== judge summary ===")
    print(run_report.render_judges(merged, prev_judges))
    print(f"\nwrote {summary_path}\n      {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
