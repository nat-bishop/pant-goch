"""Generic isolation-eval runner, on top of pydantic-evals.

A `Suite` is one agent / property: a list of `IsoCase` frozen contexts, a `task`
that builds each case's `State` and calls the agent once, and the evaluators that
score the emitted output. `run_suite` wraps the cases in a `Dataset` and evaluates
with `repeat=k` — so the K-sample loop, concurrency, and per-case aggregation come
from the framework, not a hand-rolled runner (the original standalone delta script
re-implemented all three).

Scoring follows the EVALS.md doctrine:
  * deterministic design rules → **bool assertions** → land in `case.assertions` →
    aggregated to a per-case pass-rate → **gated** by a floor.
  * reused LLM judges (`evals.judges`) → **Scored** floats → land in `case.scores` →
    trended in summary.json, never gated.

The per-case pass-rate is computed by grouping `report.cases` on `source_case_name`
(set by pydantic-evals when `repeat>1`) and taking the fraction of repeats in which
*every* bool assertion held — robust without depending on the report's internal
aggregate object.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator

from handler.state import State
from handler.world import World

from .. import report as run_report

# Anti-flake default: the floor catches a case *collapsing* (the captured attention
# bug drove its case from ~0.9 to ~0), not 0.92→0.88 drift — that is the summary.json
# diff's job. Sized to sit below a healthy ~0.9 at K≥16 without tripping on binomial
# noise. Override per case via `IsoCase.floor` where a case is genuinely harder.
DEFAULT_FLOOR = 0.8


@dataclass
class IsoCase:
    """One frozen turn context for an isolation eval.

    `build` constructs the exact `State` the agent reads. `meta` is suite-specific
    data handed to the evaluators via `Case.metadata` (the deltas `Expect`; the
    chronicler's prior case-file entries). `prose` is the fixture prose fed to the
    Phase-2 agents that take a turn's prose as input (choice author, etc.).
    `assert_case=False` marks a trend-only case (its judge score is watched, nothing
    gates); such a case is skipped by `floor_violations`.
    """

    name: str
    scenario: str
    build: Callable[[World], State]
    meta: object = None
    prose: str | None = None
    floor: float = DEFAULT_FLOOR
    assert_case: bool = True


@dataclass
class Suite:
    """One agent / property under isolation test.

    `task(case_name) -> output` builds the frozen state for that case and calls the
    agent once, returning whatever the evaluators score (a `Diff`, a `Transition`, a
    `ChronicleEntry`). `evaluators` apply to every case in the suite (mixing
    deterministic asserts and reused judges). `show_outputs` prints the distribution
    of emitted outputs per case — invaluable for the deltas suite (`9x {attention+1}`).
    """

    key: str          # summary.json block key, e.g. "phase1_deltas"
    title: str        # report heading
    cases: list[IsoCase]
    task: Callable[[str], Awaitable]
    evaluators: list[Evaluator]
    show_outputs: bool = False


async def run_suite(suite: Suite, *, k: int, concurrency: int):
    """Evaluate one suite K times per case; returns the pydantic-evals report.

    inputs = the case *name* (a str) so each of the K repeats rebuilds the state
    fresh — mirroring `evals.run`, which passes a persona name and builds the player
    in the task. Per-case data rides on `Case.metadata`.
    """
    ds = Dataset(
        name=suite.key,
        cases=[
            Case(name=c.name, inputs=c.name, metadata=c.meta) for c in suite.cases
        ],
        evaluators=suite.evaluators,
    )
    return await ds.evaluate(suite.task, max_concurrency=concurrency, repeat=k)


def _source_name(case) -> str:
    """The original case name, stripping the `[run/k]` suffix pydantic-evals adds
    under `repeat>1`. `source_case_name` is None when repeat==1."""
    return getattr(case, "source_case_name", None) or case.name


def case_pass_rates(report) -> dict[str, float]:
    """Per-case pass-rate over the K repeats: the fraction of runs in which *every*
    bool assertion on that run held. Cases that emit no assertions (judge-only,
    trend) are omitted — they do not gate."""
    by_case: dict[str, list[bool]] = defaultdict(list)
    for case in report.cases:
        if case.assertions:
            by_case[_source_name(case)].append(
                all(bool(r.value) for r in case.assertions.values())
            )
    return {n: round(sum(v) / len(v), 3) for n, v in by_case.items() if v}


def suite_metrics(report) -> dict[str, float]:
    """Flat {name: float} block for summary.json, stable across runs so it diffs
    cleanly: per-case assertion pass-rate, plus the mean of each Scored judge keyed
    `<case>/<judge>`."""
    out: dict[str, float] = dict(case_pass_rates(report))
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case in report.cases:
        src = _source_name(case)
        for sname, r in case.scores.items():
            scores[(src, sname)].append(float(r.value))
    for (src, sname), vals in scores.items():
        out[f"{src}/{sname}"] = round(sum(vals) / len(vals), 3)
    return out


def output_distribution(report) -> dict[str, dict[str, int]]:
    """Per-case histogram of emitted outputs (`str(case.output)`), most-common first
    rendered by the caller. Reads `case.output`, which pydantic-evals retains."""
    by: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in report.cases:
        by[_source_name(case)][str(getattr(case, "output", None))] += 1
    return by


def floor_violations(report, cases: list[IsoCase]) -> list[str]:
    """ASSERTED cases whose pass-rate fell below their floor — the CI gate signal."""
    rates = case_pass_rates(report)
    by_name = {c.name: c for c in cases}
    out: list[str] = []
    for name, rate in sorted(rates.items()):
        c = by_name.get(name)
        if c is not None and c.assert_case and rate < c.floor:
            out.append(f"{name}: pass-rate {rate:.0%} < floor {c.floor:.0%}")
    return out


async def run_and_report(
    suites: list[Suite],
    *,
    out_dir: Path,
    k: int,
    concurrency: int,
    gate: bool = True,
) -> int:
    """Run each suite, merge its block into summary.json, print the diffed report,
    and (when `gate`) raise on any floor breach. Returns the breach count (0 = clean)
    for callers that prefer to inspect rather than exit."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    breaches: list[str] = []
    for suite in suites:
        report = await run_suite(suite, k=k, concurrency=concurrency)
        metrics = suite_metrics(report)
        run_report.write_summary_block(out_dir, suite.key, metrics)
        prev = run_report.previous_summary(out_dir.parent, out_dir)
        print("\n" + run_report.render_summary_block(suite.title, suite.key, metrics, prev))
        if suite.show_outputs:
            for name, hist in output_distribution(report).items():
                print(f"  {name}:")
                for out_str, n in sorted(hist.items(), key=lambda kv: -kv[1]):
                    print(f"    {n:>2}x  {out_str}")
        # Errored case-runs land in `report.failures`, NOT `report.cases`, so they are
        # invisible to `case_pass_rates`/`floor_violations` — a case whose every repeat
        # errored would otherwise pass the gate silently. Gate on them explicitly: an
        # uncaught agent error is a regression the gate must not hide. (The tasks catch
        # the expected UnexpectedModelBehavior runaway and score it as a miss, so a
        # failure reaching here is genuinely unexpected.)
        if report.failures:
            breaches.append(f"[{suite.key}] {len(report.failures)} case-run(s) errored")
        breaches += [f"[{suite.key}] {v}" for v in floor_violations(report, suite.cases)]

    if breaches:
        print(f"\n{'='*60}\nFLOOR BREACHES ({len(breaches)}):")
        for b in breaches:
            print(f"  ✗ {b}")
        if gate:
            raise SystemExit(f"{len(breaches)} isolation floor breach(es)")
    else:
        print("\n✅ all asserted isolation cases above floor")
    return len(breaches)
