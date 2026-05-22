"""Unit coverage for the isolation-eval harness — no live engine.

The aggregation / gating logic (`case_pass_rates`, `suite_metrics`,
`floor_violations`) and the summary.json merge are pure functions of a report
object, so they are tested against a tiny duck-typed fake report. The live
`make eval-deltas` run is the integration check that the real pydantic-evals
report exposes these attributes (see EVALS.md / the PR's verification section).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evals import report as run_report
from evals.isolation.harness import (
    IsoCase,
    Suite,
    case_pass_rates,
    floor_violations,
    run_and_report,
    suite_metrics,
)


# ── fake report mirroring pydantic-evals' ReportCase shape ──


@dataclass
class _Res:
    value: object
    reason: str | None = None


@dataclass
class _Case:
    source_case_name: str | None
    assertions: dict
    scores: dict
    name: str = "case"
    output: object = None


@dataclass
class _Report:
    cases: list
    failures: list = field(default_factory=list)


def _assert_runs(name, bools):
    return [_Case(name, {"held": _Res(b)}, {}) for b in bools]


def _score_runs(name, scores):
    return [_Case(name, {}, {"npc_voice_adherence": _Res(s)}) for s in scores]


def test_case_pass_rate_is_fraction_of_all_assertions_holding():
    rep = _Report(_assert_runs("a", [True, True, True, False]))  # 3/4
    assert case_pass_rates(rep) == {"a": 0.75}


def test_judge_only_cases_are_omitted_from_pass_rates():
    rep = _Report(_score_runs("b", [4.0, 5.0]))
    assert case_pass_rates(rep) == {}  # no assertions → does not gate


def test_suite_metrics_carries_both_pass_rate_and_judge_mean():
    rep = _Report(_assert_runs("a", [True, False]) + _score_runs("b", [4.0, 2.0]))
    m = suite_metrics(rep)
    assert m["a"] == 0.5
    assert m["b/npc_voice_adherence"] == 3.0


def test_floor_violation_fires_below_floor_and_respects_assert_case():
    rep = _Report(_assert_runs("a", [True, True, True, False]))  # 0.75
    gated = IsoCase("a", "", build=lambda w: None, floor=0.8)
    assert floor_violations(rep, [gated]) == ["a: pass-rate 75% < floor 80%"]
    # same rate, but raise the bar passes / lower it clears
    assert floor_violations(rep, [IsoCase("a", "", build=lambda w: None, floor=0.5)]) == []
    # a trend-only case never gates even if it had assertions
    trend = IsoCase("a", "", build=lambda w: None, floor=0.9, assert_case=False)
    assert floor_violations(rep, [trend]) == []


def test_repeat1_falls_back_to_case_name_when_no_source_name():
    rep = _Report([_Case(None, {"held": _Res(True)}, {}, name="solo")])
    assert case_pass_rates(rep) == {"solo": 1.0}


def test_write_summary_block_merges_without_clobbering(tmp_path):
    # a prior block (as `make playtest` would write), then a later isolation merge
    run_report.write_summary_block(tmp_path, "metrics", {"sessions": 3})
    run_report.write_summary_block(tmp_path, "phase1_deltas", {"tom_warm_cold": 0.9})
    run_report.write_judges(tmp_path, {"npc_voice_adherence": {"value": 4.2}})
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"] == {"sessions": 3}
    assert summary["phase1_deltas"] == {"tom_warm_cold": 0.9}
    assert summary["judges"] == {"npc_voice_adherence": {"value": 4.2}}
    assert "generated_at" in summary


@dataclass
class _AlwaysPass(Evaluator):
    def build_serialization_arguments(self) -> dict:
        return {}

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


def test_run_and_report_gates_on_errored_runs(tmp_path):
    # Every repeat errors → pydantic-evals routes them to report.failures, NOT
    # report.cases, so case_pass_rates/floor_violations are blind. The harness must
    # still gate (raise SystemExit) rather than print "all cases above floor".
    async def boom(_name):
        raise RuntimeError("engine down")

    suite = Suite(
        key="s", title="S",
        cases=[IsoCase("c", "", build=lambda w: None)],
        task=boom, evaluators=[_AlwaysPass()],
    )
    with pytest.raises(SystemExit):
        asyncio.run(run_and_report([suite], out_dir=tmp_path / "r", k=3, concurrency=1))


def test_render_summary_block_shows_delta_vs_previous():
    prev = {"phase1_deltas": {"tom_warm_cold": 0.6}}
    out = run_report.render_summary_block(
        "Phase-1 deltas", "phase1_deltas", {"tom_warm_cold": 0.9}, prev
    )
    assert "tom_warm_cold: 0.9" in out
    assert "0.3" in out  # the +0.3 delta is annotated
