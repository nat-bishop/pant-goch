"""Judge layer: transition/session extraction, render, the Evaluator bridge, and
report aggregation.

No live model — judges use NativeOutput (which TestModel can't mock), so the
verdict→EvaluationReason bridge is exercised with a stub judge whose `judge()`
returns fixed verdicts, and aggregation is tested directly over a stub report.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

from evals import report as run_report
from evals.judges.base import AgentJudge
from evals.judges.transition_judges import ChoiceMeaningfulness
from evals.judges.verdicts import PassFail, Scored
from evals.session import SessionArtifact
from evals.transitions import (
    Transition,
    extract_transitions,
    session_view,
)


def _turn(scene_id, location, applied, *, kind="turn", **extra) -> dict:
    base = {
        "kind": kind, "scene_id": scene_id, "location": location, "prose": "p",
        "applied_beats": applied, "fireable": ["x"],
    }
    if kind == "turn":
        base.update(options=["a", "b"], pick_index=0, pick_text="press Tom")
    else:
        base.update(prompt="say something", typed="I lean in")
    base.update(extra)
    return base


def _art(transcript, **o) -> SessionArtifact:
    base = dict(
        persona="p", session_id="sid", model_engine="e", model_player="m",
        started_at="t", duration_s=1.0, scene_count=1, truncated=False,
        ending_id="published_story", unlocked_beats=["b1"],
        case_file_entries=["scene one"], npc_memory={}, transcript=transcript,
    )
    base.update(o)
    return SessionArtifact(**base)


def test_extract_transitions_shapes_fields():
    art = _art([
        _turn(1, "the_post_office", ["b1"]),
        {"kind": "scene_close", "scene_id": 1, "location": "the_post_office", "end": "end_scene"},
        _turn(2, "the_slaughtered_lamb", [], kind="free_text"),
    ])
    ts = extract_transitions(art)
    assert len(ts) == 2
    first, second = ts
    assert first.fired_beats == ["b1"]
    assert first.player_action == "press Tom"
    assert first.is_free_text is False
    assert first.prior_case_file == []  # no scene closed before turn 1
    # second turn is after one scene close → prior case file has one entry
    assert second.is_free_text is True
    assert second.player_action == "I lean in"
    assert second.prior_case_file == ["scene one"]


def test_extract_transitions_skips_legacy():
    art = _art([{"kind": "turn", "scene_id": 1, "location": "x", "prose": "p"}])
    assert extract_transitions(art) == []


def _transition(**o) -> Transition:
    base = dict(session_id="s", persona="p", scene_id=1, location="x",
                prior_case_file=[], player_action="look at box", prose="A box.",
                options=["look closer at the box", "examine the window"],
                fired_beats=[], fireable_beats=[])
    base.update(o)
    return Transition(**base)


def test_choice_meaningfulness_render_includes_options():
    judge = ChoiceMeaningfulness(model=None)  # render needs no model
    text = judge.render(_transition())
    assert "look closer at the box" in text and "examine the window" in text


@dataclass
class _PassFailJudge(AgentJudge):
    """Stub: maps the item (a bool) straight to a PassFail verdict."""
    name: ClassVar[str] = "pf"
    output_type: ClassVar[type] = PassFail
    async def judge(self, item):
        return PassFail(reason="r", passed=item)


@dataclass
class _ScoredJudge(AgentJudge):
    """Stub: maps the item (an int) straight to a Scored verdict."""
    name: ClassVar[str] = "sc"
    output_type: ClassVar[type] = Scored
    async def judge(self, item):
        return Scored(reason="r", score=item)


async def test_evaluate_bridges_verdict_to_evaluation_reason():
    pf = _PassFailJudge(model=None)
    r_true = await pf.evaluate(SimpleNamespace(output=True))
    r_false = await pf.evaluate(SimpleNamespace(output=False))
    assert r_true.value is True and r_true.reason == "r"
    assert r_false.value is False  # bool → assertion

    sc = _ScoredJudge(model=None)
    r = await sc.evaluate(SimpleNamespace(output=4))
    assert r.value == 4.0 and isinstance(r.value, float)  # int score → float score


def test_judge_results_aggregates_passfail_and_scored():
    """`report.judge_results` harvests pass-rate (assertions) and mean score
    (scores) from an EvaluationReport, keyed by judge name."""
    def res(v):
        return SimpleNamespace(value=v, reason="")
    cases = [
        SimpleNamespace(assertions={"pf": res(True)}, scores={"sc": res(2)}),
        SimpleNamespace(assertions={"pf": res(False)}, scores={"sc": res(4)}),
        SimpleNamespace(assertions={"pf": res(True)}, scores={}),  # judge errored here
    ]
    out = run_report.judge_results(SimpleNamespace(cases=cases))
    assert out["pf"] == round(2 / 3, 3)
    assert out["sc"] == 3.0


def test_session_view():
    art = _art([
        _turn(1, "the_old_vicarage", ["b1"]),
        {"kind": "walk", "from": "the_old_vicarage", "to": "the_post_office"},
        _turn(2, "the_post_office", []),
    ])
    sv = session_view(art)
    assert sv.location_path == ["the_old_vicarage", "the_post_office"]
    assert sv.beats_in_order == ["b1"]
    assert sv.ending_id == "published_story"
