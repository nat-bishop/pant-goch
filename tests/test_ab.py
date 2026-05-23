"""A/B harness unit tests — the deterministic logic (pairing, blinding→arm
mapping, tally, human merge, report). The pairwise/themes agent calls use
NativeOutput (which TestModel can't mock), so where a model is needed we stub
`build_judge_agent` with a fake agent returning fixed structured output —
mirroring tests/test_judges.py.
"""
from __future__ import annotations

import json

import pytest

from evals import ab
from evals.judges import pairwise as pw
from evals.judges.pairwise import Preference, PairOutcome, Theme, Themes
from evals.session import SessionArtifact
from evals.transitions import SessionView


def _art(persona: str, sid: str, arm: str) -> SessionArtifact:
    return SessionArtifact(
        persona=persona, session_id=sid, model_engine="e", model_player="p",
        started_at="t", duration_s=1.0, scene_count=1, truncated=False,
        ending_id="end", unlocked_beats=["b1"], case_file_entries=["c1"],
        npc_memory={}, transcript=[{
            "kind": "turn", "scene_id": 0, "location": "loc1", "prose": "p",
            "options": ["o"], "pick_index": 0, "pick_text": "o",
            "applied_beats": ["b1"], "applied_deltas": [], "fireable": ["b1"],
        }], arm=arm,
    )


def _sv(persona: str, sid: str) -> SessionView:
    return SessionView(session_id=sid, persona=persona, ending_id="end", truncated=False,
                       location_path=["loc1"], beats_in_order=["b1"], case_file=["c1"])


# ───────────────────────── experiment loading ─────────────────────────

def test_load_experiment_leave_mode():
    exp = ab.load_experiment("leave_mode")
    assert exp.name == "leave_mode"
    assert [a.name for a in exp.arms] == ["control", "test"]
    assert exp.arms[0].run_config == {"allow_mid_scene_leave": True}
    assert exp.arms[1].run_config == {"allow_mid_scene_leave": False}


def test_load_experiment_rejects_non_two_arms(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\narms:\n  - {name: only, run_config: {}}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ab.load_experiment(str(p))


def test_load_experiment_rejects_wrong_arm_names(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "name: bad\narms:\n  - {name: baseline, run_config: {}}\n"
        "  - {name: variant, run_config: {}}\n", encoding="utf-8")
    with pytest.raises(SystemExit):  # must be named control/test
        ab.load_experiment(str(p))


def test_load_experiment_rejects_unknown_run_config_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "name: bad\narms:\n  - {name: control, run_config: {allow_midscene_leave: true}}\n"
        "  - {name: test, run_config: {allow_mid_scene_leave: false}}\n", encoding="utf-8")
    with pytest.raises(SystemExit):  # typo'd key (missing underscore) caught at load
        ab.load_experiment(str(p))


# ───────────────────────── pairing ─────────────────────────

def test_build_pairs_matches_per_persona_and_zips_to_min():
    control = [_art("diligent", "c1", "control"), _art("diligent", "c2", "control"),
               _art("aggressive", "c3", "control")]
    test = [_art("diligent", "t1", "test"), _art("diligent", "t2", "test")]
    pairs = ab._build_pairs(control, test)
    # diligent has 2+2 → 2 pairs; aggressive only in control → 0 pairs.
    assert [pid for pid, *_ in pairs] == ["diligent-0", "diligent-1"]
    assert all(p[1] == "diligent" for p in pairs)


# ───────────────────────── pairwise blinding → arm ─────────────────────────

class _FakeAgent:
    """Returns a fixed structured output regardless of prompt."""
    def __init__(self, output):
        self._output = output

    async def run(self, _prompt):
        from types import SimpleNamespace
        return SimpleNamespace(output=self._output)


async def test_run_pairwise_maps_blinded_winner_back_to_arm(monkeypatch):
    # Judge always picks "A". With seed fixed, control_is_a is deterministic per
    # pair, so the winner must map to whichever arm was shown as A.
    monkeypatch.setattr(pw, "build_judge_agent",
                        lambda *a, **k: _FakeAgent(Preference(reason="r", winner="A")))
    pairs = [("diligent-0", "diligent", _sv("diligent", "c"), _sv("diligent", "t")),
             ("aggressive-0", "aggressive", _sv("aggressive", "c2"), _sv("aggressive", "t2"))]
    outcomes = await pw.run_pairwise(object(), pairs, concurrency=2, seed=0)
    # Reconstruct the same seeded blinding to assert the mapping is correct.
    import random
    rng = random.Random(0)
    expected = {pid: ("control" if rng.random() < 0.5 else "test")
                for pid, *_ in pairs}
    assert {o.pair_id: o.winner_arm for o in outcomes} == expected


async def test_run_pairwise_tie_yields_none(monkeypatch):
    monkeypatch.setattr(pw, "build_judge_agent",
                        lambda *a, **k: _FakeAgent(Preference(reason="even", winner="tie")))
    pairs = [("diligent-0", "diligent", _sv("diligent", "c"), _sv("diligent", "t"))]
    outcomes = await pw.run_pairwise(object(), pairs, concurrency=1, seed=0)
    assert outcomes[0].winner_arm is None


def test_tally_counts_and_rate():
    outcomes = [
        PairOutcome("p1", "x", "test", "r"),
        PairOutcome("p2", "x", "test", "r"),
        PairOutcome("p3", "x", "control", "r"),
        PairOutcome("p4", "x", None, "tie"),
    ]
    t = pw.tally(outcomes)
    assert t == {"n_pairs": 4, "wins": {"control": 1, "test": 2}, "ties": 1,
                 "win_rate_test": round(2 / 3, 3)}


# ───────────────────────── pairs.jsonl + human merge ─────────────────────────

def test_write_pairs_and_merge_human(tmp_path):
    pairs = [("diligent-0", "diligent", _sv("diligent", "c"), _sv("diligent", "t")),
             ("aggressive-0", "aggressive", _sv("aggressive", "c2"), _sv("aggressive", "t2"))]
    ab._write_pairs(tmp_path, pairs, seed=0)
    lines = (tmp_path / "pairs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert {"pair_id", "persona", "A", "B"} <= rec.keys()
    key = json.loads((tmp_path / "pairs.key.json").read_text(encoding="utf-8"))
    # Each pair's A/B map to the two distinct arms.
    for pid, m in key.items():
        assert {m["A"], m["B"]} == {"control", "test"}

    # Human picks "A" for every pair → maps to whatever arm is A in the key.
    (tmp_path / "human_ratings.jsonl").write_text(
        "\n".join(json.dumps({"pair_id": p, "choice": "A"}) for p in key), encoding="utf-8")
    # LLM picked control for both → agreement = fraction of pairs whose A is control.
    llm = {p: "control" for p in key}
    merged = ab._merge_human(tmp_path, llm)
    a_is_control = sum(1 for m in key.values() if m["A"] == "control")
    assert merged["n"] == 2
    assert merged["agreement_with_llm"] == round(a_is_control / 2, 3)


def test_merge_human_absent_returns_none(tmp_path):
    assert ab._merge_human(tmp_path, {}) is None


# ───────────────────────── report rendering ─────────────────────────

def _summary(**extra) -> dict:
    base_metrics = {
        "sessions": 8, "ending_entropy_normalized": 0.5, "beats_fired": 10,
        "filler_turn_ratio": 0.2, "post_exhaustion_turn_ratio": 0.1,
        "mean_pairwise_beat_jaccard_distance": 0.4, "persona_outcome_separation": 0.3,
        "endings_reached": 3, "beats_never_fired": 2, "location_coverage": 5,
        "median_scenes_per_session": 6, "median_distinct_locations_per_session": 4,
    }
    test_metrics = {**base_metrics, "filler_turn_ratio": 0.35, "beats_fired": 12}
    s = {
        "generated_at": "now", "experiment": "leave_mode", "description": "d",
        "arms": {"control": {"allow_mid_scene_leave": True},
                 "test": {"allow_mid_scene_leave": False}},
        "metrics": {"control": base_metrics, "test": test_metrics},
        "invariants": {"control": {}, "test": {}},
    }
    s.update(extra)
    return s


def test_render_report_north_star_diff_and_sections():
    summary = _summary(
        pairwise={"n_pairs": 6, "wins": {"control": 2, "test": 3}, "ties": 1, "win_rate_test": 0.6},
        themes=[Theme(label="momentum", favors_arm="control", frequency=4,
                      summary="control scenes advanced").model_dump()],
        human={"n": 6, "win_rate_test": 0.5, "agreement_with_llm": 0.83},
    )
    md = ab._render_report(summary)
    assert "North star (test vs control)" in md
    assert "filler_turn_ratio" in md
    # filler rose 0.2→0.35 and higher-is-worse → flagged ⚠️ in the delta.
    assert "⚠️" in md
    assert "Common themes" in md and "momentum" in md
    assert "Pairwise preference" in md
    assert "Human ratings" in md and "0.83" in md
