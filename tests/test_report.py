"""Run report: summary.json round-trip + diffed markdown render."""
from __future__ import annotations

import json
from types import SimpleNamespace

from evals import report as run_report
from evals.metrics import corpus_metrics
from evals.session import SessionArtifact
from pathlib import Path

from handler.world import load_world

FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def _art(**o) -> SessionArtifact:
    base = dict(
        persona="p", session_id="s", model_engine="e", model_player="m",
        started_at="t", duration_s=1.0, scene_count=2, truncated=False,
        ending_id="published_story", unlocked_beats=["tom_named_a_vehicle"],
        case_file_entries=[], npc_memory={}, transcript=[],
    )
    base.update(o)
    return SessionArtifact(**base)


def _fake_invariant_report(pass_map: dict[str, bool]):
    """Duck-typed EvaluationReport: one case whose assertions carry .value."""
    assertions = {n: SimpleNamespace(value=v) for n, v in pass_map.items()}
    case = SimpleNamespace(assertions=assertions)
    return SimpleNamespace(cases=[case])


def test_invariant_pass_rates():
    rep = _fake_invariant_report({"EndingReached": True, "NoPrematureEnding": False})
    rates = run_report.invariant_pass_rates(rep)
    assert rates == {"EndingReached": 1.0, "NoPrematureEnding": 0.0}


def test_emit_writes_summary_and_render(tmp_path):
    world = load_world(FIXTURE_VALID)
    out_dir = tmp_path / "20260101-000000"
    out_dir.mkdir()
    metrics = corpus_metrics([_art()], world)
    rep = _fake_invariant_report({"EndingReached": True})
    path = run_report.emit(out_dir, metrics, rep, console=False)
    assert path.exists()
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert "metrics" in summary and "invariants" in summary
    # No previous run → render notes that, no crash.
    text = run_report.render_report(summary, prev=None)
    assert "no previous run" in text
    assert "North star" in text


def test_render_report_diffs_previous():
    cur = {"generated_at": "now", "metrics": {
        "sessions": 4, "ending_entropy_normalized": 0.8, "beats_fired": 10,
        "filler_turn_ratio": 0.2, "mean_pairwise_beat_jaccard_distance": 0.5,
        "persona_outcome_separation": 0.3, "endings_reached": [], "endings_never_reached": [],
        "beats_total": 20, "beats_never_fired": [], "location_coverage": 0.5,
        "locations_never_visited": [], "beats_per_location": {},
        "median_scenes_per_session": 5, "median_distinct_locations_per_session": 4,
        "longest_common_beat_prefix": [], "ladder_compression": {},
    }, "invariants": {"NoPrematureEnding": 1.0}}
    prev = {"metrics": {**cur["metrics"], "ending_entropy_normalized": 0.3}}
    text = run_report.render_report(cur, prev)
    # entropy rose 0.3 → 0.8 (good direction)
    assert "ending_entropy_normalized" in text
    assert "▲" in text


def test_previous_summary_picks_latest(tmp_path):
    for name in ("20260101-000000", "20260102-000000"):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"tag": name}), encoding="utf-8")
    current = tmp_path / "20260103-000000"
    current.mkdir()
    prev = run_report.previous_summary(tmp_path, current)
    assert prev["tag"] == "20260102-000000"
    # first run: nothing earlier
    assert run_report.previous_summary(tmp_path, tmp_path / "20251231-000000") is None
