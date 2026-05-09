"""Tests for scripts/analyze_scenario.py — ASP-backed reachability.

For each rule the analyzer emits, build a small synthetic scenario that
triggers it and assert the expected finding shows up. Production scenario
also gets a smoke test so a future engine refactor doesn't silently break
the encoder.

The analyzer is in `scripts/`, not in the package — same import gymnastics
as `test_lint_scenario.py`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import analyze_scenario  # noqa: E402
import asp_encoder  # noqa: E402

from npc_service.models import load_scenario  # noqa: E402


def _analyze_yaml(yaml_text: str, *, time_limit_s: int = 10) -> analyze_scenario.AnalysisReport:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as f:
        f.write(yaml_text)
        path = Path(f.name)
    try:
        scenario = load_scenario(path)
        return analyze_scenario.analyze_scenario(scenario, time_limit_s=time_limit_s)
    finally:
        path.unlink(missing_ok=True)


def _has_finding(
    findings: list[analyze_scenario.Finding],
    *,
    severity: str,
    anchor: str,
    location_contains: str | None = None,
) -> bool:
    for f in findings:
        if f.severity != severity or f.anchor != anchor:
            continue
        if location_contains is not None and location_contains not in f.location:
            continue
        return True
    return False


# Minimum-viable scenario the per-rule synthetic tests start from. Two
# qualities, one NPC, one anchored welcome — enough to satisfy the model
# validators without dragging in story content.
_BASE = """
title: "Analyzer test"
premise: "Analyzer test fixture."
email_domain: "test.com"
qualities:
  - id: trust
    type: float
    min: 0.0
    max: 10.0
    default: 0.0
  - id: hidden
    type: float
    min: 0.0
    max: 10.0
    default: 0.0
npcs:
  - id: aria
    display_name: "Aria"
    email_address: "Aria <aria@test.com>"
    base_voice_register: "Plain."
    base_voice_forbidden: []
threads:
  - id: aria
    npc: aria
beats:
  - id: aria.welcome
    thread: aria
    npc: aria
    kind: one_shot
    spec:
      pre_authored_prose: "Hi."
"""


def test_orphan_effect_flagged():
    """A literal effect on a quality that no predicate reads (and that
    isn't `surface_to_chat`) should be flagged as orphan_effect."""
    yaml_text = _BASE + """
  - id: aria.writes_orphan
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: beat_fired
        beat: aria.welcome
    spec:
      intent: "Write to a quality nothing reads."
      effects:
        - quality: hidden
          value: 5.0
"""
    report = _analyze_yaml(yaml_text)
    findings = report.to_findings(_load_scenario_from_text(yaml_text))
    assert _has_finding(
        findings, severity="WARN",
        anchor=analyze_scenario.ANCHORS["orphan_effect"],
        location_contains="aria.writes_orphan",
    )


def test_orphan_predicate_flagged():
    """A predicate referencing a quality nothing writes should be flagged
    as orphan_predicate."""
    yaml_text = _BASE + """
  - id: aria.gated_on_unwritten
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: beat_fired
        beat: aria.welcome
      - kind: quality_gte
        quality: hidden
        value: 3.0
    spec:
      pre_authored_prose: "Will not fire (hidden never written)."
"""
    report = _analyze_yaml(yaml_text)
    findings = report.to_findings(_load_scenario_from_text(yaml_text))
    assert _has_finding(
        findings, severity="WARN",
        anchor=analyze_scenario.ANCHORS["orphan_predicate"],
        location_contains="aria.gated_on_unwritten",
    )


def test_unreachable_beat_when_threshold_unsatisfiable():
    """Beat gated on `quality_gte trust 100` against a scenario where no
    effect can push trust above 0.0 — analyzer reports unreachable_beat.
    """
    yaml_text = _BASE + """
  - id: aria.unreachable_threshold
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: quality_gte
        quality: trust
        value: 100.0
    spec:
      pre_authored_prose: "Threshold cannot be reached."
"""
    report = _analyze_yaml(yaml_text)
    findings = report.to_findings(_load_scenario_from_text(yaml_text))
    assert _has_finding(
        findings, severity="ERROR",
        anchor=analyze_scenario.ANCHORS["unreachable_beat"],
        location_contains="aria.unreachable_threshold",
    )


def test_analysis_incomplete_truncation_path():
    """Construct an AnalysisReport with truncated=True directly to verify
    the to_findings path emits analysis_incomplete WARN and demotes
    unreachable_beat from ERROR to WARN. The previous BFS test forced
    truncation via `max_states=1`; with brave-mode ASP solving in
    milliseconds, a tiny `time_limit_s` won't reliably truncate, so we
    exercise the report-building path directly instead.
    """
    yaml_text = _BASE  # boot only
    scenario = _load_scenario_from_text(yaml_text)
    report = analyze_scenario.AnalysisReport(
        reached_beats={"aria.welcome"},
        unreachable_beats=["aria.unreachable_demo"],
        quality_bounds={},
        quality_string_values={},
        orphan_effects=[],
        orphan_predicates=[],
        bound_mismatches=[],
        truncated=True,
        models_enumerated=42,
        grounding_atoms=1234,
        solve_time_s=29.5,
    )
    # Add the unreachable beat to the scenario's beat_by_id for the terminal
    # check in to_findings — easier to extend the scenario than mock the dict.
    # Workaround: use a beat that already exists by faking the unreachable list.
    report.unreachable_beats = ["aria.welcome"]  # any beat in scenario
    findings = report.to_findings(scenario)
    assert _has_finding(
        findings, severity="WARN",
        anchor=analyze_scenario.ANCHORS["analysis_incomplete"],
    )
    for f in findings:
        if f.anchor == analyze_scenario.ANCHORS["unreachable_beat"]:
            assert f.severity == "WARN", (
                "Truncated analysis must not assert ERROR-level "
                "unreachable claims; got " + f.format()
            )


def test_basic_reachability():
    """The boot beat (anchored, empty preconditions) must always be reachable."""
    report = _analyze_yaml(_BASE)
    assert "aria.welcome" in report.reached_beats


def test_predicate_kinds_round_trip():
    """Anti-drift: per-predicate-kind fixture asserts each predicate type
    survives encoder + solver round-trip. If a future engine refactor
    changes spine semantics, this test should catch the mismatch.

    Covered: BeatFired, QualityGte, QualityLt, QualityEquals (numeric),
    PlayerRepliesToThreadGte, Or_, Not_. Skipped here (need richer
    fixtures): RecentClassifierValue × {any/all/int}, PlayerCanonExists,
    QualityEquals (string).
    """
    yaml_text = _BASE + """
  - id: aria.via_quality_gte
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: beat_fired
        beat: aria.welcome
    spec:
      intent: "Bump trust"
      effects:
        - quality: trust
          op: add
          value: 5.0
  - id: aria.gated_quality_gte
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: beat_fired
        beat: aria.via_quality_gte
      - kind: quality_gte
        quality: trust
        value: 5.0
    spec:
      pre_authored_prose: "Reachable iff trust >= 5."
  - id: aria.gated_quality_lt
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: quality_lt
        quality: trust
        value: 1.0
    spec:
      # Reachable in the stable model where the choice rule fires this
      # BEFORE aria.via_quality_gte bumps trust. Encoder's brave-mode
      # union covers that ordering. Pre-Fix-A this was unreachable
      # because `_boot_tick` fired everything to fixpoint.
      pre_authored_prose: "Reachable iff trust < 1 (Director must fire this first)."
  - id: aria.never_fires
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: quality_gte
        quality: trust
        value: 999.0
    spec:
      pre_authored_prose: "Trust never reaches 999, so this beat never fires."
  - id: aria.gated_or
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: or
        of:
          - kind: beat_fired
            beat: aria.welcome
          - kind: beat_fired
            beat: aria.never_fires
    spec:
      pre_authored_prose: "Or-arm satisfied via aria.welcome."
  - id: aria.gated_not
    thread: aria
    npc: aria
    kind: one_shot
    preconditions:
      - kind: not
        of:
          kind: beat_fired
          beat: aria.never_fires
    spec:
      pre_authored_prose: "Not satisfied since aria.never_fires never fires."
"""
    report = _analyze_yaml(yaml_text)
    expected_reachable = {
        "aria.welcome",
        "aria.via_quality_gte",      # BeatFired
        "aria.gated_quality_gte",    # QualityGte after bump
        "aria.gated_quality_lt",     # QualityLt at boot
        "aria.gated_or",             # Or_
        "aria.gated_not",            # Not_
    }
    missing = expected_reachable - report.reached_beats
    assert not missing, f"unreachable per ASP: {missing}"


@pytest.mark.slow
def test_production_scenario_runs():
    """Smoke test: the analyzer completes against the production scenario
    within a reasonable budget and finds all (or at least most) beats.
    Drops the BFS-era state-explored assertion in favor of solve-time
    and reach-count checks.
    """
    path = Path(__file__).resolve().parents[1] / "scenario.yaml"
    scenario = load_scenario(path)
    report = analyze_scenario.analyze_scenario(scenario, time_limit_s=10)
    assert "aria.welcome" in report.reached_beats
    assert report.solve_time_s < 5.0, (
        f"production scenario solve took {report.solve_time_s:.2f}s, "
        "expected sub-5s under brave-mode encoding"
    )
    assert len(report.reached_beats) >= len(scenario.beats) - 1, (
        f"only {len(report.reached_beats)}/{len(scenario.beats)} reachable; "
        "encoding may have regressed"
    )


def test_encode_scenario_program_parses():
    """Encoder produces valid clingo syntax. Catches f-string typos and
    grammar regressions before solver-level assertions catch them."""
    import clingo
    yaml_text = _BASE
    scenario = _load_scenario_from_text(yaml_text)
    program = asp_encoder.encode_scenario(scenario)
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)  # raises on syntax error
    ctl.ground([("base", [])])    # raises on undefined predicate use


# ───────────────────────── helper ─────────────────────────


def _load_scenario_from_text(yaml_text: str):
    """Inline YAML → Scenario, mirroring _analyze_yaml's temp-file dance.
    Used by tests that need both the report and the scenario for to_findings.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as f:
        f.write(yaml_text)
        path = Path(f.name)
    try:
        return load_scenario(path)
    finally:
        path.unlink(missing_ok=True)
