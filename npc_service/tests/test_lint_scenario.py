"""Negative tests for the scenario lint script.

For each of the seven author-facing checks the lint catches, we inject the
failure mode into a minimal scenario and assert the expected finding (with
correct severity + rule anchor) appears. Also asserts the production scenario
and the existing test fixture both lint clean (exit 0).

The lint script lives outside the `npc_service` package — at
`npc_service/scripts/lint_scenario.py`. We add its directory to sys.path
here so the tests can import its check functions directly.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path
from textwrap import dedent

import pytest

# The lint script is at <repo>/npc_service/scripts/lint_scenario.py;
# this test file is at <repo>/npc_service/tests/test_lint_scenario.py.
_LINT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_LINT_DIR))
import lint_scenario  # noqa: E402

from npc_service.models import load_scenario  # noqa: E402


# ───────────────────────── helpers ─────────────────────────


def _lint_yaml(yaml_text: str) -> list[lint_scenario.Finding]:
    """Write `yaml_text` to a temp file, load via npc_service, run lint."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(yaml_text)
        path = Path(f.name)
    try:
        scenario = load_scenario(path)
        return lint_scenario.lint_scenario(scenario)
    finally:
        path.unlink(missing_ok=True)


def _scenario_with_extra_beats(extra_beats_yaml: str) -> str:
    """Append additional beats (written at zero indent) to `_BASE` at the
    correct 2-space indent for `beats:` list items."""
    body = textwrap.dedent(extra_beats_yaml).strip("\n") + "\n"
    return _BASE + textwrap.indent(body, "  ")


def _has_finding(
    findings: list[lint_scenario.Finding],
    *,
    severity: str,
    anchor: str,
    location_contains: str | None = None,
) -> bool:
    """True iff at least one finding matches the (severity, anchor[, location]) tuple."""
    for f in findings:
        if f.severity != severity or f.anchor != anchor:
            continue
        if location_contains is not None and location_contains not in f.location:
            continue
        return True
    return False


# Minimal viable base scenario. Each negative test starts from something
# close to this and injects exactly one failure mode.
_BASE = """
title: "Lint test"
premise: "Lint test fixture."
email_domain: "test.com"
qualities:
  - id: engagement
    type: float
    min: 0.0
    max: 10.0
    default: 0.0
  - id: bargain_state
    type: string
    default: untouched
    surface_to_chat: true
    allowed_values: [untouched, offered, accepted, refused]
npcs:
  - id: aria
    display_name: "Aria"
    email_address: "Aria <aria@test.com>"
    base_voice_register: "Plain."
    base_voice_forbidden:
      - "the words 'monster', 'creature'"
      - "any reference to 'father'"
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


# ───────────────────────── shipping scenarios lint clean ─────────────────────────


def test_production_scenario_lints_clean():
    """The shipping `npc_service/scenario.yaml` must lint clean (no ERROR /
    WARN; INFO findings allowed). If this regresses, fix the scenario or
    revisit the lint heuristic."""
    path = Path(__file__).resolve().parents[1] / "scenario.yaml"
    scenario = load_scenario(path)
    findings = lint_scenario.lint_scenario(scenario)
    bad = [f for f in findings if f.severity in ("ERROR", "WARN")]
    assert not bad, "production scenario triggered ERROR/WARN: " + "; ".join(
        f.format() for f in bad
    )


def test_test_fixture_lints_clean():
    """tests/fixtures/test_scenario.yaml must also lint clean."""
    path = Path(__file__).resolve().parent / "fixtures" / "test_scenario.yaml"
    scenario = load_scenario(path)
    findings = lint_scenario.lint_scenario(scenario)
    bad = [f for f in findings if f.severity in ("ERROR", "WARN")]
    assert not bad, "test fixture triggered ERROR/WARN: " + "; ".join(
        f.format() for f in bad
    )


# ───────────────────────── per-check negative tests ─────────────────────────


def test_scene_write_collision_flagged():
    """A scene beat literal-writes a quality that another beat also writes
    literally, with no `Not_(BeatFired ...)` in either direction.
    Expect ERROR with the scene-write-collision anchor."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.loop
        thread: aria
        npc: aria
        kind: scene
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 1
        spec:
          intent: "Loop forever."
          exit_when:
            - kind: player_replies_to_thread_gte
              thread: aria
              value: 99
          effects:
            - quality: engagement
              value: 0.0
      - id: aria.collision
        thread: aria
        npc: aria
        kind: one_shot
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 3
        spec:
          intent: "Write engagement to a different value."
          effects:
            - quality: engagement
              value: 5.0
    """)
    findings = _lint_yaml(yaml_text)
    assert _has_finding(
        findings,
        severity="ERROR",
        anchor="scene-write-collision",
        location_contains="aria.loop",
    ), [f.format() for f in findings]


def test_scene_write_collision_silenced_by_mutex():
    """Same setup but the colliding beat has `Not_(BeatFired aria.loop)` —
    no finding."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.loop
        thread: aria
        npc: aria
        kind: scene
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 1
          - kind: not
            of:
              kind: beat_fired
              beat: aria.collision
        spec:
          intent: "Loop until collision."
          exit_when:
            - kind: player_replies_to_thread_gte
              thread: aria
              value: 99
          effects:
            - quality: engagement
              value: 0.0
      - id: aria.collision
        thread: aria
        npc: aria
        kind: one_shot
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 3
        spec:
          intent: "Write engagement to a different value."
          effects:
            - quality: engagement
              value: 5.0
    """)
    findings = _lint_yaml(yaml_text)
    assert not _has_finding(
        findings, severity="ERROR", anchor="scene-write-collision"
    ), [f.format() for f in findings]


def test_validate_facts_on_scene_flagged():
    """validate_facts: true on a scene beat → ERROR."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.loop
        thread: aria
        npc: aria
        kind: scene
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 1
        spec:
          intent: "Loop."
          exit_when:
            - kind: player_replies_to_thread_gte
              thread: aria
              value: 99
          must_include_facts:
            - "asks one question"
          validate_facts: true
    """)
    findings = _lint_yaml(yaml_text)
    assert _has_finding(
        findings,
        severity="ERROR",
        anchor="validate-facts-on-scene",
        location_contains="aria.loop",
    ), [f.format() for f in findings]


def test_pre_authored_post_boot_flagged():
    """pre_authored_prose with player_replies_to_thread_gte ≥ 1 in
    preconditions → ERROR (linearity-trap antipattern). The base scenario's
    aria.welcome has pre_authored_prose with NO preconditions (boot, fine);
    we add a SECOND beat with pre-authored prose AND a reply gate."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.late_pre_authored
        thread: aria
        npc: aria
        kind: one_shot
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 1
        spec:
          pre_authored_prose: "Should not fire post-boot."
    """)
    findings = _lint_yaml(yaml_text)
    assert _has_finding(
        findings,
        severity="ERROR",
        anchor="pre-authored-post-boot",
        location_contains="aria.late_pre_authored",
    ), [f.format() for f in findings]


def test_forbidden_lexical_in_intent_flagged():
    """A beat's spec.intent contains a forbidden token from the assigned
    NPC's base_voice_forbidden list → WARN."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.bad_intent
        thread: aria
        npc: aria
        kind: one_shot
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 1
        spec:
          intent: "Aria addresses the player as a monster."
    """)
    findings = _lint_yaml(yaml_text)
    assert _has_finding(
        findings,
        severity="WARN",
        anchor="forbidden-lexical",
        location_contains="aria.bad_intent",
    ), [f.format() for f in findings]


def test_forbidden_lexical_in_must_not_not_flagged():
    """spec.must_not naturally contains forbidden phrases (it IS the deny
    list); the lint must NOT flag tokens appearing there."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.fine_must_not
        thread: aria
        npc: aria
        kind: one_shot
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 1
        spec:
          intent: "Aria writes a normal email."
          must_not:
            - "the words 'monster' or 'creature'"
            - "any reference to 'father'"
    """)
    findings = _lint_yaml(yaml_text)
    assert not _has_finding(
        findings, severity="WARN", anchor="forbidden-lexical",
        location_contains="aria.fine_must_not",
    ), [f.format() for f in findings]


def test_classify_on_silent_npc_flagged():
    """A beat assigned to an NPC with replies_enabled=False can't have
    classify directives — ERROR."""
    yaml_text = dedent("""
      title: "Lint test"
      premise: "."
      email_domain: "test.com"
      qualities:
        - id: engagement
          type: float
          default: 0.0
      npcs:
        - id: aria
          display_name: "Aria"
          email_address: "<aria@test.com>"
        - id: cpo
          display_name: "Office"
          email_address: "<office@test.com>"
          replies_enabled: false
      threads:
        - id: aria
          npc: aria
        - id: cpo
          npc: cpo
      beats:
        - id: aria.welcome
          thread: aria
          npc: aria
          kind: one_shot
          spec:
            pre_authored_prose: "Hi."
        - id: cpo.notice
          thread: cpo
          npc: cpo
          kind: one_shot
          preconditions:
            - kind: player_replies_to_thread_gte
              thread: aria
              value: 1
          spec:
            intent: "An institutional notice."
            effects:
              - quality: engagement
                op: set
                classify:
                  question: "Did the player engage?"
                  allowed_values: [engaged, deflected]
                  map: { engaged: 1.0, deflected: 0.0 }
    """)
    findings = _lint_yaml(yaml_text)
    assert _has_finding(
        findings,
        severity="ERROR",
        anchor="classify-on-silent-npc",
        location_contains="cpo.notice",
    ), [f.format() for f in findings]


def test_reachability_contradiction_flagged():
    """A beat that requires both BeatFired X and Not_(BeatFired X) — WARN."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.foo
        thread: aria
        npc: aria
        kind: one_shot
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 1
        spec:
          intent: "First beat."
      - id: aria.contradicts
        thread: aria
        npc: aria
        kind: one_shot
        preconditions:
          - kind: beat_fired
            beat: aria.foo
          - kind: not
            of:
              kind: beat_fired
              beat: aria.foo
        spec:
          intent: "Unreachable."
    """)
    findings = _lint_yaml(yaml_text)
    assert _has_finding(
        findings,
        severity="WARN",
        anchor="reachability-contradiction",
        location_contains="aria.contradicts",
    ), [f.format() for f in findings]


def test_terminal_overlap_hard_flagged():
    """Two terminals with overlapping hard constraints (quality ranges,
    no recent_classifier_value, no Not_(BeatFired) mutex) → WARN."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.complete_a
        thread: aria
        npc: aria
        kind: one_shot
        ends_session: true
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 5
          - kind: quality_gte
            quality: engagement
            value: 3.0
        spec:
          intent: "Terminal A."
      - id: aria.complete_b
        thread: aria
        npc: aria
        kind: one_shot
        ends_session: true
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 5
          - kind: quality_gte
            quality: engagement
            value: 4.0
        spec:
          intent: "Terminal B."
    """)
    findings = _lint_yaml(yaml_text)
    # Both terminals require engagement ≥ N (overlapping ranges) and ≥5
    # replies. No recent_classifier_value, no mutex. Hard overlap → WARN.
    assert _has_finding(
        findings, severity="WARN", anchor="terminal-overlap"
    ), [f.format() for f in findings]


def test_terminal_overlap_skipped_with_one_terminal():
    """A scenario with only one terminal — overlap check is skipped."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.complete_only
        thread: aria
        npc: aria
        kind: one_shot
        ends_session: true
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 5
        spec:
          intent: "Only terminal."
    """)
    findings = _lint_yaml(yaml_text)
    # No terminal-overlap findings of any severity.
    overlap_findings = [
        f for f in findings if f.anchor == "terminal-overlap"
    ]
    assert not overlap_findings, [f.format() for f in overlap_findings]


def test_terminal_overlap_authored_mutex_demotes_to_info():
    """Two terminals where one explicitly excludes the other via
    `Not_(BeatFired)` — demoted to INFO (not WARN)."""
    yaml_text = _scenario_with_extra_beats("""
      - id: aria.complete_a
        thread: aria
        npc: aria
        kind: one_shot
        ends_session: true
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 5
          - kind: quality_gte
            quality: engagement
            value: 3.0
        spec:
          intent: "Terminal A."
      - id: aria.complete_b
        thread: aria
        npc: aria
        kind: one_shot
        ends_session: true
        preconditions:
          - kind: player_replies_to_thread_gte
            thread: aria
            value: 5
          - kind: not
            of:
              kind: beat_fired
              beat: aria.complete_a
        spec:
          intent: "Terminal B (fallback)."
    """)
    findings = _lint_yaml(yaml_text)
    overlap_warns = [
        f for f in findings
        if f.anchor == "terminal-overlap" and f.severity == "WARN"
    ]
    assert not overlap_warns, [f.format() for f in overlap_warns]
    # Should still have an INFO-level overlap finding for the pair.
    overlap_infos = [
        f for f in findings
        if f.anchor == "terminal-overlap" and f.severity == "INFO"
    ]
    assert overlap_infos, [f.format() for f in findings]


# ───────────────────────── allow-list signal ─────────────────────────


def test_forbidden_lexical_skips_use_segment():
    """A dossier ban entry like `"any reference to X ('a', 'b'); use 'c'"` —
    the `use 'c'` segment should be skipped, so 'c' appearing in a beat's
    intent is NOT flagged."""
    yaml_text = dedent("""
      title: "Lint test"
      premise: "."
      email_domain: "test.com"
      qualities:
        - id: engagement
          type: float
          default: 0.0
      npcs:
        - id: cpo
          display_name: "Office"
          email_address: "<office@test.com>"
          base_voice_forbidden:
            - "any reference to the player as a person ('they'); use 'the recipient'"
      threads:
        - id: cpo
          npc: cpo
      beats:
        - id: cpo.welcome
          thread: cpo
          npc: cpo
          kind: one_shot
          spec:
            pre_authored_prose: "Hi."
        - id: cpo.notice
          thread: cpo
          npc: cpo
          kind: one_shot
          preconditions:
            - kind: player_replies_to_thread_gte
              thread: cpo
              value: 1
          spec:
            intent: "References 'the recipient' in proper form."
    """)
    findings = _lint_yaml(yaml_text)
    # 'they' should still be a flag candidate (in the ban segment), but
    # 'the recipient' (in the `use ...` segment) should NOT trigger.
    flagged = [
        f for f in findings
        if f.anchor == "forbidden-lexical" and f.location == "cpo.notice"
    ]
    # The intent contains 'the recipient' (allow-listed), not 'they'.
    # Expect zero findings on cpo.notice.
    assert not flagged, [f.format() for f in flagged]
