"""Static scenario analyzer (ASP-backed).

Encodes scenario reachability as Answer Set Programming via clingo (see
sibling `asp_encoder.py`), replacing the prior bounded BFS state-space
exploration. The ASP encoding handles the same five outputs:

- Which beats are reachable from boot state, and which terminals
- Min/max bounds for every quality across all reachable states
- Orphan effects (writes that no predicate reads) and orphan predicates
  (reads with no writer)
- Bound mismatches (predicate threshold outside the proven quality range)
- Truncation warning when the solver hits its time limit

Why ASP: BFS is exponential in (Director branching × verdict cartesian ×
canon-key combinations) and was already truncating at 50K states on a
13-beat scenario, producing false-positive "unreachable" findings. ASP
with `--enum-mode=brave` computes the union of all stable models in one
pass (~75x speedup on production; sub-second on 100-beat synthetic).

The encoder mirrors engine semantics (`spine.py` predicate evaluator +
effect appliers). When predicate or effect grammar changes, the encoder
must be updated in lockstep — see `asp_encoder.py` and the published
DendryScope paper (Robison & Mateas, AIIDE 2023) for the encoding
rationale and edge-case fixes.

Output: list of Finding objects (same shape as lint_scenario.py) plus a
JSON sidecar to `npc_service/_analysis/<scenario>.json` for future
ScenarioGraph overlay.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import logfire

# Make `from npc_service.X import Y` resolve when the script is invoked
# directly (`python scripts/analyze_scenario.py`) before the package is
# editable-installed. `make lint-scenario` runs via `uv run`, which already
# has the package on sys.path, so this only matters for ad-hoc invocations.
_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

# Standalone analyzer doesn't go through main.py's FastAPI startup, so we
# configure logfire here. Without this, spine's per-write `logfire.info`
# call (one per quality write × tens of thousands of explored states) used
# to fall through logfire's "not configured" path, which prints a full
# traceback to stderr on EVERY call — burning minutes of wall clock.
# `min_level="notice"` short-circuits info-level writes before any span
# work; spine's `logfire.warn` (orphan pending beats etc.) still surfaces.
# `send_to_logfire="if-token-present"` matches main.py: zero cost when
# unset, captures analyzer traces when LOGFIRE_TOKEN is exported.
logfire.configure(
    service_name="npc_service.analyze",
    send_to_logfire="if-token-present",
    console=False,
    min_level="notice",
    scrubbing=False,
)

from npc_service.models import (
    BeatFired,
    Not_,
    Or_,
    PlayerCanonExists,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityEquals,
    QualityGte,
    QualityLt,
    RecentClassifierValue,
    Scenario,
    load_scenario,
)

# `asp_encoder` is a sibling script, not in the package. The sys.path
# prepend at module top covers `npc_service.X` imports; this one resolves
# via the script directory (already on path because we're loaded from it).
import asp_encoder


# Default wall-clock cap for the clingo solve. `--enum-mode=brave` makes
# the solver internally enumerate models to compute the brave union, so
# even with the union being small the underlying search may exceed this
# on pathological scenarios (cf. plan §"grounding blowup" risk).
DEFAULT_TIME_LIMIT_S = 30


# ───────────────────────── finding shape (matches lint_scenario.py) ─────────────────────────


# Anchors to be appended to lint_rules.md.
ANCHORS = {
    "unreachable_beat": "unreachable-beat",
    "unreachable_terminal": "unreachable-terminal",
    "orphan_effect": "orphan-effect",
    "orphan_predicate": "orphan-predicate",
    "quality_bound_mismatch": "quality-bound-mismatch",
    "analysis_incomplete": "analysis-incomplete",
}


@dataclass
class Finding:
    """Same shape as lint_scenario.Finding so the lint driver can fold
    analyzer findings into its existing output stream."""

    severity: str  # "ERROR" | "WARN" | "INFO"
    location: str
    message: str
    anchor: str

    def format(self) -> str:
        return (
            f"{self.severity}: {self.location}: {self.message} "
            f"(see scripts/lint_rules.md#{self.anchor})"
        )


# ───────────────────────── scenario profile ─────────────────────────


@dataclass
class ScenarioProfile:
    """Pre-scan of the scenario identifying which dynamic state actually
    matters for predicate evaluation. Drives orphan detection and supplies
    the encoder with the per-quality / per-thread caps it needs to keep
    grounding bounded.

    `referenced_*` sets are the surface predicates read from. `qualities_*`
    sets pair them against effects to identify orphans.
    """

    referenced_canon_keys: frozenset[str]
    referenced_classifier_qualities: frozenset[str]
    thread_reply_caps: dict[str, int]  # thread → max threshold + 1
    recent_classifier_window_caps: dict[str, int]  # quality → max window in any predicate
    qualities_with_effects: frozenset[str]
    qualities_with_predicates: frozenset[str]

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "ScenarioProfile":
        canon_keys: set[str] = set()
        classifier_qs: set[str] = set()
        thread_caps: dict[str, int] = {}
        recent_caps: dict[str, int] = {}
        quals_predicates: set[str] = set()

        def walk(pred: Predicate) -> None:
            if isinstance(pred, QualityGte):
                quals_predicates.add(pred.quality)
            elif isinstance(pred, QualityLt):
                quals_predicates.add(pred.quality)
            elif isinstance(pred, QualityEquals):
                quals_predicates.add(pred.quality)
            elif isinstance(pred, PlayerCanonExists):
                canon_keys.add(pred.name)
            elif isinstance(pred, PlayerRepliesToThreadGte):
                thread_caps[pred.thread] = max(
                    thread_caps.get(pred.thread, 0), pred.value + 1,
                )
            elif isinstance(pred, RecentClassifierValue):
                classifier_qs.add(pred.quality)
                # RecentClassifierValue READS the quality (via per-axis
                # history). Counts the same as QualityGte for orphan-effect
                # detection — without this, qualities written exclusively
                # for trend-reads false-flag as orphan.
                quals_predicates.add(pred.quality)
                recent_caps[pred.quality] = max(
                    recent_caps.get(pred.quality, 0), pred.window,
                )
            elif isinstance(pred, Not_):
                walk(pred.of)
            elif isinstance(pred, Or_):
                for sub in pred.of:
                    walk(sub)

        for beat in scenario.beats:
            for top in beat.preconditions:
                walk(top)
            for top in beat.spec.exit_when:
                walk(top)
            for effect in beat.spec.effects:
                if effect.when is not None:
                    walk(effect.when)
        for effect in scenario.effects:
            if effect.when is not None:
                walk(effect.when)

        quals_with_effects: set[str] = set()
        for beat in scenario.beats:
            for effect in beat.spec.effects:
                quals_with_effects.add(effect.quality)
        for effect in scenario.effects:
            quals_with_effects.add(effect.quality)

        return cls(
            referenced_canon_keys=frozenset(canon_keys),
            referenced_classifier_qualities=frozenset(classifier_qs),
            thread_reply_caps=thread_caps,
            recent_classifier_window_caps=recent_caps,
            qualities_with_effects=frozenset(quals_with_effects),
            qualities_with_predicates=frozenset(quals_predicates),
        )


# ───────────────────────── analysis report ─────────────────────────


@dataclass
class AnalysisReport:
    reached_beats: set[str]
    unreachable_beats: list[str]
    quality_bounds: dict[str, tuple[float, float]]  # numeric only
    quality_string_values: dict[str, set[str]]  # string-typed qualities
    orphan_effects: list[tuple[str, str]]  # (location, quality)
    orphan_predicates: list[tuple[str, str]]  # (location, quality_or_canon_key)
    bound_mismatches: list[tuple[str, str, str]]  # (location, quality, reason)
    truncated: bool
    models_enumerated: int
    grounding_atoms: int = 0
    solve_time_s: float = 0.0

    def to_findings(self, scenario: Scenario) -> list[Finding]:
        out: list[Finding] = []
        terminals = {b.id for b in scenario.beats if b.ends_session}
        for beat_id in self.unreachable_beats:
            anchor_key = "unreachable_terminal" if beat_id in terminals else "unreachable_beat"
            out.append(Finding(
                severity="ERROR" if not self.truncated else "WARN",
                location=beat_id,
                message=(
                    f"beat {beat_id!r} not reachable from boot"
                    + ("; analysis truncated, claim is provisional" if self.truncated else "")
                ),
                anchor=ANCHORS[anchor_key],
            ))
        for location, quality in self.orphan_effects:
            out.append(Finding(
                severity="WARN",
                location=location,
                message=(
                    f"effect on quality {quality!r} but no predicate reads "
                    f"that quality — write is dead"
                ),
                anchor=ANCHORS["orphan_effect"],
            ))
        for location, target in self.orphan_predicates:
            out.append(Finding(
                severity="WARN",
                location=location,
                message=(
                    f"predicate references {target!r} but nothing writes it — "
                    f"predicate can never become true"
                ),
                anchor=ANCHORS["orphan_predicate"],
            ))
        for location, quality, reason in self.bound_mismatches:
            out.append(Finding(
                severity="ERROR",
                location=location,
                message=f"predicate on quality {quality!r}: {reason}",
                anchor=ANCHORS["quality_bound_mismatch"],
            ))
        if self.truncated:
            out.append(Finding(
                severity="WARN",
                location="(analyzer)",
                message=(
                    f"clingo time limit hit after {self.solve_time_s:g}s "
                    f"({self.models_enumerated} models enumerated); "
                    "reachability claims are partial"
                ),
                anchor=ANCHORS["analysis_incomplete"],
            ))
        return out

    def to_json(self, scenario: Scenario) -> dict:
        beat_records: dict[str, dict] = {}
        all_beats = {b.id for b in scenario.beats}
        for beat_id in all_beats:
            beat_records[beat_id] = {
                "reachable": beat_id in self.reached_beats,
                "is_terminal": scenario.beat_by_id[beat_id].ends_session,
            }
        return {
            "scenario_id": getattr(scenario, "id", "scenario"),
            "models_enumerated": self.models_enumerated,
            "grounding_atoms": self.grounding_atoms,
            "solve_time_s": self.solve_time_s,
            "truncated": self.truncated,
            "beats": beat_records,
            "qualities": {
                q: {
                    "min": lo, "max": hi,
                    "string_values": sorted(self.quality_string_values.get(q, set())),
                }
                for q, (lo, hi) in self.quality_bounds.items()
            } | {
                q: {"min": None, "max": None, "string_values": sorted(vals)}
                for q, vals in self.quality_string_values.items()
                if q not in self.quality_bounds
            },
            "orphans": {
                "effects": [{"location": loc, "quality": q} for loc, q in self.orphan_effects],
                "predicates": [{"location": loc, "target": t} for loc, t in self.orphan_predicates],
            },
            "bound_mismatches": [
                {"location": loc, "quality": q, "reason": r}
                for loc, q, r in self.bound_mismatches
            ],
        }


# ───────────────────────── orphan detection ─────────────────────────


def _find_orphans(
    scenario: Scenario, profile: ScenarioProfile,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Pure spec analysis (no exploration).

    Orphan effects: any effect on a quality that no predicate reads AND
    that isn't `surface_to_chat=True`. Surface qualities are deliberately
    surfaced to the chat agent for rendering — they affect prose, not
    branching, so a "no predicate reads it" lint would false-positive.

    Orphan predicates: any predicate referencing a quality that no effect
    writes. Canon-key orphans aren't reported here — extractor outputs
    are LLM-driven and we can't statically prove a key is never harvested.
    """
    surface_qualities = scenario.surface_to_chat_quality_ids
    orphan_effects: list[tuple[str, str]] = []
    orphan_predicates: list[tuple[str, str]] = []

    # Effect → no predicate reads this quality (and not surfaced to chat).
    for beat in scenario.beats:
        for i, effect in enumerate(beat.spec.effects):
            if (
                effect.quality not in profile.qualities_with_predicates
                and effect.quality not in surface_qualities
            ):
                orphan_effects.append((f"{beat.id}.effects[{i}]", effect.quality))
    for i, effect in enumerate(scenario.effects):
        if (
            effect.quality not in profile.qualities_with_predicates
            and effect.quality not in surface_qualities
        ):
            orphan_effects.append((f"scenario.effects[{i}]", effect.quality))

    # Predicate → references a quality no effect writes.
    seen: set[tuple[str, str, str]] = set()  # dedupe per (location, target, kind)

    def walk_for_orphans(pred: Predicate, location: str) -> None:
        if isinstance(pred, (QualityGte, QualityLt, QualityEquals)):
            if pred.quality not in profile.qualities_with_effects:
                key = (location, pred.quality, "quality")
                if key not in seen:
                    seen.add(key)
                    orphan_predicates.append((location, pred.quality))
        elif isinstance(pred, RecentClassifierValue):
            if pred.quality not in profile.qualities_with_effects:
                key = (location, pred.quality, "recent")
                if key not in seen:
                    seen.add(key)
                    orphan_predicates.append((location, pred.quality))
        elif isinstance(pred, Not_):
            walk_for_orphans(pred.of, location)
        elif isinstance(pred, Or_):
            for sub in pred.of:
                walk_for_orphans(sub, location)

    for beat in scenario.beats:
        for top in beat.preconditions:
            walk_for_orphans(top, f"{beat.id}.preconditions")
        for top in beat.spec.exit_when:
            walk_for_orphans(top, f"{beat.id}.exit_when")

    return orphan_effects, orphan_predicates


# ───────────────────────── bound-mismatch detection ─────────────────────────


def _find_bound_mismatches(
    scenario: Scenario,
    quality_bounds: dict[str, tuple[float, float]],
) -> list[tuple[str, str, str]]:
    """Compare each numeric predicate threshold against the proven (min, max)
    for that quality. A `QualityGte 8` predicate on a quality whose explored
    max is 7 can never become true — flag as bound mismatch.

    We only flag thresholds the analyzer is certain about — qualities with
    bounds derived from the full reachable state space. Qualities not in
    `quality_bounds` (because no reachable state ever wrote them) are skipped.
    """
    mismatches: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def check(pred: Predicate, location: str) -> None:
        if isinstance(pred, QualityGte):
            bounds = quality_bounds.get(pred.quality)
            if bounds is not None:
                _, hi = bounds
                if pred.value > hi:
                    reason = (
                        f"`>= {pred.value}` but quality maxes at {hi} "
                        "across all reachable states"
                    )
                    key = (location, pred.quality, reason)
                    if key not in seen:
                        seen.add(key)
                        mismatches.append((location, pred.quality, reason))
        elif isinstance(pred, QualityLt):
            bounds = quality_bounds.get(pred.quality)
            if bounds is not None:
                lo, _ = bounds
                if pred.value <= lo:
                    reason = (
                        f"`< {pred.value}` but quality mins at {lo} "
                        "across all reachable states"
                    )
                    key = (location, pred.quality, reason)
                    if key not in seen:
                        seen.add(key)
                        mismatches.append((location, pred.quality, reason))
        elif isinstance(pred, Not_):
            check(pred.of, location)
        elif isinstance(pred, Or_):
            for sub in pred.of:
                check(sub, location)

    for beat in scenario.beats:
        for top in beat.preconditions:
            check(top, f"{beat.id}.preconditions")
        for top in beat.spec.exit_when:
            check(top, f"{beat.id}.exit_when")
    return mismatches


# ───────────────────────── driver ─────────────────────────


def analyze_scenario(
    scenario: Scenario, *, time_limit_s: int = DEFAULT_TIME_LIMIT_S,
    progress: bool = False,
) -> AnalysisReport:
    """Run the ASP-backed static analysis. Returns an AnalysisReport ready
    to be converted to Finding objects (for lint integration) or JSON
    (for the sidecar).

    `time_limit_s` caps the clingo solver wall clock. With brave-mode
    enumeration the production scenario solves in ~0.3s; the limit only
    bites on grossly oversized scenarios (cf. plan §"grounding blowup").

    `progress=True` prints a single one-line "solving..." breadcrumb to
    stderr. ASP solving is a single shot, not iterative, so there's no
    per-step heartbeat to emit; the breadcrumb just confirms work started.
    """
    profile = ScenarioProfile.from_scenario(scenario)
    scenario_id = getattr(scenario, "id", "scenario")

    with logfire.span(
        "analyze_scenario {scenario_id}",
        _level="notice",
        scenario_id=scenario_id,
        time_limit_s=time_limit_s,
        beat_count=len(scenario.beats),
    ) as span:
        if progress:
            print(
                f"[analyze] encoding scenario ({len(scenario.beats)} beats) "
                f"and solving with clingo (time_limit={time_limit_s}s)",
                file=sys.stderr, flush=True,
            )

        program = asp_encoder.encode_scenario(scenario)
        solved = asp_encoder.solve(program, time_limit_s=time_limit_s)

        reached_beats = set(solved.reached_beats)
        all_beats = {b.id for b in scenario.beats}
        unreachable = sorted(all_beats - reached_beats)

        orphan_effects, orphan_predicates = _find_orphans(scenario, profile)
        # Bound mismatches require complete bounds to be reliable. When the
        # solver hits the time limit, the recorded max for each quality is
        # a lower bound on the true max — a predicate threshold above it
        # might still be reachable past the cap. Skip the check when
        # truncated; analysis_incomplete already tells the author to investigate.
        bound_mismatches = (
            _find_bound_mismatches(scenario, solved.quality_bounds)
            if not solved.truncated else []
        )

        span.set_attributes({
            "models_enumerated": solved.models_enumerated,
            "grounding_atoms": solved.grounding_atoms,
            "ground_time_s": solved.ground_time_s,
            "solve_time_s": solved.solve_time_s,
            "beats_reached": len(reached_beats),
            "beats_unreachable": len(unreachable),
            "truncated": solved.truncated,
        })

        if progress:
            print(
                f"[analyze] solved in {solved.solve_time_s:.2f}s "
                f"(ground {solved.ground_time_s:.2f}s, "
                f"{solved.grounding_atoms} atoms, "
                f"{solved.models_enumerated} models, "
                f"{len(reached_beats)}/{len(scenario.beats)} beats reachable)",
                file=sys.stderr, flush=True,
            )

        return AnalysisReport(
            reached_beats=reached_beats,
            unreachable_beats=unreachable,
            quality_bounds=solved.quality_bounds,
            quality_string_values=solved.quality_strings,
            orphan_effects=orphan_effects,
            orphan_predicates=orphan_predicates,
            bound_mismatches=bound_mismatches,
            truncated=solved.truncated,
            models_enumerated=solved.models_enumerated,
            grounding_atoms=solved.grounding_atoms,
            solve_time_s=solved.solve_time_s,
        )


# ───────────────────────── CLI ─────────────────────────


def write_json_sidecar(report: AnalysisReport, scenario: Scenario, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_id = getattr(scenario, "id", "scenario")
    path = out_dir / f"{scenario_id}.json"
    path.write_text(
        json.dumps(report.to_json(scenario), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static analyzer for smallville scenario.yaml (ASP-backed). "
        "Prints findings, writes JSON sidecar, exits non-zero on errors.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="npc_service/scenario.yaml",
        help="Path to scenario.yaml (default: npc_service/scenario.yaml)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_TIME_LIMIT_S,
        help=f"clingo solver wall-clock cap, seconds (default: {DEFAULT_TIME_LIMIT_S}).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write JSON sidecar to this path (default: npc_service/_analysis/<scenario>.json).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress INFO findings (analyzer doesn't emit any currently).",
    )
    args = parser.parse_args(argv)

    path = Path(args.scenario)
    if not path.exists():
        print(f"ERROR: scenario file not found: {path}", file=sys.stderr)
        return 2
    try:
        scenario = load_scenario(path)
    except Exception as exc:  # noqa: BLE001 — match lint_scenario's pattern
        print(f"ERROR: scenario failed to parse: {exc}", file=sys.stderr)
        return 2

    report = analyze_scenario(
        scenario, time_limit_s=args.time_limit, progress=not args.quiet,
    )
    findings = report.to_findings(scenario)

    has_error = False
    has_warn = False
    for f in findings:
        if args.quiet and f.severity == "INFO":
            continue
        print(f.format())
        if f.severity == "ERROR":
            has_error = True
        elif f.severity == "WARN":
            has_warn = True

    json_path = args.json if args.json is not None else (
        Path("npc_service/_analysis") if path.parts[0] == "npc_service"
        else path.parent / "_analysis"
    ) / f"{getattr(scenario, 'id', 'scenario')}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_json(scenario), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sidecar: {json_path}")
    print(
        f"Solved in {report.solve_time_s:.2f}s "
        f"({report.grounding_atoms} grounded atoms, "
        f"{'truncated' if report.truncated else 'exhaustive'}); "
        f"{len(report.reached_beats)}/{len(scenario.beats)} beats reachable.",
    )

    if has_error:
        return 1
    if has_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
