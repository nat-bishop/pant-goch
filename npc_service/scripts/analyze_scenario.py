"""Static scenario analyzer.

Bounded forward state exploration that reuses the engine's predicate
evaluator and effect appliers (no re-implementation) to determine, without
running the LLM:

- Which beats are reachable from boot state, and which terminals
- Min/max bounds for every quality across all reachable states
- Orphan effects (writes that no predicate reads) and orphan predicates
  (reads with no writer)
- Bound mismatches (predicate threshold outside the proven quality range)

State-space mitigations:

- Quality values tracked exactly. A per-scenario MAX_STATES cap halts
  exploration when the state space blows up; we emit `analysis_incomplete`
  rather than ever falsely claiming a beat unreachable.
- Canon keys: only model presence/absence for keys referenced by
  `PlayerCanonExists`. Per fire, branch over (no canon write) or (one
  referenced key added) — not full powerset, which would be 2^M per fire.
  Adds keys gradually across fires, still complete: any subset of M keys
  is reachable in at most M fires.
- Director non-determinism: when ≥2 beats are eligible on a tick, branch
  on every eligible beat (sound over-approximation: never misses a
  reachable beat).
- Player replies: each fire branches on the cartesian product of
  classifier verdicts for the fired beat's classify effects + scenario
  classify effects gated by `when` predicates that pass in the post-fire
  state. Independent-axis assumption is sound (over-approximates real
  classifier correlation).
- Classify-effect branching is approximated to fire-time, not reply-time.
  In the engine, classifier deltas land at /send time on the BEAT THE
  PLAYER REPLIED TO; here we apply them right after the beat fires. Same
  sequencing modulo one tick — the beat that fired is the beat the player
  is about to reply to, by construction.

Output: list of Finding objects (same shape as lint_scenario.py) plus a
JSON sidecar to `npc_service/_analysis/<scenario>.json` for future
ScenarioGraph overlay.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# Make `from npc_service.X import Y` resolve when the script is invoked
# directly (`python scripts/analyze_scenario.py`) before the package is
# editable-installed. `make lint-scenario` runs via `uv run`, which already
# has the package on sys.path, so this only matters for ad-hoc invocations.
_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from npc_service.models import (
    AxisResult,
    Beat,
    BeatFired,
    CanonCategory,
    Effect,
    InboxState,
    Not_,
    Or_,
    PlayerCanonEntry,
    PlayerCanonExists,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityEquals,
    QualityGte,
    QualityLt,
    RecentClassifierValue,
    Reply,
    Scenario,
    load_scenario,
)
from npc_service.spine import (
    apply_classify_effect,
    apply_literal_effects,
    evaluate_predicate,
    find_due_beats,
    scene_exit_satisfied,
)


DEFAULT_MAX_STATES = 50_000
# Replies are trimmed per thread to this floor (above the largest predicate
# threshold for that thread + the largest recent-classifier window for axes
# present, which together set the real cap). The floor prevents over-trimming
# on scenarios with no PlayerRepliesToThreadGte predicates.
REPLY_KEEP_FLOOR_PER_THREAD = 2


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
    matters for predicate evaluation. Used to compress state signatures
    (only track dimensions a predicate could ever read) and to drive
    orphan detection.

    `referenced_*` sets are the surface predicates read from. `qualities_*`
    sets pair them against effects to identify orphans.
    """

    referenced_canon_keys: frozenset[str]
    referenced_classifier_qualities: frozenset[str]
    thread_reply_caps: dict[str, int]  # thread → max threshold + 1 (states above the cap collapse)
    recent_classifier_window_caps: dict[str, int]  # quality → max window in any predicate
    qualities_with_effects: frozenset[str]
    qualities_with_predicates: frozenset[str]
    quality_thresholds: dict[str, list[float]]  # quality → all numeric thresholds referenced

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "ScenarioProfile":
        canon_keys: set[str] = set()
        classifier_qs: set[str] = set()
        thread_caps: dict[str, int] = {}
        recent_caps: dict[str, int] = {}
        quals_predicates: set[str] = set()
        thresholds: dict[str, list[float]] = {}

        def walk(pred: Predicate) -> None:
            if isinstance(pred, QualityGte):
                quals_predicates.add(pred.quality)
                thresholds.setdefault(pred.quality, []).append(pred.value)
            elif isinstance(pred, QualityLt):
                quals_predicates.add(pred.quality)
                thresholds.setdefault(pred.quality, []).append(pred.value)
            elif isinstance(pred, QualityEquals):
                quals_predicates.add(pred.quality)
                if isinstance(pred.value, (int, float)):
                    thresholds.setdefault(pred.quality, []).append(float(pred.value))
            elif isinstance(pred, PlayerCanonExists):
                canon_keys.add(pred.name)
            elif isinstance(pred, PlayerRepliesToThreadGte):
                # State signature caps reply count at threshold+1 — beyond that,
                # all states are equivalent for any PlayerRepliesToThreadGte
                # predicate evaluating against thresholds we've seen.
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
            quality_thresholds=thresholds,
        )


# ───────────────────────── state signature ─────────────────────────


# (qualities tuple, fired set, canon keys, thread reply counts, recent classifier per axis, ended)
StateKey = tuple[
    tuple[tuple[str, float | str], ...],
    frozenset[str],
    frozenset[str],
    tuple[tuple[str, int], ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    bool,
]


def state_signature(state: InboxState, profile: ScenarioProfile) -> StateKey:
    """Compress an InboxState to a hashable key for BFS dedup. Only includes
    dimensions a predicate could ever read."""
    qualities = tuple(sorted(state.qualities.items()))
    fired = frozenset(state.fired_beat_ids)
    canon_keys = frozenset(
        k for k in state.player_canon if k in profile.referenced_canon_keys
    )

    # Per-thread reply counts, capped at threshold+1 so states above any
    # threshold collapse (no predicate distinguishes "10 replies" from "11").
    thread_counts_raw: dict[str, int] = {}
    for r in state.replies:
        if r.thread_id is None:
            continue
        thread_counts_raw[r.thread_id] = thread_counts_raw.get(r.thread_id, 0) + 1
    capped: dict[str, int] = {}
    for tid, cnt in thread_counts_raw.items():
        cap = profile.thread_reply_caps.get(tid)
        if cap is None:
            # No predicate references this thread's reply count — drop entirely.
            continue
        capped[tid] = min(cnt, cap)
    thread_reply_counts = tuple(sorted(capped.items()))

    # Per-axis recent classifier history (most-recent-first), bounded to the
    # max window any RecentClassifierValue predicate uses for that quality.
    recent: dict[str, list[str]] = {}
    for r in reversed(state.replies):
        for axis in r.classifier_results:
            if axis.quality not in profile.referenced_classifier_qualities:
                continue
            cap = profile.recent_classifier_window_caps.get(axis.quality, 0)
            if cap == 0:
                continue
            slots = recent.setdefault(axis.quality, [])
            if len(slots) < cap:
                slots.append(axis.value)
    recent_classifier = tuple(
        (q, tuple(slots)) for q, slots in sorted(recent.items())
    )

    return (qualities, fired, canon_keys, thread_reply_counts, recent_classifier, state.ended)


# ───────────────────────── transitions ─────────────────────────


def _record_synthetic_reply(
    state: InboxState,
    thread_id: str | None,
    classifier_results: list[AxisResult],
    profile: ScenarioProfile,
) -> None:
    """Append a synthetic Reply so `RecentClassifierValue` and
    `PlayerRepliesToThreadGte` predicates see consistent history. Body is
    empty — predicates don't read it. Trims the per-thread reply tail
    afterward to keep state size bounded; without this, BFS deepcopies
    blow up O(states × avg_reply_count).
    """
    state.replies.append(
        Reply(
            id=f"synthetic_{len(state.replies)}",
            thread_id=thread_id,
            beat_responded_to=None,
            recipient="",
            body="",
            classifier_results=list(classifier_results),
        )
    )
    _trim_replies(state, profile)


def _trim_replies(state: InboxState, profile: ScenarioProfile) -> None:
    """Drop replies that no predicate could read.

    Per thread, keep at most `max(thread_cap, max_axis_window) + safety`
    replies. Trimming preserves within-thread order so RecentClassifierValue
    (which iterates `reversed(state.replies)`) still sees the right "most
    recent" slots. Replies with no thread (system) and replies on threads
    no PlayerRepliesToThreadGte references are trimmed first since their
    count is never read.
    """
    if not state.replies:
        return
    max_axis_window = max(
        profile.recent_classifier_window_caps.values(), default=0,
    )

    # Walk in reverse, mark kept indices per thread.
    counts: dict[str | None, int] = {}
    keep: set[int] = set()
    for i in range(len(state.replies) - 1, -1, -1):
        tid = state.replies[i].thread_id
        thread_cap = profile.thread_reply_caps.get(tid or "", 0) if tid else 0
        per_thread_keep = max(
            thread_cap, max_axis_window, REPLY_KEEP_FLOOR_PER_THREAD,
        )
        c = counts.get(tid, 0)
        if c < per_thread_keep:
            keep.add(i)
            counts[tid] = c + 1

    if len(keep) < len(state.replies):
        state.replies = [r for i, r in enumerate(state.replies) if i in keep]


def _apply_beat_fire_state(state: InboxState, beat: Beat, scenario: Scenario) -> None:
    """Apply a beat fire's state mutations: literal effects, fired_beat_ids
    update (one_shot immediate; scene only when exit_when satisfied), and
    `ends_session` flag.

    Mirrors `_fire_beat` in main.py minus the chat-agent render — the
    analyzer doesn't care about prose.
    """
    apply_literal_effects(state, scenario, beat.spec.effects, beat_id=beat.id)
    # Apply scenario-level literal effects whose `when` evaluates true now.
    apply_literal_effects(state, scenario, scenario.effects, beat_id=None)

    if beat.kind == "one_shot":
        state.fired_beat_ids.add(beat.id)
    elif beat.kind == "scene":
        if scene_exit_satisfied(beat, state):
            state.fired_beat_ids.add(beat.id)
    if beat.ends_session:
        state.ended = True


def _classify_effects_for_reply(
    state: InboxState, beat: Beat, scenario: Scenario,
) -> list[Effect]:
    """Classify-typed effects that would run when the player replies to this
    beat: per-beat classify effects + scenario.effects.is_classify gated by
    `when` predicates that pass in the post-fire state."""
    out: list[Effect] = [e for e in beat.spec.effects if e.is_classify]
    out.extend(
        e for e in scenario.effects
        if e.is_classify
        and (e.when is None or evaluate_predicate(e.when, state))
    )
    return out


def successors(
    state: InboxState, scenario: Scenario, profile: ScenarioProfile,
) -> Iterator[tuple[str | None, InboxState]]:
    """Yield (fired_beat_id, post_state) pairs reachable in one /send step.

    Each yield models one "player reply event": the player replies to some
    previously-fired beat → classifier + extractor mutations → tick →
    Director picks among eligible → (optional) beat fire.

    Branches over: which fired beat to reply to (over-approximates real
    thread/in_reply_to selection), classifier verdict combos, canon-key
    additions, and Director's pick. fired_beat_id is None when the /send
    triggered a tick but no beat fired (rare — usually means dead-end).
    """
    if state.ended:
        return

    # Player can reply to any beat that has fired (over-approximation: real
    # players reply to recent emails, but for reachability soundness we let
    # the player reply to anything previously fired). Reply to a scene beat
    # IS modeled by re-firing it through the next tick's find_due_beats.
    fired = sorted(state.fired_beat_ids)
    if not fired:
        return

    for replied_beat_id in fired:
        replied_beat = scenario.beat_by_id.get(replied_beat_id)
        if replied_beat is None:
            continue

        # Verdict combos: cartesian over classify effects on this beat +
        # scenario classify effects whose `when:` passes in current state.
        classify_effects = _classify_effects_for_reply(state, replied_beat, scenario)
        if classify_effects:
            value_sets = [
                tuple(e.classify.allowed_values)  # type: ignore[union-attr]
                for e in classify_effects
            ]
            verdict_combos: list[tuple[str, ...]] = list(itertools.product(*value_sets))
        else:
            verdict_combos = [()]

        # Pre-compute canon choices once per replied_beat — they don't
        # depend on the verdict combo (canon and classifier writes are
        # independent in the engine).
        not_yet_present = sorted(
            k for k in profile.referenced_canon_keys
            if k not in state.player_canon
        )
        canon_choices: list[str | None] = [None] + not_yet_present

        for combo in verdict_combos:
            for canon_key in canon_choices:
                # One deepcopy per (verdict combo × canon choice). All
                # state mutations land here; director branches deepcopy
                # again before applying the beat fire.
                with_writes = copy.deepcopy(state)
                classifier_results: list[AxisResult] = []
                for effect, value in zip(classify_effects, combo):
                    applied, _ = apply_classify_effect(
                        with_writes, scenario, effect, value, "",
                    )
                    if applied is not None:
                        classifier_results.append(applied)
                if canon_key is not None:
                    with_writes.player_canon[canon_key] = PlayerCanonEntry(
                        summary=f"(synthetic for analyzer: {canon_key})",
                        literal_text_quote="",
                        salience=1.0,
                        category=CanonCategory.PERSON,
                        turn_of_origin=len(with_writes.replies),
                    )
                _record_synthetic_reply(
                    with_writes, replied_beat.thread, classifier_results, profile,
                )

                # Tick: which beat fires given the post-reply state?
                eligible, _ = find_due_beats(with_writes, scenario)
                if not eligible:
                    # /send happened but no beat fired this tick; yield the
                    # post-reply state so BFS still records the classifier/
                    # canon/reply-count progress.
                    yield None, with_writes
                    continue
                for fired_beat in eligible:
                    post_fire = copy.deepcopy(with_writes)
                    _apply_beat_fire_state(post_fire, fired_beat, scenario)
                    yield fired_beat.id, post_fire


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
    states_explored: int

    def to_findings(self, scenario: Scenario) -> list[Finding]:
        out: list[Finding] = []
        terminals = {b.id for b in scenario.beats if b.ends_session}
        for beat_id in self.unreachable_beats:
            anchor_key = "unreachable_terminal" if beat_id in terminals else "unreachable_beat"
            out.append(Finding(
                severity="ERROR" if not self.truncated else "WARN",
                location=beat_id,
                message=(
                    f"beat {beat_id!r} not reachable from boot in "
                    f"{self.states_explored} explored states"
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
                    f"state-space cap hit at {self.states_explored} states; "
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
            "states_explored": self.states_explored,
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
    `quality_bounds` (because the analyzer never reached a state with that
    quality set) are skipped.
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
    scenario: Scenario, *, max_states: int = DEFAULT_MAX_STATES,
) -> AnalysisReport:
    """Run the full static analysis. Returns an AnalysisReport ready to be
    converted to Finding objects (for lint integration) or JSON (for the
    sidecar).
    """
    profile = ScenarioProfile.from_scenario(scenario)

    # Initial state — boot tick fires anchored beats with empty preconditions.
    # Mirrors `reset_session` in main.py without the LLM render: literal
    # effects apply, fired_beat_ids and ended flag update; chat-driven
    # boot beats render via the LLM in production but their state mutation
    # is captured by _apply_beat_fire_state.
    initial = InboxState.from_scenario(scenario)
    boot_fired = _boot_tick(initial, scenario)

    seen: set[StateKey] = set()
    queue: deque[InboxState] = deque([initial])
    seen.add(state_signature(initial, profile))

    # Boot fires count toward reached even for scene beats that haven't
    # exited (and so aren't in fired_beat_ids).
    reached_beats: set[str] = set(boot_fired) | set(initial.fired_beat_ids)
    quality_bounds: dict[str, tuple[float, float]] = {}
    quality_string_values: dict[str, set[str]] = {}
    truncated = False

    def update_bounds(state: InboxState) -> None:
        for q, v in state.qualities.items():
            if isinstance(v, (int, float)):
                fv = float(v)
                if q not in quality_bounds:
                    quality_bounds[q] = (fv, fv)
                else:
                    lo, hi = quality_bounds[q]
                    quality_bounds[q] = (min(lo, fv), max(hi, fv))
            elif isinstance(v, str):
                quality_string_values.setdefault(q, set()).add(v)

    update_bounds(initial)

    while queue:
        if len(seen) >= max_states:
            truncated = True
            break
        current = queue.popleft()
        for fired_id, succ in successors(current, scenario, profile):
            key = state_signature(succ, profile)
            if key in seen:
                continue
            seen.add(key)
            if fired_id is not None:
                reached_beats.add(fired_id)
            # Capture scene-beat re-fires too (entering fired_beat_ids on
            # exit); fired_id may be the same scene id across multiple yields.
            reached_beats.update(succ.fired_beat_ids)
            update_bounds(succ)
            queue.append(succ)
            if len(seen) >= max_states:
                truncated = True
                break

    all_beats = {b.id for b in scenario.beats}
    unreachable = sorted(all_beats - reached_beats)

    orphan_effects, orphan_predicates = _find_orphans(scenario, profile)
    # Bound mismatches require complete bounds to be reliable. When the
    # state-space cap is hit, the recorded max for each quality is a lower
    # bound on the true max — a predicate threshold above it might still
    # be reachable past the cap. Skip the check when truncated; the
    # analysis_incomplete warning already tells the author to investigate.
    bound_mismatches = (
        _find_bound_mismatches(scenario, quality_bounds) if not truncated else []
    )

    return AnalysisReport(
        reached_beats=reached_beats,
        unreachable_beats=unreachable,
        quality_bounds=quality_bounds,
        quality_string_values=quality_string_values,
        orphan_effects=orphan_effects,
        orphan_predicates=orphan_predicates,
        bound_mismatches=bound_mismatches,
        truncated=truncated,
        states_explored=len(seen),
    )


def _boot_tick(state: InboxState, scenario: Scenario) -> set[str]:
    """Apply the boot-tick state changes — anchored beats with empty
    preconditions fire immediately. Iterative to a fixed point: each fire
    may unlock chained boot beats (e.g., `BeatFired anchored.welcome` as
    a precondition for a follow-up anchored intro).

    Returns the set of beat ids fired during boot (so the BFS driver can
    count them as `reached`, even for scene beats that haven't exited yet
    and so aren't in `fired_beat_ids`).

    Note: the engine fires ONE beat per tick. The analyzer fires every
    boot-eligible beat in one shot — over-approximation of Director
    behavior. Sound for reachability: anything boot-eligible would
    eventually fire if the player did nothing.
    """
    fired_during_boot: set[str] = set()
    while True:
        eligible, _ = find_due_beats(state, scenario)
        if not eligible:
            return fired_during_boot
        # A "boot-eligible" beat has no player-driven preconditions:
        # nothing that requires a player reply, classifier verdict, or
        # canon write. Beats gated only on prior `BeatFired` plus quality
        # thresholds achievable from boot literal effects qualify.
        progress = False
        for beat in eligible:
            if beat.id in fired_during_boot:
                continue
            if any(_uses_player_signal(p) for p in beat.preconditions):
                continue
            _apply_beat_fire_state(state, beat, scenario)
            fired_during_boot.add(beat.id)
            progress = True
            if state.ended:
                return fired_during_boot
        if not progress:
            return fired_during_boot


def _uses_player_signal(pred: Predicate) -> bool:
    """True iff this predicate (or a sub-predicate) reads something the
    player has to provide via /send: reply count, classifier verdict, or
    canon key. Used by `_boot_tick` to decide whether a beat is boot-fireable.

    Quality predicates count as player-signal too: literal effects on boot
    are limited (only beats currently fireable have applied effects), so
    a quality threshold in the post-boot world is almost always classifier-
    driven. Conservative — boot fires fewer beats, BFS picks up the rest.
    """
    if isinstance(pred, (
        PlayerRepliesToThreadGte, RecentClassifierValue,
        PlayerCanonExists, QualityGte, QualityLt, QualityEquals,
    )):
        return True
    if isinstance(pred, Not_):
        return _uses_player_signal(pred.of)
    if isinstance(pred, Or_):
        return any(_uses_player_signal(sub) for sub in pred.of)
    return False  # BeatFired and other structural predicates


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
        description="Static analyzer for smallville scenario.yaml. "
        "Prints findings, writes JSON sidecar, exits non-zero on errors.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="npc_service/scenario.yaml",
        help="Path to scenario.yaml (default: npc_service/scenario.yaml)",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help=f"Cap state-space exploration (default: {DEFAULT_MAX_STATES}).",
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

    report = analyze_scenario(scenario, max_states=args.max_states)
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
        f"Explored {report.states_explored} states "
        f"({'truncated' if report.truncated else 'exhaustive'}); "
        f"{len(report.reached_beats)}/{len(scenario.beats)} beats reachable.",
    )

    if has_error:
        return 1
    if has_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
