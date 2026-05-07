"""scenario.yaml linter.

Loads the scenario via `npc_service.models.load_scenario` (so every
`Scenario._cross_references` violation surfaces as a parse error before lint
logic runs — we don't duplicate that surface). Then applies seven
author-facing checks the model doesn't catch on its own. Each finding is one
line, prefixed with `ERROR` / `WARN` / `INFO`, and ends with an anchor link
into the sibling `lint_rules.md` so an LLM consumer can pull up the rule's
WHAT / WHY / SCOPE writeup directly.

Severity:
- ERROR — provable invariant violation; non-zero exit.
- WARN  — heuristic finding; non-zero exit unless `--no-warn-fail`.
- INFO  — observation; never affects exit.

Invocation:
    python lint_scenario.py [--no-warn-fail] [--quiet] [scenario_path]

Default `scenario_path` is `npc_service/scenario.yaml` (relative to CWD).

The lint reuses the in-memory Pydantic models — no string scraping. Run via
`uv run` from the npc_service directory so the editable install puts
`npc_service.models` on `sys.path` (`make lint-scenario` does this for you).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from npc_service.models import (
    Beat,
    BeatFired,
    Effect,
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


# Anchors must match section IDs in the sibling lint_rules.md so the trailing
# link in each finding is a working pointer into the docs. Keep these in sync
# if you rename a rule.
ANCHORS = {
    "scene_write_collision": "scene-write-collision",
    "terminal_overlap": "terminal-overlap",
    "validate_facts_on_scene": "validate-facts-on-scene",
    "pre_authored_post_boot": "pre-authored-post-boot",
    "forbidden_lexical": "forbidden-lexical",
    "classify_on_silent_npc": "classify-on-silent-npc",
    "reachability_contradiction": "reachability-contradiction",
}


# ───────────────────────── finding type ─────────────────────────


@dataclass
class Finding:
    severity: str  # "ERROR" | "WARN" | "INFO"
    location: str  # beat id or "pair: A vs B" or similar
    message: str
    anchor: str  # one of ANCHORS values

    def format(self) -> str:
        return (
            f"{self.severity}: {self.location}: {self.message} "
            f"(see scripts/lint_rules.md#{self.anchor})"
        )


# ───────────────────────── predicate walking ─────────────────────────


def walk_predicates(pred: Predicate) -> Iterable[Predicate]:
    """Yield `pred` and every sub-predicate, recursively. Includes the
    container nodes (Not_/Or_) themselves so callers can match on them."""
    yield pred
    if isinstance(pred, Not_):
        yield from walk_predicates(pred.of)
    elif isinstance(pred, Or_):
        for sub in pred.of:
            yield from walk_predicates(sub)


def precondition_contains_not_beat_fired(
    preds: list[Predicate], target_beat: str
) -> bool:
    """True if any precondition tree in `preds` contains `Not_(BeatFired
    target_beat)` anywhere — including nested under `Or_`. Used to check
    whether one beat's preconditions exclude another's fired state.
    """
    for top in preds:
        for node in walk_predicates(top):
            if (
                isinstance(node, Not_)
                and isinstance(node.of, BeatFired)
                and node.of.beat == target_beat
            ):
                return True
    return False


def beat_writes_quality(beat: Beat, quality_id: str) -> list[Effect]:
    """Effects on `beat` that target `quality_id` (literal or classify)."""
    return [e for e in beat.spec.effects if e.quality == quality_id]


def beat_literal_writes(beat: Beat) -> dict[str, Effect]:
    """Map of quality_id → first UNGATED literal effect on this beat
    (effect.value is not None AND effect.when is None). Scene-collision
    check applies to literal effects only — classify effects fire at
    reply time, and gated literals (`when:` set) are author-controlled
    no-ops on subsequent re-fires after the gate condition flips.
    """
    out: dict[str, Effect] = {}
    for e in beat.spec.effects:
        if not e.is_classify and e.when is None and e.quality not in out:
            out[e.quality] = e
    return out


# ───────────────────────── precondition feature extraction ─────────────────────────
#
# For the terminal-overlap check we need a coarse "intersection might be
# non-empty" decision. We extract a small feature record per terminal from
# its precondition tree, treating top-level entries as an implicit AND. If
# we encounter a node we can't analyze (Or_ at the top, complex Not_, etc.)
# we mark the record as `unanalyzable=True` and the pair-comparison emits
# INFO instead of WARN.


@dataclass
class PreconditionFeatures:
    """Coarse summary of a precondition tree, sufficient for terminal-pair
    overlap analysis. Fields are conservative — we only fill them when we
    can do so without losing precision."""

    quality_lower: dict[str, float] = field(default_factory=dict)  # quality → ≥
    quality_upper: dict[str, float] = field(default_factory=dict)  # quality → <
    quality_equals: dict[str, str | int | float] = field(default_factory=dict)
    required_fired: set[str] = field(default_factory=set)
    excluded_fired: set[str] = field(default_factory=set)
    # quality_id → (allowed_values, match_mode). For terminal-overlap, we
    # treat `match=any` as "at least one slot is in values"; we approximate
    # the intersection on quality+match_mode pairs (see overlap function).
    recent_any: dict[str, set[str]] = field(default_factory=dict)
    recent_all: dict[str, set[str]] = field(default_factory=dict)
    replies_threshold: dict[str, int] = field(default_factory=dict)  # thread → ≥
    unanalyzable_reason: str | None = None


def extract_features(preds: list[Predicate]) -> PreconditionFeatures:
    """Treat the top-level list as AND. Bail (set unanalyzable) on Or_ at top
    level or any other shape we don't model. Nested Not_ on simple leaves is
    handled; Not_ wrapping Or_ / quality_gte / etc. is treated as
    unanalyzable (we'd need de Morgan + range inversion to be precise).
    """
    feats = PreconditionFeatures()
    for top in preds:
        _absorb_top_level(feats, top)
        if feats.unanalyzable_reason is not None:
            return feats
    return feats


def _absorb_top_level(feats: PreconditionFeatures, node: Predicate) -> None:
    """Each top-level entry is conjunctive: it must hold for the beat to
    fire. Absorb its constraints into `feats`, or mark unanalyzable."""
    if isinstance(node, QualityGte):
        prev = feats.quality_lower.get(node.quality)
        feats.quality_lower[node.quality] = (
            node.value if prev is None else max(prev, node.value)
        )
    elif isinstance(node, QualityLt):
        prev = feats.quality_upper.get(node.quality)
        feats.quality_upper[node.quality] = (
            node.value if prev is None else min(prev, node.value)
        )
    elif isinstance(node, QualityEquals):
        # If two qualifiers conflict on the same quality (equals X and equals
        # Y for X != Y), the precondition is itself unsatisfiable. We don't
        # treat that as "overlap with another terminal" — leave it for the
        # reachability check to flag. Here, last-wins; downstream pairing
        # will see the constraint.
        feats.quality_equals[node.quality] = node.value
    elif isinstance(node, BeatFired):
        feats.required_fired.add(node.beat)
    elif isinstance(node, PlayerRepliesToThreadGte):
        prev = feats.replies_threshold.get(node.thread, 0)
        feats.replies_threshold[node.thread] = max(prev, node.value)
    elif isinstance(node, RecentClassifierValue):
        bucket = (
            feats.recent_any if node.match == "any" or isinstance(node.match, int)
            else feats.recent_all
        )
        existing = bucket.get(node.quality)
        if existing is None:
            bucket[node.quality] = set(node.values)
        else:
            # Two AND'd constraints on same quality / match: take
            # intersection (must satisfy both).
            existing &= set(node.values)
    elif isinstance(node, Not_):
        inner = node.of
        if isinstance(inner, BeatFired):
            feats.excluded_fired.add(inner.beat)
        else:
            feats.unanalyzable_reason = (
                f"Not_ on {type(inner).__name__} requires de Morgan to analyze"
            )
    elif isinstance(node, Or_):
        # Or_ relaxes constraints rather than tightening them. For overlap
        # analysis, an Or_ subtree contributes nothing to disjointness — if
        # both terminals require Or_(A, B), any state satisfying the Or_ is
        # shared, so it can't disjoin them. We skip it; the discriminating
        # constraints live in the OTHER top-level entries.
        return
    elif isinstance(node, PlayerCanonExists):
        feats.unanalyzable_reason = "PlayerCanonExists is dynamic (extractor-driven)"
    else:
        feats.unanalyzable_reason = f"unhandled predicate kind: {type(node).__name__}"


@dataclass
class OverlapVerdict:
    """Output of pairwise overlap analysis. `can_overlap` is the headline:
    True / False / None (unanalyzable). `soft_via_recent_any_disjoint` is
    True when the two terminals share an `recent_classifier_value` quality
    constraint with disjoint value sets, both in `match=any` mode. That's
    the documented "Director picks by recent texture" pattern: a multi-slot
    window can contain slots from both value sets, so both terminals are
    eligible at once and the Director decides. INFO-level rather than WARN.
    """

    can_overlap: bool | None
    soft_via_recent_any_disjoint: bool = False


def features_can_overlap(
    a: PreconditionFeatures, b: PreconditionFeatures
) -> OverlapVerdict:
    """Decide whether the conjunctive constraints in `a` and `b` could be
    simultaneously satisfied. Returns an OverlapVerdict — see its docstring
    for the soft/hard distinction.
    """
    if a.unanalyzable_reason is not None or b.unanalyzable_reason is not None:
        return OverlapVerdict(can_overlap=None)

    # Quality range intersection.
    qualities = (
        set(a.quality_lower) | set(a.quality_upper)
        | set(b.quality_lower) | set(b.quality_upper)
    )
    for q in qualities:
        lower = max(
            a.quality_lower.get(q, float("-inf")),
            b.quality_lower.get(q, float("-inf")),
        )
        upper = min(
            a.quality_upper.get(q, float("inf")),
            b.quality_upper.get(q, float("inf")),
        )
        if lower >= upper:
            return OverlapVerdict(can_overlap=False)

    # Quality-equals (string qualities) — if both pin the same quality but
    # to different values, disjoint.
    for q, va in a.quality_equals.items():
        if q in b.quality_equals and b.quality_equals[q] != va:
            return OverlapVerdict(can_overlap=False)

    # Beat-fired sets — if A requires X fired and B requires X NOT fired
    # (or vice-versa), disjoint.
    if a.required_fired & b.excluded_fired:
        return OverlapVerdict(can_overlap=False)
    if b.required_fired & a.excluded_fired:
        return OverlapVerdict(can_overlap=False)

    # `recent_all` mode — every slot in window must be in values. If A
    # requires all in {paternal} and B requires all in {cold}, the same
    # history can't satisfy both → disjoint.
    for q in set(a.recent_all) | set(b.recent_all):
        if q in a.recent_all and q in b.recent_all:
            if not (a.recent_all[q] & b.recent_all[q]):
                return OverlapVerdict(can_overlap=False)
        # Mixing `any` and `all` on the same quality — `all` implies all
        # slots in V_all; `any` requires ≥1 slot in V_any. Both satisfiable
        # iff V_all ∩ V_any non-empty.
        if q in a.recent_all and q in b.recent_any:
            if not (a.recent_all[q] & b.recent_any[q]):
                return OverlapVerdict(can_overlap=False)
        if q in b.recent_all and q in a.recent_any:
            if not (b.recent_all[q] & a.recent_any[q]):
                return OverlapVerdict(can_overlap=False)

    # `recent_any` mode on the same quality — both want ≥1 slot in their
    # respective value sets. A multi-slot window can satisfy both even when
    # the value sets are disjoint (e.g. window=[paternal, cold, neutral]
    # satisfies A.values={paternal} AND B.values={cold}). Mark this as
    # "soft" overlap so the caller emits INFO instead of WARN — it's the
    # documented Director-discriminates-by-recent-texture pattern, not
    # a missing mutex. Reply thresholds always co-satisfy (max wins).
    soft_disjoint = False
    for q in set(a.recent_any) & set(b.recent_any):
        if not (a.recent_any[q] & b.recent_any[q]):
            soft_disjoint = True
            break

    return OverlapVerdict(
        can_overlap=True, soft_via_recent_any_disjoint=soft_disjoint
    )


# ───────────────────────── checks ─────────────────────────


def check_scene_write_collision(
    scenario: Scenario,
) -> list[Finding]:
    """Check #1. For each (scene beat, other beat) pair where both write
    the same quality literally, at least one direction must have a
    `Not_(BeatFired)` exclusion to prevent the scene re-firing next tick
    and clobbering the later write back to its own value.
    """
    findings: list[Finding] = []
    scene_beats = [b for b in scenario.beats if b.kind == "scene"]
    for sc_beat in scene_beats:
        sc_writes = beat_literal_writes(sc_beat)
        if not sc_writes:
            continue
        for other in scenario.beats:
            if other.id == sc_beat.id:
                continue
            other_qualities = {e.quality for e in other.spec.effects}
            collisions = set(sc_writes) & other_qualities
            if not collisions:
                continue
            other_excludes_sc = precondition_contains_not_beat_fired(
                other.preconditions, sc_beat.id
            )
            sc_excludes_other = precondition_contains_not_beat_fired(
                sc_beat.preconditions, other.id
            )
            if not (other_excludes_sc or sc_excludes_other):
                findings.append(
                    Finding(
                        severity="ERROR",
                        location=f"{sc_beat.id} vs {other.id}",
                        message=(
                            f"scene beat {sc_beat.id!r} writes "
                            f"{sorted(collisions)} but neither beat has "
                            f"`Not_(BeatFired ...)` against the other — "
                            "scene will re-fire and clobber"
                        ),
                        anchor=ANCHORS["scene_write_collision"],
                    )
                )
    return findings


def check_terminal_overlap(scenario: Scenario) -> list[Finding]:
    """Check #2. Pairwise overlap analysis on terminal beats. Skipped if
    fewer than two terminals exist. Emits WARN on potential overlap, INFO on
    unanalyzable predicate trees.
    """
    findings: list[Finding] = []
    terminals = [b for b in scenario.beats if b.ends_session]
    if len(terminals) < 2:
        return findings
    feats = {b.id: extract_features(b.preconditions) for b in terminals}
    for i, a in enumerate(terminals):
        for b in terminals[i + 1 :]:
            verdict = features_can_overlap(feats[a.id], feats[b.id])
            # Explicit mutex: if either terminal has `Not_(BeatFired other)`
            # in its preconditions, the author authored a fallback chain.
            # Common pattern: B = "fallback when A hasn't fired." With
            # ends_session=True on both, only one fires per run anyway, so
            # this is a coherent design — INFO, not WARN.
            explicit_mutex = (
                a.id in feats[b.id].excluded_fired
                or b.id in feats[a.id].excluded_fired
            )
            if verdict.can_overlap is None:
                reasons = []
                if feats[a.id].unanalyzable_reason:
                    reasons.append(
                        f"{a.id}: {feats[a.id].unanalyzable_reason}"
                    )
                if feats[b.id].unanalyzable_reason:
                    reasons.append(
                        f"{b.id}: {feats[b.id].unanalyzable_reason}"
                    )
                findings.append(
                    Finding(
                        severity="INFO",
                        location=f"{a.id} vs {b.id}",
                        message=(
                            "couldn't fully analyze overlap; manual review — "
                            + "; ".join(reasons)
                        ),
                        anchor=ANCHORS["terminal_overlap"],
                    )
                )
            elif verdict.can_overlap and explicit_mutex:
                findings.append(
                    Finding(
                        severity="INFO",
                        location=f"{a.id} vs {b.id}",
                        message=(
                            "terminals have analyzable overlap, but at least "
                            "one excludes the other via `Not_(BeatFired)` — "
                            "explicit fallback chain authored; with "
                            "ends_session=True on both, only one fires per run"
                        ),
                        anchor=ANCHORS["terminal_overlap"],
                    )
                )
            elif verdict.can_overlap and verdict.soft_via_recent_any_disjoint:
                # The two terminals are distinguished only by disjoint
                # `recent_classifier_value` `match=any` value sets on the
                # same quality. A multi-slot window can co-satisfy both;
                # the Director picks. INFO, not WARN — this is the
                # documented per-recent-texture branching pattern.
                findings.append(
                    Finding(
                        severity="INFO",
                        location=f"{a.id} vs {b.id}",
                        message=(
                            "terminals overlap only via disjoint "
                            "`recent_classifier_value match=any` value sets — "
                            "Director discriminates by recent texture; "
                            "confirm intentional"
                        ),
                        anchor=ANCHORS["terminal_overlap"],
                    )
                )
            elif verdict.can_overlap:
                findings.append(
                    Finding(
                        severity="WARN",
                        location=f"{a.id} vs {b.id}",
                        message=(
                            "terminal preconditions appear to overlap on "
                            "hard constraints (quality ranges, fired-beat "
                            "sets, etc.) — confirm intentional"
                        ),
                        anchor=ANCHORS["terminal_overlap"],
                    )
                )
    return findings


def check_validate_facts_on_scene(scenario: Scenario) -> list[Finding]:
    """Check #3. `validate_facts: true` on a `kind: scene` beat means one
    extra fact-checker LLM call on every re-fire — unbounded cost."""
    return [
        Finding(
            severity="ERROR",
            location=b.id,
            message=(
                "validate_facts=True on a scene beat — fact-checker "
                "would run on every re-fire (unbounded LLM cost)"
            ),
            anchor=ANCHORS["validate_facts_on_scene"],
        )
        for b in scenario.beats
        if b.kind == "scene" and b.spec.validate_facts
    ]


def check_pre_authored_post_boot(scenario: Scenario) -> list[Finding]:
    """Check #4. `pre_authored_prose` only works at boot (no `player_body`
    available post-boot per `main.py:_fire_beat`). Flag any pre-authored
    beat whose preconditions allow it to fire after the boot tick.
    """
    findings: list[Finding] = []
    for b in scenario.beats:
        if b.spec.pre_authored_prose is None:
            continue
        for top in b.preconditions:
            for node in walk_predicates(top):
                if isinstance(node, PlayerRepliesToThreadGte) and node.value >= 1:
                    findings.append(
                        Finding(
                            severity="ERROR",
                            location=b.id,
                            message=(
                                "pre_authored_prose set but preconditions "
                                f"include `player_replies_to_thread_gte "
                                f"{node.thread} {node.value}` — beat would "
                                "fire post-boot, where pre-authored render "
                                "doesn't apply (linearity-trap antipattern)"
                            ),
                            anchor=ANCHORS["pre_authored_post_boot"],
                        )
                    )
                    break
            else:
                continue
            break
    return findings


def check_forbidden_lexical(scenario: Scenario) -> list[Finding]:
    """Check #5. Extract quoted tokens from each NPC's `base_voice_forbidden`
    list and search the assigned beats' `spec.intent` and
    `spec.must_include_facts` for whole-word matches. Skip `spec.must_not`
    entirely — that list naturally contains forbidden phrases (it IS the
    deny list).
    """
    findings: list[Finding] = []
    quote_re = re.compile(r"['\"]([^'\"]+?)['\"]")

    # Allow-list signals: when a segment names a PRESCRIBED token (rather
    # than a forbidden one), skip token extraction. Authors phrase these as
    # "use 'X'", "anything but 'X'", "always 'X'", "only 'X'", etc. Without
    # this guard, the linter false-flags any beat that legitimately uses
    # the prescribed phrasing.
    allow_signals = (
        "anything but",
        "anything except",
        " use '",
        " use \"",
        "always ",
        " only ",
    )

    npc_tokens: dict[str, list[str]] = {}
    for npc in scenario.npcs:
        tokens: list[str] = []
        for entry in npc.base_voice_forbidden:
            # Split on `;` — entries combining bans with prescriptions
            # (e.g. CPO's "...; use 'the Counterpart' or 'Counterpart
            # ARIA-7'") need per-segment handling.
            for segment in entry.split(";"):
                segment_lower = segment.lower()
                if any(sig in segment_lower for sig in allow_signals):
                    continue
                tokens.extend(quote_re.findall(segment))
        # Dedupe, preserve order.
        seen: set[str] = set()
        deduped: list[str] = []
        for t in tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                deduped.append(t)
        npc_tokens[npc.id] = deduped

    for b in scenario.beats:
        tokens = npc_tokens.get(b.npc, [])
        if not tokens:
            continue
        haystacks: list[tuple[str, str]] = [("spec.intent", b.spec.intent)]
        for i, fact in enumerate(b.spec.must_include_facts):
            haystacks.append((f"spec.must_include_facts[{i}]", fact))
        for token in tokens:
            pattern = re.compile(
                r"\b" + re.escape(token) + r"\b", re.IGNORECASE
            )
            for where, text in haystacks:
                if not text:
                    continue
                if pattern.search(text):
                    findings.append(
                        Finding(
                            severity="WARN",
                            location=b.id,
                            message=(
                                f"forbidden token {token!r} (from "
                                f"npc {b.npc!r}.base_voice_forbidden) appears "
                                f"in {where} — chat agent will likely echo it"
                            ),
                            anchor=ANCHORS["forbidden_lexical"],
                        )
                    )
    return findings


def check_classify_on_silent_npc(scenario: Scenario) -> list[Finding]:
    """Check #6. Classify effects on a beat assigned to an NPC whose
    `replies_enabled=False` are dead authoring — institutional senders
    don't get replies, so the classifier never runs.
    """
    findings: list[Finding] = []
    silent = {n.id for n in scenario.npcs if not n.replies_enabled}
    for b in scenario.beats:
        if b.npc not in silent:
            continue
        for i, e in enumerate(b.spec.effects):
            if e.is_classify:
                findings.append(
                    Finding(
                        severity="ERROR",
                        location=b.id,
                        message=(
                            f"spec.effects[{i}] has a `classify:` directive "
                            f"on quality {e.quality!r}, but npc {b.npc!r} "
                            "has replies_enabled=False — classifier will "
                            "never run (dead authoring)"
                        ),
                        anchor=ANCHORS["classify_on_silent_npc"],
                    )
                )
    return findings


def check_reachability_contradiction(scenario: Scenario) -> list[Finding]:
    """Check #7. For each beat, walk its precondition tree (treating the
    top-level list as AND). If we find both `BeatFired X` and `Not_(BeatFired
    X)` for the same X anywhere in the conjunctive scope, the precondition
    is unsatisfiable. Heuristic: simple contradictions only — full SAT is
    deferred to manual review + the live-run check.
    """
    findings: list[Finding] = []
    for b in scenario.beats:
        required: set[str] = set()
        excluded: set[str] = set()
        for top in b.preconditions:
            for node in walk_predicates(top):
                if isinstance(node, BeatFired):
                    # Inside Not_, the BeatFired adds to `excluded` not
                    # `required` — we need to track parentage. Re-walk
                    # explicitly below for that nuance.
                    pass
            _collect_required_excluded(top, required, excluded, in_not=False)
        contradicting = required & excluded
        if contradicting:
            findings.append(
                Finding(
                    severity="WARN",
                    location=b.id,
                    message=(
                        "preconditions require AND exclude the same beat-fired "
                        f"set: {sorted(contradicting)} — beat is unreachable"
                    ),
                    anchor=ANCHORS["reachability_contradiction"],
                )
            )
    return findings


def _collect_required_excluded(
    pred: Predicate,
    required: set[str],
    excluded: set[str],
    *,
    in_not: bool,
) -> None:
    """Walk `pred`, accumulating BeatFired ids into `required` (when reached
    outside of `Not_`) or `excluded` (when reached inside one). Or_ is
    treated as a hatch — we don't track per-branch contradictions to avoid
    false positives on intentional branching.
    """
    if isinstance(pred, BeatFired):
        (excluded if in_not else required).add(pred.beat)
    elif isinstance(pred, Not_):
        _collect_required_excluded(
            pred.of, required, excluded, in_not=not in_not
        )
    elif isinstance(pred, Or_):
        # Each branch is a disjunctive option — don't pollute the
        # AND-context with branch-specific BeatFired refs. Skip.
        return
    # Leaf predicates without BeatFired: nothing to do.


# ───────────────────────── driver ─────────────────────────


def lint_scenario(
    scenario: Scenario,
    *,
    run_analyzer: bool = True,
    analyzer_max_states: int = 300_000,
    analyzer_progress: bool = False,
) -> list[Finding]:
    findings: list[Finding] = [
        *check_scene_write_collision(scenario),
        *check_terminal_overlap(scenario),
        *check_validate_facts_on_scene(scenario),
        *check_pre_authored_post_boot(scenario),
        *check_forbidden_lexical(scenario),
        *check_classify_on_silent_npc(scenario),
        *check_reachability_contradiction(scenario),
    ]
    if run_analyzer:
        # The static analyzer is heavier (state-space exploration) and lives
        # in a sibling module so its imports stay out of the spec-only lint
        # path. Default `analyzer_max_states=300_000` matches the standalone
        # `make analyze-scenario` cap so lint and analyze report identical
        # reachability claims on production-sized scenarios. Tests pass
        # run_analyzer=False to keep individual lint-rule tests focused on
        # spec analysis.
        from analyze_scenario import (  # noqa: E402 — sibling-module import
            analyze_scenario as _analyze,
        )
        report = _analyze(
            scenario,
            max_states=analyzer_max_states,
            progress=analyzer_progress,
        )
        for af in report.to_findings(scenario):
            findings.append(Finding(
                severity=af.severity, location=af.location,
                message=af.message, anchor=af.anchor,
            ))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lint a smallville scenario.yaml against author-facing invariants "
            "the Pydantic model doesn't catch on its own."
        )
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="npc_service/scenario.yaml",
        help="Path to scenario.yaml (default: npc_service/scenario.yaml)",
    )
    parser.add_argument(
        "--no-warn-fail",
        action="store_true",
        help="Don't fail exit code on WARN findings (ERROR still fails).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO findings.",
    )
    args = parser.parse_args(argv)

    path = Path(args.scenario)
    if not path.exists():
        print(f"ERROR: scenario file not found: {path}", file=sys.stderr)
        return 2

    try:
        scenario = load_scenario(path)
    except Exception as exc:
        print(f"ERROR: scenario failed to parse: {exc}", file=sys.stderr)
        return 2

    findings = lint_scenario(scenario, analyzer_progress=not args.quiet)

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

    if not findings:
        print(f"OK: {path} — no findings on 7 author-facing checks.")

    if has_error:
        return 1
    if has_warn and not args.no_warn_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
