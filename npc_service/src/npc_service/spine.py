"""The deterministic spine: predicate evaluation and beat scheduler.

No LLM calls live here — only code-owned state mutation.

Two rules this module enforces:

1. The chat-rendering LLM never touches state. The chat agent returns prose
   only; the spine is the only thing that mutates `InboxState`.
2. Quality writes have exactly two code-owned entry points, both in this
   module: `apply_classification` (driven by the classifier agent's per-axis
   output, with optional `accumulate`) and `apply_writes` (driven by a beat's
   authored `spec.writes`, SET-only for committed facts). Both share
   `_clamp_quality`. No other code path mutates `state.qualities`.

Observability is allowed (and explicitly used here): the spine emits
logfire info/warn lines so per-tick decisions are visible in the dev
console and queryable in the Logfire UI. The "no side effects" rule is
about state mutation, not logging.

Beat selection: `find_due_beats` returns every pending beat whose
preconditions pass, plus a per-beat eligibility report (used by the
per-tick console log and the in-browser debug overlay). When more than
one beat is eligible on the same tick, the LLM Director (in `agents.py`)
picks one — the spine no longer maintains same-tick mutex tracking.
Authoring order is not load-bearing for selection.
"""
from __future__ import annotations

import logfire

from .models import (
    AxisResult,
    Beat,
    BeatEligibility,
    BeatFired,
    ClassifyReply,
    InboxState,
    Not_,
    PendingBeat,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityDelta,
    QualityGte,
    QualityLt,
    RecentClassifierValue,
    Scenario,
    WritesByAxis,
)


# ───────────────────────── predicate evaluation ─────────────────────────


def evaluate_predicate(pred: Predicate, state: InboxState) -> bool:
    """Evaluate one predicate against current state.

    `BeatFired` only checks `state.fired_beat_ids` — the same-tick mutex
    machinery (extra_fired) is gone. With the Director picking one beat
    per tick, two beats becoming eligible together is a normal case
    resolved by the Director, not a hand-authored mutex chain.
    """
    if isinstance(pred, QualityGte):
        v = state.qualities.get(pred.quality)
        return v is not None and v >= pred.value
    if isinstance(pred, QualityLt):
        v = state.qualities.get(pred.quality)
        return v is not None and v < pred.value
    if isinstance(pred, PlayerRepliesToThreadGte):
        n = sum(1 for r in state.replies if r.thread_id == pred.thread)
        return n >= pred.value
    if isinstance(pred, BeatFired):
        return pred.beat in state.fired_beat_ids
    if isinstance(pred, RecentClassifierValue):
        return _evaluate_recent_classifier_value(pred, state)
    if isinstance(pred, Not_):
        return not evaluate_predicate(pred.of, state)
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


def _evaluate_recent_classifier_value(
    pred: RecentClassifierValue, state: InboxState
) -> bool:
    """Walk replies in reverse, collecting up to `pred.window` axis results
    whose identifier matches `pred.quality`. Replies without a matching axis
    are skipped (they don't fill a window slot). Resolve `pred.match` against
    the count of collected results whose value ∈ pred.values.

    Note: `pred.quality` is matched against `AxisResult.quality`, which is
    the axis identifier (axis_name or quality) — so verdict-only axes show
    up here under their `axis_name`.
    """
    collected: list[str] = []
    for reply in reversed(state.replies):
        for axis in reply.classifier_results:
            if axis.quality == pred.quality:
                collected.append(axis.value)
                break  # axis-uniqueness invariant: ≤1 per reply per axis
        if len(collected) >= pred.window:
            break

    matches = sum(1 for v in collected if v in pred.values)
    if pred.match == "any":
        return matches >= 1
    if pred.match == "all":
        # Under-filled window can't satisfy "all" — len(collected) < window.
        return matches >= pred.window
    return matches >= pred.match


def describe_predicate(pred: Predicate, state: InboxState) -> str:
    """Render one predicate as a concrete comparison against current state.

    Used by `find_due_beats` to populate `BeatEligibility.blocked_by` and by
    the per-tick console log to answer "why didn't this beat fire?" with
    real values rather than predicate names alone. Output is a one-line
    string designed to read naturally inline in a log message like
    "beat foo blocked: <description>".
    """
    if isinstance(pred, QualityGte):
        return f"{pred.quality}>={pred.value:g} got {_fmt_quality(state.qualities.get(pred.quality))}"
    if isinstance(pred, QualityLt):
        return f"{pred.quality}<{pred.value:g} got {_fmt_quality(state.qualities.get(pred.quality))}"
    if isinstance(pred, PlayerRepliesToThreadGte):
        n = sum(1 for r in state.replies if r.thread_id == pred.thread)
        return f"replies({pred.thread})>={pred.value} got {n}"
    if isinstance(pred, BeatFired):
        fired = pred.beat in state.fired_beat_ids
        return f"beat_fired({pred.beat})" + ("" if fired else " — not yet")
    if isinstance(pred, RecentClassifierValue):
        collected: list[str] = []
        for reply in reversed(state.replies):
            for axis in reply.classifier_results:
                if axis.quality == pred.quality:
                    collected.append(axis.value)
                    break
            if len(collected) >= pred.window:
                break
        matches = sum(1 for v in collected if v in pred.values)
        match_label = pred.match if isinstance(pred.match, str) else f">={pred.match}"
        values_text = "{" + ", ".join(pred.values) + "}"
        return (
            f"recent({pred.quality}, last {pred.window}) {match_label} "
            f"in {values_text} — got {matches}/{len(collected)}"
        )
    if isinstance(pred, Not_):
        return f"not({describe_predicate(pred.of, state)})"
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


def _fmt_quality(value: float | None) -> str:
    return "(unset)" if value is None else f"{value:g}"


# ───────────────────────── quality mutation ─────────────────────────


def _clamp_quality(value: float, quality_id: str, scenario: Scenario) -> float:
    q = scenario.quality_by_id.get(quality_id)
    if q is None:
        return value
    if q.min is not None and value < q.min:
        return q.min
    if q.max is not None and value > q.max:
        return q.max
    return value


def apply_classification(
    state: InboxState,
    scenario: Scenario,
    classify_reply: ClassifyReply,
    value: str,
    rationale: str = "",
) -> AxisResult | None:
    """Apply a classifier axis output.

    Returns the resulting `AxisResult` on success, or None if `value`
    isn't in the authored map. The caller is responsible for appending
    the returned `AxisResult` onto an active `Reply.classifier_results`
    if it wants `RecentClassifierValue` predicates to see this write.

    Two modes:
    - When `classify_reply.quality` is set: the mapped score writes to that
      quality (replace by default; add when `accumulate=True`). Both paths
      share `_clamp_quality`.
    - When `classify_reply.quality is None`: this is a verdict-only axis.
      No quality write happens; the AxisResult still returns so the caller
      can append it onto Reply.classifier_results for RecentClassifierValue.

    Quality writes log a structured before→after→delta line. Use
    `quality_delta_for_classifier` to materialise a `QualityDelta` for
    callers that thread the write into a tick-level decision payload.
    """
    score = classify_reply.map.get(value)
    if score is None:
        return None
    if classify_reply.quality is not None:
        before = state.qualities.get(classify_reply.quality)
        cur = before if before is not None else 0.0
        if classify_reply.accumulate:
            new = cur + score
        else:
            new = score
        clamped = _clamp_quality(new, classify_reply.quality, scenario)
        state.qualities[classify_reply.quality] = clamped
        if before != clamped:
            delta = clamped - cur
            sign = "+" if delta >= 0 else ""
            logfire.info(
                "quality {q}: {before} -> {after} ({sign}{delta:g}) via classifier:{axis}",
                q=classify_reply.quality,
                before=_fmt_quality(before),
                after=f"{clamped:g}",
                sign=sign,
                delta=delta,
                axis=classify_reply.identifier,
            )
    return AxisResult(
        quality=classify_reply.identifier, value=value, rationale=rationale
    )


def quality_delta_for_classifier(
    classify_reply: ClassifyReply,
    before: float | None,
    after: float | None,
) -> QualityDelta | None:
    """Build a `QualityDelta` for an applied classifier write. Returns None
    for verdict-only axes (no quality written) or no-op writes (value
    unchanged after clamp). Caller threads the delta into `TickDecision`."""
    if classify_reply.quality is None or after is None:
        return None
    if before == after:
        return None
    return QualityDelta(
        quality=classify_reply.quality,
        before=before,
        after=after,
        source=f"classifier:{classify_reply.identifier}",
    )


def apply_writes_by_axis(
    state: InboxState,
    scenario: Scenario,
    wba: WritesByAxis,
    *,
    beat_id: str | None = None,
) -> list[QualityDelta]:
    """Apply a beat's `spec.writes_by_axis` block.

    Walks `state.replies` in reverse looking for the most recent
    `AxisResult` whose identifier matches `wba.axis`. Replies without a
    matching axis are skipped (same shape as
    `_evaluate_recent_classifier_value`); the search stops after `wba.window`
    matching results have been collected. The most-recent matching verdict
    indexes into `wba.arms`; if that arm is declared, its writes are applied
    with SET semantics (clamped). Missing axis result or unmatched arm →
    no-op + a logfire.warn.

    Returns the list of `QualityDelta`s for any quality whose value
    actually changed (no-op writes are dropped). Source labels read
    `spec.writes_by_axis:<beat_id>:<arm>` for an auditable trail.
    """
    collected: list[str] = []
    for reply in reversed(state.replies):
        for axis in reply.classifier_results:
            if axis.quality == wba.axis:
                collected.append(axis.value)
                break
        if len(collected) >= wba.window:
            break

    if not collected:
        logfire.warn(
            "writes_by_axis no recent classifier value for axis {axis}",
            axis=wba.axis, beat_id=beat_id,
        )
        return []

    verdict = collected[0]  # most recent within window
    arm_writes = wba.arms.get(verdict)
    if arm_writes is None:
        logfire.warn(
            "writes_by_axis verdict {verdict!r} has no matching arm",
            verdict=verdict, axis=wba.axis, beat_id=beat_id,
            available_arms=sorted(wba.arms.keys()),
        )
        return []

    deltas: list[QualityDelta] = []
    source_label = (
        f"spec.writes_by_axis:{beat_id}:{verdict}" if beat_id
        else f"spec.writes_by_axis:{verdict}"
    )
    for quality_id, value in arm_writes.items():
        before = state.qualities.get(quality_id)
        new = _clamp_quality(value, quality_id, scenario)
        state.qualities[quality_id] = new
        if before != new:
            logfire.info(
                "quality {q}: {before} -> {after} via {source}",
                q=quality_id,
                before=_fmt_quality(before),
                after=f"{new:g}",
                source=source_label,
            )
            deltas.append(QualityDelta(
                quality=quality_id, before=before, after=new, source=source_label,
            ))
    return deltas


def apply_writes(
    state: InboxState,
    scenario: Scenario,
    writes: dict[str, float],
    *,
    beat_id: str | None = None,
) -> list[QualityDelta]:
    """Apply a beat's authored `spec.writes` to state with SET semantics
    (no accumulate). Quality bounds clamp via `_clamp_quality`. Called
    once per fire from `_fire_beat` after the fired-id guard and before
    the silent-complete check, so the fact of completion is recorded
    even on the no-render path. Multi-turn beats re-apply on every fire
    — see the BeatSpec.writes field comment for the collision contract.

    Returns the list of `QualityDelta`s for any quality whose value
    actually changed (no-op writes are dropped). `beat_id`, when
    provided, is included in the log line and the `source` of each
    returned delta.
    """
    deltas: list[QualityDelta] = []
    source_label = f"spec.writes:{beat_id}" if beat_id else "spec.writes"
    for quality_id, value in writes.items():
        before = state.qualities.get(quality_id)
        new = _clamp_quality(value, quality_id, scenario)
        state.qualities[quality_id] = new
        if before != new:
            logfire.info(
                "quality {q}: {before} -> {after} via {source}",
                q=quality_id,
                before=_fmt_quality(before),
                after=f"{new:g}",
                source=source_label,
            )
            deltas.append(QualityDelta(
                quality=quality_id, before=before, after=new, source=source_label,
            ))
    return deltas


# ───────────────────────── beat scheduler ─────────────────────────


def find_due_beats(
    state: InboxState, scenario: Scenario
) -> tuple[list[Beat], list[BeatEligibility]]:
    """Return (eligible beats, eligibility report).

    The report classifies every "interesting" pending beat as eligible or
    blocked. Beats already in `fired_beat_ids` are excluded (uninteresting
    after they've fired). Beats blocked on an unsatisfied positive-form
    `BeatFired` prereq are also excluded — they're "structurally not ready"
    (downstream of unfired work) and listing them every tick is noise the
    author doesn't need.

    For blocked beats that DO show up, `blocked_by` carries a rendered
    description of the first failing predicate (with concrete values, e.g.
    `engagement>=4.0 got 2.5`). That's what the per-tick console log
    surfaces and the debug overlay displays.

    Beats stay in `pending_beats` until they're added to `fired_beat_ids`.
    Fire-once beats land in `fired_beat_ids` immediately after rendering, so
    they're pruned on the next tick. `multi_turn` beats never enter
    `fired_beat_ids` via their own state — they re-appear in the due list
    each tick until a sibling beat's mutex (e.g. `Not_(BeatFired terminal)`)
    catches them.
    """
    eligible: list[Beat] = []
    report: list[BeatEligibility] = []
    pending_by_id = {pb.beat_id: pb for pb in state.pending_beats}

    for beat in scenario.beats:
        pb = pending_by_id.get(beat.id)
        if pb is None:
            continue
        if beat.id in state.fired_beat_ids:
            continue
        first_failed = next(
            (p for p in beat.preconditions if not evaluate_predicate(p, state)),
            None,
        )
        is_multi = beat.multi_turn
        if first_failed is None:
            eligible.append(beat)
            report.append(BeatEligibility(
                beat_id=beat.id, status="eligible", multi_turn=is_multi,
            ))
        elif _has_unsatisfied_structural_prereq(beat.preconditions, state):
            # Downstream of unfired work — omit from report (noise).
            continue
        else:
            report.append(BeatEligibility(
                beat_id=beat.id, status="blocked",
                blocked_by=describe_predicate(first_failed, state),
                multi_turn=is_multi,
            ))

    # Prune completed beats from pending; keep multi-turn beats around so
    # they re-fire on subsequent ticks until a mutex catches them.
    still_pending: list[PendingBeat] = []
    for pb in state.pending_beats:
        if pb.beat_id in state.fired_beat_ids:
            continue
        if pb.beat_id not in scenario.beat_by_id:
            logfire.warn("pending beat not in scenario", beat_id=pb.beat_id)
            continue
        still_pending.append(pb)
    state.pending_beats = still_pending
    return eligible, report


def _has_unsatisfied_structural_prereq(
    preds: list[Predicate], state: InboxState
) -> bool:
    """A "structural prereq" is a positive-form `BeatFired` predicate. If
    any such prereq isn't yet satisfied, the beat is downstream of unfired
    work. We omit such beats from the eligibility report entirely — every
    tick the author can already see the upstream beat hasn't fired, so
    repeating "downstream blocked: beat_fired(X) — not yet" N ticks running
    is just noise. Negative-form `Not_(BeatFired …)` is excluded — that's
    the mutual-exclusion idiom (always satisfied until the other beat
    actually fires), so it never causes the noise this skip is preventing.
    """
    for p in preds:
        if isinstance(p, BeatFired) and p.beat not in state.fired_beat_ids:
            return True
    return False




def thread_id_from_beat_id(beat_id: str | None) -> str | None:
    """Beat ids are namespaced as `<thread>.<local>` — extract the thread.
    Single source of truth for this convention; do NOT re-implement elsewhere.
    """
    if not beat_id:
        return None
    return beat_id.split(".", 1)[0] if "." in beat_id else beat_id
