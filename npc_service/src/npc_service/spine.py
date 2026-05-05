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
"""
from __future__ import annotations

import logfire

from .models import (
    AxisResult,
    Beat,
    BeatFired,
    ClassifyReply,
    InboxState,
    Not_,
    PendingBeat,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityGte,
    QualityLt,
    RecentClassifierValue,
    Scenario,
)


# ───────────────────────── predicate evaluation ─────────────────────────


def evaluate_predicate(
    pred: Predicate,
    state: InboxState,
    extra_fired: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """`extra_fired` is a tick-local set of beat ids that are about to fire
    on this same tick. BeatFired predicates check both `state.fired_beat_ids`
    and `extra_fired`, so a Not(BeatFired) precondition can express mutual
    exclusion between two beats that become eligible simultaneously.
    `find_due_beats` is the only caller that passes a non-empty value.
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
        return pred.beat in state.fired_beat_ids or pred.beat in extra_fired
    if isinstance(pred, RecentClassifierValue):
        return _evaluate_recent_classifier_value(pred, state)
    if isinstance(pred, Not_):
        return not evaluate_predicate(pred.of, state, extra_fired)
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


def _evaluate_recent_classifier_value(
    pred: RecentClassifierValue, state: InboxState
) -> bool:
    """Walk replies in reverse, collecting up to `pred.window` axis results
    for `pred.quality`. Replies without a matching-quality result are
    skipped (they don't fill a window slot). Resolve `pred.match` against
    the count of collected results whose value ∈ pred.values.
    """
    collected: list[str] = []
    for reply in reversed(state.replies):
        for axis in reply.classifier_results:
            if axis.quality == pred.quality:
                collected.append(axis.value)
                break  # axis-uniqueness invariant: ≤1 per reply per quality
        if len(collected) >= pred.window:
            break

    matches = sum(1 for v in collected if v in pred.values)
    if pred.match == "any":
        return matches >= 1
    if pred.match == "all":
        # Under-filled window can't satisfy "all" — len(collected) < window.
        return matches >= pred.window
    return matches >= pred.match


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
    """Apply a classifier axis output to the named quality.

    Returns the resulting `AxisResult` on success, or None if `value`
    isn't in the authored map. The caller is responsible for appending
    the returned `AxisResult` onto an active `Reply.classifier_results`
    if it wants `RecentClassifierValue` predicates to see this write.

    With `accumulate=False` (default) the mapped score replaces the quality.
    With `accumulate=True` it's added to the current value — used for
    cumulative signals like aria.loop's engagement. Both paths share
    `_clamp_quality`.
    """
    score = classify_reply.map.get(value)
    if score is None:
        return None
    if classify_reply.accumulate:
        cur = state.qualities.get(classify_reply.quality, 0.0)
        new = cur + score
    else:
        new = score
    state.qualities[classify_reply.quality] = _clamp_quality(
        new, classify_reply.quality, scenario
    )
    return AxisResult(quality=classify_reply.quality, value=value, rationale=rationale)


def apply_writes(
    state: InboxState,
    scenario: Scenario,
    writes: dict[str, float],
) -> None:
    """Apply a beat's authored `spec.writes` to state with SET semantics
    (no accumulate). Quality bounds clamp via `_clamp_quality`. Called
    once per fire from `_fire_beat` after the fired-id guard and before
    the silent-complete check, so the fact of completion is recorded
    even on the no-render path. Multi-turn beats re-apply on every fire
    — see the BeatSpec.writes field comment for the collision contract.
    """
    for quality_id, value in writes.items():
        state.qualities[quality_id] = _clamp_quality(value, quality_id, scenario)


# ───────────────────────── beat scheduler ─────────────────────────


def find_due_beats(state: InboxState, scenario: Scenario) -> list[Beat]:
    """Return all pending beats whose preconditions (authored + runtime) all
    pass on this tick. Pending beats survive to the next tick when not yet
    eligible.

    Beats stay in `pending_beats` until they're added to `fired_beat_ids`.
    Fire-once beats (empty `completes_when`) get added by `_fire_beat`
    immediately after rendering, so they're pruned on the next tick.
    Multi-turn beats stay re-fireable until their `completes_when` predicates
    pass — `_fire_beat` only marks them complete then, so they re-appear in
    the due list each tick until satisfied.

    Pending beats are evaluated in scenario authoring order — that's how
    authors break ties when two beats sharing a tick reference each other
    via BeatFired (e.g., aria.complete_warm and aria.complete_standard).
    """
    due: list[Beat] = []
    pending_by_id = {pb.beat_id: pb for pb in state.pending_beats}
    just_due_ids: set[str] = set()

    for beat in scenario.beats:
        pb = pending_by_id.get(beat.id)
        if pb is None:
            continue
        if beat.id in state.fired_beat_ids:
            continue
        all_preds = list(beat.preconditions) + list(pb.runtime_preconditions)
        if not all(evaluate_predicate(p, state, just_due_ids) for p in all_preds):
            continue
        due.append(beat)
        # Only shadow downstream `BeatFired` predicates if this beat will
        # actually complete this tick — otherwise a multi-turn beat that
        # is rendering-but-not-yet-completing would falsely satisfy them.
        if evaluate_completes_when(state, beat, just_due_ids):
            just_due_ids.add(beat.id)

    # Prune completed beats from pending; keep multi-turn beats around so
    # they re-fire on subsequent ticks until completes_when passes.
    still_pending: list[PendingBeat] = []
    for pb in state.pending_beats:
        if pb.beat_id in state.fired_beat_ids:
            continue
        if pb.beat_id not in scenario.beat_by_id:
            logfire.warn("pending beat not in scenario", beat_id=pb.beat_id)
            continue
        still_pending.append(pb)
    state.pending_beats = still_pending
    return due


def evaluate_completes_when(
    state: InboxState,
    beat: Beat,
    extra_fired: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """A beat with empty `completes_when` always completes on first render
    (legacy fire-once semantics). A beat with non-empty `completes_when`
    completes only when all predicates pass.

    `extra_fired` mirrors the parameter on `evaluate_predicate`: pass the
    tick-local "just due" set when checking from inside `find_due_beats` so
    `BeatFired` predicates can resolve same-tick. `_fire_beat` calls with
    the default empty set — by then `find_due_beats` has already returned
    and the rendering loop is processing one beat at a time.
    """
    if not beat.completes_when:
        return True
    return all(evaluate_predicate(p, state, extra_fired) for p in beat.completes_when)


# ───────────────────────── reply-subject helper ─────────────────────────


def reply_subject_for_email(state: InboxState, email_id: str | None) -> str:
    """Subject for a reply: `Re: <X>` if not already prefixed, else unchanged.

    Looks up the target by email id (not beat id). Used by both player /send
    and chat-turn render so subject formatting is consistent across all
    reply paths.
    """
    if email_id is None:
        return "(no subject)"
    target = next((e for e in state.emails if e.id == email_id), None)
    if target is None:
        return "(no subject)"
    s = target.subject
    return s if s.lower().startswith("re:") else f"Re: {s}"


def thread_id_from_beat_id(beat_id: str | None) -> str | None:
    """Beat ids are namespaced as `<thread>.<local>` — extract the thread.
    Single source of truth for this convention; do NOT re-implement elsewhere.
    """
    if not beat_id:
        return None
    return beat_id.split(".", 1)[0] if "." in beat_id else beat_id
