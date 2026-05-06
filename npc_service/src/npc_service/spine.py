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

Beat selection: `find_due_beats` returns every pending beat whose
preconditions pass. When more than one beat is eligible on the same tick,
the LLM Director (in `agents.py`) picks one — the spine no longer maintains
same-tick mutex tracking. Authoring order is not load-bearing for selection.
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
    """
    score = classify_reply.map.get(value)
    if score is None:
        return None
    if classify_reply.quality is not None:
        if classify_reply.accumulate:
            cur = state.qualities.get(classify_reply.quality, 0.0)
            new = cur + score
        else:
            new = score
        state.qualities[classify_reply.quality] = _clamp_quality(
            new, classify_reply.quality, scenario
        )
    return AxisResult(
        quality=classify_reply.identifier, value=value, rationale=rationale
    )


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
    """Return every pending beat whose preconditions all pass on this tick.
    The Director (`agents.director_agent`) picks one from this list; the
    spine no longer resolves same-tick mutex itself.

    Beats stay in `pending_beats` until they're added to `fired_beat_ids`.
    Fire-once beats land in `fired_beat_ids` immediately after rendering, so
    they're pruned on the next tick. `multi_turn` beats never enter
    `fired_beat_ids` via their own state — they re-appear in the due list
    each tick until a sibling beat's mutex (e.g. `Not_(BeatFired terminal)`)
    catches them.
    """
    due: list[Beat] = []
    pending_by_id = {pb.beat_id: pb for pb in state.pending_beats}

    for beat in scenario.beats:
        pb = pending_by_id.get(beat.id)
        if pb is None:
            continue
        if beat.id in state.fired_beat_ids:
            continue
        if not all(evaluate_predicate(p, state) for p in beat.preconditions):
            continue
        due.append(beat)

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
    return due


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
