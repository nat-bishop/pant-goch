"""The deterministic spine: predicate evaluation, beat scheduler, and
calibration application.

No LLM calls live here — only code-owned state mutation.

Two rules this module enforces:

1. The chat-rendering LLM never touches state. The chat agent returns prose
   only; the spine is the only thing that mutates `InboxState`.
2. The classifier agent (when a beat authors `classify_reply`) writes a
   single named quality via `apply_classification()`. It writes nothing else.
"""
from __future__ import annotations

import logging

from .models import (
    Beat,
    BeatFired,
    ClassifyReply,
    Email,
    InboxState,
    Not_,
    PendingBeat,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityGte,
    QualityLt,
    Scenario,
)


log = logging.getLogger("npc_service.spine")

CALIBRATION_QUALITY = "calibration"


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
    if isinstance(pred, Not_):
        return not evaluate_predicate(pred.of, state, extra_fired)
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


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
) -> bool:
    """Apply a classifier output to the named quality. Returns True if the
    value was applied; False if `value` isn't in the authored map (caller
    should log + leave the quality alone). LLM-judge → single named quality
    with `_clamp_quality` bounding.
    """
    score = classify_reply.map.get(value)
    if score is None:
        return False
    state.qualities[classify_reply.quality] = _clamp_quality(
        score, classify_reply.quality, scenario
    )
    return True


# ───────────────────────── calibration ─────────────────────────


def apply_calibration_delta(state: InboxState, scenario: Scenario, delta: float) -> float:
    """Bump the calibration quality by `delta`, clamped against its QualityDef
    bounds. Returns the delta applied for telemetry. Reaching the max ends
    the session — terminals are authored to fire well before then, so this
    is a safety net rather than the expected end.
    """
    cur = state.qualities.get(CALIBRATION_QUALITY, 0.0)
    new = _clamp_quality(cur + delta, CALIBRATION_QUALITY, scenario)
    state.qualities[CALIBRATION_QUALITY] = new
    log.debug("calibration %s -> %s", cur, new)
    q = scenario.quality_by_id.get(CALIBRATION_QUALITY)
    if not state.ended and q is not None and q.max is not None and new >= q.max:
        state.ended = True
        state.end_reason = "calibration_complete"
    return delta


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
    via BeatFired (e.g., terminal_warm and terminal_standard).
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
            log.warning("pending beat %s not in scenario", pb.beat_id)
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


def latest_npc_email_in_thread(
    state: InboxState, thread_id: str, npc_address: str
) -> Email | None:
    """Most recent email in the thread sent BY the NPC (matched by sender
    address). Used to derive the chat-turn email's `recipient` field — we
    take the NPC's prior email's `recipient` as the player's address.
    """
    for e in reversed(state.emails):
        if e.thread_id == thread_id and e.sender == npc_address:
            return e
    return None


def thread_id_from_beat_id(beat_id: str | None) -> str | None:
    """Beat ids are namespaced as `<thread>.<local>` — extract the thread.
    Single source of truth for this convention; do NOT re-implement elsewhere.
    """
    if not beat_id:
        return None
    return beat_id.split(".", 1)[0] if "." in beat_id else beat_id
