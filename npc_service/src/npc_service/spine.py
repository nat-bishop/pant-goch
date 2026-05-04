"""The deterministic spine: predicate evaluation, beat scheduler,
voice-band selector, calibration application, and PlotBeat scheduler.

No LLM calls live here — only code-owned state mutation.

Two rules this module enforces:

1. The chat-rendering LLM never touches state. The chat agent returns prose
   only; the spine is the only thing that mutates `InboxState`.
2. The judging agent (TEXTURE_AGENT) writes the *warmth* and *register*
   qualities via `apply_texture_score()`. It never writes anything else.
"""
from __future__ import annotations

import logging

from .models import (
    Beat,
    BeatFired,
    Email,
    InboxState,
    NPCDossier,
    Not_,
    PendingBeat,
    PlayerRepliesToThreadGte,
    PlotBeat,
    Predicate,
    QualityGte,
    QualityLt,
    Scenario,
    VoiceBand,
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


def apply_texture_score(
    state: InboxState,
    scenario: Scenario,
    recipient_npc_id: str | None,
    warmth: float,
) -> None:
    """Apply the texture-scorer output to qualities. Updates `warmth.<npc>`
    cumulatively and `register.<npc>` with halfway drift toward the new warmth.
    Both are no-ops if the recipient isn't a known NPC.

    Quality bounds come from scenario.yaml's QualityDef.min/max via
    `_clamp_quality` — single source of truth for clamping.
    """
    if recipient_npc_id is None:
        return
    warmth_id = f"warmth.{recipient_npc_id}"
    register_id = f"register.{recipient_npc_id}"

    if warmth_id in state.qualities:
        cur_w = state.qualities.get(warmth_id, 0.0)
        state.qualities[warmth_id] = _clamp_quality(cur_w + warmth, warmth_id, scenario)

    if register_id in state.qualities:
        cur_r = state.qualities.get(register_id, 0.0)
        new_r = cur_r + (warmth - cur_r) / 2.0
        state.qualities[register_id] = _clamp_quality(new_r, register_id, scenario)


# ───────────────────────── calibration ─────────────────────────


def apply_calibration_delta(state: InboxState, scenario: Scenario, delta: float) -> None:
    """Bump the calibration quality by `delta`, clamped against its QualityDef
    bounds. Reaching the max ends the session — terminals are authored to fire
    well before then, so this is a safety net rather than the expected end.
    """
    cur = state.qualities.get(CALIBRATION_QUALITY, 0.0)
    new = _clamp_quality(cur + delta, CALIBRATION_QUALITY, scenario)
    state.qualities[CALIBRATION_QUALITY] = new
    log.debug("calibration %s -> %s", cur, new)
    q = scenario.quality_by_id.get(CALIBRATION_QUALITY)
    if not state.ended and q is not None and q.max is not None and new >= q.max:
        state.ended = True
        state.end_reason = "calibration_complete"


def apply_calibration_from_warmth(
    state: InboxState,
    scenario: Scenario,
    warmth: float,
    *,
    base: float,
    coef: float,
) -> float:
    """Advance calibration on a /send: `delta = base + coef·|warmth|`.

    The texture-scored warmth (range -3..3) drives pacing — strong signal in
    either direction calibrates faster, lukewarm sends barely move it. Returns
    the delta applied for telemetry.
    """
    delta = base + coef * abs(warmth)
    apply_calibration_delta(state, scenario, delta)
    return delta


# ───────────────────────── beat scheduler ─────────────────────────


def find_due_beats(state: InboxState, scenario: Scenario) -> list[Beat]:
    """Return all pending beats whose preconditions (authored + runtime) all
    pass on this tick. Pending beats survive to the next tick when not yet
    eligible. Each beat fires at most once.

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
        just_due_ids.add(beat.id)

    still_pending: list[PendingBeat] = []
    for pb in state.pending_beats:
        if pb.beat_id in just_due_ids or pb.beat_id in state.fired_beat_ids:
            continue
        if pb.beat_id not in scenario.beat_by_id:
            log.warning("pending beat %s not in scenario", pb.beat_id)
            continue
        still_pending.append(pb)
    state.pending_beats = still_pending
    return due


def find_eligible_plot_beat(
    state: InboxState, scenario: Scenario, thread_id: str
) -> PlotBeat | None:
    """Walk scenario.beats in authoring order; return the first un-fired
    PlotBeat in `thread_id` whose authored preconditions pass.

    Authoring order is the tiebreaker for mutual exclusion: terminal_warm
    is authored before terminal_standard, so when both are eligible we
    return terminal_warm first. terminal_standard's `Not_(BeatFired(...))`
    precondition then keeps it from firing on subsequent ticks.

    Returns None if no PlotBeat in this thread is currently eligible — the
    caller (typically /send) falls back to a chat turn.
    """
    for beat in scenario.beats:
        if not isinstance(beat, PlotBeat):
            continue
        if beat.thread != thread_id:
            continue
        if beat.id in state.fired_beat_ids:
            continue
        # Skip if already pending — once queued we don't re-resolve.
        if any(pb.beat_id == beat.id for pb in state.pending_beats):
            continue
        if not all(evaluate_predicate(p, state) for p in beat.preconditions):
            continue
        return beat
    return None


def schedule_thread_reply(
    state: InboxState, scenario: Scenario, npc_id: str, beat: PlotBeat
) -> None:
    """Queue a resolved PlotBeat for an NPC's reply with a calibration-
    threshold runtime predicate.

    The threshold sits at `current_calibration + reply_delay_calibration`,
    above the post-send value, so the queued PlotBeat doesn't fire on its
    own send tick. A subsequent /send (or several /opens — actually no,
    /open no longer mutates calibration) ticks calibration past the
    threshold and the beat lands on the next scheduler tick. This is what
    makes plot moments feel like authored events rather than direct chat
    replies.

    Does NOT enforce one-pending-per-thread: callers (find_eligible_plot_beat)
    already filter pending. ChatTurn fallbacks fire immediately within /send
    and never go through this function — they are not stored in pending_beats.
    """
    npc = scenario.npc_by_id.get(npc_id)
    if npc is None:
        return
    cur_cal = state.qualities.get(CALIBRATION_QUALITY, 0.0)
    threshold = cur_cal + npc.reply_delay_calibration
    state.pending_beats.append(
        PendingBeat(
            beat_id=beat.id,
            runtime_preconditions=[
                QualityGte(quality=CALIBRATION_QUALITY, value=threshold)
            ],
        )
    )
    log.info(
        "scheduled plot beat %s for %s at calibration>=%.2f",
        beat.id, npc_id, threshold,
    )


# ───────────────────────── voice band selector ─────────────────────────


def select_voice_band(npc: NPCDossier, state: InboxState) -> VoiceBand | None:
    """Pick the voice band whose register range contains the current value of
    the NPC's `register_quality`. Returns None if the NPC has no bands or no
    register quality.
    """
    if not npc.voice_bands or not npc.register_quality:
        return None
    register = state.qualities.get(npc.register_quality, 0.0)
    for band in npc.voice_bands:
        if band.register_min <= register <= band.register_max:
            return band
    return npc.voice_bands[0]


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
