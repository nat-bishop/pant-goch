"""The deterministic spine: predicate/effect evaluation, beat scheduler,
voice-band selector, calibration application, and NPC-reply scheduler.

No LLM calls live here — only code-owned state mutation.

Two rules this module enforces:

1. The prose-rendering LLM never touches state. EMAIL_AGENT returns prose only;
   the spine is the only thing that mutates `InboxState`.
2. The judging agent (TEXTURE_AGENT) writes the *warmth* quality via
   `apply_texture_score()`. It never writes anything else.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import (
    All,
    AnchoredBeat,
    Any_,
    Beat,
    BeatFired,
    Effect,
    InboxState,
    NPCDossier,
    Not_,
    PendingBeat,
    PlayerRepliedTo,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityGte,
    QualityLt,
    Scenario,
    SetQuality,
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
    if isinstance(pred, PlayerRepliedTo):
        return any(r.beat_responded_to == pred.beat for r in state.replies)
    if isinstance(pred, PlayerRepliesToThreadGte):
        n = sum(1 for r in state.replies if r.thread_id == pred.thread)
        return n >= pred.value
    if isinstance(pred, BeatFired):
        return pred.beat in state.fired_beat_ids or pred.beat in extra_fired
    if isinstance(pred, All):
        return all(evaluate_predicate(p, state, extra_fired) for p in pred.of)
    if isinstance(pred, Any_):
        return any(evaluate_predicate(p, state, extra_fired) for p in pred.of)
    if isinstance(pred, Not_):
        return not evaluate_predicate(pred.of, state, extra_fired)
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


# ───────────────────────── effect application ─────────────────────────


def apply_effect(eff: Effect, state: InboxState, scenario: Scenario) -> None:
    if isinstance(eff, SetQuality):
        cur = state.qualities.get(eff.quality, 0.0)
        if eff.op == "set":
            new = float(eff.value)
        elif eff.op == "add":
            new = cur + float(eff.value)
        elif eff.op == "drift_halfway":
            new = cur + (float(eff.value) - cur) / 2.0
        else:
            raise ValueError(f"unknown SetQuality op: {eff.op}")
        new = _clamp_quality(new, eff.quality, scenario)
        state.qualities[eff.quality] = new
        log.debug("effect set_quality %s %s %s -> %s", eff.quality, eff.op, eff.value, new)
        return
    raise TypeError(f"unhandled effect: {type(eff).__name__}")


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
    """Bump the calibration quality by `delta`, clamped via the standard
    SetQuality path. Reaching the max ends the session — terminals are
    authored to fire well before then, so this is a safety net rather than
    the expected end.
    """
    apply_effect(
        SetQuality(quality=CALIBRATION_QUALITY, op="add", value=delta),
        state, scenario,
    )
    new = state.qualities[CALIBRATION_QUALITY]
    q = scenario.quality_by_id.get(CALIBRATION_QUALITY)
    if not state.ended and q is not None and q.max is not None and new >= q.max:
        state.ended = True
        state.end_reason = "calibration_complete"


# ───────────────────────── beat scheduler ─────────────────────────


@dataclass
class BeatToFire:
    """A beat the scheduler has decided should fire on this tick. Caller
    decides what to do with it (anchored → emit immediately; rendered →
    invoke EMAIL_AGENT then emit).
    """

    beat: Beat


def initialize_boot_inbox(state: InboxState, scenario: Scenario) -> list[BeatToFire]:
    """Called once at game boot. Returns the in_boot_inbox beats in scenario
    order. The caller should fire them immediately. (Auto-pending of other
    anchored beats happens in `InboxState.from_scenario`.)
    """
    return [BeatToFire(beat=b) for b in scenario.beats if getattr(b, "in_boot_inbox", False)]


def find_due_beats(state: InboxState, scenario: Scenario) -> list[BeatToFire]:
    """Return all pending beats whose preconditions (authored + runtime) all
    pass on this tick. Pending beats survive to the next tick when not yet
    eligible. Each beat fires at most once.

    Pending beats are evaluated in scenario authoring order — that's how
    authors break ties when two beats sharing a tick reference each other
    via BeatFired (e.g., terminal_warm and terminal_standard).
    """
    due: list[BeatToFire] = []
    pending_by_id = {pb.beat_id: pb for pb in state.pending_beats}
    consumed: set[str] = set()
    just_due_ids: set[str] = set()

    for beat in scenario.beats:
        pb = pending_by_id.get(beat.id)
        if pb is None:
            continue
        consumed.add(beat.id)
        if beat.id in state.fired_beat_ids:
            continue
        all_preds = list(beat.preconditions) + list(pb.runtime_preconditions)
        if not all(evaluate_predicate(p, state, just_due_ids) for p in all_preds):
            continue
        due.append(BeatToFire(beat=beat))
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


def schedule_npc_reply(state: InboxState, scenario: Scenario, npc_id: str) -> bool:
    """Player just sent to an NPC — schedule that NPC's next reply.

    Walks scenario.beats in order, finds the first un-fired beat where the
    beat's thread is the NPC's thread AND its authored preconditions pass.
    Adds it to `pending_beats` with a runtime predicate
    `quality_gte(calibration, current + midpoint(reply_delay_calibration))`.

    The calibration-threshold delay must exceed `SEND_CALIBRATION_DELTA` so a
    single send doesn't satisfy its own queued reply — this is what makes
    Aria feel asynchronous without a wait verb. A second `/send` (or several
    `/open`s) ticks calibration past the threshold and the reply lands.

    If a reply for this NPC is already pending, leaves the existing schedule
    alone — when the pending beat renders, it sees the full transcript
    including the player's new send. (Authors get batched-response behavior
    for free.)

    Returns True if a beat was newly scheduled.
    """
    npc = scenario.npc_by_id.get(npc_id)
    if npc is None:
        return False
    npc_thread_id = scenario.thread_by_npc_id.get(npc_id)
    if npc_thread_id is None:
        return False

    pending_thread_ids = {
        scenario.beat_by_id[pb.beat_id].thread
        for pb in state.pending_beats
        if pb.beat_id in scenario.beat_by_id
    }
    if npc_thread_id in pending_thread_ids:
        return False

    for beat in scenario.beats:
        if beat.thread != npc_thread_id:
            continue
        if beat.id in state.fired_beat_ids:
            continue
        if not all(evaluate_predicate(p, state) for p in beat.preconditions):
            continue
        delay_lo, delay_hi = npc.reply_delay_calibration
        delay = (delay_lo + delay_hi) / 2.0
        cur_cal = state.qualities.get(CALIBRATION_QUALITY, 0.0)
        threshold = cur_cal + delay
        state.pending_beats.append(
            PendingBeat(
                beat_id=beat.id,
                runtime_preconditions=[
                    QualityGte(quality=CALIBRATION_QUALITY, value=threshold)
                ],
            )
        )
        log.info(
            "scheduled npc reply %s for %s at calibration>=%.2f",
            beat.id, npc_id, threshold,
        )
        return True
    return False


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


# ───────────────────────── transcript renderer ─────────────────────────


def render_thread_transcript(state: InboxState, thread_id: str | None) -> str:
    """Plain-text rendering of a thread's email chain. Passed to EMAIL_AGENT
    in the user prompt; no Pydantic AI message_history.
    """
    if thread_id is None:
        return "(no thread context)"
    thread_emails = [e for e in state.emails if e.thread_id == thread_id]
    if not thread_emails:
        return "(no prior emails in this thread)"
    parts: list[str] = []
    for e in thread_emails:
        parts.append(
            f"From: {e.sender}\n"
            f"To: {e.recipient}\n"
            f"Subject: {e.subject}\n"
            f"Sent: {e.sent_at_label}\n"
            f"\n"
            f"{e.body.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def thread_id_from_beat_id(beat_id: str | None) -> str | None:
    """Beat ids are namespaced as `<thread>.<local>` — extract the thread.
    Single source of truth for this convention; do NOT re-implement elsewhere.
    """
    if not beat_id:
        return None
    return beat_id.split(".", 1)[0] if "." in beat_id else beat_id
