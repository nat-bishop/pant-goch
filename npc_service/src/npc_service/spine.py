"""The deterministic spine: predicate/effect evaluation, beat scheduler,
voice-band selector, time helpers, and NPC-reply scheduler.

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
    Effect,
    Email,
    InboxState,
    NPCDossier,
    Not_,
    PendingBeat,
    PlayerRepliedTo,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityGte,
    QualityLt,
    RenderedBeat,
    Reply,
    Scenario,
    ScheduleBeat,
    SetQuality,
    TimeHoursGte,
    VoiceBand,
)


log = logging.getLogger("npc_service.spine")


# ───────────────────────── predicate evaluation ─────────────────────────


def evaluate_predicate(pred: Predicate, state: InboxState) -> bool:
    if isinstance(pred, QualityGte):
        v = state.qualities.get(pred.quality)
        return v is not None and v >= pred.value
    if isinstance(pred, QualityLt):
        v = state.qualities.get(pred.quality)
        return v is not None and v < pred.value
    if isinstance(pred, TimeHoursGte):
        return state.game_time_hours >= pred.value
    if isinstance(pred, PlayerRepliedTo):
        return any(r.beat_responded_to == pred.beat for r in state.replies)
    if isinstance(pred, PlayerRepliesToThreadGte):
        n = sum(1 for r in state.replies if r.thread_id == pred.thread)
        return n >= pred.value
    if isinstance(pred, All):
        return all(evaluate_predicate(p, state) for p in pred.of)
    if isinstance(pred, Any_):
        return any(evaluate_predicate(p, state) for p in pred.of)
    if isinstance(pred, Not_):
        return not evaluate_predicate(pred.of, state)
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
    if isinstance(eff, ScheduleBeat):
        target_time = state.game_time_hours + eff.after_hours
        for pb in state.pending_beats:
            if pb.beat_id == eff.beat:
                if target_time < pb.earliest_time_hours:
                    pb.earliest_time_hours = target_time
                return
        if eff.beat in state.fired_beat_ids:
            log.debug("schedule_beat %s ignored (already fired)", eff.beat)
            return
        state.pending_beats.append(PendingBeat(beat_id=eff.beat, earliest_time_hours=target_time))
        log.debug("scheduled %s for time>=%.2fh", eff.beat, target_time)
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


# ───────────────────────── time advancement ─────────────────────────


def advance_time(state: InboxState, hours: float) -> None:
    state.game_time_hours += hours


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
    order. The caller should fire them immediately.
    """
    return [BeatToFire(beat=b) for b in scenario.beats if getattr(b, "in_boot_inbox", False)]


def find_due_beats(state: InboxState, scenario: Scenario) -> list[BeatToFire]:
    """Return all beats whose preconditions and `earliest_time_hours` are
    now satisfied. Each beat fires at most once.
    """
    due: list[BeatToFire] = []

    # Pass A: explicitly-scheduled (pending) beats.
    still_pending: list[PendingBeat] = []
    for pb in state.pending_beats:
        if pb.beat_id in state.fired_beat_ids:
            continue
        beat = scenario.beat_by_id.get(pb.beat_id)
        if beat is None:
            log.warning("scheduled beat %s not in scenario", pb.beat_id)
            continue
        if state.game_time_hours < pb.earliest_time_hours:
            still_pending.append(pb)
            continue
        if not all(evaluate_predicate(p, state) for p in beat.preconditions):
            still_pending.append(pb)
            continue
        due.append(BeatToFire(beat=beat))
    state.pending_beats = still_pending

    # Pass B: beats whose preconditions are now true on their own. Skip beats
    # that are in_boot_inbox (already fired by initialize_boot_inbox), already
    # fired, or already in the due list. Beats with NO preconditions are
    # excluded so they don't fire forever — those are scheduled explicitly via
    # boot or by `schedule_npc_reply`.
    due_ids = {bf.beat.id for bf in due}
    for beat in scenario.beats:
        if beat.id in state.fired_beat_ids or beat.id in due_ids:
            continue
        if getattr(beat, "in_boot_inbox", False):
            continue
        if not beat.preconditions:
            continue
        if all(evaluate_predicate(p, state) for p in beat.preconditions):
            due.append(BeatToFire(beat=beat))

    return due


def schedule_npc_reply(state: InboxState, scenario: Scenario, npc_id: str) -> bool:
    """Player just sent to an NPC — schedule that NPC's next reply.

    Walks scenario.beats in order, finds the first un-fired beat where the
    beat's thread is the NPC's thread AND preconditions pass. Schedules it
    for `current_time + midpoint(npc.reply_latency_hours)`.

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

    # If anything is already pending for this NPC's thread, don't re-schedule.
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
        latency_lo, latency_hi = npc.reply_latency_hours
        latency = (latency_lo + latency_hi) / 2.0
        target_time = state.game_time_hours + latency
        state.pending_beats.append(
            PendingBeat(beat_id=beat.id, earliest_time_hours=target_time)
        )
        log.info("scheduled npc reply %s for %s at time>=%.2fh", beat.id, npc_id, target_time)
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
    # Fallback to the first band; should not happen if bands cover the full range.
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


# ───────────────────────── end conditions ─────────────────────────


def check_end_conditions(state: InboxState) -> None:
    """The session ends when the world clock reaches `max_hours`."""
    if state.ended:
        return
    if state.game_time_hours >= state.max_hours:
        state.ended = True
        state.end_reason = "time_exhausted"
