"""The deterministic spine: predicate/effect evaluation, beat scheduler, thread
state machine evaluator, voice-band selector, search, and the coda fragment
selector. No LLM calls live here — only code-owned state mutation.

Three rules this module enforces:

1. The prose-rendering LLM never touches state. EMAIL_AGENT returns prose only;
   the spine is the only thing that mutates `InboxState`.
2. Judging agents (TEXTURE_AGENT, VOICE_FIDELITY_AGENT) write *qualities* via
   `apply_texture_score()` / `apply_voice_fidelity()` — they never flip atoms.
3. Thread state machines are invisible to the LLM. State transitions happen
   entirely here, then the model is invoked (if needed) with no awareness of
   which transition just fired.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .models import (
    AddCommitment,
    All,
    AnchoredBeat,
    Any_,
    AtomEquals,
    AtomIn,
    ActionsElapsedGte,
    AvgQualityGte,
    AvgQualityLt,
    Beat,
    CodaBeat,
    DraftVisibleBeat,
    Effect,
    Email,
    InboxState,
    LastReplyHonestyGte,
    NPCDossier,
    Not_,
    PendingBeat,
    PlayerRepliedTo,
    Predicate,
    QualityGte,
    QualityLt,
    Reply,
    RenderedBeat,
    Scenario,
    SceneUpdateBeat,
    ScheduleBeat,
    SetAtom,
    SetQuality,
    Thread,
    ThreadStateIn,
    ThreadTransition,
    VoiceBand,
)


log = logging.getLogger("npc_service.spine")


# Terminal states for Maya's thread — the set the coda gates on. Kept here
# so the values aren't restated as raw string literals across the spine,
# main.py, and tests.
MAYA_TERMINAL_STATES = frozenset({"warm", "ambivalent", "cold"})


# ───────────────────────── predicate evaluation ─────────────────────────


def evaluate_predicate(pred: Predicate, state: InboxState) -> bool:
    if isinstance(pred, AtomEquals):
        return state.atoms.get(pred.atom) == pred.value
    if isinstance(pred, AtomIn):
        return state.atoms.get(pred.atom) in pred.values
    if isinstance(pred, QualityGte):
        v = state.qualities.get(pred.quality)
        return v is not None and v >= pred.value
    if isinstance(pred, QualityLt):
        v = state.qualities.get(pred.quality)
        return v is not None and v < pred.value
    if isinstance(pred, ActionsElapsedGte):
        return state.action_count >= pred.value
    if isinstance(pred, PlayerRepliedTo):
        return any(r.beat_responded_to == pred.beat for r in state.replies)
    if isinstance(pred, ThreadStateIn):
        return state.thread_states.get(pred.thread) in pred.states
    if isinstance(pred, AvgQualityGte):
        avg = _avg_thread_reply_quality(state, pred.thread, pred.quality)
        return avg is not None and avg >= pred.value
    if isinstance(pred, AvgQualityLt):
        avg = _avg_thread_reply_quality(state, pred.thread, pred.quality)
        return avg is not None and avg < pred.value
    if isinstance(pred, LastReplyHonestyGte):
        last = _last_reply_in_thread(state, pred.thread)
        return last is not None and last.honesty >= pred.value
    if isinstance(pred, All):
        return all(evaluate_predicate(p, state) for p in pred.of)
    if isinstance(pred, Any_):
        return any(evaluate_predicate(p, state) for p in pred.of)
    if isinstance(pred, Not_):
        return not evaluate_predicate(pred.of, state)
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


def _avg_thread_reply_quality(state: InboxState, thread_id: str, quality_id: str) -> float | None:
    """For warmth.X / honesty.global trajectory predicates."""
    replies = [r for r in state.replies if r.thread_id == thread_id]
    if not replies:
        return None
    if quality_id.startswith("warmth."):
        return sum(r.warmth for r in replies) / len(replies)
    if quality_id == "honesty.global":
        return sum(r.honesty for r in replies) / len(replies)
    raise ValueError(f"avg_quality predicate: unsupported quality {quality_id!r}")


def _last_reply_in_thread(state: InboxState, thread_id: str) -> Reply | None:
    for r in reversed(state.replies):
        if r.thread_id == thread_id:
            return r
    return None


# ───────────────────────── effect application ─────────────────────────


def apply_effect(eff: Effect, state: InboxState, scenario: Scenario) -> None:
    if isinstance(eff, SetAtom):
        state.atoms[eff.atom] = eff.value
        log.info("effect set_atom %s = %r", eff.atom, eff.value)
        return
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
    if isinstance(eff, AddCommitment):
        state.commitments.add(eff.commitment)
        log.info("effect add_commitment %s", eff.commitment)
        return
    if isinstance(eff, ScheduleBeat):
        # If already pending, keep the earlier deadline.
        target_action = state.action_count + eff.after_actions
        for pb in state.pending_beats:
            if pb.beat_id == eff.beat:
                if target_action < pb.earliest_action:
                    pb.earliest_action = target_action
                return
        if eff.beat in state.fired_beat_ids:
            log.debug("schedule_beat %s ignored (already fired)", eff.beat)
            return
        state.pending_beats.append(PendingBeat(beat_id=eff.beat, earliest_action=target_action))
        log.debug("scheduled %s for action>=%d", eff.beat, target_action)
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
    honesty: float,
) -> None:
    """Apply texture-scorer outputs to qualities. Maya is the only NPC whose
    texture has cumulative effect on a thread state machine; for other NPCs
    only `honesty.global` is affected.

    Quality bounds come from scenario.yaml's QualityDef.min/max via
    `_clamp_quality` — keep that as the single source of truth.
    """
    # honesty.global rolling average across all replies.
    cur_h = state.qualities.get("honesty.global", 0.0)
    n = len(state.replies)  # already includes the just-recorded reply
    new_h = (cur_h * (n - 1) + honesty) / n if n > 0 else honesty
    state.qualities["honesty.global"] = _clamp_quality(new_h, "honesty.global", scenario)

    if recipient_npc_id == "maya":
        cur_w = state.qualities.get("warmth.maya", 0.0)
        state.qualities["warmth.maya"] = _clamp_quality(cur_w + warmth, "warmth.maya", scenario)
        # Register drifts halfway toward last reply's warmth — closes the
        # texture loop visibly within one round.
        cur_r = state.qualities.get("register.maya", 0.0)
        new_r = cur_r + (warmth - cur_r) / 2.0
        state.qualities["register.maya"] = _clamp_quality(new_r, "register.maya", scenario)


def apply_voice_fidelity(state: InboxState, voice_fidelity: float, rationale: str) -> None:
    state.voice_fidelity = max(0.0, min(1.0, voice_fidelity))
    state.voice_fidelity_rationale = rationale


# ───────────────────────── beat scheduler ─────────────────────────


@dataclass
class BeatToFire:
    """A beat the scheduler has decided should fire on this tick. Caller
    decides what to do with it (anchored → emit immediately; rendered →
    invoke EMAIL_AGENT then emit; scene_update → apply effects only).
    """

    beat: Beat


def initialize_boot_inbox(state: InboxState, scenario: Scenario) -> list[BeatToFire]:
    """Called once at game boot. Returns the in_boot_inbox beats in scenario
    order. The caller should fire them immediately (they're all anchored).
    """
    boot_beats: list[BeatToFire] = []
    for beat in scenario.beats:
        if getattr(beat, "in_boot_inbox", False):
            boot_beats.append(BeatToFire(beat=beat))
    return boot_beats


def find_due_beats(state: InboxState, scenario: Scenario) -> list[BeatToFire]:
    """Walk the scenario's beats and pending-schedule list; return all beats
    whose preconditions and arrives_after are now satisfied. Each beat fires
    at most once. Caller is responsible for actually firing.
    """
    due: list[BeatToFire] = []

    # Beats explicitly scheduled by an effect.
    still_pending: list[PendingBeat] = []
    for pb in state.pending_beats:
        if pb.beat_id in state.fired_beat_ids:
            continue
        beat = scenario.beat_by_id.get(pb.beat_id)
        if beat is None:
            log.warning("scheduled beat %s not in scenario", pb.beat_id)
            continue
        if state.action_count < pb.earliest_action:
            still_pending.append(pb)
            continue
        if not all(evaluate_predicate(p, state) for p in beat.preconditions):
            still_pending.append(pb)
            continue
        due.append(BeatToFire(beat=beat))
    state.pending_beats = still_pending

    # Beats whose preconditions just became true on their own (no schedule
    # required). Skip beats that are in_boot_inbox (already fired by
    # initialize_boot_inbox), already fired, or already in the due list.
    due_ids = {bf.beat.id for bf in due}
    for beat in scenario.beats:
        if beat.id in state.fired_beat_ids or beat.id in due_ids:
            continue
        if getattr(beat, "in_boot_inbox", False):
            continue
        if not beat.preconditions:
            continue  # an unscheduled beat with no preconditions would fire forever
        if all(evaluate_predicate(p, state) for p in beat.preconditions):
            due.append(BeatToFire(beat=beat))

    return due


# ───────────────────────── thread state machine ─────────────────────────


def evaluate_thread_transitions(state: InboxState, scenario: Scenario) -> list[BeatToFire]:
    """For every thread, evaluate transitions in order. First matching transition
    fires; its effects are applied, the thread state moves, and any `fire_beat`
    is collected for the caller to fire.
    """
    fire: list[BeatToFire] = []

    for thread in scenario.threads:
        cur_state = state.thread_states.get(thread.id, thread.initial_state)
        for trans in thread.transitions:
            if trans.from_state != cur_state:
                continue
            if not all(evaluate_predicate(p, state) for p in trans.when):
                continue
            log.info("thread %s: %s → %s", thread.id, trans.from_state, trans.to_state)
            state.thread_states[thread.id] = trans.to_state
            for eff in trans.effects:
                apply_effect(eff, state, scenario)
            if trans.fire_beat:
                beat = scenario.beat_by_id.get(trans.fire_beat)
                if beat is None:
                    log.warning("transition fires unknown beat %s", trans.fire_beat)
                else:
                    fire.append(BeatToFire(beat=beat))
            break  # only one transition per tick per thread

    return fire


# ───────────────────────── voice band selector ─────────────────────────


def select_voice_band(npc: NPCDossier, state: InboxState) -> VoiceBand | None:
    """Pick the voice band whose register range contains the current value of
    the NPC's `register_quality`. Returns None if the NPC has no bands (e.g.
    Tess) or no register quality.
    """
    if not npc.voice_bands or not npc.register_quality:
        return None
    register = state.qualities.get(npc.register_quality, 0.0)
    for band in npc.voice_bands:
        if band.register_min <= register <= band.register_max:
            return band
    # Fallback to the first band; should not happen if bands cover the full range.
    return npc.voice_bands[0]


# ───────────────────────── search ─────────────────────────


def search_inbox(state: InboxState, scenario: Scenario, query: str) -> list[dict]:
    """Substring search across emails currently in the inbox plus the archive
    corpus. Free for the player — does not advance actions.elapsed.
    Case-insensitive. Returns ranked hits with snippet.
    """
    q = query.strip().lower()
    if not q:
        return []
    needle = re.compile(re.escape(q), re.IGNORECASE)
    hits: list[tuple[int, dict]] = []

    # Archive emails: pre-existing world content.
    for ae in scenario.archive_emails:
        haystack = "\n".join([ae.subject, ae.from_, ae.body])
        if needle.search(haystack):
            score = haystack.lower().count(q)
            hits.append((score, {
                "id": ae.id,
                "from": ae.from_,
                "to": ae.to,
                "subject": ae.subject,
                "sent_at": ae.sent_at,
                "folder": ae.folder,
                "body": ae.body,
                "snippet": _snippet(ae.body, q),
                "is_archive": True,
            }))

    # Live emails from this session.
    for e in state.emails:
        haystack = "\n".join([e.subject, e.sender, e.body])
        if needle.search(haystack):
            score = haystack.lower().count(q)
            hits.append((score, {
                "id": e.id,
                "from": e.sender,
                "to": e.recipient,
                "subject": e.subject,
                "sent_at": e.sent_at_label,
                "folder": e.folder,
                "body": e.body,
                "snippet": _snippet(e.body, q),
                "is_archive": False,
            }))

    hits.sort(key=lambda t: -t[0])
    return [h[1] for h in hits]


def _snippet(body: str, q: str, radius: int = 80) -> str:
    idx = body.lower().find(q.lower())
    if idx < 0:
        return body[:160]
    start = max(0, idx - radius)
    end = min(len(body), idx + len(q) + radius)
    snippet = body[start:end]
    if start > 0:
        snippet = "…" + snippet
    if end < len(body):
        snippet = snippet + "…"
    return snippet


# ───────────────────────── coda fragment selector ─────────────────────────


@dataclass
class CodaFragment:
    label: str  # short tag the framing-line generator can reference
    excerpt: str  # the actual text shown in the montage


def select_coda_fragments(state: InboxState, scenario: Scenario) -> list[CodaFragment]:
    """Pick deterministic fragments from session content for the coda.

    Foregrounds the player's choices verbatim — the model never paraphrases
    these, only writes the framing lines between them.
    """
    fragments: list[CodaFragment] = []

    # 1. Player's most warmth-positive reply (if any).
    warmest_id: str | None = None
    if state.replies:
        warmest = max(state.replies, key=lambda r: r.warmth)
        if warmest.warmth > 0:
            warmest_id = warmest.id
            fragments.append(CodaFragment(
                label="player_warmest_reply",
                excerpt=_excerpt(warmest.body, 120),
            ))

    # 2. Player's most honesty-positive reply (if any, and a distinct reply
    #    from the warmest — compared by reply.id, not by excerpt prefix).
    if state.replies:
        most_honest = max(state.replies, key=lambda r: r.honesty)
        if most_honest.honesty > 0 and most_honest.id != warmest_id:
            fragments.append(CodaFragment(
                label="player_most_honest_reply",
                excerpt=_excerpt(most_honest.body, 120),
            ))

    # 3. Maya's last received email (her terminal beat). Beat-id namespace is
    # the source of truth for thread membership; see thread_id_from_beat_id.
    maya_emails = [
        e for e in state.emails
        if e.beat_id and thread_id_from_beat_id(e.beat_id) == "maya"
    ]
    if maya_emails:
        last_maya = maya_emails[-1]
        fragments.append(CodaFragment(
            label="maya_final_email",
            excerpt=_closing_lines(last_maya.body, 2),
        ))

    # 4. Eileen's "looking forward to having you back" line, if seen.
    if state.atoms.get("knowledge.tess.was_continuing"):
        fragments.append(CodaFragment(
            label="eileen_welcome_back",
            excerpt="really looking forward to having you back",
        ))

    # 5. Class registration status.
    if "class_registration" in state.commitments:
        fragments.append(CodaFragment(
            label="class_registration_preserved",
            excerpt="CER101 — confirmed.",
        ))
    else:
        # Show the deadline note as a fragment (the thing that didn't happen).
        if state.action_count >= 8:  # only mention if the reminder fired
            fragments.append(CodaFragment(
                label="class_registration_expired",
                excerpt="CER101 — released to the waitlist.",
            ))

    # 6. Unsent draft disposition.
    disp = state.atoms.get("unsent_draft_disposition", "pending")
    if disp == "sent_unchanged":
        # Pull the opening of the draft text.
        draft_email = next((e for e in state.emails if e.sender.startswith("Tess (draft)")), None)
        if draft_email:
            fragments.append(CodaFragment(
                label="draft_sent_unchanged",
                excerpt=_excerpt(draft_email.body, 120),
            ))
    elif disp == "sent_edited":
        if state.voice_fidelity_rationale:
            fragments.append(CodaFragment(
                label="draft_sent_edited",
                excerpt=state.voice_fidelity_rationale,
            ))
    elif disp == "held":
        fragments.append(CodaFragment(
            label="draft_held",
            excerpt="The draft sat in the Drafts folder, unsent.",
        ))

    return fragments


def _excerpt(body: str, max_chars: int) -> str:
    body = body.strip()
    if len(body) <= max_chars:
        return body
    cut = body[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.6:
        cut = cut[:last_space]
    return cut + "…"


def _closing_lines(body: str, n: int) -> str:
    lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
    return " ".join(lines[-n:]) if len(lines) >= n else " ".join(lines)


# ───────────────────────── transcript renderer (P3a) ─────────────────────────


def render_thread_transcript(state: InboxState, thread_id: str) -> str:
    """Plain-text rendering of a thread's email chain. Passed to EMAIL_AGENT
    in the user prompt; no Pydantic AI message_history.
    """
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
    """The slice ends when either (a) the action budget is exhausted, or
    (b) Maya's thread reached a terminal state and the coda's actions_min
    has been crossed.
    """
    if state.ended:
        return
    maya_state = state.thread_states.get("maya")
    if state.action_count >= state.max_actions:
        state.ended = True
        state.end_reason = "budget_exhausted"
        return
    if maya_state in MAYA_TERMINAL_STATES and state.action_count >= state.coda_actions_min:
        state.ended = True
        state.end_reason = "natural"
        return
