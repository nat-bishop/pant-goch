"""Event-sourced session log: every state change in the engine is an Event.

Two flavors:
- **Observation events** — recorded LLM outputs (classifier verdicts, extractor
  candidates, Director picks, chat-agent bodies). Fold does NOT mutate state
  on these; they're held so deterministic replay can re-apply downstream
  effects without calling llama, and so re-execution replay can identify
  what to drop.
- **Mutation events** — the actual state writes (email appends, quality
  writes, fired_beat_ids adds, etc.). `fold` performs the mutation.

Authoring discipline: NEVER remove a `kind` value, only deprecate. Adding a
new event type is `+1 class, +1 union member, +1 fold case` — old saved logs
still deserialize because their kinds are still in the union.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessage

from .models import (
    AxisResult,
    CanonCandidate,
    CanonDropRecord,
    Email,
    FactCheckTrace,
    InboxState,
    PlayerCanonEntry,
    QualityDelta,
    Reply,
)


# ───────────────────────── observation events ─────────────────────────


class ClassifierObserved(BaseModel):
    """Recorded LLM output: classifier verdicts on the player's reply.
    Kept verbatim; downstream `ClassifyEffectApplied` carries the actual
    quality writes derived from these verdicts.
    """
    kind: Literal["classifier_observed"] = "classifier_observed"
    reply_id: str
    verdicts: list[AxisResult]


class ExtractorObserved(BaseModel):
    """Recorded LLM output: extractor candidates and any drops.
    Promotion to player_canon is its own `CanonPromoted` event.
    """
    kind: Literal["extractor_observed"] = "extractor_observed"
    reply_id: str
    candidates: list[CanonCandidate]
    drops: list[CanonDropRecord] = Field(default_factory=list)


class DirectorChose(BaseModel):
    """The Director's per-tick selection (or short-circuit). For `none`,
    `sole`, and `continue` the LLM was not actually called — recorded for
    audit completeness so the log shows every selection, not just the
    director-driven ones.
    """
    kind: Literal["director_chose"] = "director_chose"
    selection_method: Literal["none", "sole", "continue", "director"]
    chosen_beat_id: str | None
    rationale: str | None = None
    eligible: list[str] = Field(default_factory=list)


class ChatBodyRendered(BaseModel):
    """Recorded LLM output: an email body produced by the chat or epilogue
    agent. The corresponding `EmailEmitted` event carries the email itself;
    this event is the trace-level record of the LLM call.
    """
    kind: Literal["chat_body_rendered"] = "chat_body_rendered"
    beat_id: str | None
    npc_id: str
    thread_id: str | None
    body: str
    epilogue: bool = False
    fact_check: FactCheckTrace | None = None


# ───────────────────────── mutation events ─────────────────────────


class SendRecorded(BaseModel):
    """The player's /send entering the system."""
    kind: Literal["send_recorded"] = "send_recorded"
    email: Email
    reply: Reply


class EmailEmitted(BaseModel):
    """An email appended to the inbox (NPC→player or chat-fallback)."""
    kind: Literal["email_emitted"] = "email_emitted"
    email: Email


class EmailRead(BaseModel):
    """A /open marked the email as read."""
    kind: Literal["email_read"] = "email_read"
    email_id: str


class ThreadHistoryAppended(BaseModel):
    """`state.thread_messages[thread_id].extend(messages)`. Producer decides
    what to include (e.g., chat-agent renders strip the synthetic-trigger
    user-turn in outbound mode); fold just appends.
    """
    kind: Literal["thread_history_appended"] = "thread_history_appended"
    thread_id: str
    messages: list[ModelMessage]


class LiteralEffectApplied(BaseModel):
    """One quality write from a beat's literal effect."""
    kind: Literal["literal_effect_applied"] = "literal_effect_applied"
    delta: QualityDelta
    beat_id: str | None = None


class ClassifyEffectApplied(BaseModel):
    """One classifier verdict applied to its target quality. `delta` is None
    when the verdict was recorded on the reply but not written (predicate
    gate failed at apply time, or value unchanged).
    """
    kind: Literal["classify_effect_applied"] = "classify_effect_applied"
    reply_id: str
    axis: AxisResult
    delta: QualityDelta | None = None


class CanonPromoted(BaseModel):
    """One canon entry written. REPLACE semantics on key collision."""
    kind: Literal["canon_promoted"] = "canon_promoted"
    key: str
    entry: PlayerCanonEntry


class BeatFired(BaseModel):
    """A beat fired this tick. `completed=True` adds the beat to
    `fired_beat_ids` (one-shots always; scenes only when scene_exit is
    satisfied).
    """
    kind: Literal["beat_fired"] = "beat_fired"
    beat_id: str
    beat_kind: Literal["one_shot", "scene"]
    scene_turn: int | None = None
    completed: bool


class SessionEnded(BaseModel):
    """A terminal beat triggered. Emitted alongside the terminal `BeatFired`."""
    kind: Literal["session_ended"] = "session_ended"
    beat_id: str


# ───────────────────────── union ─────────────────────────


Event = Annotated[
    Union[
        SendRecorded,
        EmailEmitted,
        EmailRead,
        ThreadHistoryAppended,
        ClassifierObserved,
        ExtractorObserved,
        DirectorChose,
        ChatBodyRendered,
        LiteralEffectApplied,
        ClassifyEffectApplied,
        CanonPromoted,
        BeatFired,
        SessionEnded,
    ],
    Field(discriminator="kind"),
]


# ───────────────────────── fold ─────────────────────────


def fold(state: InboxState, event: Event) -> None:
    """Apply a single event to state. The only state-write entry point in
    the engine. Pure mutation, no I/O.
    """
    match event:
        case SendRecorded():
            state.emails.append(event.email)
            state.replies.append(event.reply)
        case EmailEmitted():
            state.emails.append(event.email)
        case EmailRead():
            for e in state.emails:
                if e.id == event.email_id:
                    e.is_read = True
                    break
        case ThreadHistoryAppended():
            history = state.thread_messages.setdefault(event.thread_id, [])
            history.extend(event.messages)
        case LiteralEffectApplied():
            state.qualities[event.delta.quality] = event.delta.after
        case ClassifyEffectApplied():
            for r in reversed(state.replies):
                if r.id == event.reply_id:
                    r.classifier_results.append(event.axis)
                    break
            if event.delta is not None:
                state.qualities[event.delta.quality] = event.delta.after
        case CanonPromoted():
            state.player_canon[event.key] = event.entry
        case BeatFired():
            if event.completed:
                state.fired_beat_ids.add(event.beat_id)
        case SessionEnded():
            state.ended = True
            state.end_reason = event.beat_id
        case ClassifierObserved() | ExtractorObserved() | DirectorChose() | ChatBodyRendered():
            pass
