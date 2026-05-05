"""Pydantic schemas for Inbox.

Two layers:

- **Authoring** (loaded from `scenario.yaml`, immutable at runtime): QualityDef,
  NPCDossier, Beat, Thread, Scenario.
- **Runtime state** (mutable, owned by the spine): Email, Reply, InboxState.
  The LLM never writes any of this directly — only the classifier agent
  (when a beat authors `classify_reply`) writes a single named quality.

Predicates are tagged-union Pydantic models, not strings. YAML authoring is
slightly more verbose; in exchange, every predicate is type-checked at load
time and impossible to misspell silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai.messages import ModelMessage


# ───────────────────────── classifier authoring ─────────────────────────


class ClassifyReply(BaseModel):
    """Beat-local LLM classifier authored on a beat. When the player replies
    to that beat's email, the classifier judges the reply and writes the
    result to a named quality.

    The single LLM-judge → single named quality boundary keeps the classifier
    from acting as a general state actor.
    """

    quality: str
    question: str
    allowed_values: list[str] = Field(min_length=2)
    map: dict[str, float]
    # When True, apply_classification adds the mapped score to the existing
    # quality value (clamped) rather than overwriting it. Used for cumulative
    # signals like aria.loop's engagement score, where every reply moves the
    # quality by the mapped delta.
    accumulate: bool = False

    @model_validator(mode="after")
    def _map_keys_match_allowed(self) -> "ClassifyReply":
        if set(self.map.keys()) != set(self.allowed_values):
            raise ValueError(
                f"classify_reply: map keys {sorted(self.map.keys())} "
                f"must equal allowed_values {sorted(self.allowed_values)}"
            )
        return self


# ───────────────────────── predicate DSL ─────────────────────────


# Predicates compose. Authors write trees of these.


class QualityGte(BaseModel):
    kind: Literal["quality_gte"] = "quality_gte"
    quality: str
    value: float


class QualityLt(BaseModel):
    kind: Literal["quality_lt"] = "quality_lt"
    quality: str
    value: float


class PlayerRepliesToThreadGte(BaseModel):
    """True iff the player has sent at least `value` replies in the given thread."""

    kind: Literal["player_replies_to_thread_gte"] = "player_replies_to_thread_gte"
    thread: str
    value: int = Field(ge=1)


class BeatFired(BaseModel):
    """True iff the named beat has completed (lives in `state.fired_beat_ids`).

    For fire-once beats, "completed" == "rendered once" — they're added to
    `fired_beat_ids` immediately after rendering. For multi-turn beats
    (those with non-empty `completes_when`), "completed" means the gate
    passed: such a beat may render multiple times before completing, and
    only the completing render adds it to `fired_beat_ids`.

    Use under `Not_` to express mutual exclusion between two beats whose
    other preconditions overlap.
    """

    kind: Literal["beat_fired"] = "beat_fired"
    beat: str


class Not_(BaseModel):
    """Logical NOT."""

    kind: Literal["not"] = "not"
    of: "Predicate"


Predicate = Annotated[
    Union[
        QualityGte,
        QualityLt,
        PlayerRepliesToThreadGte,
        BeatFired,
        Not_,
    ],
    Field(discriminator="kind"),
]


Not_.model_rebuild()


# ───────────────────────── authoring schemas ─────────────────────────


class QualityDef(BaseModel):
    """A continuous numeric state. Mutated by the classifier when a beat
    authors `classify_reply` (see `apply_classification`). Predicates read
    qualities through `quality_gte`/`quality_lt`, not as discrete values.
    """

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    type: Literal["int", "float"] = "float"
    min: float | None = None
    max: float | None = None
    default: float | int | None = None
    description: str = ""


class NPCDossier(BaseModel):
    """Static authoring artifact. Never mutated at runtime. Injected into the
    chat agent via deps + `@instructions`."""

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    # Sender address used for chat-turn emails (and recommended as the `from`
    # value for the NPC's authored beats too).
    email_address: str
    relationship_to_owner: str = ""
    base_voice_register: str = ""
    base_voice_forbidden: list[str] = Field(default_factory=list)
    will_say: list[str] = Field(default_factory=list)
    will_not_say: list[str] = Field(default_factory=list)


class BeatSpec(BaseModel):
    """Inputs to the chat agent for a `plot` beat — author directive layered
    onto the conversation as per-run instructions."""

    intent: str
    must_include_facts: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    length_words: tuple[int, int] | None = None  # min, max


# Beat is a tagged union on `type`. Each variant carries the fields that make
# sense for it. Common to all: id, thread, preconditions.

class _BeatBase(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    thread: str
    preconditions: list[Predicate] = Field(default_factory=list)
    # Beat-local classifier. When the player replies to this beat's email,
    # the classifier runs and writes the mapped score to the named quality.
    # None → no classifier runs for replies to this beat.
    classify_reply: ClassifyReply | None = None
    # Multi-turn lifecycle. Empty → fire-once (legacy semantics: completes on
    # first render). Non-empty → beat is "completed" only when all predicates
    # pass; until then, find_due_beats keeps returning it and _fire_beat
    # re-renders. The chat agent's message_history carries prior asks +
    # player replies, so on a re-render the LLM differentiates initial vs
    # rephrase from context (the spec's intent should describe both modes).
    completes_when: list[Predicate] = Field(default_factory=list)


class AnchoredBeat(_BeatBase):
    """Pre-authored email; ships verbatim. No LLM call.

    Recipient is `Scenario.player_email` — anchored beats are NPC→player by
    construction. If a future scenario needs a non-player anchored email,
    add a `to` field back at that time.
    """

    type: Literal["anchored"] = "anchored"
    from_: str = Field(alias="from")
    subject: str
    authored_prose: str
    sent_at: str = ""  # in-fiction timestamp label (e.g. "just now")

    model_config = {"populate_by_name": True}


class PlotBeat(_BeatBase):
    """Authored directive that modulates one chat-agent turn.

    A PlotBeat fires when its preconditions pass; the spine runs the chat
    agent with the beat's `spec` injected as per-run instructions. The
    agent's reply IS the plot moment.

    `multi_turn` and `ends_session` are independent flags that change firing
    semantics — see field comments below.
    """

    type: Literal["plot"] = "plot"
    npc: str  # which dossier to use
    subject: str
    spec: BeatSpec
    # When True, the beat re-fires on every tick — never enters
    # `fired_beat_ids` based on its own state. Stop firing by adding mutex
    # preconditions (e.g. `Not_(BeatFired aria.complete_warm)`) so terminal
    # beats can take over. Cleaner than abusing `completes_when` with a
    # never-True predicate.
    multi_turn: bool = False
    # When True, `state.ended` flips to True immediately after this beat's
    # email is emitted, with `state.end_reason = beat.id`. Subsequent /open
    # and /send return HTTP 409.
    ends_session: bool = False


Beat = Annotated[
    Union[AnchoredBeat, PlotBeat],
    Field(discriminator="type"),
]


class Thread(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    npc: str | None = None  # dossier id; None for system threads (e.g. system)


class Scenario(BaseModel):
    """Authored content for one playthrough. Loaded once; immutable.

    Cross-references validated at load: every thread referenced by a beat
    must exist; every dossier id used by a plot beat must exist; every
    quality/beat referenced by a predicate must exist.

    Lookup dicts (`beat_by_id`, `npc_by_id`, `quality_by_id`, `thread_by_id`)
    are built lazily and cached. Use them anywhere you'd otherwise scan a list.
    """

    model_config = {"ignored_types": (cached_property,)}

    title: str
    premise: str
    # The player's email address. Used as the `recipient` on every NPC→player
    # email — anchored beats and chat-turn renders both source it from here
    # so the address has exactly one home rather than being read back off the
    # boot welcome's `to:` field.
    player_email: str = Field(min_length=1)

    qualities: list[QualityDef]
    npcs: list[NPCDossier]
    threads: list[Thread]
    beats: list[Beat]

    @cached_property
    def beat_by_id(self) -> dict[str, Beat]:
        return {b.id: b for b in self.beats}

    @cached_property
    def npc_by_id(self) -> dict[str, NPCDossier]:
        return {n.id: n for n in self.npcs}

    @cached_property
    def quality_by_id(self) -> dict[str, QualityDef]:
        return {q.id: q for q in self.qualities}

    @cached_property
    def thread_by_id(self) -> dict[str, Thread]:
        return {t.id: t for t in self.threads}

    @field_validator("qualities")
    @classmethod
    def _quality_ids_unique(cls, v: list[QualityDef]) -> list[QualityDef]:
        ids = [q.id for q in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate quality ids in {ids}")
        return v

    @field_validator("npcs")
    @classmethod
    def _npc_ids_unique(cls, v: list[NPCDossier]) -> list[NPCDossier]:
        ids = [n.id for n in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate npc ids in {ids}")
        return v

    @field_validator("threads")
    @classmethod
    def _thread_ids_unique(cls, v: list[Thread]) -> list[Thread]:
        ids = [t.id for t in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate thread ids in {ids}")
        return v

    @field_validator("beats")
    @classmethod
    def _beat_ids_unique(cls, v: list[Beat]) -> list[Beat]:
        ids = [b.id for b in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate beat ids in {ids}")
        return v

    @model_validator(mode="after")
    def _cross_references(self) -> "Scenario":
        quality_ids = {q.id for q in self.qualities}
        npc_ids = {n.id for n in self.npcs}
        thread_ids = {t.id for t in self.threads}
        beat_ids = {b.id for b in self.beats}

        for beat in self.beats:
            if beat.thread not in thread_ids:
                raise ValueError(f"beat {beat.id}: unknown thread {beat.thread!r}")
            # The spine treats the beat-id prefix as the source of truth for
            # thread membership (see `thread_id_from_beat_id`). Fail loudly
            # here so authoring mistakes don't silently break reply threading.
            id_prefix = beat.id.split(".", 1)[0] if "." in beat.id else beat.id
            if id_prefix != beat.thread:
                raise ValueError(
                    f"beat {beat.id}: id prefix {id_prefix!r} must equal thread {beat.thread!r}"
                )
            if isinstance(beat, PlotBeat) and beat.npc not in npc_ids:
                raise ValueError(f"beat {beat.id}: unknown npc {beat.npc!r}")
            for pred in beat.preconditions:
                _check_predicate_refs(
                    pred, f"beat {beat.id}", quality_ids, thread_ids, beat_ids
                )
            for pred in beat.completes_when:
                _check_predicate_refs(
                    pred,
                    f"beat {beat.id} completes_when",
                    quality_ids, thread_ids, beat_ids,
                )
            if beat.classify_reply is not None:
                if beat.classify_reply.quality not in quality_ids:
                    raise ValueError(
                        f"beat {beat.id}: classify_reply.quality "
                        f"{beat.classify_reply.quality!r} is not a known quality"
                    )

        for thread in self.threads:
            if thread.npc is not None and thread.npc not in npc_ids:
                raise ValueError(f"thread {thread.id}: unknown npc {thread.npc!r}")

        return self


def _check_predicate_refs(
    pred: Predicate,
    where: str,
    quality_ids: set[str],
    thread_ids: set[str],
    beat_ids: set[str],
) -> None:
    if isinstance(pred, (QualityGte, QualityLt)) and pred.quality not in quality_ids:
        raise ValueError(f"{where}: predicate references unknown quality {pred.quality!r}")
    if isinstance(pred, PlayerRepliesToThreadGte) and pred.thread not in thread_ids:
        raise ValueError(f"{where}: PlayerRepliesToThreadGte references unknown thread {pred.thread!r}")
    if isinstance(pred, BeatFired) and pred.beat not in beat_ids:
        raise ValueError(f"{where}: BeatFired references unknown beat {pred.beat!r}")
    if isinstance(pred, Not_):
        _check_predicate_refs(pred.of, where, quality_ids, thread_ids, beat_ids)


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate scenario YAML. Fails loudly on bad data."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario.model_validate(data)


# ───────────────────────── runtime state ─────────────────────────


@dataclass
class Email:
    """A concrete email instance in the inbox.

    Sources: anchored beat (`authored_prose`), chat-turn render, or PlotBeat
    render (chat agent with author directive). `thread_id` is set at emission
    so neither client nor spine has to parse it back out of beat ids.
    """

    id: str
    beat_id: str | None  # None for player sends and chat turns
    thread_id: str | None  # set at emission so consumers don't parse beat ids
    folder: str
    sender: str
    recipient: str
    subject: str
    body: str
    sent_at_label: str
    is_read: bool = False
    in_reply_to_beat: str | None = None


@dataclass
class Reply:
    """One player send. Captured for trajectory analysis."""

    id: str
    thread_id: str | None  # None for cross-thread sends
    beat_responded_to: str | None
    recipient: str
    body: str


@dataclass
class PendingBeat:
    """A beat queued to fire when its preconditions become satisfied.

    `runtime_preconditions` are AND'd with the beat's authored preconditions
    at scheduling time. Currently unused at runtime — kept on the schema for
    extensibility and exercised by the spine tests.
    """

    beat_id: str
    runtime_preconditions: list[Predicate] = field(default_factory=list)


@dataclass
class InboxState:
    """The mutable runtime state. The spine fully owns this; the LLM never
    writes to it directly except through classifier outputs (when a beat
    authors `classify_reply`) which target a single named quality.

    `thread_messages` holds the Pydantic-AI-native conversation per NPC
    thread. Anchored emails seed it as `ModelResponse(TextPart)` entries;
    player sends append `ModelRequest(UserPromptPart)`; chat agent runs
    append `result.new_messages()` to keep the conversation continuous.
    System threads (no NPC) don't get a history entry.
    """

    qualities: dict[str, float] = field(default_factory=dict)
    fired_beat_ids: set[str] = field(default_factory=set)
    pending_beats: list[PendingBeat] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
    thread_messages: dict[str, list[ModelMessage]] = field(default_factory=dict)
    ended: bool = False
    end_reason: str | None = None

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "InboxState":
        qualities: dict[str, float] = {}
        for q in scenario.qualities:
            if q.default is None:
                qualities[q.id] = 0.0
            elif isinstance(q.default, (int, float)):
                qualities[q.id] = float(q.default)
        state = cls(qualities=qualities)
        # Auto-pend every beat; `find_due_beats` evaluates authored
        # preconditions each tick and only fires the eligible ones.
        for beat in scenario.beats:
            state.pending_beats.append(PendingBeat(beat_id=beat.id))
        return state

    # The view exposed to the client. Hides internal scheduler state.
    def public_view(self) -> dict:
        return {
            "ended": self.ended,
            "end_reason": self.end_reason,
            "unread_counts": self._unread_counts(),
        }

    def _unread_counts(self) -> dict[str, int]:
        # Count distinct threads with any unread email, not raw email count —
        # the inbox list collapses to one row per thread, so the badge should
        # match that visual unit.
        by_folder: dict[str, set[str | None]] = {}
        for e in self.emails:
            if not e.is_read:
                by_folder.setdefault(e.folder, set()).add(e.thread_id)
        return {folder: len(threads) for folder, threads in by_folder.items()}


# ───────────────────────── HTTP DTOs ─────────────────────────


class OpenRequest(BaseModel):
    email_id: str = Field(min_length=1)


class SendRequest(BaseModel):
    in_reply_to: str | None = None  # email id, optional
    recipient: str = Field(min_length=1)
    body: str  # rejected as 400 if empty/whitespace


class ClassificationResult(BaseModel):
    """CLASSIFIER_AGENT output. `value` MUST be one of the beat-authored
    `allowed_values` — caller validates and skips the quality write if not."""

    value: str = Field(min_length=1)
    rationale: str = ""
