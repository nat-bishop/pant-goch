"""Pydantic schemas for Inbox.

Three layers:

- **Authoring** (loaded from `scenario.yaml`, immutable at runtime): QualityDef,
  NPCDossier, Beat, Thread, Scenario.
- **Runtime state** (mutable, owned by the spine): Email, Reply, InboxState.
  The LLM never writes any of this directly — only via beat effects
  (deterministic) or texture scorer outputs (continuous warmth quality).
- **HTTP DTOs**: OpenRequest, SendRequest.

Effects and predicates are tagged-union Pydantic models, not strings. YAML
authoring is slightly more verbose; in exchange, every effect and every
predicate is type-checked at load time and impossible to misspell silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ───────────────────────── effect / predicate DSL ─────────────────────────


class SetQuality(BaseModel):
    """Mutate a numeric quality. `op: set` overwrites; `op: add` increments;
    `op: drift_halfway` moves halfway toward `value` (used for register drift)."""

    kind: Literal["set_quality"] = "set_quality"
    quality: str
    op: Literal["set", "add", "drift_halfway"] = "set"
    value: float


# Only one effect type for now; kept as a discriminated union so adding a
# second effect later doesn't require touching call sites that already
# pattern-match on `kind`.
Effect = Annotated[
    Union[SetQuality],
    Field(discriminator="kind"),
]


# Predicates compose. Authors write trees of these.


class QualityGte(BaseModel):
    kind: Literal["quality_gte"] = "quality_gte"
    quality: str
    value: float


class QualityLt(BaseModel):
    kind: Literal["quality_lt"] = "quality_lt"
    quality: str
    value: float


class PlayerRepliedTo(BaseModel):
    """True iff the player has sent a reply addressed to (in response to) the
    given beat."""

    kind: Literal["player_replied_to"] = "player_replied_to"
    beat: str


class PlayerRepliesToThreadGte(BaseModel):
    """True iff the player has sent at least `value` replies in the given thread."""

    kind: Literal["player_replies_to_thread_gte"] = "player_replies_to_thread_gte"
    thread: str
    value: int = Field(ge=1)


class BeatFired(BaseModel):
    """True iff the named beat has already fired (lives in `state.fired_beat_ids`).

    Use under `Not_` to express mutual exclusion between two beats whose
    other preconditions overlap.
    """

    kind: Literal["beat_fired"] = "beat_fired"
    beat: str


class All(BaseModel):
    """Logical AND."""

    kind: Literal["all"] = "all"
    of: list["Predicate"]


class Any_(BaseModel):
    """Logical OR."""

    kind: Literal["any"] = "any"
    of: list["Predicate"]


class Not_(BaseModel):
    """Logical NOT."""

    kind: Literal["not"] = "not"
    of: "Predicate"


Predicate = Annotated[
    Union[
        QualityGte,
        QualityLt,
        PlayerRepliedTo,
        PlayerRepliesToThreadGte,
        BeatFired,
        All,
        Any_,
        Not_,
    ],
    Field(discriminator="kind"),
]


All.model_rebuild()
Any_.model_rebuild()
Not_.model_rebuild()


# ───────────────────────── authoring schemas ─────────────────────────


class QualityDef(BaseModel):
    """A continuous numeric state. Mutated by beat effects or TEXTURE_AGENT —
    never read by predicates as discrete values (use `quality_gte`/`quality_lt`).
    """

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    type: Literal["int", "float"] = "float"
    min: float | None = None
    max: float | None = None
    default: float | int | None = None
    description: str = ""


class VoiceBand(BaseModel):
    """One register-conditioned voice variant. EMAIL_AGENT picks a band per
    render based on the NPC's `register` quality.
    """

    name: str
    register_min: float = -3.0
    register_max: float = 3.0
    selected_when: str
    vocabulary_examples: list[str] = Field(default_factory=list)
    sentence_register: str
    tone_examples: list[str] = Field(default_factory=list)


class NPCDossier(BaseModel):
    """Static authoring artifact. Never mutated at runtime. Injected into
    EMAIL_AGENT or TEXTURE_AGENT prompts as needed."""

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    relationship_to_owner: str = ""
    base_voice_register: str = ""
    base_voice_forbidden: list[str] = Field(default_factory=list)
    will_say: list[str] = Field(default_factory=list)
    will_not_say: list[str] = Field(default_factory=list)
    voice_bands: list[VoiceBand] = Field(default_factory=list)
    register_quality: str | None = None  # which quality drives band selection
    # Min/max delay (in calibration units) before this NPC sends their next reply
    # after the player writes to them. Sampled at the midpoint for determinism.
    # The delay must exceed SEND_CALIBRATION_DELTA so a single /send doesn't
    # immediately satisfy its own queued reply — see schedule_npc_reply.
    reply_delay_calibration: tuple[float, float] = (2.0, 5.0)


class BeatSpec(BaseModel):
    """Inputs to EMAIL_AGENT for a `rendered` beat."""

    intent: str
    must_include_facts: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    length_words: tuple[int, int] | None = None  # min, max


# Beat is a tagged union on `type`. Each variant carries the fields that make
# sense for it. Common to all: id, thread, preconditions, in_boot_inbox, effects.

class _BeatBase(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    thread: str
    preconditions: list[Predicate] = Field(default_factory=list)
    in_boot_inbox: bool = False
    effects: list[Effect] = Field(default_factory=list)


class AnchoredBeat(_BeatBase):
    """Pre-authored email; ships verbatim. No LLM call."""

    type: Literal["anchored"] = "anchored"
    from_: str = Field(alias="from")
    to: str = "you"
    subject: str
    authored_prose: str
    sent_at: str = ""  # in-fiction timestamp label (e.g. "just now")
    folder: Literal["Inbox", "Sent"] = "Inbox"

    model_config = {"populate_by_name": True}


class RenderedBeat(_BeatBase):
    """LLM-generated email from spec + dossier + transcript + voice band."""

    type: Literal["rendered"] = "rendered"
    npc: str  # which dossier to use
    from_: str = Field(alias="from")
    to: str = "you"
    subject: str
    spec: BeatSpec
    folder: Literal["Inbox"] = "Inbox"

    model_config = {"populate_by_name": True}


Beat = Annotated[
    Union[AnchoredBeat, RenderedBeat],
    Field(discriminator="type"),
]


class Thread(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    npc: str | None = None  # dossier id; None for system threads (e.g. system)


class Scenario(BaseModel):
    """Authored content for one playthrough. Loaded once; immutable.

    Cross-references validated at load: every thread referenced by a beat
    must exist; every dossier id used by a rendered beat must exist; every
    quality/beat referenced by an effect or predicate must exist.

    Lookup dicts (`beat_by_id`, `npc_by_id`, `quality_by_id`, `thread_by_id`)
    are built lazily and cached. Use them anywhere you'd otherwise scan a list.
    """

    model_config = {"ignored_types": (cached_property,)}

    title: str
    premise: str

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

    @cached_property
    def thread_by_npc_id(self) -> dict[str, str]:
        """Map npc id → its thread id (for `schedule_npc_reply`)."""
        return {t.npc: t.id for t in self.threads if t.npc is not None}

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
            if isinstance(beat, RenderedBeat) and beat.npc not in npc_ids:
                raise ValueError(f"beat {beat.id}: unknown npc {beat.npc!r}")
            for eff in beat.effects:
                _check_effect_refs(eff, beat.id, quality_ids)
            for pred in beat.preconditions:
                _check_predicate_refs(
                    pred, f"beat {beat.id}", quality_ids, thread_ids, beat_ids
                )

        for npc in self.npcs:
            if npc.register_quality and npc.register_quality not in quality_ids:
                raise ValueError(
                    f"npc {npc.id}: register_quality {npc.register_quality!r} is not a known quality"
                )

        for thread in self.threads:
            if thread.npc is not None and thread.npc not in npc_ids:
                raise ValueError(f"thread {thread.id}: unknown npc {thread.npc!r}")

        return self


def _check_effect_refs(
    eff: Effect,
    where: str,
    quality_ids: set[str],
) -> None:
    if isinstance(eff, SetQuality) and eff.quality not in quality_ids:
        raise ValueError(f"{where}: SetQuality references unknown quality {eff.quality!r}")


def _check_predicate_refs(
    pred: Predicate,
    where: str,
    quality_ids: set[str],
    thread_ids: set[str],
    beat_ids: set[str],
) -> None:
    if isinstance(pred, (QualityGte, QualityLt)) and pred.quality not in quality_ids:
        raise ValueError(f"{where}: predicate references unknown quality {pred.quality!r}")
    if isinstance(pred, PlayerRepliedTo) and pred.beat not in beat_ids:
        raise ValueError(f"{where}: PlayerRepliedTo references unknown beat {pred.beat!r}")
    if isinstance(pred, PlayerRepliesToThreadGte) and pred.thread not in thread_ids:
        raise ValueError(f"{where}: PlayerRepliesToThreadGte references unknown thread {pred.thread!r}")
    if isinstance(pred, BeatFired) and pred.beat not in beat_ids:
        raise ValueError(f"{where}: BeatFired references unknown beat {pred.beat!r}")
    if isinstance(pred, (All, Any_)):
        for sub in pred.of:
            _check_predicate_refs(sub, where, quality_ids, thread_ids, beat_ids)
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
    """A concrete email instance in the inbox. Produced when a beat fires.

    `body` is populated from `authored_prose` (anchored) or from EMAIL_AGENT
    output (rendered). `thread_id` is set at emission time so neither client
    nor spine has to parse it back out of beat ids.
    """

    id: str
    beat_id: str | None  # None for player sends
    thread_id: str | None  # set at emission so consumers don't parse beat ids
    folder: str
    sender: str
    recipient: str
    subject: str
    body: str
    sent_at_label: str
    is_read: bool = False
    in_reply_to_beat: str | None = None
    voice_band_used: str | None = None


@dataclass
class Reply:
    """One player send. Captured for trajectory analysis."""

    id: str
    thread_id: str | None  # None for cross-thread sends
    beat_responded_to: str | None
    recipient: str
    body: str
    warmth: float
    rationale: str


@dataclass
class PendingBeat:
    """A beat queued to fire when its preconditions become satisfied.

    `runtime_preconditions` are added at scheduling time (e.g. by
    `schedule_npc_reply` to encode a calibration-based reply delay) and
    AND'd with the beat's authored preconditions.
    """

    beat_id: str
    runtime_preconditions: list[Predicate] = field(default_factory=list)


@dataclass
class InboxState:
    """The mutable runtime state. The spine fully owns this; the LLM never
    writes to it directly except through TEXTURE_AGENT outputs that target
    specific quality fields."""

    qualities: dict[str, float] = field(default_factory=dict)
    fired_beat_ids: set[str] = field(default_factory=set)
    pending_beats: list[PendingBeat] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
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
        # Auto-pend every anchored beat that doesn't ship in the boot inbox.
        # Their authored preconditions are evaluated as runtime predicates by
        # `find_due_beats`. Rendered per-NPC reply beats stay out — those are
        # added to pending only by `schedule_npc_reply` on player send.
        for beat in scenario.beats:
            if isinstance(beat, AnchoredBeat) and not beat.in_boot_inbox:
                state.pending_beats.append(PendingBeat(beat_id=beat.id))
        return state

    # The view exposed to the client. Hides internal scheduler state.
    def public_view(self) -> dict:
        return {
            "calibration": self.qualities.get("calibration", 0.0),
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


class TextureScore(BaseModel):
    """TEXTURE_AGENT output."""

    warmth: float = Field(ge=-3.0, le=3.0)
    rationale: str = ""


class EmailRender(BaseModel):
    """EMAIL_AGENT output. Prose only — no state changes proposed."""

    prose: str = Field(min_length=1)
