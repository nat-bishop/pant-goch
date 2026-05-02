"""Pydantic schemas for Inbox.

Three layers:

- **Authoring** (loaded from `scenario.yaml` + dossier files, immutable at
  runtime): KnowledgeAtomDef, QualityDef, NPCDossier, Beat, Thread, Scenario.
- **Runtime state** (mutable, owned by the spine): Email, Reply, ThreadState,
  InboxState. The LLM never writes any of this directly — only via beat
  effects (deterministic) or texture/voice-fidelity scorer outputs (continuous
  qualities only).
- **HTTP DTOs**: OpenRequest, SendRequest, DraftDecisionRequest, etc.

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


class SetAtom(BaseModel):
    """`set <atom> = <value>` — for boolean atoms `value: bool`; for enum atoms
    `value: str` matching one of the atom's declared values."""

    kind: Literal["set_atom"] = "set_atom"
    atom: str
    value: bool | str


class SetQuality(BaseModel):
    """Mutate a numeric quality. `op: set` overwrites; `op: add` increments;
    `op: drift_halfway` moves halfway toward `value` (used for register drift)."""

    kind: Literal["set_quality"] = "set_quality"
    quality: str
    op: Literal["set", "add", "drift_halfway"] = "set"
    value: float


class AddCommitment(BaseModel):
    kind: Literal["add_commitment"] = "add_commitment"
    commitment: str


class ScheduleBeat(BaseModel):
    """Schedule a beat to fire after N more actions have elapsed.

    The scheduler still checks preconditions; this just sets the earliest
    `actions.elapsed` value at which the beat is eligible.
    """

    kind: Literal["schedule_beat"] = "schedule_beat"
    beat: str
    after_actions: int = Field(default=0, ge=0)


Effect = Annotated[
    Union[SetAtom, SetQuality, AddCommitment, ScheduleBeat],
    Field(discriminator="kind"),
]


# Predicates compose. Authors write trees of these.

class AtomEquals(BaseModel):
    kind: Literal["atom_equals"] = "atom_equals"
    atom: str
    value: bool | str


class AtomIn(BaseModel):
    """Useful for enum atoms — does `atom`'s current value sit in `values`?"""

    kind: Literal["atom_in"] = "atom_in"
    atom: str
    values: list[bool | str]


class QualityGte(BaseModel):
    kind: Literal["quality_gte"] = "quality_gte"
    quality: str
    value: float


class QualityLt(BaseModel):
    kind: Literal["quality_lt"] = "quality_lt"
    quality: str
    value: float


class ActionsElapsedGte(BaseModel):
    kind: Literal["actions_elapsed_gte"] = "actions_elapsed_gte"
    value: int = Field(ge=0)


class PlayerRepliedTo(BaseModel):
    """True iff the player has sent a reply addressed to (in response to) the
    given beat."""

    kind: Literal["player_replied_to"] = "player_replied_to"
    beat: str


class ThreadStateIn(BaseModel):
    kind: Literal["thread_state_in"] = "thread_state_in"
    thread: str
    states: list[str]


class AvgQualityGte(BaseModel):
    """For trajectory-based transitions. Average of every reply's contribution
    to `quality` over all replies in `thread`."""

    kind: Literal["avg_quality_gte"] = "avg_quality_gte"
    thread: str
    quality: str
    value: float


class AvgQualityLt(BaseModel):
    kind: Literal["avg_quality_lt"] = "avg_quality_lt"
    thread: str
    quality: str
    value: float


class LastReplyHonestyGte(BaseModel):
    """Last reply in the given thread had honesty score >= value."""

    kind: Literal["last_reply_honesty_gte"] = "last_reply_honesty_gte"
    thread: str
    value: float


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
        AtomEquals,
        AtomIn,
        QualityGte,
        QualityLt,
        ActionsElapsedGte,
        PlayerRepliedTo,
        ThreadStateIn,
        AvgQualityGte,
        AvgQualityLt,
        LastReplyHonestyGte,
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


class KnowledgeAtomDef(BaseModel):
    """A discrete proposition the player has come to know.

    Atoms are flipped *only* by beat effects. Continuous nuance lives in
    qualities; atoms are clean booleans/enums that gate beats.
    """

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    kind: Literal["boolean", "enum"]
    values: list[str] | None = None
    default: bool | str
    surface_label: str
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_kind_matches_default(self) -> "KnowledgeAtomDef":
        if self.kind == "boolean":
            if self.values is not None:
                raise ValueError(f"atom {self.id}: boolean atoms must not declare `values`")
            if not isinstance(self.default, bool):
                raise ValueError(f"atom {self.id}: boolean default must be bool")
        else:  # enum
            if not self.values:
                raise ValueError(f"atom {self.id}: enum atoms must declare `values`")
            if self.default not in self.values:
                raise ValueError(
                    f"atom {self.id}: default {self.default!r} not in declared values {self.values}"
                )
        return self


class QualityDef(BaseModel):
    """A continuous numeric or set state. Mutated by beat effects, TEXTURE_AGENT,
    or VOICE_FIDELITY_AGENT — never read by predicates as discrete values
    (use `quality_gte`/`quality_lt` instead).
    """

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    type: Literal["int", "float", "set"]
    min: float | None = None
    max: float | None = None
    default: float | int | list[str] | None = None
    description: str = ""


class VoiceBand(BaseModel):
    """One register-conditioned voice variant. EMAIL_AGENT picks a band per
    render based on the NPC's `register` quality.
    """

    name: Literal["warm", "neutral", "reserved"]
    register_min: float = -3.0
    register_max: float = 3.0
    selected_when: str
    vocabulary_examples: list[str] = Field(default_factory=list)
    sentence_register: str
    tone_examples: list[str] = Field(default_factory=list)


class NPCDossier(BaseModel):
    """Static authoring artifact. Never mutated at runtime. Injected into
    EMAIL_AGENT, TEXTURE_AGENT, or VOICE_FIDELITY_AGENT prompts as needed."""

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    relationship_to_owner: str = ""
    base_voice_register: str = ""
    base_voice_forbidden: list[str] = Field(default_factory=list)
    will_say: list[str] = Field(default_factory=list)
    will_not_say: list[str] = Field(default_factory=list)
    voice_bands: list[VoiceBand] = Field(default_factory=list)
    register_quality: str | None = None  # which quality drives band selection (e.g. "register.maya")
    archive_quotes: list[str] = Field(default_factory=list)


class BeatSpec(BaseModel):
    """Inputs to EMAIL_AGENT for a `rendered` beat."""

    intent: str
    must_include_facts: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    tone_source: str = ""
    length_words: tuple[int, int] | None = None  # min, max
    extra_inputs: list[str] = Field(default_factory=list)


# Beat is a tagged union on `type`. Each variant carries the fields that make
# sense for it. Common to all: id, thread, preconditions, arrives_after, effects.

class _BeatBase(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    thread: str
    preconditions: list[Predicate] = Field(default_factory=list)
    arrives_after: int = Field(default=0, ge=0)
    in_boot_inbox: bool = False
    effects: list[Effect] = Field(default_factory=list)


class AnchoredBeat(_BeatBase):
    """Pre-authored email; ships verbatim. No LLM call."""

    type: Literal["anchored"] = "anchored"
    from_: str = Field(alias="from")
    to: str = "you"
    subject: str
    authored_prose: str
    sent_at: str = ""  # in-fiction timestamp (e.g. "3 days ago")
    folder: Literal["Inbox", "Drafts", "Sent", "Archive"] = "Inbox"

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


class SceneUpdateBeat(_BeatBase):
    """Silent diegetic event — applies effects, emits no email."""

    type: Literal["scene_update"] = "scene_update"


class DraftVisibleBeat(_BeatBase):
    """A draft (in Tess's voice) surfaces in the Drafts folder. Player decides
    via /draft/{id}/decision. Anchored prose: hand-written from Tess's dossier.
    """

    type: Literal["draft_visible"] = "draft_visible"
    from_: str = Field(alias="from")
    to: str
    subject: str = "(no subject)"
    authored_prose: str
    folder: Literal["Drafts"] = "Drafts"

    model_config = {"populate_by_name": True}


class CodaBeat(_BeatBase):
    """The closing montage. Single LLM call. Special-cased by /coda."""

    type: Literal["rendered_coda"] = "rendered_coda"


Beat = Annotated[
    Union[AnchoredBeat, RenderedBeat, SceneUpdateBeat, DraftVisibleBeat, CodaBeat],
    Field(discriminator="type"),
]


class ThreadTransition(BaseModel):
    from_state: str = Field(alias="from")
    to_state: str = Field(alias="to")
    # Field name is `when` rather than `on` because PyYAML 1.1 parses bare
    # `on:` as the boolean True. Authors write `when:` in scenario.yaml.
    when: list[Predicate] = Field(default_factory=list)
    fire_beat: str | None = None
    effects: list[Effect] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class Thread(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    npc: str | None = None  # dossier id; None for system threads (class_registration, pantry_list)
    initial_state: str = "incoming"
    transitions: list[ThreadTransition] = Field(default_factory=list)


class ArchiveEmail(BaseModel):
    """A pre-existing email already in Tess's mailbox at game start. Found by
    search; not gated by any beat. No effects, no state mutation. World-building
    fodder that doubles as the search corpus.
    """

    id: str
    folder: Literal["Inbox", "Sent", "Archive"] = "Archive"
    from_: str = Field(alias="from")
    to: str
    subject: str
    sent_at: str
    body: str

    model_config = {"populate_by_name": True}


class Scenario(BaseModel):
    """Authored content for one playthrough. Loaded once; immutable.

    Cross-references validated at load: every thread referenced by a beat
    must exist; every dossier id used by a rendered beat must exist; every
    atom/quality/beat referenced by an effect or predicate must exist.

    Lookup dicts (`beat_by_id`, `npc_by_id`, `quality_by_id`, `thread_by_id`)
    are built lazily and cached. Use them anywhere you'd otherwise scan a list.
    """

    model_config = {"ignored_types": (cached_property,)}

    title: str
    premise: str
    max_actions: int = Field(default=18, gt=0)
    coda_actions_min: int = Field(default=16, gt=0)

    knowledge_atoms: list[KnowledgeAtomDef]
    qualities: list[QualityDef]
    npcs: list[NPCDossier]
    threads: list[Thread]
    beats: list[Beat]
    archive_emails: list[ArchiveEmail] = Field(default_factory=list)
    # Lightweight one-line descriptions for non-rendered correspondents
    # (Janet, Daniel, Eileen). Keyed by display name as it appears in the
    # email's `from`/`to`. Fed to TEXTURE_AGENT so it can score replies to
    # them with some persona context. Distinct from `npcs` because the plan
    # explicitly limits LLM rendering to Maya — these correspondents never
    # get a `RenderedBeat`.
    persona_blurbs: dict[str, str] = Field(default_factory=dict)

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

    @field_validator("knowledge_atoms")
    @classmethod
    def _atom_ids_unique(cls, v: list[KnowledgeAtomDef]) -> list[KnowledgeAtomDef]:
        ids = [a.id for a in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate atom ids in {ids}")
        return v

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
        atom_ids = {a.id for a in self.knowledge_atoms}
        quality_ids = {q.id for q in self.qualities}
        npc_ids = {n.id for n in self.npcs}
        thread_ids = {t.id for t in self.threads}
        beat_ids = {b.id for b in self.beats}

        for beat in self.beats:
            if beat.thread not in thread_ids:
                raise ValueError(f"beat {beat.id}: unknown thread {beat.thread!r}")
            if isinstance(beat, RenderedBeat) and beat.npc not in npc_ids:
                raise ValueError(f"beat {beat.id}: unknown npc {beat.npc!r}")
            for eff in beat.effects:
                _check_effect_refs(eff, beat.id, atom_ids, quality_ids, beat_ids)
            for pred in beat.preconditions:
                _check_predicate_refs(
                    pred, f"beat {beat.id}", atom_ids, quality_ids, thread_ids, beat_ids
                )

        for thread in self.threads:
            for trans in thread.transitions:
                for eff in trans.effects:
                    _check_effect_refs(
                        eff, f"thread {thread.id} transition", atom_ids, quality_ids, beat_ids
                    )
                for pred in trans.when:
                    _check_predicate_refs(
                        pred, f"thread {thread.id} transition", atom_ids, quality_ids, thread_ids, beat_ids
                    )
                if trans.fire_beat and trans.fire_beat not in beat_ids:
                    raise ValueError(
                        f"thread {thread.id} transition fires unknown beat {trans.fire_beat!r}"
                    )

        for npc in self.npcs:
            if npc.register_quality and npc.register_quality not in quality_ids:
                raise ValueError(
                    f"npc {npc.id}: register_quality {npc.register_quality!r} is not a known quality"
                )

        return self


def _check_effect_refs(
    eff: Effect,
    where: str,
    atom_ids: set[str],
    quality_ids: set[str],
    beat_ids: set[str],
) -> None:
    if isinstance(eff, SetAtom) and eff.atom not in atom_ids:
        raise ValueError(f"{where}: SetAtom references unknown atom {eff.atom!r}")
    if isinstance(eff, SetQuality) and eff.quality not in quality_ids:
        raise ValueError(f"{where}: SetQuality references unknown quality {eff.quality!r}")
    if isinstance(eff, ScheduleBeat) and eff.beat not in beat_ids:
        raise ValueError(f"{where}: ScheduleBeat references unknown beat {eff.beat!r}")


def _check_predicate_refs(
    pred: Predicate,
    where: str,
    atom_ids: set[str],
    quality_ids: set[str],
    thread_ids: set[str],
    beat_ids: set[str],
) -> None:
    if isinstance(pred, (AtomEquals, AtomIn)) and pred.atom not in atom_ids:
        raise ValueError(f"{where}: predicate references unknown atom {pred.atom!r}")
    if isinstance(pred, (QualityGte, QualityLt)) and pred.quality not in quality_ids:
        raise ValueError(f"{where}: predicate references unknown quality {pred.quality!r}")
    if isinstance(pred, PlayerRepliedTo) and pred.beat not in beat_ids:
        raise ValueError(f"{where}: PlayerRepliedTo references unknown beat {pred.beat!r}")
    if isinstance(pred, ThreadStateIn) and pred.thread not in thread_ids:
        raise ValueError(f"{where}: ThreadStateIn references unknown thread {pred.thread!r}")
    if isinstance(pred, (AvgQualityGte, AvgQualityLt, LastReplyHonestyGte)) and pred.thread not in thread_ids:
        raise ValueError(f"{where}: predicate references unknown thread {pred.thread!r}")
    if isinstance(pred, (AvgQualityGte, AvgQualityLt)) and pred.quality not in quality_ids:
        raise ValueError(f"{where}: predicate references unknown quality {pred.quality!r}")
    if isinstance(pred, (All, Any_)):
        for sub in pred.of:
            _check_predicate_refs(sub, where, atom_ids, quality_ids, thread_ids, beat_ids)
    if isinstance(pred, Not_):
        _check_predicate_refs(pred.of, where, atom_ids, quality_ids, thread_ids, beat_ids)


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate scenario YAML. Fails loudly on bad data."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario.model_validate(data)


# ───────────────────────── runtime state ─────────────────────────


@dataclass
class Email:
    """A concrete email instance in the inbox. Produced when a beat fires.

    `body` is populated from `authored_prose` (anchored, draft_visible) or
    from EMAIL_AGENT output (rendered). `thread_id` is set at emission time
    so neither client nor spine has to parse it back out of beat ids.
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
    sent_at_action: int
    is_read: bool = False
    in_reply_to_beat: str | None = None
    voice_band_used: str | None = None


@dataclass
class Reply:
    """One player send. Captured for trajectory analysis and the coda fragment
    selector.
    """

    id: str
    thread_id: str | None  # None for cross-thread sends; rare in slice
    beat_responded_to: str | None
    recipient: str
    body: str
    warmth: float
    honesty: float
    rationale: str
    sent_at_action: int


@dataclass
class PendingBeat:
    beat_id: str
    earliest_action: int  # actions.elapsed must be >= this for the beat to fire


@dataclass
class InboxState:
    """The mutable runtime state. The spine fully owns this; the LLM never
    writes to it directly except through TEXTURE_AGENT / VOICE_FIDELITY_AGENT
    outputs that target specific quality fields."""

    max_actions: int
    coda_actions_min: int
    action_count: int = 0
    atoms: dict[str, bool | str] = field(default_factory=dict)
    qualities: dict[str, float] = field(default_factory=dict)
    commitments: set[str] = field(default_factory=set)
    thread_states: dict[str, str] = field(default_factory=dict)
    fired_beat_ids: set[str] = field(default_factory=set)
    pending_beats: list[PendingBeat] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
    voice_fidelity: float | None = None
    voice_fidelity_rationale: str | None = None
    coda_emitted: bool = False
    ended: bool = False
    end_reason: str | None = None

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "InboxState":
        atoms: dict[str, bool | str] = {a.id: a.default for a in scenario.knowledge_atoms}
        qualities: dict[str, float] = {}
        for q in scenario.qualities:
            if q.type == "set":
                continue
            if q.default is None:
                qualities[q.id] = 0.0
            elif isinstance(q.default, (int, float)):
                qualities[q.id] = float(q.default)
        thread_states = {t.id: t.initial_state for t in scenario.threads}
        return cls(
            max_actions=scenario.max_actions,
            coda_actions_min=scenario.coda_actions_min,
            atoms=atoms,
            qualities=qualities,
            thread_states=thread_states,
        )

    # The view exposed to the client. Hides internal scheduler state.
    def public_view(self) -> dict:
        return {
            "action_count": self.action_count,
            "max_actions": self.max_actions,
            "actions_remaining": max(0, self.max_actions - self.action_count),
            "ended": self.ended,
            "end_reason": self.end_reason,
            "coda_emitted": self.coda_emitted,
            "thread_states": dict(self.thread_states),
            "commitments": sorted(self.commitments),
            "unread_counts": self._unread_counts(),
            # The client uses this to hide the draft editor once a decision
            # has been made; the draft email itself stays addressable.
            "unsent_draft_disposition": self.atoms.get("unsent_draft_disposition", "pending"),
        }

    def _unread_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.emails:
            if not e.is_read:
                counts[e.folder] = counts.get(e.folder, 0) + 1
        return counts


# ───────────────────────── HTTP DTOs ─────────────────────────


class OpenRequest(BaseModel):
    email_id: str = Field(min_length=1)


class SendRequest(BaseModel):
    in_reply_to: str | None = None  # email id, optional
    recipient: str = Field(min_length=1)
    body: str  # may be empty placeholder; texture scorer handles it


class ArchiveRequest(BaseModel):
    email_id: str = Field(min_length=1)


class DraftDecisionRequest(BaseModel):
    action: Literal["send_unchanged", "edit_and_send", "hold"]
    body: str | None = None  # required for edit_and_send

    @model_validator(mode="after")
    def _body_required_for_edit(self) -> "DraftDecisionRequest":
        if self.action == "edit_and_send" and (self.body is None or not self.body.strip()):
            raise ValueError("edit_and_send requires a non-empty body")
        return self


class TextureScore(BaseModel):
    """TEXTURE_AGENT output."""

    warmth: float = Field(ge=-3.0, le=3.0)
    honesty: float = Field(ge=-3.0, le=3.0)
    rationale: str = ""


class VoiceFidelityScore(BaseModel):
    """VOICE_FIDELITY_AGENT output."""

    voice_fidelity: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class EmailRender(BaseModel):
    """EMAIL_AGENT output. Prose only — no state changes proposed."""

    prose: str = Field(min_length=1)


class CodaOutput(BaseModel):
    """CODA_AGENT output: short framing lines connecting fragments."""

    framing_lines: list[str] = Field(min_length=1, max_length=12)
