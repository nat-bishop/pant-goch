"""Pydantic schemas for Inbox.

Two layers:

- **Authoring** (loaded from `scenario.yaml`, immutable at runtime): QualityDef,
  NPCDossier, Beat, Thread, Scenario.
- **Runtime state** (mutable, owned by the spine): Email, Reply, InboxState.
  The LLM never writes any of this directly. There are exactly two code-owned
  state-write sites for qualities: `apply_classification` (driven by the
  classifier agent's per-axis output) and `apply_writes` (driven by a beat's
  authored `spec.writes`). Both live in spine.py.

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
    """Beat-local LLM classifier authored on one axis of a beat's
    `classify_reply_axes` list. When the player replies to that beat's email,
    each axis is judged and the result is recorded.

    Two modes:
    - Quality-writing axis (`quality` set): the mapped score is written to
      the named quality (replace or accumulate per `accumulate` flag).
    - Verdict-only axis (`quality=None`, `axis_name` set): no quality write,
      but the AxisResult still lands on `Reply.classifier_results` so
      `RecentClassifierValue` predicates keyed on the same axis identifier
      can read it. Used when the classifier verdict is consumed by a
      preconditional gate, not a quality-driven one.

    `axis_name` is the identifier `RecentClassifierValue` matches on. If
    omitted, it defaults to `quality`. At least one of the two must be set.

    Multiple axes per beat are batched into a single classifier call.
    Within one beat, axis identifiers (axis_name or quality) must be unique.
    """

    quality: str | None = None
    axis_name: str | None = None
    question: str
    allowed_values: list[str] = Field(min_length=2)
    map: dict[str, float]
    # When True, apply_classification adds the mapped score to the existing
    # quality value (clamped) rather than overwriting it. Used for cumulative
    # signals like aria.loop's engagement score, where every reply moves the
    # quality by the mapped delta. Ignored when `quality is None` (verdict-
    # only axes don't write at all).
    accumulate: bool = False

    @model_validator(mode="after")
    def _map_keys_match_allowed(self) -> "ClassifyReply":
        if set(self.map.keys()) != set(self.allowed_values):
            raise ValueError(
                f"classify_reply axis: map keys {sorted(self.map.keys())} "
                f"must equal allowed_values {sorted(self.allowed_values)}"
            )
        return self

    @model_validator(mode="after")
    def _identifier_required(self) -> "ClassifyReply":
        if self.quality is None and self.axis_name is None:
            raise ValueError(
                "classify_reply axis: at least one of `quality` or "
                "`axis_name` must be set"
            )
        return self

    @property
    def identifier(self) -> str:
        """The key used by AxisResult and RecentClassifierValue to refer to
        this axis. Prefer `axis_name`; fall back to `quality` when omitted."""
        return self.axis_name or self.quality  # type: ignore[return-value]


class AxisResult(BaseModel):
    """One classifier axis's output. Stored on `Reply.classifier_results`
    immediately after the classifier writes (or, for verdict-only axes,
    after the value is accepted), so recent-window predicates can read
    player history off `state.replies` directly rather than maintaining a
    parallel ring buffer.

    `quality` is the axis identifier — populated from
    `ClassifyReply.identifier` (axis_name or quality). The field name is
    historical; it identifies an axis, which may or may not correspond to
    a state quality.
    """

    quality: str  # axis identifier (axis_name or quality)
    value: str
    rationale: str = ""


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

    Fire-once beats land in `fired_beat_ids` immediately after rendering.
    Multi-turn beats (`Beat.multi_turn=True`) never enter `fired_beat_ids`
    via their own state — author a sibling beat that gates on `BeatFired` of
    the loop's eventual successor under `Not_` if you need mutex.

    Use under `Not_` to express mutual exclusion between two beats whose
    other preconditions overlap.
    """

    kind: Literal["beat_fired"] = "beat_fired"
    beat: str


class RecentClassifierValue(BaseModel):
    """True iff, of the last `window` replies whose classifier wrote
    `quality`, at least `match` (or "any" / "all") of them landed on a value
    in `values`.

    Replies without a matching-quality classifier result are skipped — they
    don't fill a window slot. So the window walks "the last N replies that
    were classified on this quality," not "the last N replies overall."

    Use cases (from the Frankenstein adaptation): "the last three replies
    were clinical or horrified" (any-of, match=all); "the most recent reply
    was a refusal" (window=1, match=any).
    """

    kind: Literal["recent_classifier_value"] = "recent_classifier_value"
    quality: str
    values: list[str] = Field(min_length=1)  # ANY-OF match
    window: int = Field(ge=1)
    # "any" → ≥1 match; "all" → every slot in window matches (under-filled
    # window cannot satisfy "all"); int N → ≥N matches.
    match: Literal["any", "all"] | int = "any"

    @model_validator(mode="after")
    def _match_in_range(self) -> "RecentClassifierValue":
        if isinstance(self.match, int) and not isinstance(self.match, bool):
            if not (1 <= self.match <= self.window):
                raise ValueError(
                    f"RecentClassifierValue: match int {self.match} "
                    f"must be in 1..{self.window}"
                )
        return self


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
        RecentClassifierValue,
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
    # When False, the frontend hides the composer for this NPC's thread —
    # the NPC sends but the player cannot reply. Used for institutional
    # senders (e.g. a Compliance Office) that exist as one-way notices.
    # The backend does not enforce this; a /send to a no-reply NPC would
    # fall through silently (the chat-turn fallback only fires when the
    # thread has no covering plot beat, which institutional one-shots
    # naturally don't have either).
    replies_enabled: bool = True


class WritesByAxis(BaseModel):
    """Conditional writes keyed on the most recent verdict of a classifier
    axis. A beat declaring `writes_by_axis` looks up the last `window`
    classified replies, finds the most recent `AxisResult` whose identifier
    matches `axis`, and applies the matching `arms[<verdict>]` map of
    quality writes (SET semantics, clamped to bounds).

    Used for collapsed plot-router beats: one beat replaces what would
    otherwise be N parallel followup beats, each writing a different
    quality pin based on the player's prior verdict. The arm whose key
    matches the verdict wins; non-matching verdicts and missing axis
    results are no-ops (logged but non-fatal).

    Validator (`Scenario._cross_references`) checks: `axis` resolves to a
    declared axis identifier; every per-arm quality id is a known quality;
    every arm key appears in some declared axis's `allowed_values` for the
    same identifier (catches typos like `accpeted`).
    """

    axis: str
    window: int = Field(ge=1, default=1)
    arms: dict[str, dict[str, float]]


class BeatSpec(BaseModel):
    """Inputs to the chat agent for a `plot` beat — author directive layered
    onto the conversation as per-run instructions.

    `pre_authored_prose` short-circuits the chat agent entirely: when set,
    the beat fires the prose verbatim (no LLM call). Use for moments where
    exact phrasing matters — typically the boot welcome. `intent` becomes
    optional in that case (it's only consumed by the chat agent).
    """

    intent: str = ""
    must_include_facts: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)

    # Pre-authored email body. When set, the beat ships verbatim — no LLM
    # call, no chat-agent retries, no fact-checker. The body seeds
    # `thread_messages` as a `ModelResponse(TextPart)` so subsequent chat
    # agent runs see it as a prior assistant turn, exactly as the chat
    # agent's own outputs do.
    pre_authored_prose: str | None = None

    # Committed facts the beat fire stamps onto state with SET semantics
    # (no accumulate). Quality bounds clamp via `_clamp_quality`. Authoring
    # convention: keys begin with `commitment.` so the chat agent's
    # `_commitments_block` instructions decorator surfaces them on every
    # render. Multi-turn-collision contract: if a `writes` key on a
    # `multi_turn` beat is also written by another beat, that other beat's
    # mutex must take effect before its tick — otherwise the multi_turn
    # beat will re-fire on the next tick and clobber the later write back
    # to its own value. (Concrete: multi_turn A writes x=2.0; B later
    # writes x=3.0; on the next tick A re-fires unless A has
    # `Not_(BeatFired B)` and clobbers B back to 2.0.)
    writes: dict[str, float] = Field(default_factory=dict)
    # Conditional writes keyed on a classifier verdict. See WritesByAxis.
    # Applied by `apply_writes_by_axis` in spine.py AFTER the flat `writes`
    # block, so per-arm values win on collision (lets `writes` carry a
    # default that arms override).
    writes_by_axis: WritesByAxis | None = None
    # When True, the chat agent's output_validator runs the fact-checker
    # agent against this beat's `must_include_facts` + `must_not` and
    # raises ModelRetry on any miss/violation. Default False — every
    # un-flagged beat skips the validator entirely (zero per-render LLM
    # cost). Authors flag this on critical beats (terminals, pivotal
    # forks); leave it off on chatty multi-turn loops or pre-authored beats.
    validate_facts: bool = False

    @model_validator(mode="after")
    def _validate_field_combinations(self) -> "BeatSpec":
        if not self.intent.strip() and not self.pre_authored_prose:
            raise ValueError(
                "BeatSpec: at least one of `intent` or `pre_authored_prose` "
                "must be set."
            )
        if self.validate_facts and not (self.must_include_facts or self.must_not):
            raise ValueError(
                "BeatSpec: validate_facts=True requires at least one entry "
                "in must_include_facts or must_not — otherwise the validator "
                "has nothing to check."
            )
        if self.pre_authored_prose and self.validate_facts:
            raise ValueError(
                "BeatSpec: validate_facts has no effect when "
                "`pre_authored_prose` is set (the prose ships verbatim, "
                "no chat-agent output to validate)."
            )
        return self


class Beat(BaseModel):
    """One authored beat. Fires when its preconditions pass.

    Two render modes, selected by `spec`:
    - `spec.pre_authored_prose` set → ships verbatim (no LLM call). Used
      for moments where exact phrasing matters (e.g. the boot welcome).
    - otherwise → runs the chat agent with `spec` injected as per-run
      instructions. The agent's reply IS the plot moment.

    `multi_turn` and `ends_session` are independent flags that change firing
    semantics — see field comments below.
    """

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    thread: str
    npc: str  # which dossier to use (sender + voice + relationship blurb)
    spec: BeatSpec
    preconditions: list[Predicate] = Field(default_factory=list)
    # Beat-local classifier axes. Empty → no classifier runs for replies
    # to this beat. Multiple axes are batched into a single classifier
    # LLM call; each axis writes independently to its own quality. Within
    # one beat, axis quality ids must be unique (and must not collide
    # with any key in this beat's `spec.writes`).
    classify_reply_axes: list[ClassifyReply] = Field(default_factory=list)
    # When True, the beat re-fires on every tick — never enters
    # `fired_beat_ids` based on its own state. Stop firing by adding mutex
    # preconditions (e.g. `Not_(BeatFired aria.complete_warm)`) so terminal
    # beats can take over.
    multi_turn: bool = False
    # When True, `state.ended` flips to True immediately after this beat's
    # email is emitted, with `state.end_reason = beat.id`. Subsequent /open
    # and /send return HTTP 409.
    ends_session: bool = False


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
    def quality_defaults(self) -> dict[str, float]:
        """Frozen at load — the chat-agent commitments snapshot reads this
        on every render and would otherwise rebuild it per call."""
        return {
            q.id: float(q.default) if q.default is not None else 0.0
            for q in self.qualities
        }

    @cached_property
    def quality_descriptions(self) -> dict[str, str]:
        return {q.id: q.description for q in self.qualities}

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

        # Axis identifiers (axis_name or quality) declared anywhere in the
        # scenario — used to validate RecentClassifierValue references that
        # may target a verdict-only axis_name rather than a quality. Also
        # used by `writes_by_axis` validation: each axis identifier maps to
        # the union of `allowed_values` across every declaration of that
        # identifier in the scenario, so arm keys can be checked for typos.
        axis_identifiers: set[str] = set()
        axis_allowed_values: dict[str, set[str]] = {}
        for beat in self.beats:
            for axis in beat.classify_reply_axes:
                axis_identifiers.add(axis.identifier)
                axis_allowed_values.setdefault(axis.identifier, set()).update(
                    axis.allowed_values
                )

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
            if beat.npc not in npc_ids:
                raise ValueError(f"beat {beat.id}: unknown npc {beat.npc!r}")
            for pred in beat.preconditions:
                _check_predicate_refs(
                    pred, f"beat {beat.id}",
                    quality_ids, thread_ids, beat_ids, axis_identifiers,
                )
            local_axis_identifiers: set[str] = set()
            axis_quality_writes: set[str] = set()
            for axis in beat.classify_reply_axes:
                if axis.quality is not None and axis.quality not in quality_ids:
                    raise ValueError(
                        f"beat {beat.id}: classify_reply_axes quality "
                        f"{axis.quality!r} is not a known quality"
                    )
                ident = axis.identifier
                if ident in local_axis_identifiers:
                    raise ValueError(
                        f"beat {beat.id}: classify_reply_axes has two axes "
                        f"with the same identifier {ident!r} (set distinct "
                        f"`axis_name` values to disambiguate)"
                    )
                local_axis_identifiers.add(ident)
                if axis.quality is not None:
                    axis_quality_writes.add(axis.quality)
            for write_q in beat.spec.writes:
                if write_q not in quality_ids:
                    raise ValueError(
                        f"beat {beat.id}: spec.writes targets unknown "
                        f"quality {write_q!r}"
                    )
                if write_q in axis_quality_writes:
                    raise ValueError(
                        f"beat {beat.id}: spec.writes key {write_q!r} "
                        f"collides with a classify_reply_axes axis on "
                        f"the same beat (the multi_turn re-fire would "
                        f"clobber the classifier each tick)"
                    )
            wba = beat.spec.writes_by_axis
            if wba is not None:
                if wba.axis not in axis_identifiers:
                    raise ValueError(
                        f"beat {beat.id}: spec.writes_by_axis.axis "
                        f"{wba.axis!r} is not a declared axis identifier"
                    )
                allowed = axis_allowed_values.get(wba.axis, set())
                for arm_key, arm_writes in wba.arms.items():
                    if allowed and arm_key not in allowed:
                        raise ValueError(
                            f"beat {beat.id}: spec.writes_by_axis arm "
                            f"{arm_key!r} is not an allowed_value of axis "
                            f"{wba.axis!r} (declared values: "
                            f"{sorted(allowed)})"
                        )
                    for arm_q in arm_writes:
                        if arm_q not in quality_ids:
                            raise ValueError(
                                f"beat {beat.id}: spec.writes_by_axis "
                                f"arm {arm_key!r} targets unknown quality "
                                f"{arm_q!r}"
                            )
                        if arm_q in axis_quality_writes:
                            raise ValueError(
                                f"beat {beat.id}: spec.writes_by_axis "
                                f"arm {arm_key!r} writes quality {arm_q!r} "
                                f"that this beat also writes via "
                                f"classify_reply_axes (would clobber the "
                                f"classifier each tick)"
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
    axis_identifiers: set[str],
) -> None:
    if isinstance(pred, (QualityGte, QualityLt)) and pred.quality not in quality_ids:
        raise ValueError(f"{where}: predicate references unknown quality {pred.quality!r}")
    if isinstance(pred, PlayerRepliesToThreadGte) and pred.thread not in thread_ids:
        raise ValueError(f"{where}: PlayerRepliesToThreadGte references unknown thread {pred.thread!r}")
    if isinstance(pred, BeatFired) and pred.beat not in beat_ids:
        raise ValueError(f"{where}: BeatFired references unknown beat {pred.beat!r}")
    if isinstance(pred, RecentClassifierValue):
        # RecentClassifierValue.quality is an axis identifier — it may name
        # a state quality (when the axis writes one) OR a verdict-only
        # axis_name (when the axis is gate-driven, no quality write).
        if pred.quality not in quality_ids and pred.quality not in axis_identifiers:
            raise ValueError(
                f"{where}: RecentClassifierValue references unknown axis "
                f"identifier {pred.quality!r} (must be a quality id or a "
                f"classify_reply_axes axis_name)"
            )
    if isinstance(pred, Not_):
        _check_predicate_refs(
            pred.of, where, quality_ids, thread_ids, beat_ids, axis_identifiers,
        )


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate scenario YAML. Fails loudly on bad data."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario.model_validate(data)


# ───────────────────────── runtime state ─────────────────────────


@dataclass
class Email:
    """A concrete email instance in the inbox.

    Sources: pre-authored beat (`spec.pre_authored_prose`), chat-agent
    render, or player send. `thread_id` is set at emission so neither
    client nor spine has to parse it back out of beat ids.
    """

    id: str
    beat_id: str | None  # None for player sends and chat turns
    thread_id: str | None  # set at emission so consumers don't parse beat ids
    folder: str
    sender: str
    recipient: str
    body: str
    sent_at_label: str
    is_read: bool = False


@dataclass
class Reply:
    """One player send. Captured for trajectory analysis.

    `classifier_results` holds the per-axis classifier output for this
    reply, populated by `apply_classification` after each successful
    write. Recent-window predicates walk `state.replies` in reverse and
    read this field — single source of truth for "what did the classifier
    say on reply N?"
    """

    id: str
    thread_id: str | None  # None for cross-thread sends
    beat_responded_to: str | None
    recipient: str
    body: str
    classifier_results: list[AxisResult] = field(default_factory=list)


@dataclass
class PendingBeat:
    """A beat queued to fire when its preconditions become satisfied."""

    beat_id: str


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
            "unread_counts_by_thread": self._unread_counts_by_thread(),
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

    def _unread_counts_by_thread(self) -> dict[str, int]:
        """Per-thread unread email count for the Inbox folder. Frontend uses
        this to badge each chat row independently — the Inbox-wide count
        from `_unread_counts` collapses across threads."""
        by_thread: dict[str, int] = {}
        for e in self.emails:
            if e.is_read or e.folder != "Inbox" or e.thread_id is None:
                continue
            by_thread[e.thread_id] = by_thread.get(e.thread_id, 0) + 1
        return by_thread


# ───────────────────────── HTTP DTOs ─────────────────────────


class OpenRequest(BaseModel):
    email_id: str = Field(min_length=1)


class SendRequest(BaseModel):
    in_reply_to: str | None = None  # email id, optional
    recipient: str = Field(min_length=1)  # display string for the email "to" field
    # NPC dossier id the client knows it's writing to. Optional for
    # backwards compat — when omitted, the server fuzzy-matches `recipient`
    # against dossier names. Newer clients always pass this directly.
    recipient_npc_id: str | None = None
    body: str  # rejected as 400 if empty/whitespace


# ───────────────────────── tick decision payload ─────────────────────────


class BeatEligibility(BaseModel):
    """One beat's eligibility classification on a single tick. The
    `find_due_beats` reporter returns one of these per "interesting" beat:
    eligible beats, and beats whose authored value gates failed. Beats
    blocked on an unsatisfied positive-form `BeatFired` prereq (i.e.
    downstream of unfired work) are omitted entirely — the author already
    knows they need to fire the upstream first, and listing them every tick
    is noise.
    """

    beat_id: str
    status: Literal["eligible", "blocked"]
    blocked_by: str | None = None  # describe_predicate output for the first failing pred
    multi_turn: bool = False


class QualityDelta(BaseModel):
    """One quality mutation applied during a tick. `before` is None when the
    quality didn't have a prior value (first write). `source` names the
    origin: `"classifier:<axis>"` or `"spec.writes:<beat_id>"`.
    """

    quality: str
    before: float | None
    after: float
    source: str


class TickDecision(BaseModel):
    """Structured per-tick decision payload. Built once in `_tick` and used
    three ways: as logfire span attributes (queryable in the Logfire UI),
    as the input to per-tick console logging, and as the SSE event payload
    that drives the in-browser debug overlay.
    """

    in_response_to: str | None
    eligible: list[BeatEligibility]  # eligible + value-blocked beats only
    selection: Literal["none", "sole", "director"]
    chosen_beat_id: str | None
    chosen_rationale: str | None = None  # only when selection == "director"
    quality_deltas: list[QualityDelta] = Field(default_factory=list)
    fired_beat_id: str | None = None
    multi_turn_turn: int | None = None  # nth fire of a multi-turn beat (1-indexed)


