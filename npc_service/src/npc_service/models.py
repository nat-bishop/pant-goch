"""Pydantic schemas for Inbox.

Two layers:

- **Authoring** (loaded from `scenario.yaml`, immutable at runtime): Quality,
  NPCDossier, Beat, Thread, Scenario, Effect.
- **Runtime state** (mutable, owned by the spine): Email, Reply, InboxState.
  The LLM never writes any of this directly. Quality writes go through
  the spine's effect appliers (`apply_literal_effects` for literal `value:`
  effects fired at beat-fire time, `apply_classify_effect` for
  classifier-driven writes at reply time), all driven by `Effect` entries
  on `BeatSpec.effects` or `Scenario.effects`.

Predicates are tagged-union Pydantic models, not strings. YAML authoring is
slightly more verbose; in exchange, every predicate is type-checked at load
time and impossible to misspell silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai.messages import ModelMessage


# ───────────────────────── classifier authoring ─────────────────────────


class ClassifyDirective(BaseModel):
    """A classifier prompt block authored on an `Effect`. When the player
    replies to a beat that owns this effect (or to any reply, when scoped at
    `Scenario.effects`), the classifier judges the reply against this block
    and the verdict feeds the parent effect's quality write.

    For numeric qualities, `map` is required: the verdict string indexes into
    `map` to produce the float written. For string qualities, `map` is
    forbidden — the verdict string IS the value written.
    """

    question: str = Field(min_length=1)
    allowed_values: list[str] = Field(min_length=2)
    # Only set for numeric quality targets. Keys must equal `allowed_values`
    # when set (validated at the parent Effect's load time, where we know
    # the quality type).
    map: dict[str, float] | None = None


class AxisResult(BaseModel):
    """One classifier verdict's output. Stored on `Reply.classifier_results`
    immediately after the classifier writes, so recent-window predicates can
    read player history off `state.replies` directly rather than maintaining
    a parallel ring buffer.

    `quality` names the target quality the verdict was for (every classify
    effect targets a quality now — there are no verdict-only axes).
    """

    quality: str
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


class QualityEquals(BaseModel):
    """True iff the named quality is set and equals `value`. Works for both
    numeric and string qualities — for string qualities, `value` is the
    expected string verdict (e.g. `bargain_state == "refused"`)."""

    kind: Literal["quality_equals"] = "quality_equals"
    quality: str
    value: str | int | float


class PlayerRepliesToThreadGte(BaseModel):
    """True iff the player has sent at least `value` replies in the given thread."""

    kind: Literal["player_replies_to_thread_gte"] = "player_replies_to_thread_gte"
    thread: str
    value: int = Field(ge=1)


class BeatFired(BaseModel):
    """True iff the named beat has completed (lives in `state.fired_beat_ids`).

    One-shot beats (`Beat.kind == "one_shot"`) land in `fired_beat_ids`
    immediately after rendering. Scene beats (`Beat.kind == "scene"`)
    enter `fired_beat_ids` only when their `spec.exit_when` predicates
    all pass on a fire — until then they re-appear in the eligible list
    each tick.

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


class Or_(BaseModel):
    """Logical OR over a non-empty list of sub-predicates. True iff any
    element is true.

    Predicate-list AND already exists at the precondition / completes_when
    level (every entry must pass), so `And_` is not provided — wrap a list
    instead. `Or_` is needed because list-level OR is not expressible
    otherwise (the calibration scene's exit condition is the canonical
    use: name canonized OR turn cap reached).
    """

    kind: Literal["or"] = "or"
    of: list["Predicate"] = Field(min_length=1)


class PlayerCanonExists(BaseModel):
    """True iff the named key is present in `state.player_canon`. The
    extractor agent populates canon on every player reply; predicates in
    this family are how the spine reads the player's introduced specifics
    back into branching decisions.

    Canon names are dynamic (extractor-determined), so this predicate is
    NOT validated against any scenario-load set in `_check_predicate_refs`.
    """

    kind: Literal["player_canon_exists"] = "player_canon_exists"
    name: str = Field(min_length=1)


Predicate = Annotated[
    Union[
        QualityGte,
        QualityLt,
        QualityEquals,
        PlayerRepliesToThreadGte,
        BeatFired,
        RecentClassifierValue,
        Not_,
        Or_,
        PlayerCanonExists,
    ],
    Field(discriminator="kind"),
]


Not_.model_rebuild()
Or_.model_rebuild()


# ───────────────────────── effects ─────────────────────────


class Effect(BaseModel):
    """One state mutation triggered by a beat's lifecycle. Two value sources,
    mutually exclusive — exactly one of `value` or `classify` must be set:

    - **Literal effect** (`value` set): the literal is written to `quality`
      when the beat fires. For scenario-scope effects this branch is unused
      in practice; scenario-scope effects are typically `classify`-driven.
    - **Classify effect** (`classify` set): the classifier judges the
      player's reply per the directive; the verdict is mapped (numeric
      qualities) or written verbatim (string qualities) to `quality`.

    `op` controls how the value is combined with the existing quality:
    `set` (default; replaces), `add` (accumulates), `max` (floor — write
    only if greater), `min` (cap — write only if lesser).

    `when` is an optional predicate gating the effect. For literal effects
    it's evaluated at fire time; for classify effects it's evaluated at
    reply time before applying the verdict.
    """

    quality: str = Field(min_length=1)
    value: float | int | str | None = None
    classify: ClassifyDirective | None = None
    op: Literal["set", "add", "max", "min"] = "set"
    when: Predicate | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "Effect":
        has_value = self.value is not None
        has_classify = self.classify is not None
        if has_value == has_classify:
            raise ValueError(
                "Effect on quality "
                f"{self.quality!r}: must set exactly one of `value` or "
                f"`classify` (got value={self.value!r}, classify="
                f"{'<set>' if has_classify else 'None'})"
            )
        return self

    @property
    def is_classify(self) -> bool:
        return self.classify is not None


# ───────────────────────── authoring schemas ─────────────────────────


class Quality(BaseModel):
    """A scenario-level state variable. Mutated only via `apply_effects`
    (driven by `Effect`s on beats or at scenario scope).

    Two type families:
    - Numeric (`int` / `float`): supports the full `op` range — set / add /
      max / min — and `min` / `max` clamps. Classify-driven writes require
      a `map` from verdict string to float.
    - String: supports only `op: set` — there's no meaningful `add` for a
      string. Classify-driven writes write the verdict string verbatim.

    `surface_to_chat: true` flags this quality for the chat-agent's
    commitments block — its current value is included on every render so
    the LLM can read the "where the run currently stands" ledger. Replaces
    the old `commitment.*` namespace convention.
    """

    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    type: Literal["int", "float", "string"] = "float"
    min: float | None = None
    max: float | None = None
    default: float | int | str | None = None
    description: str = ""
    # When True, `commitments_snapshot` includes this quality (with current
    # value) in the chat agent's per-render prompt block. Used for
    # ledger-style state the LLM should know about (bargain_state,
    # aria_acted_in_player_name). Default False — trajectory-shaping
    # qualities like creator_affect stay internal.
    surface_to_chat: bool = False
    # Optional value-set restriction for string qualities. When non-empty,
    # writes that produce a value outside this set are rejected at
    # apply-time (logged + dropped). For string qualities the union of
    # every `Effect.classify.allowed_values` writing this quality must be a
    # subset of `allowed_values` — load-time validator catches typos.
    allowed_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _type_constraints(self) -> "Quality":
        if self.type == "string":
            if self.min is not None or self.max is not None:
                raise ValueError(
                    f"quality {self.id}: min/max are numeric-only "
                    f"(remove them on string-typed quality)"
                )
            if self.default is not None and not isinstance(self.default, str):
                raise ValueError(
                    f"quality {self.id}: string-typed quality default must "
                    f"be a string (got {self.default!r})"
                )
        else:
            if self.allowed_values:
                raise ValueError(
                    f"quality {self.id}: allowed_values is string-quality-only"
                )
            if self.default is not None and not isinstance(
                self.default, (int, float)
            ):
                raise ValueError(
                    f"quality {self.id}: numeric quality default must be a "
                    f"number (got {self.default!r})"
                )
        return self


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


class BeatSpec(BaseModel):
    """Inputs to the chat agent for a `plot` beat — author directive layered
    onto the conversation as per-run instructions.

    `pre_authored_prose` short-circuits the chat agent entirely: when set,
    the beat fires the prose verbatim (no LLM call). Use for moments where
    exact phrasing matters — typically the boot welcome. `intent` becomes
    optional in that case (it's only consumed by the chat agent).

    `effects` is the unified state-mutation list. Each entry writes one
    quality, with the value coming from a literal (`value`) or a classifier
    verdict (`classify`). See `Effect` for shape.

    `exit_when`, `voice_register_override`, and `escalation_notes` only
    apply on scene beats (`Beat.kind == "scene"`); the parent `Beat`
    validator policies their combinations.
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

    # State mutations triggered by this beat. Literal (`value`) effects fire
    # when the beat fires; classify (`classify`) effects schedule a
    # classifier batch on the next player reply. Multiple classify effects
    # on the same beat are batched into one classifier LLM call. See
    # Effect for the per-entry shape; see spine.apply_effects for ordering.
    #
    # Scene-collision contract: if a literal effect's `quality` on a scene
    # beat is also written by another beat, that other beat's mutex (or
    # the scene's `exit_when`) must take effect before its tick —
    # otherwise the scene will re-fire on the next tick and clobber the
    # later write back to its own value.
    effects: list[Effect] = Field(default_factory=list)

    # Scene exit predicates (list-AND). Required when `Beat.kind ==
    # "scene"`, forbidden when `kind == "one_shot"`. Evaluated in
    # `_fire_beat` after each scene fire; when every predicate passes the
    # beat enters `fired_beat_ids` so downstream `BeatFired` predicates
    # become true. Until then the scene stays pending and may continue
    # firing each tick.
    exit_when: list[Predicate] = Field(default_factory=list)

    # Scene voice override. When set on a scene beat, the chat agent's
    # `_dossier_block` substitutes this string in for the dossier's
    # `base_voice_register` line during the beat's renders. The forbidden
    # / will-say / will-not-say lists from the dossier still apply.
    voice_register_override: str | None = None

    # Per-scene cross-turn escalation directive. Surfaced via
    # `_plot_directive_block` under an "ESCALATION (across turns)" section
    # so the chat agent can modulate within a scene without an author
    # branching the intent string per turn count.
    escalation_notes: str | None = None

    # When True, the chat agent's output_validator runs the fact-checker
    # agent against this beat's `must_include_facts` + `must_not` and
    # raises ModelRetry on any miss/violation. Default False — every
    # un-flagged beat skips the validator entirely (zero per-render LLM
    # cost). Authors flag this on critical beats (terminals, pivotal
    # forks); leave it off on chatty scenes or pre-authored beats.
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

    `kind` controls multi-fire semantics:
    - `one_shot` — fires exactly once. Enters `fired_beat_ids`
      immediately after rendering. The default for transitions, bridges,
      terminals, and the boot welcome.
    - `scene` — fires across multiple ticks until `spec.exit_when` is
      satisfied. Enters `fired_beat_ids` only on the firing tick that
      satisfies every entry of `exit_when`. Replaces the old `multi_turn`
      flag.

    `ends_session=True` requires `kind="one_shot"` (terminals are always
    one-shots).
    """

    # Beat ids are namespaced as `<thread>.<local>` — the spine relies on
    # this convention (see `thread_id_from_beat_id`). The cross-reference
    # validator enforces that the prefix matches the beat's `thread` field.
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.]*$")
    thread: str
    npc: str  # which dossier to use (sender + voice + relationship blurb)
    spec: BeatSpec
    preconditions: list[Predicate] = Field(default_factory=list)
    # Per-beat fire-mode discriminator. Required, no default — authors
    # must explicitly choose `one_shot` or `scene` (no silent default
    # so a missing field never silently makes a one_shot scene-y or
    # vice versa).
    kind: Literal["scene", "one_shot"]
    # When True, `state.ended` flips to True immediately after this beat's
    # email is emitted, with `state.end_reason = beat.id`. Subsequent /open
    # and /send return HTTP 409.
    ends_session: bool = False

    @model_validator(mode="after")
    def _validate_kind_combinations(self) -> "Beat":
        if self.kind == "scene":
            if not self.spec.exit_when:
                raise ValueError(
                    f"Beat {self.id!r}: kind='scene' requires "
                    "spec.exit_when to be non-empty (the scene must have "
                    "a defined exit condition)."
                )
            if self.spec.pre_authored_prose is not None:
                raise ValueError(
                    f"Beat {self.id!r}: kind='scene' is incompatible with "
                    "spec.pre_authored_prose (scenes are played out by the "
                    "chat agent across turns)."
                )
            if self.ends_session:
                raise ValueError(
                    f"Beat {self.id!r}: ends_session=True requires "
                    "kind='one_shot' (terminals are always one-shots)."
                )
        else:  # one_shot
            if self.spec.exit_when:
                raise ValueError(
                    f"Beat {self.id!r}: kind='one_shot' forbids "
                    "spec.exit_when (one-shots fire exactly once)."
                )
        return self


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

    `effects` is the scenario-scope effects list. Classify effects here run
    on every player reply (gated by their own `when:` predicate). Used for
    always-on trajectory scoring that doesn't belong to any single beat.
    Empty by default — the production scenario currently has no
    scenario-scope effects; see Phase 2 (lively-allen) for the planned hoist.
    """

    model_config = {"ignored_types": (cached_property,)}

    title: str
    premise: str
    # The player's email DOMAIN. The local-part is derived at runtime from
    # `state.player_canon[player_canon_name_key]` once a name has been
    # canonized by the extractor. Before that, `fallback_player_local` is
    # used. Recipient resolution lives in `resolve_player_email` (main.py).
    email_domain: str = Field(min_length=1)
    # Optional explicit override. Set by tests / legacy fixtures; if set, it
    # wins over the domain-derived address when no name has been canonized
    # yet. Kept as a backward-compat hatch — production scenarios should
    # leave this None and rely on email_domain + fallback_player_local.
    player_email: str | None = None
    # Local-part used when the player has not yet been canonized.
    # Tonally on-key for Continuity Program: the system assigns you an
    # identifier if you don't give one. Override per scenario as needed.
    fallback_player_local: str = "candidate"
    # Which canon key holds the player's name for email derivation. The
    # extractor's prompt is responsible for keying the player's name as
    # exactly this string when extracted.
    player_canon_name_key: str = "name"

    qualities: list[Quality]
    npcs: list[NPCDossier]
    threads: list[Thread]
    beats: list[Beat]
    effects: list[Effect] = Field(default_factory=list)

    @cached_property
    def beat_by_id(self) -> dict[str, Beat]:
        return {b.id: b for b in self.beats}

    @cached_property
    def npc_by_id(self) -> dict[str, NPCDossier]:
        return {n.id: n for n in self.npcs}

    @cached_property
    def quality_by_id(self) -> dict[str, Quality]:
        return {q.id: q for q in self.qualities}

    @cached_property
    def quality_defaults(self) -> dict[str, float | str]:
        """Frozen at load — the chat-agent commitments snapshot reads this
        on every render and would otherwise rebuild it per call. For
        numeric qualities, an unset default becomes 0.0; for string
        qualities, an unset default becomes the empty string."""
        out: dict[str, float | str] = {}
        for q in self.qualities:
            if q.default is not None:
                out[q.id] = q.default if q.type == "string" else float(q.default)
            else:
                out[q.id] = "" if q.type == "string" else 0.0
        return out

    @cached_property
    def quality_descriptions(self) -> dict[str, str]:
        return {q.id: q.description for q in self.qualities}

    @cached_property
    def thread_by_id(self) -> dict[str, Thread]:
        return {t.id: t for t in self.threads}

    @cached_property
    def surface_to_chat_quality_ids(self) -> set[str]:
        """The qualities the chat agent's commitments block surfaces. Frozen
        at load so the per-render snapshot doesn't re-scan every quality."""
        return {q.id for q in self.qualities if q.surface_to_chat}

    @field_validator("qualities")
    @classmethod
    def _quality_ids_unique(cls, v: list[Quality]) -> list[Quality]:
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
        quality_by_id = {q.id: q for q in self.qualities}
        quality_ids = set(quality_by_id.keys())
        npc_ids = {n.id for n in self.npcs}
        thread_ids = {t.id for t in self.threads}
        beat_ids = {b.id for b in self.beats}

        # Walk every effect (beat-scope + scenario-scope) and validate its
        # quality target + classify directive shape.
        all_effect_sources: list[tuple[str, list[Effect]]] = [
            (f"beat {beat.id}", beat.spec.effects) for beat in self.beats
        ]
        all_effect_sources.append(("scenario.effects", self.effects))

        for where, effects in all_effect_sources:
            for i, effect in enumerate(effects):
                _validate_effect(
                    effect, f"{where}.effects[{i}]",
                    quality_by_id, thread_ids, beat_ids,
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
                    quality_ids, thread_ids, beat_ids,
                )
            # Within one beat, a quality should not be the target of two
            # classify effects (one classifier call answers each axis once,
            # so two classify effects on the same quality is ambiguous).
            seen_classify_qualities: set[str] = set()
            for i, effect in enumerate(beat.spec.effects):
                if effect.is_classify:
                    if effect.quality in seen_classify_qualities:
                        raise ValueError(
                            f"beat {beat.id}: two classify effects target the "
                            f"same quality {effect.quality!r}"
                        )
                    seen_classify_qualities.add(effect.quality)

        for thread in self.threads:
            if thread.npc is not None and thread.npc not in npc_ids:
                raise ValueError(f"thread {thread.id}: unknown npc {thread.npc!r}")

        return self


def _validate_effect(
    effect: Effect,
    where: str,
    quality_by_id: dict[str, Quality],
    thread_ids: set[str],
    beat_ids: set[str],
) -> None:
    """Validate a single effect's quality target, value/classify shape, and
    `when:` predicate references. Raises ValueError on any inconsistency.
    """
    q = quality_by_id.get(effect.quality)
    if q is None:
        raise ValueError(
            f"{where}: targets unknown quality {effect.quality!r}"
        )

    # Per-type value/op constraints.
    if effect.is_classify:
        directive = effect.classify  # type: ignore[union-attr]
        if q.type == "string":
            if directive.map is not None:
                raise ValueError(
                    f"{where}: classify.map is forbidden when target "
                    f"quality {q.id!r} is type:string (the verdict string "
                    f"IS the value)"
                )
            if q.allowed_values:
                disallowed = [
                    v for v in directive.allowed_values
                    if v not in q.allowed_values
                ]
                if disallowed:
                    raise ValueError(
                        f"{where}: classify.allowed_values {disallowed} not "
                        f"in quality {q.id!r}.allowed_values "
                        f"{q.allowed_values}"
                    )
        else:
            if directive.map is None:
                raise ValueError(
                    f"{where}: classify.map is required when target "
                    f"quality {q.id!r} is type:{q.type} (numeric)"
                )
            if set(directive.map.keys()) != set(directive.allowed_values):
                raise ValueError(
                    f"{where}: classify.map keys "
                    f"{sorted(directive.map.keys())} must equal "
                    f"allowed_values {sorted(directive.allowed_values)}"
                )
    else:
        # Literal effect — type-check the value against the target.
        if q.type == "string":
            if not isinstance(effect.value, str):
                raise ValueError(
                    f"{where}: literal value {effect.value!r} must be a "
                    f"string for type:string quality {q.id!r}"
                )
            if q.allowed_values and effect.value not in q.allowed_values:
                raise ValueError(
                    f"{where}: literal value {effect.value!r} not in "
                    f"quality {q.id!r}.allowed_values {q.allowed_values}"
                )
            if effect.op != "set":
                raise ValueError(
                    f"{where}: op={effect.op!r} is not supported on "
                    f"string-typed quality {q.id!r} (only `set`)"
                )
        else:
            if not isinstance(effect.value, (int, float)) or isinstance(
                effect.value, bool
            ):
                raise ValueError(
                    f"{where}: literal value {effect.value!r} must be a "
                    f"number for type:{q.type} quality {q.id!r}"
                )

    if effect.when is not None:
        _check_predicate_refs(
            effect.when, where, set(quality_by_id.keys()),
            thread_ids, beat_ids,
        )


def _check_predicate_refs(
    pred: Predicate,
    where: str,
    quality_ids: set[str],
    thread_ids: set[str],
    beat_ids: set[str],
) -> None:
    if isinstance(pred, (QualityGte, QualityLt, QualityEquals)) and pred.quality not in quality_ids:
        raise ValueError(f"{where}: predicate references unknown quality {pred.quality!r}")
    if isinstance(pred, PlayerRepliesToThreadGte) and pred.thread not in thread_ids:
        raise ValueError(f"{where}: PlayerRepliesToThreadGte references unknown thread {pred.thread!r}")
    if isinstance(pred, BeatFired) and pred.beat not in beat_ids:
        raise ValueError(f"{where}: BeatFired references unknown beat {pred.beat!r}")
    if isinstance(pred, RecentClassifierValue):
        # RecentClassifierValue.quality is the quality id whose verdicts the
        # window walks. Every classify effect targets a quality (no more
        # verdict-only axes), so this must be a known quality.
        if pred.quality not in quality_ids:
            raise ValueError(
                f"{where}: RecentClassifierValue references unknown quality "
                f"{pred.quality!r}"
            )
    if isinstance(pred, Not_):
        _check_predicate_refs(
            pred.of, where, quality_ids, thread_ids, beat_ids,
        )
    if isinstance(pred, Or_):
        for sub in pred.of:
            _check_predicate_refs(
                sub, where, quality_ids, thread_ids, beat_ids,
            )
    # PlayerCanonExists references no scenario-load set (canon names are
    # dynamic, populated by the extractor at runtime); nothing to validate.


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

    `classifier_results` holds the per-quality classifier output for this
    reply, populated by `apply_effects` (classify branch) after each
    successful write. Recent-window predicates walk `state.replies` in
    reverse and read this field — single source of truth for "what did the
    classifier say on reply N?"
    """

    id: str
    thread_id: str | None  # None for cross-thread sends
    beat_responded_to: str | None
    recipient: str
    body: str
    classifier_results: list[AxisResult] = field(default_factory=list)


class CanonCategory(StrEnum):
    """Categories of player-introduced specifics the extractor can produce.

    `name` and `role` are special: the extractor's prompt instructs it to key
    those entries as exactly those literals so downstream gating
    (`PlayerCanonExists("name")`) and email derivation can rely on them.
    Other categories use descriptive free-form snake_case keys.
    """

    NAME = "name"
    ROLE = "role"
    PERSON = "person"
    PLACE = "place"
    EVENT = "event"
    DATE = "date"
    POSSESSION = "possession"
    CLAIM_ABOUT_SELF = "claim_about_self"


@dataclass
class PlayerCanonEntry:
    """One entry in the per-playthrough memory of player-introduced specifics.

    Written by the extractor agent (only) when a player reply contains a
    concrete claim above the salience threshold. REPLACE semantics on key
    collision: a later mention of the same key overwrites the earlier
    summary (so "actually call me Robert" canonically replaces "Bob").

    `literal_text_quote` is the verbatim substring of the player's reply
    that anchored this entry. Required: the extractor's output is dropped
    if the quote is not a substring of the reply (anti-hallucination guard).

    `salience` is the extractor's confidence that this is a deliberate
    claim worth remembering (0.0–1.0). The promotion threshold is applied
    by `apply_canon_extraction`; entries below threshold never reach state.
    """

    summary: str
    literal_text_quote: str
    salience: float
    category: CanonCategory
    turn_of_origin: int


class CanonCandidate(BaseModel):
    """One extractor-produced canon candidate. Promoted to `state.player_canon`
    by `apply_canon_extraction` if salience ≥ threshold AND `literal_text_quote`
    is a verbatim substring of the player's reply.

    Lives in models (not agents.py) so the debug-panel TickDecision payload
    can reference it without a cyclic import. The extractor agent's
    ExtractorOutput re-imports this type.
    """

    key: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    summary: str = Field(min_length=1)
    literal_text_quote: str = Field(min_length=1)
    salience: float = Field(ge=0.0, le=1.0)
    category: CanonCategory


@dataclass
class PendingBeat:
    """A beat queued to fire when its preconditions become satisfied."""

    beat_id: str


@dataclass
class InboxState:
    """The mutable runtime state. The spine fully owns this; the LLM never
    writes to it directly except through classifier outputs (consumed by
    `apply_effects`) which target a single named quality.

    `qualities` holds the current value for each authored quality. The
    type is `dict[str, float | str]` because string-typed qualities are
    supported (their values are verdict strings).

    `thread_messages` holds the Pydantic-AI-native conversation per NPC
    thread. Anchored emails seed it as `ModelResponse(TextPart)` entries;
    player sends append `ModelRequest(UserPromptPart)`; chat agent runs
    append `result.new_messages()` to keep the conversation continuous.
    System threads (no NPC) don't get a history entry.
    """

    qualities: dict[str, float | str] = field(default_factory=dict)
    fired_beat_ids: set[str] = field(default_factory=set)
    pending_beats: list[PendingBeat] = field(default_factory=list)
    emails: list[Email] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
    thread_messages: dict[str, list[ModelMessage]] = field(default_factory=dict)
    # Per-playthrough memory of player-introduced specifics. Written only
    # by `apply_canon_extraction` (driven by the extractor agent). Surfaced
    # to the chat agent via `_player_canon_block` (top-K window) and to
    # predicates via `PlayerCanonExists`. The full ledger feeds the
    # epilogue digest.
    player_canon: dict[str, PlayerCanonEntry] = field(default_factory=dict)
    ended: bool = False
    end_reason: str | None = None

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "InboxState":
        qualities: dict[str, float | str] = {}
        for q in scenario.qualities:
            if q.default is not None:
                qualities[q.id] = (
                    q.default if q.type == "string" else float(q.default)
                )
            else:
                qualities[q.id] = "" if q.type == "string" else 0.0
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


class PredicateTrace(BaseModel):
    """Per-node pass/fail trace mirroring a Predicate's tree shape.

    Produced by `evaluate_predicate_with_trace` for blocked beats so the
    debug overlay can render the predicate tree with each node colored by
    its evaluation result (which arm of an `Or_` satisfied vs. didn't,
    which leaf failed under a `Not_`).

    `label` carries the per-node `describe_predicate` output (concrete
    values inline, e.g. `engagement>=4.0 got 2.5`) — the frontend renders
    it verbatim instead of reconstructing leaf descriptions from live
    state it doesn't have.

    Children layout matches the source Predicate:
      - `Not_`  → exactly one child (the inverted sub-predicate's trace)
      - `Or_`   → one child per `of[i]`
      - leaves  → empty list
    """

    passed: bool
    label: str
    children: list["PredicateTrace"] = Field(default_factory=list)


class BeatEligibility(BaseModel):
    """One beat's eligibility classification on a single tick. The
    `find_due_beats` reporter returns one of these per "interesting" beat:
    eligible beats, and beats whose authored value gates failed. Beats
    blocked on an unsatisfied positive-form `BeatFired` prereq (i.e.
    downstream of unfired work) are omitted entirely — the author already
    knows they need to fire the upstream first, and listing them every tick
    is noise.

    `predicate_tree` and `predicate_trace` are populated only for blocked
    beats — the structured form of the same predicate that `blocked_by`
    renders as one line. Frontend renders them as an interactive tree.
    """

    beat_id: str
    status: Literal["eligible", "blocked"]
    blocked_by: str | None = None  # describe_predicate output for the first failing pred
    is_scene: bool = False
    predicate_tree: Predicate | None = None
    predicate_trace: PredicateTrace | None = None


class QualityDelta(BaseModel):
    """One quality mutation applied during a tick. `before` is None when the
    quality didn't have a prior value (first write). `source` names the
    origin: `"classify:<quality>"` for classify-driven writes, or
    `"effect:<beat_id>:<quality>"` for literal effects.
    """

    quality: str
    before: float | str | None
    after: float | str
    source: str


class CanonDropRecord(BaseModel):
    """One extractor candidate that did NOT get promoted to player_canon,
    paired with the reason. Surfaced in the debug overlay so authors can
    see what the extractor proposed but the engine rejected — currently
    invisible (only logfire-warned) which makes debugging extractor prompt
    drift painful.

    Reasons:
      - "anchor_fail"     — `literal_text_quote` was not a verbatim substring
                            of the player's reply (anti-hallucination floor)
      - "below_salience"  — anchored, but extractor confidence was below the
                            promotion threshold (default 0.5)
    """

    candidate: CanonCandidate
    reason: Literal["anchor_fail", "below_salience"]


class FactCheckAttempt(BaseModel):
    """One pass through the chat agent's fact-check validator on a beat
    flagged `validate_facts: true`. `flagged` attempts trigger ModelRetry
    so the chat agent re-renders; `passed` attempts let the email through.
    """

    status: Literal["passed", "flagged"]
    missing_facts: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class FactCheckTrace(BaseModel):
    """Per-fire fact-check trace, populated only when the firing beat had
    `validate_facts: true`. Empty `attempts` is impossible — the validator
    always runs at least once when enabled.

    The frontend reads `attempts[-1].status` to decide whether to render a
    pass badge or a refusal indicator, and `len(attempts)` for retry count.
    """

    attempts: list[FactCheckAttempt]


class TickDecision(BaseModel):
    """Structured per-tick decision payload. Built once in `_tick` and used
    three ways: as logfire span attributes (queryable in the Logfire UI),
    as the input to per-tick console logging, and as the SSE event payload
    that drives the in-browser debug overlay.
    """

    in_response_to: str | None
    eligible: list[BeatEligibility]  # eligible + value-blocked beats only
    selection: Literal["none", "sole", "director", "continue"]
    chosen_beat_id: str | None
    chosen_rationale: str | None = None  # only when selection == "director"
    quality_deltas: list[QualityDelta] = Field(default_factory=list)
    fired_beat_id: str | None = None
    scene_turn: int | None = None  # nth fire of a scene beat (1-indexed)
    # Canon writes from the extractor on this tick. `canon_writes` are the
    # promoted candidates (anchored + above salience threshold).
    # `canon_drops` are the candidates the extractor proposed but the
    # engine rejected, with reason. Both empty on ticks where the extractor
    # found no candidates (the common case).
    canon_writes: list[CanonCandidate] = Field(default_factory=list)
    canon_drops: list[CanonDropRecord] = Field(default_factory=list)
    # Snapshot of `state.player_canon` AFTER this tick's writes. Lets the
    # debug overlay show the current ledger without accumulating diffs
    # locally, and recovers correctly after page reload (next /send
    # repopulates). Keys map to a flat dict shape mirroring PlayerCanonEntry
    # plus the key — frontend renders without needing the dataclass.
    player_canon: dict[str, dict] = Field(default_factory=dict)
    # Populated only when the fired beat had `validate_facts: true`.
    fact_check: FactCheckTrace | None = None
    # Per-axis classifier history snapshot, capped per axis. Built from
    # state.replies after this tick's classifier writes — so the most
    # recent verdict appears first in each list. Used by the debug
    # overlay's classifier section to render sparkline / pill sequences.
    classifier_history: dict[str, list[str]] = Field(default_factory=dict)
    # Snapshot of `state.fired_beat_ids` AFTER this tick. Lets the scenario
    # graph color nodes without the frontend having to accumulate firings
    # itself (and lets it recover correctly after a page reload).
    fired_beat_ids: list[str] = Field(default_factory=list)


# `PredicateTrace` self-references via `children`; rebuild so Pydantic resolves
# the forward ref before any instance is constructed. Same pattern as
# `Not_.model_rebuild()` / `Or_.model_rebuild()` above.
PredicateTrace.model_rebuild()
