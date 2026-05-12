"""Handler engine models — predicates, effects, archetypes, events,
missions, options, state.

Two layers:

- **Authoring** (loaded from a mission YAML, immutable at runtime):
  `Mission`, `AgentDossier`, `Archetype`, `Event`. Quality refs are
  opaque dotted strings — `mission.cycles_remaining`, `mission.extraction_stage`,
  `agent[LYNX].cover`. Predicates are tagged-union Pydantic models, not
  free-form strings — every predicate is type-checked at load time.

- **Runtime** (mutable, owned by `cycle.py` via `spine.py`'s pure functions):
  `State`. Holds the flat qualities map plus engine-internal trackers
  (`fired` ledger, per-agent fire ledger, outcome).

The LLM never writes to state. State writes happen only through the three
spine entry points (`apply_archetype_effects`, `apply_forced_event_effects`,
`fire_due_passive_events`) plus `tick_cycle_close`.

Adding a new `Predicate` variant — coordinated edits:
  1. Define the class here, inheriting `_PredicateLeaf` (evaluated directly)
     or `_PredicateComposite` (must implement `children()`).
  2. Add the class to the `Predicate` Union below.
  3. spine.py: extend `evaluate_predicate` (ends in `assert_never`).
  4. tests/test_spine.py: add a unit test.

The single composite walker `iter_predicate_leaves` covers analysis-side
consumers; new leaves work automatically.
"""
from __future__ import annotations

from typing import Annotated, Any, Iterable, Iterator, Literal, Union, assert_never

from pydantic import BaseModel, Field, model_validator


# ───────────────────────── predicate DSL ─────────────────────────


class _PredicateLeaf(BaseModel):
    """Leaf — evaluated against state directly. No sub-predicates."""


class _PredicateComposite(BaseModel):
    """Composite — must implement `children()` yielding sub-predicates."""

    def children(self) -> Iterable["Predicate"]:
        raise NotImplementedError(
            f"{type(self).__name__}: composite predicate must implement children()"
        )


class QualityGte(_PredicateLeaf):
    kind: Literal["quality_gte"] = "quality_gte"
    quality: str
    value: float


class QualityGt(_PredicateLeaf):
    kind: Literal["quality_gt"] = "quality_gt"
    quality: str
    value: float


class QualityLte(_PredicateLeaf):
    kind: Literal["quality_lte"] = "quality_lte"
    quality: str
    value: float


class QualityLt(_PredicateLeaf):
    kind: Literal["quality_lt"] = "quality_lt"
    quality: str
    value: float


class QualityEquals(_PredicateLeaf):
    """True iff the named quality is set and equals `value`. Numeric or string."""

    kind: Literal["quality_equals"] = "quality_equals"
    quality: str
    value: str | int | float


class EventFired(_PredicateLeaf):
    """True iff the named event is in `state.fired`. Used under `Not_`
    to express mutual exclusion."""

    kind: Literal["event_fired"] = "event_fired"
    event: str


class Not_(_PredicateComposite):
    kind: Literal["not"] = "not"
    of: "Predicate"

    def children(self) -> Iterable["Predicate"]:
        yield self.of


class And_(_PredicateComposite):
    """Conjunction over a non-empty list. Top-level `when:` lists already
    AND their elements, but inside `Or_`/`Not_` you need an explicit And_."""

    kind: Literal["and"] = "and"
    of: list["Predicate"] = Field(min_length=1)

    def children(self) -> Iterable["Predicate"]:
        return iter(self.of)


class Or_(_PredicateComposite):
    kind: Literal["or"] = "or"
    of: list["Predicate"] = Field(min_length=1)

    def children(self) -> Iterable["Predicate"]:
        return iter(self.of)


Predicate = Annotated[
    Union[
        QualityGte,
        QualityGt,
        QualityLte,
        QualityLt,
        QualityEquals,
        EventFired,
        Not_,
        And_,
        Or_,
    ],
    Field(discriminator="kind"),
]

Not_.model_rebuild()
And_.model_rebuild()
Or_.model_rebuild()


def iter_predicate_leaves(pred: "Predicate") -> Iterator["Predicate"]:
    """Yield every leaf predicate inside `pred`, walking composites recursively.
    Single source of truth for "walk the predicate tree."
    """
    if isinstance(pred, _PredicateComposite):
        for child in pred.children():
            yield from iter_predicate_leaves(child)
    else:
        yield pred


# ───────────────────────── effects ─────────────────────────


class Effect(BaseModel):
    """One literal state mutation. `op` controls how `value` combines with the
    existing quality. `when` optionally gates the effect on a predicate; when
    set, the effect only fires if `when` holds at apply-time.

    For string qualities only `op: set` is meaningful — there is no add/max/min.
    """

    quality: str = Field(min_length=1)
    value: float | int | str | bool
    op: Literal["set", "add", "max", "min"] = "set"
    when: Predicate | None = None


# ───────────────────────── authoring schemas ─────────────────────────


class AgentDossier(BaseModel):
    """An operative deployed on the mission. The codename is the
    canonical id used in quality refs (`agent[LYNX].cover`)."""

    id: str = Field(min_length=1)  # canonical codename, e.g. "LYNX"
    codename: str = Field(min_length=1)  # display form, e.g. "Lynx"
    file_number: str = Field(default="")  # e.g. "19-X"
    voice_register: str = Field(default="")  # short character pitch
    voice_details: str = Field(default="")  # longer character notes for renderer
    # In-fiction ground-level posture for this agent at mission start. Surfaced
    # to every agent's system prompt so the SoP/renderer have specific anchors
    # (the city, what the agent is holding, the threat pressing on them) rather
    # than only the static opening_prose. Mid-mission drift is carried by
    # recent dispatches, not by re-authoring this field.
    situation: str = Field(default="")
    starting_cover: int = Field(ge=0, le=10)


class Quality(BaseModel):
    """One authored quality — initial value plus the in-fiction descriptor
    used by the narrative state block. The single per-quality authoring unit;
    every entry in `Mission.qualities` is a `Quality`.

    Use `buckets` for numeric qualities, `values` for discrete enums. Exactly
    one MUST be non-empty — that requirement is what eliminates the older
    "missing descriptor" footgun (initial value and descriptor are co-located,
    so an author can't write one without the other).

    Bucket lookup walks the list in authored order; the first bucket whose
    `threshold` >= the actual value wins. Author bottom-up:
        buckets:
          - [0, "blown"]
          - [2, "compromised"]   # 1..2
          - [5, "degraded"]      # 3..5
          - [8, "intact"]        # 6..8
          - [10, "solid"]        # 9..10

    `values` is keyed by the literal value coerced to its string form
    (`"0": "papers not yet surfaced"`).

    `label` (optional) prefixes every descriptor — e.g. label="extraction"
    yields "extraction: papers not yet surfaced".

    `initial` is omitted only for universal qualities the engine injects
    (`agent[*].cover`, `mission.cycles_remaining`); a `qualities:` entry for
    those exists purely to override the built-in display buckets.
    """

    initial: float | int | str | None = None
    buckets: list[tuple[float, str]] = Field(default_factory=list)
    values: dict[str, str] = Field(default_factory=dict)
    label: str = ""

    @model_validator(mode="after")
    def _has_descriptor(self) -> "Quality":
        if not self.buckets and not self.values:
            raise ValueError(
                "Quality: must set one of `buckets` or `values` (got neither). "
                "Every quality must carry its narrative descriptor."
            )
        if self.buckets and self.values:
            raise ValueError(
                "Quality: set EITHER `buckets` OR `values`, not both"
            )
        return self

    def describe(self, value: float | int | str | bool) -> str | None:
        """Look up the descriptor for `value`. Returns None if no bucket/value
        matches (numeric value above all bucket thresholds, or string value
        absent from the values map). Callers fall back to built-ins or skip.
        """
        body: str | None = None
        if self.values:
            body = self.values.get(str(value))
        elif self.buckets and isinstance(value, (int, float)) and not isinstance(value, bool):
            for threshold, descriptor in self.buckets:
                if value <= threshold:
                    body = descriptor
                    break
        if body is None:
            return None
        return f"{self.label}: {body}" if self.label else body


class DramaticQuestion(BaseModel):
    """Author-written framing for a cycle: when the predicate matches, the SoP
    leads with this question instead of summarising state. Selection walks the
    mission's list in authored order and picks the first match — author from
    most-specific to least-specific.
    """

    when: Predicate
    question: str = Field(min_length=1)


class Archetype(BaseModel):
    """Player-driven action template. Owns a precondition (`when`), a list of
    effects (deterministic state writes), and an optional `params` declaration
    that lets the drafter expand the archetype over the available agents.

    `params: ["agent"]` means: the drafter can choose any agent with cover > 0
    as the actor. The chosen agent is bound to `$agent` in `when` and `effects`.
    """

    id: str = Field(min_length=1)
    when: Predicate
    effects: list[Effect] = Field(default_factory=list)
    params: list[Literal["agent"]] = Field(default_factory=list)
    # Short descriptor of the archetype's intent. Surfaced to the drafter so
    # it can write prose that matches the mechanic. Not displayed to the player.
    intent: str = ""


class Event(BaseModel):
    """Predicate-gated mission event. One authored concept, three runtime shapes
    selected by flag:

    - **Forced** (`force_into_drafter=True`): the engine adds the event to the
      drafter's forced-option requirement; effects fire when the player picks
      the satisfying option. Carries the LLM-facing prose constraints
      (`intent` / `must_communicate` / `must_not` / `tone`).
    - **Passive** (`force_into_drafter=False`): the engine fires the event
      after archetype effects apply when `when` matches. Three sub-shapes:
        * `terminal=True` — sets `state.outcome` and ends the mission.
        * `only_when_no_progress_this_cycle=True` — passive pressure that
          fires only if the authorized option didn't progress the mission.
        * Reaction — fires unconditionally when predicate matches; lifts
          flags ("agent lost") in response to state changes.
    - **Per-agent** (`for_each_agent=True`, passive only): expands `$agent`
      over the mission's agents — fires one instance per matching agent.

    `fires_at_most` controls re-firing across cycles. Mutual-exclusion
    invariants on the flag combinations are enforced by `_consistency`.
    """

    id: str = Field(min_length=1)
    when: Predicate
    effects: list[Effect] = Field(default_factory=list)
    fires_at_most: Literal["unlimited", "once_per_mission", "once_per_agent"] = "once_per_mission"

    # Forced-event (drafter-surfaced) fields. Ignored when force_into_drafter=False.
    force_into_drafter: bool = False
    intent: str = ""
    must_communicate: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    tone: str = ""

    # Passive-event (post-authorization) fields. Ignored when force_into_drafter=True.
    terminal: bool = False
    outcome_key: str | None = None  # for terminal events: sets state.outcome
    only_when_no_progress_this_cycle: bool = False
    for_each_agent: bool = False
    # Authoring-order priority; lower fires first when multiple events are due
    # in the same passive-fire pass. Tie-breaker is authored order.
    priority: int = 0

    @model_validator(mode="after")
    def _consistency(self) -> "Event":
        if self.terminal and not self.outcome_key:
            raise ValueError(
                f"Event {self.id!r}: terminal=True requires outcome_key"
            )
        if self.for_each_agent and self.fires_at_most == "once_per_mission":
            # `once_per_mission` doesn't make sense with for_each_agent —
            # the natural pairing is `once_per_agent`.
            raise ValueError(
                f"Event {self.id!r}: for_each_agent=True requires "
                f"fires_at_most='once_per_agent' or 'unlimited'"
            )
        if self.fires_at_most == "once_per_agent" and not self.for_each_agent:
            # `once_per_agent` is only meaningful when the event has an agent
            # dimension to gate on — i.e. when `for_each_agent=True`.
            raise ValueError(
                f"Event {self.id!r}: fires_at_most='once_per_agent' requires "
                f"for_each_agent=True"
            )
        if self.force_into_drafter:
            # Forced events fire pre-effects via drafter selection; the passive
            # post-effects flags have no engine semantics in that path. The
            # fires_at_most restriction also keeps the live spine mutation in
            # sync with the fold-replay path (both append `event.id` to
            # `state.fired` exactly once per authored event).
            for attr in (
                "terminal",
                "for_each_agent",
                "only_when_no_progress_this_cycle",
            ):
                if getattr(self, attr):
                    raise ValueError(
                        f"Event {self.id!r}: force_into_drafter cannot combine with {attr}"
                    )
            if self.fires_at_most != "once_per_mission":
                raise ValueError(
                    f"Event {self.id!r}: force_into_drafter=True requires "
                    f"fires_at_most='once_per_mission' (forced events are one-shot)"
                )
        return self


class Mission(BaseModel):
    """Authored mission content. Loaded once from YAML at process start;
    immutable thereafter."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)  # display name
    codename: str = ""  # e.g. "LYNX-EXFIL"
    opening_prose: str = ""  # shown on mission start
    # World frame shown to the SoP/renderer system prompts as in-fiction
    # grounding (the city, the sweep, the threat). Optional; falls back to
    # opening_prose alone if unset.
    setting: str = ""
    max_cycles: int = Field(gt=0)
    agents: list[AgentDossier] = Field(min_length=1)
    # Quality authoring. Each entry carries the initial value AND the narrative
    # descriptor in one block — co-locating them eliminates the older "missing
    # descriptor" footgun. Universal qualities (agent[*].cover,
    # mission.cycles_remaining) appear here only when overriding the engine's
    # built-in display; their initial values come from `max_cycles` and
    # `agents[*].starting_cover` and are injected by `make_initial_state`.
    qualities: dict[str, Quality] = Field(default_factory=dict)
    # Per-mission persistent prose constraints, injected into all three agent
    # system prompts. Use for facts the renderer/SoP must honor every cycle
    # (e.g. "LYNX must not be photographed again", "the Vienna network is
    # offline"). Authored, no LLM in the loop.
    standing_orders: list[str] = Field(default_factory=list)
    # Per-state-shape SoP framing. Engine selects the first matching question
    # each cycle; SoP leads with it instead of summarising state. Author from
    # most-specific predicate to most-general.
    dramatic_questions: list[DramaticQuestion] = Field(default_factory=list)
    archetypes: list[Archetype] = Field(min_length=1)
    events: list[Event] = Field(min_length=1)
    # Map of outcome_key → human prose summary for the outcome card.
    outcomes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _id_uniqueness(self) -> "Mission":
        for collection, label in (
            (self.agents, "agent"),
            (self.archetypes, "archetype"),
            (self.events, "event"),
        ):
            seen: set[str] = set()
            for item in collection:
                if item.id in seen:
                    raise ValueError(f"duplicate {label} id: {item.id!r}")
                seen.add(item.id)
        # Outcome keys referenced by terminal events must be in `outcomes`.
        for event in self.events:
            if event.terminal and event.outcome_key not in self.outcomes:
                raise ValueError(
                    f"Event {event.id!r}: outcome_key {event.outcome_key!r} "
                    f"is not in mission.outcomes ({sorted(self.outcomes)})"
                )
        return self

    def initial_qualities_for_state(self) -> dict[str, float | int | str]:
        """Return the (name → initial value) mapping the engine should seed
        into State at mission start. Universal qualities (cycles_remaining,
        agent[*].cover) are injected separately by `make_initial_state`."""
        return {
            name: q.initial
            for name, q in self.qualities.items()
            if q.initial is not None
        }

    def archetype(self, id: str) -> Archetype:
        for a in self.archetypes:
            if a.id == id:
                return a
        raise KeyError(f"archetype {id!r} not in mission {self.id!r}")

    def agent(self, id: str) -> AgentDossier:
        for a in self.agents:
            if a.id == id:
                return a
        raise KeyError(f"agent {id!r} not in mission {self.id!r}")


# ───────────────────────── drafter output ─────────────────────────


class Option(BaseModel):
    """One option the drafter offers the player. `archetype`, `agent`, `params`
    drive deterministic resolution; `title` and `prose` are display-only —
    they never influence state.
    """

    archetype: str = Field(min_length=1)
    agent: str | None = None
    params: dict[str, str] = Field(default_factory=dict)
    title: str = Field(min_length=1)
    prose: str = Field(min_length=1)
    # Id of the forced event this option satisfies, if any. The LLM-facing
    # field name keeps the narrative word "moment" (the drafter prompt and
    # system-prompt prefix speak of FORCED MOMENT); the engine reads it as
    # an event id and looks the event up in `mission.events`.
    satisfies_moment: str | None = None


class OptionSet(BaseModel):
    """Drafter structured output: 3–4 options per cycle."""

    options: list[Option] = Field(min_length=3, max_length=4)


# ───────────────────────── runtime state ─────────────────────────


class State(BaseModel):
    """Per-game runtime state. Qualities are namespaced flat strings —
    `mission.cycles_remaining`, `mission.extraction_stage`, `agent[LYNX].cover`.

    Universal qualities the engine guarantees (set up by `make_initial_state`):
    - `mission.cycles_remaining: int` (counts down to 0)
    - `agent[<id>].cover: int` per agent (0 = lost)
    """

    qualities: dict[str, float | int | str | bool] = Field(default_factory=dict)
    # One entry per fire of an event (forced or passive). May contain duplicates
    # for events with `fires_at_most="unlimited"`. The `EventFired(event)`
    # predicate is just a membership check against this list.
    fired: list[str] = Field(default_factory=list)
    # Per-(event, agent) firing ledger for `for_each_agent` events with
    # `fires_at_most='once_per_agent'`. Key is f"{event_id}:{agent_id}".
    fired_per_agent: list[str] = Field(default_factory=list)
    outcome: str | None = None
    # Set True for the duration of one cycle by forced-event effects like
    # kestrel_sacrifice (LYNX.cover_protected). Cleared by `tick_cycle_close`.
    one_cycle_flags: dict[str, bool] = Field(default_factory=dict)

    def get_quality(self, name: str) -> float | int | str | bool | None:
        return self.qualities.get(name)

    def set_quality(self, name: str, value: float | int | str | bool) -> None:
        self.qualities[name] = value


def make_initial_state(mission: Mission) -> State:
    """Construct the starting `State` for a mission. Injects universal
    qualities (`mission.cycles_remaining`, `agent[*].cover`) and authored
    initial values from `mission.qualities`."""
    qualities: dict[str, float | int | str | bool] = {
        "mission.cycles_remaining": mission.max_cycles,
    }
    for agent in mission.agents:
        qualities[f"agent[{agent.id}].cover"] = agent.starting_cover
    qualities.update(mission.initial_qualities_for_state())
    return State(qualities=qualities)


# ───────────────────────── resolution ─────────────────────────


class MissionResolution(BaseModel):
    """Snapshot at terminal — what the outcome card shows."""

    outcome_key: str
    terminal_event_id: str
    cycles_used: int
    cycles_max: int
    agents_lost: list[str]
    agents_alive: list[str]
    summary: str


# ───────────────────────── quality-ref substitution ─────────────────────────


def substitute_placeholders(s: str, placeholders: dict[str, str]) -> str:
    """Replace each `$name` in `s` with the value bound to that name. Empty
    dict is a no-op. Generalises beyond `$agent` so future shape templates can
    bind arbitrary parameters (e.g. `{"pays": "KESTREL", "benefits": "LYNX"}`)."""
    for name, value in placeholders.items():
        s = s.replace(f"${name}", value)
    return s


def substitute_predicate(pred: Predicate, placeholders: dict[str, str]) -> Predicate:
    """Return a new `Predicate` with every `$name` reference resolved per
    `placeholders`. Walks composites; leaves are reconstructed with substituted
    quality refs."""
    if isinstance(pred, QualityGte):
        return QualityGte(quality=substitute_placeholders(pred.quality, placeholders), value=pred.value)
    if isinstance(pred, QualityGt):
        return QualityGt(quality=substitute_placeholders(pred.quality, placeholders), value=pred.value)
    if isinstance(pred, QualityLte):
        return QualityLte(quality=substitute_placeholders(pred.quality, placeholders), value=pred.value)
    if isinstance(pred, QualityLt):
        return QualityLt(quality=substitute_placeholders(pred.quality, placeholders), value=pred.value)
    if isinstance(pred, QualityEquals):
        return QualityEquals(quality=substitute_placeholders(pred.quality, placeholders), value=pred.value)
    if isinstance(pred, EventFired):
        return EventFired(event=substitute_placeholders(pred.event, placeholders))
    if isinstance(pred, Not_):
        return Not_(of=substitute_predicate(pred.of, placeholders))
    if isinstance(pred, And_):
        return And_(of=[substitute_predicate(p, placeholders) for p in pred.of])
    if isinstance(pred, Or_):
        return Or_(of=[substitute_predicate(p, placeholders) for p in pred.of])
    assert_never(pred)


def substitute_effect(eff: Effect, placeholders: dict[str, str]) -> Effect:
    """Return a new `Effect` with placeholders substituted in quality and
    (if set) `when`."""
    return Effect(
        quality=substitute_placeholders(eff.quality, placeholders),
        value=eff.value,
        op=eff.op,
        when=substitute_predicate(eff.when, placeholders) if eff.when else None,
    )
