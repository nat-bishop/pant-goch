"""Handler engine models — predicates, effects, archetypes, moments, beats,
missions, options, state.

Two layers:

- **Authoring** (loaded from a mission YAML, immutable at runtime):
  `Mission`, `AgentDossier`, `Archetype`, `Moment`, `Beat`. Quality refs are
  opaque dotted strings — `mission.cycles_remaining`, `mission.extraction_stage`,
  `agent[LYNX].cover`. Predicates are tagged-union Pydantic models, not
  free-form strings — every predicate is type-checked at load time.

- **Runtime** (mutable, owned by `cycle.py` via `spine.py`'s pure functions):
  `State`. Holds the flat qualities map plus engine-internal trackers
  (fired_beats, fired_moments, per-agent fire ledger, outcome).

The LLM never writes to state. State writes happen only through the three
spine entry points (`apply_archetype_effects`, `apply_moment_effects`,
`fire_due_beats`) plus `tick_cycle_close`.

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


class BeatFired(_PredicateLeaf):
    """True iff the named beat is in `state.fired_beats`. Used under `Not_`
    to express mutual exclusion."""

    kind: Literal["beat_fired"] = "beat_fired"
    beat: str


class MomentFired(_PredicateLeaf):
    """True iff the named moment is in `state.fired_moments`."""

    kind: Literal["moment_fired"] = "moment_fired"
    moment: str


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
        BeatFired,
        MomentFired,
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
    starting_cover: int = Field(ge=0, le=10)


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


class Moment(BaseModel):
    """Authored decision point. Engine forces the drafter to include an option
    satisfying this moment when its `when` predicate fires. The mechanical
    `effects` are fixed; the LLM writes situated prose around them per
    `intent` / `must_communicate` / `must_not` / `tone`.
    """

    id: str = Field(min_length=1)
    when: Predicate
    fires_at_most: Literal["once_per_mission", "once_per_agent"] = "once_per_mission"
    effects: list[Effect] = Field(default_factory=list)
    intent: str = ""
    must_communicate: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    tone: str = ""


class Beat(BaseModel):
    """Predicate-gated engine event. No player input. Three flavors via flags:

    - **Terminal** (`terminal=True`): sets `state.outcome` and ends the mission.
    - **Passive pressure** (`only_when_no_progress_this_cycle=True`): fires
      only if the authorized option didn't progress the mission this cycle.
    - **Reaction**: fires unconditionally when predicate matches; useful for
      lifting flags ("agent lost") in response to state changes.

    `fires_at_most` controls re-firing across cycles. `for_each_agent: true`
    expands `$agent` placeholders in `when` and `effects` over the mission's
    agents — fires one instance per matching agent.
    """

    id: str = Field(min_length=1)
    when: Predicate
    effects: list[Effect] = Field(default_factory=list)
    terminal: bool = False
    outcome_key: str | None = None  # for terminal beats: sets state.outcome
    fires_at_most: Literal["unlimited", "once_per_mission", "once_per_agent"] = "once_per_mission"
    only_when_no_progress_this_cycle: bool = False
    for_each_agent: bool = False
    # Authoring-order priority; lower fires first when multiple beats are due
    # in the same evaluation pass. Tie-breaker is authored order.
    priority: int = 0

    @model_validator(mode="after")
    def _terminal_has_outcome(self) -> "Beat":
        if self.terminal and not self.outcome_key:
            raise ValueError(
                f"Beat {self.id!r}: terminal=True requires outcome_key"
            )
        if self.for_each_agent and self.fires_at_most == "once_per_mission":
            # `once_per_mission` doesn't make sense with for_each_agent —
            # the natural pairing is `once_per_agent`.
            raise ValueError(
                f"Beat {self.id!r}: for_each_agent=True requires "
                f"fires_at_most='once_per_agent' or 'unlimited'"
            )
        return self


class Mission(BaseModel):
    """Authored mission content. Loaded once from YAML at process start;
    immutable thereafter."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)  # display name
    codename: str = ""  # e.g. "LYNX-EXFIL"
    opening_prose: str = ""  # shown on mission start
    max_cycles: int = Field(gt=0)
    agents: list[AgentDossier] = Field(min_length=1)
    # Mission-authored initial qualities (e.g. extraction_stage=0). Universal
    # qualities (cycles_remaining, agent[*].cover) are auto-injected from
    # `max_cycles` and `agents` — not redundantly declared here.
    initial_qualities: dict[str, float | int | str] = Field(default_factory=dict)
    archetypes: list[Archetype] = Field(min_length=1)
    moments: list[Moment] = Field(default_factory=list)
    beats: list[Beat] = Field(min_length=1)
    # Map of outcome_key → human prose summary for the outcome card.
    outcomes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _id_uniqueness(self) -> "Mission":
        for collection, label in (
            (self.agents, "agent"),
            (self.archetypes, "archetype"),
            (self.moments, "moment"),
            (self.beats, "beat"),
        ):
            seen: set[str] = set()
            for item in collection:
                if item.id in seen:
                    raise ValueError(f"duplicate {label} id: {item.id!r}")
                seen.add(item.id)
        # Outcome keys referenced by terminal beats must be in `outcomes`.
        for beat in self.beats:
            if beat.terminal and beat.outcome_key not in self.outcomes:
                raise ValueError(
                    f"Beat {beat.id!r}: outcome_key {beat.outcome_key!r} "
                    f"is not in mission.outcomes ({sorted(self.outcomes)})"
                )
        return self

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
    # Set True for options that satisfy a forced moment this cycle.
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
    fired_beats: list[str] = Field(default_factory=list)  # one entry per fire
    fired_moments: list[str] = Field(default_factory=list)
    # Per-(beat, agent) firing ledger for `for_each_agent` beats with
    # `fires_at_most='once_per_agent'`. Key is f"{beat_id}:{agent_id}".
    fired_per_agent: list[str] = Field(default_factory=list)
    outcome: str | None = None
    # Set True for the duration of one cycle by moment effects like
    # kestrel_sacrifice (LYNX.cover_protected). Cleared by `tick_cycle_close`.
    one_cycle_flags: dict[str, bool] = Field(default_factory=dict)

    def get_quality(self, name: str) -> float | int | str | bool | None:
        return self.qualities.get(name)

    def set_quality(self, name: str, value: float | int | str | bool) -> None:
        self.qualities[name] = value


def make_initial_state(mission: Mission) -> State:
    """Construct the starting `State` for a mission. Injects universal
    qualities (`mission.cycles_remaining`, `agent[*].cover`) and authored
    initial qualities."""
    qualities: dict[str, float | int | str | bool] = {
        "mission.cycles_remaining": mission.max_cycles,
    }
    for agent in mission.agents:
        qualities[f"agent[{agent.id}].cover"] = agent.starting_cover
    qualities.update(mission.initial_qualities)
    return State(qualities=qualities)


# ───────────────────────── resolution ─────────────────────────


class MissionResolution(BaseModel):
    """Snapshot at terminal — what the outcome card shows."""

    outcome_key: str
    terminal_beat_id: str
    cycles_used: int
    cycles_max: int
    agents_lost: list[str]
    agents_alive: list[str]
    summary: str


# ───────────────────────── quality-ref substitution ─────────────────────────


def substitute_agent(s: str, agent_id: str) -> str:
    """Replace `$agent` in `s` with the concrete agent id. Used by the engine
    when expanding `for_each_agent` beats and `params: [agent]` archetypes."""
    return s.replace("$agent", agent_id)


def substitute_predicate(pred: Predicate, agent_id: str) -> Predicate:
    """Return a new `Predicate` with every `$agent` reference resolved to
    `agent_id`. Walks composites; leaves are reconstructed with substituted
    quality refs.
    """
    if isinstance(pred, QualityGte):
        return QualityGte(quality=substitute_agent(pred.quality, agent_id), value=pred.value)
    if isinstance(pred, QualityGt):
        return QualityGt(quality=substitute_agent(pred.quality, agent_id), value=pred.value)
    if isinstance(pred, QualityLte):
        return QualityLte(quality=substitute_agent(pred.quality, agent_id), value=pred.value)
    if isinstance(pred, QualityLt):
        return QualityLt(quality=substitute_agent(pred.quality, agent_id), value=pred.value)
    if isinstance(pred, QualityEquals):
        return QualityEquals(quality=substitute_agent(pred.quality, agent_id), value=pred.value)
    if isinstance(pred, BeatFired):
        return BeatFired(beat=substitute_agent(pred.beat, agent_id))
    if isinstance(pred, MomentFired):
        return MomentFired(moment=substitute_agent(pred.moment, agent_id))
    if isinstance(pred, Not_):
        return Not_(of=substitute_predicate(pred.of, agent_id))
    if isinstance(pred, And_):
        return And_(of=[substitute_predicate(p, agent_id) for p in pred.of])
    if isinstance(pred, Or_):
        return Or_(of=[substitute_predicate(p, agent_id) for p in pred.of])
    assert_never(pred)


def substitute_effect(eff: Effect, agent_id: str) -> Effect:
    """Return a new `Effect` with `$agent` substituted in quality and (if set) `when`."""
    return Effect(
        quality=substitute_agent(eff.quality, agent_id),
        value=eff.value,
        op=eff.op,
        when=substitute_predicate(eff.when, agent_id) if eff.when else None,
    )
