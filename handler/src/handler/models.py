"""Handler engine models — predicates, effects, archetypes, events,
missions, options, state.

Two layers:

- **Authoring** (loaded from a mission YAML, immutable at runtime):
  `Mission`, `CastMember`, `Archetype`, `Event`. Quality refs are
  opaque dotted strings — `mission.days_remaining`, `mission.extraction_stage`,
  `agent[LYNX].cover`. Predicates are tagged-union Pydantic models, not
  free-form strings — every predicate is type-checked at load time.

- **Runtime** (mutable, owned by `day.py` via `spine.py`'s pure functions):
  `State`. Holds the flat qualities map plus engine-internal trackers
  (`fired` ledger, per-agent fire ledger, outcome).

The LLM never writes to state. State writes happen only through the three
spine entry points (`apply_archetype_effects`, `apply_forced_event_effects`,
`fire_due_passive_events`) plus `tick_day_close`.

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

    Opposed-check resolution lives in `Action.outcomes` (first-match-wins on
    sibling `when:` predicates) — Effects no longer carry branch author-tags.
    """

    quality: str = Field(min_length=1)
    value: float | int | str | bool
    op: Literal["set", "add", "max", "min"] = "set"
    when: Predicate | None = None


# ───────────────────────── authoring schemas ─────────────────────────


class CastMember(BaseModel):
    """A character in the mission. `role="agent"` is an operative deployed in
    the field (canonical id used in quality refs like `agent[LYNX].cover`);
    `role="handler"` is the player's in-fiction identity; `role="boss"` is
    the player's superior; `role="opposition"` is a named antagonist;
    `role="asset"` is an off-stage support character (sleeper, source,
    contact) named in prose but never field-deployed — surfaces in the
    onboarding roster and in renderer prompts but contributes no cover
    quality and is not eligible for archetype expansion.

    Only role="agent" members get a `cover` quality injected by
    `make_initial_state` and appear in archetype expansion / for_each_agent
    — all other roles are renderer-prompt content only.
    """

    id: str = Field(min_length=1)  # canonical codename, e.g. "LYNX"
    codename: str = Field(min_length=1)  # display form, e.g. "Lynx"
    role: Literal["agent", "handler", "boss", "opposition", "asset"] = "agent"
    file_number: str = Field(default="")  # e.g. "19-X"
    voice_register: str = Field(default="")  # short character pitch
    voice_details: str = Field(default="")  # longer character notes for renderer
    # In-fiction ground-level posture for this agent at mission start. Surfaced
    # to every agent's system prompt so the SoP/renderer have specific anchors
    # (the city, what the agent is holding, the threat pressing on them) rather
    # than only the static opening_prose. Mid-mission drift is carried by
    # recent dispatches, not by re-authoring this field.
    situation: str = Field(default="")
    # Required for role="agent" (where the engine reads it to inject the
    # `agent[<id>].cover` quality at mission start). Non-agent members omit it.
    starting_cover: int | None = Field(default=None, ge=0, le=10)

    @model_validator(mode="after")
    def _agents_have_starting_cover(self) -> "CastMember":
        if self.role == "agent" and self.starting_cover is None:
            raise ValueError(
                f"CastMember {self.id!r}: role='agent' requires starting_cover"
            )
        if self.role != "agent" and self.starting_cover is not None:
            raise ValueError(
                f"CastMember {self.id!r}: role={self.role!r} must not set "
                f"starting_cover (only role='agent' members get cover)"
            )
        return self


class Band(BaseModel):
    """One band in a `Quality`'s descriptor ladder: a threshold paired with
    one or more phrasing variants. Single-variant bands are the simple case
    (the band describes a state level); multi-variant bands rotate by
    `day_index` for hidden qualities so the player can't phrase-decode
    the band by reading the same line every day.

    YAML may author a Band as:
      - the explicit form `{threshold: 5, variants: ["v"]}`,
      - a 2-tuple `[5, "v"]` (single-variant sugar), or
      - a 2-tuple `[5, ["v1", "v2"]]` (multi-variant tuple form).
    All forms normalise to the explicit shape in-memory.
    """

    threshold: float
    variants: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_tuple(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)) and len(data) == 2 and not isinstance(data, dict):
            threshold, body = data
            if isinstance(body, str):
                return {"threshold": threshold, "variants": [body]}
            if isinstance(body, list):
                return {"threshold": threshold, "variants": body}
        return data


class Quality(BaseModel):
    """One authored quality — initial value plus the in-fiction descriptor
    used by the narrative state block. The single per-quality authoring unit;
    every entry in `Mission.qualities` is a `Quality`.

    Descriptor shape: `bands` — an ordered ladder. The first band whose
    `threshold` covers the value wins (walk in authored order; numeric
    comparison uses `value <= threshold`). Single-variant bands handle the
    common case; multi-variant bands rotate by day_index for hidden-state
    phrasing variety.

    Ladder layout (bottom-up — first match wins on `value <= threshold`):
        bands:
          - [0, "blown"]
          - [2, "compromised"]
          - [5, "degraded"]
          - [8, "intact"]
          - [10, "solid"]

    Enum-like qualities author each level as its own single-variant band
    keyed on the literal value (`bands: [[0, "..."], [1, "..."]]`).

    `label` (optional) prefixes every descriptor — e.g. label="extraction"
    yields "extraction: papers not yet surfaced".

    `visible` (default True): when False, the quality is hidden from the
    OPERATION POSTURE state block and the RESULT footer (`describe_delta`
    skips invisible-quality entries). It surfaces only through the dedicated
    hidden-state hint block in the drafter / log-writer user prompts.

    `initial` is omitted only for universal qualities the engine injects
    (`agent[*].cover`, `mission.days_remaining`); a `qualities:` entry for
    those exists purely to override the built-in display.
    """

    initial: float | int | str | None = None
    bands: list[Band] = Field(default_factory=list)
    label: str = ""
    visible: bool = True

    @model_validator(mode="after")
    def _has_bands(self) -> "Quality":
        if not self.bands:
            raise ValueError(
                "Quality: `bands` must be non-empty. Every quality needs at "
                "least one band carrying its narrative descriptor."
            )
        return self

    def _match_band(self, value: float | int | str | bool) -> Band | None:
        """Walk `bands` and return the first whose threshold covers value
        (`value <= band.threshold`). Returns None for non-numeric values or
        when no band matches (value above the top threshold)."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        for band in self.bands:
            if value <= band.threshold:
                return band
        return None

    def _label(self, body: str) -> str:
        return f"{self.label}: {body}" if self.label else body

    def describe(self, value: float | int | str | bool) -> str | None:
        """Look up the descriptor for `value` — the first variant of the
        matching band. Returns None when no band covers the value. Callers
        wanting the rotated variant for multi-variant bands should call
        `describe_variant(value, day_index)` instead.
        """
        band = self._match_band(value)
        if band is None:
            return None
        return self._label(band.variants[0])

    def describe_variant(
        self, value: float | int | str | bool, day_index: int
    ) -> str | None:
        """Look up the rotated variant of the matching band, keyed on
        `day_index % len(variants)`. Deterministic so test replay is
        reproducible and consecutive days in the same band surface
        different phrasings. Returns None when no band matches."""
        band = self._match_band(value)
        if band is None:
            return None
        return self._label(band.variants[day_index % len(band.variants)])


class Beat(BaseModel):
    """Author-written observational fragment threaded into a day's prompts
    when its predicate matches. Replaces the older `DramaticQuestion`.

    Selection is highest-priority (lowest `priority` int) authored-order winner
    among `when`-matching beats that haven't already fired this mission.
    On selection, the beat's `id` is appended to `state.fired` so the same
    fragment never surfaces twice — fixing the audit §2b symptom of the same
    framing reappearing every day.

    `fragment` is the prose handed to the LLM. The drafter sees it as day
    framing color; the log writer sees it as an additional lean directive.
    Authors write the fragment as one *example* — the prompts instruct
    translation through the day's specific situation, not verbatim quoting.

    `fires_at_most`:
      - "once_per_mission" (default): consume on fire, never repeats.
      - "unlimited": legacy semantics — re-fires every day the predicate
        matches. Use sparingly; the whole point of the rename is fire-once.
    """

    id: str = Field(min_length=1)
    when: Predicate
    fragment: str = Field(min_length=1)
    priority: int = 0
    fires_at_most: Literal["unlimited", "once_per_mission"] = "once_per_mission"


class Outcome(BaseModel):
    """One resolution branch of an Action. First-match-wins on sibling `when:`
    predicates — sibling outcomes are mutually exclusive (enforced at mission
    load via `_predicates_may_overlap`).

    `leans` are renderer directives threaded into the log writer's LEAN-ON
    block when this outcome resolves. Use them to thread the dramatic shape
    of an opposed-check (clean / partial / failure) without authoring a
    separate `branch_leans` dict.
    """

    when: Predicate
    effects: list[Effect] = Field(default_factory=list)
    leans: list[str] = Field(default_factory=list)


class Action(BaseModel):
    """A player-authorizable action. Subsumes the older Archetype (option
    template) AND forced Event (drafter-surfaced choice with prose constraints)
    into one type.

    Authoring shapes:
      - **Flat** — `effects:` (one deterministic outcome).
      - **Opposed** — `outcomes:` (first-match-wins on sibling `when:`,
        mutually exclusive at load time).
    `effects` and `outcomes` are mutually exclusive — author one or the other.

    Forced shape: `forced: true` + optional prose-constraint fields. A forced
    Action MUST appear as one of the day's options when its `when:` holds
    and its id is not yet in `state.fired`. The drafter's prose writer
    honours `intent` / `must_communicate` / `must_not` / `tone` on the
    option that carries it. The forced Action IS the option — there is no
    separate carrier archetype.

    `for_each_agent: bool` expands `$agent` in `when` / `effects` / `outcomes`
    over agents with cover > 0 (one catalog entry per matching agent).
    """

    id: str = Field(min_length=1)
    when: Predicate
    effects: list[Effect] = Field(default_factory=list)
    outcomes: list[Outcome] = Field(default_factory=list)
    for_each_agent: bool = False
    intent: str = ""
    # Forced-Action fields. Ignored when forced=False.
    forced: bool = False
    must_communicate: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    tone: str = ""

    @model_validator(mode="after")
    def _effects_xor_outcomes(self) -> "Action":
        if self.effects and self.outcomes:
            raise ValueError(
                f"Action {self.id!r}: author `effects` OR `outcomes`, not both"
            )
        return self

    @model_validator(mode="after")
    def _outcomes_mutually_exclusive(self) -> "Action":
        """Sibling outcomes must be mutually exclusive. The check is best-effort
        — handles AND-of-comparisons (the common opposed-check shape). Predicates
        the analyser can't decompose (Or_, Not_, EventFired, string Equals)
        are treated as "possibly overlapping" and raise; authors can refactor
        or pre-validate manually if needed.
        """
        if len(self.outcomes) < 2:
            return self
        for i in range(len(self.outcomes)):
            for j in range(i + 1, len(self.outcomes)):
                if _predicates_may_overlap(self.outcomes[i].when, self.outcomes[j].when):
                    raise ValueError(
                        f"Action {self.id!r}: outcomes {i} and {j} have "
                        f"overlapping `when:` predicates — sibling outcomes must "
                        f"be mutually exclusive (first-match-wins)"
                    )
        return self


class Reaction(BaseModel):
    """Engine-applied consequence. Fires after the player authorizes their
    Action and its effects apply. Replaces the old passive `Event`.

    `terminal=True` + `outcome_key` sets `state.outcome` and ends the mission.
    `only_when_no_progress_this_day=True` gates the reaction on the day's
    Action having NOT progressed the mission (so HOLD has bite).
    `for_each_agent=True` expands `$agent` over agents (one fire per matching).
    """

    id: str = Field(min_length=1)
    when: Predicate
    effects: list[Effect] = Field(default_factory=list)
    fires_at_most: Literal["unlimited", "once_per_mission", "once_per_agent"] = "once_per_mission"
    priority: int = 0
    terminal: bool = False
    outcome_key: str | None = None
    only_when_no_progress_this_day: bool = False
    for_each_agent: bool = False
    # Free-form prose directives the renderer should foreground when this
    # reaction fires. Aggregated by `day._collect_leans` and threaded into
    # the log-writer's user prompt verbatim.
    lean_on: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "Reaction":
        if self.terminal and not self.outcome_key:
            raise ValueError(
                f"Reaction {self.id!r}: terminal=True requires outcome_key"
            )
        if self.for_each_agent and self.fires_at_most == "once_per_mission":
            raise ValueError(
                f"Reaction {self.id!r}: for_each_agent=True requires "
                f"fires_at_most='once_per_agent' or 'unlimited'"
            )
        if self.fires_at_most == "once_per_agent" and not self.for_each_agent:
            raise ValueError(
                f"Reaction {self.id!r}: fires_at_most='once_per_agent' requires "
                f"for_each_agent=True"
            )
        return self


# ───────────────────────── outcome-overlap analysis ─────────────────────────


def _predicates_may_overlap(p1: Predicate, p2: Predicate) -> bool:
    """True iff there's any state where both predicates can hold simultaneously.
    Used by Action's mutual-exclusion validator on sibling Outcomes.

    Strategy: extract per-quality numeric intervals (closed bounds with epsilon
    for strict comparisons). For each shared quality, check intervals are
    disjoint. If any shared quality is disjoint → predicates are mutually
    exclusive (return False). Otherwise → may overlap (return True).

    Conservative on shapes the analyser can't decompose (Or_, Not_, EventFired,
    non-numeric Equals) — those return True so the validator raises, prompting
    the author to flatten or to write each outcome's `when:` as an
    AND-of-comparisons over a numeric quality.
    """
    iv1 = _quality_intervals(p1)
    iv2 = _quality_intervals(p2)
    if iv1 is None or iv2 is None:
        return True
    for q, (lo1, hi1) in iv1.items():
        if q in iv2:
            lo2, hi2 = iv2[q]
            if hi1 < lo2 or hi2 < lo1:
                return False
    return True


def _quality_intervals(pred: Predicate) -> dict[str, tuple[float, float]] | None:
    """Extract per-quality closed numeric intervals from an AND-of-comparisons.
    Returns None for shapes the analyser can't decompose."""
    intervals: dict[str, tuple[float, float]] = {}
    epsilon = 1e-9

    def add(q: str, lo: float, hi: float) -> bool:
        existing = intervals.get(q)
        if existing is None:
            intervals[q] = (lo, hi)
            return True
        new_lo = max(existing[0], lo)
        new_hi = min(existing[1], hi)
        if new_lo > new_hi:
            return False
        intervals[q] = (new_lo, new_hi)
        return True

    def walk(p: Predicate) -> bool | None:
        if isinstance(p, QualityGte):
            return add(p.quality, float(p.value), float("inf"))
        if isinstance(p, QualityGt):
            return add(p.quality, float(p.value) + epsilon, float("inf"))
        if isinstance(p, QualityLte):
            return add(p.quality, float("-inf"), float(p.value))
        if isinstance(p, QualityLt):
            return add(p.quality, float("-inf"), float(p.value) - epsilon)
        if isinstance(p, QualityEquals):
            if not isinstance(p.value, (int, float)) or isinstance(p.value, bool):
                return None
            return add(p.quality, float(p.value), float(p.value))
        if isinstance(p, And_):
            for sub in p.of:
                r = walk(sub)
                if r is None:
                    return None
                if not r:
                    return False
            return True
        return None

    result = walk(pred)
    if result is None:
        return None
    return intervals


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
    max_days: int = Field(gt=0)
    # Full cast for the mission — operatives (role="agent"), the player's
    # in-fiction handler (role="handler"), the boss (role="boss"), named
    # opposition (role="opposition"). Only role="agent" members are seen by
    # the engine (archetype expansion, for_each_agent, cover injection); the
    # rest are renderer-prompt content. Access agents via `mission.agents`
    # (property); access any member via `mission.member(id)`.
    cast: list[CastMember] = Field(min_length=1)
    # Quality authoring. Each entry carries the initial value AND the narrative
    # descriptor in one block — co-locating them eliminates the older "missing
    # descriptor" footgun. Universal qualities (agent[*].cover,
    # mission.days_remaining) appear here only when overriding the engine's
    # built-in display; their initial values come from `max_days` and
    # `cast[*].starting_cover` (agent role only) and are injected by `make_initial_state`.
    qualities: dict[str, Quality] = Field(default_factory=dict)
    # Per-mission persistent prose constraints, injected into all three agent
    # system prompts. Use for facts the renderer/SoP must honor every day
    # (e.g. "LYNX must not be photographed again", "the Vienna network is
    # offline"). Authored, no LLM in the loop.
    standing_orders: list[str] = Field(default_factory=list)
    # Per-state-shape authored fragments. The engine selects one beat per
    # day (highest-priority `when`-matching unfired entry); it's threaded
    # into the drafter's user prompt as framing color AND into the log
    # writer's leans as a foregrounding directive. Fire-once by default so
    # the same observation never surfaces twice in a run (audit §2b).
    beats: list[Beat] = Field(default_factory=list)
    actions: list[Action] = Field(min_length=1)
    reactions: list[Reaction] = Field(min_length=1)
    # Day-1-only briefing prose. Prepended to the SoP user prompt on day 1
    # so the player meets the cast and the stakes before the operational
    # days begin. Empty = no briefing branch (skips the block). Front-loads
    # investment without restructuring days or adding a mechanic.
    opening_briefing: str = Field(default="")
    # Map of outcome_key → human prose summary for the outcome card.
    outcomes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _id_uniqueness(self) -> "Mission":
        for collection, label in (
            (self.cast, "cast member"),
            (self.actions, "action"),
            (self.reactions, "reaction"),
            (self.beats, "beat"),
        ):
            seen: set[str] = set()
            for item in collection:
                if item.id in seen:
                    raise ValueError(f"duplicate {label} id: {item.id!r}")
                seen.add(item.id)
        # Actions, Reactions, and Beats all share the `state.fired` namespace
        # (EventFired predicate is a membership check). Authored ids must be
        # disjoint across all three so a single id always refers to one thing.
        fired_ns: dict[str, str] = {}
        for action in self.actions:
            fired_ns[action.id] = "action"
        for reaction in self.reactions:
            if reaction.id in fired_ns:
                raise ValueError(
                    f"reaction id {reaction.id!r} collides with a "
                    f"{fired_ns[reaction.id]} id; both share the fired-ledger namespace"
                )
            fired_ns[reaction.id] = "reaction"
        for beat in self.beats:
            if beat.id in fired_ns:
                raise ValueError(
                    f"beat id {beat.id!r} collides with a {fired_ns[beat.id]} id; "
                    f"both share the fired-ledger namespace"
                )
        # Outcome keys referenced by terminal reactions must be in `outcomes`.
        for reaction in self.reactions:
            if reaction.terminal and reaction.outcome_key not in self.outcomes:
                raise ValueError(
                    f"Reaction {reaction.id!r}: outcome_key {reaction.outcome_key!r} "
                    f"is not in mission.outcomes ({sorted(self.outcomes)})"
                )
        return self

    @model_validator(mode="after")
    def _at_least_one_agent(self) -> "Mission":
        if not self.agents:
            raise ValueError(
                "Mission must include at least one cast member with role='agent'."
            )
        return self

    def initial_qualities_for_state(self) -> dict[str, float | int | str]:
        """Return the (name → initial value) mapping the engine should seed
        into State at mission start. Universal qualities (days_remaining,
        agent[*].cover) are injected separately by `make_initial_state`."""
        return {
            name: q.initial
            for name, q in self.qualities.items()
            if q.initial is not None
        }

    def action(self, id: str) -> Action:
        for a in self.actions:
            if a.id == id:
                return a
        raise KeyError(f"action {id!r} not in mission {self.id!r}")

    def reaction(self, id: str) -> Reaction:
        for r in self.reactions:
            if r.id == id:
                return r
        raise KeyError(f"reaction {id!r} not in mission {self.id!r}")

    @property
    def agents(self) -> list[CastMember]:
        """Agent-role cast members. The engine reads this for archetype
        expansion, for_each_agent, and cover injection — non-agent roles
        are renderer content only and must not appear here."""
        return [m for m in self.cast if m.role == "agent"]

    def agent(self, id: str) -> CastMember:
        for a in self.agents:
            if a.id == id:
                return a
        raise KeyError(f"agent {id!r} not in mission {self.id!r}")

    def member(self, id: str) -> CastMember:
        """Look up any cast member by id, regardless of role."""
        for m in self.cast:
            if m.id == id:
                return m
        raise KeyError(f"cast member {id!r} not in mission {self.id!r}")



# ───────────────────────── drafter output ─────────────────────────


class Option(BaseModel):
    """One option the drafter offers the player. Assembled by the engine from
    the selector's `(action, agent)` slot + the prose writer's
    `(title, prose)` pair. `action` and `agent` drive deterministic
    resolution; `title` and `prose` are display-only — they never influence
    state.

    A forced Action surfaces here like any other — the engine knows it's
    forced from the Action's `forced: true` flag, not from any field on
    Option. No carrier coupling: the option IS the action.
    """

    action: str = Field(min_length=1)
    agent: str | None = None
    title: str = Field(min_length=1)
    prose: str = Field(min_length=1)


class OptionProse(BaseModel):
    """Prose-writer structured output for one slot. The engine pairs each
    `OptionProse` (positionally) with a selector slot to build a final
    `Option`. No catalog identifiers here — those are pinned by the engine
    before this call runs."""

    title: str = Field(min_length=1)
    prose: str = Field(min_length=1)


# ───────────────────────── runtime state ─────────────────────────


class State(BaseModel):
    """Per-game runtime state. Qualities are namespaced flat strings —
    `mission.days_remaining`, `mission.extraction_stage`, `agent[LYNX].cover`.

    Universal qualities the engine guarantees (set up by `make_initial_state`):
    - `mission.days_remaining: int` (counts down to 0)
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

    def get_quality(self, name: str) -> float | int | str | bool | None:
        return self.qualities.get(name)

    def set_quality(self, name: str, value: float | int | str | bool) -> None:
        self.qualities[name] = value


def make_initial_state(mission: Mission) -> State:
    """Construct the starting `State` for a mission. Injects universal
    qualities (`mission.days_remaining`, `agent[*].cover`) and authored
    initial values from `mission.qualities`."""
    qualities: dict[str, float | int | str | bool] = {
        "mission.days_remaining": mission.max_days,
    }
    for agent in mission.agents:
        # CastMember validator guarantees role="agent" has starting_cover set.
        assert agent.starting_cover is not None
        qualities[f"agent[{agent.id}].cover"] = agent.starting_cover
    qualities.update(mission.initial_qualities_for_state())
    return State(qualities=qualities)


# ───────────────────────── resolution ─────────────────────────


class MissionResolution(BaseModel):
    """Snapshot at terminal — what the outcome card shows."""

    outcome_key: str
    terminal_reaction_id: str
    days_used: int
    days_max: int
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
