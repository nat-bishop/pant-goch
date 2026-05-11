"""Engine primitives: predicates, effects, qualities, and minimal State/Mission
placeholders.

Predicates are tagged-union Pydantic models — every authored predicate is
type-checked at load time and impossible to misspell silently.

Handler-specific schemas (AgentDossier, Archetype, Moment, Beat, Option,
OptionSet, full State, full Mission) are authored in Phase 1.
"""
from __future__ import annotations

from typing import Annotated, Any, Iterable, Iterator, Literal, Union

from pydantic import BaseModel, Field


# ───────────────────────── predicate DSL ─────────────────────────


class _PredicateLeaf(BaseModel):
    """Mixin base for leaf predicate variants — evaluated against state directly."""


class _PredicateComposite(BaseModel):
    """Mixin base for composite predicate variants. Must yield sub-predicates."""

    def children(self) -> Iterable["Predicate"]:
        raise NotImplementedError(
            f"{type(self).__name__}: composite predicate must implement children()"
        )


class QualityGte(_PredicateLeaf):
    kind: Literal["quality_gte"] = "quality_gte"
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


class Not_(_PredicateComposite):
    kind: Literal["not"] = "not"
    of: "Predicate"

    def children(self) -> Iterable["Predicate"]:
        yield self.of


class Or_(_PredicateComposite):
    """OR over a non-empty list. List-level AND is already expressible by
    putting multiple predicates in a `when:` list (every entry must pass)."""

    kind: Literal["or"] = "or"
    of: list["Predicate"] = Field(min_length=1)

    def children(self) -> Iterable["Predicate"]:
        return iter(self.of)


Predicate = Annotated[
    Union[QualityGte, QualityLt, QualityEquals, Not_, Or_],
    Field(discriminator="kind"),
]

Not_.model_rebuild()
Or_.model_rebuild()


def iter_predicate_leaves(pred: "Predicate") -> Iterator["Predicate"]:
    """Yield every leaf predicate inside `pred`, walking composites recursively.

    Single source of truth for "walk the predicate tree, give me each leaf."
    A new `_PredicateLeaf` subclass works automatically; a new
    `_PredicateComposite` works as soon as its `children()` method is implemented.
    """
    if isinstance(pred, _PredicateComposite):
        for child in pred.children():
            yield from iter_predicate_leaves(child)
    else:
        yield pred  # _PredicateLeaf


# ───────────────────────── effect primitive ─────────────────────────


class Effect(BaseModel):
    """One literal state mutation. `op` controls how the value combines with
    the existing quality: `set` (replace), `add` (accumulate), `max` (floor),
    `min` (cap). `when` optionally gates the effect on a predicate.
    """

    quality: str = Field(min_length=1)
    value: float | int | str
    op: Literal["set", "add", "max", "min"] = "set"
    when: Predicate | None = None


# ───────────────────────── Phase 1 placeholders ─────────────────────────
#
# Phase 1 replaces these with the real Handler schemas — Mission (authored
# YAML), State (runtime), AgentDossier, Archetype, Moment, Beat, Option,
# OptionSet, MissionResolution. These stubs exist only so `session.py` and
# other Phase 0 survivors can import without ImportError until Phase 1.


class State(BaseModel):
    """Phase 1 fleshes out the real quality scopes (mission, per-agent,
    engine-internal). For now, a flat dict suffices for `get_quality`."""

    qualities: dict[str, float | int | str] = Field(default_factory=dict)

    def get_quality(self, name: str) -> float | int | str | None:
        return self.qualities.get(name)

    def set_quality(self, name: str, value: float | int | str) -> None:
        self.qualities[name] = value


class Mission(BaseModel):
    """Phase 1 fleshes out: agents, archetypes, moments, beats, max_cycles,
    opening prose, initial qualities."""

    id: str = "placeholder"
