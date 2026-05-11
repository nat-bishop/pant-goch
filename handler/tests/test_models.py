"""Unit tests for the engine model schemas — predicate algebra, effect
validation, id uniqueness."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from handler.models import (
    AgentDossier,
    And_,
    Archetype,
    Beat,
    Effect,
    Mission,
    Moment,
    Not_,
    Or_,
    QualityEquals,
    QualityGte,
    QualityGt,
    QualityLt,
    QualityLte,
    State,
    iter_predicate_leaves,
    make_initial_state,
    substitute_effect,
    substitute_predicate,
)


# ───────────────────────── predicate algebra ─────────────────────────


def test_predicate_discriminator_roundtrip() -> None:
    """Every leaf variant survives a model_validate round-trip."""
    for variant in [
        {"kind": "quality_gte", "quality": "x", "value": 1.0},
        {"kind": "quality_gt", "quality": "x", "value": 1.0},
        {"kind": "quality_lte", "quality": "x", "value": 1.0},
        {"kind": "quality_lt", "quality": "x", "value": 1.0},
        {"kind": "quality_equals", "quality": "x", "value": "foo"},
        {"kind": "beat_fired", "beat": "b"},
        {"kind": "moment_fired", "moment": "m"},
    ]:
        adapter = Effect.model_fields["when"].annotation
        # smoke: we can dump and reload via Effect's `when:` field
        e = Effect(quality="x", value=1, when=variant)  # type: ignore[arg-type]
        dumped = e.model_dump(exclude_none=True)
        Effect.model_validate(dumped)


def test_iter_predicate_leaves_walks_composites() -> None:
    p = And_(of=[
        QualityGte(quality="a", value=1),
        Or_(of=[QualityLt(quality="b", value=2), Not_(of=QualityEquals(quality="c", value="x"))]),
    ])
    leaves = list(iter_predicate_leaves(p))
    qualities = [getattr(leaf, "quality", None) for leaf in leaves]
    assert qualities == ["a", "b", "c"]


def test_substitute_predicate_replaces_agent_placeholder() -> None:
    pred = And_(of=[
        QualityGt(quality="agent[$agent].cover", value=0),
        QualityEquals(quality="mission.extraction_stage", value=1),
    ])
    out = substitute_predicate(pred, "KESTREL")
    leaves = list(iter_predicate_leaves(out))
    assert leaves[0].quality == "agent[KESTREL].cover"
    assert leaves[1].quality == "mission.extraction_stage"  # untouched


def test_substitute_effect_carries_through_when() -> None:
    eff = Effect(
        quality="agent[$agent].cover",
        value=-2,
        op="add",
        when=QualityGt(quality="agent[$agent].cover", value=0),
    )
    out = substitute_effect(eff, "LYNX")
    assert out.quality == "agent[LYNX].cover"
    assert isinstance(out.when, QualityGt)
    assert out.when.quality == "agent[LYNX].cover"


# ───────────────────────── id uniqueness ─────────────────────────


def _make_mission(*, beats: list[Beat] | None = None) -> Mission:
    return Mission(
        id="t",
        name="t",
        max_cycles=2,
        agents=[AgentDossier(id="A", codename="A", starting_cover=5)],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        beats=beats or [
            Beat(
                id="end",
                terminal=True,
                outcome_key="win",
                when=QualityGte(quality="mission.progress", value=1),
            )
        ],
        outcomes={"win": ""},
    )


def test_mission_rejects_duplicate_archetype_ids() -> None:
    with pytest.raises(ValidationError) as exc:
        Mission(
            id="t",
            name="t",
            max_cycles=2,
            agents=[AgentDossier(id="A", codename="A", starting_cover=5)],
            archetypes=[
                Archetype(id="X", when=QualityGte(quality="a", value=0)),
                Archetype(id="X", when=QualityGte(quality="a", value=0)),
            ],
            beats=[
                Beat(
                    id="end",
                    terminal=True,
                    outcome_key="win",
                    when=QualityGte(quality="a", value=0),
                )
            ],
            outcomes={"win": ""},
        )
    assert "duplicate archetype id" in str(exc.value)


def test_terminal_beat_requires_known_outcome_key() -> None:
    bad = Beat(
        id="end",
        terminal=True,
        outcome_key="unknown_outcome",
        when=QualityGte(quality="x", value=0),
    )
    with pytest.raises(ValidationError) as exc:
        _make_mission(beats=[bad])
    assert "outcome_key" in str(exc.value)


def test_terminal_beat_without_outcome_key_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        Beat(
            id="end",
            terminal=True,
            outcome_key=None,
            when=QualityGte(quality="x", value=0),
        )
    assert "outcome_key" in str(exc.value)


def test_for_each_agent_requires_per_agent_or_unlimited() -> None:
    with pytest.raises(ValidationError) as exc:
        Beat(
            id="b",
            for_each_agent=True,
            fires_at_most="once_per_mission",
            when=QualityGte(quality="x", value=0),
        )
    assert "for_each_agent" in str(exc.value)


# ───────────────────────── initial state injection ─────────────────────────


def test_make_initial_state_injects_universal_qualities() -> None:
    m = Mission(
        id="t",
        name="t",
        max_cycles=4,
        agents=[
            AgentDossier(id="A", codename="A", starting_cover=6),
            AgentDossier(id="B", codename="B", starting_cover=9),
        ],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        beats=[
            Beat(
                id="end",
                terminal=True,
                outcome_key="win",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"win": ""},
        initial_qualities={"mission.stage": 0},
    )
    s = make_initial_state(m)
    assert s.get_quality("mission.cycles_remaining") == 4
    assert s.get_quality("agent[A].cover") == 6
    assert s.get_quality("agent[B].cover") == 9
    assert s.get_quality("mission.stage") == 0
