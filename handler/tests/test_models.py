"""Unit tests for the engine model schemas — predicate algebra, effect
validation, id uniqueness, Action/Reaction validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from handler.models import (
    Action,
    And_,
    Band,
    Beat,
    CastMember,
    Effect,
    Mission,
    Not_,
    Option,
    Or_,
    Outcome,
    Quality,
    QualityEquals,
    QualityGt,
    QualityGte,
    QualityLt,
    QualityLte,
    Reaction,
    iter_predicate_leaves,
    make_initial_state,
    substitute_effect,
    substitute_predicate,
)


# ───────────────────────── predicate algebra ─────────────────────────


def test_predicate_discriminator_roundtrip() -> None:
    for variant in [
        {"kind": "quality_gte", "quality": "x", "value": 1.0},
        {"kind": "quality_gt", "quality": "x", "value": 1.0},
        {"kind": "quality_lte", "quality": "x", "value": 1.0},
        {"kind": "quality_lt", "quality": "x", "value": 1.0},
        {"kind": "quality_equals", "quality": "x", "value": "foo"},
        {"kind": "event_fired", "event": "e"},
    ]:
        e = Effect(quality="x", value=1, when=variant)  # type: ignore[arg-type]
        dumped = e.model_dump(exclude_none=True)
        Effect.model_validate(dumped)


def test_iter_predicate_leaves_walks_composites() -> None:
    p = And_(of=[
        QualityGte(quality="a", value=1),
        Or_(of=[QualityLt(quality="b", value=2), Not_(of=QualityEquals(quality="c", value="x"))]),
    ])
    qualities = [getattr(leaf, "quality", None) for leaf in iter_predicate_leaves(p)]
    assert qualities == ["a", "b", "c"]


def test_substitute_predicate_replaces_agent_placeholder() -> None:
    pred = And_(of=[
        QualityGt(quality="agent[$agent].cover", value=0),
        QualityEquals(quality="mission.extraction_stage", value=1),
    ])
    out = substitute_predicate(pred, {"agent": "KESTREL"})
    leaves = list(iter_predicate_leaves(out))
    assert leaves[0].quality == "agent[KESTREL].cover"
    assert leaves[1].quality == "mission.extraction_stage"


def test_substitute_effect_carries_through_when() -> None:
    eff = Effect(
        quality="agent[$agent].cover",
        value=-2,
        op="add",
        when=QualityGt(quality="agent[$agent].cover", value=0),
    )
    out = substitute_effect(eff, {"agent": "LYNX"})
    assert out.quality == "agent[LYNX].cover"
    assert isinstance(out.when, QualityGt)
    assert out.when.quality == "agent[LYNX].cover"


# ───────────────────────── Mission factory ─────────────────────────


def _make_mission(*, actions=None, reactions=None, beats=None) -> Mission:
    return Mission(
        id="t",
        name="t",
        max_days=2,
        cast=[CastMember(id="A", codename="A", starting_cover=5)],
        actions=actions or [Action(id="X", when=QualityGte(quality="a", value=0))],
        reactions=reactions or [
            Reaction(
                id="end",
                terminal=True,
                outcome_key="win",
                when=QualityGte(quality="mission.progress", value=1),
            )
        ],
        beats=beats or [],
        outcomes={"win": ""},
    )


# ───────────────────────── id uniqueness ─────────────────────────


def test_mission_rejects_duplicate_action_ids() -> None:
    with pytest.raises(ValidationError) as exc:
        _make_mission(
            actions=[
                Action(id="X", when=QualityGte(quality="a", value=0)),
                Action(id="X", when=QualityGte(quality="a", value=0)),
            ],
        )
    assert "duplicate action id" in str(exc.value)


def test_mission_rejects_action_reaction_id_collision() -> None:
    with pytest.raises(ValidationError) as exc:
        _make_mission(
            actions=[Action(id="shared", when=QualityGte(quality="a", value=0))],
            reactions=[
                Reaction(
                    id="shared",
                    terminal=True,
                    outcome_key="win",
                    when=QualityGte(quality="x", value=0),
                )
            ],
        )
    assert "collides" in str(exc.value)


def test_mission_rejects_beat_id_colliding_with_action_id() -> None:
    with pytest.raises(ValidationError) as exc:
        _make_mission(
            actions=[Action(id="b", when=QualityGte(quality="a", value=0))],
            beats=[
                Beat(id="b", when=QualityGte(quality="x", value=0), fragment="f"),
            ],
        )
    assert "collides" in str(exc.value)


# ───────────────────────── Reaction validators ─────────────────────────


def test_terminal_reaction_requires_known_outcome_key() -> None:
    bad = Reaction(
        id="end",
        terminal=True,
        outcome_key="unknown",
        when=QualityGte(quality="x", value=0),
    )
    with pytest.raises(ValidationError):
        _make_mission(reactions=[bad])


def test_terminal_reaction_without_outcome_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Reaction(
            id="end",
            terminal=True,
            outcome_key=None,
            when=QualityGte(quality="x", value=0),
        )


def test_for_each_agent_reaction_requires_per_agent_or_unlimited() -> None:
    with pytest.raises(ValidationError):
        Reaction(
            id="r",
            for_each_agent=True,
            fires_at_most="once_per_mission",
            when=QualityGte(quality="x", value=0),
        )


def test_once_per_agent_requires_for_each_agent() -> None:
    with pytest.raises(ValidationError):
        Reaction(
            id="r",
            for_each_agent=False,
            fires_at_most="once_per_agent",
            when=QualityGte(quality="x", value=0),
        )


# ───────────────────────── Action validators ─────────────────────────


def test_action_effects_xor_outcomes_rejects_both() -> None:
    with pytest.raises(ValidationError):
        Action(
            id="X",
            when=QualityGte(quality="a", value=0),
            effects=[Effect(quality="a", value=1, op="add")],
            outcomes=[
                Outcome(when=QualityGte(quality="b", value=0), effects=[]),
            ],
        )


def test_action_outcomes_overlap_rejected() -> None:
    with pytest.raises(ValidationError):
        Action(
            id="X",
            when=QualityGte(quality="a", value=0),
            outcomes=[
                Outcome(when=QualityGte(quality="b", value=0), effects=[]),
                Outcome(when=QualityGte(quality="b", value=0), effects=[]),
            ],
        )


def test_action_outcomes_disjoint_intervals_accepted() -> None:
    """Mutually-exclusive AND-of-comparisons pass the validator."""
    action = Action(
        id="X",
        when=QualityGte(quality="a", value=0),
        outcomes=[
            Outcome(when=QualityLt(quality="b", value=5), effects=[]),
            Outcome(
                when=And_(of=[
                    QualityGte(quality="b", value=5),
                    QualityLte(quality="b", value=7),
                ]),
                effects=[],
            ),
            Outcome(when=QualityGte(quality="b", value=8), effects=[]),
        ],
    )
    assert len(action.outcomes) == 3


# ───────────────────────── initial state injection ─────────────────────────


def test_make_initial_state_injects_universal_qualities() -> None:
    m = Mission(
        id="t",
        name="t",
        max_days=4,
        cast=[
            CastMember(id="A", codename="A", starting_cover=6),
            CastMember(id="B", codename="B", starting_cover=9),
        ],
        actions=[Action(id="X", when=QualityGte(quality="a", value=0))],
        reactions=[
            Reaction(
                id="end",
                terminal=True,
                outcome_key="win",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"win": ""},
        qualities={"mission.stage": Quality(initial=0, bands=[Band(threshold=0, variants=["stage 0"])])},
    )
    s = make_initial_state(m)
    assert s.get_quality("mission.days_remaining") == 4
    assert s.get_quality("agent[A].cover") == 6
    assert s.get_quality("agent[B].cover") == 9
    assert s.get_quality("mission.stage") == 0


# ───────────────────────── Quality ─────────────────────────


def test_quality_requires_bands() -> None:
    with pytest.raises(ValidationError):
        Quality(initial=0)


def test_quality_describe_uses_enum_bands() -> None:
    q = Quality(
        initial=0,
        bands=[
            Band(threshold=0, variants=["papers not yet surfaced"]),
            Band(threshold=1, variants=["in hand"]),
        ],
    )
    assert q.describe(0) == "papers not yet surfaced"
    assert q.describe(1) == "in hand"
    assert q.describe(99) is None


def test_quality_describe_bands_walk_with_label() -> None:
    q = Quality(
        label="extraction",
        bands=[Band(threshold=0.0, variants=["blown"]), Band(threshold=10.0, variants=["alive"])],
    )
    assert q.describe(0) == "extraction: blown"
    assert q.describe(5) == "extraction: alive"


def test_quality_describe_variant_rotates_by_day_index() -> None:
    q = Quality(
        initial=0,
        visible=False,
        bands=[Band(threshold=5.0, variants=["v0", "v1", "v2"])],
    )
    assert q.describe_variant(3, 0) == "v0"
    assert q.describe_variant(3, 1) == "v1"
    assert q.describe_variant(3, 2) == "v2"
    assert q.describe_variant(3, 3) == "v0"


def test_band_accepts_tuple_form() -> None:
    q = Quality.model_validate(
        {"initial": 0, "bands": [[0, "low"], [5, ["mid-a", "mid-b"]]]}
    )
    assert q.describe(0) == "low"
    assert q.describe(3) == "mid-a"
    assert q.describe_variant(3, 1) == "mid-b"


# ───────────────────────── Beat ─────────────────────────


def test_beat_requires_id_and_fragment() -> None:
    with pytest.raises(ValidationError):
        Beat(id="", when=QualityGte(quality="x", value=0), fragment="x")
    with pytest.raises(ValidationError):
        Beat(id="x", when=QualityGte(quality="x", value=0), fragment="")


# ───────────────────────── Option ─────────────────────────


def test_option_carries_action_field() -> None:
    opt = Option(action="ACT", title="t", prose="p")
    assert opt.action == "ACT"
    assert opt.agent is None
