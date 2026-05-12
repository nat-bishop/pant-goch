"""Unit tests for the engine model schemas — predicate algebra, effect
validation, id uniqueness."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from handler.models import (
    AgentDossier,
    And_,
    Archetype,
    DramaticQuestion,
    Effect,
    Event,
    Mission,
    Not_,
    Or_,
    Quality,
    QualityEquals,
    QualityGte,
    QualityGt,
    QualityLt,
    QualityLte,
    State,
    iter_predicate_leaves,
    make_initial_state,
    substitute_effect,
    substitute_placeholders,
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
        {"kind": "event_fired", "event": "e"},
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
    out = substitute_predicate(pred, {"agent": "KESTREL"})
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
    out = substitute_effect(eff, {"agent": "LYNX"})
    assert out.quality == "agent[LYNX].cover"
    assert isinstance(out.when, QualityGt)
    assert out.when.quality == "agent[LYNX].cover"


# ───────────────────────── id uniqueness ─────────────────────────


def _make_mission(*, events: list[Event] | None = None) -> Mission:
    return Mission(
        id="t",
        name="t",
        max_cycles=2,
        agents=[AgentDossier(id="A", codename="A", starting_cover=5)],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        events=events or [
            Event(
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
            events=[
                Event(
                    id="end",
                    terminal=True,
                    outcome_key="win",
                    when=QualityGte(quality="a", value=0),
                )
            ],
            outcomes={"win": ""},
        )
    assert "duplicate archetype id" in str(exc.value)


def test_terminal_event_requires_known_outcome_key() -> None:
    bad = Event(
        id="end",
        terminal=True,
        outcome_key="unknown_outcome",
        when=QualityGte(quality="x", value=0),
    )
    with pytest.raises(ValidationError) as exc:
        _make_mission(events=[bad])
    assert "outcome_key" in str(exc.value)


def test_terminal_event_without_outcome_key_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        Event(
            id="end",
            terminal=True,
            outcome_key=None,
            when=QualityGte(quality="x", value=0),
        )
    assert "outcome_key" in str(exc.value)


def test_for_each_agent_requires_per_agent_or_unlimited() -> None:
    with pytest.raises(ValidationError) as exc:
        Event(
            id="e",
            for_each_agent=True,
            fires_at_most="once_per_mission",
            when=QualityGte(quality="x", value=0),
        )
    assert "for_each_agent" in str(exc.value)


def test_force_into_drafter_rejects_terminal() -> None:
    with pytest.raises(ValidationError) as exc:
        Event(
            id="e",
            force_into_drafter=True,
            terminal=True,
            outcome_key="win",
            when=QualityGte(quality="x", value=0),
        )
    assert "force_into_drafter" in str(exc.value) and "terminal" in str(exc.value)


def test_force_into_drafter_rejects_for_each_agent() -> None:
    with pytest.raises(ValidationError) as exc:
        Event(
            id="e",
            force_into_drafter=True,
            for_each_agent=True,
            fires_at_most="once_per_agent",
            when=QualityGte(quality="x", value=0),
        )
    assert "force_into_drafter" in str(exc.value) and "for_each_agent" in str(exc.value)


def test_force_into_drafter_rejects_only_when_no_progress() -> None:
    with pytest.raises(ValidationError) as exc:
        Event(
            id="e",
            force_into_drafter=True,
            only_when_no_progress_this_cycle=True,
            when=QualityGte(quality="x", value=0),
        )
    assert "force_into_drafter" in str(exc.value)
    assert "only_when_no_progress" in str(exc.value)


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
        events=[
            Event(
                id="end",
                terminal=True,
                outcome_key="win",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"win": ""},
        qualities={"mission.stage": Quality(initial=0, values={"0": "stage 0"})},
    )
    s = make_initial_state(m)
    assert s.get_quality("mission.cycles_remaining") == 4
    assert s.get_quality("agent[A].cover") == 6
    assert s.get_quality("agent[B].cover") == 9
    assert s.get_quality("mission.stage") == 0


# ───────────────────────── Quality block ─────────────────────────


def _basic_mission_kwargs() -> dict:
    return dict(
        id="t",
        name="t",
        max_cycles=2,
        agents=[AgentDossier(id="A", codename="A", starting_cover=5)],
        archetypes=[Archetype(id="X", when=QualityGte(quality="a", value=0))],
        events=[
            Event(
                id="end",
                terminal=True,
                outcome_key="win",
                when=QualityGte(quality="x", value=0),
            )
        ],
        outcomes={"win": ""},
    )


def test_quality_requires_descriptor() -> None:
    """The whole point of the unified block: an authored initial without a
    descriptor is rejected at the per-quality level."""
    with pytest.raises(ValidationError):
        Quality(initial=0)


def test_quality_rejects_both_buckets_and_values() -> None:
    with pytest.raises(ValidationError):
        Quality(initial=0, buckets=[(1.0, "a")], values={"0": "b"})


def test_quality_describe_uses_values_map() -> None:
    q = Quality(initial=0, values={"0": "papers not yet surfaced", "1": "in hand"})
    assert q.describe(0) == "papers not yet surfaced"
    assert q.describe(1) == "in hand"
    assert q.describe(99) is None  # out of map


def test_quality_describe_uses_bucket_walk_with_label() -> None:
    q = Quality(label="extraction", buckets=[(0.0, "blown"), (10.0, "alive")])
    assert q.describe(0) == "extraction: blown"
    assert q.describe(5) == "extraction: alive"


def test_mission_qualities_block() -> None:
    """A mission authors qualities via `qualities:`. Initial value comes from
    the same block as the descriptor."""
    m = Mission(
        **_basic_mission_kwargs(),
        qualities={
            "mission.stage": Quality(
                initial=0, values={"0": "papers not yet surfaced", "1": "papers in hand"}
            ),
        },
    )
    assert m.initial_qualities_for_state() == {"mission.stage": 0}
    s = make_initial_state(m)
    assert s.get_quality("mission.stage") == 0


def test_mission_qualities_universal_override_omits_initial() -> None:
    """A `qualities:` entry without `initial:` is a descriptor-only override
    for a universal quality (`agent[*].cover`, `mission.cycles_remaining`).
    The engine still injects the initial value from `max_cycles` / `agents`."""
    m = Mission(
        **_basic_mission_kwargs(),
        qualities={
            "agent[*].cover": Quality(
                buckets=[(0.0, "blown"), (10.0, "alive")],
            ),
        },
    )
    # No authored initials — initial_qualities_for_state filters None.
    assert m.initial_qualities_for_state() == {}
    s = make_initial_state(m)
    # Universal cover injection still works.
    assert s.get_quality("agent[A].cover") == 5


# ───────────────────────── DramaticQuestion ─────────────────────────


def test_dramatic_question_requires_non_empty_question() -> None:
    with pytest.raises(ValidationError):
        DramaticQuestion(when=QualityGte(quality="x", value=0), question="")


def test_mission_dramatic_questions_default_empty() -> None:
    m = Mission(**_basic_mission_kwargs())
    assert m.dramatic_questions == []
    assert m.standing_orders == []


def test_mission_with_dramatic_questions_and_standing_orders() -> None:
    m = Mission(
        **_basic_mission_kwargs(),
        standing_orders=[
            "LYNX must not be photographed again.",
            "The Vienna network is offline.",
        ],
        dramatic_questions=[
            DramaticQuestion(
                when=QualityLte(quality="agent[A].cover", value=2),
                question="Is A still useful, or do we cut her loose?",
            ),
        ],
    )
    assert len(m.standing_orders) == 2
    assert len(m.dramatic_questions) == 1
    assert "useful" in m.dramatic_questions[0].question


# ───────────────────────── generalised substitution ─────────────────────────


def test_substitute_placeholders_dict_form() -> None:
    """Arbitrary placeholders are bound by name, not just `$agent`."""
    s = substitute_placeholders(
        "agent[$pays].cover bound to $bench",
        {"pays": "KESTREL", "bench": "LYNX"},
    )
    assert s == "agent[KESTREL].cover bound to LYNX"


def test_substitute_placeholders_empty_dict_is_noop() -> None:
    assert substitute_placeholders("no $name here", {}) == "no $name here"


def test_substitute_predicate_resolves_arbitrary_names() -> None:
    """Substitution walks quality / event refs (not predicate `value`
    fields — those carry concrete numbers or fixed enum strings)."""
    pred = And_(of=[
        QualityGt(quality="agent[$pays].cover", value=0),
        QualityGte(quality="mission.contributions[$pays]", value=1),
    ])
    out = substitute_predicate(pred, {"pays": "KESTREL"})
    leaves = list(iter_predicate_leaves(out))
    assert leaves[0].quality == "agent[KESTREL].cover"
    assert leaves[1].quality == "mission.contributions[KESTREL]"


def test_substitute_effect_resolves_arbitrary_names() -> None:
    eff = Effect(
        quality="agent[$pays].cover",
        value=-2,
        op="add",
        when=QualityGt(quality="agent[$pays].cover", value=0),
    )
    out = substitute_effect(eff, {"pays": "KESTREL"})
    assert out.quality == "agent[KESTREL].cover"
    assert isinstance(out.when, QualityGt)
    assert out.when.quality == "agent[KESTREL].cover"
