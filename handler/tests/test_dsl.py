"""DSL parser tests. The parsers must produce dicts equivalent to what the
canonical tagged-union YAML form would produce — i.e. round-trippable through
`Mission.model_validate` to identical model instances."""
from __future__ import annotations

import pytest

from handler.dsl import (
    DSLParseError,
    normalise_mission_dsl,
    parse_effect_string,
    parse_predicate_string,
)
from handler.models import (
    And_,
    Effect,
    EventFired,
    Not_,
    Or_,
    Predicate,
    QualityEquals,
    QualityGt,
    QualityGte,
    QualityLt,
    QualityLte,
)
from handler.mission_loader import load_mission
from pathlib import Path


def _adapt_predicate(payload: dict) -> Predicate:
    """Validate a parsed dict through `Effect.when` → returns the canonical
    `Predicate` model. Doing this through Effect's discriminated union is the
    cleanest way to exercise pydantic without writing a TypeAdapter."""
    return Effect(quality="x", value=0, when=payload).when  # type: ignore[return-value]


# ───────────────────────── predicate: simple comparisons ─────────────────────────


@pytest.mark.parametrize(
    "source,kind,quality,value",
    [
        ("agent[LYNX].cover >= 1",          "quality_gte",    "agent[LYNX].cover", 1),
        ("agent[LYNX].cover > 0",           "quality_gt",     "agent[LYNX].cover", 0),
        ("agent[LYNX].cover <= 5",          "quality_lte",    "agent[LYNX].cover", 5),
        ("agent[LYNX].cover < 3",           "quality_lt",     "agent[LYNX].cover", 3),
        ("mission.extraction_stage == 2",   "quality_equals", "mission.extraction_stage", 2),
        ("mission.extraction_stage = 2",    "quality_equals", "mission.extraction_stage", 2),
    ],
)
def test_parse_predicate_simple_comparison(source, kind, quality, value) -> None:
    out = parse_predicate_string(source)
    assert out == {"kind": kind, "quality": quality, "value": value}
    # Round-trip via pydantic to confirm the dict is canonical.
    pred = _adapt_predicate(out)
    assert pred.kind == kind  # type: ignore[union-attr]


# ───────────────────────── predicate: range, bool, string ─────────────────────────


def test_parse_predicate_range_expands_to_and() -> None:
    out = parse_predicate_string("agent[KESTREL].cover in 1..5")
    assert out == {
        "kind": "and",
        "of": [
            {"kind": "quality_gte", "quality": "agent[KESTREL].cover", "value": 1},
            {"kind": "quality_lte", "quality": "agent[KESTREL].cover", "value": 5},
        ],
    }
    pred = _adapt_predicate(out)
    assert isinstance(pred, And_)


def test_parse_predicate_bool_value() -> None:
    out = parse_predicate_string("agent[LYNX].cover_protected == true")
    assert out == {
        "kind": "quality_equals",
        "quality": "agent[LYNX].cover_protected",
        "value": True,
    }


def test_parse_predicate_string_value() -> None:
    out = parse_predicate_string('mission.outcome == "win"')
    assert out["value"] == "win"


def test_parse_predicate_bare_identifier_value() -> None:
    """Allow unquoted bare identifiers as enum-ish values (matches existing
    tagged-union behaviour where `value: foo` is fine in YAML)."""
    out = parse_predicate_string("mission.flag == active")
    assert out["value"] == "active"


# ───────────────────────── predicate: composites ─────────────────────────


def test_parse_predicate_and() -> None:
    out = parse_predicate_string(
        "mission.extraction_stage >= 1 and agent[LYNX].cover > 0"
    )
    assert out["kind"] == "and"
    assert len(out["of"]) == 2


def test_parse_predicate_or() -> None:
    out = parse_predicate_string(
        "agent[LYNX].cover <= 0 or agent[KESTREL].cover <= 0"
    )
    assert out["kind"] == "or"


def test_parse_predicate_not() -> None:
    out = parse_predicate_string("not event_fired:kestrel_sacrifice")
    assert out["kind"] == "not"
    assert out["of"] == {"kind": "event_fired", "event": "kestrel_sacrifice"}


def test_parse_predicate_precedence_not_over_and_over_or() -> None:
    """`not A and B or C` must parse as `(not A) and B) or C`."""
    out = parse_predicate_string("not agent[A].cover > 0 and agent[B].cover > 0 or mission.flag == true")
    assert out["kind"] == "or"
    left = out["of"][0]
    assert left["kind"] == "and"
    not_clause, _ = left["of"]
    assert not_clause["kind"] == "not"


def test_parse_predicate_parens_override_precedence() -> None:
    out = parse_predicate_string("agent[A].cover > 0 and (agent[B].cover > 0 or agent[C].cover > 0)")
    assert out["kind"] == "and"
    right = out["of"][1]
    assert right["kind"] == "or"


def test_parse_predicate_event_fired_flag() -> None:
    out = parse_predicate_string("event_fired:world_attention_tightens")
    assert out == {"kind": "event_fired", "event": "world_attention_tightens"}


def test_parse_predicate_substitution_placeholder_in_qref() -> None:
    """`$agent` (and other placeholders) survive parsing — the substitution
    layer resolves them later."""
    out = parse_predicate_string("agent[$agent].cover > 0")
    assert out == {"kind": "quality_gt", "quality": "agent[$agent].cover", "value": 0}


# ───────────────────────── effect parser ─────────────────────────


@pytest.mark.parametrize(
    "source,expected",
    [
        ("agent[LYNX].cover -= 2", {"quality": "agent[LYNX].cover", "value": -2, "op": "add"}),
        ("agent[LYNX].cover += 1", {"quality": "agent[LYNX].cover", "value": 1, "op": "add"}),
        ("mission.extraction_stage = 2", {"quality": "mission.extraction_stage", "value": 2, "op": "set"}),
        ("agent[$agent].cover_protected = true", {"quality": "agent[$agent].cover_protected", "value": True, "op": "set"}),
        ("agent[LYNX].cover max= 10", {"quality": "agent[LYNX].cover", "value": 10, "op": "max"}),
        ("agent[LYNX].cover min= 0", {"quality": "agent[LYNX].cover", "value": 0, "op": "min"}),
    ],
)
def test_parse_effect_basic_ops(source, expected) -> None:
    assert parse_effect_string(source) == expected


def test_parse_effect_with_when_guard() -> None:
    out = parse_effect_string("agent[LYNX].cover -= 4 if agent[LYNX].cover < 5")
    assert out["quality"] == "agent[LYNX].cover"
    assert out["value"] == -4
    assert out["op"] == "add"
    assert out["when"] == {"kind": "quality_lt", "quality": "agent[LYNX].cover", "value": 5}
    # Round-trip through pydantic.
    eff = Effect.model_validate(out)
    assert isinstance(eff.when, QualityLt)


# ───────────────────────── error reporting ─────────────────────────


def test_parse_predicate_unexpected_trailing() -> None:
    with pytest.raises(DSLParseError) as exc:
        parse_predicate_string("agent[A].cover > 0 garbage")
    assert "trailing" in str(exc.value)


def test_parse_predicate_empty() -> None:
    with pytest.raises(DSLParseError):
        parse_predicate_string("")


def test_parse_effect_missing_op() -> None:
    with pytest.raises(DSLParseError):
        parse_effect_string("agent[A].cover")


def test_parse_predicate_unexpected_char() -> None:
    with pytest.raises(DSLParseError) as exc:
        parse_predicate_string("agent[A].cover @ 0")
    assert "unexpected character" in str(exc.value)


# ───────────────────────── normalise_mission_dsl ─────────────────────────


def test_normalise_converts_string_when_in_archetype() -> None:
    raw = {
        "archetypes": [
            {"id": "X", "when": "mission.stage > 0", "effects": []},
        ]
    }
    out = normalise_mission_dsl(raw)
    assert out["archetypes"][0]["when"] == {
        "kind": "quality_gt",
        "quality": "mission.stage",
        "value": 0,
    }


def test_normalise_passes_through_dict_when() -> None:
    """Authors who wrote the explicit tagged-union dict still work."""
    raw = {
        "archetypes": [
            {
                "id": "X",
                "when": {"kind": "quality_gt", "quality": "x", "value": 0},
                "effects": [],
            }
        ]
    }
    out = normalise_mission_dsl(raw)
    assert out["archetypes"][0]["when"]["kind"] == "quality_gt"


def test_normalise_converts_string_effects_in_list() -> None:
    raw = {
        "events": [
            {
                "id": "e",
                "when": "mission.stage >= 1",
                "effects": ["agent[LYNX].cover -= 2", "mission.stage = 2"],
            }
        ]
    }
    out = normalise_mission_dsl(raw)
    effects = out["events"][0]["effects"]
    assert effects[0]["op"] == "add" and effects[0]["value"] == -2
    assert effects[1]["op"] == "set" and effects[1]["value"] == 2


def test_normalise_recurses_into_beats() -> None:
    raw = {
        "beats": [
            {
                "id": "lynx_brink",
                "when": "agent[LYNX].cover <= 2",
                "fragment": "LYNX is hours from collapse.",
            }
        ]
    }
    out = normalise_mission_dsl(raw)
    assert out["beats"][0]["when"]["kind"] == "quality_lte"


def test_normalise_handles_effect_dict_with_inline_when() -> None:
    """An effect authored as a dict can still carry a string `when:` shorthand."""
    raw = {
        "events": [
            {
                "id": "e",
                "when": "mission.stage >= 1",
                "effects": [
                    {"quality": "agent[LYNX].cover", "value": -2, "op": "add",
                     "when": "agent[LYNX].cover > 0"},
                ],
            }
        ]
    }
    out = normalise_mission_dsl(raw)
    effect = out["events"][0]["effects"][0]
    assert effect["when"]["kind"] == "quality_gt"


# ───────────────────────── end-to-end via load_mission ─────────────────────────


def test_dsl_round_trip_via_inline_mission(tmp_path: Path) -> None:
    """A mission authored entirely in DSL shorthand loads, validates, and
    structurally matches its tagged-union equivalent."""
    shorthand = tmp_path / "shorthand.yaml"
    shorthand.write_text(
        """\
id: tt
name: "DSL test"
max_cycles: 3
cast:
  - id: A
    codename: A
    starting_cover: 5
qualities:
  mission.stage:
    initial: 0
    values: { "0": "init", "1": "mid", "2": "done" }
archetypes:
  - id: PUSH
    when: "mission.stage < 2"
    effects:
      - "mission.stage += 1"
  - id: BURN
    when: "agent[$agent].cover > 0 and mission.stage in 0..1"
    params: [agent]
    effects:
      - "agent[$agent].cover -= 2"
      - "mission.stage += 1"
events:
  - id: win
    terminal: true
    outcome_key: w
    when: "mission.stage >= 2"
  - id: timeout
    terminal: true
    outcome_key: lo
    when: "mission.cycles_remaining == 0"
outcomes:
  w: ""
  lo: ""
""",
        encoding="utf-8",
    )
    m = load_mission(shorthand)
    assert m.id == "tt"
    push = m.archetype("PUSH")
    assert isinstance(push.when, QualityLt)
    burn = m.archetype("BURN")
    # The `in 0..1` clause expands to And_.
    assert isinstance(burn.when, And_)
    assert burn.effects[0].op == "add"
    assert burn.effects[0].value == -2
