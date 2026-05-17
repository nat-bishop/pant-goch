"""Engine helpers (reachability, state application) + rule enforcement.

After the M3 rearch, rules that used to live in `validate_state_change_bundle`
and `validate_prose_part` are split between:
  - Canonical schemas (state-INDEPENDENT @model_validators) — tested here
    by directly constructing the canonical models.
  - Per-call dynamic builders (state-DEPENDENT Literal narrowing) — tested
    here against `_build_state_change_bundle_type` / `_build_prose_part_type`
    via `ValidationError` at parse time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from handler.agents import (
    _build_prose_part_type,
    _build_state_change_bundle_type,
)
from handler.engine import (
    apply_state_changes,
    reachable_beat_ids,
)
from handler.models import Beat, Brief, Ending, Location, NPC
from handler.schemas import (
    ContinueScene,
    EndGame,
    EndScene,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from handler.state import State, make_initial_state
from handler.world import World, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def _beat(id: str, **kwargs) -> Beat:
    """Engine-test Beat factory. Supplies `name` (derived from id) and the
    boilerplate description/notes_to_llm so each call stays one-line.
    Pass `name=` explicitly to override the derived name."""
    kwargs.setdefault("name", id.replace("_", " ").capitalize())
    kwargs.setdefault("description", "x")
    kwargs.setdefault("notes_to_llm", "y")
    return Beat(id=id, **kwargs)


def _make_world(
    *,
    locations: list[Location] | None = None,
    npcs: list[NPC] | None = None,
    beats: list[Beat] | None = None,
    endings: list[Ending] | None = None,
    starts_at: str = "loc_a",
) -> World:
    """Inline world builder for engine tests. Bypasses YAML + invariant
    checking (the tests need to construct shapes the loader would reject,
    e.g. permanently-blocked beats)."""
    locations = locations or [
        Location(id="loc_a", name="A", adjacent_to=["loc_b"], description="a"),
        Location(id="loc_b", name="B", adjacent_to=["loc_a"], description="b"),
    ]
    locations_dict = {loc.id: loc for loc in locations}
    npcs = npcs or []
    npcs_by_location: dict[str, list[NPC]] = {loc_id: [] for loc_id in locations_dict}
    for npc in npcs:
        npcs_by_location[npc.default_location].append(npc)
    beat_dict = {b.id: b for b in (beats or [])}
    return World(
        locations=locations_dict,
        npcs={n.id: n for n in npcs},
        crime="test",
        beats=beat_dict,
        endings={e.id: e for e in (endings or [])},
        brief=Brief(player_starts_at=starts_at, opening_seed="x"),
        npcs_by_location=npcs_by_location,
        few_shots=[],
        map_coords={},
        beat_name_to_id={b.name: b.id for b in beat_dict.values()},
    )


# ───────────────────────── apply_state_changes ─────────────────────────


def test_apply_unlock_beat_mutates_unlocked_beats():
    state = State(player_location="loc_a")
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="b1")])
    apply_state_changes(bundle, state)
    assert state.unlocked_beats == {"b1"}


def test_apply_two_unlock_beats_in_order():
    state = State(player_location="loc_a")
    bundle = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="b1"), UnlockBeat(beat_id="b2")]
    )
    apply_state_changes(bundle, state)
    assert state.unlocked_beats == {"b1", "b2"}


def test_apply_empty_bundle_is_noop():
    state = State(player_location="loc_a")
    apply_state_changes(StateChangeBundle(), state)
    assert state.unlocked_beats == set()


# ───────────────────────── canonical StateChangeBundle validators ─────────────────────────


def test_state_change_bundle_max_length_two_passes():
    StateChangeBundle(state_changes=[
        UnlockBeat(beat_id="b1"), UnlockBeat(beat_id="b2"),
    ])


def test_state_change_bundle_max_length_three_rejected():
    with pytest.raises(ValidationError):
        StateChangeBundle(state_changes=[
            UnlockBeat(beat_id="b1"),
            UnlockBeat(beat_id="b2"),
            UnlockBeat(beat_id="b3"),
        ])


def test_state_change_bundle_duplicate_beat_rejected():
    with pytest.raises(ValidationError, match="appears twice"):
        StateChangeBundle(state_changes=[
            UnlockBeat(beat_id="b1"), UnlockBeat(beat_id="b1"),
        ])


# ───────────────────────── canonical ContinueScene validator ─────────────────────────


def test_continue_scene_with_leave_option_passes():
    ContinueScene(options=[
        Option(text="stay"),
        Option(text="leave for B", leaves_for="loc_b"),
    ])


def test_continue_scene_without_leave_option_rejected():
    with pytest.raises(ValidationError, match="leaves_for"):
        ContinueScene(options=[
            Option(text="stay 1"),
            Option(text="stay 2"),
        ])


def test_continue_scene_empty_options_rejected():
    with pytest.raises(ValidationError, match="leaves_for"):
        ContinueScene(options=[])


# ───────────────────────── dynamic StateChangeBundle (Literal narrowing) ─────────────────────────


def test_dynamic_bundle_accepts_reachable_beat():
    world = _make_world(beats=[_beat("b1")])
    state = State(player_location="loc_a")
    BundleType = _build_state_change_bundle_type(world, state)
    bundle = BundleType.model_validate({
        "state_changes": [{"type": "unlock_beat", "beat_name": "B1"}]
    })
    assert bundle.state_changes[0].beat_name == "B1"  # type: ignore[union-attr]


def test_dynamic_bundle_rejects_unknown_beat():
    world = _make_world(beats=[_beat("b1")])
    state = State(player_location="loc_a")
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [{"type": "unlock_beat", "beat_name": "Ghost"}]
        })


def test_dynamic_bundle_rejects_already_fired_beat():
    """Reachability excludes already-fired beats → grammar Literal omits
    them → ValidationError at parse time."""
    world = _make_world(beats=[_beat("b1")])
    state = State(player_location="loc_a", unlocked_beats={"b1"})
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [{"type": "unlock_beat", "beat_name": "B1"}]
        })


def test_dynamic_bundle_rejects_unsatisfied_requires():
    """Reachability requires every prereq to be fired or transitively reachable.
    If prereq is unfired and standalone-reachable, b1 is reachable too — so
    we author a permanently-blocked prereq to test the rejection path."""
    world = _make_world(beats=[
        _beat("blocker"),
        _beat("prereq", blocked_by_beats=["blocker"]),
        _beat("b1", requires_beats=["prereq"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker"})
    # blocker is fired → prereq is permanently blocked → b1 is unreachable.
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [{"type": "unlock_beat", "beat_name": "B1"}]
        })


def test_dynamic_bundle_rejects_active_blocker():
    """blocked_by_beats with an active blocker → reachability excludes →
    grammar Literal omits → ValidationError."""
    world = _make_world(beats=[
        _beat("blocker"),
        _beat("b1", blocked_by_beats=["blocker"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker"})
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [{"type": "unlock_beat", "beat_name": "B1"}]
        })


def test_dynamic_bundle_empty_when_no_reachable_beats():
    """No reachable beats → bundle must be empty (max_length=0)."""
    world = _make_world(beats=[])
    state = State(player_location="loc_a")
    BundleType = _build_state_change_bundle_type(world, state)
    bundle = BundleType.model_validate({"state_changes": []})
    assert bundle.state_changes == []


def test_dynamic_bundle_max_length_three_rejected():
    world = _make_world(beats=[
        _beat("b1"),
        _beat("b2"),
        _beat("b3"),
    ])
    state = State(player_location="loc_a")
    BundleType = _build_state_change_bundle_type(world, state)
    with pytest.raises(ValidationError):
        BundleType.model_validate({
            "state_changes": [
                {"type": "unlock_beat", "beat_name": "B1"},
                {"type": "unlock_beat", "beat_name": "B2"},
                {"type": "unlock_beat", "beat_name": "B3"},
            ]
        })


# ───────────────────────── dynamic ProsePart (Literal narrowing) ─────────────────────────


def test_dynamic_prose_endgame_with_satisfied_prereqs_passes():
    world = _make_world(
        beats=[_beat("decided_to_file")],
        endings=[Ending(
            id="published_story",
            requires_beats=["decided_to_file"],
            notes_to_llm="x",
            epilogue="y",
        )],
    )
    state = State(player_location="loc_a", unlocked_beats={"decided_to_file"})
    ProseType = _build_prose_part_type(world, state)
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {"type": "end_game", "ending_id": "published_story"},
    })
    assert pp.next.ending_id == "published_story"  # type: ignore[union-attr]


def test_dynamic_prose_endgame_with_missing_prereqs_rejected():
    """Ending whose requires_beats aren't satisfied → not in the satisfied
    set → grammar Literal omits → ValidationError."""
    world = _make_world(
        beats=[_beat("decided_to_file")],
        endings=[Ending(
            id="published_story",
            requires_beats=["decided_to_file"],
            notes_to_llm="x",
            epilogue="y",
        )],
    )
    state = State(player_location="loc_a")  # decision beat not fired
    ProseType = _build_prose_part_type(world, state)
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {"type": "end_game", "ending_id": "published_story"},
        })


def test_dynamic_prose_endgame_undefined_ending_rejected():
    world = _make_world()
    state = State(player_location="loc_a")
    ProseType = _build_prose_part_type(world, state)
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {"type": "end_game", "ending_id": "ghost"},
        })


def test_dynamic_prose_continue_without_leave_option_rejected():
    """The ≥1-leave-option rule lives on canonical ContinueScene as
    @model_validator. The dynamic ContinueScene inherits it via __base__=
    so the rule fires at parse time, inside pydantic-ai's retry loop."""
    world = _make_world()
    state = State(player_location="loc_a")
    ProseType = _build_prose_part_type(world, state)
    with pytest.raises(ValidationError, match="leaves_for"):
        ProseType.model_validate({
            "prose": "...",
            "next": {
                "type": "continue",
                "options": [{"text": "stay 1"}, {"text": "stay 2"}],
            },
        })


def test_dynamic_prose_continue_with_leave_option_passes():
    world = _make_world()
    state = State(player_location="loc_a")
    ProseType = _build_prose_part_type(world, state)
    ProseType.model_validate({
        "prose": "...",
        "next": {
            "type": "continue",
            "options": [
                {"text": "stay"},
                {"text": "leave", "leaves_for": "loc_b"},
            ],
        },
    })


def test_dynamic_prose_continue_with_non_adjacent_leave_rejected():
    """leaves_for narrowed to Literal[*adjacent_to] → non-adjacent
    rejected by grammar."""
    world = _make_world(
        locations=[
            Location(id="loc_a", name="A", adjacent_to=["loc_b"], description="a"),
            Location(id="loc_b", name="B", adjacent_to=["loc_a"], description="b"),
            Location(id="loc_c", name="C", adjacent_to=[], description="c"),
        ],
    )
    state = State(player_location="loc_a")
    ProseType = _build_prose_part_type(world, state)
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {
                "type": "continue",
                "options": [
                    {"text": "leave for C", "leaves_for": "loc_c"},
                ],
            },
        })


# ───────────────────────── leave_pick → EndScene (grammar-enforced) ─────────────────────────


def test_leave_pick_turn_grammar_only_accepts_end_scene():
    """The regression that originally motivated `__base__=`. When leave_pick
    is set, the dynamic ProsePart's `next` is `EndScene` (no union) — so
    Continue/EndGame are rejected at parse time, before any application
    validation can run."""
    world = _make_world()
    state = State(player_location="loc_a")
    leave_pick = Option(text="leave", leaves_for="loc_b")
    ProseType = _build_prose_part_type(world, state, leave_pick=leave_pick)

    # ContinueScene rejected.
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {
                "type": "continue",
                "options": [
                    {"text": "stay"},
                    {"text": "leave", "leaves_for": "loc_b"},
                ],
            },
        })

    # EndGame rejected.
    with pytest.raises(ValidationError):
        ProseType.model_validate({
            "prose": "...",
            "next": {"type": "end_game", "ending_id": "anything"},
        })

    # EndScene accepted.
    pp = ProseType.model_validate({
        "prose": "...",
        "next": {"type": "end_scene"},
    })
    assert pp.next.type == "end_scene"


# ───────────────────────── reachability filter (transitive blocking) ─────────────────────────


def test_reachable_includes_standalone_unfired_beats():
    world = _make_world(beats=[
        _beat("standalone"),
    ])
    state = State(player_location="loc_a")
    assert reachable_beat_ids(world, state) == {"standalone"}


def test_reachable_excludes_fired_beats():
    world = _make_world(beats=[
        _beat("b1"),
    ])
    state = State(player_location="loc_a", unlocked_beats={"b1"})
    assert reachable_beat_ids(world, state) == set()


def test_reachable_excludes_directly_blocked_beats():
    world = _make_world(beats=[
        _beat("blocker"),
        _beat("b1", blocked_by_beats=["blocker"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker"})
    # blocker is fired (excluded); b1 is blocked (excluded).
    assert reachable_beat_ids(world, state) == set()


def test_reachable_includes_chain_when_root_unfired():
    world = _make_world(beats=[
        _beat("root"),
        _beat("leaf", requires_beats=["root"]),
    ])
    state = State(player_location="loc_a")
    # Both reachable: root is firable now, leaf is firable after root.
    assert reachable_beat_ids(world, state) == {"root", "leaf"}


def test_reachable_excludes_transitively_blocked_beats():
    """The whole point of the closure walk: if A requires B and B is
    permanently blocked by an already-fired C, then A is also unreachable
    even though A's own requires/blocks list doesn't mention C."""
    world = _make_world(beats=[
        _beat("C"),                                  # the blocker
        _beat("B", blocked_by_beats=["C"]),           # blocked by C
        _beat("A", requires_beats=["B"]),             # requires B
    ])
    # C has fired → B is permanently blocked → A is transitively unreachable.
    state = State(player_location="loc_a", unlocked_beats={"C"})
    assert reachable_beat_ids(world, state) == set()


def test_reachable_excludes_chain_when_intermediate_blocked():
    """Longer chain: A requires B requires Z; Z blocked by fired Y → A and B
    both unreachable."""
    world = _make_world(beats=[
        _beat("Y"),
        _beat("Z", blocked_by_beats=["Y"]),
        _beat("B", requires_beats=["Z"]),
        _beat("A", requires_beats=["B"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"Y"})
    assert reachable_beat_ids(world, state) == set()


def test_reachable_includes_alternative_when_one_branch_blocked():
    """If a beat has multiple independent paths that aren't blocked, it's
    still reachable through the open path. (Not currently in scope — beats
    have a single requires_beats list with AND semantics — but verifies the
    closure handles AND correctly.)"""
    world = _make_world(beats=[
        _beat("A"),
        _beat("B"),
        # C requires both A and B; both are independent and both reachable.
        _beat("C", requires_beats=["A", "B"]),
    ])
    state = State(player_location="loc_a")
    assert reachable_beat_ids(world, state) == {"A", "B", "C"}


def test_reachable_with_fired_prereq():
    """A requires B; B has fired → A is reachable now (no closure needed)."""
    world = _make_world(beats=[
        _beat("B"),
        _beat("A", requires_beats=["B"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"B"})
    assert reachable_beat_ids(world, state) == {"A"}


# ───────────────────────── integration: valid fixture ─────────────────────────


def test_dynamic_builder_against_loaded_fixture():
    """Sanity check: dynamic builder accepts valid changes against the
    fixture world. Dynamic items carry `beat_name`; the engine boundary
    (`emit_state_changes`) translates name→id via `World.beat_id_for_name`
    before constructing the canonical bundle."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    BundleType = _build_state_change_bundle_type(world, state)
    # tom_named_a_vehicle has no requires, and its blocker (tom_closes_off)
    # hasn't fired — so it's reachable.
    bundle = BundleType.model_validate({
        "state_changes": [
            {"type": "unlock_beat", "beat_name": "Tom named a vehicle"}
        ]
    })
    canonical = StateChangeBundle(state_changes=[
        UnlockBeat(beat_id=world.beat_id_for_name(c.beat_name))
        for c in bundle.state_changes
    ])
    apply_state_changes(canonical, state)
    assert "tom_named_a_vehicle" in state.unlocked_beats
    # Now tom_admits_more (requires tom_named_a_vehicle) is reachable.
    BundleType2 = _build_state_change_bundle_type(world, state)
    BundleType2.model_validate({
        "state_changes": [
            {"type": "unlock_beat", "beat_name": "Tom admits more"}
        ]
    })
