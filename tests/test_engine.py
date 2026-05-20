"""Engine helpers (reachability, state application, ending resolution) +
rule enforcement.

After the Phase-2-split refactor, rules that used to live in
`validate_state_change_bundle` and `validate_prose_part` are split between:
  - Canonical schemas (state-INDEPENDENT @model_validators / Field
    constraints) — tested here by directly constructing the canonical
    models.
  - Per-call dynamic builders (state-DEPENDENT Literal narrowing) —
    tested here against `_build_state_change_bundle_type` via
    `ValidationError` at parse time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from handler.agents import _build_state_change_bundle_type
import random

from handler.engine import (
    FREE_TEXT_COOLDOWN,
    FREE_TEXT_FALLBACK,
    FREE_TEXT_OFFER_P,
    _turns_since_typed,
    apply_state_changes,
    free_text_cadence_ok,
    reachable_beat_ids,
    resolved_ending_id_at_close,
)
from handler.models import Beat, Brief, Ending, Location, NPC
from handler.schemas import (
    ChoiceBundle,
    Continue,
    FreeText,
    Option,
    StateChangeBundle,
    UnlockBeat,
)
from handler.state import State, TurnRecord, make_initial_state
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
        glosses=[],
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


# ───────────────────────── canonical Continue validator ─────────────────────────


def test_continue_with_two_options_passes():
    Continue(options=[
        Option(text="stay"),
        Option(text="ask"),
    ])


def test_continue_with_four_options_passes():
    Continue(options=[
        Option(text="a"),
        Option(text="b"),
        Option(text="c"),
        Option(text="d"),
    ])


def test_continue_with_one_option_rejected():
    with pytest.raises(ValidationError):
        Continue(options=[Option(text="only one")])


def test_continue_with_five_options_rejected():
    with pytest.raises(ValidationError):
        Continue(options=[
            Option(text="a"),
            Option(text="b"),
            Option(text="c"),
            Option(text="d"),
            Option(text="e"),
        ])


def test_continue_empty_options_rejected():
    with pytest.raises(ValidationError):
        Continue(options=[])


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


# ───────────────────────── dynamic ChoiceBundle (Literal narrowing) ─────────────────────────


def test_choice_continue_with_two_options_passes():
    """The choice author uses the canonical `ChoiceBundle` directly (no
    per-call narrowing now that FreeText is gone)."""
    bundle = ChoiceBundle.model_validate({
        "next": {
            "type": "continue",
            "options": [{"text": "a"}, {"text": "b"}],
        },
    })
    assert bundle.next.type == "continue"  # type: ignore[union-attr]


def test_choice_continue_with_one_option_rejected():
    """Continue's `min_length=2` lives on the canonical schema."""
    with pytest.raises(ValidationError):
        ChoiceBundle.model_validate({
            "next": {"type": "continue", "options": [{"text": "only"}]},
        })


def test_choice_accepts_leave_marker():
    """`Leave` is a marker variant with no payload."""
    bundle = ChoiceBundle.model_validate({"next": {"type": "leave"}})
    assert bundle.next.type == "leave"  # type: ignore[union-attr]


def test_choice_rejects_free_text():
    """FreeText is no longer a choice-author arm — the gate + engine own it."""
    with pytest.raises(ValidationError):
        ChoiceBundle.model_validate({
            "next": {"type": "free_text", "prompt": "go on", "npc_id": None},
        })


def test_free_text_fallback_is_general_agency():
    """The fixed fallback box (caption author failed): neutral caption, no
    addressee, no suggestion."""
    assert isinstance(FREE_TEXT_FALLBACK, FreeText)
    assert FREE_TEXT_FALLBACK.prompt == "What do you do next?"
    assert FREE_TEXT_FALLBACK.npc_id is None
    assert FREE_TEXT_FALLBACK.suggested is None


# ───────────────────────── free-text cadence ─────────────────────────


def _state_with_history(typed_flags: list[bool]) -> State:
    """A State whose dialogue_history is one completed TurnRecord per flag
    (oldest first), each marking whether the player typed that turn."""
    state = State(player_location="x")
    for typed in typed_flags:
        state.dialogue_history.append(
            TurnRecord(prose="p", applied_state_changes=[], player_pick_typed=typed)
        )
    return state


def test_turns_since_typed_counts_back_to_last_typed_move():
    # most recent typed move is 2 completed turns back
    assert _turns_since_typed(_state_with_history([True, False, False])) == 2
    # the last completed turn was typed → 0 turns since
    assert _turns_since_typed(_state_with_history([False, True])) == 0


def test_turns_since_typed_large_when_never_typed():
    state = _state_with_history([False, False])
    assert _turns_since_typed(state) > FREE_TEXT_COOLDOWN


def test_cadence_suppresses_within_cooldown():
    """A typed move within FREE_TEXT_COOLDOWN turns suppresses the offer
    regardless of the RNG draw."""
    # last turn was typed → 0 < cooldown → suppressed even with rng→0.0
    state = _state_with_history([True])
    rng = random.Random(0)
    assert free_text_cadence_ok(state, rng) is False


def test_cadence_probability_is_seedable_and_honored_past_cooldown():
    """Past the cooldown, the offer is honored only with probability
    FREE_TEXT_OFFER_P — deterministic under a seeded RNG."""
    state = _state_with_history([False, False, False])  # no recent typed move

    class _Stub:
        def __init__(self, val): self.val = val
        def random(self): return self.val

    assert free_text_cadence_ok(state, _Stub(FREE_TEXT_OFFER_P - 0.01)) is True
    assert free_text_cadence_ok(state, _Stub(FREE_TEXT_OFFER_P + 0.01)) is False


# ───────────────────────── resolved_ending_id_at_close ─────────────────────────


def test_resolved_ending_id_picks_first_satisfied_in_author_order():
    """Two endings, both with their requires_beats fired → returns the
    one declared first in YAML (preserved by world.endings being an
    insertion-ordered dict)."""
    world = _make_world(
        beats=[_beat("filed"), _beat("walked_away")],
        endings=[
            Ending(id="published", requires_beats=["filed"], notes_to_llm="x", epilogue="y"),
            Ending(id="abandoned", requires_beats=["walked_away"], notes_to_llm="x", epilogue="y"),
        ],
    )
    state = State(player_location="loc_a", unlocked_beats={"filed", "walked_away"})
    assert resolved_ending_id_at_close(world, state) == "published"


def test_resolved_ending_id_picks_only_satisfied():
    """One ending satisfied, one not → returns the satisfied one even if
    it's not first in author order."""
    world = _make_world(
        beats=[_beat("filed"), _beat("walked_away")],
        endings=[
            Ending(id="published", requires_beats=["filed"], notes_to_llm="x", epilogue="y"),
            Ending(id="abandoned", requires_beats=["walked_away"], notes_to_llm="x", epilogue="y"),
        ],
    )
    state = State(player_location="loc_a", unlocked_beats={"walked_away"})
    assert resolved_ending_id_at_close(world, state) == "abandoned"


def test_resolved_ending_id_returns_none_when_none_satisfied():
    world = _make_world(
        beats=[_beat("filed")],
        endings=[Ending(
            id="published", requires_beats=["filed"], notes_to_llm="x", epilogue="y",
        )],
    )
    state = State(player_location="loc_a")
    assert resolved_ending_id_at_close(world, state) is None


def test_resolved_ending_id_returns_none_with_no_endings_authored():
    world = _make_world(endings=[])
    state = State(player_location="loc_a")
    assert resolved_ending_id_at_close(world, state) is None


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


# ───────────────────────── requires_any_beats (OR-clause) ─────────────────────────


def test_reachable_with_requires_any_all_unfired_but_reachable():
    """A requires_any_beats [X, Y]; both X and Y are unfired but both
    standalone-reachable → A is reachable."""
    world = _make_world(beats=[
        _beat("X"),
        _beat("Y"),
        _beat("A", requires_any_beats=["X", "Y"]),
    ])
    state = State(player_location="loc_a")
    assert reachable_beat_ids(world, state) == {"X", "Y", "A"}


def test_reachable_with_requires_any_one_branch_blocked():
    """A requires_any_beats [X, Y]; X is permanently blocked but Y is
    reachable → A is still reachable through Y."""
    world = _make_world(beats=[
        _beat("blocker"),
        _beat("X", blocked_by_beats=["blocker"]),
        _beat("Y"),
        _beat("A", requires_any_beats=["X", "Y"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker"})
    assert reachable_beat_ids(world, state) == {"Y", "A"}


def test_reachable_with_requires_any_all_branches_blocked():
    """A requires_any_beats [X, Y]; both X and Y permanently blocked →
    A is unreachable."""
    world = _make_world(beats=[
        _beat("blocker"),
        _beat("X", blocked_by_beats=["blocker"]),
        _beat("Y", blocked_by_beats=["blocker"]),
        _beat("A", requires_any_beats=["X", "Y"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker"})
    assert reachable_beat_ids(world, state) == set()


def test_reachable_with_requires_any_one_fired():
    """A requires_any_beats [X, Y]; X has fired → A is reachable even
    if Y is permanently blocked."""
    world = _make_world(beats=[
        _beat("blocker"),
        _beat("X"),
        _beat("Y", blocked_by_beats=["blocker"]),
        _beat("A", requires_any_beats=["X", "Y"]),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker", "X"})
    assert reachable_beat_ids(world, state) == {"A"}


def test_reachable_combines_requires_and_requires_any():
    """A requires_beats=[hard] AND requires_any_beats=[X, Y]; hard fires
    and Y fires → A reachable. hard fires but neither X nor Y → A
    unreachable."""
    world_ok = _make_world(beats=[
        _beat("hard"),
        _beat("X"),
        _beat("Y"),
        _beat("A", requires_beats=["hard"], requires_any_beats=["X", "Y"]),
    ])
    state_ok = State(player_location="loc_a", unlocked_beats={"hard", "Y"})
    assert "A" in reachable_beat_ids(world_ok, state_ok)

    # hard fires, X and Y both blocked → A unreachable
    world_blocked = _make_world(beats=[
        _beat("blocker"),
        _beat("hard"),
        _beat("X", blocked_by_beats=["blocker"]),
        _beat("Y", blocked_by_beats=["blocker"]),
        _beat("A", requires_beats=["hard"], requires_any_beats=["X", "Y"]),
    ])
    state_blocked = State(
        player_location="loc_a", unlocked_beats={"hard", "blocker"},
    )
    assert "A" not in reachable_beat_ids(world_blocked, state_blocked)


def test_resolved_ending_id_with_requires_any_one_satisfied():
    """Ending requires_any_beats=[X, Y, Z]; only Y fired → ending fires."""
    world = _make_world(
        beats=[_beat("X"), _beat("Y"), _beat("Z")],
        endings=[Ending(
            id="published", requires_any_beats=["X", "Y", "Z"],
            notes_to_llm="x", epilogue="y",
        )],
    )
    state = State(player_location="loc_a", unlocked_beats={"Y"})
    assert resolved_ending_id_at_close(world, state) == "published"


def test_resolved_ending_id_with_requires_any_none_satisfied():
    """Ending requires_any_beats=[X, Y]; neither fired → None."""
    world = _make_world(
        beats=[_beat("X"), _beat("Y")],
        endings=[Ending(
            id="published", requires_any_beats=["X", "Y"],
            notes_to_llm="x", epilogue="y",
        )],
    )
    state = State(player_location="loc_a")
    assert resolved_ending_id_at_close(world, state) is None


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
