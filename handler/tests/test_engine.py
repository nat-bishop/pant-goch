"""M2 — Engine validators + reachability + state-change application."""
from __future__ import annotations

from pathlib import Path

import pytest

from handler.engine import (
    ProsePartError,
    StateChangeError,
    UNLOCK_BEAT_CAP_PER_BUNDLE,
    apply_state_changes,
    reachable_beat_ids,
    validate_prose_part,
    validate_state_change_bundle,
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
    return World(
        locations=locations_dict,
        npcs={n.id: n for n in npcs},
        crime="test",
        beats={b.id: b for b in (beats or [])},
        endings={e.id: e for e in (endings or [])},
        brief=Brief(player_starts_at=starts_at, opening_seed="x"),
        npcs_by_location=npcs_by_location,
        few_shots=[],
        map_coords={},
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


# ───────────────────────── validate_state_change_bundle ─────────────────────────


def test_unlock_beat_undefined_id_rejected():
    world = _make_world(beats=[Beat(id="b1", description="x", notes_to_llm="y")])
    state = State(player_location="loc_a")
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="ghost")])
    with pytest.raises(StateChangeError, match="not defined"):
        validate_state_change_bundle(bundle, world, state)


def test_unlock_beat_already_fired_rejected():
    world = _make_world(beats=[Beat(id="b1", description="x", notes_to_llm="y")])
    state = State(player_location="loc_a", unlocked_beats={"b1"})
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="b1")])
    with pytest.raises(StateChangeError, match="already fired"):
        validate_state_change_bundle(bundle, world, state)


def test_requires_beats_unsatisfied_rejected():
    world = _make_world(beats=[
        Beat(id="prereq", description="x", notes_to_llm="y"),
        Beat(id="b1", description="x", requires_beats=["prereq"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a")  # prereq not fired
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="b1")])
    with pytest.raises(StateChangeError, match="requires_beats not yet satisfied"):
        validate_state_change_bundle(bundle, world, state)


def test_requires_beats_satisfied_passes():
    world = _make_world(beats=[
        Beat(id="prereq", description="x", notes_to_llm="y"),
        Beat(id="b1", description="x", requires_beats=["prereq"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a", unlocked_beats={"prereq"})
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="b1")])
    # Should not raise
    validate_state_change_bundle(bundle, world, state)


def test_blocked_by_beats_active_rejected():
    world = _make_world(beats=[
        Beat(id="blocker", description="x", notes_to_llm="y"),
        Beat(id="b1", description="x", blocked_by_beats=["blocker"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker"})
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="b1")])
    with pytest.raises(StateChangeError, match="blocked by"):
        validate_state_change_bundle(bundle, world, state)


def test_blocked_by_inactive_passes():
    world = _make_world(beats=[
        Beat(id="blocker", description="x", notes_to_llm="y"),
        Beat(id="b1", description="x", blocked_by_beats=["blocker"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a")  # blocker NOT fired
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="b1")])
    validate_state_change_bundle(bundle, world, state)


# ───────────────────────── ≤2 unlock_beat per bundle cap ─────────────────────────


def test_one_unlock_beat_passes():
    world = _make_world(beats=[
        Beat(id="b1", description="x", notes_to_llm="y"),
    ])
    bundle = StateChangeBundle(state_changes=[UnlockBeat(beat_id="b1")])
    validate_state_change_bundle(bundle, world, State(player_location="loc_a"))


def test_two_unlock_beats_passes():
    world = _make_world(beats=[
        Beat(id="b1", description="x", notes_to_llm="y"),
        Beat(id="b2", description="x", notes_to_llm="y"),
    ])
    bundle = StateChangeBundle(state_changes=[
        UnlockBeat(beat_id="b1"), UnlockBeat(beat_id="b2"),
    ])
    validate_state_change_bundle(bundle, world, State(player_location="loc_a"))


def test_three_unlock_beats_rejected():
    world = _make_world(beats=[
        Beat(id="b1", description="x", notes_to_llm="y"),
        Beat(id="b2", description="x", notes_to_llm="y"),
        Beat(id="b3", description="x", notes_to_llm="y"),
    ])
    bundle = StateChangeBundle(state_changes=[
        UnlockBeat(beat_id="b1"),
        UnlockBeat(beat_id="b2"),
        UnlockBeat(beat_id="b3"),
    ])
    with pytest.raises(StateChangeError, match=f"> {UNLOCK_BEAT_CAP_PER_BUNDLE}"):
        validate_state_change_bundle(bundle, world, State(player_location="loc_a"))


def test_zero_unlock_beats_passes():
    world = _make_world()
    validate_state_change_bundle(
        StateChangeBundle(), world, State(player_location="loc_a")
    )


def test_duplicate_beat_in_bundle_rejected():
    world = _make_world(beats=[Beat(id="b1", description="x", notes_to_llm="y")])
    bundle = StateChangeBundle(state_changes=[
        UnlockBeat(beat_id="b1"), UnlockBeat(beat_id="b1"),
    ])
    with pytest.raises(StateChangeError, match="appears twice"):
        validate_state_change_bundle(bundle, world, State(player_location="loc_a"))


# ───────────────────────── reachability filter (transitive blocking) ─────────────────────────


def test_reachable_includes_standalone_unfired_beats():
    world = _make_world(beats=[
        Beat(id="standalone", description="x", notes_to_llm="y"),
    ])
    state = State(player_location="loc_a")
    assert reachable_beat_ids(world, state) == {"standalone"}


def test_reachable_excludes_fired_beats():
    world = _make_world(beats=[
        Beat(id="b1", description="x", notes_to_llm="y"),
    ])
    state = State(player_location="loc_a", unlocked_beats={"b1"})
    assert reachable_beat_ids(world, state) == set()


def test_reachable_excludes_directly_blocked_beats():
    world = _make_world(beats=[
        Beat(id="blocker", description="x", notes_to_llm="y"),
        Beat(id="b1", description="x", blocked_by_beats=["blocker"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a", unlocked_beats={"blocker"})
    # blocker is fired (excluded); b1 is blocked (excluded).
    assert reachable_beat_ids(world, state) == set()


def test_reachable_includes_chain_when_root_unfired():
    world = _make_world(beats=[
        Beat(id="root", description="x", notes_to_llm="y"),
        Beat(id="leaf", description="x", requires_beats=["root"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a")
    # Both reachable: root is firable now, leaf is firable after root.
    assert reachable_beat_ids(world, state) == {"root", "leaf"}


def test_reachable_excludes_transitively_blocked_beats():
    """The whole point of the closure walk: if A requires B and B is
    permanently blocked by an already-fired C, then A is also unreachable
    even though A's own requires/blocks list doesn't mention C."""
    world = _make_world(beats=[
        Beat(id="C", description="x", notes_to_llm="y"),                                  # the blocker
        Beat(id="B", description="x", blocked_by_beats=["C"], notes_to_llm="y"),           # blocked by C
        Beat(id="A", description="x", requires_beats=["B"], notes_to_llm="y"),             # requires B
    ])
    # C has fired → B is permanently blocked → A is transitively unreachable.
    state = State(player_location="loc_a", unlocked_beats={"C"})
    assert reachable_beat_ids(world, state) == set()


def test_reachable_excludes_chain_when_intermediate_blocked():
    """Longer chain: A requires B requires Z; Z blocked by fired Y → A and B
    both unreachable."""
    world = _make_world(beats=[
        Beat(id="Y", description="x", notes_to_llm="y"),
        Beat(id="Z", description="x", blocked_by_beats=["Y"], notes_to_llm="y"),
        Beat(id="B", description="x", requires_beats=["Z"], notes_to_llm="y"),
        Beat(id="A", description="x", requires_beats=["B"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a", unlocked_beats={"Y"})
    assert reachable_beat_ids(world, state) == set()


def test_reachable_includes_alternative_when_one_branch_blocked():
    """If a beat has multiple independent paths that aren't blocked, it's
    still reachable through the open path. (Not currently in scope — beats
    have a single requires_beats list with AND semantics — but verifies the
    closure handles AND correctly.)"""
    world = _make_world(beats=[
        Beat(id="A", description="x", notes_to_llm="y"),
        Beat(id="B", description="x", notes_to_llm="y"),
        # C requires both A and B; both are independent and both reachable.
        Beat(id="C", description="x", requires_beats=["A", "B"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a")
    assert reachable_beat_ids(world, state) == {"A", "B", "C"}


def test_reachable_with_fired_prereq():
    """A requires B; B has fired → A is reachable now (no closure needed)."""
    world = _make_world(beats=[
        Beat(id="B", description="x", notes_to_llm="y"),
        Beat(id="A", description="x", requires_beats=["B"], notes_to_llm="y"),
    ])
    state = State(player_location="loc_a", unlocked_beats={"B"})
    assert reachable_beat_ids(world, state) == {"A"}


# ───────────────────────── validate_prose_part ─────────────────────────


def test_endgame_with_satisfied_prereqs_passes():
    world = _make_world(
        beats=[Beat(id="decided_to_file", description="x", notes_to_llm="y")],
        endings=[Ending(
            id="published_story",
            requires_beats=["decided_to_file"],
            notes_to_llm="x",
            epilogue="y",
        )],
    )
    state = State(player_location="loc_a", unlocked_beats={"decided_to_file"})
    prose_part = ProsePart(prose="...", next=EndGame(ending_id="published_story"))
    validate_prose_part(prose_part, world, state)


def test_endgame_with_missing_prereqs_rejected():
    world = _make_world(
        beats=[Beat(id="decided_to_file", description="x", notes_to_llm="y")],
        endings=[Ending(
            id="published_story",
            requires_beats=["decided_to_file"],
            notes_to_llm="x",
            epilogue="y",
        )],
    )
    state = State(player_location="loc_a")  # decision beat not fired
    prose_part = ProsePart(prose="...", next=EndGame(ending_id="published_story"))
    with pytest.raises(ProsePartError, match="not yet satisfied"):
        validate_prose_part(prose_part, world, state)


def test_endgame_undefined_ending_rejected():
    world = _make_world()
    state = State(player_location="loc_a")
    prose_part = ProsePart(prose="...", next=EndGame(ending_id="ghost"))
    with pytest.raises(ProsePartError, match="not defined"):
        validate_prose_part(prose_part, world, state)


def test_continue_without_leave_option_rejected():
    world = _make_world()
    state = State(player_location="loc_a")
    prose_part = ProsePart(
        prose="...",
        next=ContinueScene(options=[
            Option(text="stay 1"), Option(text="stay 2"),
        ]),
    )
    with pytest.raises(ProsePartError, match="leaves_for"):
        validate_prose_part(prose_part, world, state)


def test_continue_with_leave_option_passes():
    world = _make_world()
    state = State(player_location="loc_a")
    prose_part = ProsePart(
        prose="...",
        next=ContinueScene(options=[
            Option(text="stay"),
            Option(text="leave for B", leaves_for="loc_b"),
        ]),
    )
    validate_prose_part(prose_part, world, state)


def test_continue_with_non_adjacent_leave_option_rejected():
    world = _make_world(
        locations=[
            Location(id="loc_a", name="A", adjacent_to=["loc_b"], description="a"),
            Location(id="loc_b", name="B", adjacent_to=["loc_a"], description="b"),
            Location(id="loc_c", name="C", adjacent_to=[], description="c"),
        ],
    )
    state = State(player_location="loc_a")
    prose_part = ProsePart(
        prose="...",
        next=ContinueScene(options=[
            Option(text="leave for C", leaves_for="loc_c"),  # not adjacent
        ]),
    )
    with pytest.raises(ProsePartError, match="not adjacent"):
        validate_prose_part(prose_part, world, state)


# ───────────────────────── leave-pick forces EndScene ─────────────────────────


def test_leave_pick_turn_forces_end_scene_continue_rejected():
    world = _make_world()
    state = State(player_location="loc_a")
    leave_pick = Option(text="leave", leaves_for="loc_b")
    prose_part = ProsePart(
        prose="...",
        next=ContinueScene(options=[Option(text="stay", leaves_for=None),
                                    Option(text="leave", leaves_for="loc_b")]),
    )
    with pytest.raises(ProsePartError, match="MUST emit EndScene"):
        validate_prose_part(prose_part, world, state, leave_pick=leave_pick)


def test_leave_pick_turn_forces_end_scene_endgame_rejected():
    world = _make_world(
        beats=[Beat(id="d", description="x", notes_to_llm="y")],
        endings=[Ending(id="e", requires_beats=["d"], notes_to_llm="x", epilogue="y")],
    )
    state = State(player_location="loc_a", unlocked_beats={"d"})
    leave_pick = Option(text="leave", leaves_for="loc_b")
    prose_part = ProsePart(prose="...", next=EndGame(ending_id="e"))
    with pytest.raises(ProsePartError, match="MUST emit EndScene"):
        validate_prose_part(prose_part, world, state, leave_pick=leave_pick)


def test_leave_pick_turn_with_end_scene_passes():
    world = _make_world()
    state = State(player_location="loc_a")
    leave_pick = Option(text="leave", leaves_for="loc_b")
    prose_part = ProsePart(prose="...", next=EndScene())
    validate_prose_part(prose_part, world, state, leave_pick=leave_pick)


def test_normal_turn_no_leave_pick_continue_passes():
    """Sanity: when leave_pick is None, ContinueScene with a leave option
    is the normal-case path."""
    world = _make_world()
    state = State(player_location="loc_a")
    prose_part = ProsePart(
        prose="...",
        next=ContinueScene(options=[
            Option(text="stay"), Option(text="leave", leaves_for="loc_b"),
        ]),
    )
    validate_prose_part(prose_part, world, state, leave_pick=None)


# ───────────────────────── integration: valid fixture ─────────────────────────


def test_validators_play_with_loaded_fixture():
    """Sanity check: validators don't reject valid changes against the
    fixture world."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    # tom_named_a_vehicle has no requires, and its blocker (tom_closes_off)
    # hasn't fired — so it's firable.
    bundle = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")]
    )
    validate_state_change_bundle(bundle, world, state)
    apply_state_changes(bundle, state)
    assert "tom_named_a_vehicle" in state.unlocked_beats
    # tom_admits_more requires tom_named_a_vehicle — now satisfied.
    bundle = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_admits_more")]
    )
    validate_state_change_bundle(bundle, world, state)
