"""Engine helpers for the quality engine: confession bands, `when`
evaluation, delta application + milestone derivation, emittable beats,
ending resolution, and the per-call dynamic bundle (Literal narrowing).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from handler.agents import _build_state_change_bundle_type
from handler.engine import (
    FREE_TEXT_COOLDOWN,
    FREE_TEXT_FALLBACK,
    SCENE_FINDABLE_GRACE_TURNS,
    _trust_adjustable_here,
    _turns_since_typed,
    apply_state_changes,
    apply_village_attention,
    band_name,
    confession_depth,
    effective_deltas,
    emittable_beat_ids,
    free_text_cadence_ok,
    has_live_findable,
    present_npc_ids,
    resolved_ending_id_at_close,
    scene_is_spent,
    seed_present_npcs,
    village_ejected,
    when_satisfied,
)
from handler.models import Beat, Brief, Disposition, Ending, Location, NPC, When
from handler.prompts import (
    build_choice_author_user_message,
    build_prose_writer_user_message,
)
from handler.schemas import (
    Continue,
    FreeText,
    Option,
    QualityDelta,
    StateChangeBundle,
    UnlockBeat,
)
from handler.state import State, TurnRecord, make_initial_state
from handler.world import World, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


def _beat(id: str, **kwargs) -> Beat:
    kwargs.setdefault("name", id.replace("_", " ").capitalize())
    kwargs.setdefault("description", "x")
    kwargs.setdefault("notes_to_llm", "y")
    return Beat(id=id, **kwargs)


def _npc(id: str, default_location: str, disposition: Disposition | None = None) -> NPC:
    return NPC(
        id=id, name=id.capitalize(), default_location=default_location,
        knowledge="x", attitude="y", disposition=disposition,
    )


def _make_world(
    *,
    locations: list[Location] | None = None,
    npcs: list[NPC] | None = None,
    beats: list[Beat] | None = None,
    endings: list[Ending] | None = None,
    starts_at: str = "loc_a",
) -> World:
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


def _state(world: World, location: str = "loc_a", **kwargs) -> State:
    """A State with trust/guard seeded for every NPC (mirrors make_initial_state)."""
    s = State(player_location=location, **kwargs)
    s.trust = {n: 0 for n in world.npcs}
    s.guard = {n: 0 for n in world.npcs}
    return s


# ───────────────────────── confession bands ─────────────────────────


def test_confession_depth_default_width():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    assert confession_depth(world, s, "tom") == 0
    s.trust["tom"] = 2
    assert confession_depth(world, s, "tom") == 2  # default width 1 → depth=trust
    assert band_name(world, s, "tom") == "open"


def test_confession_depth_slow_width():
    tom = _npc("tom", "loc_a", disposition=Disposition(warms="slow"))
    world = _make_world(npcs=[tom])
    s = _state(world)
    s.trust["tom"] = 2  # slow width 2 → depth 1
    assert confession_depth(world, s, "tom") == 1
    assert band_name(world, s, "tom") == "thawing"
    s.trust["tom"] = 4  # depth 2 → open
    assert confession_depth(world, s, "tom") == 2


def test_guard_lowers_effective_warmth():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    s.trust["tom"] = 3
    s.guard["tom"] = 2
    assert confession_depth(world, s, "tom") == 1  # effective 3-2=1


def test_breach_returns_lost_depth_and_closed_band():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    s.trust["tom"] = 3
    s.guard_breached.add("tom")
    assert confession_depth(world, s, "tom") == -1
    assert band_name(world, s, "tom") == "closed"


# ───────────────────────── when_satisfied ─────────────────────────


def test_when_npc_trust_at_least():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    w = When(npc="tom", trust_at_least="open")
    assert not when_satisfied(world, s, w)
    s.trust["tom"] = 2
    assert when_satisfied(world, s, w)


def test_when_any_evidence_of():
    world = _make_world(beats=[_beat("m", when=When(npc="tom", trust_at_least="open"))])
    s = _state(world)
    w = When(any_evidence_of=["m"])
    assert not when_satisfied(world, s, w)
    s.unlocked_beats.add("m")
    assert when_satisfied(world, s, w)


def test_when_npcs_thawed_at_least():
    world = _make_world(npcs=[_npc("a", "loc_a"), _npc("b", "loc_a"), _npc("c", "loc_a")])
    s = _state(world)
    w = When(npcs_thawed_at_least=2)
    assert not when_satisfied(world, s, w)
    s.trust["a"] = 1  # thawing (depth ≥ 1)
    s.trust["b"] = 1
    assert when_satisfied(world, s, w)


def test_when_attention_bands():
    world = _make_world()
    s = _state(world)
    s.village_attention = 1  # noticed
    assert when_satisfied(world, s, When(village_attention_at_least="noticed"))
    assert not when_satisfied(world, s, When(village_attention_at_least="hostile"))
    assert when_satisfied(world, s, When(village_attention_below="hostile"))
    s.village_attention = 3  # hostile
    assert not when_satisfied(world, s, When(village_attention_below="hostile"))


def test_village_ejected_at_hostile():
    s = State(player_location="loc_a")
    assert not village_ejected(s)
    s.village_attention = 3
    assert village_ejected(s)


# ───────────────────────── _trust_adjustable_here ─────────────────────────


def test_trust_not_adjustable_when_breached():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    s.guard_breached.add("tom")
    assert not _trust_adjustable_here(world, s, "tom")


def test_trust_pinned_to_opens_up_only_at():
    dylan = _npc("dylan", "loc_a", disposition=Disposition(opens_up_only_at=["loc_b"]))
    world = _make_world(npcs=[dylan])
    assert not _trust_adjustable_here(world, _state(world, "loc_a"), "dylan")
    assert _trust_adjustable_here(world, _state(world, "loc_b"), "dylan")


# ───────────────────────── apply_state_changes ─────────────────────────


def test_apply_quality_delta_clamped():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    bundle = StateChangeBundle(deltas=[
        QualityDelta(npc_id="tom", trust="+1", guard="+1"),
    ])
    apply_state_changes(bundle, s, world)
    assert s.trust["tom"] == 1
    assert s.guard["tom"] == 1


def test_apply_village_attention_climbs_fast_cools_slow():
    """village_attention is set by the dedicated scene-close agent via
    apply_village_attention — it jumps UP to a verdict at/above the current heat
    (suspicion spreads fast) but only steps DOWN one band per calmer scene (the
    village cools slowly), so a pushy stretch is recoverable but not instantly."""
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    assert s.village_attention == 0
    assert apply_village_attention(s, "talked_about") == 2  # jumps up to verdict
    assert apply_village_attention(s, "none") == 1          # cools ONE band, not to 0
    assert apply_village_attention(s, "none") == 0          # another calm scene → 0
    assert apply_village_attention(s, "hostile") == 3       # spreads fast again
    assert apply_village_attention(s, "noticed") == 2       # then eases one band


def test_apply_breach_latches_guard():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    apply_state_changes(
        StateChangeBundle(deltas=[QualityDelta(npc_id="tom", guard="BREACH")]), s, world
    )
    assert "tom" in s.guard_breached
    # Further deltas are ignored once breached.
    apply_state_changes(
        StateChangeBundle(deltas=[QualityDelta(npc_id="tom", trust="+1")]), s, world
    )
    assert s.trust["tom"] == 0


def test_apply_positive_trust_pinned_to_location():
    dylan = _npc("dylan", "loc_a", disposition=Disposition(opens_up_only_at=["loc_b"]))
    world = _make_world(npcs=[dylan])
    s = _state(world, "loc_a")  # not where he opens up
    apply_state_changes(
        StateChangeBundle(deltas=[QualityDelta(npc_id="dylan", trust="+1")]), s, world
    )
    assert s.trust["dylan"] == 0  # positive trust dropped here
    # Guard still applies anywhere.
    apply_state_changes(
        StateChangeBundle(deltas=[QualityDelta(npc_id="dylan", guard="+1")]), s, world
    )
    assert s.guard["dylan"] == 1


def test_apply_derives_milestone_when_trust_crosses():
    world = _make_world(
        npcs=[_npc("tom", "loc_a")],
        beats=[_beat("tom_named", when=When(npc="tom", trust_at_least="open"))],
    )
    s = _state(world)
    s.trust["tom"] = 1
    newly = apply_state_changes(
        StateChangeBundle(deltas=[QualityDelta(npc_id="tom", trust="+1")]), s, world
    )
    assert newly == ["tom_named"]
    assert "tom_named" in s.unlocked_beats


def test_apply_commitment_then_derive_in_order():
    world = _make_world(
        npcs=[_npc("tom", "loc_a")],
        beats=[
            _beat("tom_named", when=When(npc="tom", trust_at_least="open")),
            _beat("decided_to_file", commitment=True,
                  when=When(any_evidence_of=["tom_named"])),
        ],
    )
    s = _state(world)
    s.trust["tom"] = 2  # already open → tom_named will derive
    s.unlocked_beats.add("tom_named")  # evidence present so the commitment is valid
    newly = apply_state_changes(
        StateChangeBundle(commitments=[UnlockBeat(beat_id="decided_to_file")]), s, world
    )
    assert "decided_to_file" in newly
    assert "decided_to_file" in s.unlocked_beats


def test_apply_empty_bundle_is_noop():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    assert apply_state_changes(StateChangeBundle(), s, world) == []


# ───────────────────────── emittable_beat_ids ─────────────────────────


def test_emittable_observation_milestone_location_gated():
    world = _make_world(beats=[
        _beat("read_letter", commitment=True, requires_location=["loc_b"]),
    ])
    assert "read_letter" not in emittable_beat_ids(world, _state(world, "loc_a"))
    assert "read_letter" in emittable_beat_ids(world, _state(world, "loc_b"))


def test_emittable_decision_requires_when():
    world = _make_world(beats=[
        _beat("m", when=When(npc="tom", trust_at_least="open")),
        _beat("decided_to_file", commitment=True, when=When(any_evidence_of=["m"])),
    ])
    s = _state(world)
    assert "decided_to_file" not in emittable_beat_ids(world, s)  # no evidence yet
    s.unlocked_beats.add("m")
    assert "decided_to_file" in emittable_beat_ids(world, s)


def test_emittable_derived_milestones_never_emittable():
    world = _make_world(beats=[_beat("m", when=When(npc="tom", trust_at_least="open"))])
    # Not a commitment → never in the emittable (LLM-fired) set.
    assert "m" not in emittable_beat_ids(world, _state(world))


def test_emittable_decision_mutual_exclusion():
    world = _make_world(beats=[
        _beat("m", when=When(npc="tom", trust_at_least="open")),
        _beat("decided_to_file", commitment=True, when=When(any_evidence_of=["m"])),
        _beat("decided_to_walk_away", commitment=True, when=When(any_evidence_of=["m"])),
    ])
    s = _state(world, unlocked_beats={"m", "decided_to_file"})
    # one decision taken → the other is no longer emittable
    assert "decided_to_walk_away" not in emittable_beat_ids(world, s)


def test_emittable_npc_findable_requires_presence_and_trust():
    """A person-anchored findable (commitment + when.npc) is pursuable only when
    the NPC is BOTH present and trust-gated — the unified findable model."""
    tom = _npc("tom", "loc_a")
    world = _make_world(
        npcs=[tom],
        beats=[_beat("tom_tells", commitment=True,
                     when=When(npc="tom", trust_at_least="open"))],
    )
    # present (loc_a) + open (depth 2 at default width) → emittable
    s_here_open = _state(world, "loc_a")
    s_here_open.trust["tom"] = 2
    assert "tom_tells" in emittable_beat_ids(world, s_here_open)
    # present but below the band → trust gate withholds it
    s_here_cold = _state(world, "loc_a")
    assert "tom_tells" not in emittable_beat_ids(world, s_here_cold)
    # open but player elsewhere (Tom absent) → presence gate withholds it
    s_away_open = _state(world, "loc_b")
    s_away_open.trust["tom"] = 2
    assert "tom_tells" not in emittable_beat_ids(world, s_away_open)


# ───────────────────────── scene_is_spent (grace window) ─────────────────────────


def _dry_turn() -> TurnRecord:
    return TurnRecord(prose="p", applied_state_changes=[], applied_deltas=[])


def _moving_turn() -> TurnRecord:
    return TurnRecord(prose="p", applied_state_changes=[],
                      applied_deltas=[QualityDelta(npc_id="tom", trust="+1")])


def test_scene_not_spent_within_grace_with_findable():
    """An un-pursued findable keeps the scene alive during the opening grace
    window — the room gets a few turns to breathe."""
    world = _make_world(beats=[
        _beat("look", commitment=True, requires_location=["loc_a"]),
    ])
    s = _state(world, "loc_a")
    s.dialogue_history = [_dry_turn() for _ in range(2)]
    assert len(s.dialogue_history) <= SCENE_FINDABLE_GRACE_TURNS
    assert scene_is_spent(world, s) is False


def test_scene_spent_past_grace_on_dry_turn_despite_findable():
    """Past the grace window, a still-available but un-pursued findable no
    longer immortalises the room — a dry turn closes it (the post-office /
    vicarage 'circles forever' tail)."""
    world = _make_world(beats=[
        _beat("look", commitment=True, requires_location=["loc_a"]),
    ])
    s = _state(world, "loc_a")
    s.dialogue_history = [_dry_turn() for _ in range(SCENE_FINDABLE_GRACE_TURNS + 1)]
    assert scene_is_spent(world, s) is True


def test_scene_not_spent_past_grace_when_last_turn_moved():
    """Movement always holds a scene open: even past grace, a turn that shifted
    standing is not a dry turn, so a productive conversation never auto-closes."""
    world = _make_world(npcs=[_npc("tom", "loc_a")], beats=[
        _beat("look", commitment=True, requires_location=["loc_a"]),
    ])
    s = _state(world, "loc_a")
    s.dialogue_history = [_dry_turn() for _ in range(SCENE_FINDABLE_GRACE_TURNS)]
    s.dialogue_history.append(_moving_turn())
    assert scene_is_spent(world, s) is False


def test_scene_spent_dead_room_past_grace_no_findable():
    """A room with nothing live and a dry last turn is spent past grace."""
    world = _make_world()
    s = _state(world, "loc_a")
    s.dialogue_history = [_dry_turn() for _ in range(SCENE_FINDABLE_GRACE_TURNS + 1)]
    assert scene_is_spent(world, s) is True


# ───────────────────────── resolved_ending_id_at_close ─────────────────────────


def test_resolved_ending_requires_beats():
    world = _make_world(
        beats=[_beat("decided_to_file", commitment=True,
                     when=When(any_evidence_of=[]))],
        endings=[Ending(id="published", requires_beats=["decided_to_file"],
                        notes_to_llm="x", epilogue="y")],
    )
    s = _state(world)
    assert resolved_ending_id_at_close(world, s) is None
    s.unlocked_beats.add("decided_to_file")
    assert resolved_ending_id_at_close(world, s) == "published"


def test_resolved_ending_quality_gate():
    world = _make_world(
        npcs=[_npc("iwan", "loc_a")],
        beats=[_beat("decided_to_tell", commitment=True, when=When(any_evidence_of=[]))],
        endings=[Ending(
            id="iwan_peace", requires_beats=["decided_to_tell"],
            when=When(npc="iwan", trust_at_least="open"), notes_to_llm="x", epilogue="y",
        )],
    )
    s = _state(world, unlocked_beats={"decided_to_tell"})
    assert resolved_ending_id_at_close(world, s) is None  # iwan not open
    s.trust["iwan"] = 2
    assert resolved_ending_id_at_close(world, s) == "iwan_peace"


def test_resolved_ending_run_out_precedence():
    world = _make_world(
        beats=[_beat("decided_to_file", commitment=True, when=When(any_evidence_of=[]))],
        endings=[
            Ending(id="run_out", when=When(village_attention_at_least="hostile"),
                   notes_to_llm="x", epilogue="y"),
            Ending(id="published", requires_beats=["decided_to_file"],
                   notes_to_llm="x", epilogue="y"),
        ],
    )
    s = _state(world, unlocked_beats={"decided_to_file"})
    s.village_attention = 3
    # both satisfied → run_out wins (authored first)
    assert resolved_ending_id_at_close(world, s) == "run_out"


def test_resolved_ending_ejection_outranks_regardless_of_author_order():
    """Ejection precedence is enforced by the engine (the hostile-gated ending
    wins), NOT by YAML order — so authoring the decision ending FIRST must not
    flip which ending fires when the player is both ejected and decided."""
    world = _make_world(
        beats=[_beat("decided_to_file", commitment=True, when=When(any_evidence_of=[]))],
        endings=[
            Ending(id="published", requires_beats=["decided_to_file"],
                   notes_to_llm="x", epilogue="y"),
            Ending(id="run_out", when=When(village_attention_at_least="hostile"),
                   notes_to_llm="x", epilogue="y"),
        ],
    )
    s = _state(world, unlocked_beats={"decided_to_file"})
    s.village_attention = 3
    assert resolved_ending_id_at_close(world, s) == "run_out"
    # Not ejected → the decision ending fires as before.
    s.village_attention = 0
    assert resolved_ending_id_at_close(world, s) == "published"


def test_resolved_ending_none_when_no_endings():
    assert resolved_ending_id_at_close(_make_world(endings=[]), _state(_make_world())) is None


# ───────────────────────── dynamic StateChangeBundle (Literal narrowing) ─────────────────────────


def _bundle_world():
    return _make_world(
        npcs=[_npc("tom", "loc_a"), _npc("maud", "loc_b")],
        beats=[
            _beat("read_letter", commitment=True, requires_location=["loc_a"]),
        ],
    )


def test_dynamic_bundle_accepts_present_npc_quality_delta():
    world = _bundle_world()
    Bundle = _build_state_change_bundle_type(world, _state(world, "loc_a"))
    b = Bundle.model_validate({
        "npc_deltas": [{"npc_id": "tom", "trust": "+1", "reason": "warmed"}],
        "commitments": [],
    })
    assert b.npc_deltas[0].npc_id == "tom"  # type: ignore[union-attr]


def test_dynamic_bundle_rejects_absent_npc():
    world = _bundle_world()
    Bundle = _build_state_change_bundle_type(world, _state(world, "loc_a"))
    with pytest.raises(ValidationError):
        Bundle.model_validate({
            "npc_deltas": [{"npc_id": "maud", "trust": "+1"}],
        })


def test_dynamic_bundle_rejects_out_of_range_trust():
    world = _bundle_world()
    Bundle = _build_state_change_bundle_type(world, _state(world, "loc_a"))
    with pytest.raises(ValidationError):
        Bundle.model_validate({
            "npc_deltas": [{"npc_id": "tom", "trust": "+2"}],
        })


def test_dynamic_bundle_accepts_emittable_commitment():
    world = _bundle_world()
    Bundle = _build_state_change_bundle_type(world, _state(world, "loc_a"))
    b = Bundle.model_validate({
        "commitments": [
            {"type": "unlock_beat", "beat_name": "Read letter", "warrant": "I read the letter"}
        ],
    })
    assert b.commitments[0].beat_name == "Read letter"  # type: ignore[union-attr]


def test_dynamic_bundle_rejects_non_emittable_commitment():
    world = _bundle_world()
    # at loc_b the location-gated observation milestone is not emittable
    Bundle = _build_state_change_bundle_type(world, _state(world, "loc_b"))
    with pytest.raises(ValidationError):
        Bundle.model_validate({
            "commitments": [{"type": "unlock_beat", "beat_name": "Read letter"}],
        })


def test_dynamic_bundle_commitment_max_one():
    world = _make_world(
        npcs=[_npc("tom", "loc_a")],
        beats=[
            _beat("read_letter", commitment=True, requires_location=["loc_a"]),
            _beat("read_again", commitment=True, requires_location=["loc_a"]),
        ],
    )
    Bundle = _build_state_change_bundle_type(world, _state(world, "loc_a"))
    with pytest.raises(ValidationError):
        Bundle.model_validate({
            "commitments": [
                {"type": "unlock_beat", "beat_name": "Read letter"},
                {"type": "unlock_beat", "beat_name": "Read again"},
            ],
        })


# ───────────────────────── canonical Continue validator ─────────────────────────


def test_continue_with_two_options_passes():
    Continue(options=[Option(text="stay"), Option(text="ask")])


def test_continue_with_one_option_rejected():
    with pytest.raises(ValidationError):
        Continue(options=[Option(text="only one")])


def test_continue_with_five_options_rejected():
    with pytest.raises(ValidationError):
        Continue(options=[Option(text=t) for t in "abcde"])


# ───────────────────────── free-text cadence ─────────────────────────


def _state_with_history(typed_flags: list[bool]) -> State:
    state = State(player_location="x")
    for typed in typed_flags:
        state.dialogue_history.append(
            TurnRecord(prose="p", applied_state_changes=[], player_pick_typed=typed)
        )
    return state


def test_turns_since_typed_counts_back_to_last_typed_move():
    assert _turns_since_typed(_state_with_history([True, False, False])) == 2
    assert _turns_since_typed(_state_with_history([False, True])) == 0


def test_turns_since_typed_large_when_never_typed():
    assert _turns_since_typed(_state_with_history([False, False])) > FREE_TEXT_COOLDOWN


def test_cadence_suppresses_within_cooldown():
    # Last move was typed one turn ago — inside the cooldown, so suppressed.
    assert free_text_cadence_ok(_state_with_history([True])) is False


def test_cadence_allows_past_cooldown():
    # No recent typed move — the cooldown is clear, so an offer is honored.
    assert free_text_cadence_ok(_state_with_history([False, False, False])) is True


def test_cadence_is_cooldown_only_and_deterministic():
    # No probability, no rng: the same state always yields the same answer.
    state = _state_with_history([False, False, False])
    assert free_text_cadence_ok(state) is free_text_cadence_ok(state) is True


def test_free_text_fallback_is_general_agency():
    assert isinstance(FREE_TEXT_FALLBACK, FreeText)
    assert FREE_TEXT_FALLBACK.npc_id is None
    assert FREE_TEXT_FALLBACK.suggested is None


# ───────────────────────── integration: valid fixture ─────────────────────────


def test_quality_engine_against_loaded_fixture():
    """End-to-end against the fixture world: warm Tom to `open` → the engine
    derives tom_named_a_vehicle; then the player can commit to filing."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"  # Tom present
    # Warm Tom across turns to the open band.
    for _ in range(2):
        apply_state_changes(
            StateChangeBundle(deltas=[QualityDelta(npc_id="tom", trust="+1")]),
            state, world,
        )
    assert "tom_named_a_vehicle" in state.unlocked_beats
    # Now the file decision is emittable.
    assert "decided_to_file" in emittable_beat_ids(world, state)
    apply_state_changes(
        StateChangeBundle(commitments=[UnlockBeat(beat_id="decided_to_file")]),
        state, world,
    )
    assert resolved_ending_id_at_close(world, state) == "published_story"


# ───────────────────────── findable surfacing in prompts ─────────────────────────


def test_prose_surfaces_present_findable_as_set_dressing():
    """A live location findable is surfaced (by name) in the prose writer's
    '## Present In The Scene' block, and drops out once unlocked."""
    world = _make_world(beats=[
        _beat("read_letter", name="Read the letter",
              commitment=True, requires_location=["loc_a"]),
    ])
    s = _state(world, "loc_a")
    msg = build_prose_writer_user_message(s, world, [], is_first_turn=True)
    assert "## Present In The Scene" in msg
    assert "Read the letter" in msg
    s.unlocked_beats.add("read_letter")
    msg2 = build_prose_writer_user_message(s, world, [], is_first_turn=True)
    assert "## Present In The Scene" not in msg2


def test_prose_does_not_set_dress_npc_findable():
    """Person-anchored findables are pursued in conversation, not pre-planted as
    scenery — they must not appear in the prose '## Present In The Scene' block."""
    tom = _npc("tom", "loc_a")
    world = _make_world(npcs=[tom], beats=[
        _beat("tom_tells", name="Tom tells", commitment=True,
              when=When(npc="tom", trust_at_least="open")),
    ])
    s = _state(world, "loc_a")
    s.trust["tom"] = 2
    msg = build_prose_writer_user_message(s, world, [], is_first_turn=True)
    assert "## Present In The Scene" not in msg


def test_choice_author_worth_pursuing_names_present_npc_findable():
    """The choice author's '## Worth Pursuing Here' steer surfaces a present,
    open NPC's findable spoiler-free (the person, not the payoff)."""
    tom = _npc("tom", "loc_a")
    world = _make_world(npcs=[tom], beats=[
        _beat("tom_tells", name="Tom tells", commitment=True,
              when=When(npc="tom", trust_at_least="open")),
    ])
    s = _state(world, "loc_a")
    s.trust["tom"] = 2
    msg = build_choice_author_user_message(s, world, prose="The bar is quiet.")
    assert "## Worth Pursuing Here" in msg
    assert "Tom" in msg


# ───────────────────── effective deltas (realized changes) ─────────────────────


def _world_dylan() -> World:
    """A world where dylan opens up only at loc_b (NOT his default loc_a) and
    carries a person-anchored findable gated on his trust — the Dylan trap in
    miniature."""
    dylan = _npc("dylan", "loc_a", disposition=Disposition(opens_up_only_at=["loc_b"]))
    beat = _beat(
        "dylan_admits", name="Dylan admits", commitment=True,
        when=When(npc="dylan", trust_at_least="open"),
    )
    return _make_world(npcs=[dylan], beats=[beat])


def _apply_and_realize(world, s, bundle):
    tb, gb, bb = dict(s.trust), dict(s.guard), set(s.guard_breached)
    apply_state_changes(bundle, s, world)
    return effective_deltas(
        bundle, s, trust_before=tb, guard_before=gb, breached_before=bb
    )


def test_effective_deltas_zeroed_off_opens_location():
    world = _world_dylan()
    s = _state(world, location="loc_a")  # not where dylan warms
    bundle = StateChangeBundle(deltas=[QualityDelta(npc_id="dylan", trust="+1", reason="r")])
    eff = _apply_and_realize(world, s, bundle)
    assert eff[0].trust == "0"      # warmth dropped at a non-opens-up location
    assert s.trust["dylan"] == 0


def test_effective_deltas_applied_on_opens_location():
    world = _world_dylan()
    s = _state(world, location="loc_b")  # dylan opens up here
    bundle = StateChangeBundle(deltas=[QualityDelta(npc_id="dylan", trust="+1", reason="r")])
    eff = _apply_and_realize(world, s, bundle)
    assert eff[0].trust == "+1"
    assert s.trust["dylan"] == 1


def test_effective_deltas_clamp_saturation_reads_zero():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    s.trust["tom"] = 8  # already at TRUST_MAX
    bundle = StateChangeBundle(deltas=[QualityDelta(npc_id="tom", trust="+1")])
    eff = _apply_and_realize(world, s, bundle)
    assert eff[0].trust == "0"      # saturated → nothing actually applied


def test_effective_deltas_breach_records_breach():
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    bundle = StateChangeBundle(deltas=[QualityDelta(npc_id="tom", guard="BREACH")])
    eff = _apply_and_realize(world, s, bundle)
    assert eff[0].guard == "BREACH"


# ───────────────────── seed_present_npcs (reachability) ─────────────────────


def test_seed_present_npcs_places_dylan_where_warmable():
    world = _world_dylan()
    s = make_initial_state(world)  # dylan at default loc_a
    s.player_location = "loc_b"    # an opens-up spot, findable still live
    assert seed_present_npcs(world, s) == ["dylan"]
    assert s.npc_locations["dylan"] == "loc_b"


def test_seed_present_npcs_skips_non_opens_location():
    world = _world_dylan()
    s = make_initial_state(world)
    s.player_location = "loc_a"    # default, but not an opens-up spot
    assert seed_present_npcs(world, s) == []


def test_seed_present_npcs_skips_when_findable_done():
    world = _world_dylan()
    s = make_initial_state(world)
    s.player_location = "loc_b"
    s.unlocked_beats.add("dylan_admits")
    assert not has_live_findable(world, s, "dylan")
    assert seed_present_npcs(world, s) == []


def test_seed_present_npcs_skips_when_breached():
    world = _world_dylan()
    s = make_initial_state(world)
    s.player_location = "loc_b"
    s.guard_breached.add("dylan")
    assert seed_present_npcs(world, s) == []


# ───────────────────── not-here steer + bible reachability ─────────────────────


def test_choice_author_steers_away_from_dead_location():
    """A present NPC who can't warm HERE gets a 'won't talk here / somewhere
    quieter' steer instead of another futile press."""
    world = _world_dylan()
    s = _state(world, location="loc_a")  # dylan present (default) but unwarmable here
    msg = build_choice_author_user_message(s, world, prose="Dylan looks away.")
    assert "## They Won't Talk Here" in msg


def test_choice_author_no_dead_location_steer_where_warmable():
    world = _world_dylan()
    s = _state(world, location="loc_b")
    s.npc_locations = {"dylan": "loc_b"}  # present where he opens up
    msg = build_choice_author_user_message(s, world, prose="Dylan exhales.")
    assert "## They Won't Talk Here" not in msg


REAL_WORLD = Path(__file__).parent.parent / "world"


def test_every_person_findable_has_a_warmable_present_spot():
    """Regression for the Dylan trap, on the real bible: every person-anchored
    findable must have at least one location where its NPC is BOTH present
    (default or engine-seeded) AND warmable (`_trust_adjustable_here`). Before
    `seed_present_npcs`, dylan_admits_he_knows failed — Dylan was present only at
    Tan-y-coed (where he won't warm) and warmable only where the populator rarely
    placed him."""
    world = load_world(REAL_WORLD)
    person_beats = {
        bid: b.when.npc
        for bid, b in world.beats.items()
        if b.commitment and not bid.startswith("decided_")
        and b.when is not None and b.when.npc is not None
    }
    assert person_beats, "expected person-anchored findables in the bible"
    for bid, nid in person_beats.items():
        reachable = False
        for loc in world.locations:
            s = make_initial_state(world)
            s.player_location = loc
            seed_present_npcs(world, s)
            if nid in present_npc_ids(world, s) and _trust_adjustable_here(world, s, nid):
                reachable = True
                break
        assert reachable, f"{bid}: NPC {nid!r} is never both present and warmable"
