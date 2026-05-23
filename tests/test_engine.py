"""Engine helpers for the quality engine: confession bands, `when`
evaluation, delta application + milestone derivation, emittable beats,
ending resolution, and the per-call dynamic bundle (Literal narrowing).
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from handler.agents import _build_state_change_bundle_type
from handler.engine import (
    FREE_TEXT_COOLDOWN,
    FREE_TEXT_FALLBACK,
    FREE_TEXT_OFFER_P,
    _trust_adjustable_here,
    _turns_since_typed,
    apply_state_changes,
    apply_village_attention,
    band_name,
    confession_depth,
    emittable_beat_ids,
    free_text_cadence_ok,
    resolved_ending_id_at_close,
    village_ejected,
    when_satisfied,
)
from handler.models import Beat, Brief, Disposition, Ending, Location, NPC, When
from handler.schemas import (
    ChoiceBundle,
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


def test_apply_village_attention_ratchets_up_only():
    """village_attention is set by the dedicated scene-close agent via
    apply_village_attention — it climbs to the verdict level and never drops."""
    world = _make_world(npcs=[_npc("tom", "loc_a")])
    s = _state(world)
    assert s.village_attention == 0
    assert apply_village_attention(s, "talked_about") == 2
    assert apply_village_attention(s, "none") == 2      # never lowers
    assert apply_village_attention(s, "noticed") == 2   # nor a lower verdict
    assert apply_village_attention(s, "hostile") == 3   # but climbs further


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
        "npc_deltas": [{"npc_id": "tom", "trust": "+1"}],
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
        "commitments": [{"type": "unlock_beat", "beat_name": "Read letter"}],
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
    assert free_text_cadence_ok(_state_with_history([True]), random.Random(0)) is False


def test_cadence_probability_is_seedable_and_honored_past_cooldown():
    state = _state_with_history([False, False, False])

    class _Stub:
        def __init__(self, val): self.val = val
        def random(self): return self.val

    assert free_text_cadence_ok(state, _Stub(FREE_TEXT_OFFER_P - 0.01)) is True
    assert free_text_cadence_ok(state, _Stub(FREE_TEXT_OFFER_P + 0.01)) is False


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
