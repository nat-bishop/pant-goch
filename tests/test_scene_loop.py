"""Scene-loop orchestration tests.

Mocks emit_state_changes / emit_prose_text / emit_choices / emit_free_text_gate
/ emit_caption (and the engine cadence gate) to avoid LLM dependency. Verifies
that run_turn sequences phase 1 → apply → prose writer → choice author ‖ gate →
(cadence + caption) → assemble ProsePart → append, that fallbacks fire on
UnexpectedModelBehavior, and that play_scene drives Continue picks and FreeText
typed input until termination.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from handler.agents import (
    build_scene_runner_bundle,
    make_chronicler_runtime,
    make_scene_runner_runtime,
)
from handler.engine import FREE_TEXT_FALLBACK, enter_scene
from handler.schemas import (
    ChoiceBundle,
    ChronicleEntry,
    Continue,
    EndGame,
    EndScene,
    FreeText,
    FreeTextCaption,
    Leave,
    NpcPlacementBundle,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from handler.scene_loop import (
    _assemble_prose_part,
    play_scene,
    play_session,
    run_turn,
)
from handler.state import State, make_initial_state
from handler.world import World, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── populator / chronicler no-ops ─────────────────────────


async def _empty_placements(*args, **kwargs):
    """Default populator stub for tests that don't care about NPC
    placement. Returns an empty bundle (defaults stand)."""
    return NpcPlacementBundle()


def _patch_populator(monkeypatch):
    monkeypatch.setattr("handler.scene_loop.emit_placements", _empty_placements)


def _patch_chronicler(monkeypatch, text: str = "entry"):
    async def fake_emit(*args, **kwargs):
        return ChronicleEntry(
            case_file_entry=text,
            npc_memory_updates=[],
        )
    monkeypatch.setattr("handler.scene_loop.emit_chronicle_entry", fake_emit)


def _patch_phase1(monkeypatch, bundle: StateChangeBundle | None = None):
    """Default Phase 1 stub: returns the supplied bundle (or empty)."""
    payload = bundle or StateChangeBundle()

    async def fake(*args, **kwargs):
        return payload

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake)


def _patch_prose_writer(monkeypatch, proses):
    """`proses` is an iterable of strings — one per expected call. The
    test should provide enough to cover all turns; if it runs out, the
    iterator raises StopIteration → the test fails loudly."""
    it = iter(proses)

    async def fake(*args, **kwargs):
        return next(it)

    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake)


def _patch_choice_author(monkeypatch, choices):
    """`choices` is an iterable of `ChoiceBundle` — one per expected call.
    Mirrors `_patch_prose_writer`."""
    it = iter(choices)

    async def fake(*args, **kwargs):
        return next(it)

    monkeypatch.setattr("handler.scene_loop.emit_choices", fake)


def _patch_gate(monkeypatch, offers):
    """`offers` is a bool (every turn) or an iterable of bools (one per
    expected call). Patches the free-text gate."""
    if isinstance(offers, bool):
        async def fake(*args, **kwargs):
            return offers
    else:
        it = iter(offers)

        async def fake(*args, **kwargs):
            return next(it)

    monkeypatch.setattr("handler.scene_loop.emit_free_text_gate", fake)


def _patch_caption(monkeypatch, captions):
    """`captions` is an iterable of `FreeTextCaption` — one per expected
    call. Mirrors `_patch_choice_author`. Use when the gate+cadence will
    surface a box; the helper also asserts (implicitly, via StopIteration)
    that it's called exactly as many times as captions supplied."""
    it = iter(captions)

    async def fake(*args, **kwargs):
        return next(it)

    monkeypatch.setattr("handler.scene_loop.emit_caption", fake)


def _patch_cadence(monkeypatch, ok: bool):
    """Force the engine-side free-text cadence gate to a fixed result,
    isolating run_turn/play_scene tests from the seeded probability draw."""
    monkeypatch.setattr(
        "handler.scene_loop.free_text_cadence_ok", lambda state, rng: ok
    )


@pytest.fixture(autouse=True)
def _default_gate_off(monkeypatch):
    """Default: the free-text gate offers nothing, so run_turn/play_scene
    tests get deterministic canned-option behavior without a real TestModel
    gate call. Tests exercising free text re-patch via `_patch_gate` (the
    later setattr in the test body wins)."""
    async def fake(*args, **kwargs):
        return False

    monkeypatch.setattr("handler.scene_loop.emit_free_text_gate", fake)


@pytest.fixture
def bundle_and_state():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")  # tom is here
    return bundle, runtime, state


# ───────────────────────── _assemble_prose_part ─────────────────────────


def test_assemble_continue_maps_to_continue_scene():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    choices = ChoiceBundle(next=Continue(options=[
        Option(text="a"), Option(text="b"),
    ]))
    pp = _assemble_prose_part(world, state, "prose", choices)
    assert isinstance(pp.next, Continue)
    assert len(pp.next.options) == 2


def test_assemble_leave_with_no_satisfied_ending_maps_to_end_scene():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    pp = _assemble_prose_part(
        world, state, "prose", ChoiceBundle(next=Leave()),
    )
    assert isinstance(pp.next, EndScene)


def test_assemble_leave_with_satisfied_ending_maps_to_end_game():
    """If the decision beat for an authored ending has fired, Leave
    resolves to EndGame with that ending id."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    ending = world.endings["published_story"]
    state.unlocked_beats = set(ending.requires_beats)
    pp = _assemble_prose_part(
        world, state, "prose", ChoiceBundle(next=Leave()),
    )
    assert isinstance(pp.next, EndGame)
    assert pp.next.ending_id == "published_story"


def test_assemble_free_text_offer_with_caption_builds_per_turn_free_text():
    """Offer + a caption → engine builds a per-turn FreeText carrying the
    caption author's caption and suggested move, dropping canned options."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    choices = ChoiceBundle(next=Continue(options=[
        Option(text="a"), Option(text="b"),
    ]))
    caption = FreeTextCaption(caption="What do you say to Tom?", suggested="I ask Tom about the keys.")
    pp = _assemble_prose_part(
        world, state, "prose", choices, free_text_offer=True, caption=caption,
    )
    assert isinstance(pp.next, FreeText)
    assert pp.next.prompt == "What do you say to Tom?"
    assert pp.next.suggested == "I ask Tom about the keys."
    assert pp.next.npc_id is None


def test_assemble_free_text_offer_without_caption_uses_fallback():
    """Offer stands but the caption author failed (caption=None) → engine
    falls back to the fixed-caption box with no suggestion."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    choices = ChoiceBundle(next=Continue(options=[
        Option(text="a"), Option(text="b"),
    ]))
    pp = _assemble_prose_part(world, state, "prose", choices, free_text_offer=True)
    assert pp.next == FREE_TEXT_FALLBACK
    assert pp.next.prompt == "What do you do next?"
    assert pp.next.suggested is None


def test_assemble_leave_beats_free_text_offer():
    """Precedence: Leave wins over a gate offer — a closing scene never
    gets a free-text box."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    pp = _assemble_prose_part(
        world, state, "prose", ChoiceBundle(next=Leave()), free_text_offer=True,
    )
    assert isinstance(pp.next, EndScene)


# ───────────────────────── run_turn happy path ─────────────────────────


async def test_run_turn_applies_state_changes_and_appends_record(monkeypatch, bundle_and_state):
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch, StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")],
    ))
    _patch_prose_writer(monkeypatch, [
        "Tom looks down at his pint, and then up at you.",
    ])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="say nothing"),
            Option(text="ask gently"),
        ])),
    ])

    result = await run_turn(bundle, runtime, state)

    assert isinstance(result.next, Continue)
    # State change applied.
    assert "tom_named_a_vehicle" in state.unlocked_beats
    # TurnRecord appended with prose + applied changes.
    assert len(state.dialogue_history) == 1
    record = state.dialogue_history[0]
    assert record.prose.startswith("Tom looks down")
    assert record.applied_state_changes[0].beat_id == "tom_named_a_vehicle"


async def test_run_turn_phase1_fallback_on_unexpected_behavior(monkeypatch, bundle_and_state):
    """Phase 1 retries exhausted → empty bundle, no state change applied."""
    bundle, runtime, state = bundle_and_state

    async def fake_phase1_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated retry exhaustion")

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1_fails)
    _patch_prose_writer(monkeypatch, ["Tom drinks."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="watch"), Option(text="go"),
        ])),
    ])

    result = await run_turn(bundle, runtime, state)
    assert state.unlocked_beats == set()
    assert isinstance(result.next, Continue)
    # TurnRecord still appended (no state changes).
    assert state.dialogue_history[-1].applied_state_changes == []


async def test_run_turn_prose_writer_fallback(monkeypatch, bundle_and_state):
    """Prose writer retries exhausted → engine-authored generic prose +
    choice author still runs to author options."""
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)

    async def fake_prose_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")

    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose_fails)
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="wait"), Option(text="look"),
        ])),
    ])

    result = await run_turn(bundle, runtime, state)
    # No engine metatext leaks into prose.
    assert "[engine" not in result.prose.lower()
    assert isinstance(result.next, Continue)


async def test_run_turn_prose_writer_fallback_does_not_pollute_message_history(
    monkeypatch, bundle_and_state
):
    """Failed prose-writer retries → fallback prose to player, but NOT
    appended to runtime.prose_writer_message_history."""
    bundle, runtime, state = bundle_and_state
    runtime.prose_writer_message_history = ["existing-anchor-marker"]  # type: ignore[list-item]

    _patch_phase1(monkeypatch)

    async def fake_prose_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")

    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose_fails)
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="a"), Option(text="b"),
        ])),
    ])

    await run_turn(bundle, runtime, state)
    assert runtime.prose_writer_message_history == ["existing-anchor-marker"], (
        "prose-writer fallback must not mutate prose_writer_message_history"
    )


async def test_run_turn_choice_author_fallback(monkeypatch, bundle_and_state):
    """Choice author retries exhausted → generic 2-option Continue."""
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The peat fire crackles."])

    async def fake_choices_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")

    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices_fails)

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, Continue)
    assert len(result.next.options) == 2


async def test_run_turn_leave_resolves_to_endgame_when_satisfied(monkeypatch, bundle_and_state):
    """Choice author Leave + satisfied ending in state → engine assembles
    EndGame with the resolved ending id."""
    bundle, runtime, state = bundle_and_state
    # Satisfy the published_story ending's prereqs.
    state.unlocked_beats = set(bundle.world.endings["published_story"].requires_beats)

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["A pause."])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, EndGame)
    assert result.next.ending_id == "published_story"


async def test_run_turn_leave_resolves_to_endscene_when_no_ending_satisfied(monkeypatch, bundle_and_state):
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["A pause."])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, EndScene)


async def test_run_turn_gate_offers_free_text_in_exploration(monkeypatch):
    """General agency: the gate can offer free text even with no NPC
    present (exploration). With cadence allowing, the caption author runs
    and its caption + suggestion ride the FreeText, overriding Continue."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_old_vicarage")  # solo location — no NPC

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The wood is silent."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_gate(monkeypatch, True)
    _patch_cadence(monkeypatch, True)
    _patch_caption(monkeypatch, [
        FreeTextCaption(caption="What do you do?", suggested="I search the desk."),
    ])

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, FreeText)
    assert result.next.prompt == "What do you do?"
    assert result.next.suggested == "I search the desk."
    assert result.next.npc_id is None


async def test_run_turn_cadence_suppresses_gate_offer(monkeypatch):
    """Gate offers but the engine cadence declines → no caption call, the
    choice author's Continue stands."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_old_vicarage")

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The wood is silent."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_gate(monkeypatch, True)
    _patch_cadence(monkeypatch, False)

    async def _fail_caption(*args, **kwargs):
        raise AssertionError("caption author must not be called when cadence declines")

    monkeypatch.setattr("handler.scene_loop.emit_caption", _fail_caption)

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, Continue)


async def test_run_turn_caption_failure_falls_back_to_fixed_box(monkeypatch):
    """Gate+cadence say offer but the caption author exhausts retries → the
    box still surfaces, with the fixed fallback caption and no suggestion."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_old_vicarage")

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The wood is silent."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_gate(monkeypatch, True)
    _patch_cadence(monkeypatch, True)

    async def _boom(*args, **kwargs):
        raise UnexpectedModelBehavior("retries exhausted")

    monkeypatch.setattr("handler.scene_loop.emit_caption", _boom)

    result = await run_turn(bundle, runtime, state)
    assert result.next == FREE_TEXT_FALLBACK


# ───────────────────────── play_scene ─────────────────────────


async def test_play_scene_drives_until_endscene(monkeypatch):
    """Three turns: turn 1+2 Continue, turn 3 Leave → EndScene."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2", "t3"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="stay"), Option(text="ask")])),
        ChoiceBundle(next=Continue(options=[Option(text="watch"), Option(text="speak")])),
        ChoiceBundle(next=Leave()),
    ])

    result = await play_scene(bundle, runtime, state, pick_option=lambda pp: 0)
    assert isinstance(result.next, EndScene)
    # Three TurnRecords.
    assert len(state.dialogue_history) == 3
    # Two Continue turns had a pick; the EndScene turn had none.
    assert state.dialogue_history[0].player_pick is not None
    assert state.dialogue_history[1].player_pick is not None
    assert state.dialogue_history[2].player_pick is None


async def test_play_scene_dispatches_free_text(monkeypatch):
    """Gate offers free text on turn 1 → play_scene calls pick_free_text →
    typed string becomes the turn's player_pick (flagged typed). Turn 2 the
    gate declines and the choice author emits Leave → EndScene."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2"])
    _patch_choice_author(monkeypatch, [
        # Turn 1 Continue is overridden by the gate offer; turn 2 Leave wins.
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
        ChoiceBundle(next=Leave()),
    ])
    _patch_gate(monkeypatch, [True, False])
    _patch_cadence(monkeypatch, True)
    _patch_caption(monkeypatch, [
        FreeTextCaption(caption="What do you say?", suggested="I tell him about Megan."),
    ])

    typed_replies = iter(["I came because of Megan."])

    def pick_free_text(req: FreeText) -> str:
        assert req.prompt == "What do you say?"
        assert req.suggested == "I tell him about Megan."
        assert req.npc_id is None
        return next(typed_replies)

    result = await play_scene(
        bundle, runtime, state,
        pick_option=lambda pp: 0,
        pick_free_text=pick_free_text,
    )
    assert isinstance(result.next, EndScene)
    # First TurnRecord got the typed string as the player's pick, flagged typed.
    assert state.dialogue_history[0].player_pick is not None
    assert state.dialogue_history[0].player_pick.text == "I came because of Megan."
    assert state.dialogue_history[0].player_pick_typed is True
    # The clicked-option turn (none here) would have player_pick_typed False;
    # the Leave turn has no pick.
    assert state.dialogue_history[1].player_pick is None


async def test_play_scene_resets_prose_writer_message_history_at_scene_start(monkeypatch):
    """enter_scene → runtime.prose_writer_message_history must reset to
    [] even if the previous scene left it populated."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    runtime.prose_writer_message_history = ["leftover-from-prior-scene"]  # type: ignore[list-item]
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)

    captured: list[list] = []

    async def fake_prose(bundle, rt, *args, **kwargs):
        # Observe message_history at the time of the first prose-writer
        # call.
        captured.append(list(rt.prose_writer_message_history))
        return "end"

    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose)
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    await play_scene(bundle, runtime, state, pick_option=lambda pp: 0)
    # The first observed prose-writer call MUST see an empty history.
    assert captured == [[]]


# ───────────────────────── play_session ─────────────────────────


def _make_session_fixtures():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    chronicler = make_chronicler_runtime()
    state = make_initial_state(world)
    return world, bundle, runtime, chronicler, state


async def test_play_session_calls_both_callbacks_and_terminates_on_end_game(monkeypatch):
    """Two scenes. Scene 2's third turn fires the decision beat via
    Phase 1, then the choice author emits Leave → engine resolves
    EndGame."""
    world, bundle, runtime, chronicler, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_chronicler(monkeypatch)
    # Phase 1 fires nothing until scene 2 turn 2 (the fourth call).
    phase1_outputs = iter([
        StateChangeBundle(),  # scene 1 turn 1
        StateChangeBundle(),  # scene 1 turn 2 (Leave)
        StateChangeBundle(),  # scene 2 turn 1
        StateChangeBundle(   # scene 2 turn 2: fire the decision beat
            state_changes=[UnlockBeat(beat_id=b) for b in decision_beats],
        ),
    ])

    async def fake_phase1(*args, **kwargs):
        return next(phase1_outputs)

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    _patch_prose_writer(monkeypatch, ["t1", "t1-end", "t2", "t2-end"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="ask"), Option(text="go")])),
        ChoiceBundle(next=Leave()),  # scene 1 EndScene (no ending satisfied)
        ChoiceBundle(next=Continue(options=[Option(text="ask"), Option(text="go")])),
        ChoiceBundle(next=Leave()),  # scene 2 EndGame (decision beat now in state)
    ])

    log_entries: list[tuple[State, ChronicleEntry]] = []
    closes: list[ProsePart] = []

    def policy_pick(pp: ProsePart) -> int:
        return 0

    def policy_walk(state: State, world: World) -> str:
        return world.locations[state.player_location].adjacent_to[0]

    def record_log(s: State, o: ChronicleEntry) -> None:
        log_entries.append((s, o))

    terminal = await play_session(
        bundle, runtime, chronicler, state, world,
        pick_option=policy_pick,
        pick_walk=policy_walk,
        on_chronicle_entry=record_log,
        on_scene_close=closes.append,
    )

    # Terminated on EndGame, state reflects it.
    assert isinstance(terminal.next, EndGame)
    assert state.game_over is not None
    assert state.game_over.ending_id == "published_story"
    # Both callbacks fired once per scene end.
    assert len(closes) == 2
    assert len(log_entries) == 2
    assert len(state.case_file_entries) == 2


async def test_play_session_pick_walk_only_called_between_scenes(monkeypatch):
    """If the session ends in EndGame on the first scene, pick_walk is
    never called. Phase 1 fires the decision beat on the only turn so
    Leave resolves to EndGame."""
    world, bundle, runtime, chronicler, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_phase1(monkeypatch, StateChangeBundle(
        state_changes=[UnlockBeat(beat_id=b) for b in decision_beats],
    ))
    _patch_prose_writer(monkeypatch, ["opening"])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    walk_calls: list[str] = []

    def policy_walk(state: State, world: World) -> str:
        walk_calls.append(state.player_location)
        return world.locations[state.player_location].adjacent_to[0]

    terminal = await play_session(
        bundle, runtime, chronicler, state, world,
        pick_option=lambda pp: 0,
        pick_walk=policy_walk,
    )
    assert isinstance(terminal.next, EndGame)
    assert walk_calls == []


async def test_play_session_case_file_entries_accumulate_on_state(monkeypatch):
    world, bundle, runtime, chronicler, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_phase1(monkeypatch, StateChangeBundle(
        state_changes=[UnlockBeat(beat_id=b) for b in decision_beats],
    ))
    _patch_prose_writer(monkeypatch, ["t1"])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    await play_session(
        bundle, runtime, chronicler, state, world,
        pick_option=lambda pp: 0,
        pick_walk=lambda s, w: w.locations[s.player_location].adjacent_to[0],
    )
    assert len(state.case_file_entries) == 1
    assert state.case_file_entries[0] == "entry"
