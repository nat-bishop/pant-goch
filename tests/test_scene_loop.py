"""Scene-loop orchestration tests.

Mocks emit_state_changes / emit_prose_text / emit_choices /
emit_departure_prose to avoid LLM dependency. Verifies that run_turn
sequences phase 1 → apply → prose writer → choice author → assemble
ProsePart → append, that fallbacks fire on UnexpectedModelBehavior, that
play_scene drives Continue picks and FreeText typed input until
termination, and that scene closure flows through `_close_via_departure`
on both player-leave (LeaveRequested raised from pick callbacks) and
the engine's MAX_TURNS_PER_SCENE guardrail.
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
from handler.engine import enter_scene
from handler.schemas import (
    ChoiceBundle,
    ChronicleEntry,
    Continue,
    EndGame,
    EndScene,
    FreeText,
    NpcPlacementBundle,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from handler.scene_loop import (
    LeaveRequested,
    MAX_TURNS_PER_SCENE,
    close_scene,
    play_scene,
    play_session,
    run_turn,
)
from handler.state import State, make_initial_state
from handler.world import World, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── populator / chronicler / departure no-ops ─────────────────────────


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


def _patch_departure_prose(monkeypatch, text: str = "You stand. The door."):
    """Patch the departure-prose agent call used by the close path."""
    async def fake(*args, **kwargs):
        return text
    monkeypatch.setattr("handler.scene_loop.emit_departure_prose", fake)


def _leave_after(n: int):
    """Build a pick_option callback that picks 0 the first `n` times and
    then raises LeaveRequested (simulating POST /leave)."""
    state = {"calls": 0}

    def cb(pp):
        state["calls"] += 1
        if state["calls"] > n:
            raise LeaveRequested()
        return 0
    return cb


@pytest.fixture
def bundle_and_state():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")  # tom is here
    return bundle, runtime, state


# ───────────────────────── close_scene (pure helper) ─────────────────────────


def test_close_scene_with_no_satisfied_ending_returns_end_scene():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    pp = close_scene(world, state, "prose")
    assert isinstance(pp.next, EndScene)
    assert pp.prose == "prose"


def test_close_scene_with_satisfied_ending_returns_end_game():
    """If the decision beats for an authored ending have fired, close_scene
    resolves to EndGame with that ending id."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    ending = world.endings["published_story"]
    state.unlocked_beats = set(ending.requires_beats)
    pp = close_scene(world, state, "prose")
    assert isinstance(pp.next, EndGame)
    assert pp.next.ending_id == "published_story"


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


async def test_run_turn_passes_through_free_text(monkeypatch, bundle_and_state):
    """Choice author FreeText → run_turn pass-through to ProsePart.next."""
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The pub falls quiet."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=FreeText(prompt="What do you ask?", npc_id="tom")),
    ])

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, FreeText)
    assert result.next.prompt == "What do you ask?"
    assert result.next.npc_id == "tom"


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


# ───────────────────────── play_scene: leave + turn-cap ─────────────────────────


async def test_play_scene_drives_until_player_leave(monkeypatch):
    """Two Continue turns; on the second turn's pick the player raises
    LeaveRequested → engine writes departure prose + EndScene. First
    turn's pick succeeded; the second turn's prose was streamed but the
    player chose to leave instead of picking."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="stay"), Option(text="ask")])),
        ChoiceBundle(next=Continue(options=[Option(text="watch"), Option(text="speak")])),
    ])
    _patch_departure_prose(monkeypatch, text="You stand and step into the rain.")

    result = await play_scene(
        bundle, runtime, state, pick_option=_leave_after(1),
    )
    assert isinstance(result.next, EndScene)
    # Three TurnRecords: 2 normal turns + 1 departure turn.
    assert len(state.dialogue_history) == 3
    # First turn picked; second turn no pick (player chose leave); third
    # is the departure prose.
    assert state.dialogue_history[0].player_pick is not None
    assert state.dialogue_history[1].player_pick is None
    assert state.dialogue_history[2].prose == "You stand and step into the rain."
    assert state.dialogue_history[2].player_pick is None


async def test_play_scene_player_leave_resolves_to_end_game_when_ending_satisfied(monkeypatch):
    """Player /leave with satisfied ending prereqs → close_scene yields
    EndGame."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    state.unlocked_beats = set(world.endings["published_story"].requires_beats)

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_departure_prose(monkeypatch)

    result = await play_scene(
        bundle, runtime, state, pick_option=_leave_after(0),
    )
    assert isinstance(result.next, EndGame)
    assert result.next.ending_id == "published_story"


async def test_play_scene_dispatches_free_text_and_then_player_leave(monkeypatch):
    """Choice author emits FreeText → play_scene calls pick_free_text →
    on the next turn the player picks Leave via the typed-input callback
    raising LeaveRequested."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=FreeText(prompt="What do you say?", npc_id="tom")),
        ChoiceBundle(next=Continue(options=[Option(text="watch"), Option(text="go")])),
    ])
    _patch_departure_prose(monkeypatch)

    def pick_free_text(req: FreeText) -> str:
        assert req.npc_id == "tom"
        return "I came because of Megan."

    # pick_option leaves on its very first call (turn 2's Continue → wait).
    result = await play_scene(
        bundle, runtime, state,
        pick_option=_leave_after(0),
        pick_free_text=pick_free_text,
    )
    assert isinstance(result.next, EndScene)
    assert state.dialogue_history[0].player_pick is not None
    assert state.dialogue_history[0].player_pick.text == "I came because of Megan."


async def test_play_scene_leave_from_free_text_callback(monkeypatch):
    """Player raises LeaveRequested from the typed-input wait → close_scene
    fires just like from a pick wait."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=FreeText(prompt="What do you say?", npc_id=None)),
    ])
    _patch_departure_prose(monkeypatch)

    def pick_free_text(_req):
        raise LeaveRequested()

    result = await play_scene(
        bundle, runtime, state,
        pick_option=lambda pp: 0,
        pick_free_text=pick_free_text,
    )
    assert isinstance(result.next, EndScene)


async def test_play_scene_turn_cap_engine_close(monkeypatch):
    """If a scene runs to MAX_TURNS_PER_SCENE turns without a player
    leave, the engine synthesises EndScene via the departure flow."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    # MAX_TURNS_PER_SCENE proses, then the guardrail closes.
    _patch_prose_writer(
        monkeypatch, [f"t{i}" for i in range(MAX_TURNS_PER_SCENE)],
    )
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")]))
        for _ in range(MAX_TURNS_PER_SCENE)
    ])
    _patch_departure_prose(monkeypatch, text="A long silence and the door.")

    result = await play_scene(
        bundle, runtime, state, pick_option=lambda pp: 0,
    )
    assert isinstance(result.next, EndScene)
    # MAX_TURNS_PER_SCENE normal turns + 1 departure turn.
    assert len(state.dialogue_history) == MAX_TURNS_PER_SCENE + 1
    assert state.dialogue_history[-1].prose == "A long silence and the door."


async def test_play_scene_departure_prose_fallback(monkeypatch):
    """Departure prose retries exhausted → engine fallback line; close
    still resolves."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    async def fake_departure_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")

    monkeypatch.setattr("handler.scene_loop.emit_departure_prose", fake_departure_fails)

    result = await play_scene(
        bundle, runtime, state, pick_option=_leave_after(0),
    )
    assert isinstance(result.next, EndScene)
    # Fallback prose was used.
    assert state.dialogue_history[-1].prose != ""


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
    _patch_departure_prose(monkeypatch)

    captured: list[list] = []

    async def fake_prose(bundle, rt, *args, **kwargs):
        # Observe message_history at the time of the first prose-writer
        # call.
        captured.append(list(rt.prose_writer_message_history))
        return "t1"

    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose)
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    await play_scene(bundle, runtime, state, pick_option=_leave_after(0))
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
    """Two scenes. Scene 2's second turn fires the decision beat via
    Phase 1, then the player leaves → close_scene yields EndGame."""
    world, bundle, runtime, chronicler, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_departure_prose(monkeypatch)
    # Phase 1 fires nothing until scene 2's second turn.
    phase1_outputs = iter([
        StateChangeBundle(),  # scene 1 turn 1
        StateChangeBundle(),  # scene 1 turn 2
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
        ChoiceBundle(next=Continue(options=[Option(text="ask"), Option(text="go")])),
        ChoiceBundle(next=Continue(options=[Option(text="ask"), Option(text="go")])),
        ChoiceBundle(next=Continue(options=[Option(text="ask"), Option(text="go")])),
    ])

    # Per-scene leave: pick once, then leave on the second wait.
    scene_pick_counts = {"calls": 0}

    def policy_pick(pp):
        scene_pick_counts["calls"] += 1
        # 1st call: scene 1 turn 1 → pick. 2nd call: scene 1 turn 2 → leave.
        # 3rd call: scene 2 turn 1 → pick. 4th call: scene 2 turn 2 → leave.
        if scene_pick_counts["calls"] in (2, 4):
            raise LeaveRequested()
        return 0

    log_entries: list[tuple[State, ChronicleEntry]] = []
    closes: list[ProsePart] = []

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
    the player's leave resolves to EndGame."""
    world, bundle, runtime, chronicler, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_departure_prose(monkeypatch)
    _patch_phase1(monkeypatch, StateChangeBundle(
        state_changes=[UnlockBeat(beat_id=b) for b in decision_beats],
    ))
    _patch_prose_writer(monkeypatch, ["opening"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    walk_calls: list[str] = []

    def policy_walk(state: State, world: World) -> str:
        walk_calls.append(state.player_location)
        return world.locations[state.player_location].adjacent_to[0]

    terminal = await play_session(
        bundle, runtime, chronicler, state, world,
        pick_option=_leave_after(0),
        pick_walk=policy_walk,
    )
    assert isinstance(terminal.next, EndGame)
    assert walk_calls == []


async def test_play_session_case_file_entries_accumulate_on_state(monkeypatch):
    world, bundle, runtime, chronicler, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_departure_prose(monkeypatch)
    _patch_phase1(monkeypatch, StateChangeBundle(
        state_changes=[UnlockBeat(beat_id=b) for b in decision_beats],
    ))
    _patch_prose_writer(monkeypatch, ["t1"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    await play_session(
        bundle, runtime, chronicler, state, world,
        pick_option=_leave_after(0),
        pick_walk=lambda s, w: w.locations[s.player_location].adjacent_to[0],
    )
    assert len(state.case_file_entries) == 1
    assert state.case_file_entries[0] == "entry"
