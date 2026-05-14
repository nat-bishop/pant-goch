"""M3 — Scene-loop orchestration tests.

Mocks emit_state_changes / emit_prose to avoid LLM dependency. Verifies
that run_turn sequences phase 1 → apply → phase 2 → append, that
fallbacks fire on UnexpectedModelBehavior, and that play_scene drives
ContinueScene picks until termination.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from handler.agents import (
    build_scene_runner_bundle,
    make_log_writer_runtime,
)
from handler.engine import enter_scene
from handler.schemas import (
    ContinueScene,
    EndGame,
    EndScene,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from handler.scene_loop import (
    _fallback_prose_part,
    play_scene,
    play_session,
    run_turn,
)
from handler.state import LogEntry, State, make_initial_state
from handler.world import World, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


@pytest.fixture
def bundle_and_state():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")  # tom is here
    return bundle, state


# ───────────────────────── run_turn happy path ─────────────────────────


async def test_run_turn_applies_state_changes_and_appends_record(monkeypatch, bundle_and_state):
    bundle, state = bundle_and_state
    canned_bundle = StateChangeBundle(
        state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")]
    )
    canned_prose = ProsePart(
        prose="Tom looks down at his pint, and then up at you.",
        next=ContinueScene(options=[
            Option(text="say nothing"),
            Option(text="leave for the post office", leaves_for="the_post_office"),
        ]),
    )

    async def fake_phase1(*args, **kwargs):
        return canned_bundle

    async def fake_phase2(*args, **kwargs):
        return canned_prose

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    result = await run_turn(bundle, state)

    assert result is canned_prose
    # State change applied.
    assert "tom_named_a_vehicle" in state.unlocked_beats
    # TurnRecord appended with scene anchor + applied changes.
    assert len(state.dialogue_history) == 1
    record = state.dialogue_history[0]
    assert record.scene_id == state.scene_id
    assert record.location_id == "the_slaughtered_lamb"
    assert record.npcs_present == ["tom"]
    assert record.next_kind == "continue"
    assert len(record.options) == 2
    assert record.applied_state_changes[0].beat_id == "tom_named_a_vehicle"


async def test_run_turn_phase1_fallback_on_unexpected_behavior(monkeypatch, bundle_and_state):
    """Phase 1 retries exhausted → empty bundle, no state change applied."""
    bundle, state = bundle_and_state
    canned_prose = ProsePart(
        prose="Tom drinks.",
        next=ContinueScene(options=[
            Option(text="watch"), Option(text="go", leaves_for="the_post_office"),
        ]),
    )

    async def fake_phase1_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated retry exhaustion")

    async def fake_phase2(*args, **kwargs):
        return canned_prose

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1_fails)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    result = await run_turn(bundle, state)
    assert state.unlocked_beats == set()  # no fallout from the failure
    assert result is canned_prose
    # TurnRecord still appended (no state changes).
    assert state.dialogue_history[-1].applied_state_changes == []


async def test_run_turn_phase2_fallback_on_unexpected_behavior(monkeypatch, bundle_and_state):
    """Phase 2 retries exhausted → engine-authored generic prose +
    ContinueScene with 'Wait', 'Look around', and an arbitrary leave."""
    bundle, state = bundle_and_state

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated retry exhaustion")

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2_fails)

    result = await run_turn(bundle, state)
    # Generic prose (from _fallback_prose_part).
    assert isinstance(result.next, ContinueScene)
    assert any(o.leaves_for is not None for o in result.next.options)
    # No engine metatext leaks into prose (per plan §"Drift mitigation").
    assert "[engine" not in result.prose.lower()


async def test_run_turn_phase2_fallback_on_leave_pick_emits_endscene(monkeypatch, bundle_and_state):
    """Plan §"Drift mitigation": leave-pick + phase 2 failure forces EndScene."""
    bundle, state = bundle_and_state

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2_fails)

    leave = Option(text="leave", leaves_for="the_post_office")
    result = await run_turn(bundle, state, leave_pick=leave)
    assert isinstance(result.next, EndScene)


# ───────────────────────── _fallback_prose_part ─────────────────────────


def test_fallback_endscene_when_leave_pick(bundle_and_state):
    bundle, state = bundle_and_state
    leave = Option(text="leave", leaves_for="the_post_office")
    result = _fallback_prose_part(state, bundle, leave_pick=leave)
    assert isinstance(result.next, EndScene)


def test_fallback_continue_includes_leave_option(bundle_and_state):
    bundle, state = bundle_and_state
    result = _fallback_prose_part(state, bundle, leave_pick=None)
    assert isinstance(result.next, ContinueScene)
    leave_opts = [o for o in result.next.options if o.leaves_for is not None]
    assert len(leave_opts) >= 1


# ───────────────────────── play_scene ─────────────────────────


async def test_play_scene_drives_until_endscene(monkeypatch):
    """Scene with three turns: turn 1 + 2 ContinueScene, turn 3 EndScene.
    play_scene picks an option each turn; returns the EndScene ProsePart."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    # Three canned ProsePart responses for three turns.
    proses = iter([
        ProsePart(prose="t1", next=ContinueScene(options=[
            Option(text="stay"), Option(text="go", leaves_for="the_post_office"),
        ])),
        ProsePart(prose="t2", next=ContinueScene(options=[
            Option(text="ask"), Option(text="go", leaves_for="the_post_office"),
        ])),
        ProsePart(prose="t3", next=EndScene()),
    ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        return next(proses)

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    # Pick the first option each turn (the non-leave one).
    result = await play_scene(bundle, state, pick_option=lambda pp: 0)
    assert isinstance(result.next, EndScene)
    # Three TurnRecords.
    assert len(state.dialogue_history) == 3
    # Each non-terminal record has player_pick set.
    assert state.dialogue_history[0].player_pick is not None
    assert state.dialogue_history[1].player_pick is not None
    assert state.dialogue_history[2].player_pick is None  # EndScene; no pick


async def test_play_scene_picks_leave_option_records_leave_pick(monkeypatch):
    """When the player picks a leave option, the next turn must be invoked
    with leave_pick set (validated by phase 2 forcing EndScene)."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    captured_leave_picks: list[Option | None] = []

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        # Capture the leave_pick kwarg for inspection.
        captured_leave_picks.append(kwargs.get("leave_pick"))
        # Turn 1: ContinueScene with leave option. Turn 2: EndScene.
        if len(captured_leave_picks) == 1:
            return ProsePart(prose="t1", next=ContinueScene(options=[
                Option(text="stay"),
                Option(text="leave for post office", leaves_for="the_post_office"),
            ]))
        return ProsePart(prose="t2", next=EndScene())

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    # Pick the leave option (index 1).
    result = await play_scene(bundle, state, pick_option=lambda pp: 1)
    assert isinstance(result.next, EndScene)
    # First turn: no leave_pick. Second turn (after picking leave): leave_pick set.
    assert captured_leave_picks[0] is None
    assert captured_leave_picks[1] is not None
    assert captured_leave_picks[1].leaves_for == "the_post_office"


# ───────────────────────── play_session ─────────────────────────


def _make_session_fixtures():
    """Helper: load world + build bundle + log writer runtime + state."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    log_writer = make_log_writer_runtime(TestModel(custom_output_text="entry"), world)
    state = make_initial_state(world)
    return world, bundle, log_writer, state


async def test_play_session_calls_both_callbacks_and_terminates_on_end_game(monkeypatch):
    """A scripted policy ends the session on EndGame; the LogEntry and
    closing-prose callbacks both fire; state.game_over is set."""
    world, bundle, log_writer, state = _make_session_fixtures()

    proses = iter([
        # Scene 1: one Continue, then EndScene
        ProsePart(prose="t1", next=ContinueScene(options=[
            Option(text="ask"),
            Option(text="go", leaves_for="the_post_office"),
        ])),
        ProsePart(prose="t1-end", next=EndScene()),
        # Scene 2 at the post office: ends in EndGame after the validator
        # accepts because evidence prereqs are stubbed (we patch validator).
        ProsePart(prose="t2", next=ContinueScene(options=[
            Option(text="ask"),
            Option(text="leave", leaves_for="the_slaughtered_lamb"),
        ])),
        ProsePart(prose="t2-end", next=EndGame(ending_id="published_story")),
    ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        return next(proses)

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    log_entries: list[LogEntry] = []
    closes: list[ProsePart] = []

    def policy_pick(pp: ProsePart) -> int:
        return 0  # always the first option (non-leave)

    def policy_walk(state: State, world: World) -> str:
        return world.locations[state.player_location].adjacent_to[0]

    terminal = await play_session(
        bundle, log_writer, state, world,
        pick_option=policy_pick,
        pick_walk=policy_walk,
        on_log_entry=log_entries.append,
        on_scene_close=closes.append,
    )

    # Terminated on EndGame, state reflects it.
    assert isinstance(terminal.next, EndGame)
    assert state.game_over is not None
    assert state.game_over.ending_id == "published_story"

    # Both callbacks fired once per scene end (two scenes = two events each).
    assert len(closes) == 2
    assert len(log_entries) == 2
    # case_file_entries (caller-managed) is None here; play_session builds one.


async def test_play_session_pick_walk_only_called_between_scenes(monkeypatch):
    """pick_walk fires once per EndScene-but-not-EndGame transition. If
    the session ends in EndGame on the first scene, pick_walk is never
    called."""
    world, bundle, log_writer, state = _make_session_fixtures()

    proses = iter([
        ProsePart(prose="opening", next=EndGame(ending_id="published_story")),
    ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        return next(proses)

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    walk_calls: list[str] = []

    def policy_pick(pp: ProsePart) -> int:
        return 0

    def policy_walk(state: State, world: World) -> str:
        walk_calls.append(state.player_location)
        return world.locations[state.player_location].adjacent_to[0]

    terminal = await play_session(
        bundle, log_writer, state, world,
        pick_option=policy_pick,
        pick_walk=policy_walk,
    )

    assert isinstance(terminal.next, EndGame)
    # Session ended immediately on EndGame; pick_walk never fired.
    assert walk_calls == []


async def test_play_session_case_file_entries_passed_through(monkeypatch):
    """The caller-provided case_file_entries list is the same list the
    log writer appends to — eval drivers can inspect it directly."""
    world, bundle, log_writer, state = _make_session_fixtures()

    proses = iter([
        ProsePart(prose="t1", next=EndGame(ending_id="published_story")),
    ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        return next(proses)

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    caller_owned: list[str] = []

    await play_session(
        bundle, log_writer, state, world,
        pick_option=lambda pp: 0,
        pick_walk=lambda s, w: w.locations[s.player_location].adjacent_to[0],
        case_file_entries=caller_owned,
    )

    # One entry written (one scene ended).
    assert len(caller_owned) == 1
    assert caller_owned[0] == "entry"
