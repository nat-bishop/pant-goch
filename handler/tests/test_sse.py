"""M4 — SSE event emission + paced word emitter.

Tests the event surface from two angles:
1. `paced_word_emitter` directly (unit) — words emerge at >=80ms cadence,
   indices increment, sentinel fires with is_last=True.
2. `play_scene` / `play_session` with stubbed agents + an event-recording
   `emit_event` callback — verifies which events fire at which boundaries
   (scene_start, state_change, prose_word, options/scene_exit/game_over,
   location_change, log_entry_word).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from handler.agents import (
    build_scene_runner_bundle,
    make_log_writer_runtime,
    make_scene_runner_runtime,
)
from handler.schemas import (
    ContinueScene,
    EndGame,
    EndScene,
    LogWriterOutput,
    NpcPlacementBundle,
    Option,
    ProsePart,
    StateChangeBundle,
    UnlockBeat,
)
from handler.scene_loop import play_scene, play_session
from handler.sse import (
    WORD_PACE_S,
    _words_from_stream,
    _words_from_text,
    paced_word_emitter,
)
from handler.state import State, make_initial_state
from handler.world import World, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── populator / log writer no-ops ─────────────────────────


async def _empty_placements(*args, **kwargs):
    return NpcPlacementBundle()


def _patch_populator(monkeypatch):
    monkeypatch.setattr("handler.scene_loop.emit_placements", _empty_placements)


def _patch_log_writer(monkeypatch, text: str = "case-file entry"):
    async def fake_emit(*args, **kwargs):
        return LogWriterOutput(
            case_file_entry=text,
            npc_memory_updates=[],
        )
    monkeypatch.setattr("handler.scene_loop.emit_log_entry", fake_emit)


# ───────────────────────── paced_word_emitter ─────────────────────────


async def test_words_from_text_yields_complete_words():
    """Static text: yields one item per word, trailing whitespace preserved."""
    words = [w async for w in _words_from_text("Hello world.")]
    assert words == ["Hello ", "world."]


async def test_words_from_text_handles_punctuation():
    """No special handling of punctuation — token = non-whitespace run."""
    words = [w async for w in _words_from_text("It's fine, mostly. Yes!")]
    assert words == ["It's ", "fine, ", "mostly. ", "Yes!"]


async def _async_chunks(chunks: list[str]) -> AsyncIterator[str]:
    for c in chunks:
        yield c


async def test_words_from_stream_buffers_partial_words():
    """Streaming input split mid-word: emitter holds until whitespace
    arrives, then yields the complete word."""
    words = [w async for w in _words_from_stream(
        _async_chunks(["Hel", "lo wo", "rld."])
    )]
    assert words == ["Hello ", "world."]


async def test_words_from_stream_yields_final_word_without_trailing_space():
    """Last word in the stream lacks trailing whitespace — must still yield."""
    words = [w async for w in _words_from_stream(_async_chunks(["one two three"]))]
    assert words == ["one ", "two ", "three"]


async def test_paced_word_emitter_throttles_at_min_interval():
    """Emissions must be at least min_interval apart wall-clock."""
    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    interval = 0.05  # short interval for fast tests
    start = time.monotonic()
    await paced_word_emitter(
        _words_from_text("one two three four"),
        emit,
        event_kind="prose_word",
        min_interval=interval,
    )
    elapsed = time.monotonic() - start

    # 4 word events + 1 sentinel = 5 events.
    assert len(events) == 5
    # First word + sentinel are immediate; words 2-4 each wait one
    # interval. Total ~= 3 * interval.
    assert elapsed >= 3 * interval * 0.9  # loose tolerance for scheduler
    # idx increments monotonically
    for i, e in enumerate(events):
        assert e["idx"] == i
    # Last event is the sentinel.
    assert events[-1]["is_last"] is True
    assert events[-1]["word"] == ""
    # Earlier events have the actual words.
    assert events[0]["word"] == "one "
    assert events[3]["word"] == "four"


async def test_paced_word_emitter_passes_extra_payload():
    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    await paced_word_emitter(
        _words_from_text("hi"),
        emit,
        event_kind="log_entry_word",
        min_interval=0.001,
        extra={"scene_id": 7},
    )
    for e in events:
        assert e["scene_id"] == 7
        assert e["event"] == "log_entry_word"


# ───────────────────────── play_scene event emission ─────────────────────────


@pytest.fixture
def world_and_bundle():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    return world, bundle


async def test_play_scene_emits_scene_start_then_terminal_events(monkeypatch, world_and_bundle):
    """One-turn scene ending in EndScene: scene_start -> state_change*
    -> prose_word* -> scene_exit."""
    world, bundle = world_and_bundle
    state = make_initial_state(world)

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle(state_changes=[UnlockBeat(beat_id="tom_named_a_vehicle")])

    async def fake_phase2(*args, **kwargs):
        return ProsePart(prose="Tom spoke at last.", next=EndScene())

    _patch_populator(monkeypatch)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    # Use a very short pacing so the test runs fast — set on the imported
    # paced emitter via monkeypatch.
    monkeypatch.setattr("handler.sse.WORD_PACE_S", 0.001)
    monkeypatch.setattr("handler.scene_loop.paced_word_emitter",
                        _patched_paced_emitter(0.001))

    state.player_location = "the_slaughtered_lamb"
    await play_scene(bundle, make_scene_runner_runtime(), state, pick_option=lambda pp: 0, emit_event=emit)

    kinds = [e["event"] for e in events]
    assert kinds[0] == "scene_start"
    assert "state_change" in kinds
    # Prose word events
    assert any(e["event"] == "prose_word" for e in events)
    # Final event for an EndScene is scene_exit
    assert kinds[-1] == "scene_exit"
    # state_change carries the beat_id
    sc = next(e for e in events if e["event"] == "state_change")
    assert sc["kind"] == "unlock_beat"
    assert sc["payload"]["beat_id"] == "tom_named_a_vehicle"


def _patched_paced_emitter(interval: float):
    """Helper that returns a paced_word_emitter wrapper using a custom
    interval. Used by tests to skip the 80 ms wait per word."""
    from handler import sse

    async def _wrapped(word_source, emit, *, event_kind, min_interval=None, extra=None):
        return await sse.paced_word_emitter(
            word_source, emit,
            event_kind=event_kind, min_interval=interval, extra=extra,
        )

    return _wrapped


async def test_play_scene_emits_options_event_for_continue(monkeypatch, world_and_bundle):
    """ContinueScene turn fires an options event with text+leaves_for per
    option."""
    world, bundle = world_and_bundle
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    # Phase 2: one Continue then EndScene.
    proses = iter([
        ProsePart(prose="A pause.", next=ContinueScene(options=[
            Option(text="Stay quiet"),
            Option(text="Walk to post", leaves_for="the_post_office"),
        ])),
        ProsePart(prose="The hour ends.", next=EndScene()),
    ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        return next(proses)

    _patch_populator(monkeypatch)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)
    monkeypatch.setattr("handler.scene_loop.paced_word_emitter",
                        _patched_paced_emitter(0.001))

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    await play_scene(bundle, make_scene_runner_runtime(), state, pick_option=lambda pp: 0, emit_event=emit)

    options_events = [e for e in events if e["event"] == "options"]
    assert len(options_events) == 1
    opts = options_events[0]["options"]
    assert opts[0] == {"text": "Stay quiet", "leaves_for": None}
    assert opts[1] == {"text": "Walk to post", "leaves_for": "the_post_office"}


async def test_play_scene_emits_game_over_for_end_game(monkeypatch, world_and_bundle):
    world, bundle = world_and_bundle
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        return ProsePart(prose="The story runs.", next=EndGame(ending_id="published_story"))

    _patch_populator(monkeypatch)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)
    monkeypatch.setattr("handler.scene_loop.paced_word_emitter",
                        _patched_paced_emitter(0.001))

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    await play_scene(bundle, make_scene_runner_runtime(), state, pick_option=lambda pp: 0, emit_event=emit)
    kinds = [e["event"] for e in events]
    assert kinds[-1] == "game_over"
    go = events[-1]
    assert go["ending_id"] == "published_story"


# ───────────────────────── play_session event emission ─────────────────────────


async def test_play_session_emits_location_change_and_log_entry_words(
    monkeypatch, world_and_bundle
):
    """Cross-scene events: location_change fires between scenes;
    log_entry_word events stream the case file entry."""
    world, bundle = world_and_bundle
    log_writer = make_log_writer_runtime(
        TestModel(custom_output_text="The pub, the road. Tom was at the bar."), world
    )
    state = make_initial_state(world)

    # First scene at pub: ends in EndScene. Second scene at post office:
    # ends in EndGame.
    proses = iter([
        ProsePart(prose="Pub scene.", next=EndScene()),
        ProsePart(prose="Post scene.", next=EndGame(ending_id="published_story")),
    ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_phase2(*args, **kwargs):
        return next(proses)

    _patch_populator(monkeypatch)
    _patch_log_writer(monkeypatch, text="The pub, the road. Tom was at the bar.")
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose", fake_phase2)
    monkeypatch.setattr("handler.scene_loop.paced_word_emitter",
                        _patched_paced_emitter(0.001))

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    async def pick_option(_pp):
        return 0

    async def pick_walk(_state, _world):
        # Walk to post office from pub.
        return "the_post_office"

    state.player_location = "the_slaughtered_lamb"

    await play_session(
        bundle, make_scene_runner_runtime(), log_writer, state, world,
        pick_option=pick_option,
        pick_walk=pick_walk,
        emit_event=emit,
    )

    kinds = [e["event"] for e in events]
    # Two scenes -> two scene_starts.
    assert kinds.count("scene_start") == 2
    # One location_change between them.
    lc = [e for e in events if e["event"] == "location_change"]
    assert len(lc) == 1
    assert lc[0]["from"] == "the_slaughtered_lamb"
    assert lc[0]["to"] == "the_post_office"
    # Log entry words streamed (at least one per scene).
    log_words = [e for e in events if e["event"] == "log_entry_word"]
    # Two scenes -> two log entries -> at least 2 sentinels.
    sentinels = [e for e in log_words if e.get("is_last")]
    assert len(sentinels) == 2
    # game_over fires BEFORE the final log entry's words finish streaming
    # — the frontend can render the end-game panel while the case file
    # sidebar finishes its last entry.
    assert "game_over" in kinds
    game_over_idx = kinds.index("game_over")
    # After game_over, only log_entry_word events follow.
    assert all(k == "log_entry_word" for k in kinds[game_over_idx + 1:])
