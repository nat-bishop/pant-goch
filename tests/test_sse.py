"""M4 — SSE event emission + paced clause emitter.

Tests the event surface from two angles:
1. `paced_clause_emitter` directly (unit) — words group into clauses on
   punctuation / word-cap boundaries, emissions throttle to the per-clause
   hold, indices increment, sentinel fires with is_last=True.
2. `play_scene` / `play_session` with stubbed agents + an event-recording
   `emit_event` callback — verifies which events fire at which boundaries
   (scene_start, state_change, prose_chunk, options/scene_exit/game_over,
   location_change, log_entry_chunk).
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from handler.agents import (
    build_scene_runner_bundle,
    make_chronicler_runtime,
    make_scene_runner_runtime,
)
from handler.schemas import (
    ChoiceBundle,
    ChronicleEntry,
    Continue,
    NpcPlacementBundle,
    Option,
    StateChangeBundle,
    UnlockBeat,
)
from handler.scene_loop import LeaveRequested, play_scene, play_session
from handler.sse import (
    _clauses_from_words,
    _words_from_stream,
    _words_from_text,
    paced_clause_emitter,
)
from handler.state import make_initial_state
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── populator / chronicler / departure no-ops ─────────────────────────


async def _empty_placements(*args, **kwargs):
    return NpcPlacementBundle()


def _patch_populator(monkeypatch):
    monkeypatch.setattr("handler.scene_loop.emit_placements", _empty_placements)


def _patch_chronicler(monkeypatch, text: str = "case-file entry"):
    async def fake_emit(*args, **kwargs):
        return ChronicleEntry(
            case_file_entry=text,
            npc_memory_updates=[],
        )
    monkeypatch.setattr("handler.scene_loop.emit_chronicle_entry", fake_emit)


def _patch_departure_prose(monkeypatch, text: str = "You step out into the rain."):
    async def fake(*args, **kwargs):
        return text
    monkeypatch.setattr("handler.scene_loop.emit_departure_prose", fake)


def _leave_after(n: int):
    """pick_option callback that picks 0 the first `n` times and then
    raises LeaveRequested. Mirrors POST /leave on the SSE backend."""
    state = {"calls": 0}

    def cb(pp):
        state["calls"] += 1
        if state["calls"] > n:
            raise LeaveRequested()
        return 0
    return cb


# ───────────────────────── paced_clause_emitter ─────────────────────────


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


async def test_clauses_split_on_terminal_punctuation():
    """Comma / period / semicolon end a clause."""
    pairs = [
        p async for p in _clauses_from_words(
            _words_from_text("She paused, then spoke. The room held still.")
        )
    ]
    texts = [t for t, _ in pairs]
    assert texts == ["She paused, ", "then spoke. ", "The room held still."]
    # word_count tracks the words emitted into each clause.
    assert [n for _, n in pairs] == [2, 2, 4]


async def test_clauses_respect_paragraph_break():
    """A paragraph break (\\n\\n in trailing whitespace) closes a clause
    even without punctuation."""
    pairs = [
        p async for p in _clauses_from_words(
            _words_from_text("first paragraph\n\nsecond")
        )
    ]
    texts = [t for t, _ in pairs]
    assert "".join(texts) == "first paragraph\n\nsecond"
    assert len(pairs) == 2


async def test_clauses_cap_on_word_count():
    """No internal punctuation → soft cap (MAX_CLAUSE_WORDS = 8) kicks in."""
    text = "one two three four five six seven eight nine ten"
    pairs = [p async for p in _clauses_from_words(_words_from_text(text))]
    # 10 words, cap 8 → first clause is 8 words, then 2 leftover.
    assert [n for _, n in pairs] == [8, 2]
    assert "".join(t for t, _ in pairs) == text


async def test_paced_clause_emitter_throttles_at_clause_hold(monkeypatch):
    """Emissions must hold for the per-clause cadence."""
    pace = 0.02
    monkeypatch.setattr("handler.sse.CLAUSE_WORD_PACE_S", pace)
    monkeypatch.setattr("handler.sse.MIN_CLAUSE_PACE_S", pace)

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    start = time.monotonic()
    # Four single-word clauses (each comma-terminated).
    await paced_clause_emitter(
        _words_from_text("one, two, three, four."),
        emit,
        event_kind="prose_chunk",
    )
    elapsed = time.monotonic() - start

    # 4 clause events + 1 sentinel = 5 events.
    assert len(events) == 5
    # First emit is immediate; the three subsequent clauses each hold
    # ≥ pace before they emit.
    assert elapsed >= 3 * pace * 0.9
    # idx increments monotonically.
    for i, e in enumerate(events):
        assert e["idx"] == i
    # Last event is the sentinel.
    assert events[-1]["is_last"] is True
    assert events[-1]["text"] == ""
    # Earlier events carry the clauses.
    assert events[0]["text"] == "one, "
    assert events[3]["text"] == "four."


async def test_paced_clause_emitter_passes_extra_payload(monkeypatch):
    monkeypatch.setattr("handler.sse.CLAUSE_WORD_PACE_S", 0.001)
    monkeypatch.setattr("handler.sse.MIN_CLAUSE_PACE_S", 0.001)

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    await paced_clause_emitter(
        _words_from_text("hi"),
        emit,
        event_kind="log_entry_chunk",
        extra={"scene_id": 7},
    )
    for e in events:
        assert e["scene_id"] == 7
        assert e["event"] == "log_entry_chunk"


# ───────────────────────── play_scene event emission ─────────────────────────


@pytest.fixture
def world_and_bundle():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    return world, bundle


def _fast_pacing(monkeypatch, pace: float = 0.001) -> None:
    """Monkeypatch the clause pacer's hold to near-zero so scene-level
    tests aren't wall-clock-bound."""
    monkeypatch.setattr("handler.sse.CLAUSE_WORD_PACE_S", pace)
    monkeypatch.setattr("handler.sse.MIN_CLAUSE_PACE_S", pace)


async def test_play_scene_emits_scene_start_then_terminal_events(monkeypatch, world_and_bundle):
    """One-turn scene ending in EndScene (via player leave):
    scene_start -> state_change* -> prose_chunk* (normal turn) ->
    prose_chunk* (departure) -> scene_exit."""
    world, bundle = world_and_bundle
    state = make_initial_state(world)

    async def fake_phase1(*args, **kwargs):
        # tom_closes_off is a negative beat — not in any ending's
        # requires_beats — so close_scene yields EndScene, not EndGame.
        return StateChangeBundle(state_changes=[UnlockBeat(beat_id="tom_closes_off")])

    async def fake_prose(*args, **kwargs):
        return "Tom shuts down."

    async def fake_choices(*args, **kwargs):
        return ChoiceBundle(next=Continue(options=[
            Option(text="a"), Option(text="b"),
        ]))

    _patch_populator(monkeypatch)
    _patch_departure_prose(monkeypatch)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose)
    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices)

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    _fast_pacing(monkeypatch)

    state.player_location = "the_slaughtered_lamb"
    await play_scene(
        bundle, make_scene_runner_runtime(), state,
        pick_option=_leave_after(0),
        emit_event=emit,
    )

    kinds = [e["event"] for e in events]
    assert kinds[0] == "scene_start"
    assert "state_change" in kinds
    # Prose chunk events (normal turn AND departure turn).
    assert any(e["event"] == "prose_chunk" for e in events)
    # Final event for an EndScene is scene_exit.
    assert kinds[-1] == "scene_exit"
    # state_change carries the beat_id.
    sc = next(e for e in events if e["event"] == "state_change")
    assert sc["kind"] == "unlock_beat"
    assert sc["payload"]["beat_id"] == "tom_closes_off"


async def test_play_scene_emits_options_event_for_continue(monkeypatch, world_and_bundle):
    """A Continue turn fires an options event with one `{text}` entry per
    option (no leaves_for field; the map is the navigation surface)."""
    world, bundle = world_and_bundle
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    proses = iter(["A pause.", "The hour ends."])
    choices = iter([
        ChoiceBundle(next=Continue(options=[
            Option(text="Stay quiet"),
            Option(text="Walk to post"),
        ])),
        ChoiceBundle(next=Continue(options=[
            Option(text="watch"), Option(text="speak"),
        ])),
    ])

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_prose(*args, **kwargs):
        return next(proses)

    async def fake_choices(*args, **kwargs):
        return next(choices)

    _patch_populator(monkeypatch)
    _patch_departure_prose(monkeypatch)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose)
    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices)
    _fast_pacing(monkeypatch)

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    await play_scene(
        bundle, make_scene_runner_runtime(), state,
        pick_option=_leave_after(1),
        emit_event=emit,
    )

    options_events = [e for e in events if e["event"] == "options"]
    assert len(options_events) >= 1
    opts = options_events[0]["options"]
    assert opts[0] == {"text": "Stay quiet"}
    assert opts[1] == {"text": "Walk to post"}


async def test_play_scene_emits_game_over_for_end_game(monkeypatch, world_and_bundle):
    """Player leave with a satisfied ending in state → close_scene
    resolves EndGame → game_over event with the resolved id."""
    world, bundle = world_and_bundle
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    # Satisfy the published_story ending's decision beat.
    state.unlocked_beats = set(world.endings["published_story"].requires_beats)

    async def fake_phase1(*args, **kwargs):
        return StateChangeBundle()

    async def fake_prose(*args, **kwargs):
        return "The story runs."

    async def fake_choices(*args, **kwargs):
        return ChoiceBundle(next=Continue(options=[
            Option(text="a"), Option(text="b"),
        ]))

    _patch_populator(monkeypatch)
    _patch_departure_prose(monkeypatch)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose)
    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices)
    _fast_pacing(monkeypatch)

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    await play_scene(
        bundle, make_scene_runner_runtime(), state,
        pick_option=_leave_after(0),
        emit_event=emit,
    )
    kinds = [e["event"] for e in events]
    assert kinds[-1] == "game_over"
    go = events[-1]
    assert go["ending_id"] == "published_story"


# ───────────────────────── play_session event emission ─────────────────────────


async def test_play_session_emits_location_change_and_log_entry_chunks(
    monkeypatch, world_and_bundle
):
    """Cross-scene events: location_change fires between scenes;
    log_entry_chunk events stream the case file entry."""
    world, bundle = world_and_bundle
    chronicler = make_chronicler_runtime()
    state = make_initial_state(world)

    # First scene at pub: player leaves with no satisfied ending → EndScene.
    # Second scene at post office: Phase 1 fires the published_story
    # decision beat just before the player leaves → EndGame.
    proses = iter(["Pub scene.", "Post scene."])
    choices = iter([
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    decision_beats = world.endings["published_story"].requires_beats
    phase1_outputs = iter([
        StateChangeBundle(),  # scene 1 turn 1: nothing
        StateChangeBundle(   # scene 2 turn 1: fire the decision beat
            state_changes=[UnlockBeat(beat_id=b) for b in decision_beats],
        ),
    ])

    async def fake_phase1(*args, **kwargs):
        return next(phase1_outputs)

    async def fake_prose(*args, **kwargs):
        return next(proses)

    async def fake_choices(*args, **kwargs):
        return next(choices)

    _patch_populator(monkeypatch)
    _patch_chronicler(monkeypatch, text="The pub, the road. Tom was at the bar.")
    _patch_departure_prose(monkeypatch)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    monkeypatch.setattr("handler.scene_loop.emit_prose_text", fake_prose)
    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices)
    _fast_pacing(monkeypatch)

    events: list[dict] = []

    async def emit(e: dict) -> None:
        events.append(e)

    def pick_option(_pp):
        # Both scenes: leave on the first pick wait.
        raise LeaveRequested()

    async def pick_walk(_state, _world):
        # Walk to post office from pub.
        return "the_post_office"

    state.player_location = "the_slaughtered_lamb"

    await play_session(
        bundle, make_scene_runner_runtime(), chronicler, state, world,
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
    # Log entry chunks streamed (at least one per scene).
    log_chunks = [e for e in events if e["event"] == "log_entry_chunk"]
    # Two scenes -> two log entries -> at least 2 sentinels.
    sentinels = [e for e in log_chunks if e.get("is_last")]
    assert len(sentinels) == 2
    # game_over fires BEFORE the final log entry's chunks finish streaming.
    assert "game_over" in kinds
    game_over_idx = kinds.index("game_over")
    # After game_over, only log_entry_chunk events follow.
    assert all(k == "log_entry_chunk" for k in kinds[game_over_idx + 1:])
