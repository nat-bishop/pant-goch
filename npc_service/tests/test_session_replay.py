"""Deterministic-replay regression tests.

The promise of event sourcing: state is a fold over the event log. Two
sessions that apply the same event sequence in the same order MUST land
on byte-identical state. These tests validate that property end-to-end:
drive a session with a scripted /send sequence (against TestModel-stubbed
agents), then replay the recorded events into a fresh session and assert
the final state matches.

If this regresses, the fold logic has lost referential integrity — a
change to one of the `apply_event` cases mutated state that subsequent
events depend on inconsistently. Catches the kind of bug that would
make the save/load feature silently produce wrong replays.
"""
from __future__ import annotations

import pytest

from npc_service.events import (
    BeatFired,
    CanonPromoted,
    ClassifyEffectApplied,
    EmailEmitted,
    Event,
    LiteralEffectApplied,
    SendRecorded,
    SessionEnded,
)
from npc_service.models import (
    AxisResult,
    CanonCategory,
    Email,
    InboxState,
    PlayerCanonEntry,
    QualityDelta,
    Reply,
    Scenario,
)
from npc_service.session import Session, make_session, replay_into_session


@pytest.fixture
def small_scenario() -> Scenario:
    return Scenario.model_validate({
        "title": "Replay Test",
        "premise": "Synthetic.",
        "email_domain": "x",
        "qualities": [
            {"id": "engagement", "type": "float", "min": 0.0, "max": 10.0, "default": 0.0},
        ],
        "npcs": [{"id": "aria", "display_name": "Aria", "email_address": "aria@x"}],
        "threads": [{"id": "aria", "npc": "aria"}],
        "beats": [
            {
                "id": "aria.welcome",
                "thread": "aria",
                "npc": "aria",
                "kind": "one_shot",
                "spec": {"pre_authored_prose": "Welcome."},
            },
            {
                "id": "aria.terminal",
                "thread": "aria",
                "npc": "aria",
                "kind": "one_shot",
                "ends_session": True,
                "preconditions": [
                    {"kind": "quality_gte", "quality": "engagement", "value": 5.0},
                ],
                "spec": {"pre_authored_prose": "We're done."},
            },
        ],
    })


def _make_session(scenario: Scenario) -> Session:
    """Construct a Session without an AgentBundle. The replay tests never
    invoke agents — fold operates on state, not on LLMs — so a placeholder
    AgentBundle suffices."""
    from npc_service.agents import AgentBundle
    bundle = AgentBundle.__new__(AgentBundle)
    return make_session(scenario, bundle)


def _scripted_events() -> list[Event]:
    """A representative event sequence that exercises every fold case
    except ThreadHistoryAppended (which carries pydantic_ai ModelMessages
    not under test here)."""
    aria_welcome_email = Email(
        id="email_w", beat_id="aria.welcome", thread_id="aria",
        folder="Inbox", sender="aria@x", recipient="player@x",
        body="Welcome.", sent_at_label="just now",
    )
    player_email = Email(
        id="email_p1", beat_id=None, thread_id="aria",
        folder="Sent", sender="you", recipient="aria@x",
        body="Hi I'm Marcus", sent_at_label="just now", is_read=True,
    )
    player_reply = Reply(
        id="reply_p1", thread_id="aria", beat_responded_to="email_w",
        recipient="aria@x", body="Hi I'm Marcus",
    )
    return [
        EmailEmitted(email=aria_welcome_email),
        BeatFired(beat_id="aria.welcome", beat_kind="one_shot", completed=True),
        SendRecorded(email=player_email, reply=player_reply),
        ClassifyEffectApplied(
            reply_id="reply_p1",
            axis=AxisResult(quality="engagement", value="warm", rationale=""),
            delta=QualityDelta(
                quality="engagement", before=0.0, after=3.0,
                source="classify:engagement",
            ),
        ),
        CanonPromoted(
            key="name",
            entry=PlayerCanonEntry(
                summary="Marcus", literal_text_quote="I'm Marcus",
                salience=0.9, category=CanonCategory.NAME, turn_of_origin=1,
            ),
        ),
        LiteralEffectApplied(
            delta=QualityDelta(
                quality="engagement", before=3.0, after=6.0,
                source="effect:nudge:engagement",
            ),
        ),
        BeatFired(beat_id="aria.terminal", beat_kind="one_shot", completed=True),
        SessionEnded(beat_id="aria.terminal"),
    ]


def test_deterministic_replay_matches_recorded_state(small_scenario):
    """Live session and replayed session land on identical state."""
    live = _make_session(small_scenario)
    for ev in _scripted_events():
        live.apply(ev)

    replayed = _make_session(small_scenario)
    replay_into_session(replayed, list(live.event_log))

    # State equality across all fields the fold cases write.
    assert replayed.state.qualities == live.state.qualities
    assert replayed.state.fired_beat_ids == live.state.fired_beat_ids
    assert replayed.state.ended == live.state.ended
    assert replayed.state.end_reason == live.state.end_reason
    assert [e.id for e in replayed.state.emails] == [e.id for e in live.state.emails]
    assert [r.id for r in replayed.state.replies] == [r.id for r in live.state.replies]
    assert (
        replayed.state.player_canon["name"].summary
        == live.state.player_canon["name"].summary
    )
    # Replayed reply received the same classifier verdict as the live one.
    assert (
        [(a.quality, a.value) for a in replayed.state.replies[0].classifier_results]
        == [(a.quality, a.value) for a in live.state.replies[0].classifier_results]
    )


def test_deterministic_replay_event_log_round_trips_through_json(small_scenario):
    """Pydantic discriminated-union round-trip: dump every event to JSON,
    re-validate, fold into a fresh session, assert state matches."""
    from pydantic import TypeAdapter
    adapter: TypeAdapter[Event] = TypeAdapter(Event)

    live = _make_session(small_scenario)
    for ev in _scripted_events():
        live.apply(ev)

    json_blobs = [adapter.dump_json(ev) for ev in live.event_log]
    rehydrated = [adapter.validate_json(blob) for blob in json_blobs]

    replayed = _make_session(small_scenario)
    replay_into_session(replayed, rehydrated)

    assert replayed.state.qualities == live.state.qualities
    assert replayed.state.fired_beat_ids == live.state.fired_beat_ids
    assert replayed.state.ended == live.state.ended
    assert replayed.state.end_reason == live.state.end_reason


def test_state_at_intermediate_index_matches_partial_fold(small_scenario):
    """The Timeline tab depends on `state_at(N)` matching what the live
    session's state looked like after exactly N applies. Fold has no hidden
    state beyond what's in InboxState, so this should always hold."""
    sess = _make_session(small_scenario)
    events = _scripted_events()
    for ev in events:
        sess.apply(ev)

    # Snapshot state at the boundary BETWEEN classify-write and canon-write.
    # That's index 4 (after EmailEmitted, BeatFired, SendRecorded, Classify):
    snap = sess.state_at(4)
    assert snap.qualities["engagement"] == 3.0
    assert "name" not in snap.player_canon  # canon_promoted hasn't fired yet
    assert "aria.welcome" in snap.fired_beat_ids
    assert "aria.terminal" not in snap.fired_beat_ids
    assert not snap.ended
