"""Unit tests for the Event union + fold + Session.

Every event type has a fold case; this file covers each one in isolation
plus the round-trip through Pydantic JSON to confirm the discriminated
union deserializes cleanly (the property the save/load feature depends on).
"""
from __future__ import annotations

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelResponse, TextPart

from npc_service.events import (
    BeatFired,
    CanonPromoted,
    ChatBodyRendered,
    ClassifierObserved,
    ClassifyEffectApplied,
    DirectorChose,
    Email,
    EmailEmitted,
    EmailRead,
    Event,
    ExtractorObserved,
    LiteralEffectApplied,
    Reply,
    SendRecorded,
    SessionEnded,
    ThreadHistoryAppended,
    fold,
)
from npc_service.models import (
    AxisResult,
    CanonCandidate,
    CanonCategory,
    InboxState,
    PlayerCanonEntry,
    QualityDelta,
    Scenario,
)
from npc_service.session import Session, make_session


def _scenario() -> Scenario:
    return Scenario.model_validate({
        "title": "T",
        "premise": "p",
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
        ],
    })


def _state() -> InboxState:
    return InboxState.from_scenario(_scenario())


def _email(eid: str = "e1", body: str = "hi") -> Email:
    return Email(
        id=eid, beat_id=None, thread_id="aria",
        folder="Inbox", sender="aria@x", recipient="p@x",
        body=body, sent_at_label="now",
    )


def _reply(rid: str = "r1") -> Reply:
    return Reply(
        id=rid, thread_id="aria", beat_responded_to=None,
        recipient="aria@x", body="hello",
    )


def test_send_recorded_appends_email_and_reply():
    s = _state()
    fold(s, SendRecorded(email=_email("e1"), reply=_reply("r1")))
    assert len(s.emails) == 1 and s.emails[0].id == "e1"
    assert len(s.replies) == 1 and s.replies[0].id == "r1"


def test_email_emitted_appends_to_emails():
    s = _state()
    fold(s, EmailEmitted(email=_email("e2")))
    assert [e.id for e in s.emails] == ["e2"]


def test_email_read_flips_is_read():
    s = _state()
    fold(s, EmailEmitted(email=_email("e1")))
    assert s.emails[0].is_read is False
    fold(s, EmailRead(email_id="e1"))
    assert s.emails[0].is_read is True


def test_thread_history_appended():
    s = _state()
    msg = ModelResponse(parts=[TextPart(content="seed body")])
    fold(s, ThreadHistoryAppended(thread_id="aria", messages=[msg]))
    assert "aria" in s.thread_messages
    assert len(s.thread_messages["aria"]) == 1


def test_literal_effect_applied_writes_quality():
    s = _state()
    fold(s, LiteralEffectApplied(
        delta=QualityDelta(quality="engagement", before=0.0, after=2.5, source="effect:b:engagement"),
        beat_id="b",
    ))
    assert s.qualities["engagement"] == 2.5


def test_classify_effect_applied_appends_axis_and_writes_quality():
    s = _state()
    fold(s, SendRecorded(email=_email("e1"), reply=_reply("r1")))
    axis = AxisResult(quality="engagement", value="warm", rationale="r")
    delta = QualityDelta(quality="engagement", before=0.0, after=3.0, source="classify:engagement")
    fold(s, ClassifyEffectApplied(reply_id="r1", axis=axis, delta=delta))
    assert s.qualities["engagement"] == 3.0
    assert s.replies[0].classifier_results == [axis]


def test_classify_effect_applied_with_no_delta_still_records_axis():
    s = _state()
    fold(s, SendRecorded(email=_email("e1"), reply=_reply("r1")))
    axis = AxisResult(quality="engagement", value="cold", rationale="")
    fold(s, ClassifyEffectApplied(reply_id="r1", axis=axis, delta=None))
    assert s.qualities["engagement"] == 0.0  # unchanged
    assert s.replies[0].classifier_results == [axis]


def test_canon_promoted_writes_entry():
    s = _state()
    entry = PlayerCanonEntry(
        summary="Marcus", literal_text_quote="I'm Marcus",
        salience=0.9, category=CanonCategory.NAME, turn_of_origin=1,
    )
    fold(s, CanonPromoted(key="name", entry=entry))
    assert s.player_canon["name"] is entry


def test_beat_fired_completed_adds_to_set():
    s = _state()
    fold(s, BeatFired(beat_id="aria.welcome", beat_kind="one_shot", completed=True))
    assert "aria.welcome" in s.fired_beat_ids


def test_beat_fired_not_completed_skips():
    s = _state()
    fold(s, BeatFired(beat_id="aria.scene", beat_kind="scene", scene_turn=1, completed=False))
    assert s.fired_beat_ids == set()


def test_session_ended_sets_flags():
    s = _state()
    fold(s, SessionEnded(beat_id="aria.terminal"))
    assert s.ended is True and s.end_reason == "aria.terminal"


def test_observation_events_are_no_op_in_fold():
    s = _state()
    before = (set(s.fired_beat_ids), dict(s.qualities), len(s.emails))
    fold(s, ClassifierObserved(reply_id="r1", verdicts=[]))
    fold(s, ExtractorObserved(reply_id="r1", candidates=[], drops=[]))
    fold(s, DirectorChose(selection_method="none", chosen_beat_id=None, eligible=[]))
    fold(s, ChatBodyRendered(
        beat_id=None, npc_id="aria", thread_id="aria", body="hi",
    ))
    after = (set(s.fired_beat_ids), dict(s.qualities), len(s.emails))
    assert before == after


def test_event_round_trips_through_json():
    """Discriminated union deserializes cleanly — the property save/load relies on."""
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    events: list[Event] = [
        SendRecorded(email=_email("e1"), reply=_reply("r1")),
        LiteralEffectApplied(
            delta=QualityDelta(quality="engagement", before=0.0, after=2.0, source="effect:b:engagement"),
            beat_id="b",
        ),
        BeatFired(beat_id="aria.welcome", beat_kind="one_shot", completed=True),
        SessionEnded(beat_id="aria.terminal"),
    ]
    for ev in events:
        blob = adapter.dump_json(ev)
        loaded = adapter.validate_json(blob)
        assert type(loaded) is type(ev)
        assert loaded.kind == ev.kind


def test_session_apply_and_state_at():
    scenario = _scenario()
    # AgentBundle is constructed without instances; use a minimal stub via dataclass default.
    # For this test we don't run any agents; just verify the apply/state_at mechanics.
    from npc_service.agents import AgentBundle
    bundle = AgentBundle.__new__(AgentBundle)  # bypass __init__; the test never calls agents
    sess = make_session(scenario, bundle)

    e1 = SendRecorded(email=_email("e1"), reply=_reply("r1"))
    e2 = LiteralEffectApplied(
        delta=QualityDelta(quality="engagement", before=0.0, after=4.0, source="effect:b:engagement"),
    )
    e3 = BeatFired(beat_id="aria.welcome", beat_kind="one_shot", completed=True)
    sess.apply(e1)
    sess.apply(e2)
    sess.apply(e3)

    assert len(sess.event_log) == 3
    assert sess.state.qualities["engagement"] == 4.0
    assert "aria.welcome" in sess.state.fired_beat_ids

    # Time travel: state at index 2 should reflect e1+e2 but not e3.
    snap = sess.state_at(2)
    assert snap.qualities["engagement"] == 4.0
    assert "aria.welcome" not in snap.fired_beat_ids
    # Live state untouched.
    assert "aria.welcome" in sess.state.fired_beat_ids


def test_state_at_bounds():
    from npc_service.agents import AgentBundle
    bundle = AgentBundle.__new__(AgentBundle)
    sess = make_session(_scenario(), bundle)
    # Empty log: index 0 returns a fresh state.
    snap = sess.state_at(0)
    assert snap.emails == []

    import pytest as _pytest
    with _pytest.raises(IndexError):
        sess.state_at(-1)
    with _pytest.raises(IndexError):
        sess.state_at(99)
