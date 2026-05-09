"""FastAPI smoke tests with Pydantic AI's TestModel — no llama-server, no
real LLM. Verifies wiring: lifespan boots, the welcome anchored beat lands,
/open is read-only, /send rejects empty bodies and runs the classifier on
beats that author classify_reply, the scene loop re-fires across sends,
engagement accumulates, and ends_session terminals flip state.ended.
"""
import json
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

import npc_service.main as main
from npc_service.agents import ChatDeps, build_agents
from npc_service.models import BeatSpec, NPCDossier

from .conftest import consume_sse


CHAT_TURN_PROSE = (
    "Thank you for your reply. I have logged it as a calibration sample "
    "and look forward to our continued partnership."
)


@pytest.fixture
def client(monkeypatch):
    """Boot the app with no llama-server and override agents with TestModels
    that return canned outputs.

    Loads `tests/fixtures/test_scenario.yaml` (a synthesised single-quality
    Continuity-Program-shaped scenario) instead of the production
    scenario.yaml — keeps app-wiring tests focused on engine plumbing
    rather than story content.

    Default classifier value is 'deflected' so engagement starts moving
    backwards by default; individual tests override per-call when they need
    'engaged' to push past the warm-ending threshold.

    `_select_beat` is monkeypatched to "first eligible" so beat selection
    is deterministic in tests — replicates the old extra_fired ordering
    (terminals are authored before the loop in the fixture, so first-eligible
    picks the terminal when both are due). The real Director (LLM-driven)
    is exercised via integration tests against llama.
    """
    from pathlib import Path
    fixture_path = Path(__file__).parent / "fixtures" / "test_scenario.yaml"
    monkeypatch.setenv("SCENARIO_PATH", str(fixture_path))
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)

    async def _first_eligible(due, *, reply_thread_id=None):
        # Mirrors the production `_select_beat` 3-tuple return shape:
        # (chosen, selection-method, rationale). Selection is "sole" since
        # we deterministically pick the first due — bypasses the Director
        # AND the scene-continuity short-circuit (we want first-eligible
        # for fixture determinism regardless of thread state).
        if not due:
            return None, "none", None
        return due[0], "sole", None
    monkeypatch.setattr(main, "_select_beat", _first_eligible)

    with TestClient(main.app) as c:
        assert main.AGENTS is not None
        with ExitStack() as stack:
            stack.enter_context(main.AGENTS.chat_agent.override(model=_test_model_for_chat()))
            stack.enter_context(main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("deflected")))
            # Without this override the extractor_agent runs against the
            # OpenAI provider configured for localhost:8080 on every /send
            # (asyncio.gather'd with the classifier). When llama isn't up,
            # each call hangs on httpx connect/read until timing out — adding
            # ~5-7 s per send and dragging the suite past 4 minutes.
            stack.enter_context(main.AGENTS.extractor_agent.override(model=_test_model_for_extractor_empty()))
            # Fact-checker auto-engages whenever a beat declares
            # must_include_facts / must_not (the test fixture's loop +
            # terminals all do). Default to a clean pass so the validator
            # is a no-op for tests that don't care; per-test overrides
            # exercise the flagged path.
            stack.enter_context(main.AGENTS.fact_checker_agent.override(
                model=_test_model_for_fact_checker_clean(),
            ))
            yield c


def _test_model_for_chat(text: str = CHAT_TURN_PROSE) -> TestModel:
    return TestModel(custom_output_text=text)


# NativeOutput requires the model to advertise json-schema support; TestModel's
# default profile is tool-mode only.
_NATIVE_OUTPUT_PROFILE = ModelProfile(
    supports_json_schema_output=True, default_structured_output_mode="native"
)


def _test_model_for_classifier(value: str, quality: str = "engagement") -> TestModel:
    """Single-axis convenience — returns one AxisResult inside the list
    output type the multi-axis classifier now uses. Continuity Program
    tests use the default quality='engagement' for aria.loop's only axis.

    The classifier uses NativeOutput; Pydantic AI wraps a `list[X]` schema
    as `{"response": [...]}` for OpenAI-compat structured outputs."""
    return TestModel(
        custom_output_text=json.dumps({
            "response": [{"quality": quality, "value": value, "rationale": "(test)"}],
        }),
        profile=_NATIVE_OUTPUT_PROFILE,
    )


def _test_model_for_classifier_multi(results: list[dict]) -> TestModel:
    """Multi-axis variant — pass a list of {quality, value, rationale} dicts.
    Used by tests that exercise the batched per-axis flow and the
    defense-in-depth paths (missing / duplicate / unknown axes)."""
    return TestModel(
        custom_output_text=json.dumps({"response": results}),
        profile=_NATIVE_OUTPUT_PROFILE,
    )


def _test_model_for_fact_checker_clean() -> TestModel:
    """Fact-checker that always returns a clean verdict (no missing facts,
    no violations). Used as the default in the `client` fixture so the
    auto-engaged validator no-ops unless a test explicitly overrides."""
    return TestModel(
        custom_output_text=json.dumps({"missing_facts": [], "violations": []}),
        profile=_NATIVE_OUTPUT_PROFILE,
    )


def _test_model_for_extractor_empty() -> TestModel:
    """Stub the extractor to return zero candidates so app-wiring tests
    don't trigger player_canon writes nor wait on a real LLM call."""
    return TestModel(
        custom_output_text=json.dumps({"candidates": []}),
        profile=_NATIVE_OUTPUT_PROFILE,
    )


# ───────────────────────── boot + read-only endpoints ─────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["scenario_loaded"] is True


def test_scenario(client):
    r = client.get("/scenario")
    assert r.status_code == 200
    j = r.json()
    assert j["title"] == "Test Scenario"
    assert "premise" in j
    # The /scenario endpoint now returns the npcs list with thread mapping
    # and replies_enabled — used by the frontend to render multi-NPC chats.
    assert "npcs" in j
    assert any(n["id"] == "aria" for n in j["npcs"])
    aria = next(n for n in j["npcs"] if n["id"] == "aria")
    assert aria["thread_id"] == "aria"
    assert aria["replies_enabled"] is True


def test_state_starts_clean(client):
    r = client.get("/state").json()
    assert r["ended"] is False
    assert r["end_reason"] is None


def test_boot_inbox_has_one_email(client):
    r = client.get("/inbox")
    assert r.status_code == 200
    j = r.json()
    inbox = j["folders"]["Inbox"]
    assert len(inbox) == 1
    assert any("Aria" in e["from"] for e in inbox), [e["from"] for e in inbox]


def test_boot_seeds_thread_messages(client):
    """The anchored aria.welcome fires at boot. Its body lands in
    thread_messages["aria"] as a ModelResponse(TextPart) so the chat agent's
    next run sees Aria's welcome as her prior turn."""
    history = main.STATE.thread_messages.get("aria", [])
    assert len(history) == 1
    assert isinstance(history[0], ModelResponse)
    assert any(isinstance(p, TextPart) for p in history[0].parts)


# ───────────────────────── /open ─────────────────────────


def test_open_is_read_only(client):
    """/open marks the email read but mutates no qualities — engagement
    stays at default after /open."""
    inbox = client.get("/inbox").json()
    welcome = inbox["folders"]["Inbox"][0]
    initial_engagement = main.STATE.qualities["engagement"]
    r = client.post("/open", json={"email_id": welcome["id"]})
    assert r.status_code == 200
    assert main.STATE.qualities["engagement"] == initial_engagement


def test_open_marks_email_read(client):
    inbox = client.get("/inbox").json()
    welcome = inbox["folders"]["Inbox"][0]
    client.post("/open", json={"email_id": welcome["id"]})
    inbox_after = client.get("/inbox").json()
    welcome_after = inbox_after["folders"]["Inbox"][0]
    assert welcome_after["is_read"] is True


# ───────────────────────── /send ─────────────────────────


def test_send_rejects_empty_body(client):
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    r = client.post("/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "   ",
    })
    assert r.status_code == 400


def test_first_send_fires_loop(client):
    """After the first reply (replies≥1 satisfied), aria.loop becomes
    eligible and renders. Welcome has no classify_reply so the classifier
    doesn't run on this send."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to be here",
    })))
    new_email_events = [e for e in events if e["type"] == "new_email"]
    loop_emails = [
        e["email"] for e in new_email_events
        if e["email"]["beat_id"] == "aria.loop"
    ]
    assert len(loop_emails) == 1
    # Scene beat: doesn't enter fired_beat_ids until exit_when satisfied.
    assert "aria.loop" not in main.STATE.fired_beat_ids
    # Welcome has no classify_reply → no quality moved.
    assert main.STATE.qualities["engagement"] == 0.0
    assert main.STATE.replies[-1].classifier_results == []


def test_send_emits_tick_decision_event(client):
    """Every successful /send emits a structured `tick_decision` SSE event
    after `new_email` and before `state` + `done`. Frontend's debug panel
    consumes it; backend always emits regardless of client. Shape mirrors
    the `TickDecision` Pydantic model — eligibility report, selection
    method, fired beat id, scene turn count."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to be here",
    })))
    decisions = [e for e in events if e["type"] == "tick_decision"]
    assert len(decisions) == 1
    decision = decisions[0]["decision"]

    # Required fields present (full TickDecision shape).
    for field in (
        "in_response_to", "eligible", "selection",
        "chosen_beat_id", "quality_deltas", "fired_beat_id",
    ):
        assert field in decision

    # First send fired aria.loop (scene, turn 1) under the test
    # harness's first-eligible Director stub.
    assert decision["fired_beat_id"] == "aria.loop"
    assert decision["chosen_beat_id"] == "aria.loop"
    assert decision["scene_turn"] == 1
    assert decision["selection"] == "sole"

    # Eligibility report includes aria.loop as eligible.
    eligible_ids = {be["beat_id"] for be in decision["eligible"]}
    assert "aria.loop" in eligible_ids

    # Event ordering: tick_decision lands after new_email and before state+done.
    types = [e["type"] for e in events]
    assert types.index("tick_decision") > types.index("new_email")
    assert types.index("tick_decision") < types.index("state")
    assert types.index("state") < types.index("done")


def test_scene_retired_when_exit_when_already_satisfied_on_entry(client):
    """A scene whose `exit_when` is satisfied at selection time gets retired
    without firing — `_tick` adds it to `fired_beat_ids`, marks the
    eligibility report entry `retired`, and falls through to the chat-turn
    fallback so the player still gets an in-character reply.

    Regression: previously such a scene would fire once and exit on the
    same tick (e.g. calibration_scene asking one calibration question
    after the welcome reply already canonized the player's name)."""
    # aria.loop's exit_when is `engagement>=4.0 OR replies>=8`. Bumping
    # engagement above the threshold pre-emptively means the scene's
    # exit_when is true the first time it would have fired.
    main.STATE.qualities["engagement"] = 4.0

    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"],
        "recipient": "Aria",
        "body": "happy to be here",
    })))

    # aria.loop landed in fired_beat_ids — but never produced a new_email
    # under that beat_id. The chat-turn fallback covered the thread with
    # a chat-turn email (beat_id=None).
    assert "aria.loop" in main.STATE.fired_beat_ids
    new_email_events = [e for e in events if e["type"] == "new_email"]
    assert all(
        e["email"]["beat_id"] != "aria.loop" for e in new_email_events
    ), "aria.loop should not have fired"
    assert any(
        e["email"]["beat_id"] is None for e in new_email_events
    ), "chat-turn fallback should have fired"

    # tick_decision reflects the retirement.
    decision = next(e for e in events if e["type"] == "tick_decision")["decision"]
    loop_entry = next(be for be in decision["eligible"] if be["beat_id"] == "aria.loop")
    assert loop_entry["status"] == "retired"
    assert decision["fired_beat_id"] is None
    assert decision["chosen_beat_id"] is None
    assert decision["selection"] == "none"


def test_loop_scene_re_fires_across_sends(client):
    """Two /sends to Aria. The loop is kind=scene, so it re-fires every
    tick until exit_when promotes it. thread_messages accumulates:
    welcome (ModelResponse) + two (ModelRequest, ModelResponse) pairs =
    5 entries."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello there",
    })))
    # Reply to the loop's email this time so the classifier runs.
    loop_email = next(
        e for e in main.STATE.emails if e.beat_id == "aria.loop"
    )
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": loop_email.id, "recipient": "Aria", "body": "another note",
    })))

    # Two loop emails should now exist.
    loop_emails = [e for e in main.STATE.emails if e.beat_id == "aria.loop"]
    assert len(loop_emails) == 2

    history = main.STATE.thread_messages["aria"]
    assert len(history) == 5
    assert isinstance(history[0], ModelResponse)
    assert isinstance(history[1], ModelRequest)
    assert isinstance(history[2], ModelResponse)
    assert isinstance(history[3], ModelRequest)
    assert isinstance(history[4], ModelResponse)


def test_loop_body_does_not_leak_email_headers(client):
    """Regression guard: with output_type=str (no JSON schema enforcement)
    the chat agent could in principle drift into emitting `Subject:` or
    `From:` headers. The base instructions forbid this."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello",
    })))
    loop_emails = [
        e["email"] for e in events
        if e["type"] == "new_email" and e["email"]["beat_id"] == "aria.loop"
    ]
    assert loop_emails
    body = loop_emails[0]["body"]
    for forbidden in ("Subject:", "From:", "To:", "Date:", "```"):
        assert forbidden not in body, f"loop body leaked {forbidden!r}: {body!r}"


def test_classifier_runs_on_reply_to_loop_and_accumulates(client):
    """Replying to a loop email triggers classify_reply. The classifier
    writes to `engagement` with `accumulate=True`, so each engaged reply
    pushes the quality up by 1.0."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    # First send: replies to welcome → loop fires; welcome has no classifier.
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    assert main.STATE.qualities["engagement"] == 0.0
    loop_email = next(e for e in main.STATE.emails if e.beat_id == "aria.loop")

    # Override classifier to 'engaged' for the next two sends; engagement
    # should accumulate to 2.0.
    with main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("engaged")):
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": loop_email.id, "recipient": "Aria",
            "body": "Tuesday afternoons I usually triage email, then deep work.",
        })))
        # Per-axis classifier write landed on the just-recorded reply.
        first_results = main.STATE.replies[-1].classifier_results
        assert len(first_results) == 1
        assert first_results[0].quality == "engagement"
        assert first_results[0].value == "engaged"

        next_loop_email = next(
            e for e in reversed(main.STATE.emails) if e.beat_id == "aria.loop"
        )
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": next_loop_email.id, "recipient": "Aria",
            "body": "and Wednesdays are deep work mornings.",
        })))
    assert main.STATE.qualities["engagement"] == pytest.approx(2.0)


def test_complete_warm_fires_and_ends_session(client):
    """Engineer the warm ending: 5 replies on the aria thread, classifier
    returning 'engaged' so engagement accumulates past the 4.0 threshold.
    On the 5th send (replies≥5 AND engagement≥4.0), complete_warm fires
    and state.ended flips."""
    with main.AGENTS.classifier_agent.override(model=_test_model_for_classifier("engaged")):
        _play_five_replies(client)

    assert main.STATE.ended is True
    assert main.STATE.end_reason == "aria.complete_warm"
    warm_emails = [e for e in main.STATE.emails if e.beat_id == "aria.complete_warm"]
    assert len(warm_emails) == 1


def test_complete_standard_fires_when_engagement_low(client):
    """5 replies but classifier always 'deflected' → engagement floors at 0.0,
    complete_warm precondition fails, complete_standard wins via mutex."""
    # Default fixture classifier is 'deflected' — no override needed.
    _play_five_replies(client)

    assert main.STATE.ended is True
    assert main.STATE.end_reason == "aria.complete_standard"
    standard_emails = [
        e for e in main.STATE.emails if e.beat_id == "aria.complete_standard"
    ]
    assert len(standard_emails) == 1
    assert main.STATE.qualities["engagement"] == 0.0  # floor-clamped


def test_send_emits_structured_error_when_chat_agent_fails(client, monkeypatch):
    """When chat_agent.run raises, _send_stream must emit a structured SSE
    `error` event (matching `web/src/api.ts` SseEvent.error) followed by a
    final state + done — never a torn HTTP stream. The player's email is
    already persisted so the client sees their send accepted before the
    error.

    Replaces the real chat_agent.run with one that raises RuntimeError so
    the test exercises the ChatAgentFailure → SSE error path without
    needing to break the TestModel internals.
    """
    async def _raise(*args, **kwargs):
        raise RuntimeError("simulated llama failure")

    monkeypatch.setattr(main.AGENTS.chat_agent, "run", _raise)
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))

    # Player's send is recorded first.
    assert events[0]["type"] == "send_recorded"

    # Exactly one error event with a player-friendly message.
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert "Aria" in errors[0]["message"]

    # Stream still terminates cleanly with state + done.
    assert events[-2]["type"] == "state"
    assert events[-1]["type"] == "done"

    # No NPC reply was emitted (rendering failed before any new_email event).
    assert not any(e["type"] == "new_email" for e in events)

    # Player email IS in state — the send was accepted before the failure.
    assert any(e.body == "hi" and e.sender == "you" for e in main.STATE.emails)


def test_send_returns_409_after_session_ends(client):
    """Once a terminal beat fires, /send rejects further activity."""
    _play_five_replies(client)

    assert main.STATE.ended is True
    last_npc = next(e for e in reversed(main.STATE.emails) if e.sender != "you")
    r = client.post("/send", json={
        "in_reply_to": last_npc.id, "recipient": "Aria", "body": "more?",
    })
    assert r.status_code == 409


# ───────────────────────── multi-axis classifier defense-in-depth ─────────────────────────


def test_classifier_handles_unknown_axis_returned_by_model(client):
    """LLM returns a result for a quality the beat didn't author. The
    unknown axis is logged and dropped; the authored axis still applies."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    loop_email = next(e for e in main.STATE.emails if e.beat_id == "aria.loop")

    multi = _test_model_for_classifier_multi([
        {"quality": "engagement", "value": "engaged", "rationale": "(t)"},
        {"quality": "phantom", "value": "anything", "rationale": "(t)"},
    ])
    with main.AGENTS.classifier_agent.override(model=multi):
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": loop_email.id, "recipient": "Aria", "body": "yes",
        })))
    # Only the authored 'engagement' axis applied.
    results = main.STATE.replies[-1].classifier_results
    assert len(results) == 1
    assert results[0].quality == "engagement"
    assert main.STATE.qualities["engagement"] == 1.0


def test_classifier_handles_duplicate_axis_returned_by_model(client):
    """LLM returns two results for the same authored axis. First kept,
    second logged."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    loop_email = next(e for e in main.STATE.emails if e.beat_id == "aria.loop")

    multi = _test_model_for_classifier_multi([
        {"quality": "engagement", "value": "engaged", "rationale": "(first)"},
        {"quality": "engagement", "value": "deflected", "rationale": "(dupe)"},
    ])
    with main.AGENTS.classifier_agent.override(model=multi):
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": loop_email.id, "recipient": "Aria", "body": "yes",
        })))
    results = main.STATE.replies[-1].classifier_results
    assert len(results) == 1
    assert results[0].value == "engaged"  # first kept


def test_classifier_handles_off_spec_value(client):
    """Model returns a value outside allowed_values. apply_classify_effect
    returns None; no append onto Reply.classifier_results, quality unchanged."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    loop_email = next(e for e in main.STATE.emails if e.beat_id == "aria.loop")

    multi = _test_model_for_classifier_multi([
        {"quality": "engagement", "value": "garbage_value", "rationale": "(t)"},
    ])
    starting_engagement = main.STATE.qualities["engagement"]
    with main.AGENTS.classifier_agent.override(model=multi):
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": loop_email.id, "recipient": "Aria", "body": "yes",
        })))
    assert main.STATE.qualities["engagement"] == starting_engagement
    # The reply was recorded but its classifier_results stayed empty.
    last_reply = main.STATE.replies[-1]
    assert last_reply.classifier_results == []


def test_sse_event_order_send_recorded_then_new_email_then_state(client):
    """SSE ordering: send_recorded first, then new_email(s), then state,
    then done."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    types = [e["type"] for e in events]
    sr_idx = types.index("send_recorded")
    new_email_idx = types.index("new_email")
    state_idx = types.index("state")
    done_idx = types.index("done")
    assert sr_idx < new_email_idx < state_idx < done_idx


def test_apply_classification_appends_to_active_reply_via_send(client):
    """Smoke test for the Reply.classifier_results plumbing through /send:
    after a successful classification, the just-recorded Reply has the
    AxisResult."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    loop_email = next(e for e in main.STATE.emails if e.beat_id == "aria.loop")

    with main.AGENTS.classifier_agent.override(
        model=_test_model_for_classifier("engaged")
    ):
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": loop_email.id, "recipient": "Aria", "body": "yes",
        })))
    last_reply = main.STATE.replies[-1]
    assert len(last_reply.classifier_results) == 1
    assert last_reply.classifier_results[0].quality == "engagement"
    assert last_reply.classifier_results[0].value == "engaged"


# ───────────────────────── render validator (fact-checker) ─────────────────────────


def _validator_test_bundle():
    """Construct an isolated AgentBundle for unit-testing the chat-agent
    output_validator. Chat agent and fact-checker need different TestModel
    output shapes, so each test overrides them in its own ExitStack."""
    placeholder = OpenAIChatModel(
        "test", provider=OpenAIProvider(base_url="http://x", api_key="k")
    )
    return build_agents(placeholder)


def _validator_test_deps() -> ChatDeps:
    return ChatDeps(
        npc=NPCDossier(id="aria", display_name="Aria", email_address="aria@x"),
        plot_directive=BeatSpec(
            intent="x",
            must_include_facts=["thank the player"],
        ),
    )


@pytest.mark.asyncio
async def test_validator_passes_when_fact_checker_returns_clean():
    """must_include_facts present + fact-checker returning empty lists →
    body passes through unchanged."""
    bundle = _validator_test_bundle()
    with ExitStack() as stack:
        stack.enter_context(bundle.chat_agent.override(
            model=TestModel(custom_output_text="ok"),
        ))
        stack.enter_context(bundle.fact_checker_agent.override(
            model=TestModel(
                custom_output_text=json.dumps({
                    "missing_facts": [], "violations": [],
                }),
                profile=_NATIVE_OUTPUT_PROFILE,
            ),
        ))
        result = await bundle.chat_agent.run("hello", deps=_validator_test_deps())
    assert result.output.strip() == "ok"


@pytest.mark.asyncio
async def test_validator_raises_when_fact_checker_flags():
    """must_include_facts present + non-empty missing_facts → ModelRetry.
    With chat_agent retries=2, exhaustion surfaces as
    UnexpectedModelBehavior; capturing the raise confirms the validator
    engaged."""
    bundle = _validator_test_bundle()
    with ExitStack() as stack:
        stack.enter_context(bundle.chat_agent.override(
            model=TestModel(custom_output_text="ok"),
        ))
        stack.enter_context(bundle.fact_checker_agent.override(
            model=TestModel(
                custom_output_text=json.dumps({
                    "missing_facts": ["thank the player"], "violations": [],
                }),
                profile=_NATIVE_OUTPUT_PROFILE,
            ),
        ))
        with pytest.raises(UnexpectedModelBehavior):
            await bundle.chat_agent.run("hello", deps=_validator_test_deps())


# ───────────────────────── /scenario_meta ─────────────────────────


def test_scenario_meta_returns_qualities_beats_and_graph(client):
    """The DEV-only debug overlay fetches /scenario_meta once at boot to
    populate hover tooltips (quality descriptions, beat metadata) and the
    scenario-graph view. Shape must be stable: each section keyed by id,
    edges have from/to/kind."""
    payload = client.get("/scenario_meta").json()
    assert "qualities" in payload
    assert "beats" in payload
    assert "beat_graph" in payload

    # Quality entries carry the authored description + bounds + flags.
    eng = payload["qualities"]["engagement"]
    assert "description" in eng
    assert eng["type"] == "float"
    assert eng["surface_to_chat"] is False

    # Beat entries carry npc/thread/flags/summary.
    welcome = payload["beats"]["aria.welcome"]
    assert welcome["npc"] == "aria"
    assert welcome["thread"] == "aria"
    assert welcome["pre_authored"] is True

    # Graph: nodes for every beat, edges only for beats with BeatFired refs.
    node_ids = {n["id"] for n in payload["beat_graph"]["nodes"]}
    assert node_ids == set(payload["beats"].keys())
    # Mutex edge: aria.complete_standard's `Not_(BeatFired complete_warm)`
    # produces a mutex edge from complete_warm → complete_standard.
    edges = payload["beat_graph"]["edges"]
    mutex_pairs = {
        (e["from"], e["to"]) for e in edges if e["kind"] == "mutex"
    }
    assert ("aria.complete_warm", "aria.complete_standard") in mutex_pairs


def test_debug_scenario_returns_predicate_ast(client):
    """The DEV-only Beats inspector fetches /debug/scenario once at boot.
    Pairs with TickDecision.beats_evaluation: this endpoint ships the
    structural shape (entry_when AST, exit_when, effects); evaluation
    ships the verdicts. Frontend zips them per beat to render trees.
    """
    payload = client.get("/debug/scenario").json()
    assert "beats" in payload
    beats = payload["beats"]
    # Every scenario beat present.
    expected = {"aria.welcome", "aria.complete_warm", "aria.complete_standard", "aria.loop"}
    assert set(beats.keys()) == expected

    # Welcome is a pre-authored one_shot with no preconditions.
    welcome = beats["aria.welcome"]
    assert welcome["kind"] == "one_shot"
    assert welcome["preconditions"] == []
    assert welcome["exit_when"] == []

    # The scene beat has both preconditions and exit_when AST.
    loop = beats["aria.loop"]
    assert loop["kind"] == "scene"
    assert len(loop["preconditions"]) > 0
    assert len(loop["exit_when"]) > 0
    # Predicates serialize via discriminated union; `kind` field carries
    # the discriminator. First precondition is player_replies_to_thread_gte.
    assert loop["preconditions"][0]["kind"] == "player_replies_to_thread_gte"


def test_send_emits_beats_evaluation(client):
    """tick_decision.beats_evaluation covers EVERY scenario beat, not just
    eligible+blocked. Structurally-blocked beats (like complete_standard
    when complete_warm hasn't fired but the BeatFired-not condition still
    permits eligibility) appear with full traces. Drives the Beats
    inspector — author can select any beat and see entry_when atom-by-atom.
    """
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hi",
    })))
    decision = next(e for e in events if e["type"] == "tick_decision")["decision"]
    assert "beats_evaluation" in decision
    evals = decision["beats_evaluation"]
    eval_ids = {e["beat_id"] for e in evals}
    expected = {"aria.welcome", "aria.complete_warm", "aria.complete_standard", "aria.loop"}
    assert eval_ids == expected

    by_id = {e["beat_id"]: e for e in evals}
    # aria.welcome fired at boot, status flips to "fired".
    assert by_id["aria.welcome"]["status"] == "fired"
    # complete_warm and complete_standard remain blocked — replies < 5.
    assert by_id["aria.complete_warm"]["status"] == "blocked"
    # aria.loop is a scene; carries exit_traces parallel to its exit_when.
    loop_eval = by_id["aria.loop"]
    assert loop_eval["is_scene"] is True
    assert len(loop_eval["exit_traces"]) > 0
    # entry_traces parallel to preconditions (3 preconds on aria.loop).
    assert len(loop_eval["entry_traces"]) == 3


def test_send_emits_canon_and_classifier_history_on_decision(client):
    """A successful /send's tick_decision payload includes the new
    structured fields the debug overlay reads — canon snapshot, fired
    beat ids, classifier history, and (empty) canon_writes when the
    extractor stub returned no candidates."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    events = list(consume_sse(client.stream("POST", "/send", json={
        "in_reply_to": welcome["id"], "recipient": "Aria", "body": "hello",
    })))
    decision = next(e for e in events if e["type"] == "tick_decision")["decision"]
    # Structural fields the panel relies on — present (possibly empty).
    assert "canon_writes" in decision
    assert "canon_drops" in decision
    assert "player_canon" in decision
    assert "fact_check" in decision
    assert "classifier_history" in decision
    assert "fired_beat_ids" in decision
    # Stub returns zero candidates; canon_writes/drops should be empty.
    assert decision["canon_writes"] == []
    assert decision["canon_drops"] == []
    # Boot-tick fires aria.welcome (pre-authored); first /send fires
    # aria.loop. The welcome enters fired_beat_ids; the loop (kind=scene)
    # does NOT until its exit_when is satisfied.
    assert "aria.welcome" in decision["fired_beat_ids"]


# ───────────────────────── helpers ─────────────────────────


def _play_five_replies(client, body: str = "k") -> None:
    """Send five replies on the aria thread, chaining each to the most
    recent NPC email so the classifier always lands on the loop beat (or
    its terminal). Caller controls the classifier via Agent.override."""
    welcome = client.get("/inbox").json()["folders"]["Inbox"][0]
    last_in_reply_to = welcome["id"]
    for _ in range(5):
        list(consume_sse(client.stream("POST", "/send", json={
            "in_reply_to": last_in_reply_to, "recipient": "Aria", "body": body,
        })))
        last_npc = next(
            (e for e in reversed(main.STATE.emails) if e.sender != "you"),
            None,
        )
        if last_npc is not None:
            last_in_reply_to = last_npc.id
