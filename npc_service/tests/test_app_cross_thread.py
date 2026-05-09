"""Cross-thread routing tests — exercises the two-bug fix:

  Part A (scheduler-side filter): a scene that has already had at least
  one fire and whose thread differs from the player's reply thread is
  marked `deferred` and dropped from the eligible set, preventing the
  Director from re-firing it on consecutive same-thread player ticks.

  Part B (render-side reply/outbound split): when a chat-driven beat
  fires on a thread the player did NOT just reply to, the chat agent
  runs in `outbound_mode` — it receives a synthetic trigger (not the
  player's other-thread body) and `ChatDeps.outbound_mode=True` so the
  OUTBOUND_MODE_FRAMING block tells it to compose unprompted, never
  treating the trigger as input.

Uses a dedicated multi-NPC fixture (test_scenario_cross_thread.yaml) so
the assertions don't depend on the single-thread test_scenario.yaml that
the rest of test_app.py builds against.
"""
from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

import npc_service.main as main
from npc_service.agents import ChatDeps

from .conftest import consume_sse
from .test_app import _test_model_for_fact_checker_clean


CHAT_TURN_PROSE = "Acknowledged. Calibration sample logged."


@pytest.fixture
def client_multi(monkeypatch):
    """App boot against the cross-thread fixture. The chat agent returns
    canned prose so we can assert routing without llama. `_select_beat`
    is stubbed to first-eligible so beat selection is deterministic — the
    fixture orders marcus.intro_scene before aria.scene so Marcus wins
    the first cross-thread tick."""
    fixture = Path(__file__).parent / "fixtures" / "test_scenario_cross_thread.yaml"
    monkeypatch.setenv("SCENARIO_PATH", str(fixture))
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setattr(main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)

    async def _first_eligible(due, *, reply_thread_id=None):
        if not due:
            return None, "none", None
        return due[0], "sole", None
    monkeypatch.setattr(main, "_select_beat", _first_eligible)

    with TestClient(main.app) as c:
        assert main.AGENTS is not None
        with ExitStack() as stack:
            stack.enter_context(
                main.AGENTS.chat_agent.override(
                    model=TestModel(custom_output_text=CHAT_TURN_PROSE),
                ),
            )
            # The fixture authors a scenario-scope classify effect
            # (`thread: aria`) so the thread-filter test can observe a
            # gated dispatch — that means classifier_agent will run on any
            # aria-thread /send even though the beats themselves have no
            # classify-typed effects. Stub it so it doesn't reach
            # localhost:8080. Empty response list means no quality writes,
            # which keeps the unrelated routing tests in this file
            # behaviorally identical to their pre-fixture-extension state.
            #
            # The extractor likewise fires on every /send via asyncio.gather
            # and needs the same treatment.
            from pydantic_ai.profiles import ModelProfile
            import json
            _native_profile = ModelProfile(
                supports_json_schema_output=True,
                default_structured_output_mode="native",
            )
            stack.enter_context(
                main.AGENTS.classifier_agent.override(
                    model=TestModel(
                        custom_output_text=json.dumps({"response": []}),
                        profile=_native_profile,
                    ),
                ),
            )
            stack.enter_context(
                main.AGENTS.extractor_agent.override(
                    model=TestModel(
                        custom_output_text=json.dumps({"candidates": []}),
                        profile=_native_profile,
                    ),
                ),
            )
            # The fact-checker auto-engages on every beat fire (both scenes
            # in this fixture declare must_include_facts). Default to a
            # clean pass so the validator no-ops and routing tests stay
            # focused on dispatch rather than re-render retries.
            stack.enter_context(
                main.AGENTS.fact_checker_agent.override(
                    model=_test_model_for_fact_checker_clean(),
                ),
            )
            yield c


def _aria_welcome_id(client) -> str:
    inbox = client.get("/inbox").json()
    welcome = next(
        e for e in inbox["folders"]["Inbox"] if "Aria" in e["from"]
    )
    return welcome["id"]


def _install_chat_run_capture(monkeypatch) -> list[dict]:
    """Wrap chat_agent.run so each invocation appends one record per call:
    the user_prompt, which npc the run was for, and ChatDeps.outbound_mode.
    Tests call this after client_multi has booted (which sets up the agent
    overrides) so the wrapper sits on top of the overridden agent."""
    captured: list[dict] = []
    real_run = main.AGENTS.chat_agent.run

    async def _capturing_run(prompt, *args, deps=None, **kwargs):
        captured.append({
            "prompt": prompt,
            "outbound_mode": deps.outbound_mode if isinstance(deps, ChatDeps) else None,
            "npc_id": deps.npc.id if isinstance(deps, ChatDeps) else None,
        })
        return await real_run(prompt, *args, deps=deps, **kwargs)

    monkeypatch.setattr(main.AGENTS.chat_agent, "run", _capturing_run)
    return captured


# ───────────────────────── Part A: scheduler-side defer ─────────────────────────


def test_marcus_scene_deferred_after_first_fire_when_replying_to_aria(
    client_multi,
):
    """First aria-tick: marcus.intro_scene fires cross-thread (the
    fixture-ordered first-eligible stub picks it). Second aria-tick:
    marcus.intro_scene is now an in-flight off-thread scene, so the
    Part-A filter marks it `deferred`. Marcus must NOT emit a second
    email, and the eligibility report must surface the deferral."""
    welcome_id = _aria_welcome_id(client_multi)

    # Tick 1: Marcus introduces himself cross-thread.
    list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": welcome_id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": "hello aria",
    })))
    marcus_after_first = [
        e for e in main.STATE.emails if e.beat_id == "marcus.intro_scene"
    ]
    assert len(marcus_after_first) == 1, (
        "marcus.intro_scene should have fired once cross-thread on the "
        "first aria-tick under the first-eligible stub"
    )

    # Tick 2: reply to Aria again. marcus.intro_scene is in-flight off-thread.
    aria_email = next(
        e for e in main.STATE.emails
        if e.thread_id == "aria" and e.sender != "you" and e.beat_id != "aria.welcome"
    )
    events = list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": aria_email.id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": "another message to aria",
    })))

    decision = next(e for e in events if e["type"] == "tick_decision")["decision"]
    marcus_entry = next(
        be for be in decision["eligible"] if be["beat_id"] == "marcus.intro_scene"
    )
    assert marcus_entry["status"] == "deferred", (
        f"marcus.intro_scene should be deferred, got {marcus_entry['status']!r}"
    )
    assert "marcus" in (marcus_entry["blocked_by"] or ""), (
        f"deferred note should mention the awaited thread, "
        f"got {marcus_entry['blocked_by']!r}"
    )

    marcus_after_second = [
        e for e in main.STATE.emails if e.beat_id == "marcus.intro_scene"
    ]
    assert len(marcus_after_second) == 1, (
        "marcus.intro_scene must NOT have fired a second time during an "
        "aria-thread tick"
    )


def test_first_fire_of_offthread_scene_is_not_deferred(client_multi):
    """The defer filter must only catch scenes with prior fires. A scene's
    very first fire is allowed cross-thread (Marcus needs to introduce
    himself on an aria-tick because the player can't reply on the marcus
    thread before he sends anything). This asserts the first fire happens
    normally."""
    welcome_id = _aria_welcome_id(client_multi)
    events = list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": welcome_id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": "hello",
    })))
    decision = next(e for e in events if e["type"] == "tick_decision")["decision"]
    marcus_entry = next(
        be for be in decision["eligible"] if be["beat_id"] == "marcus.intro_scene"
    )
    # Marcus is either the chosen beat (eligible-then-fired) or marked
    # eligible but not chosen — but it must NOT be deferred on its first
    # fire opportunity.
    assert marcus_entry["status"] != "deferred"


# ───────────────────────── Part B: render-side outbound mode ─────────────────────────


def test_offthread_chat_render_runs_in_outbound_mode(client_multi, monkeypatch):
    """Marcus's first fire on an aria-tick must use outbound_mode: the
    chat agent receives the synthetic trigger as user_prompt and
    ChatDeps.outbound_mode=True. Without this, Marcus would read the
    player's aria-directed body as if it were addressed to him."""
    captured = _install_chat_run_capture(monkeypatch)

    welcome_id = _aria_welcome_id(client_multi)
    PLAYER_BODY = "secret aria-directed text that marcus should not see as input"
    list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": welcome_id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": PLAYER_BODY,
    })))

    # The marcus run must be present and must have fired in outbound mode.
    marcus_runs = [c for c in captured if c["npc_id"] == "marcus"]
    assert len(marcus_runs) == 1, (
        f"expected exactly one marcus run, captured: {captured!r}"
    )
    marcus_run = marcus_runs[0]
    assert marcus_run["outbound_mode"] is True, (
        "marcus's render must carry outbound_mode=True when firing on an "
        "aria-thread tick"
    )
    assert PLAYER_BODY not in marcus_run["prompt"], (
        f"player's aria-directed body leaked into marcus's user_prompt: "
        f"{marcus_run['prompt']!r}"
    )
    assert marcus_run["prompt"] == main._OUTBOUND_TRIGGER_PROMPT


def test_onthread_chat_render_runs_in_reply_mode(client_multi, monkeypatch):
    """Sanity inverse: when a beat fires on the player's reply thread, the
    chat agent receives the player's body as user_prompt and outbound_mode
    is False. Tests the chat-turn fallback path in particular — on the
    second aria-tick (after marcus is deferred and aria.scene fires),
    aria.scene's render is reply mode."""
    welcome_id = _aria_welcome_id(client_multi)
    # Tick 1: marcus fires cross-thread (outbound), aria gets chat-turn fallback (reply).
    list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": welcome_id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": "first",
    })))

    captured = _install_chat_run_capture(monkeypatch)

    # Tick 2: marcus is deferred, aria.scene fires on-thread (reply mode).
    aria_email = next(
        e for e in main.STATE.emails
        if e.thread_id == "aria" and e.sender != "you" and e.beat_id != "aria.welcome"
    )
    PLAYER_BODY = "second message addressed to aria"
    list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": aria_email.id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": PLAYER_BODY,
    })))

    aria_runs = [c for c in captured if c["npc_id"] == "aria"]
    assert aria_runs, f"aria should have rendered at least once, captured: {captured!r}"
    for run in aria_runs:
        assert run["outbound_mode"] is False, (
            "aria's on-thread render must be reply-mode (outbound_mode=False)"
        )
        # At least one aria run sees the player body verbatim (the
        # reply-mode fire). The chat-turn fallback would also see it.
    assert any(PLAYER_BODY in r["prompt"] for r in aria_runs), (
        f"player body should be the user_prompt in at least one aria reply-mode "
        f"run, got: {[r['prompt'] for r in aria_runs]!r}"
    )


def test_outbound_render_does_not_pollute_thread_history(client_multi):
    """The synthetic outbound trigger must NOT be persisted into the
    target thread's message_history — only the assistant response is
    appended. Without this, a later reply-mode render on the same thread
    would see the trigger as if the player had typed it."""
    welcome_id = _aria_welcome_id(client_multi)
    list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": welcome_id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": "hello",
    })))

    # marcus.intro_scene fired in outbound mode. The marcus thread history
    # should now contain ONLY the assistant response — no synthetic-
    # trigger ModelRequest, and no aria-directed body.
    marcus_history = main.STATE.thread_messages.get("marcus", [])
    from pydantic_ai.messages import ModelRequest, ModelResponse

    assert all(isinstance(m, ModelResponse) for m in marcus_history), (
        f"marcus thread history must contain only assistant turns after an "
        f"outbound fire, got types: {[type(m).__name__ for m in marcus_history]}"
    )
    # And critically, none of the message text should contain the trigger
    # sentinel or the player's other-thread body.
    rendered = " ".join(
        p.content for m in marcus_history for p in m.parts
        if hasattr(p, "content") and isinstance(p.content, str)
    )
    assert main._OUTBOUND_TRIGGER_PROMPT not in rendered
    assert "hello" not in rendered


# ───────────────────────── Part C: scenario-scope thread filter ─────────────────────────


def _install_classifier_run_capture(monkeypatch) -> list[dict]:
    """Wrap classifier_agent.run so each invocation appends a record. Used
    by the thread-filter test to assert classifier dispatch happens (or
    doesn't) per the `Effect.thread` gate."""
    captured: list[dict] = []
    real_run = main.AGENTS.classifier_agent.run

    async def _capturing_run(prompt, *args, **kwargs):
        captured.append({"prompt": prompt})
        return await real_run(prompt, *args, **kwargs)

    monkeypatch.setattr(main.AGENTS.classifier_agent, "run", _capturing_run)
    return captured


def test_scenario_classify_effect_fires_on_matching_thread_only(
    client_multi, monkeypatch,
):
    """The cross-thread fixture authors a scenario-scope classify effect
    with `thread: aria`. A reply on the aria thread should batch that
    effect into the classifier call; a reply on the marcus thread should
    drop it (no other classify effects → no classifier call at all)."""
    captured = _install_classifier_run_capture(monkeypatch)
    welcome_id = _aria_welcome_id(client_multi)

    # Tick 1: aria-thread reply. The scenario-scope effect's `thread: aria`
    # gate matches → classifier_agent runs once.
    list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": welcome_id,
        "recipient": "Aria",
        "recipient_npc_id": "aria",
        "body": "first reply on aria",
    })))
    aria_calls_after_first = len(captured)
    assert aria_calls_after_first == 1, (
        f"expected one classifier call on aria-thread reply, got "
        f"{aria_calls_after_first}"
    )

    # Tick 2: marcus-thread reply. The scenario-scope effect's gate does
    # NOT match. No beat-scope classify effects in this fixture either →
    # classify_effects ends up empty → classifier_agent must NOT run.
    marcus_email = next(
        e for e in main.STATE.emails
        if e.thread_id == "marcus" and e.sender != "you"
    )
    list(consume_sse(client_multi.stream("POST", "/send", json={
        "in_reply_to": marcus_email.id,
        "recipient": "Marcus",
        "recipient_npc_id": "marcus",
        "body": "first reply on marcus",
    })))
    assert len(captured) == aria_calls_after_first, (
        f"classifier_agent must NOT run on a marcus-thread reply when "
        f"the only directive is gated `thread: aria`; captured "
        f"{len(captured) - aria_calls_after_first} unexpected call(s)"
    )


def test_effect_thread_field_round_trips_via_load_scenario(tmp_path):
    """Author a minimal scenario YAML with `Effect.thread` set and assert
    the field survives load_scenario without being silently dropped."""
    from npc_service.models import load_scenario

    yaml_text = """\
title: "Thread-Field Round-Trip"
premise: "Minimal."
email_domain: "halcyon.com"
fallback_player_local: "x"

qualities:
  - id: q
    type: float
    min: 0.0
    max: 1.0
    default: 0.0
    description: "test"

npcs:
  - id: a
    display_name: "A"
    email_address: "A <a@halcyon.com>"
    relationship_to_owner: "x"
    base_voice_register: "x"
    base_voice_forbidden: ["x"]
    will_say: ["x"]
    will_not_say: ["x"]

threads:
  - id: a
    npc: a

effects:
  - quality: q
    op: set
    thread: a
    classify:
      question: "x"
      allowed_values: [hot, cold]
      map: { hot: 1.0, cold: 0.0 }

beats:
  - id: a.welcome
    thread: a
    npc: a
    kind: one_shot
    spec:
      pre_authored_prose: "hi"
"""
    p = tmp_path / "s.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    s = load_scenario(p)
    assert len(s.effects) == 1
    assert s.effects[0].thread == "a"
    assert s.effects[0].is_classify
