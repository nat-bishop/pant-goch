"""End-to-end cycle integration test against the FastAPI surface.

llama is not booted — the per-cycle agent-call builders are monkeypatched to
produce deterministic stub outputs. The cycle orchestration + authorize +
continue endpoints + event sequencing run for real.

The test inspects the per-session out-queue + futures directly rather than
consuming SSE through TestClient.stream — TestClient's sync streaming mode
doesn't interleave with concurrent asyncio tasks reliably enough for an
event-driven test, and the runtime queue carries the same information.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from handler.models import Option


_TINY_PATH = str(Path(__file__).parent / "fixtures" / "tiny_mission.yaml")


def _completed_task() -> asyncio.Task:
    """Fills the `pacing_task` slot in the stub log-writer return value.
    The real callable overlaps LLM streaming and pacing; the stub just
    hands back an already-resolving task."""
    return asyncio.create_task(asyncio.sleep(0))


def _stub_drafter_llm(rt):
    async def _drafter_llm(session, cycle_index, catalog, forced_event_id):
        # Sort PUSH first so /authorize index 0 reliably progresses the mission.
        ordered = sorted(catalog, key=lambda c: (0 if c[0] == "PUSH" else 1, c[0]))
        options: list[Option] = []
        seen: set[tuple[str, str | None]] = set()
        for (arch, agent, params) in ordered:
            if (arch, agent) in seen:
                continue
            seen.add((arch, agent))
            satisfies = forced_event_id if (forced_event_id and not any(o.satisfies_event for o in options)) else None
            options.append(
                Option(
                    archetype=arch,
                    agent=agent,
                    params=params,
                    title=f"Authorize {arch}",
                    prose=f"Take action {arch}.",
                    satisfies_event=satisfies,
                )
            )
            if len(options) == 3:
                break
        return options, False
    return _drafter_llm


def _stub_drafter_emit(rt):
    async def _drafter_emit(cycle_index, options, engine_fallback, forced_event_id):
        rt.current_options = options
        for option_idx, option in enumerate(options):
            await rt.out_queue.put(
                {
                    "event": "option_started",
                    "cycle": cycle_index,
                    "option_idx": option_idx,
                    "archetype": option.archetype,
                    "agent": option.agent,
                    "params": option.params,
                    "title": option.title,
                    "satisfies_event": option.satisfies_event,
                }
            )
        await rt.out_queue.put(
            {
                "event": "options_complete",
                "cycle": cycle_index,
                "count": len(options),
                "has_forced_event": forced_event_id is not None,
                "event_id": forced_event_id,
                "engine_fallback": engine_fallback,
            }
        )
    return _drafter_emit


def _stub_log_writer(rt):
    async def _log_writer(
        session, cycle_index, option, delta, prior_state,
        *, leans=(),
    ):
        # Accumulate every leans payload the log writer receives across the
        # session so tests can assert that lean_on directives thread through
        # from fired events into the call. Each entry is the list of resolved
        # lean lines for one cycle's entry.
        if not hasattr(rt, "lean_history"):
            rt.lean_history = []
        rt.lean_history.append(list(leans))
        text = f"Case-file entry for cycle {cycle_index} ({option.archetype})."
        await rt.out_queue.put(
            {"event": "entry_started", "cycle": cycle_index, "archetype": option.archetype}
        )
        await rt.out_queue.put(
            {"event": "entry_word", "cycle": cycle_index, "idx": 0, "word": "", "is_last": True}
        )
        from handler.agents import LogEntry, LogShorthand
        rt.log_entries.append(
            LogEntry(cycle_index=cycle_index, archetype=option.archetype, text=text)
        )
        # Emit a synthetic shorthand event so the integration test can verify
        # the entry_shorthand path lights up without hitting the real LLM.
        shorthand = f"C{cycle_index} — {option.archetype} — stub consequence"
        rt.log_shorthands.append(
            LogShorthand(cycle_index=cycle_index, shorthand=shorthand)
        )
        await rt.out_queue.put(
            {"event": "entry_shorthand", "cycle": cycle_index, "shorthand": shorthand}
        )
        return text, _completed_task()
    return _log_writer


def _stub_log_text_ready_hook(rt):
    """Disable cross-cycle prefetch in this integration test — prefetch would
    try to hit the drafter agent against the (non-booted) llama-server. The
    cycle orchestration is what we're testing here; prefetch correctness is
    verified separately."""
    async def _hook(_cycle_index: int) -> None:
        return
    return _hook


@pytest.fixture
def stubbed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setenv("MISSION_PATH", _TINY_PATH)
    from handler import main as _main
    monkeypatch.setattr(_main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(_main, "_build_drafter_llm_callable", _stub_drafter_llm)
    monkeypatch.setattr(_main, "_build_drafter_emit_callable", _stub_drafter_emit)
    monkeypatch.setattr(_main, "_build_log_writer_callable", _stub_log_writer)
    monkeypatch.setattr(_main, "_build_log_text_ready_hook", _stub_log_text_ready_hook)
    monkeypatch.setattr(_main, "WORD_PACE_S", 0.0)
    with TestClient(_main.app) as client:
        yield client


def _drain_until(rt, predicate, *, max_iters: int = 200) -> list[dict]:
    """Pull events off the runtime's out_queue (synchronously via the test
    thread's loop) until `predicate(event)` is true. Returns the list of
    events drained. The TestClient's portal pumps the underlying loop on
    each `queue.get()`."""
    out: list[dict] = []
    for _ in range(max_iters):
        ev = rt.out_queue.get_nowait() if not rt.out_queue.empty() else None
        if ev is None:
            return out
        out.append(ev)
        if predicate(ev):
            return out
    raise RuntimeError("drained max_iters without satisfying predicate")


def _pump(client: TestClient) -> None:
    """Run a no-op request against the client so the underlying portal
    pumps the asyncio loop a few times. This lets background tasks (the
    cycle loop) progress without us actually consuming the SSE stream."""
    r = client.get("/health")
    assert r.status_code == 200


def _await_event(client: TestClient, rt, kind: str, *, max_pumps: int = 200) -> dict:
    """Pump the loop until an event with `event == kind` appears on the
    out_queue. Returns the event."""
    for _ in range(max_pumps):
        _pump(client)
        while not rt.out_queue.empty():
            ev = rt.out_queue.get_nowait()
            if ev.get("event") == kind:
                return ev
    raise AssertionError(f"event {kind!r} never produced after {max_pumps} pumps")


def _await_future(client: TestClient, getter, *, max_pumps: int = 200) -> None:
    """Pump the loop until `getter()` returns a future that exists and is
    not yet done (i.e. the cycle is awaiting that gate)."""
    for _ in range(max_pumps):
        _pump(client)
        fut = getter()
        if fut is not None and not fut.done():
            return
    raise AssertionError("future never armed")


def test_full_cycle_to_terminal(stubbed_client: TestClient) -> None:
    """Drive the cycle loop through to a terminal outcome."""
    from handler import main as _main

    r = stubbed_client.post("/mission/start")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    rt = _main.SESSIONS[sid]

    all_events: list[dict] = []
    resolved: dict | None = None

    for _ in range(20):  # up to 20 alternations; tiny mission resolves in ≤4
        _await_future(stubbed_client, lambda: rt.auth_future)
        # Drain everything currently queued.
        while not rt.out_queue.empty():
            all_events.append(rt.out_queue.get_nowait())
        # Authorize index 0 (PUSH first by stub ordering).
        ar = stubbed_client.post(f"/mission/{sid}/authorize", json={"option_index": 0})
        assert ar.status_code == 200, ar.text

        # Now wait for either awaiting_continue (cycle ended, mission active)
        # or mission_resolved (terminal). Pump until one of those events lands.
        for _ in range(200):
            _pump(stubbed_client)
            while not rt.out_queue.empty():
                ev = rt.out_queue.get_nowait()
                all_events.append(ev)
                if ev.get("event") == "mission_resolved":
                    resolved = ev
            if resolved is not None:
                break
            if rt.continue_future is not None and not rt.continue_future.done():
                break
        if resolved is not None:
            break
        # POST /continue to advance to next cycle.
        cr = stubbed_client.post(f"/mission/{sid}/continue")
        assert cr.status_code == 200, cr.text

    assert resolved is not None, f"mission never resolved; kinds={[e.get('event') for e in all_events]}"

    # Server stats round-trip through SSE.
    assert resolved["outcome"] in {"win", "loss_timeout", "loss_agent_lost"}
    assert resolved["cycles_max"] == 4
    assert resolved["terminal_event"] in {"win", "timeout", "agent_lost"}
    assert isinstance(resolved["agents_alive"], list)
    assert isinstance(resolved["agents_lost"], list)

    # Critical sequence assertions per cycle 1.
    kinds = [e.get("event") for e in all_events]
    assert "cycle_opened" in kinds
    assert "options_complete" in kinds
    assert "effects_applied" in kinds
    assert "entry_started" in kinds
    assert "entry_shorthand" in kinds
    assert "mission_resolved" in kinds
    # cycle_opened precedes effects_applied for the first cycle.
    assert kinds.index("cycle_opened") < kinds.index("effects_applied")

    # /state reflects terminal outcome.
    s = stubbed_client.get(f"/mission/{sid}/state").json()
    assert s["outcome"] is not None


def test_forced_event_lean_threads_to_log_writer(stubbed_client: TestClient) -> None:
    """The tiny mission's `turning_point` forced event carries
    `lean_on: ["test lean directive for ALPHA"]`. When the drafter stub
    surfaces that event and the player authorizes the satisfying option,
    the log writer must receive that lean line."""
    from handler import main as _main

    r = stubbed_client.post("/mission/start")
    sid = r.json()["session_id"]
    rt = _main.SESSIONS[sid]

    resolved = None
    for _ in range(20):
        _await_future(stubbed_client, lambda: rt.auth_future)
        while not rt.out_queue.empty():
            rt.out_queue.get_nowait()
        # Authorize index 0 — PUSH on cycle 1; on cycle 2 turning_point is due
        # and the drafter stub assigns satisfies_event to the first option.
        stubbed_client.post(f"/mission/{sid}/authorize", json={"option_index": 0})
        for _ in range(200):
            _pump(stubbed_client)
            while not rt.out_queue.empty():
                ev = rt.out_queue.get_nowait()
                if ev.get("event") == "mission_resolved":
                    resolved = ev
            if resolved is not None:
                break
            if rt.continue_future is not None and not rt.continue_future.done():
                break
        if resolved is not None:
            break
        stubbed_client.post(f"/mission/{sid}/continue")

    assert resolved is not None
    # The turning_point lean must have surfaced in at least one log-writer call.
    all_leans = [line for batch in getattr(rt, "lean_history", []) for line in batch]
    assert "test lean directive for ALPHA" in all_leans, (
        f"expected the tiny mission's forced-event lean to thread through "
        f"to the log writer; got lean_history={getattr(rt, 'lean_history', [])}"
    )


def test_authorize_rejects_bad_index(stubbed_client: TestClient) -> None:
    """When awaiting_auth is pending, an out-of-range option_index must 400."""
    from handler import main as _main

    r = stubbed_client.post("/mission/start")
    sid = r.json()["session_id"]
    rt = _main.SESSIONS[sid]

    _await_future(stubbed_client, lambda: rt.auth_future)

    ar = stubbed_client.post(f"/mission/{sid}/authorize", json={"option_index": 99})
    assert ar.status_code == 400
    # Recovery: a valid index should still succeed.
    ar2 = stubbed_client.post(f"/mission/{sid}/authorize", json={"option_index": 0})
    assert ar2.status_code == 200


def test_continue_409_without_pending(stubbed_client: TestClient) -> None:
    """POST /continue right after /start must 409 — cycle 1 hasn't finished."""
    r = stubbed_client.post("/mission/start")
    sid = r.json()["session_id"]
    cr = stubbed_client.post(f"/mission/{sid}/continue")
    assert cr.status_code == 409


def test_authorize_409_without_pending(stubbed_client: TestClient) -> None:
    """POST /authorize after a successful authorize (before next cycle's
    awaiting_auth arms) must 409."""
    from handler import main as _main

    r = stubbed_client.post("/mission/start")
    sid = r.json()["session_id"]
    rt = _main.SESSIONS[sid]
    _await_future(stubbed_client, lambda: rt.auth_future)
    stubbed_client.post(f"/mission/{sid}/authorize", json={"option_index": 0})
    # Immediately try again — the cycle is now in effects/dispatch/close phase,
    # awaiting continue (or done).
    ar2 = stubbed_client.post(f"/mission/{sid}/authorize", json={"option_index": 0})
    assert ar2.status_code == 409
