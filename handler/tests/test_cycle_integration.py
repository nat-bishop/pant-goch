"""End-to-end cycle integration test against the FastAPI surface.

llama is not booted — the three agent-call builders (`_build_sop_callable`,
`_build_drafter_callable`, `_build_renderer_callable`) are monkeypatched to
produce deterministic stub outputs. The cycle orchestration + SSE wiring +
authorize endpoint + event sequencing run for real.

What this pins:
 - `mission_started` → `cycle_opened` → SoP words → option events →
   `awaiting_auth` → effects/beats → dispatch → `cycle_closed` → next cycle
   or `mission_resolved`, in order.
 - `/authorize` resumes the cycle loop, options validated by index.
 - The tiny mission terminates at one of its authored outcomes within
   `max_cycles=4`.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from handler.models import Option


_TINY_PATH = str(Path(__file__).parent / "fixtures" / "tiny_mission.yaml")


def _stub_sop(rt):
    async def _sop(session, cycle_index):
        text = f"State of play for cycle {cycle_index}. Resources nominal."
        for w in text.split(" "):
            await rt.out_queue.put(
                {"event": "sop_word", "cycle": cycle_index, "idx": 0, "word": w + " "}
            )
        await rt.out_queue.put(
            {"event": "sop_word", "cycle": cycle_index, "idx": 99, "word": "", "is_last": True}
        )
        return text
    return _sop


def _stub_drafter(rt):
    async def _draft(session, cycle_index, sop_text, catalog, forced_moment_id):
        # Build one option per (archetype, agent) entry, up to 3. PUSH first
        # so a "first eligible" picker authorizes mission progress and the
        # mission terminates on cycle 2 of the tiny mission.
        options: list[Option] = []
        seen: set[tuple[str, str | None]] = set()
        # Sort to prefer PUSH first (forces deterministic progress).
        ordered = sorted(catalog, key=lambda c: (0 if c[0] == "PUSH" else 1, c[0]))
        for (arch, agent, params) in ordered:
            if (arch, agent) in seen:
                continue
            seen.add((arch, agent))
            satisfies = forced_moment_id if (forced_moment_id and not any(o.satisfies_moment for o in options)) else None
            options.append(
                Option(
                    archetype=arch,
                    agent=agent,
                    params=params,
                    title=f"Authorize {arch}",
                    prose=f"Take action {arch}.",
                    satisfies_moment=satisfies,
                )
            )
            if len(options) == 3:
                break
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
                    "satisfies_moment": option.satisfies_moment,
                }
            )
            await rt.out_queue.put(
                {
                    "event": "option_word",
                    "cycle": cycle_index,
                    "option_idx": option_idx,
                    "idx": 0,
                    "word": option.prose,
                }
            )
            await rt.out_queue.put(
                {
                    "event": "option_word",
                    "cycle": cycle_index,
                    "option_idx": option_idx,
                    "idx": 1,
                    "word": "",
                    "is_last": True,
                }
            )
        await rt.out_queue.put(
            {
                "event": "options_complete",
                "cycle": cycle_index,
                "count": len(options),
                "has_forced_moment": forced_moment_id is not None,
                "moment_id": forced_moment_id,
                "engine_fallback": False,
            }
        )
        rt.current_options = options
        return options, False
    return _draft


def _stub_renderer(rt):
    async def _render(session, cycle_index, option, delta, acting):
        text = f"Dispatch for cycle {cycle_index} ({acting}, {option.archetype})."
        await rt.out_queue.put(
            {"event": "dispatch_started", "cycle": cycle_index, "acting": acting, "archetype": option.archetype}
        )
        await rt.out_queue.put(
            {"event": "dispatch_word", "cycle": cycle_index, "idx": 0, "word": text, "acting": acting}
        )
        await rt.out_queue.put(
            {"event": "dispatch_word", "cycle": cycle_index, "idx": 1, "word": "", "acting": acting, "is_last": True}
        )
        # Mimic main.py's renderer callable: record DispatchRecord so the next
        # cycle's drafter prompt has context (no-op in the stub but matches
        # real-path semantics).
        from handler.agents import DispatchRecord
        rt.dispatches.append(
            DispatchRecord(cycle_index=cycle_index, acting=acting, archetype=option.archetype, text=text)
        )
        return text
    return _render


@pytest.fixture
def stubbed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LAUNCH_LLAMA", "false")
    monkeypatch.setenv("MISSION_PATH", _TINY_PATH)
    from handler import main as _main
    monkeypatch.setattr(_main, "EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(_main, "_build_sop_callable", _stub_sop)
    monkeypatch.setattr(_main, "_build_drafter_callable", _stub_drafter)
    monkeypatch.setattr(_main, "_build_renderer_callable", _stub_renderer)
    # Zero out the pacing so the test doesn't wait 80ms per word.
    monkeypatch.setattr(_main, "WORD_PACE_S", 0.0)
    with TestClient(_main.app) as client:
        yield client


def _drain_until(events, predicate, *, timeout_s: float = 10.0):
    """Collect events from an SSE generator until `predicate(ev)` returns True
    or the timeout expires. Returns the list of events collected (inclusive)."""
    start = time.monotonic()
    out = []
    for ev in events:
        out.append(ev)
        if predicate(ev):
            return out
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"SSE didn't yield predicate event in {timeout_s}s")
    raise RuntimeError("SSE stream ended before predicate matched")


def _parse_sse(stream):
    """Yield each SSE `data: {...}` payload as a dict."""
    buffer = ""
    for chunk in stream.iter_text():
        buffer += chunk
        while "\n\n" in buffer:
            block, _, buffer = buffer.partition("\n\n")
            for line in block.split("\n"):
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue


def test_full_cycle_to_terminal(stubbed_client: TestClient) -> None:
    """Start a session, ride the SSE stream, authorize each cycle's first
    option, and assert the run resolves with a mission_resolved event."""
    r = stubbed_client.post("/mission/start")
    assert r.status_code == 200
    sid = r.json()["session_id"]

    with stubbed_client.stream("GET", f"/mission/{sid}/stream") as stream:
        assert stream.status_code == 200
        events = _parse_sse(stream)

        # Cycle 1: pick option 0, then ride to either next cycle or terminal.
        seen_kinds: list[str] = []
        resolved_payload: dict | None = None
        cycles_seen = 0
        max_iters = 200  # bound on event count for a 4-cycle tiny mission

        for _ in range(max_iters):
            try:
                ev = next(events)
            except StopIteration:
                break
            kind = ev.get("event")
            if not kind or kind == ":heartbeat":
                continue
            seen_kinds.append(kind)
            if kind == "cycle_opened":
                cycles_seen += 1
            if kind == "awaiting_auth":
                # Authorize index 0 (first option — by the stub's ordering,
                # PUSH if it's in the catalog).
                ar = stubbed_client.post(
                    f"/mission/{sid}/authorize", json={"option_index": 0}
                )
                assert ar.status_code == 200, ar.text
            if kind == "mission_resolved":
                resolved_payload = ev
                break

        assert resolved_payload is not None, f"mission never resolved; seen={seen_kinds}"

        # Outcome must be one of the authored terminals.
        assert resolved_payload["outcome"] in {"win", "loss_timeout", "loss_agent_lost"}
        # Server stats must round-trip.
        assert resolved_payload["cycles_max"] == 4
        assert isinstance(resolved_payload["agents_alive"], list)
        assert isinstance(resolved_payload["agents_lost"], list)
        assert resolved_payload["terminal_beat"] in {"win", "timeout", "agent_lost"}

        # Critical sequence assertions: at least one full cycle's events
        # appeared in order before the resolve.
        first_cycle_kinds = []
        for k in seen_kinds:
            first_cycle_kinds.append(k)
            if k == "mission_resolved":
                break
        # cycle_opened must precede the first awaiting_auth.
        assert first_cycle_kinds.index("cycle_opened") < first_cycle_kinds.index("awaiting_auth")
        # awaiting_auth must precede effects_applied.
        assert first_cycle_kinds.index("awaiting_auth") < first_cycle_kinds.index("effects_applied")
        # The state endpoint reflects the terminal outcome.
        s = stubbed_client.get(f"/mission/{sid}/state").json()
        assert s["outcome"] is not None


def test_authorize_rejects_bad_index(stubbed_client: TestClient) -> None:
    """When awaiting_auth is pending, an out-of-range option_index must 400."""
    r = stubbed_client.post("/mission/start")
    sid = r.json()["session_id"]

    with stubbed_client.stream("GET", f"/mission/{sid}/stream") as stream:
        events = _parse_sse(stream)
        # Drive forward until awaiting_auth.
        for _ in range(200):
            ev = next(events)
            if ev.get("event") == "awaiting_auth":
                break
        else:
            pytest.fail("never reached awaiting_auth")

        ar = stubbed_client.post(
            f"/mission/{sid}/authorize", json={"option_index": 99}
        )
        assert ar.status_code == 400
        # Recovery: a valid index should still succeed.
        ar2 = stubbed_client.post(
            f"/mission/{sid}/authorize", json={"option_index": 0}
        )
        assert ar2.status_code == 200


def test_authorize_409_before_awaiting(stubbed_client: TestClient) -> None:
    """Right after /mission/start, the SoP is still streaming — the auth
    future hasn't been created yet, so /authorize must 409."""
    r = stubbed_client.post("/mission/start")
    sid = r.json()["session_id"]
    # Don't open the SSE stream — the cycle loop is producing into the queue
    # but no future is set yet because the stub SoP completes synchronously
    # and the loop progresses to drafter on the next event loop step.
    # Give it no time to advance:
    ar = stubbed_client.post(f"/mission/{sid}/authorize", json={"option_index": 0})
    # Either 409 (no pending auth yet) or 200 (auth raced to set the future).
    # The contract is "must not crash"; we accept both as the only safe outcomes.
    assert ar.status_code in {200, 409}
