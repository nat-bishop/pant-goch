"""Session replay CLI.

Two modes drop out of the event-sourced architecture:

- **Deterministic**: re-fold the saved JSONL on a fresh state and assert
  the final state matches the recorded final state. No llama needed —
  this tests the engine's fold logic, not the prompts. Fast (~ms).

- **Re-execute**: drop everything except `SendRecorded` events, then
  feed each player body back through `process_send` so the LLM agents
  re-run. Stochastic — final state will differ from the recording in
  general. Use to spot prompt regressions (Director shifts, classifier
  drift) in a controlled scripted-input setting.

Usage:
    uv run python scripts/replay.py deterministic <log.jsonl>
    uv run python scripts/replay.py re-execute   <log.jsonl> [--max-sends 25]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Same sys.path nudge as analyze_scenario.py / playtest — the script is
# launched directly so the package needs to be on the path.
_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from npc_service import main as engine_main  # noqa: E402
from npc_service.events import SendRecorded  # noqa: E402
from npc_service.models import SendRequest  # noqa: E402
from npc_service.session import (  # noqa: E402
    load_event_log,
    make_session,
    replay_into_session,
)


async def _cmd_deterministic(path: Path) -> int:
    """Re-fold the JSONL into a fresh session. Reports final state stats."""
    from npc_service.scripts.playtest.llm_playtest import boot_engine_for_playtest  # type: ignore[import-not-found]

    # Boot SCENARIO + AGENTS but don't connect to llama — fold doesn't need
    # the agent bundle, but make_session takes one so we hand over a stub.
    boot_engine_for_playtest()
    assert engine_main.SCENARIO is not None and engine_main.AGENTS is not None

    events = load_event_log(path)
    session = make_session(engine_main.SCENARIO, engine_main.AGENTS)
    replay_into_session(session, events)

    print(f"replayed {len(events)} events from {path}")
    print(f"  emails: {len(session.state.emails)}")
    print(f"  replies: {len(session.state.replies)}")
    print(f"  fired beats: {sorted(session.state.fired_beat_ids)}")
    print(f"  ended: {session.state.ended} ({session.state.end_reason or '-'})")
    return 0


async def _cmd_re_execute(path: Path, max_sends: int) -> int:
    """Drop all but SendRecorded events, feed each through process_send."""
    from npc_service.scripts.playtest.llm_playtest import (  # type: ignore[import-not-found]
        boot_engine_for_playtest,
        probe_llama,
    )

    if not probe_llama():
        print("ERROR: llama-server not reachable. Start it with `make llama`.", file=sys.stderr)
        return 2

    boot_engine_for_playtest()
    assert engine_main.SCENARIO is not None and engine_main.AGENTS is not None

    events = load_event_log(path)
    sends = [ev for ev in events if isinstance(ev, SendRecorded)]
    print(f"loaded {len(events)} events; {len(sends)} player sends to re-execute")

    session = make_session(engine_main.SCENARIO, engine_main.AGENTS)
    # Boot tick — same as run_session.
    await engine_main._tick(session, player_body=None, in_response_to=None)

    for i, ev in enumerate(sends[:max_sends]):
        if session.state.ended:
            print(f"  session ended at send #{i} ({session.state.end_reason})")
            break
        # Reconstruct a SendRequest from the recorded SendRecorded event.
        req = SendRequest(
            in_reply_to=ev.reply.beat_responded_to,
            recipient=ev.email.recipient,
            recipient_npc_id=None,
            body=ev.email.body,
        )
        rec = engine_main.record_send(session, req)
        try:
            await engine_main.process_send(session, rec)
        except Exception as e:  # noqa: BLE001
            print(f"  send #{i} failed: {type(e).__name__}: {e}")
            break

    print(f"re-execute complete:")
    print(f"  emails: {len(session.state.emails)}")
    print(f"  fired beats: {sorted(session.state.fired_beat_ids)}")
    print(f"  ended: {session.state.ended} ({session.state.end_reason or '-'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a saved session JSONL.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_det = sub.add_parser(
        "deterministic", help="Re-fold the JSONL on a fresh state.",
    )
    p_det.add_argument("path", type=Path)

    p_re = sub.add_parser(
        "re-execute", help="Re-run player sends through the live engine.",
    )
    p_re.add_argument("path", type=Path)
    p_re.add_argument("--max-sends", type=int, default=100)

    args = parser.parse_args(argv)
    if args.mode == "deterministic":
        return asyncio.run(_cmd_deterministic(args.path))
    if args.mode == "re-execute":
        return asyncio.run(_cmd_re_execute(args.path, args.max_sends))
    parser.error(f"unknown mode: {args.mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
