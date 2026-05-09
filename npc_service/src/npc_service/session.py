"""Per-game `Session` — replaces the module globals (`STATE`, `ACTION_LOCK`)
that previously lived in `main.py`.

A `Session` bundles:
- the per-game state (`InboxState`)
- the canonical event log (the source of truth — state is derived)
- a per-session asyncio.Lock (so `/send` requests on different sessions
  don't block each other)
- references (NOT copies) to `Scenario` and `AgentBundle`, which stay
  process singletons
- an optional `persist` callback fired on every committed event, used to
  autosave the log to a JSONL file when `Settings.session_save_path` is set

`apply(event)` is the only state-write entry point in the engine.
`state_at(index)` re-folds the first N events on a fresh state for the
read-only debug-overlay timeline.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import TypeAdapter

from .agents import AgentBundle
from .events import Event, fold
from .models import InboxState, Scenario


# Module-level adapter so we don't re-build it per session.
_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


@dataclass
class Session:
    scenario: Scenario
    agents: AgentBundle
    state: InboxState
    event_log: list[Event] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Optional sink for committed events. Set to a file-appender when
    # session autosave is enabled. Called synchronously after fold + append;
    # exceptions propagate (callers should make the sink no-throw).
    persist: Callable[[Event], None] | None = None

    def apply(self, event: Event) -> None:
        fold(self.state, event)
        self.event_log.append(event)
        if self.persist is not None:
            self.persist(event)

    def state_at(self, index: int) -> InboxState:
        if index < 0 or index > len(self.event_log):
            raise IndexError(
                f"event index {index} out of range [0, {len(self.event_log)}]"
            )
        snapshot = InboxState.from_scenario(self.scenario)
        for ev in self.event_log[:index]:
            fold(snapshot, ev)
        return snapshot


def make_session(
    scenario: Scenario,
    agents: AgentBundle,
    *,
    persist: Callable[[Event], None] | None = None,
) -> Session:
    """Build a fresh session with empty state. Caller is responsible for
    running the boot tick (which emits welcome-beat events through
    `session.apply`)."""
    return Session(
        scenario=scenario,
        agents=agents,
        state=InboxState.from_scenario(scenario),
        persist=persist,
    )


# ───────────────────────── JSONL autosave + replay ─────────────────────────


def make_jsonl_persistor(path: Path) -> Callable[[Event], None]:
    """Return a `persist` callback that appends each event to `path` as a
    single JSON line. The file is opened-and-closed per write so partial
    writes after a crash are bounded to one event. Caller ensures the
    parent directory exists before passing `path`.
    """
    def _persist(event: Event) -> None:
        line = _EVENT_ADAPTER.dump_json(event).decode("utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
    return _persist


def load_event_log(path: Path) -> list[Event]:
    """Read a JSONL file written by `make_jsonl_persistor`. Returns the
    event list (validated through the discriminated union — bad lines
    raise; the file is assumed write-trusted by us).
    """
    events: list[Event] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            events.append(_EVENT_ADAPTER.validate_python(json.loads(stripped)))
    return events


def replay_into_session(session: Session, events: list[Event]) -> None:
    """Fold a saved event list into a fresh session. Bypasses the persist
    hook (we don't want to re-write what we just read). Use this on
    lifespan boot when `session_save_path` already exists.
    """
    saved_persist = session.persist
    session.persist = None
    try:
        for ev in events:
            session.apply(ev)
    finally:
        session.persist = saved_persist
