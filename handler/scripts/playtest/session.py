"""Session record + HTTP driver for the option-picking playtest.

`SessionRecord` captures everything one playtest run produced for evaluators
+ report serialization. `run_session` drives one game end-to-end:
  - POST /mission/start
  - GET /mission/{id}/stream (SSE)
  - per cycle: collect SoP + options, call the picker, POST /authorize
  - on terminal: record outcome
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .option_picker import PickedOption, pick
from .personas import PersonaSpec


# ───────────────────────── per-cycle trace ─────────────────────────


@dataclass
class CycleTrace:
    """What happened in one cycle. Mirrors the SSE event stream — collected
    by `run_session` as events flow in."""

    cycle_index: int
    sop_text: str = ""
    options: list[dict[str, Any]] = field(default_factory=list)
    chosen_index: int | None = None
    chosen_archetype: str | None = None
    effects_delta: dict[str, Any] = field(default_factory=dict)
    beats_fired: list[str] = field(default_factory=list)
    dispatch_text: str = ""
    dispatch_acting: str | None = None
    used_engine_fallback: bool = False
    forced_moment_id: str | None = None


# ───────────────────────── session record ─────────────────────────


@dataclass
class SessionRecord:
    persona_name: str
    persona_archetype: str
    seed: int
    cycles: list[CycleTrace] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    outcome: str | None = None  # outcome_key on terminal, None on error/timeout
    outcome_summary: str = ""
    error: str | None = None

    def reached_terminal(self) -> bool:
        return self.outcome is not None

    def cycles_used(self) -> int:
        return len(self.cycles)

    def fired_beats(self) -> list[str]:
        out: list[str] = []
        for c in self.cycles:
            out.extend(c.beats_fired)
        return out

    def engine_fallback_count(self) -> int:
        return sum(1 for c in self.cycles if c.used_engine_fallback)

    def to_json(self) -> dict[str, Any]:
        return {
            "persona_name": self.persona_name,
            "persona_archetype": self.persona_archetype,
            "seed": self.seed,
            "cycles": [asdict(c) for c in self.cycles],
            "final_state": self.final_state,
            "outcome": self.outcome,
            "outcome_summary": self.outcome_summary,
            "error": self.error,
        }


# ───────────────────────── HTTP / SSE driver ─────────────────────────


async def _sse_events(response: httpx.Response):
    """Async iterator over SSE `data: {...}` JSON payloads."""
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        try:
            yield json.loads(line[6:])
        except json.JSONDecodeError:
            continue


async def run_session(
    persona: PersonaSpec,
    *,
    seed: int,
    engine_base_url: str,
    timeout_s: float = 600.0,
) -> SessionRecord:
    """Run one playtest session end-to-end. Returns a populated `SessionRecord`.

    The engine sidecar must already be running at `engine_base_url`. Each
    SSE event mutates the in-progress `CycleTrace`; when options finish
    streaming and `awaiting_auth` arrives, the picker fires.
    """
    record = SessionRecord(
        persona_name=persona.name,
        persona_archetype=persona.archetype,
        seed=seed,
    )
    try:
        async with httpx.AsyncClient(base_url=engine_base_url, timeout=timeout_s) as client:
            r = await client.post("/mission/start")
            r.raise_for_status()
            session_id = r.json()["session_id"]

            current: CycleTrace | None = None
            current_options_payload: list[dict[str, Any]] = []
            awaiting_auth = False

            async with client.stream(
                "GET", f"/mission/{session_id}/stream"
            ) as stream_resp:
                stream_resp.raise_for_status()
                async for ev in _sse_events(stream_resp):
                    kind = ev.get("event")
                    if kind == "mission_started":
                        record.final_state = dict(ev.get("state", {}))
                    elif kind == "cycle_opened":
                        current = CycleTrace(cycle_index=ev["cycle"])
                        record.cycles.append(current)
                        current_options_payload = []
                        awaiting_auth = False
                    elif kind == "sop_word" and current is not None:
                        if not ev.get("is_last"):
                            current.sop_text += ev["word"]
                    elif kind == "option_started" and current is not None:
                        current_options_payload.append(
                            {
                                "option_idx": ev["option_idx"],
                                "archetype": ev["archetype"],
                                "agent": ev.get("agent"),
                                "params": ev.get("params", {}),
                                "title": ev["title"],
                                "satisfies_moment": ev.get("satisfies_moment"),
                                "prose": "",
                            }
                        )
                    elif kind == "option_word" and current is not None:
                        idx = ev["option_idx"]
                        if 0 <= idx < len(current_options_payload) and not ev.get("is_last"):
                            current_options_payload[idx]["prose"] += ev["word"]
                    elif kind == "options_complete" and current is not None:
                        current.options = list(current_options_payload)
                        if ev.get("moment_id"):
                            current.forced_moment_id = ev["moment_id"]
                        if ev.get("engine_fallback"):
                            current.used_engine_fallback = True
                    elif kind == "awaiting_auth" and current is not None:
                        # Pick + post authorize.
                        picks = [
                            PickedOption(
                                archetype=o["archetype"],
                                agent=o.get("agent"),
                                satisfies_moment=o.get("satisfies_moment"),
                            )
                            for o in current.options
                        ]
                        chosen_idx = pick(persona, picks)
                        current.chosen_index = chosen_idx
                        current.chosen_archetype = current.options[chosen_idx]["archetype"]
                        await client.post(
                            f"/mission/{session_id}/authorize",
                            json={"option_index": chosen_idx},
                        )
                        awaiting_auth = False
                    elif kind == "effects_applied" and current is not None:
                        current.effects_delta = ev.get("delta", {})
                    elif kind == "beats_fired" and current is not None:
                        current.beats_fired.extend(ev.get("beats", []))
                    elif kind == "dispatch_started" and current is not None:
                        current.dispatch_acting = ev.get("acting")
                    elif kind == "dispatch_word" and current is not None:
                        if not ev.get("is_last"):
                            current.dispatch_text += ev["word"]
                    elif kind == "cycle_closed":
                        record.final_state = dict(ev.get("state", {}))
                    elif kind == "mission_resolved":
                        record.outcome = ev["outcome"]
                        record.outcome_summary = ev.get("summary", "")
                    elif kind == "session_closed":
                        break
                    elif kind == "error":
                        record.error = ev.get("code", "unknown")
                        break
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        record.error = f"http: {e}"
    except Exception as e:
        record.error = f"exception: {type(e).__name__}: {e}"
    return record
