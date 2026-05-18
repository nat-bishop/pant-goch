"""Server-Sent Events: paced word emitter + v2 event surface.

Two responsibilities:

1. **Paced word emitter.** The 80 ms pace floor is non-negotiable —
   it's the source of truth for streaming cadence, NOT the frontend's
   job to re-pace. Two call sites: the scene runner's phase-2 prose
   stream and the chronicler's entry stream.

2. **Event helpers.** Small builder functions for the v2 SSE event
   surface (SPEC §5 table). Plain dicts on the wire so the frontend's
   `readSseStream` can JSON.parse and consume directly.

Wire format: each SSE message is one JSON object with a top-level
`event` key naming the event kind. Other keys are event-specific. Sent
via `Server-Sent Events` framing (`data: <json>\\n\\n`). Heartbeats are
ignored by the client (`readSseStream` lifted from v1 already filters
empty lines).
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any


# ───────────────────────── paced word-stream emitter ─────────────────────────

WORD_PACE_S = 0.08  # min seconds between paced word emissions

_WORD_RE = re.compile(r"\S+\s*")


async def _words_from_text(text: str) -> AsyncIterator[str]:
    """Split a complete text into words, preserving trailing whitespace per word."""
    for match in _WORD_RE.finditer(text):
        yield match.group(0)


async def _words_from_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Buffer streaming text chunks; yield each complete word as it crosses a
    whitespace boundary. The final word (no trailing whitespace) is yielded on
    stream end. Each yielded word includes its trailing whitespace, if any."""
    buf = ""
    async for chunk in chunks:
        buf += chunk
        # Repeatedly extract leading complete words.
        while True:
            match = _WORD_RE.match(buf)
            if not match:
                break
            # Word is "complete" iff there's whitespace after it OR we know
            # more content is coming. Trailing-whitespace match means complete.
            word = match.group(0)
            # If the buffer is exactly the match and the match has no trailing
            # whitespace, the word might still be growing. Hold it.
            if match.end() == len(buf) and not re.search(r"\s$", word):
                break
            yield word
            buf = buf[match.end():]
    if buf.strip():
        yield buf


async def paced_word_emitter(
    word_source: AsyncIterator[str],
    emit: Callable[[dict], Awaitable[None]],
    *,
    event_kind: str,
    min_interval: float = WORD_PACE_S,
    extra: dict | None = None,
) -> str:
    """Emit each word from `word_source` as an SSE event, throttled to at
    least `min_interval` seconds between emissions. Returns the accumulated
    full text.

    Two call sites in v2:
    1. Phase-2 prose: `word_source = _words_from_stream(llm_stream)` —
       words emerge as the LLM produces them; the emitter throttles when
       the LLM outpaces the cadence and passes through when the LLM is
       slower.
    2. Log entry: same shape (`_words_from_stream(log_stream)`).
    """
    next_slot = time.monotonic()
    idx = 0
    full = ""
    async for word in word_source:
        now = time.monotonic()
        if now < next_slot:
            await asyncio.sleep(next_slot - now)
        payload: dict[str, Any] = {"event": event_kind, "idx": idx, "word": word}
        if extra:
            payload.update(extra)
        await emit(payload)
        full += word
        idx += 1
        next_slot = time.monotonic() + min_interval
    sentinel: dict[str, Any] = {
        "event": event_kind, "idx": idx, "word": "", "is_last": True,
    }
    if extra:
        sentinel.update(extra)
    await emit(sentinel)
    return full


# ───────────────────────── v2 event builders ─────────────────────────
#
# Plain dicts on the wire. Builder functions keep the schema honest in one
# place — if a new event is added, it lands here and the frontend sees it
# named consistently.


def scene_start_event(scene_id: int, location_id: str, location_name: str) -> dict:
    return {
        "event": "scene_start",
        "scene_id": scene_id,
        "location_id": location_id,
        "location_name": location_name,
    }


def state_change_event(kind: str, payload: dict) -> dict:
    """e.g. kind='unlock_beat', payload={'beat_id': 'tom_named_a_vehicle'}."""
    return {"event": "state_change", "kind": kind, "payload": payload}


def options_event(options: list[dict]) -> dict:
    """`options` is a list of `{text}` dicts. The map is the navigation
    surface; options no longer carry destinations."""
    return {"event": "options", "options": options}


def free_text_request_event(req) -> dict:
    """Emitted when the choice author asks for typed input from the
    player (e.g. interrogation). Frontend / CLI opens an input box with
    `prompt` as the placeholder; `npc_id` (when set) names the NPC the
    typed line is addressed to. Resolved via POST /free_text on the SSE
    backend or stdin on the CLI.

    `req` is a `schemas.FreeText`; typed loosely to avoid a circular
    import.
    """
    return {
        "event": "free_text_request",
        "prompt": req.prompt,
        "npc_id": req.npc_id,
    }


def scene_exit_event() -> dict:
    return {"event": "scene_exit"}


def location_change_event(from_id: str, to_id: str) -> dict:
    return {"event": "location_change", "from": from_id, "to": to_id}


def npc_moved_event(npc_id: str, from_id: str, to_id: str) -> dict:
    """Emitted by `scene_loop.play_scene` after the populator places an NPC
    somewhere other than their previous location. Frontend may surface
    this as a quiet marker on the map; for v1 it's informational only.
    Sent once per moved NPC at the start of the scene (before any prose).
    """
    return {
        "event": "npc_moved",
        "npc_id": npc_id,
        "from": from_id,
        "to": to_id,
    }


def game_over_event(ending_id: str) -> dict:
    return {"event": "game_over", "ending_id": ending_id}


def error_event(message: str) -> dict:
    """For caller-side errors that the frontend should surface (bad walk
    target, invalid pick index, etc.). Stays out of the prose stream."""
    return {"event": "error", "message": message}
