"""Server-Sent Events: paced word emitter + v2 event surface.

Two responsibilities:

1. **Paced word emitter.** The 150 ms pace floor is non-negotiable —
   it's the source of truth for streaming cadence, NOT the frontend's
   job to re-pace. The floor lengthens after punctuation so the reveal
   breathes like read speech — clause < sentence < paragraph — see
   `_pause_after` below. Two call sites: the scene runner's phase-2
   prose stream and the chronicler's entry stream. An optional
   `skip_event` lets the session flush the remainder at full speed
   in response to a player click — see `paced_word_emitter` below.

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

WORD_PACE_S = 0.15  # base min seconds between paced word emissions

# Punctuation lengthens the hold *after* a word, as multiples of the base
# interval, so WORD_PACE_S stays the single authoritative knob (tests that
# monkeypatch it down scale the whole curve). Grounded in read-speech prosody
# (comma ≈ 0.47 s, sentence boundary ≈ 0.6 s, between-sentence 0.6–1.2 s),
# scaled down to suit a reveal brisker than spoken delivery: clause ≈ 0.32 s,
# sentence ≈ 0.50 s, paragraph ≈ 0.70 s at the default base.
PACE_CLAUSE_X = 2.1     # , ; : — –        clause break
PACE_SENTENCE_X = 3.3   # . ! ? …          sentence end
PACE_PARAGRAPH_X = 4.7  # blank line       paragraph break

# Closing marks that can trail the operative punctuation (e.g. '"Stop." ').
_CLOSERS = "\"'”’)]}»"

_WORD_RE = re.compile(r"\S+\s*")


def _pause_after(word: str, base: float) -> float:
    """Seconds to wait AFTER emitting `word`, derived from its trailing
    punctuation. `word` includes its trailing whitespace (per `_WORD_RE`),
    so a blank line is detectable as a paragraph break.

    Abbreviations ("Mr.", "St.") trip the sentence tier — the standard TTS
    edge case; rare here and harmless (one slightly longer hold)."""
    trailing_ws = word[len(word.rstrip()):]
    if trailing_ws.count("\n") >= 2:  # blank line = paragraph break
        return base * PACE_PARAGRAPH_X
    core = word.rstrip().rstrip(_CLOSERS)
    if not core:
        return base
    last = core[-1]
    if last in ".!?…":
        return base * PACE_SENTENCE_X
    if last in ",;:—–":
        return base * PACE_CLAUSE_X
    return base


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
    skip_event: asyncio.Event | None = None,
) -> str:
    """Emit each word from `word_source` as an SSE event, throttled to at
    least `min_interval` seconds between emissions. The hold after a word
    lengthens with its trailing punctuation (`_pause_after`) so the reveal
    breathes at clause / sentence / paragraph boundaries. Returns the
    accumulated full text.

    Two call sites in v2:
    1. Phase-2 prose: `word_source = _words_from_stream(llm_stream)` —
       words emerge as the LLM produces them; the emitter throttles when
       the LLM outpaces the cadence and passes through when the LLM is
       slower.
    2. Log entry: same shape (`_words_from_stream(log_stream)`).

    `skip_event`: when set (e.g. by POST /skip in response to a player
    click), the remaining words flush without pacing until the source is
    exhausted. Each emitter call clears the event on entry so a stray
    click landing between two paced streams doesn't leak into the next
    one — and so a mid-stream exception still leaves a clean slate.
    """
    if skip_event is not None:
        skip_event.clear()
    next_slot = time.monotonic()
    idx = 0
    full = ""
    async for word in word_source:
        if skip_event is None or not skip_event.is_set():
            now = time.monotonic()
            if now < next_slot:
                await asyncio.sleep(next_slot - now)
        payload: dict[str, Any] = {"event": event_kind, "idx": idx, "word": word}
        if extra:
            payload.update(extra)
        await emit(payload)
        full += word
        idx += 1
        next_slot = time.monotonic() + _pause_after(word, min_interval)
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
    """Emitted when the engine surfaces a free-text box. Frontend / CLI
    opens an input box captioned with `prompt`; `suggested` (when set) is a
    pre-loaded, editable backup move; `npc_id` (when set) names the NPC the
    typed line is addressed to. Resolved via POST /free_text on the SSE
    backend or stdin on the CLI.

    `req` is a `schemas.FreeText`; typed loosely to avoid a circular
    import.
    """
    return {
        "event": "free_text_request",
        "prompt": req.prompt,
        "suggested": req.suggested,
        "npc_id": req.npc_id,
    }


def scene_exit_event() -> dict:
    """Emitted when a scene has fully ended — the player walked and the
    closing prose has streamed. Marks the scene closed and (with the walk
    consumed) disarms the map until the next input-await. Followed by
    `location_change` once the chronicler + case file settle."""
    return {"event": "scene_exit"}


def scene_spent_event() -> dict:
    """Emitted on a **soft-close**: the choice author's Leave with no
    ending satisfied. The scene is spent and the only way on is to leave,
    so the frontend flips the left pane to the map (map-as-home). Distinct
    from `scene_exit` (which fires after the player has actually walked):
    the scene is not yet closed here — a closing line still streams once
    the player picks a destination."""
    return {"event": "scene_spent"}


def walk_ready_event() -> dict:
    """Emitted right after every in-scene input cue (options /
    free_text_request / scene_spent), once the engine is awaiting the
    player's action. Arms the map — adjacent pins become clickable so a
    WalkTo can end any turn (mid-scene voluntary leave or a forced
    soft-close). Closes the window where an early POST /walk would 409."""
    return {"event": "walk_ready"}


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


def npc_standing_event(npc_id: str, band: str, impression: str) -> dict:
    """The journalist's read on a dispositioned NPC, for the dossier "People"
    card. Emitted once per present dispositioned NPC at scene entry (seeding the
    board / first meetings) and again only when an NPC's confession band changes
    within a scene — never per turn, so the board reads like *noticing* a thaw,
    not a gauge. `band` is the raw engine band (`engine.band_name`); `impression`
    is the authored `disposition.impression_by_band[band]` line, resolved
    server-side ("" when that band is unauthored). No numbers cross the wire."""
    return {
        "event": "npc_standing",
        "npc_id": npc_id,
        "band": band,
        "impression": impression,
    }


def game_over_event(ending_id: str) -> dict:
    return {"event": "game_over", "ending_id": ending_id}


def error_event(message: str) -> dict:
    """For caller-side errors that the frontend should surface (bad walk
    target, invalid pick index, etc.). Stays out of the prose stream."""
    return {"event": "error", "message": message}
