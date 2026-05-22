"""Gemini-backed simulated player.

`GeminiPlayer` implements the three callbacks `handler.scene_loop.play_session`
expects (`pick_option`, `pick_free_text`, `pick_walk`) against one pydantic-ai
`Agent` whose message history persists across the whole session — so the player
remembers its own playthrough. Persona lives in the agent's `instructions`.

The engine's ids are grammar-sampled and so always valid; the player has no
such guarantee, so every numeric choice is clamped into range here rather than
trusted.
"""
from __future__ import annotations

import asyncio
import random

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from handler.schemas import FreeText, ProsePart
from handler.state import State
from handler.world import World

# The cloud player rides a shared Gemini endpoint that intermittently returns
# 5xx / 429 under load. Without a retry these transient blips raise out of
# run_session and the WHOLE session is dropped from the run — a Gemini hiccup
# silently shrank a 16-session playtest to 2 surviving artifacts, and the report
# presented those 2 as the corpus. Retry with backoff so a transient provider
# error costs a few seconds, not a session.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 6
# Hard wall-clock cap on a single player call. The Gemini endpoint occasionally
# does not return a 5xx but simply HANGS (connection open, no body) — which the
# status/hint retry below never catches, so one stuck call blocks the session and,
# because run.py awaits every session in one gather, the WHOLE run (and its
# summary.json) hangs behind it. A normal call is ~1–3s, so a 120s ceiling only
# trips on a true stall; the resulting TimeoutError is treated as transient and
# retried like any other blip. (Harness resilience only — not a scoring change.)
_RUN_TIMEOUT_S = 120.0
# Free-text fallback for transient errors that arrive WITHOUT a structured
# status_code (raw httpx/network failures). Phrase hints only — bare numbers
# like "500" were dropped because a non-retryable message that merely contains
# such a substring (a quota note mentioning "1500", a validation error quoting
# an HTTP 500 example) would be retried 6× and mask the real failure. Genuine
# 5xx/429 responses carry status_code and are caught structurally above.
_TRANSIENT_HINTS = (
    "overloaded", "temporarily unavailable", "service unavailable",
    "timeout", "timed out", "connection reset", "connection error",
    "connection aborted", "server disconnected",
)


_BASE_INSTRUCTIONS = (
    "You are playing a text-based investigation game. You are a freelance "
    "journalist who has come to the Welsh Marches village of Pant Goch to look "
    "into a death the local police closed too quickly. Stay fully in character "
    "as this journalist for the entire session. Never mention that this is a "
    "game or that you are an AI; never describe the mechanics. You will be shown "
    "the current scene and a set of choices, and asked to decide what your "
    "character does. Make every decision consistent with the persona below."
)


class PlayerChoice(BaseModel):
    """Structured pick. `reasoning` forces a moment of deliberation (and gives a
    readable trace); `choice` is the 1-indexed option the player selects."""

    reasoning: str
    choice: int


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


# The always-available map exit, mirroring the real UI's "Step outside" entry
# (engine appends it to the options after LEAVE_OPTION_MIN_HISTORY turns). The
# eval player picks this when a scene has stopped being productive — the agency
# a human has and the harness previously denied (it could only keep picking
# options or leave on a rare coin flip).
LEAVE_CHOICE_LABEL = (
    "Leave — you've got what there is to get here for now; step out and move on "
    "elsewhere in the village."
)


def render_scene(prose: str, labels: list[str]) -> str:
    lines = [prose.strip(), "", "Your options:"]
    for i, label in enumerate(labels, start=1):
        lines.append(f"  {i}. {label}")
    lines.append("")
    lines.append(
        "Reply with the number of the option your character chooses, and a one-"
        "sentence reason in character."
    )
    return "\n".join(lines)


def render_free_text(prompt: str) -> str:
    return (
        f"{prompt.strip()}\n\n"
        "Write your character's next move — what they say or do — in their own "
        "voice. One or two sentences."
    )


def render_walk(state: State, world: World, adjacents: list[str]) -> str:
    here = world.locations[state.player_location].name
    lines = [f"You are leaving {here}. Where do you go next?", "", "You can walk to:"]
    for i, loc_id in enumerate(adjacents, start=1):
        loc = world.locations[loc_id]
        # First sentence of the authored description — the same place-character
        # a human reads off the map — so the player navigates with intent
        # rather than picking a bare name blind.
        gist = loc.description.strip().split(". ")[0].replace("\n", " ").strip()
        lines.append(f"  {i}. {loc.name} — {gist}.")
    lines.append("")
    lines.append(
        "Reply with the number of the place your character heads to. As a "
        "journalist, favour leads and places you have not yet followed up."
    )
    return "\n".join(lines)


class GeminiPlayer:
    """One simulated player for one session. Holds its own conversation history.

    `model_name` defaults are not set here — the caller (run_session / run.py)
    passes the configured model so there is a single source of truth.
    """

    def __init__(
        self, persona_prompt: str, *, model_name: str, api_key: str | None = None,
    ) -> None:
        instructions = f"{_BASE_INSTRUCTIONS}\n\n{persona_prompt.strip()}"
        self.agent = Agent(
            GoogleModel(model_name, provider=GoogleProvider(api_key=api_key)),
            instructions=instructions,
            name="gemini_player",
        )
        self.history: list = []

    async def _run(self, prompt: str, **kwargs):
        """Run the agent with retry+backoff on transient provider errors,
        advancing `self.history` only on success."""
        delay = 1.0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                result = await asyncio.wait_for(
                    self.agent.run(prompt, message_history=self.history, **kwargs),
                    timeout=_RUN_TIMEOUT_S,
                )
                self.history = result.all_messages()
                return result
            except Exception as e:  # noqa: BLE001 — narrow to transient below
                status = getattr(e, "status_code", None)
                transient = (
                    isinstance(e, (asyncio.TimeoutError, TimeoutError))
                    or (isinstance(e, ModelHTTPError) and status in _RETRYABLE_STATUS)
                    or (status in _RETRYABLE_STATUS)
                    or any(h in str(e).lower() for h in _TRANSIENT_HINTS)
                )
                if not transient or attempt == _MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(delay + random.uniform(0, 0.75))
                delay = min(delay * 2, 30.0)

    async def _pick_index(self, prompt: str, n: int) -> int:
        result = await self._run(prompt, output_type=PlayerChoice)
        return _clamp(result.output.choice - 1, 0, n - 1)

    async def pick_option(self, prose_part: ProsePart, *, can_leave: bool = False) -> int:
        """Choose among the offered options. When `can_leave` is True, a final
        Leave entry is appended (index == len(options)); the caller routes that
        index to a map walk. Mirrors the real UI, where the map exit is live on
        every turn once a scene has a couple of turns behind it."""
        options = prose_part.next.options  # type: ignore[union-attr]
        labels = [o.text for o in options]
        if can_leave:
            labels.append(LEAVE_CHOICE_LABEL)
        return await self._pick_index(
            render_scene(prose_part.prose, labels), len(labels),
        )

    async def pick_free_text(self, req: FreeText) -> str:
        result = await self._run(render_free_text(req.prompt), output_type=str)
        return result.output.strip() or "I take a moment to look around."

    async def pick_walk(self, state: State, world: World) -> str:
        adjacents = world.locations[state.player_location].adjacent_to
        idx = await self._pick_index(
            render_walk(state, world, adjacents), len(adjacents),
        )
        return adjacents[idx]
