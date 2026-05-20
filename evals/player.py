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

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from handler.schemas import FreeText, ProsePart
from handler.state import State
from handler.world import World


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


def render_scene(prose: str, options: list) -> str:
    lines = [prose.strip(), "", "Your options:"]
    for i, opt in enumerate(options, start=1):
        lines.append(f"  {i}. {opt.text}")
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
        lines.append(f"  {i}. {world.locations[loc_id].name}")
    lines.append("")
    lines.append("Reply with the number of the place your character heads to.")
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

    async def _pick_index(self, prompt: str, n: int) -> int:
        result = await self.agent.run(
            prompt, output_type=PlayerChoice, message_history=self.history,
        )
        self.history = result.all_messages()
        return _clamp(result.output.choice - 1, 0, n - 1)

    async def pick_option(self, prose_part: ProsePart) -> int:
        options = prose_part.next.options  # type: ignore[union-attr]
        return await self._pick_index(
            render_scene(prose_part.prose, options), len(options),
        )

    async def pick_free_text(self, req: FreeText) -> str:
        result = await self.agent.run(
            render_free_text(req.prompt), output_type=str, message_history=self.history,
        )
        self.history = result.all_messages()
        return result.output.strip() or "I take a moment to look around."

    async def pick_walk(self, state: State, world: World) -> str:
        adjacents = world.locations[state.player_location].adjacent_to
        idx = await self._pick_index(
            render_walk(state, world, adjacents), len(adjacents),
        )
        return adjacents[idx]
