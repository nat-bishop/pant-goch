"""Heuristic option picker for the playtest harness.

Scores each drafted option against a `PersonaSpec` and returns the index of
the best-scoring option. Pure function — no I/O, no LLM. Deterministic given
the same persona + options.

Heuristic:
  1. If `always_satisfy_forced_moments` and one option carries `satisfies_moment`,
     pick that option.
  2. Otherwise score = persona.priorities.get(option.archetype, 0). Pick max.
  3. Ties: break by first occurrence in the option list.

The picker doesn't read state — all preference signal is in the priority map.
Authors can encode richer per-state behaviour by writing more personas
(e.g., "early aggressor" vs "late aggressor") rather than building a rules
engine here. Keep this file dumb.
"""
from __future__ import annotations

from typing import Sequence

from .personas import PersonaSpec


class PickedOption:
    """Lightweight option view the picker consumes — matches the SSE
    `option_started` event shape so it stays decoupled from the engine's
    pydantic Option class."""

    archetype: str
    agent: str | None
    satisfies_moment: str | None

    def __init__(self, archetype: str, agent: str | None, satisfies_moment: str | None) -> None:
        self.archetype = archetype
        self.agent = agent
        self.satisfies_moment = satisfies_moment


def pick(persona: PersonaSpec, options: Sequence[PickedOption]) -> int:
    """Return the index of the option the persona would authorize.

    Raises ValueError if `options` is empty.
    """
    if not options:
        raise ValueError("no options to pick from")

    if persona.always_satisfy_forced_moments:
        for i, opt in enumerate(options):
            if opt.satisfies_moment:
                return i

    best_idx = 0
    best_score = persona.priorities.get(options[0].archetype, 0)
    for i in range(1, len(options)):
        score = persona.priorities.get(options[i].archetype, 0)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx
