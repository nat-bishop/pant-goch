"""Base for one-thing LLM judges.

A judge owns three things: (1) the exact text it shows the model — `render(item)`,
which serializes the authored ground truth alongside the transition/session so the
judge scores against the bible, not vibes; (2) a frozen single-purpose rubric —
`system_prompt`; (3) one structured verdict — `output_type` (PassFail or Scored).

Judges are deliberately NOT pydantic-evals `Evaluator`s: their unit is a turn or a
session extracted from a finished artifact, not a task to run, so they are driven
by the purpose-built `scorer.py` (gather → aggregate → persist) rather than a
`Dataset.evaluate` over a no-op task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models import Model

from .verdicts import PassFail, Scored

Verdict = PassFail | Scored


@dataclass
class AgentJudge:
    """Wraps a shared judge model. Subclasses set `name`, `system_prompt`,
    `output_type` and implement `render`."""

    model: Model

    name: ClassVar[str] = "judge"
    system_prompt: ClassVar[str] = ""
    output_type: ClassVar[type[BaseModel]] = PassFail

    def render(self, item) -> str:
        raise NotImplementedError

    def _agent(self) -> Agent:
        # Built once and reused across this judge's cases (model, output_type and
        # system_prompt are stable). NativeOutput (response_format json_schema),
        # not tool-calling: the local Qwen/inferno endpoint returns an error body
        # for the default tool-output mode but handles native JSON-schema cleanly.
        cached = getattr(self, "_agent_cache", None)
        if cached is None:
            cached = Agent(
                self.model, output_type=NativeOutput(self.output_type),
                instructions=self.system_prompt,
            )
            self._agent_cache = cached
        return cached

    async def judge(self, item) -> Verdict:
        """Score one item; returns the structured verdict (PassFail | Scored)."""
        result = await self._agent().run(self.render(item))
        return result.output
