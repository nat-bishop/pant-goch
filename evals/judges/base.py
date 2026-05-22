"""Base for one-thing LLM judges.

A judge owns three things: (1) the exact text it shows the model — `render(item)`,
which serializes the authored ground truth alongside the transition/session so the
judge scores against the bible, not vibes; (2) a frozen single-purpose rubric —
`system_prompt`; (3) one structured verdict — `output_type` (PassFail or Scored).

Judges ARE pydantic-evals `Evaluator`s: `judge_run.py` wraps the per-scope item
lists in `Dataset`s over an identity task and calls `Dataset.evaluate`, so judge
results land in the same Logfire experiment surface (and run-over-run diff) as the
deterministic invariants. The bridge is `evaluate()` below: it maps the structured
verdict onto pydantic-evals' value model — `PassFail.passed` → a bool assertion
(pass-rate), `Scored.score` → a float score (averaged) — carrying the reason via
`EvaluationReason`. Per-judge `name` becomes the report key; `model`/`world` are
code-level parameters, not data, so they are kept out of the serialized spec.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models import Model
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from .verdicts import PassFail, Scored

Verdict = PassFail | Scored

# Per-case wall-clock ceiling. Without it, one stuck response (a vLLM request that
# never returns its body) hangs the whole evaluation indefinitely. On timeout the
# raised TimeoutError is recorded by pydantic-evals as an evaluator failure (visible
# in the report) and the run still terminates.
CASE_TIMEOUT_S = 180.0


@dataclass
class AgentJudge(Evaluator):
    """Wraps a shared judge model as a pydantic-evals `Evaluator`. Subclasses set
    `name`, `system_prompt`, `output_type` and implement `render`."""

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

    # --- pydantic-evals Evaluator interface ---------------------------------

    def get_default_evaluation_name(self) -> str:
        """Use the judge's own `name` (e.g. `beat_fired_appropriately`) as the
        report/Logfire key, not the class name."""
        return self.name

    def build_serialization_arguments(self) -> dict:
        """The judge is parameterized by code (model, world), not by serializable
        data, so emit an empty spec rather than dumping a Model/World."""
        return {}

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        """Score `ctx.output` (a Transition or SessionView) and bridge the verdict
        onto pydantic-evals: PassFail → bool assertion (pass-rate), Scored → float
        score (averaged). The reason rides along for the report."""
        v = await asyncio.wait_for(self.judge(ctx.output), timeout=CASE_TIMEOUT_S)
        value = v.passed if isinstance(v, PassFail) else float(v.score)
        return EvaluationReason(value=value, reason=v.reason)
