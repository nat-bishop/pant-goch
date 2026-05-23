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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from .verdicts import PassFail, Scored

Verdict = PassFail | Scored

# Per-case wall-clock ceiling. Without it, one stuck response (a vLLM request that
# never returns its body) hangs the whole evaluation indefinitely. On timeout the
# raised TimeoutError is recorded by pydantic-evals as an evaluator failure (visible
# in the report) and the run still terminates.
CASE_TIMEOUT_S = 180.0


def _judge_model_settings(model: Model) -> dict:
    # Settings are split by judge model family. The local Qwen/inferno judge gets
    # its vendor-recommended sampling + a hard token ceiling; a cloud judge
    # (gemini:/claude:, opt-in A/B) is left to its own defaults — pinning ours
    # onto it fights the vendor's guidance (Gemini 3.x wants temp≈1.0) and risks
    # truncating a thinking model whose reasoning spends the same token budget.
    inner = getattr(model, "wrapped", model)  # unwrap RetryModel
    if isinstance(inner, OpenAIChatModel):
        # temp=0.6: Qwen3's recommended value — NOT greedy (temp=0), which Qwen
        # warns degrades into repetition. Reproducibility across runs is bounded
        # well enough at 0.6 for a trended metric. max_tokens=768: a verdict is
        # tiny (a PassFail/Scored + a reason string), so cap hard — without it a
        # runaway response generates until it trips CASE_TIMEOUT_S, the dominant
        # cause of errored judge cases. The cap (not temp=0) is now what bounds
        # the runaway, so the recommended non-greedy temp is safe.
        return {"temperature": 0.6, "max_tokens": 768}
    return {}


def build_judge_agent(model: Model, output_type: type[BaseModel], system_prompt: str) -> Agent:
    """A judge agent with the shared wiring every HANDLER judge uses: NativeOutput
    (json_schema response_format — the local Qwen/inferno endpoint errors on the
    default tool-output mode) and family-aware sampling (`_judge_model_settings`:
    the local Qwen gets temp=0.6 + a hard token cap; a cloud judge keeps its own
    defaults). Used by both the single-item `AgentJudge` and the pairwise judge so
    they can't drift apart."""
    return Agent(
        model,
        output_type=NativeOutput(output_type),
        instructions=system_prompt,
        model_settings=_judge_model_settings(model),
    )


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
        # system_prompt are stable). Family-aware sampling lives in build_judge_agent
        # (shared with the pairwise judge), so both judge kinds get the local-Qwen
        # token cap / cloud-default split from one place.
        cached = getattr(self, "_agent_cache", None)
        if cached is None:
            cached = build_judge_agent(self.model, self.output_type, self.system_prompt)
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
