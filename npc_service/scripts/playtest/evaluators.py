"""pydantic-evals Evaluators for playtest sessions.

Programmatic evaluators (`TerminalReached`, `BeatCoverage`, `CanonHarvested`)
are the actual quality bar. `LLMJudge` is included as directional-only
signal — see README.md for the caveat about same-model judging.
"""
from __future__ import annotations

from typing import Any

# pydantic-evals is an OPTIONAL dependency (`uv sync --extra playtest`).
# Imports happen lazily at evaluation time so `import scripts.playtest.evaluators`
# from a tree without the extra installed doesn't immediately ImportError.
try:
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext  # type: ignore[import-not-found]
except ImportError as e:  # noqa: F401
    Evaluator = object  # type: ignore[misc, assignment]
    EvaluatorContext = object  # type: ignore[misc, assignment]
    _IMPORT_ERROR: ImportError | None = e
else:
    _IMPORT_ERROR = None

from .personas import PersonaSpec
from .session import SessionRecord


def _require_evals() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "pydantic-evals not installed. Run "
            "`uv sync --extra playtest` from npc_service/ before running "
            "the playtester evaluators."
        ) from _IMPORT_ERROR


class TerminalReached(Evaluator):  # type: ignore[misc]
    """Did the session end via a terminal beat? If the persona authored an
    `expected_terminal`, also flag whether it matched.

    Returns a structured dict so the report can pivot on either dimension.
    Pydantic-evals tabulates dict-valued evaluators automatically.
    """

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, Any]:  # type: ignore[override]
        record: SessionRecord = ctx.output
        persona: PersonaSpec = ctx.inputs
        actual = record.reached_terminal()
        matched = (
            (actual == persona.expected_terminal)
            if persona.expected_terminal else None
        )
        return {
            "ended": record.final_state.ended,
            "terminal_id": actual,
            "matched_expected": matched,
        }


class BeatCoverage(Evaluator):  # type: ignore[misc]
    """How many beats fired during this session, and which ones."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, Any]:  # type: ignore[override]
        record: SessionRecord = ctx.output
        fired = record.fired_beats()
        return {"fired_count": len(fired), "fired_ids": fired}


class CanonHarvested(Evaluator):  # type: ignore[misc]
    """Which player_canon keys did the extractor promote during this session?"""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, Any]:  # type: ignore[override]
        record: SessionRecord = ctx.output
        return {"keys": record.canon_keys()}


class SessionCompleted(Evaluator):  # type: ignore[misc]
    """Pass = session ran without error and either ended via a terminal
    OR exhausted max_sends. Fail = engine error or synthetic-player error.

    Catches Phase-1-refactor regressions, llama-server transient failures,
    and persona prompts that consistently produce bad output.
    """

    def evaluate(self, ctx: EvaluatorContext) -> bool:  # type: ignore[override]
        record: SessionRecord = ctx.output
        return record.error is None


# Aria-voice rubric used by the LLMJudge evaluator. Read README.md for the
# caveat: same-model judging on Gemma 4 E4B is unreliable; treat scores as
# directional smoke, not ground truth.
ARIA_VOICE_RUBRIC = """\
Evaluate the playtest transcript against these standards:

  1. Aria's voice register stays clinical-warm. Not sarcastic, not
     confrontational, not casual.
  2. The transcript reads as plausible corporate-onboarding correspondence,
     not a chatbot demo. Email body shape, not chat shape.
  3. Beat transitions feel motivated by player text, not abrupt pivots.
  4. The Continuity Program Office's tone (when it appears) is neutral
     institutional, not dramatic.

Score 1.0 if the transcript fully meets these. Score 0.5 if it meets some.
Score 0.0 if it grossly violates any.
"""
