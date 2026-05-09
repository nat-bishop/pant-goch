"""pydantic-evals Evaluators for playtest sessions.

Programmatic evaluators (`TerminalReached`, `BeatCoverage`, `CanonHarvested`)
are the actual quality bar. `LLMJudge` is included as directional-only
signal — see README.md for the caveat about same-model judging.

Each evaluator returns one `EvaluationReason` (scalar value + descriptive
reason). pydantic-evals' Evaluator protocol allows scalars, EvaluationReason,
or `Mapping[str, scalar | EvaluationReason]` — but the mapping form rejects
`None`-valued or list-valued entries, which we'd hit with `terminal_id=None`
and `fired_ids=[…]`. Single EvaluationReason keeps the structured detail
in the human-readable `reason` and the assertion bit/scalar in `value`.
"""
from __future__ import annotations

# pydantic-evals is an OPTIONAL dependency (`uv sync --extra playtest`).
# Imports happen lazily at evaluation time so `import scripts.playtest.evaluators`
# from a tree without the extra installed doesn't immediately ImportError.
try:
    from pydantic_evals.evaluators import (  # type: ignore[import-not-found]
        EvaluationReason,
        Evaluator,
        EvaluatorContext,
    )
except ImportError as e:  # noqa: F401
    Evaluator = object  # type: ignore[misc, assignment]
    EvaluatorContext = object  # type: ignore[misc, assignment]
    EvaluationReason = None  # type: ignore[misc, assignment]
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
    """Pass when the session ended via a terminal AND (if a persona had an
    `expected_terminal`) the actual terminal matches it. The reason string
    carries the actual terminal id for grouping/debugging in the report.
    """

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        persona: PersonaSpec = ctx.inputs[0]
        actual = record.reached_terminal()
        if persona.expected_terminal is not None:
            passed = (actual == persona.expected_terminal)
            reason = (
                f"expected={persona.expected_terminal} actual={actual or 'none'}"
            )
        else:
            passed = actual is not None
            reason = f"actual={actual or 'none'} (persona did not author an expected terminal)"
        return EvaluationReason(value=passed, reason=reason)


class BeatCoverage(Evaluator):  # type: ignore[misc]
    """Score = number of beats fired this session. The reason lists the
    fired ids so they're inspectable in the report without expanding to
    a separate evaluator per id.
    """

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        fired = record.fired_beats()
        return EvaluationReason(value=len(fired), reason=", ".join(fired) or "(none)")


class CanonHarvested(Evaluator):  # type: ignore[misc]
    """Score = number of player_canon keys promoted. Reason lists them.

    Useful as a smoke metric for the extractor — zero across all sessions
    means the extractor is silent (or broken); 20+ means it's overharvesting
    near-synonyms.
    """

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        keys = record.canon_keys()
        return EvaluationReason(value=len(keys), reason=", ".join(keys) or "(none)")


class SessionCompleted(Evaluator):  # type: ignore[misc]
    """Pass = session ran without error and either ended via a terminal
    OR exhausted max_sends. Fail = engine error or synthetic-player error.

    Catches Phase-1-refactor regressions, llama-server transient failures,
    and persona prompts that consistently produce bad output.
    """

    def evaluate(self, ctx: EvaluatorContext) -> bool:  # type: ignore[override]
        record: SessionRecord = ctx.output
        return record.error is None


def _judge_view(record: SessionRecord, *, max_turns: int = 12) -> str:
    """Build a compact judge-readable transcript view.

    The full SessionRecord serializes to ~30k+ tokens (transcript + every
    tick decision + full final state). LLMJudge's default behaviour is to
    JSON-dump `ctx.output` into the prompt, which blows past any sane
    context cap. We trim to `max_turns` of the most recent NPC↔player
    exchange — enough for a voice-register judgement, small enough to
    leave room for the rubric.
    """
    transcript = record.transcript
    tail = transcript[-max_turns:]
    lines = [f"PERSONA: {record.persona_name}"]
    if record.error:
        lines.append(f"SESSION ERRORED: {record.error}")
    lines.append(f"TERMINAL: {record.reached_terminal() or '(none)'}")
    lines.append(f"FIRED BEATS: {', '.join(record.fired_beats()) or '(none)'}")
    lines.append("")
    lines.append(
        f"TRANSCRIPT (last {len(tail)} of {len(transcript)} emails):"
    )
    for email in tail:
        sender = "PLAYER" if email.sender == "you" else email.sender
        lines.append(f"--- from {sender} ---")
        lines.append(email.body.rstrip())
    return "\n".join(lines)


class AriaVoiceJudge(Evaluator):  # type: ignore[misc]
    """LLM-as-judge wrapper that hands the model a condensed transcript
    view rather than the full SessionRecord. Stand-in for `LLMJudge` —
    bypasses pydantic-evals' default `json.dumps(output)` prompt builder
    so the judge's prompt fits in the model's context window.

    Same disclaimer as LLMJudge in this codebase: same-model judging on
    Gemma 4 E4B is directional smoke, not ground truth.
    """

    rubric: str
    model: object  # OpenAIChatModel; kept untyped to avoid the optional dep here.
    model_settings: dict | None = None

    def __init__(self, rubric: str, model: object, model_settings: dict | None = None):
        self.rubric = rubric
        self.model = model
        self.model_settings = model_settings

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        from pydantic_evals.evaluators.llm_as_a_judge import judge_output  # type: ignore[import-not-found]
        record: SessionRecord = ctx.output
        condensed = _judge_view(record)
        grading = await judge_output(
            condensed, self.rubric, self.model, self.model_settings,
        )
        return EvaluationReason(
            value=grading.pass_, reason=grading.reason or "",
        )


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
