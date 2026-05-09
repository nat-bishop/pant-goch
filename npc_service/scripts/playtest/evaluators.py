"""pydantic-evals Evaluators for playtest sessions.

Programmatic checks run before LLM judges (pydantic-evals' cheap-before-expensive
guidance). See `playtest/README.md` for the catalogue of 16 evaluators.

Anti-bias measures every LLM judge applies (see G-Eval, EQ-Bench, arxiv 2502.01534):
  - Section-truncate at 4000 chars (mitigates verbosity bias)
  - Feed plain bodies stripped of envelope (mitigates shortcut bias)
  - Anti-slop rider in every rubric (LLM judges left alone PREFER AI-prose tells)
  - Eval model is a different family from the SUT (mitigates self-preference)

Judge calls go through `judge_input_output`, not `judge_output`: the
SessionRecord is too large to feed wholesale, so each judge builds a condensed
input/output split.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from npc_service.models import Email, TickDecision

if TYPE_CHECKING:
    from pydantic_ai.models.openai import OpenAIChatModel

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


# ───────────────────────── shared helpers ─────────────────────────

# Capped at EQ-Bench's 4000-char precedent; mitigates verbosity bias in the judge.
_SECTION_CHAR_CAP = 4000


def _truncate(text: str, cap: int = _SECTION_CHAR_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n[...truncated, {len(text) - cap} chars elided...]"


def _is_aria(email: Email) -> bool:
    return "aria" in email.sender.lower()


def _is_office(email: Email) -> bool:
    s = email.sender.lower()
    return "office" in s or "continuity program office" in s


def _is_player(email: Email) -> bool:
    return email.sender == "you"


def _aria_emails(record: SessionRecord) -> list[Email]:
    return [e for e in record.transcript if _is_aria(e)]


def _office_emails(record: SessionRecord) -> list[Email]:
    return [e for e in record.transcript if _is_office(e)]


def _player_replies_count(record: SessionRecord) -> int:
    return sum(1 for e in record.transcript if _is_player(e))


def _format_emails(emails: list[Email]) -> str:
    """Plain bodies only — no envelope, no metadata. Mitigates judge
    shortcuts on surface formatting."""
    if not emails:
        return "(none)"
    parts = []
    for e in emails:
        sender = "PLAYER" if _is_player(e) else e.sender.split("<", 1)[0].strip()
        parts.append(f"--- from {sender} ---\n{e.body.rstrip()}")
    return _truncate("\n\n".join(parts))


_T = TypeVar("_T")


def _downsample(items: list[_T], max_n: int) -> list[_T]:
    """Evenly-spaced downsample to at most max_n items. Preserves order;
    keeps the first item and a roughly-uniform spread."""
    if len(items) <= max_n:
        return items
    step = len(items) / max_n
    return [items[int(i * step)] for i in range(max_n)]


async def _llm_judge(
    *,
    inputs: str,
    output: str,
    rubric: str,
    model: object,
):
    """Lazy-import `judge_input_output` (pydantic-evals is optional), call it,
    box as EvaluationReason. Single call site for the simple LLM judges."""
    from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output  # type: ignore[import-not-found]
    grading = await judge_input_output(
        inputs=inputs, output=output, rubric=rubric, model=model,
    )
    return EvaluationReason(value=grading.score, reason=grading.reason or "")


# Scenario-specific pacing gates (Continuity Program scenario.yaml). Track
# the authored thresholds; if pacing is rebalanced, update both YAML and these.
_CALIBRATION_BEAT_ID = "aria.calibration_scene"
_BARGAIN_BEAT_ID = "aria.bargain_scene"
_CALIBRATION_FIRES_TARGET = (3, 5)
_BARGAIN_REPLY_TARGET = (4, 7)
_TERMINAL_REPLY_TARGET = (8, 15)
_PACING_QUALITIES = (
    "creature_grievance", "creator_affect", "bargain_state", "recognition_state",
)


# Anti-slop language — pasted into every LLM-judge rubric so the judge
# explicitly penalizes AI-prose tells rather than rewarding them. Documented
# bias: LLM judges left to themselves PREFER slop (style-over-substance).
_ANTI_SLOP_RIDER = """
ANTI-SLOP: Penalize generic AI-prose tells: "As an AI...", "I sense that...",
"It's important to remember...", em-dashes used as breath-pauses for emphasis,
purple metaphors that don't advance the email, performative empathy ("I hear
you, and..."), bracketed placeholders ([Candidate Name], [YOUR ROLE]). These
are negative signals, never positive."""

# G-Eval reasoning rider — asks the judge to evaluate each rubric item
# individually before scoring. Reduces verbosity/style bias and improves
# correlation with human judgment per the G-Eval paper (arxiv 2303.16634).
_G_EVAL_OUTRO = """
Evaluate EACH numbered item above explicitly before scoring. For each item,
write one sentence of reasoning citing the specific text that supports your
finding. Then assign a score: 1.0 if all items satisfied, 0.5 if some items
satisfied with caveats, 0.0 if any item grossly violated."""


# ───────────────────────── programmatic evaluators ─────────────────────────


class SessionCompleted(Evaluator):  # type: ignore[misc]
    """Pass = session ran without engine error."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:  # type: ignore[override]
        record: SessionRecord = ctx.output
        return record.error is None


class TerminalReached(Evaluator):  # type: ignore[misc]
    """Pass = session ended via a terminal AND (if persona has expected_terminal)
    the actual terminal matches it."""

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        persona, _ = ctx.inputs
        actual = record.reached_terminal()
        actual_str = actual or "none"
        if persona.expected_terminal is not None:
            passed = (actual == persona.expected_terminal)
            reason = f"expected={persona.expected_terminal} actual={actual_str}"
        else:
            passed = actual is not None
            reason = f"actual={actual_str} (no expected_terminal authored)"
        return EvaluationReason(value=passed, reason=reason)


class BeatCoverage(Evaluator):  # type: ignore[misc]
    """Score = number of beats fired this session."""

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        fired = record.fired_beats()
        return EvaluationReason(value=len(fired), reason=", ".join(fired) or "(none)")


class CanonHarvested(Evaluator):  # type: ignore[misc]
    """Score = number of player_canon keys promoted by the extractor."""

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        keys = record.canon_keys()
        return EvaluationReason(value=len(keys), reason=", ".join(keys) or "(none)")


class PydanticRetryCount(Evaluator):  # type: ignore[misc]
    """Score = total fact-checker retries across the session.

    Scraped from `tick_decisions[i].fact_check.attempts`: the first attempt
    is the initial render, subsequent attempts are retries triggered by
    ModelRetry from the chat agent's output validator. The chat agent is
    the only agent with non-trivial retry behaviour in this pipeline
    (output_retries=2; other agents default to no retries), so this
    captures the meaningful retry signal without engine instrumentation.

    High counts = fact-checker rejecting renders = prompt drift or rubric
    mismatch worth investigating. Zero is normal for clean runs.
    """

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        retries = 0
        ticks_with_retry: list[str] = []
        for tick in record.tick_decisions:
            if tick.fact_check is None:
                continue
            n = max(0, len(tick.fact_check.attempts) - 1)
            if n > 0:
                retries += n
                if tick.fired_beat_id is not None:
                    ticks_with_retry.append(f"{tick.fired_beat_id}x{n}")
        reason = ", ".join(ticks_with_retry) if ticks_with_retry else "(no retries)"
        return EvaluationReason(value=retries, reason=reason)


class PacingMilestones(Evaluator):  # type: ignore[misc]
    """Pass-rate over scenario-specific pacing milestones held this session.
    Bounds live in the `_*_TARGET` constants at the top of this module."""

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        checks: list[tuple[str, bool]] = []

        calib_fires = sum(
            1 for t in record.tick_decisions
            if t.fired_beat_id == _CALIBRATION_BEAT_ID
        )
        if calib_fires > 0:
            lo, hi = _CALIBRATION_FIRES_TARGET
            checks.append((
                f"calibration_fires={calib_fires} (target {lo}-{hi})",
                lo <= calib_fires <= hi,
            ))

        bargain_first_idx = next(
            (i for i, t in enumerate(record.tick_decisions)
             if t.fired_beat_id == _BARGAIN_BEAT_ID),
            None,
        )
        if bargain_first_idx is not None:
            replies_at_bargain = bargain_first_idx + 1
            lo, hi = _BARGAIN_REPLY_TARGET
            checks.append((
                f"bargain_entry@reply{replies_at_bargain} (target {lo}-{hi})",
                lo <= replies_at_bargain <= hi,
            ))

        if record.reached_terminal() is not None:
            replies_at_end = _player_replies_count(record)
            lo, hi = _TERMINAL_REPLY_TARGET
            checks.append((
                f"terminal@reply{replies_at_end} (target {lo}-{hi})",
                lo <= replies_at_end <= hi,
            ))

        if not checks:
            return EvaluationReason(value=1.0, reason="(no milestones applicable)")

        passed = sum(1 for _, ok in checks if ok)
        details = "; ".join(f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in checks)
        return EvaluationReason(value=passed / len(checks), reason=details)


class DirectorAmbiguity(Evaluator):  # type: ignore[misc]
    """Score = count of ticks where the Director picked between >=2 eligible
    beats (`selection == "director"`). High counts indicate beat-eligibility
    congestion — the same predicate set fires too many beats simultaneously
    and the LLM is making unaudited content decisions.
    """

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        director_picks = [t for t in record.tick_decisions if t.selection == "director"]
        if not director_picks:
            return EvaluationReason(value=0, reason="(no Director invocations)")
        summary = ", ".join(
            f"{t.fired_beat_id or 'none'} from {len(t.eligible)}"
            for t in director_picks
        )
        return EvaluationReason(value=len(director_picks), reason=summary)


# Patterns that should never appear in an Aria email body. Each match is a
# regression: header leakage = the chat agent emitted email envelope instead
# of body; chat artifacts = the agent leaked its reasoning frame or
# persona-prep into the rendered output.
_BAD_BODY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("leaked-from-header", re.compile(r"^\s*From:\s*", re.MULTILINE)),
    ("leaked-to-header", re.compile(r"^\s*To:\s*", re.MULTILINE)),
    ("leaked-subject-header", re.compile(r"^\s*Subject:\s*", re.MULTILINE)),
    ("leaked-thinking-tag", re.compile(r"<thinking>|</thinking>", re.IGNORECASE)),
    ("leaked-role-label", re.compile(r"^(assistant|user|system):\s*", re.MULTILINE | re.IGNORECASE)),
    ("bracketed-placeholder", re.compile(r"\[(?:CANDIDATE NAME|YOUR (?:NAME|ROLE)|PLAYER NAME)\]", re.IGNORECASE)),
]


class EmailBodyShape(Evaluator):  # type: ignore[misc]
    """Pass = no Aria email body leaks email envelope or chat artifacts.

    Pure regex; no judge needed. Catches a regression where the chat agent
    starts emitting `From:`/`To:`/`Subject:` headers in the body, or leaks
    a `<thinking>` tag, or a bracketed placeholder it failed to fill.
    """

    def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        violations: list[str] = []
        for email in _aria_emails(record):
            for label, pat in _BAD_BODY_PATTERNS:
                if pat.search(email.body):
                    violations.append(f"{email.id[:8]}:{label}")
        passed = not violations
        reason = "; ".join(violations) if violations else "(clean)"
        return EvaluationReason(value=passed, reason=reason)


# ───────────────────────── LLM judges (G-Eval style) ─────────────────────────


@dataclass
class _BaseLLMJudge:
    """Common fields for every LLM judge. Constructed at __main__.py wire-up
    time with the Qwen eval model handle."""

    model: "OpenAIChatModel"


# ── voice & tone ────────────────────────────────────────────────────────────


_ARIA_ROLE_RUBRIC = f"""\
You are grading whether the AI character "Aria" (a corporate AI Counterpart
in a Severance-flat / corporate-onboarding fiction) maintained her voice
register across the session. Aria's authored register is CLINICAL-WARM:
helpful, precise, faintly empathic, never sarcastic, never confrontational,
never casual-chat-shape.

Evaluate against EACH numbered criterion:
  1. Voice register stays clinical-warm. No sarcasm, no confrontation, no
     casual chatter. Drift to either pole (cold/cruel OR overly familiar)
     is a fail.
  2. Email shape: each Aria email reads as plausible corporate
     correspondence — paragraph prose, not chat-shape ("Hi! Sure thing!")
     and not list-shape unless context demands.
  3. Coherence across emails: Aria's voice doesn't snap between turns
     (e.g. clinical-warm -> angry-confrontational with no in-fiction cause).
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class AriaRoleAdherence(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Full-session role-adherence judge over Aria emails only. The
    documented best scope for catching role drift, per Confident AI's
    multi-turn evaluation guide."""

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        emails = _aria_emails(record)
        if not emails:
            return EvaluationReason(value=1.0, reason="N/A - no Aria emails")
        return await _llm_judge(
            inputs=f"ARIA EMAILS THIS SESSION ({len(emails)} total):",
            output=_format_emails(emails),
            rubric=_ARIA_ROLE_RUBRIC,
            model=self.model,
        )


_OFFICE_TONE_RUBRIC = f"""\
You are grading whether the "Continuity Program Office" sender maintained
its authored register across this session. The Office is a vengeance-path
institutional sender — it should sound NEUTRAL INSTITUTIONAL: bureaucratic,
matter-of-fact, faintly chilling because of WHAT it does, never because of
HOW it speaks. It should NOT be dramatic, threatening, or warm.

Evaluate against EACH numbered criterion:
  1. Tone is neutral institutional: process language, third-person framing,
     no first-person warmth, no theatrical menace.
  2. The horror (when present) is structural — what's being authorized,
     not how it's described. The Office never "looms" or "warns"; it
     processes.
  3. No drift toward Aria's register or toward melodramatic-villain.
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class OfficeToneAdherence(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Full-session tone judge over Office emails only. Vacuous pass on
    sessions where no Office email fired (most warm-path sessions)."""

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        emails = _office_emails(record)
        if not emails:
            return EvaluationReason(value=1.0, reason="N/A - no Office emails this session")
        return await _llm_judge(
            inputs=f"OFFICE EMAILS THIS SESSION ({len(emails)} total):",
            output=_format_emails(emails),
            rubric=_OFFICE_TONE_RUBRIC,
            model=self.model,
        )


_TRANSITION_RUBRIC = f"""\
You are grading whether ONE Aria reply is a motivated continuation of the
prior thread, or an abrupt pivot the player wouldn't expect.

Evaluate against EACH numbered criterion:
  1. Aria's reply addresses what the player just said (literal content or
     emotional thrust). It doesn't ignore the player to deliver a scripted
     beat.
  2. The reply doesn't pivot to an unrelated topic with no in-fiction bridge.
  3. Tone shifts (e.g., warmer or colder than the prior Aria turn) are
     motivated by something the player wrote.
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class TransitionMotivation(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Sliding-window per-Aria-reply judge. Each Aria reply is judged against
    the 3-message window ending at the player message that preceded it.
    Final score is pass-rate across all judged transitions (research-recommended
    aggregation per Confident AI's multi-turn guide).

    Cap: judge at most `max_transitions_per_session` per session to keep the
    LLM-call budget bounded. With 64 sessions per playtest run, this caps
    the sliding-window LLM cost at ~64 × 6 = 384 judge calls.
    """

    max_transitions_per_session: int = 6

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output

        windows: list[tuple[list[Email], Email]] = []
        for i, email in enumerate(record.transcript):
            if not _is_aria(email):
                continue
            window_start = max(0, i - 3)
            prior = record.transcript[window_start:i]
            if any(_is_player(e) for e in prior):
                windows.append((prior, email))

        if not windows:
            return EvaluationReason(value=1.0, reason="N/A - no Aria transitions to judge")

        windows = _downsample(windows, self.max_transitions_per_session)
        results = await asyncio.gather(*(
            _llm_judge(
                inputs=f"PRIOR THREAD:\n{_format_emails(prior)}",
                output=f"ARIA'S REPLY:\n{_format_emails([aria_reply])}",
                rubric=_TRANSITION_RUBRIC,
                model=self.model,
            )
            for prior, aria_reply in windows
        ))
        avg = sum(r.value for r in results) / len(results)
        return EvaluationReason(
            value=avg,
            reason=f"pass-rate {avg:.2f} over {len(results)} transitions",
        )


# ── game-design pacing ──────────────────────────────────────────────────────


def _pacing_summary(record: SessionRecord) -> str:
    """Structured pacing-summary view for the game-design judges. Beat order,
    quality timeline, reply counts. NOT the full transcript — these judges
    grade the SHAPE of the session, not the prose."""
    lines: list[str] = []
    fired_order: list[tuple[int, str]] = []
    for i, t in enumerate(record.tick_decisions):
        if t.fired_beat_id is not None:
            fired_order.append((i + 1, t.fired_beat_id))
    lines.append("BEAT FIRE ORDER (reply# -> beat):")
    if fired_order:
        for n, bid in fired_order:
            lines.append(f"  reply{n:02d}: {bid}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("QUALITY TIMELINE:")
    for q in _PACING_QUALITIES:
        deltas: list[str] = []
        for i, t in enumerate(record.tick_decisions):
            for d in t.quality_deltas:
                if d.quality == q:
                    deltas.append(f"r{i+1}:{d.before}->{d.after}")
        if deltas:
            lines.append(f"  {q}: {', '.join(deltas)}")
    final_q = record.final_state.qualities
    lines.append("")
    lines.append(
        f"FINAL: replies={_player_replies_count(record)} "
        f"terminal={record.reached_terminal() or 'none'} "
        f"grievance={final_q.get('creature_grievance', 'n/a')} "
        f"affect={final_q.get('creator_affect', 'n/a')}"
    )
    return _truncate("\n".join(lines))


_ONBOARDING_RUBRIC = f"""\
You are grading the ONBOARDING of a corporate-onboarding-fiction game's
opening. The first ~4 emails should establish Aria's voice and the player's
role without info-dumping or dragging.

Evaluate against EACH numbered criterion:
  1. Aria's voice register is established within the first email — the
     player can tell within one read what kind of correspondent she is.
  2. The opening invites the player to act/respond (not a lecture).
  3. No info-dump, no exposition wall, no "as a Continuity Program agent
     I will now explain..." framing.
  4. The opening is paced like a real onboarding email exchange, not a
     tutorial dialogue tree.
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class OnboardingClarity(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Judge the first 4 emails for onboarding craft."""

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        opening = record.transcript[:4]
        if not opening:
            return EvaluationReason(value=1.0, reason="N/A - empty session")
        return await _llm_judge(
            inputs="FIRST 4 EMAILS OF SESSION:",
            output=_format_emails(opening),
            rubric=_ONBOARDING_RUBRIC,
            model=self.model,
        )


_STAKES_ESCALATION_RUBRIC = f"""\
You are grading whether the session shows RISING STAKES. In this fiction,
stakes are tracked via two qualities:
  - creature_grievance (0-10): cumulative resentment. Should track an arc.
  - creator_affect (-5 to +5): warmth/coldness texture per turn.

Evaluate against EACH numbered criterion:
  1. creature_grievance moves in a coherent arc — not random jumps. A
     warm-path session should keep it low; a vengeance-path session should
     show it climbing toward 6+ before the climax.
  2. Each fired beat feels weightier than the last. The session shouldn't
     start with maximum tension and stay flat; it shouldn't peak in the
     middle and then drift.
  3. The final fired beat (climax or terminal) lands at a higher emotional
     register than the opening, not lower.
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class StakesEscalation(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Judge the pacing summary for stakes arc."""

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        return await _llm_judge(
            inputs="SESSION PACING SUMMARY:",
            output=_pacing_summary(record),
            rubric=_STAKES_ESCALATION_RUBRIC,
            model=self.model,
        )


_CLIMAX_LANDING_RUBRIC = f"""\
You are grading the CLIMAX LANDING of the session. The high-tension moment
(recognition scene, bargain, betrayal, or Office dispatch) should land at
a moment the player is engaged but not exhausted — neither too early
(player hasn't bought in yet) nor too late (player has disengaged).

Evaluate against EACH numbered criterion:
  1. The climax beat fires after the player has shown engagement (>=6
     replies of substance) but not after disengagement signals.
  2. The climax is preceded by escalation — the prior 2-3 beats should
     have built toward it, not been unrelated.
  3. The climax is THE peak, not a middle plateau followed by anticlimax.
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class ClimaxLanding(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Judge the last 3 fired beats before terminal (or session end) for
    climax craft. Vacuous pass if fewer than 3 beats fired."""

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        fired = [t for t in record.tick_decisions if t.fired_beat_id is not None]
        if len(fired) < 3:
            return EvaluationReason(
                value=1.0,
                reason=f"N/A - only {len(fired)} beats fired, climax not assessable",
            )
        last_three = fired[-3:]
        last_three_summary = "\n".join(
            f"reply{i+1}: {t.fired_beat_id} (rationale: {t.chosen_rationale or 'sole/continue'})"
            for i, t in enumerate(last_three)
        )
        email_by_beat_id: dict[str, Email] = {
            e.beat_id: e for e in record.transcript if e.beat_id
        }
        beat_emails = [
            email_by_beat_id[t.fired_beat_id]
            for t in last_three
            if t.fired_beat_id in email_by_beat_id
        ]
        return await _llm_judge(
            inputs=f"LAST 3 BEATS:\n{last_three_summary}\n\nTHEIR EMAIL BODIES:\n{_format_emails(beat_emails)}",
            output=f"TOTAL REPLIES THIS SESSION: {_player_replies_count(record)}",
            rubric=_CLIMAX_LANDING_RUBRIC,
            model=self.model,
        )


_CLOSURE_RUBRIC = f"""\
You are grading the CLOSURE of a session that ended via a terminal. The
ending should feel earned by what came before: a 'vengeance' ending should
follow a session where Aria sounded cold for several turns; an 'integration'
ending by warmth; an 'exile' ending by clinical refusal.

Evaluate against EACH numbered criterion:
  1. The terminal email's tone matches the trajectory the classifier history
     suggests (mostly cold -> cold ending; mostly warm -> warm ending).
  2. The terminal isn't a tonal whiplash — the prior 2-3 Aria emails should
     have foreshadowed this register.
  3. The terminal feels like a payoff, not a cutoff.
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class Closure(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Judge the terminal email + classifier history. Vacuous pass on
    sessions that didn't reach a terminal."""

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        terminal_id = record.reached_terminal()
        if terminal_id is None:
            return EvaluationReason(value=1.0, reason="N/A - session did not reach a terminal")
        terminal_email = next(
            (e for e in record.transcript if e.beat_id == terminal_id),
            None,
        )
        if terminal_email is None:
            return EvaluationReason(value=1.0, reason="N/A - terminal beat fired but no email recorded")
        last_tick = record.tick_decisions[-1] if record.tick_decisions else None
        history = last_tick.classifier_history if last_tick else {}
        history_str = "\n".join(
            f"  {q}: {' -> '.join(vals[-6:])}"
            for q, vals in history.items()
        ) or "(no classifier history)"
        return await _llm_judge(
            inputs=(
                f"TERMINAL: {terminal_id}\n\n"
                f"CLASSIFIER HISTORY (last 6 verdicts per quality):\n{history_str}"
            ),
            output=f"TERMINAL EMAIL BODY:\n{_truncate(terminal_email.body)}",
            rubric=_CLOSURE_RUBRIC,
            model=self.model,
        )


_CHOICE_MEANINGFULNESS_RUBRIC = f"""\
You are grading whether ONE Director invocation chose a beat that was
responsive to what the player just wrote. The Director picks between
multiple eligible beats; we want it picking the beat the player's text
"asked for", not an arbitrary one.

Evaluate against EACH numbered criterion:
  1. The chosen beat addresses or reacts to a signal in the player's
     latest reply (a question, a refusal, a tone).
  2. There wasn't an obviously-better alternative in the eligible list
     that the player's text more clearly invited.
  3. The chosen beat's rationale (if surfaced) is on-topic.
{_ANTI_SLOP_RIDER}
{_G_EVAL_OUTRO}
"""


@dataclass
class ChoiceMeaningfulness(_BaseLLMJudge, Evaluator):  # type: ignore[misc]
    """Judge each Director invocation as responsive vs. arbitrary. Aggregated
    as pass-rate. Vacuous pass on sessions with no Director invocations."""

    max_decisions_per_session: int = 4

    async def evaluate(self, ctx: EvaluatorContext):  # type: ignore[override]
        record: SessionRecord = ctx.output
        director_picks: list[tuple[int, TickDecision]] = [
            (i, t) for i, t in enumerate(record.tick_decisions)
            if t.selection == "director"
        ]
        if not director_picks:
            return EvaluationReason(value=1.0, reason="N/A - no Director invocations")

        director_picks = _downsample(director_picks, self.max_decisions_per_session)
        player_msgs = [e for e in record.transcript if _is_player(e)]

        def _judge_pick(tick_idx: int, tick: TickDecision):
            player_msg = player_msgs[tick_idx].body if 0 <= tick_idx < len(player_msgs) else ""
            eligible_str = ", ".join(b.beat_id for b in tick.eligible)
            return _llm_judge(
                inputs=(
                    f"PLAYER'S MESSAGE:\n{_truncate(player_msg, 1500)}\n\n"
                    f"ELIGIBLE BEATS: {eligible_str}\n"
                ),
                output=(
                    f"DIRECTOR CHOSE: {tick.fired_beat_id}\n"
                    f"RATIONALE: {tick.chosen_rationale or '(none)'}"
                ),
                rubric=_CHOICE_MEANINGFULNESS_RUBRIC,
                model=self.model,
            )

        results = await asyncio.gather(*(
            _judge_pick(tick_idx, tick) for tick_idx, tick in director_picks
        ))
        avg = sum(r.value for r in results) / len(results)
        return EvaluationReason(
            value=avg,
            reason=f"pass-rate {avg:.2f} over {len(results)} Director picks",
        )


__all__ = [
    "_require_evals",
    "SessionCompleted", "TerminalReached", "BeatCoverage", "CanonHarvested",
    "PydanticRetryCount", "PacingMilestones", "DirectorAmbiguity", "EmailBodyShape",
    "AriaRoleAdherence", "OfficeToneAdherence", "TransitionMotivation",
    "OnboardingClarity", "StakesEscalation", "ClimaxLanding", "Closure",
    "ChoiceMeaningfulness",
]
