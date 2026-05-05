"""The four LLM agents: CHAT_AGENT, CLASSIFIER_AGENT, SUMMARY_AGENT, FACT_CHECKER_AGENT.

CHAT_AGENT renders one email body per turn. It uses Pydantic AI's native
`message_history` for the per-thread conversation, plain-text `output_type=str`
so the assistant turn lands as a `TextPart` (clean for subsequent turns), and
dynamic `@instructions` keyed on `ChatDeps` to layer dossier, optional plot
directive, and current commitments per run. An `@output_validator` engages
ONLY when the firing beat sets `spec.validate_facts=True`: it runs the
fact-checker against the beat's must_include_facts / must_not lists and
raises ModelRetry on any miss. Default-off keeps loop beats free of
validator latency.

CLASSIFIER_AGENT is one-shot, ToolOutput. It returns a list of per-axis
results (one entry per beat-authored axis in `classify_reply_axes`), each
with a `quality`, `value`, `rationale`. The allowed-values list per axis
is enforced via the user prompt (not the schema) since values vary per
axis. The caller validates post-hoc per axis and skips the quality write
if a returned value is off-spec. Single batched call regardless of axis
count keeps per-send cost flat.

SUMMARY_AGENT runs only inside the chat agent's history_processors, when
message_history has grown past the configured token budget. It takes the
oldest messages and produces a single recap that replaces them. Same model
handle as the chat agent (we don't have a separate cheaper provider in this
local-llama setup), so the cost is one extra LLM round-trip on the rare
overflow turn — preferred over drop-oldest because rapport carries forward.

FACT_CHECKER_AGENT is opt-in, ToolOutput, and only invoked from the chat
agent's output validator on beats with `spec.validate_facts=True`. Returns
`{missing_facts, violations}`; both empty → output passes through; either
non-empty → ModelRetry. Pay the cost only on critical beats.

Construction is deferred to `build_agents()` (called from `lifespan`) so the
shared OpenAI-compatible model handle can be passed in once.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.messages import ModelMessage

from .models import (
    AxisResult,
    BeatSpec,
    ClassifyReply,
    NPCDossier,
)


COMMITMENT_PREFIX = "commitment."


# Type alias: a function that estimates the token cost of a list of messages.
# Production uses llama-server's `/tokenize`; tests inject a stub so neither
# the network nor the tokenizer is exercised.
TokenCounter = Callable[[list[ModelMessage]], Awaitable[int]]


# ───────────────────────── chat-agent deps ─────────────────────────


@dataclass
class ChatDeps:
    """Per-run injection for the chat agent.

    `plot_directive` is None for chat-turn fallbacks (free conversation in
    character) and is a `BeatSpec` when a PlotBeat is firing on this turn —
    in which case the directive is layered into the prompt as authored
    intent / must-include / must-not.

    `commitments` is a snapshot of qualities matching the `commitment.*`
    convention captured at render-call time, mapping the full quality id
    to (current value, description). The decorator `_commitments_block`
    surfaces them with the `commitment.` prefix stripped. Snapshotting
    in deps keeps deps immutable for the duration of one agent run, which
    matches the `plot_directive` pattern.
    """

    npc: NPCDossier
    plot_directive: BeatSpec | None
    commitments: dict[str, tuple[float, str]] = field(default_factory=dict)


# ───────────────────────── system prompts ─────────────────────────


_BASE_CHAT_INSTRUCTIONS = """You are writing a single email body in the voice of the NPC described below.

OUTPUT FORMAT (strict):
- Body text only. No subject line. No "From:" / "To:" / "Date:" headers.
- No sign-off / signature unless the dossier explicitly specifies one.
- No quoted prior message. No "Re:" prefix in the output.
- No JSON, no markdown code fences, no meta-commentary.

VOICE RULES:
- Stay inside the dossier's base voice register.
- Do not invent named characters or places not in the dossier or transcript.
- Do not mention game mechanics, scoring, or the meta-fiction.
- Address the recipient by name or as "you", never as "the player".
"""


_CLASSIFIER_SYSTEM_PROMPT = """You classify a player-written email reply against one or more beat-authored axes.

You will receive:
- The recipient's persona blurb (who the player is writing to).
- The most recent prior email in the thread (what the player is replying to).
- The player's just-typed reply (the new send).
- A list of CLASSIFICATION AXES. Each axis has:
  - quality: an internal id used to route the result.
  - question: what the author wants to know.
  - allowed_values: the exact strings you may return for that axis.

For each axis, pick the single allowed value that best matches the player's
reply on that axis. Axes are independent — judge each one against its own
question, and do not let a value from one axis bias another. Do not return
values outside the listed allowed_values for the relevant axis. Write a
one-sentence rationale per axis (telemetry only).

Return one result per axis, in the order the axes were given:
[{"quality": "<axis quality>", "value": "<one of that axis's allowed_values>", "rationale": "<one sentence>"}, ...]
"""


_FACT_CHECKER_SYSTEM_PROMPT = """You audit a candidate email body against authored constraints.

You will receive:
- The candidate body the chat agent just produced.
- A list of REQUIRED FACTS — each entry must be present (in spirit) in the body.
- A list of FORBIDDEN ITEMS — none of these may appear in the body (in spirit).

Return:
- missing_facts: required entries that are NOT meaningfully present in the body.
- violations: forbidden entries that ARE present in the body.

Be strict but charitable: a fact is present if the body conveys the same
meaning, even if the wording differs. A violation is a clear hit, not a
suspicion. Both lists empty → the body passes.
"""


_SUMMARY_INSTRUCTIONS = """You are summarising the older portion of an email exchange between a human player and Aria, an AI counterpart in a corporate-satire scenario. Your output replaces the messages you have been shown — write it as a single message that captures everything the next conversation turn might need to recall.

Focus on:
- Topics the player and Aria discussed
- Facts disclosed by either party (names, schedules, opinions, preferences)
- Decisions, agreements, or unfinished questions
- The emotional tenor of the exchange (warm, distant, deflective, engaged)

Omit small talk, pleasantries, and repetition.

Output: a single concise paragraph in third person. No header, no list formatting, no meta-commentary.
"""


# ───────────────────────── agent construction ─────────────────────────


class FactCheckResult(BaseModel):
    """Output of the fact-checker agent.

    `missing_facts` lists `must_include_facts` entries the candidate body
    fails to convey; `violations` lists `must_not` entries it does convey.
    Both empty → the body passes the validator.
    """

    missing_facts: list[str]
    violations: list[str]


@dataclass
class AgentBundle:
    """Four agents wired up against one shared model handle."""

    chat_agent: Agent
    classifier_agent: Agent
    summary_agent: Agent
    fact_checker_agent: Agent


def build_agents(
    model,
    *,
    token_counter: TokenCounter | None = None,
    token_budget: int | None = None,
    history_keep_tail: int = 4,
) -> AgentBundle:
    """Construct the four agents. The chat agent's per-run customization is
    in `@instructions` decorators below — they read from `ctx.deps` (ChatDeps)
    so a single shared agent handles every NPC and every turn type. The
    `@output_validator` engages only when the firing beat sets
    `spec.validate_facts=True`; the fact-checker agent is referenced via
    closure so the validator never needs it on `ChatDeps`.

    When `token_counter` and `token_budget` are both provided, the chat agent
    gets a `history_processors` callable that summarises the oldest messages
    (via `summary_agent`) when the conversation exceeds the budget. Tests
    typically omit both — there's no need to exercise the compactor in
    smoke tests, and omitting them keeps the chat agent transparent.
    """
    summary_agent = Agent(
        model,
        name="summary_agent",
        output_type=str,
        instructions=_SUMMARY_INSTRUCTIONS,
        model_settings={"max_tokens": 512, "temperature": 0.3},
    )

    fact_checker_agent = Agent(
        model,
        name="fact_checker_agent",
        output_type=ToolOutput(FactCheckResult),
        instructions=_FACT_CHECKER_SYSTEM_PROMPT,
        # Tight budget — the agent only emits two short lists. The
        # 2048 budget the classifier uses is overkill here.
        model_settings={"max_tokens": 512, "temperature": 0.1},
    )

    history_processors: list = []
    if token_counter is not None and token_budget is not None:
        history_processors.append(
            make_history_compactor(
                summary_agent=summary_agent,
                token_counter=token_counter,
                token_budget=token_budget,
                keep_tail=history_keep_tail,
            )
        )

    chat_agent: Agent[ChatDeps, str] = Agent(
        model,
        name="chat_agent",
        deps_type=ChatDeps,
        output_type=str,
        instructions=_BASE_CHAT_INSTRUCTIONS,
        model_settings={"max_tokens": 800, "temperature": 0.7},
        history_processors=history_processors,
        # Allow one corrective shot when the fact-checker raises ModelRetry.
        # Exhaustion is caught at the render call site (`_render_via_chat_agent`)
        # so a fact-check miss never breaks a player's session.
        retries=2,
    )

    @chat_agent.instructions
    def _dossier_block(ctx: RunContext[ChatDeps]) -> str:
        return _format_dossier(ctx.deps.npc)

    @chat_agent.instructions
    def _plot_directive_block(ctx: RunContext[ChatDeps]) -> str:
        spec = ctx.deps.plot_directive
        if spec is None:
            return ""
        must_include = "\n".join(f"  - {f}" for f in spec.must_include_facts) or "  (none specified)"
        must_not = "\n".join(f"  - {f}" for f in spec.must_not) or "  (none specified)"
        return (
            f"PLOT DIRECTIVE (this turn only):\n"
            f"Intent: {spec.intent.strip()}\n"
            f"Must include:\n{must_include}\n"
            f"Must not:\n{must_not}"
        )

    @chat_agent.instructions
    def _commitments_block(ctx: RunContext[ChatDeps]) -> str:
        lines = format_commitments(ctx.deps.commitments)
        if not lines:
            return ""
        body = "\n".join(f"  {line}" for line in lines)
        return f"CURRENT COMMITMENTS:\n{body}"

    @chat_agent.output_validator
    async def _validate_against_directive(
        ctx: RunContext[ChatDeps], output: str
    ) -> str:
        spec = ctx.deps.plot_directive
        if spec is None or not spec.validate_facts:
            return output
        if not spec.must_include_facts and not spec.must_not:
            return output  # load-time validator should reject this; defensive
        result = await fact_checker_agent.run(
            build_fact_checker_user_prompt(
                output, spec.must_include_facts, spec.must_not
            )
        )
        fc = result.output
        if fc.missing_facts or fc.violations:
            logfire.info(
                "fact_checker flagged candidate body",
                missing_facts=fc.missing_facts,
                violations=fc.violations,
            )
            raise ModelRetry(_format_fact_retry_message(fc))
        return output

    classifier_agent = Agent(
        model,
        name="classifier_agent",
        output_type=ToolOutput(list[AxisResult]),
        instructions=_CLASSIFIER_SYSTEM_PROMPT,
        # Headroom: Gemma 4 E4B sometimes preambles before emitting the
        # tool call. At max_tokens=256 it regularly hit `finish_reasons=
        # ["length"]` on the first attempt, forcing Pydantic AI to retry
        # and ~2x the latency. 2048 leaves ample budget for the call plus
        # any preamble the model wants to write — and now scales over
        # multiple per-axis result entries in one tool call.
        model_settings={"max_tokens": 2048, "temperature": 0.1},
    )

    return AgentBundle(
        chat_agent=chat_agent,
        classifier_agent=classifier_agent,
        summary_agent=summary_agent,
        fact_checker_agent=fact_checker_agent,
    )


# ───────────────────────── history compaction ─────────────────────────


def _extract_text(messages: list[ModelMessage]) -> str:
    """Concatenate every string `content` across all parts of all messages.

    Pydantic AI's `ModelMessage` is a union (`ModelRequest` | `ModelResponse`)
    with heterogeneous parts (TextPart, UserPromptPart, ToolCallPart, ...).
    Our chat agent never registers tools, so in practice every part has
    string content. Non-string content is skipped — token counts will run
    slightly low, which is the safe direction (we keep messages we'd
    otherwise drop).
    """
    chunks: list[str] = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                chunks.append(content)
    return "\n".join(chunks)


def make_llama_token_counter(tokenize_url: str) -> TokenCounter:
    """Return a TokenCounter that hits llama.cpp's `/tokenize` endpoint.

    `/tokenize` is a llama-server-specific endpoint (NOT under `/v1/...`):
    POST `{"content": "..."}` returns `{"tokens": [...]}`. Sub-millisecond
    against the loaded model — we'd otherwise need tiktoken or HuggingFace
    tokenizers, neither of which knows about Gemma 4 E4B's exact vocabulary.
    """
    async def _count(messages: list[ModelMessage]) -> int:
        text = _extract_text(messages)
        if not text:
            return 0
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.post(tokenize_url, json={"content": text})
            r.raise_for_status()
            return len(r.json().get("tokens", []))
    return _count


def make_history_compactor(
    *,
    summary_agent: Agent,
    token_counter: TokenCounter,
    token_budget: int,
    keep_tail: int,
) -> Callable[[list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Build a Pydantic AI `history_processors` callable.

    Strategy: summarise-then-replace. On every chat agent run:
    1. Count tokens via `token_counter`.
    2. If under `token_budget`, return messages unchanged.
    3. Otherwise hand `messages[:-keep_tail]` to `summary_agent` and
       splice its `new_messages()` in front of the kept tail.

    The most recent `keep_tail` messages are always preserved verbatim so
    the model sees the immediate exchange without paraphrase loss.
    """
    async def compact(messages: list[ModelMessage]) -> list[ModelMessage]:
        if len(messages) <= keep_tail:
            return messages
        token_count = await token_counter(messages)
        if token_count <= token_budget:
            return messages
        oldest = list(messages[:-keep_tail])
        recent = list(messages[-keep_tail:])
        with logfire.span(
            "history_compact",
            token_count_before=token_count,
            token_budget=token_budget,
            messages_summarised=len(oldest),
            messages_kept_tail=len(recent),
        ):
            summary = await summary_agent.run(message_history=oldest)
            return list(summary.new_messages()) + recent
    return compact


# ───────────────────────── prompt formatting helpers ─────────────────────────


def _format_dossier(npc: NPCDossier) -> str:
    return (
        f"DOSSIER for {npc.display_name} ({npc.id}):\n"
        f"Relationship: {npc.relationship_to_owner}\n"
        f"Base voice: {npc.base_voice_register}\n"
        f"Forbidden:\n{_bullets(npc.base_voice_forbidden)}\n"
        f"Will say:\n{_bullets(npc.will_say)}\n"
        f"Will not say:\n{_bullets(npc.will_not_say)}"
    )


def _bullets(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items) or "  (none)"


def build_classifier_user_prompt(
    recipient_blurb: str,
    prior_email_body: str,
    reply_body: str,
    axes: list[ClassifyReply],
) -> str:
    """Build the classifier's user prompt for one or more axes.

    `recipient_blurb` and `prior_email_body` are shared across all axes by
    construction — there is no per-axis context selection in Stage 1. If a
    future axis needs to score against, say, the original welcome rather
    than the most recent prior, that's a follow-up.
    """
    axis_blocks: list[str] = []
    for i, axis in enumerate(axes, start=1):
        values = ", ".join(f'"{v}"' for v in axis.allowed_values)
        axis_blocks.append(
            f"AXIS {i} — quality: {axis.quality}\n"
            f"  Question: {axis.question.strip()}\n"
            f"  Allowed values: {values}"
        )
    axes_text = "\n\n".join(axis_blocks)
    return f"""RECIPIENT (who the player is writing to):
{recipient_blurb}

MOST RECENT PRIOR EMAIL (what the player is replying to):
{prior_email_body.strip() or '(no prior email)'}

PLAYER'S REPLY (just typed):
{reply_body}

CLASSIFICATION AXES (return one result per axis, in order):
{axes_text}

For each axis, return {{"quality": "<axis quality id>", "value": "<one allowed value, verbatim>", "rationale": "<one sentence>"}}."""


def build_fact_checker_user_prompt(
    body: str,
    must_include_facts: list[str],
    must_not: list[str],
) -> str:
    required = _bullets(must_include_facts)
    forbidden = _bullets(must_not)
    return f"""CANDIDATE BODY:
{body}

REQUIRED FACTS (each must be conveyed):
{required}

FORBIDDEN ITEMS (none may appear):
{forbidden}

Return missing_facts and violations."""


def _format_fact_retry_message(fc: FactCheckResult) -> str:
    """Concise corrective message handed back to the chat agent on retry.
    The agent already has the full directive in its instructions block."""
    parts: list[str] = []
    if fc.missing_facts:
        parts.append("Required facts missing from your draft:\n" + _bullets(fc.missing_facts))
    if fc.violations:
        parts.append("Forbidden items present in your draft:\n" + _bullets(fc.violations))
    parts.append(
        "Rewrite the body so every required fact is conveyed and every "
        "forbidden item is removed. Keep voice, length, and intent."
    )
    return "\n\n".join(parts)


def format_commitments(commitments: dict[str, tuple[float, str]]) -> list[str]:
    """Filter the commitments snapshot down to qualities whose current
    value differs from their default (or 0.0 if default is None) and
    format each as a single line with the `commitment.` prefix stripped.

    A commitment that has never been set is noise the chat agent has to
    ignore on every render — empty list is the right state until the
    bargain (or whatever else the scenario authors) actually commits.
    """
    lines: list[str] = []
    for full_id, (value, description) in commitments.items():
        if not full_id.startswith(COMMITMENT_PREFIX):
            continue
        short = full_id[len(COMMITMENT_PREFIX):]
        line = f"{short} = {value}"
        if description.strip():
            line += f"  # {description.strip()}"
        lines.append(line)
    return lines


def commitments_snapshot(
    qualities: dict[str, float],
    defaults: dict[str, float],
    descriptions: dict[str, str],
) -> dict[str, tuple[float, str]]:
    """Build the per-render commitments snapshot for ChatDeps. Filters to
    `commitment.*` ids whose current value differs from the authored
    default (or 0.0 if no default). Caller derives `defaults` and
    `descriptions` from `Scenario.qualities`.
    """
    out: dict[str, tuple[float, str]] = {}
    for full_id, value in qualities.items():
        if not full_id.startswith(COMMITMENT_PREFIX):
            continue
        default = defaults.get(full_id, 0.0)
        if value == default:
            continue
        out[full_id] = (value, descriptions.get(full_id, ""))
    return out
