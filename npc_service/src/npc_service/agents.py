"""The six LLM agents: CHAT_AGENT, CLASSIFIER_AGENT, EXTRACTOR_AGENT, SUMMARY_AGENT, FACT_CHECKER_AGENT, DIRECTOR_AGENT.

CHAT_AGENT renders one email body per turn — both ordinary chat turns and
the closing epilogue. It uses Pydantic AI's native `message_history` for the
per-thread conversation, plain-text `output_type=str` so the assistant turn
lands as a `TextPart` (clean for subsequent turns), and dynamic
`@instructions` keyed on `ChatDeps` to layer dossier, optional plot directive,
current commitments, player canon, and — when ChatDeps carries an
`epilogue_digest` — the closing-email framing plus the per-run digest. An
`@output_validator` engages ONLY when the firing beat sets
`spec.validate_facts=True`: it runs the fact-checker against the beat's
must_include_facts / must_not lists and
raises ModelRetry on any miss. Default-off keeps loop beats free of
validator latency.

CLASSIFIER_AGENT is one-shot, ToolOutput. It returns a list of per-quality
results (one entry per classify-typed `Effect` in scope on the player's
reply — the beat the player just replied to plus any scenario-scope
effects), each with a `quality`, `value`, `rationale`. The allowed-values
list per quality is enforced via the user prompt (not the schema) since
values vary per quality. The caller validates post-hoc per effect and
skips the quality write if a returned value is off-spec. Single batched
call regardless of effect count keeps per-send cost flat.

EXTRACTOR_AGENT runs in parallel with the classifier on every player /send
(via `asyncio.gather`, so it's latency-neutral). It harvests open-ended
player-introduced specifics — name, role, named people, places, claims
about self — into `state.player_canon` after a substring-anchor and
salience-threshold filter. Distinct from the classifier: closed-set verdict
on author-defined axes vs. open-ended fact harvesting. The chat agent reads
canon back through the `_player_canon_block` instructions decorator.

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

DIRECTOR_AGENT picks one beat per tick from the currently-eligible set. The
output schema is constructed PER CALL with `chosen_beat_id` constrained to
a closed Literal of the eligible ids plus "none". Inputs: a state digest
(qualities, recent classifier history, recent fire history) and the eligible
list. NOT raw chat history. Every selection carries a mandatory `rationale`
field — that is the audit trail for the non-determinism the Director adds.
Two short-circuits keep cost down: 0 eligible → return "none" without an
LLM call; 1 eligible → return that id without an LLM call.

Construction is deferred to `build_agents()` (called from `lifespan`) so the
shared OpenAI-compatible model handle can be passed in once.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx
import logfire
from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.messages import ModelMessage

from .models import (
    AxisResult,
    BeatSpec,
    CanonCategory,
    Effect,
    NPCDossier,
    PlayerCanonEntry,
)


# Top-K cap on the player_canon snapshot the chat agent sees per render.
# The full ledger remains in state.player_canon for predicates and the
# epilogue digest; this only bounds the chat-agent prompt block size.
PLAYER_CANON_SURFACE_LIMIT = 10
# Per-turn salience decay for snapshot ranking — older entries are weighted
# down by 0.85 ** turns_since_origin, so a recently-introduced fact beats
# an older one of the same salience.
PLAYER_CANON_RECENCY_DECAY = 0.85


# Type alias: a function that estimates the token cost of a list of messages.
# Production uses llama-server's `/tokenize`; tests inject a stub so neither
# the network nor the tokenizer is exercised.
TokenCounter = Callable[[list[ModelMessage]], Awaitable[int]]


# ───────────────────────── chat-agent deps ─────────────────────────


@dataclass
class ChatDeps:
    """Per-run injection for the chat agent.

    `plot_directive` is None for chat-turn fallbacks (free conversation in
    character) and is a `BeatSpec` when a beat is firing on this turn —
    in which case the directive is layered into the prompt as authored
    intent / must-include / must-not.

    `commitments` is a snapshot of qualities flagged `surface_to_chat: true`
    captured at render-call time, mapping the quality id to (current value,
    description). The decorator `_commitments_block` surfaces them on every
    render. Snapshotting in deps keeps deps immutable for the duration of
    one agent run, which matches the `plot_directive` pattern.

    `epilogue_digest`, when set, switches the chat agent into closing-email
    mode: the `_epilogue_framing_block` instructions decorator emits the
    closing-email rules and `_epilogue_digest_block` surfaces the run
    digest verbatim. Caller passes `message_history=None` for epilogue
    runs so the agent works off the digest, not the live thread history.
    """

    npc: NPCDossier
    plot_directive: BeatSpec | None
    commitments: dict[str, tuple[float | str, str]] = field(default_factory=dict)
    # Per-render top-K slice of state.player_canon (most relevant entries
    # by recency × salience). The decorator `_player_canon_block` surfaces
    # them as a prompt section so the chat agent can quote the player's
    # stated specifics back without reasoning over message history.
    player_canon: dict[str, PlayerCanonEntry] = field(default_factory=dict)
    epilogue_digest: str | None = None


# ───────────────────────── system prompts ─────────────────────────


_BASE_CHAT_INSTRUCTIONS = """You are writing a single email body in the voice of the NPC described below.

OUTPUT FORMAT (strict):
- Body text only. No subject line. No "From:" / "To:" / "Date:" headers.
- No sign-off / signature unless the dossier explicitly specifies one.
- No quoted prior message. No "Re:" prefix in the output.
- No JSON, no markdown code fences, no meta-commentary.
- Length: chat-message length, not memo length. Aim for 2-3 short sentences; rarely more than two short paragraphs. Cover the required facts and stop.

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


_EXTRACTOR_SYSTEM_PROMPT = """You read a single player-written email reply and extract any concrete specifics the player has introduced — facts the world should remember about them.

You will receive:
- The recipient's persona blurb (who the player is writing to).
- The most recent prior email in the thread (what the player is replying to).
- The player's just-typed reply (the new send).

Extract candidates fitting any of these categories:
- name: the player's preferred name (key it as exactly "name").
- role: the player's job, team, or professional position (key it as exactly "role").
- person: a third party the player mentions by name (a relative, colleague, friend).
- place: a named location the player mentions (a workplace, city, room).
- event: a specific event the player references (a meeting, an incident, a milestone).
- date: a specific date or time the player names.
- possession: a concrete object the player claims to own.
- claim_about_self: a specific claim the player makes about themselves (a hobby, an opinion held as identity, a habit).

For each candidate, return:
- key: short snake_case identifier. Use exactly "name" / "role" for those categories. For others, pick a descriptive snake_case key (e.g. "gretchen", "manager_dispute", "tuesday_meditation").
- summary: a one-line summary of the fact.
- literal_text_quote: an EXACT verbatim substring from the player's reply that anchors this fact. Required and load-bearing — if your quote is not a substring of the reply, the candidate will be dropped.
- salience: 0.0–1.0. High (≥0.7) for direct claims ("I'm Gretchen's cousin"). Medium (0.4–0.7) for incidental but specific mentions ("anyway, my brother in Brisbane says..."). Low (<0.4) for ambiguous, hedged, or speculative text.
- category: one of the categories above.

Do NOT invent facts. Do NOT extrapolate. If the reply contains no concrete specifics, return an empty candidates list. A polite "Hi, looking forward to it" should produce no candidates.

Return: {"candidates": [<one entry per concrete specific>, ...]}.
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


_DIRECTOR_SYSTEM_PROMPT = """You are the Director — you decide which authored beat fires next in an interactive scenario.

You will receive:
- A STATE DIGEST: current qualities, recent classifier verdicts on player replies (per axis, most-recent-first), and the recent fire history.
- An ELIGIBLE BEATS list: each entry has an id and a short intent summary describing what that beat does to the conversation.

Pick exactly one of these as `chosen_beat_id`:
- One of the listed beat ids — fires that beat this tick.
- "none" — skip the tick (no beat fires; the player will see no NPC reply this turn). Use sparingly, only when no listed beat is a fit for the current state.

Heuristics:
- Prefer beats that ADVANCE the arc over beats that REPEAT recent activity. If you fired a multi-turn loop on the previous tick and a fire-once plot beat is now eligible, prefer the plot beat.
- Read the state digest. A "creature_grievance high" with a "creator_affect cold" recent-window is asking for the cold/high-grievance branch — pick a beat that fits, not the warm one.
- Do not pick a terminal (one that ends the session) unless the digest clearly signals the run is settled into one of the four endings.
- If two beats look equally good, prefer the one that has not fired anything similar recently.

Provide a one-sentence rationale: which signal in the digest decided it.
"""


_EPILOGUE_FRAMING = """EPILOGUE MODE: This is the closing email of the scenario. The player has reached a terminal beat — this email IS the ending.

Tailor the closing email to THIS run. Reference the digest below — quote or echo a pivot reply, name a specific commitment, acknowledge a register the player held throughout — without reading like a template. Stay strictly inside the dossier voice and the directive's must-include / must-not lists.

Length: one short paragraph plus the sign-off. Two paragraphs only if the digest demands it. Land it; don't unfurl.
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


class CanonCandidate(BaseModel):
    """One extractor-produced canon candidate. Promoted to `state.player_canon`
    by `apply_canon_extraction` if salience ≥ threshold AND `literal_text_quote`
    is a verbatim substring of the player's reply.
    """

    key: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    summary: str = Field(min_length=1)
    literal_text_quote: str = Field(min_length=1)
    salience: float = Field(ge=0.0, le=1.0)
    category: CanonCategory


class ExtractorOutput(BaseModel):
    """Structured output of the extractor agent. An empty candidates list is
    a valid (and common) result — most replies contain no concrete specifics.
    """

    candidates: list[CanonCandidate] = Field(default_factory=list)


@dataclass
class AgentBundle:
    """Six agents wired up against one shared model handle."""

    chat_agent: Agent
    classifier_agent: Agent
    extractor_agent: Agent
    summary_agent: Agent
    fact_checker_agent: Agent
    director_agent: Agent


# ───────────────────────── director output (dynamic) ─────────────────────────


class DirectorChoiceFallback(BaseModel):
    """Static base output schema for the director — used as a fallback when
    the per-call dynamic Literal-bound model can't be constructed (e.g. tests
    that override the agent without per-call output_type). Real call sites
    pass `output_type=build_director_output_type([...])` for the closed enum.
    """

    chosen_beat_id: str
    rationale: str = Field(min_length=1)


def build_director_output_type(eligible_beat_ids: list[str]):
    """Construct a fresh Pydantic model whose `chosen_beat_id` is a closed
    Literal of the eligible ids plus the literal "none" (skip-tick).

    Used at the call site so the LLM cannot return a beat id that isn't
    eligible — the schema enforces it. Built per call because the eligible
    list changes every tick.
    """
    options = tuple(eligible_beat_ids) + ("none",)
    # Literal[tuple] is the runtime construction — accepted by Pydantic for
    # JSON-schema enum generation. The created model's __name__ stays
    # generic; we don't expose the type to user-facing surfaces.
    literal_t = Literal[options]  # type: ignore[valid-type]
    return create_model(
        "DirectorChoice",
        chosen_beat_id=(literal_t, ...),
        rationale=(str, Field(..., min_length=1)),
        __base__=BaseModel,
    )


def _attach_chat_deps_instructions(
    agent: Agent,
    *,
    fact_checker_agent: Agent,
) -> None:
    """Wire the dossier / directive / commitments / epilogue / fact-check
    decorators onto the chat agent. The epilogue decorators are no-ops
    when `ChatDeps.epilogue_digest is None` (the regular chat-turn case)."""

    @agent.instructions
    def _dossier_block(ctx: RunContext[ChatDeps]) -> str:
        return _format_dossier(ctx.deps.npc)

    @agent.instructions
    def _plot_directive_block(ctx: RunContext[ChatDeps]) -> str:
        spec = ctx.deps.plot_directive
        if spec is None:
            return ""
        directive_label = (
            "this terminal" if ctx.deps.epilogue_digest is not None else "this turn only"
        )
        must_include = "\n".join(f"  - {f}" for f in spec.must_include_facts) or "  (none specified)"
        must_not = "\n".join(f"  - {f}" for f in spec.must_not) or "  (none specified)"
        return (
            f"PLOT DIRECTIVE ({directive_label}):\n"
            f"Intent: {spec.intent.strip()}\n"
            f"Must include:\n{must_include}\n"
            f"Must not:\n{must_not}"
        )

    @agent.instructions
    def _commitments_block(ctx: RunContext[ChatDeps]) -> str:
        lines = format_commitments(ctx.deps.commitments)
        if not lines:
            return ""
        body = "\n".join(f"  {line}" for line in lines)
        return f"CURRENT COMMITMENTS:\n{body}"

    @agent.instructions
    def _player_canon_block(ctx: RunContext[ChatDeps]) -> str:
        lines = format_player_canon(ctx.deps.player_canon)
        if not lines:
            return ""
        body = "\n".join(f"  {line}" for line in lines)
        return f"PLAYER CANON (specifics the player has introduced — quote the literal_text where natural):\n{body}"

    @agent.instructions
    def _epilogue_framing_block(ctx: RunContext[ChatDeps]) -> str:
        if ctx.deps.epilogue_digest is None:
            return ""
        return _EPILOGUE_FRAMING

    @agent.instructions
    def _epilogue_digest_block(ctx: RunContext[ChatDeps]) -> str:
        if ctx.deps.epilogue_digest is None:
            return ""
        return ctx.deps.epilogue_digest

    @agent.output_validator
    async def _validate_against_directive(
        ctx: RunContext[ChatDeps], output: str
    ) -> str:
        spec = ctx.deps.plot_directive
        if spec is None or not spec.validate_facts:
            return output
        if not spec.must_include_facts and not spec.must_not:
            return output  # load-time validator should reject this; defensive
        # `ctx.retries` is a `dict[str, int]` in current Pydantic AI (per-tool
        # retry counters). Sum the values for an overall attempt count and
        # 1-index for human-readable logs ("attempt 1" reads better than
        # "attempt 0"). Both pass and fail log it so "succeeded after 2
        # retries" is as visible as "still flagged on attempt 1."
        retries = ctx.retries
        prior = sum(retries.values()) if isinstance(retries, dict) else int(retries or 0)
        attempt = prior + 1
        result = await fact_checker_agent.run(
            build_fact_checker_user_prompt(
                output, spec.must_include_facts, spec.must_not
            )
        )
        fc = result.output
        validator_label = (
            "epilogue body" if ctx.deps.epilogue_digest is not None else "candidate body"
        )
        if fc.missing_facts or fc.violations:
            logfire.info(
                f"fact_checker flagged {validator_label} on attempt {{attempt}}",
                attempt=attempt,
                missing_facts=fc.missing_facts,
                violations=fc.violations,
            )
            raise ModelRetry(_format_fact_retry_message(fc))
        logfire.info(
            f"fact_checker passed {validator_label} on attempt {{attempt}}",
            attempt=attempt,
        )
        return output


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
        model_settings={"max_tokens": 512, "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 64}},
    )

    fact_checker_agent = Agent(
        model,
        name="fact_checker_agent",
        output_type=NativeOutput(FactCheckResult),
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
        model_settings={"max_tokens": 800, "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 64}},
        history_processors=history_processors,
        # Allow one corrective shot when the fact-checker raises ModelRetry.
        # Exhaustion is caught at the render call site (`_render_via_chat_agent`)
        # so a fact-check miss never breaks a player's session.
        retries=2,
    )

    _attach_chat_deps_instructions(
        chat_agent,
        fact_checker_agent=fact_checker_agent,
    )

    classifier_agent = Agent(
        model,
        name="classifier_agent",
        output_type=NativeOutput(list[AxisResult]),
        instructions=_CLASSIFIER_SYSTEM_PROMPT,
        # Headroom: Gemma 4 E4B sometimes preambles before emitting the
        # tool call. At max_tokens=256 it regularly hit `finish_reasons=
        # ["length"]` on the first attempt, forcing Pydantic AI to retry
        # and ~2x the latency. 2048 leaves ample budget for the call plus
        # any preamble the model wants to write — and now scales over
        # multiple per-axis result entries in one tool call.
        model_settings={"max_tokens": 2048, "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 64}},
    )

    # Lower temperature than the classifier — extraction is fact-shaped,
    # not vibes-shaped. 1024 token budget covers preamble + several
    # candidate entries.
    extractor_agent = Agent(
        model,
        name="extractor_agent",
        output_type=NativeOutput(ExtractorOutput),
        instructions=_EXTRACTOR_SYSTEM_PROMPT,
        model_settings={"max_tokens": 1024, "temperature": 0.3, "top_p": 0.95, "extra_body": {"top_k": 64}},
    )

    # Director defaults to the static fallback output_type; the call site
    # overrides per run with `output_type=NativeOutput(build_director_output_type(ids))`.
    # Lower temperature than the chat agent — selection is a judgement, not
    # a creative exercise; we want consistency on identical digests.
    director_agent = Agent(
        model,
        name="director_agent",
        output_type=NativeOutput(DirectorChoiceFallback),
        instructions=_DIRECTOR_SYSTEM_PROMPT,
        model_settings={"max_tokens": 512, "temperature": 0.3},
    )

    return AgentBundle(
        chat_agent=chat_agent,
        classifier_agent=classifier_agent,
        extractor_agent=extractor_agent,
        summary_agent=summary_agent,
        fact_checker_agent=fact_checker_agent,
        director_agent=director_agent,
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
    effects: list[Effect],
) -> str:
    """Build the classifier's user prompt for one or more classify-typed
    effects. All effects in `effects` must have `effect.is_classify` true.

    `recipient_blurb` and `prior_email_body` are shared across all effects
    by construction — there is no per-effect context selection in Stage 1.
    If a future effect needs to score against, say, the original welcome
    rather than the most recent prior, that's a follow-up.
    """
    axis_blocks: list[str] = []
    for i, effect in enumerate(effects, start=1):
        directive = effect.classify
        if directive is None:
            continue
        values = ", ".join(f'"{v}"' for v in directive.allowed_values)
        axis_blocks.append(
            f"AXIS {i} — quality: {effect.quality}\n"
            f"  Question: {directive.question.strip()}\n"
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


def build_extractor_user_prompt(
    recipient_blurb: str,
    prior_email_body: str,
    reply_body: str,
) -> str:
    """Build the extractor's user prompt. Same input shape as the classifier
    by design — the two agents run on identical inputs in parallel via
    `asyncio.gather` in the /send orchestration."""
    return f"""RECIPIENT (who the player is writing to):
{recipient_blurb}

MOST RECENT PRIOR EMAIL (what the player is replying to):
{prior_email_body.strip() or '(no prior email)'}

PLAYER'S REPLY (just typed):
{reply_body}

Extract every concrete specific the player has introduced. Each candidate must include a literal_text_quote that is a verbatim substring of the reply. If the reply contains no concrete specifics, return an empty candidates list."""


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


def format_commitments(
    commitments: dict[str, tuple[float | str, str]]
) -> list[str]:
    """Format the commitments snapshot — one line per surfaced quality
    whose current value differs from its default. Caller is responsible
    for the snapshot's filter (qualities flagged `surface_to_chat`); this
    just renders.

    A commitment that has never been set is noise the chat agent has to
    ignore on every render — empty list is the right state until the
    bargain (or whatever else the scenario authors) actually commits.
    """
    lines: list[str] = []
    for quality_id, (value, description) in commitments.items():
        if isinstance(value, str):
            line = f"{quality_id} = {value!r}"
        else:
            line = f"{quality_id} = {value}"
        if description.strip():
            line += f"  # {description.strip()}"
        lines.append(line)
    return lines


def commitments_snapshot(
    qualities: dict[str, float | str],
    defaults: dict[str, float | str],
    descriptions: dict[str, str],
    surfaced_ids: set[str],
) -> dict[str, tuple[float | str, str]]:
    """Build the per-render commitments snapshot for ChatDeps.

    Includes only qualities whose id is in `surfaced_ids` (i.e. authored
    `surface_to_chat: true`) AND whose current value differs from its
    default. Caller derives `defaults`, `descriptions`, and `surfaced_ids`
    from `Scenario.quality_defaults`, `quality_descriptions`, and
    `surface_to_chat_quality_ids` respectively.
    """
    out: dict[str, tuple[float | str, str]] = {}
    for quality_id, value in qualities.items():
        if quality_id not in surfaced_ids:
            continue
        default = defaults.get(quality_id, 0.0 if isinstance(value, (int, float)) else "")
        if value == default:
            continue
        out[quality_id] = (value, descriptions.get(quality_id, ""))
    return out


def format_player_canon(canon: dict[str, PlayerCanonEntry]) -> list[str]:
    """Render a player_canon snapshot as one line per entry. Caller is
    responsible for the snapshot's bounded size; this just formats.

    Lines are ordered by descending `recency_score × salience` — the same
    ranking `player_canon_snapshot` applies — so the most-relevant entry
    appears first under the prompt header.
    """
    def _rank(item: tuple[str, PlayerCanonEntry]) -> float:
        # Recency_score = decay ** turns_since_origin; turns_since_origin is
        # encoded into the entry by the snapshot-builder, so here we just
        # rank by salience (snapshot already applied recency).
        return -item[1].salience
    lines: list[str] = []
    for key, entry in sorted(canon.items(), key=_rank):
        line = f'{key}: "{entry.summary}" — quoted: "{entry.literal_text_quote}"'
        lines.append(line)
    return lines


def player_canon_snapshot(
    canon: dict[str, PlayerCanonEntry],
    *,
    current_turn: int,
    limit: int = PLAYER_CANON_SURFACE_LIMIT,
    decay: float = PLAYER_CANON_RECENCY_DECAY,
) -> dict[str, PlayerCanonEntry]:
    """Build the per-render player_canon snapshot for ChatDeps.

    Returns at most `limit` entries from the full ledger, ranked by
    `(decay ** turns_since_origin) × salience` and ordered descending.
    The full `state.player_canon` remains intact for predicate evaluation
    and the epilogue digest; only the chat-agent-facing prompt block is
    bounded.

    `current_turn` should be the count of player replies so far (after
    appending the just-received reply, if applicable).
    """
    if not canon:
        return {}

    def _rank(item: tuple[str, PlayerCanonEntry]) -> float:
        _, entry = item
        age = max(0, current_turn - entry.turn_of_origin)
        return (decay ** age) * entry.salience

    ranked = sorted(canon.items(), key=_rank, reverse=True)[:limit]
    return dict(ranked)


# ───────────────────────── director helpers ─────────────────────────


def build_director_user_prompt(
    *,
    qualities: dict[str, float | str],
    recent_classifier_history: dict[str, list[str]],
    recent_fire_history: list[str],
    eligible: list[dict],
) -> str:
    """Build the per-tick prompt for the Director.

    `recent_classifier_history` is `{quality: [most-recent-first values]}`.
    `recent_fire_history` is a list of beat ids in fire order, most recent last.
    `eligible` is `[{"id": str, "intent_summary": str}]` plus we always offer
    "none" as an implicit option (declared in the system prompt).
    """
    qualities_text = ", ".join(
        f"{k}={v!r}" if isinstance(v, str) else f"{k}={v:g}"
        for k, v in sorted(qualities.items())
    ) or "(none)"
    history_lines: list[str] = []
    for axis, values in recent_classifier_history.items():
        if not values:
            continue
        history_lines.append(f"  {axis}: {', '.join(values)}")
    history_block = "\n".join(history_lines) or "  (no replies classified yet)"
    fired_block = ", ".join(recent_fire_history) if recent_fire_history else "(none yet)"
    eligible_block = "\n".join(
        f"  - {e['id']}: {e['intent_summary']}" for e in eligible
    ) or "  (no beats eligible — pick \"none\")"

    return f"""STATE DIGEST:
Qualities: {qualities_text}
Recent classifier verdicts (most recent first, per axis):
{history_block}
Recent fire history (oldest → newest): {fired_block}

ELIGIBLE BEATS:
{eligible_block}
  - none: skip this tick

Pick one. Provide a one-sentence rationale identifying which signal in the digest decided it."""


def build_recent_classifier_history(
    replies,
    *,
    axes: list[str] | None = None,
    per_axis: int = 5,
) -> dict[str, list[str]]:
    """Walk replies in reverse and collect, per axis, the last `per_axis`
    classifier values seen. If `axes` is None, every axis encountered shows up.

    `replies` is duck-typed as an iterable of objects with a
    `.classifier_results` attribute (list of `AxisResult`).
    """
    out: dict[str, list[str]] = {}
    for reply in reversed(list(replies)):
        for axis in reply.classifier_results:
            if axes is not None and axis.quality not in axes:
                continue
            bucket = out.setdefault(axis.quality, [])
            if len(bucket) >= per_axis:
                continue
            bucket.append(axis.value)
    return out
