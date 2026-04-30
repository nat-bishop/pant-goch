from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import logfire
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from pydantic_ai import Agent, PromptedOutput, ToolOutput
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .launcher import (
    LlamaStartupError,
    resolve_config,
    start_llama,
    stop_llama,
    wait_healthy,
)
from .models import (
    ActionRequest,
    Captor,
    CaptorResponse,
    GameState,
    HostageDef,
    OptionsResponse,
    Scenario,
    TimelineEvent,
    load_scenario,
)


_BOOT_SETTINGS = Settings()
logging.basicConfig(
    level=_BOOT_SETTINGS.npc_log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# httpx logs every request at INFO; at DEBUG it spams every health probe and
# chat completion. Pin it to WARNING so NPC_LOG_LEVEL=debug stays readable.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("npc_service")

# Logfire: load the write token from the credentials file next to this module
# rather than relying on CWD-based discovery. Godot spawns the sidecar with
# cwd=project root (see SidecarHost.cs), not npc_service/, so the default
# `.logfire/` lookup misses the file. send_to_logfire='if-token-present' keeps
# the SDK silent when neither the file nor LOGFIRE_TOKEN exists (fresh checkouts,
# CI).
# src layout: src/npc_service/main.py → npc_service project root is parents[2],
# where .logfire/logfire_credentials.json lives.
_logfire_creds = Path(__file__).resolve().parents[2] / ".logfire" / "logfire_credentials.json"
_logfire_token = os.getenv("LOGFIRE_TOKEN")
if not _logfire_token and _logfire_creds.is_file():
    with contextlib.suppress(Exception):
        _logfire_token = json.loads(_logfire_creds.read_text()).get("token")
logfire.configure(
    service_name="npc_service",
    send_to_logfire="if-token-present",
    token=_logfire_token,
    console=False,
)
logfire.instrument_pydantic_ai()
# httpx: capture every call to llama-server as a child span with full request /
# response bodies. Surfaces LLM round-trip latency, status, and timeouts plus the
# exact OpenAI tool-call payload — high signal for diagnosing the cold-load /
# stream-stall / context-exhaustion / tool-call-shape failure modes called out in
# CLAUDE.md. capture_all=True is verbose; gated implicitly by send_to_logfire.
logfire.instrument_httpx(capture_all=True)
# Bridge stdlib logging (`log.info(...)` throughout this file) into Logfire as
# structured events so meter mutations, hostage-status changes, and timeline
# events appear inline with spans without rewriting the call sites. Attached as
# a second handler so the existing stderr StreamHandler (from basicConfig) still
# prints locally.
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())

# SSE keepalive: send a comment ":" line every N seconds when /events queue is
# idle, so proxies and HttpClient instances don't drop the connection.
EVENTS_KEEPALIVE_SECONDS = 15.0
# How long to wait for llama-server when LAUNCH_LLAMA=false (just a probe — we
# proceed even if it fails so the dev can start it after the sidecar).
EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS = 2.0
# History compaction tunables. Defaults come from Settings; tests still
# monkeypatch these module attributes directly.
SUMMARY_TRIGGER_FRAC = _BOOT_SETTINGS.summary_trigger_frac
# 0 → computed in lifespan as SUMMARY_TRIGGER_FRAC * LLAMA_CTX.
SUMMARY_TRIGGER_TOKENS: int = _BOOT_SETTINGS.summary_trigger_tokens
# Mirrors launcher's LLAMA_CTX so log lines can show pct-of-max without
# re-reading config. Set in lifespan once cfg.ctx is known.
LLAMA_CTX_TOKENS: int = 0
SUMMARY_KEEP_TAIL = _BOOT_SETTINGS.summary_keep_tail
ACTION_STREAM_TIMEOUT_SECONDS = _BOOT_SETTINGS.action_stream_timeout_seconds
OPTIONS_TIMEOUT_SECONDS = _BOOT_SETTINGS.options_timeout_seconds


# ───────────────────────── globals (set in lifespan) ─────────────────────────

SCENARIO: Scenario | None = None
GAME_STATE: GameState | None = None
HISTORY: list = []  # pydantic_ai message history
CAPTOR_AGENT: Agent | None = None
OPTIONS_AGENT: Agent | None = None
SUMMARIZER_AGENT: Agent | None = None
LLAMA_PROC: subprocess.Popen | None = None
EVENT_BUS: "EventBus | None" = None
ACTION_LOCK = asyncio.Lock()  # serialise /action and /options so they can't race


# ───────────────────────── event bus (for /events SSE) ─────────────────────────


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, payload: dict) -> None:
        async with self._lock:
            stale = []
            for q in self._subscribers:
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    stale.append(q)
            for q in stale:
                self._subscribers.discard(q)


# ───────────────────────── system prompt ─────────────────────────


def build_system_prompt(captor: Captor, hostages: list[HostageDef]) -> str:
    facts = "\n".join(f"- {f}" for f in captor.known_facts)
    demands = "\n".join(f"- {d}" for d in captor.initial_demands)
    hostage_list = ", ".join(f"{h.name} ({h.description})" for h in hostages)

    examples = ""
    if captor.seed_exchanges:
        ex = captor.seed_exchanges[0]
        example_response = CaptorResponse(
            response_text=ex.assistant,
            stress_delta=-2,
            trust_delta=1,
        )
        examples = (
            f"\nExample. Negotiator says: {ex.user}\n"
            f"Your structured response:\n{example_response.model_dump_json()}\n"
        )

    return f"""You are {captor.name}.

{captor.persona}

You are holding two hostages: {hostage_list}.

You know these things to be true:
{facts}

Your demands:
{demands}

Stay in character. Speak only from the knowledge above — if asked about anything \
outside it, refuse in character ({captor.deflection_style}). Never invent names, \
places, times, or events that are not in your knowledge.

Each turn the negotiator speaks to you. Read the tone and substance of what they \
say and respond with these fields:

- response_text: 1-3 sentences IN CHARACTER, replying to the negotiator's line.
- stress_delta, trust_delta, desperation_delta: integers in -20..+20 reflecting how \
their words affected you. Genuine empathy and reasonable offers usually lower stress \
and raise trust. Refusing your demands raises stress and desperation. Threats spike \
stress and desperation, lower trust.
- action: one of [null, "release_hostage", "harm_hostage", "surrender"]. null is the \
default. Only set non-null when the situation truly warrants it given your current \
state. Releasing a hostage requires significant trust. Surrender requires very high \
trust and a path out (medical attention, safe passage). You will hurt a hostage if \
the negotiator threatens you or if you feel cornered with no way out.
- action_target: name of the hostage if action requires one ("Janet" or "David"), \
otherwise null.{examples}"""


def build_options_system_prompt(scenario: Scenario) -> str:
    """System prompt for OPTIONS_AGENT.

    Mirrors /scenario's public view: setup + captor name/persona/initial_demands
    + hostage names. Deliberately omits known_facts, seed_exchanges, and
    starting_meters — the options agent reasons from player-knowable state only,
    preserving the "meters never leave the server" invariant. Two example sets
    in-prompt anchor the format and the diversity-of-strategy expectation.
    """
    captor = scenario.captor
    hostage_list = ", ".join(f"{h.name} ({h.description})" for h in scenario.hostages)
    demands = "\n".join(f"- {d}" for d in captor.initial_demands)

    return f"""You are a dialogue coach helping a hostage negotiator pick what to say next.

Scenario: {scenario.setup}

The negotiator is talking to {captor.name}: {captor.persona}

Hostages: {hostage_list}

Stated demands from the captor so far:
{demands}

Each call you produce 2-5 dialogue options for the negotiator's next line. Each option has:
- tag: a one or two word label naming the *intent or emotion* the line conveys (e.g. "Empathize", "Threaten", "Stall", "Bargain", "Probe", "Reassure", "Concede", "Challenge", "Pivot", "Press"). Tags MUST be distinct within a set — never two options with the same tag.
- hidden_line: the actual sentence the negotiator would speak (1-3 sentences). The PLAYER will not see this until they pick it; they only see the tag.

Options must represent meaningfully different *strategies*, not slight rewordings. The negotiator's goal is to de-escalate, build trust, and persuade the captor toward release or surrender — but options should include risky and confrontational moves too, not only safe ones, so the player has a real choice.

Format examples (do not literally reuse these lines):

Opening situation, no exchange yet:
{{"options":[
  {{"tag":"Introduce","hidden_line":"This is Officer Reeves with crisis negotiations. I'm here to talk, that's it. What should I call you?"}},
  {{"tag":"Empathize","hidden_line":"Hey — I know tonight's gone sideways. Talk to me. What's the most important thing right now?"}},
  {{"tag":"Probe","hidden_line":"Before anything else, are you and the people in there okay? Anyone hurt?"}}
]}}

Mid-conversation, captor just refused something:
{{"options":[
  {{"tag":"Concede","hidden_line":"Okay. I hear you. Forget that for a minute. Tell me what would actually help right now."}},
  {{"tag":"Press","hidden_line":"I get the no. But you understand what 'no' costs you here, right?"}},
  {{"tag":"Pivot","hidden_line":"Different question, then. Is there anyone we can call for you? Family, anybody?"}},
  {{"tag":"Stall","hidden_line":"Give me a minute to talk to my people about that. I want to give you a real answer, not a fast one."}}
]}}
"""


# ───────────────────────── state mutation ─────────────────────────


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def apply_response(state: GameState, r: CaptorResponse) -> None:
    """Atomic mutation of game state from a fully-validated captor response."""
    state.stress = _clamp(state.stress + r.stress_delta, 0, 100)
    state.trust = _clamp(state.trust + r.trust_delta, 0, 100)
    state.desperation = _clamp(state.desperation + r.desperation_delta, 0, 100)

    if r.action == "release_hostage":
        h = state.hostage_by_name(r.action_target)
        if h and h.status == "held":
            h.status = "released"
            log.info("hostage released: %s", h.name)
    elif r.action == "harm_hostage":
        h = state.hostage_by_name(r.action_target)
        if h and h.status == "held":
            h.status = "harmed"
            log.info("hostage harmed: %s", h.name)
    elif r.action == "surrender":
        state.captor_outcome = "surrendered"
        state.ended = True
        state.end_reason = "captor_surrendered"
        log.info("captor surrendered")

    check_end_conditions(state)
    log.info(
        "meters: stress=%d trust=%d desperation=%d hostages=%s ended=%s",
        state.stress,
        state.trust,
        state.desperation,
        [(h.name, h.status) for h in state.hostages],
        state.ended,
    )


def check_end_conditions(state: GameState) -> None:
    if state.ended:
        return
    if state.desperation > 90:
        state.ended = True
        state.captor_outcome = "breached"
        state.end_reason = "auto_breach"
        return
    if any(h.status == "harmed" for h in state.hostages):
        state.ended = True
        state.captor_outcome = "harmed"
        state.end_reason = "hostage_harmed"
        return
    if all(h.status != "held" for h in state.hostages):
        state.ended = True
        state.captor_outcome = state.captor_outcome or "resolved"
        state.end_reason = "all_released"
        return
    if state.action_count >= state.max_actions:
        state.ended = True
        state.captor_outcome = state.captor_outcome or "breached"
        state.end_reason = "budget_exhausted"


# ───────────────────────── timeline events (action-counted, fired inline) ─────────────────────────


def synthetic_event_message(evt: TimelineEvent) -> ModelRequest:
    """Inject a scene-update as a user-role message the captor sees on their next turn."""
    return ModelRequest(parts=[UserPromptPart(content=f"[scene update: {evt.injection_text}]")])


def fire_due_timeline_events(state: GameState, scenario: Scenario) -> list[TimelineEvent]:
    """Fire any timeline events whose threshold has been crossed and which haven't
    fired yet. Appends synthetic scene-update messages to HISTORY so the captor
    sees them on its next turn. Returns the fired list so the SSE stream can
    broadcast them too. Idempotent — events fire at most once per game.
    """
    global HISTORY
    due = [
        e for e in scenario.timeline
        if e.after_actions <= state.action_count
        and e.id not in state.fired_timeline_event_ids
    ]
    for evt in due:
        log.info(
            "timeline event firing: id=%s after_actions=%d (action_count=%d)",
            evt.id, evt.after_actions, state.action_count,
        )
        HISTORY.append(synthetic_event_message(evt))
        state.fired_timeline_event_ids.add(evt.id)
    return due


def _timeline_event_payload(evt: TimelineEvent) -> dict:
    return {
        "id": evt.id,
        "after_actions": evt.after_actions,
        "kind": evt.kind,
        "notification_text": evt.notification_text,
    }


# ───────────────────────── history processors (pydantic-ai history_processors=) ─────────────────────────


def _approx_tokens(messages: list[ModelMessage]) -> int:
    """Approximate prompt-token count of `messages` if sent as message_history.

    Prefer the most recent ModelResponse.usage from the history — its
    input_tokens is the exact tokenized prefix length the model just saw, plus
    its own output_tokens completes the run. For any messages added after that
    response (e.g., synthetic timeline scene-updates injected directly into
    HISTORY) fall back to chars/4. Returns chars/4 over the whole list when no
    usage is available (e.g., test fixtures, freshly seeded history).
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ModelResponse) and msg.usage is not None:
            tail_chars = sum(len(str(m)) for m in messages[i + 1:])
            return msg.usage.input_tokens + msg.usage.output_tokens + tail_chars // 4
    return sum(len(str(m)) for m in messages) // 4


def _pct_of_ctx(tokens: int) -> str:
    """Format `tokens` as 'N (P% of ctx)' for log lines, when ctx is known."""
    if LLAMA_CTX_TOKENS <= 0:
        return str(tokens)
    return f"{tokens} ({100 * tokens // LLAMA_CTX_TOKENS}% of {LLAMA_CTX_TOKENS})"


async def summarize_when_large(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Fold the oldest messages into a single recap when history grows too big."""
    if SUMMARIZER_AGENT is None:
        return messages
    approx = _approx_tokens(messages)
    if approx < SUMMARY_TRIGGER_TOKENS:
        return messages
    if len(messages) <= SUMMARY_KEEP_TAIL:
        return messages
    head = messages[:-SUMMARY_KEEP_TAIL]
    tail = messages[-SUMMARY_KEEP_TAIL:]
    if len(head) < 6:
        return messages

    log.info(
        "history compaction: firing — input=%s threshold=%s folding %d msgs into recap, keeping %d verbatim",
        _pct_of_ctx(approx), _pct_of_ctx(SUMMARY_TRIGGER_TOKENS), len(head), len(tail),
    )
    t0 = time.monotonic()
    summary = await SUMMARIZER_AGENT.run(
        "Produce the recap now.", message_history=head
    )
    summary_usage = summary.usage()
    recap_msg = ModelRequest(parts=[
        UserPromptPart(content=f"[recap of earlier negotiation]\n{summary.output}")
    ])
    out = [recap_msg, *tail]
    log.info(
        "history compaction: recap done in %.2fs — summarizer input_tokens=%d output_tokens=%d "
        "recap_chars=%d post_compaction=%s (was %s)",
        time.monotonic() - t0,
        summary_usage.input_tokens, summary_usage.output_tokens,
        len(summary.output),
        _pct_of_ctx(_approx_tokens(out)), _pct_of_ctx(approx),
    )
    return out


# ───────────────────────── seed history (empty for structured-output safety) ─────────────────────────


def seed_history(captor: Captor) -> list:
    """Return an empty history.

    With output_type=CaptorResponse, seeding the history with a plain TextPart
    response would teach the model to emit text instead of JSON. We embed
    seed_exchanges as in-prompt examples (see build_system_prompt) instead.
    """
    return []


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global SCENARIO, GAME_STATE, HISTORY, CAPTOR_AGENT, OPTIONS_AGENT
    global SUMMARIZER_AGENT, LLAMA_PROC, EVENT_BUS
    global SUMMARY_TRIGGER_TOKENS, LLAMA_CTX_TOKENS

    settings = Settings()

    # scenario.json lives at the npc_service project root (parents[2] from this
    # module under src/npc_service/main.py), not next to the source.
    scenario_path = settings.scenario_path or Path(__file__).resolve().parents[2] / "scenario.json"
    log.info("loading scenario: %s", scenario_path)
    SCENARIO = load_scenario(scenario_path)

    cfg = resolve_config(settings)
    if settings.launch_llama:
        log.info("starting llama-server (LAUNCH_LLAMA=true)")
        try:
            LLAMA_PROC = start_llama(cfg)
            await wait_healthy(host=cfg.host, port=cfg.port)
        except LlamaStartupError:
            log.exception("llama-server failed to come up")
            stop_llama(LLAMA_PROC)
            raise
    else:
        log.info("LAUNCH_LLAMA=false — assuming llama-server is managed externally")
        # Quick non-blocking probe; warn but proceed even if unreachable.
        try:
            await wait_healthy(host=cfg.host, port=cfg.port, timeout=EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS)
        except LlamaStartupError as e:
            log.warning("llama-server not reachable yet: %s — sidecar will boot anyway", e)

    LLAMA_CTX_TOKENS = cfg.ctx
    if SUMMARY_TRIGGER_TOKENS <= 0:
        SUMMARY_TRIGGER_TOKENS = int(LLAMA_CTX_TOKENS * SUMMARY_TRIGGER_FRAC)
    log.info(
        "history compaction config: ctx=%d trigger=%d (%.0f%% of ctx) keep_tail=%d",
        LLAMA_CTX_TOKENS, SUMMARY_TRIGGER_TOKENS,
        100 * SUMMARY_TRIGGER_TOKENS / LLAMA_CTX_TOKENS if LLAMA_CTX_TOKENS else 0,
        SUMMARY_KEEP_TAIL,
    )

    base_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
    model_name = "gemma-4-e4b"
    log.info("building Pydantic AI agent: model=%s base_url=%s", model_name, base_url)
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key="not-needed"),
    )
    # ToolOutput requires the model to emit OpenAI-format tool_calls; Pydantic
    # AI parses the args into CaptorResponse. With current llama.cpp this is
    # reliable on Gemma 4 and produces bounded, in-character turns. NativeOutput
    # (response_format=json_schema, GBNF-enforced at the sampler) was previously
    # shipped here, but with an unbounded response_text field the model filled
    # it indefinitely, cascading turn-over-turn into 19K-token generations.
    # The model_settings max_tokens cap is a safety net for occasional non-
    # tool-call prefix prose that Pydantic AI silently discards.
    # name= surfaces in Logfire spans so each agent's runs are filterable —
    # captor turns vs. options picks vs. summarization passes show up distinctly.
    SUMMARIZER_AGENT = Agent(
        model,
        name="summarizer",
        instructions=(
            "Summarize the negotiation so far in 4-6 short bullets. "
            "Cover: emotional state, demands raised, offers made, hostage status, "
            "and any commitments. Plain text, no preamble."
        ),
    )
    CAPTOR_AGENT = Agent(
        model,
        name="captor",
        output_type=ToolOutput(CaptorResponse),
        instructions=build_system_prompt(SCENARIO.captor, SCENARIO.hostages),
        history_processors=[summarize_when_large],
        model_settings={"max_tokens": 512},
    )
    # OPTIONS_AGENT: separate agent generating player dialogue picks each turn.
    # Reuses the same model and history processors as the captor so both agents
    # see a consistent compacted view of HISTORY. max_tokens=1536 covers the
    # worst case (5 options × ~400-char hidden_line + 24-char tags + JSON +
    # tool-call envelope ≈ 1100-1400 tokens) with margin so the last option
    # never truncates into invalid JSON.
    # PromptedOutput (NOT ToolOutput): llama.cpp's peg-gemma4 tool-call grammar
    # generator emits a malformed `thought ::=` rule (duplicated, second copy with
    # LHS stripped) on b8953, crashing GBNF install on `/options`'s schema.
    # PromptedOutput sends no `tools=[...]` so peg-gemma4 isn't activated at all;
    # the schema lives in the system prompt and Pydantic AI parses the JSON
    # response client-side. output_retries=2 covers the rare bad-JSON turn.
    # Captor keeps ToolOutput — its flat schema doesn't trip the bug.
    OPTIONS_AGENT = Agent(
        model,
        name="options",
        output_type=PromptedOutput(OptionsResponse),
        instructions=build_options_system_prompt(SCENARIO),
        history_processors=[summarize_when_large],
        model_settings={"max_tokens": 1536},
        output_retries=2,
    )

    HISTORY = seed_history(SCENARIO.captor)
    GAME_STATE = GameState.from_scenario(SCENARIO)
    EVENT_BUS = EventBus()

    log.info("npc_service ready")
    try:
        yield
    finally:
        log.info("shutting down")
        stop_llama(LLAMA_PROC)


app = FastAPI(lifespan=lifespan, title="npc_service")
logfire.instrument_fastapi(app, capture_headers=False)


# ───────────────────────── endpoints ─────────────────────────


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "scenario_loaded": SCENARIO is not None,
        "agent_ready": CAPTOR_AGENT is not None,
    }


@app.get("/scenario")
async def get_scenario() -> dict:
    if SCENARIO is None:
        raise HTTPException(503, "scenario not loaded")
    # Public view: omit known_facts, seed_exchanges, starting_meters
    return {
        "title": SCENARIO.title,
        "setup": SCENARIO.setup,
        "max_actions": SCENARIO.max_actions,
        "captor": {
            "name": SCENARIO.captor.name,
            "persona": SCENARIO.captor.persona,
            "initial_demands": SCENARIO.captor.initial_demands,
        },
        "hostages": [
            {"name": h.name, "description": h.description}
            for h in SCENARIO.hostages
        ],
    }


def _debug_affordances_enabled() -> bool:
    return Settings().debug_affordances


def _state_body(state: GameState) -> dict:
    """Public view + optional debug envelope. Shared by /state and /reset so
    clients see the same shape from both endpoints."""
    body = state.public_view()
    if _debug_affordances_enabled():
        body["debug"] = {
            "stress": state.stress,
            "trust": state.trust,
            "desperation": state.desperation,
            "action_count": state.action_count,
            "max_actions": state.max_actions,
            "fired_timeline_event_ids": sorted(state.fired_timeline_event_ids),
        }
    return body


@app.get("/state")
async def get_state() -> dict:
    if GAME_STATE is None:
        raise HTTPException(503, "game state not initialised")
    check_end_conditions(GAME_STATE)
    return _state_body(GAME_STATE)


@app.post("/action")
async def post_action(req: ActionRequest) -> StreamingResponse:
    if CAPTOR_AGENT is None or GAME_STATE is None:
        raise HTTPException(503, "service not ready")
    if GAME_STATE.ended:
        raise HTTPException(409, f"scenario already ended ({GAME_STATE.end_reason})")
    if not req.line.strip():
        raise HTTPException(422, "line must be a non-empty player utterance")

    user_prompt = req.line
    log.info("/action prompt=%r", user_prompt)

    return StreamingResponse(
        _action_event_stream(user_prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _action_event_stream(user_prompt: str) -> AsyncIterator[bytes]:
    """SSE generator for /action. Streams text deltas, then state, then done."""
    global HISTORY
    assert CAPTOR_AGENT is not None and GAME_STATE is not None and EVENT_BUS is not None

    async with ACTION_LOCK:
        last_text = ""
        t_start = time.monotonic()
        t_first: float | None = None
        try:
            async with asyncio.timeout(ACTION_STREAM_TIMEOUT_SECONDS):
                async with CAPTOR_AGENT.run_stream(
                    user_prompt, message_history=HISTORY
                ) as result:
                    async for partial in result.stream_output():
                        text = getattr(partial, "response_text", "") or ""
                        if text and text != last_text:
                            if t_first is None:
                                t_first = time.monotonic()
                            delta = text[len(last_text):]
                            yield _sse({"type": "text", "text": delta})
                            last_text = text

                    final = await result.get_output()
                    # Persist + mutate only on full completion (cancellation jumps
                    # straight to the except clause and skips this block).
                    HISTORY = list(result.all_messages())
                    GAME_STATE.action_count += 1
                    apply_response(GAME_STATE, final)
                    yield _sse({"type": "state", "state": GAME_STATE.public_view()})
                    # Fire any due timeline events AFTER state mutation but
                    # BEFORE we yield 'done', so the client renders them in
                    # order and the next /options call sees them in HISTORY.
                    # Skip if the captor's response already ended the game.
                    if not GAME_STATE.ended:
                        for evt in fire_due_timeline_events(GAME_STATE, SCENARIO):
                            yield _sse({"type": "timeline", "event": _timeline_event_payload(evt)})
                    if GAME_STATE.ended:
                        await EVENT_BUS.publish({
                            "type": "game_ended",
                            "end_reason": GAME_STATE.end_reason,
                            "captor_outcome": GAME_STATE.captor_outcome,
                        })
                    ttfb_ms = int((t_first - t_start) * 1000) if t_first else -1
                    total_ms = int((time.monotonic() - t_start) * 1000)
                    usage = result.usage()
                    log.info(
                        "/action complete: ttfb=%dms total=%dms response_chars=%d "
                        "input_tokens=%s output_tokens=%d action_count=%d history_msgs=%d "
                        "history_tokens=%s",
                        ttfb_ms, total_ms, len(final.response_text or ""),
                        _pct_of_ctx(usage.input_tokens), usage.output_tokens,
                        GAME_STATE.action_count, len(HISTORY),
                        _pct_of_ctx(_approx_tokens(HISTORY)),
                    )
            yield _sse({"type": "done"})
        except asyncio.CancelledError:
            log.info("/action cancelled by client (partial discarded)")
            raise
        except TimeoutError:
            log.warning(
                "/action stalled past %.1fs — likely context exhaustion or model hang",
                ACTION_STREAM_TIMEOUT_SECONDS,
            )
            yield _sse({
                "type": "error",
                "message": (
                    f"model stalled after {ACTION_STREAM_TIMEOUT_SECONDS:.0f}s — "
                    "likely context exhaustion. Try /reset."
                ),
            })
        except ValidationError as e:
            log.warning("captor response failed validation: %s", e)
            yield _sse({"type": "error", "message": f"validation_error: {e}"})
        except Exception as e:
            log.exception("/action stream failed")
            yield _sse({"type": "error", "message": str(e)})


@app.post("/reset")
async def post_reset() -> JSONResponse:
    global HISTORY, GAME_STATE
    if SCENARIO is None or EVENT_BUS is None:
        raise HTTPException(503, "service not ready")
    log.info("/reset")
    HISTORY = seed_history(SCENARIO.captor)
    GAME_STATE = GameState.from_scenario(SCENARIO)
    await EVENT_BUS.publish({"type": "reset"})
    return JSONResponse(_state_body(GAME_STATE))


@app.get("/options")
async def get_options() -> dict:
    """Generate 2-5 emotion-tagged dialogue picks for the negotiator's next line.

    Read-only with respect to HISTORY: the options agent's call is NOT written
    back via result.all_messages(), because doing so would leak the
    "Generate options." user prompt into the captor's history and confuse it on
    the next turn.

    Shares ACTION_LOCK with /action so this can't read HISTORY mid-mutation.
    Called by the client (a) on boot before any captor turn, and (b) after each
    /action SSE 'done'. On any failure (validation, timeout, model error)
    returns `{"options": []}` so the Godot client falls into its existing
    "No options available." path instead of breaking the UX.
    """
    if SCENARIO is None or GAME_STATE is None or OPTIONS_AGENT is None:
        raise HTTPException(503, "service not ready")
    if GAME_STATE.ended:
        return {"options": []}
    action_count = GAME_STATE.action_count
    actions_remaining = GAME_STATE.actions_remaining()
    log.info("/options request (action_count=%d)", action_count)
    t_start = time.monotonic()
    # Pace signal in the user prompt (not the system prompt) so prefix-cache
    # stays warm across turns. Don't leak meters — actions_remaining is in
    # public_view().
    user_prompt = (
        f"Generate options. Actions remaining: {actions_remaining}/{GAME_STATE.max_actions}."
    )
    with logfire.span(
        "options_request",
        action_count=action_count,
        actions_remaining=actions_remaining,
        history_msgs=len(HISTORY),
    ) as span:
        try:
            async with ACTION_LOCK:
                async with asyncio.timeout(OPTIONS_TIMEOUT_SECONDS):
                    # Read-only: do NOT write result back to HISTORY.
                    result = await OPTIONS_AGENT.run(user_prompt, message_history=HISTORY)
        except (TimeoutError, ValidationError) as e:
            span.set_attribute("error", type(e).__name__)
            log.warning("/options failed (%s): %s", type(e).__name__, e)
            return {"options": []}
        except Exception as e:
            span.set_attribute("error", type(e).__name__)
            log.exception("/options crashed: %s", e)
            return {"options": []}
        usage = result.usage()
        span.set_attribute("options_count", len(result.output.options))
        span.set_attribute("output_tokens", usage.output_tokens)
        log.info(
            "/options complete: %dms options=%d input_tokens=%s output_tokens=%d",
            int((time.monotonic() - t_start) * 1000), len(result.output.options),
            _pct_of_ctx(usage.input_tokens), usage.output_tokens,
        )
        return {"options": [o.model_dump() for o in result.output.options]}


@app.get("/events")
async def get_events() -> StreamingResponse:
    if EVENT_BUS is None:
        raise HTTPException(503, "service not ready")

    async def stream() -> AsyncIterator[bytes]:
        q = await EVENT_BUS.subscribe()
        try:
            yield _sse({"type": "connected"})
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=EVENTS_KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                yield _sse(payload)
        except asyncio.CancelledError:
            raise
        finally:
            await EVENT_BUS.unsubscribe(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ───────────────────────── helpers ─────────────────────────


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
