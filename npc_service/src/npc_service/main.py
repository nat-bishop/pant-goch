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
from pydantic_ai import Agent, ToolOutput
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
    AgentDef,
    AgentReport,
    AgentState,
    HandlerState,
    Operation,
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
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("npc_service")

# Logfire: load the write token from the credentials file next to this module
# rather than relying on CWD-based discovery. Godot spawns the sidecar with
# cwd=project root (see SidecarHost.cs), not npc_service/, so the default
# `.logfire/` lookup misses the file. send_to_logfire='if-token-present' keeps
# the SDK silent when neither the file nor LOGFIRE_TOKEN exists.
# src layout: src/npc_service/main.py → npc_service project root is parents[2].
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
logfire.instrument_httpx(capture_all=True)
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())

EVENTS_KEEPALIVE_SECONDS = 15.0
EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS = 2.0
SUMMARY_TRIGGER_FRAC = _BOOT_SETTINGS.summary_trigger_frac
SUMMARY_TRIGGER_TOKENS: int = _BOOT_SETTINGS.summary_trigger_tokens
LLAMA_CTX_TOKENS: int = 0
SUMMARY_KEEP_TAIL = _BOOT_SETTINGS.summary_keep_tail
ACTION_STREAM_TIMEOUT_SECONDS = _BOOT_SETTINGS.action_stream_timeout_seconds
OPTIONS_TIMEOUT_SECONDS = _BOOT_SETTINGS.options_timeout_seconds


# ───────────────────────── globals (set in lifespan) ─────────────────────────

SCENARIO: Scenario | None = None
HANDLER_STATE: HandlerState | None = None
HISTORIES: dict[str, list[ModelMessage]] = {}
AGENTS: dict[str, Agent] = {}
OPTIONS_AGENT: Agent | None = None
SUMMARIZER_AGENT: Agent | None = None
LLAMA_PROC: subprocess.Popen | None = None
EVENT_BUS: "EventBus | None" = None
ACTION_LOCK = asyncio.Lock()  # serialise /action and /options across all channels


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


# ───────────────────────── system prompts ─────────────────────────


def build_agent_system_prompt(agent: AgentDef, op: Operation) -> str:
    facts = "\n".join(f"- {f}" for f in agent.known_facts)

    examples = ""
    if agent.seed_exchanges:
        ex = agent.seed_exchanges[0]
        example_response = AgentReport(
            report_text=ex.assistant,
            stress_delta=-2,
            trust_delta=1,
        )
        examples = (
            f"\nExample. Handler says: {ex.user}\n"
            f"Your structured response:\n{example_response.model_dump_json()}\n"
        )

    return f"""You are {agent.callsign} ({agent.name}), a field asset reporting to your handler.

Cover / role: {agent.role}
City: {agent.city}

{agent.persona}

Operation codename: {op.codename}
Objective: {op.objective}

You know these things to be true. Speak only from this knowledge — if asked \
about anything outside it, deflect in character ({agent.deflection_style}). \
Never invent names, places, times, or events that are not in your knowledge:
{facts}

Each turn the handler sends you a message over the secure channel. Read it and \
respond with these fields:

- report_text: 1-3 sentences IN CHARACTER, addressing what the handler said. \
Use the voice of your persona. Do not narrate scene; speak as yourself, on the \
radio, to the handler.
- stress_delta: integer in -20..+20. How the handler's message affected your \
stress. Reassurance lowers it; demands or alarming intel raise it; threats \
spike it.
- trust_delta: integer in -20..+20. How the handler's message affected your \
trust in them. Coherent direction and acknowledgement raise it; contradictions, \
risk-shifting onto you, or going dark on prior commitments lower it.{examples}"""


def build_options_system_prompt(scenario: Scenario) -> str:
    """System prompt for OPTIONS_AGENT.

    Stays generic across all channels — the active agent's identity is provided
    in the per-call user prompt so the prefix stays cache-warm. Mirrors
    /scenario's public view: setup + op codename/objective + agent roster.
    Deliberately omits known_facts, seed_exchanges, starting_meters — the
    options agent reasons from player-knowable state only, preserving the
    "meters never leave the server" invariant.
    """
    op = scenario.operation
    roster = "\n".join(
        f"- {a.callsign} ({a.name}), {a.role}, {a.city}"
        for a in scenario.agents
    )

    return f"""You are a dialogue coach helping a covert-operations handler decide what to say next \
to one of their field assets over a secure channel.

Operation: {op.codename} — {op.objective}

Brief: {scenario.setup}

Asset roster:
{roster}

Each call, you produce 2-5 dialogue options for the handler's next line to a SPECIFIC asset \
(named in the user prompt). Each option has:
- tag: a one or two word label naming the *intent or stance* the line conveys (e.g. "Probe", \
"Reassure", "Press", "Pivot", "Direct", "Acknowledge", "Hedge", "Verify", "Stand Down"). \
Tags MUST be distinct within a set — never two options with the same tag.
- hidden_line: the actual sentence the handler would speak (1-3 sentences). The PLAYER will \
not see this until they pick it; they only see the tag.

Options must represent meaningfully different *strategies*, not slight rewordings. The handler's \
goal is to keep the operation moving and the assets safe and trusting — but options should \
include risky, confrontational, and command-style moves too, not only safe ones, so the \
player has a real choice.

Format example (do not literally reuse these lines):

{{"options":[
  {{"tag":"Verify","hidden_line":"Walk me through the timestamps once more — slow."}},
  {{"tag":"Reassure","hidden_line":"You're doing the work I sent you to do. Hold position."}},
  {{"tag":"Press","hidden_line":"You're sure that's a separate vehicle and not a tail?"}},
  {{"tag":"Hedge","hidden_line":"Stand down for the next exchange. We re-task once Vienna comes back up."}}
]}}
"""


# ───────────────────────── state mutation ─────────────────────────


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def apply_report(state: HandlerState, agent_id: str, r: AgentReport) -> None:
    """Atomic mutation of agent state from a fully-validated report."""
    a = state.agents[agent_id]
    a.stress = _clamp(a.stress + r.stress_delta, 0, 100)
    a.trust = _clamp(a.trust + r.trust_delta, 0, 100)
    a.last_report_action_count = state.action_count
    log.info(
        "agent[%s] meters: stress=%d trust=%d (after report at action_count=%d)",
        agent_id, a.stress, a.trust, state.action_count,
    )
    check_end_conditions(state)


def check_end_conditions(state: HandlerState) -> None:
    if state.ended:
        return
    if all(a.lost for a in state.agents.values()):
        state.ended = True
        state.end_reason = "all_lost"
        return
    if state.action_count >= state.max_actions:
        state.ended = True
        state.end_reason = "budget_exhausted"


# ───────────────────────── scene-update injections ─────────────────────────


def synthetic_event_message(text: str) -> ModelRequest:
    """Inject a scene-update as a user-role message the agent sees on their next turn."""
    return ModelRequest(parts=[UserPromptPart(content=f"[scene update: {text}]")])


def fire_due_timeline_events(state: HandlerState, scenario: Scenario) -> list[TimelineEvent]:
    """Fire timeline events whose threshold has been crossed and which haven't fired.

    For each event, append synthetic scene-update messages to each addressed
    agent's HISTORY based on event.injections. Idempotent — events fire at most
    once per game.
    """
    due = [
        e for e in scenario.timeline
        if e.after_actions <= state.action_count
        and e.id not in state.fired_timeline_event_ids
    ]
    for evt in due:
        log.info(
            "timeline event firing: id=%s after_actions=%d (action_count=%d) injections=%s",
            evt.id, evt.after_actions, state.action_count, sorted(evt.injections),
        )
        for agent_id, text in evt.injections.items():
            HISTORIES[agent_id].append(synthetic_event_message(text))
        state.fired_timeline_event_ids.add(evt.id)
    return due


def _timeline_event_payload(evt: TimelineEvent) -> dict:
    return {
        "id": evt.id,
        "after_actions": evt.after_actions,
        "kind": evt.kind,
        "notification_text": evt.notification_text,
        "affected_agents": sorted(evt.injections),
    }


# ───────────────────────── history processors (per-agent compaction) ─────────────────────────


def _approx_tokens(messages: list[ModelMessage]) -> int:
    """Approximate prompt-token count of `messages` if sent as message_history."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ModelResponse) and msg.usage is not None:
            tail_chars = sum(len(str(m)) for m in messages[i + 1:])
            return msg.usage.input_tokens + msg.usage.output_tokens + tail_chars // 4
    return sum(len(str(m)) for m in messages) // 4


def _pct_of_ctx(tokens: int) -> str:
    if LLAMA_CTX_TOKENS <= 0:
        return str(tokens)
    return f"{tokens} ({100 * tokens // LLAMA_CTX_TOKENS}% of {LLAMA_CTX_TOKENS})"


async def summarize_when_large(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Fold the oldest messages of a single agent's HISTORY into a recap when it grows too big."""
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
        UserPromptPart(content=f"[recap of earlier transmissions]\n{summary.output}")
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


# ───────────────────────── seeding histories ─────────────────────────


def seed_history(agent: AgentDef) -> list[ModelMessage]:
    """Seed an agent's HISTORY with their opening transmission as a scene-update.

    Returns the agent's initial HISTORY: a single scene-update UserPromptPart
    informing them of their own prior transmission to the handler. NOT a
    ModelResponse with the opening text — that would teach the model to emit
    plain text instead of tool-calling structured output (CLAUDE.md). The agent
    treats this as scene context; the *client* renders agent.opening_transmission
    directly into the chat as the agent's first line.
    """
    return [synthetic_event_message(
        f"Your prior transmission to the handler was: \"{agent.opening_transmission}\""
    )]


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global SCENARIO, HANDLER_STATE, HISTORIES, AGENTS, OPTIONS_AGENT
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
    log.info("building Pydantic AI agents: model=%s base_url=%s", model_name, base_url)
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key="not-needed"),
    )
    SUMMARIZER_AGENT = Agent(
        model,
        name="summarizer",
        instructions=(
            "Summarize the transmissions between the handler and this asset so far in 4-6 short bullets. "
            "Cover: asset's emotional state, intel reported, commitments made or refused, current orders, "
            "and any threads still open. Plain text, no preamble."
        ),
    )

    # One Agent per asset persona. Each gets its own ToolOutput(AgentReport)
    # + max_tokens=512 + summarize_when_large processor. Same shape as the
    # single-captor architecture, just N×. name= surfaces in Logfire spans so
    # each agent's runs are filterable.
    AGENTS = {
        a.id: Agent(
            model,
            name=f"agent_{a.id}",
            output_type=ToolOutput(AgentReport),
            instructions=build_agent_system_prompt(a, SCENARIO.operation),
            history_processors=[summarize_when_large],
            model_settings={"max_tokens": 512},
        )
        for a in SCENARIO.agents
    }

    # Single shared OPTIONS_AGENT, parameterized via user prompt per call.
    # Reuses the same model and history processors so it sees a consistent
    # compacted view of the active channel's HISTORY. max_tokens=1536 covers
    # the worst case (5 options × ~400-char hidden_line + 24-char tags + JSON
    # + tool-call envelope ≈ 1100-1400 tokens) with margin.
    OPTIONS_AGENT = Agent(
        model,
        name="options",
        output_type=ToolOutput(OptionsResponse),
        instructions=build_options_system_prompt(SCENARIO),
        history_processors=[summarize_when_large],
        model_settings={"max_tokens": 1536},
    )

    HISTORIES = {a.id: seed_history(a) for a in SCENARIO.agents}
    HANDLER_STATE = HandlerState.from_scenario(SCENARIO)
    EVENT_BUS = EventBus()

    log.info("npc_service ready: agents=%s", sorted(AGENTS))
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
        "agents_ready": sorted(AGENTS),
    }


@app.get("/scenario")
async def get_scenario() -> dict:
    if SCENARIO is None:
        raise HTTPException(503, "scenario not loaded")
    # Public view: omit known_facts, seed_exchanges, starting_meters,
    # operation.world_state. Include opening_transmission so the client can
    # render the agent's first line into the chat at boot.
    return {
        "title": SCENARIO.title,
        "setup": SCENARIO.setup,
        "max_actions": SCENARIO.max_actions,
        "operation": {
            "codename": SCENARIO.operation.codename,
            "objective": SCENARIO.operation.objective,
        },
        "agents": [
            {
                "id": a.id,
                "callsign": a.callsign,
                "name": a.name,
                "city": a.city,
                "role": a.role,
                "persona": a.persona,
                "opening_transmission": a.opening_transmission,
            }
            for a in SCENARIO.agents
        ],
    }


def _debug_affordances_enabled() -> bool:
    return Settings().debug_affordances


def _state_body(state: HandlerState) -> dict:
    """Public view + optional debug envelope. Shared by /state and /reset."""
    body = state.public_view()
    if _debug_affordances_enabled():
        body["debug"] = {
            "agents": [
                {"id": a.id, "stress": a.stress, "trust": a.trust}
                for a in state.agents.values()
            ],
            "action_count": state.action_count,
            "max_actions": state.max_actions,
            "fired_timeline_event_ids": sorted(state.fired_timeline_event_ids),
        }
    return body


@app.get("/state")
async def get_state() -> dict:
    if HANDLER_STATE is None:
        raise HTTPException(503, "handler state not initialised")
    check_end_conditions(HANDLER_STATE)
    return _state_body(HANDLER_STATE)


@app.post("/action")
async def post_action(req: ActionRequest) -> StreamingResponse:
    if not AGENTS or HANDLER_STATE is None:
        raise HTTPException(503, "service not ready")
    if HANDLER_STATE.ended:
        raise HTTPException(409, f"scenario already ended ({HANDLER_STATE.end_reason})")
    if req.agent_id not in AGENTS:
        raise HTTPException(404, f"unknown agent_id: {req.agent_id}")
    if HANDLER_STATE.agents[req.agent_id].lost:
        raise HTTPException(409, f"agent {req.agent_id} is lost — channel closed")
    if not req.line.strip():
        raise HTTPException(422, "line must be a non-empty handler utterance")

    log.info("/action agent=%s prompt=%r", req.agent_id, req.line)

    return StreamingResponse(
        _action_event_stream(req.agent_id, req.line),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _action_event_stream(agent_id: str, user_prompt: str) -> AsyncIterator[bytes]:
    """SSE generator for /action. Streams text deltas, then state, then done.

    Lock is global across all channels so /options never reads HISTORIES mid-mutation.
    """
    assert HANDLER_STATE is not None and EVENT_BUS is not None and SCENARIO is not None
    agent = AGENTS[agent_id]

    async with ACTION_LOCK:
        last_text = ""
        t_start = time.monotonic()
        t_first: float | None = None
        try:
            async with asyncio.timeout(ACTION_STREAM_TIMEOUT_SECONDS):
                async with agent.run_stream(
                    user_prompt, message_history=HISTORIES[agent_id]
                ) as result:
                    async for partial in result.stream_output():
                        text = getattr(partial, "report_text", "") or ""
                        if text and text != last_text:
                            if t_first is None:
                                t_first = time.monotonic()
                            delta = text[len(last_text):]
                            yield _sse({"type": "text", "agent_id": agent_id, "text": delta})
                            last_text = text

                    final = await result.get_output()
                    # Persist + mutate only on full completion (cancellation jumps
                    # straight to the except clause and skips this block).
                    HISTORIES[agent_id] = list(result.all_messages())
                    HANDLER_STATE.action_count += 1
                    apply_report(HANDLER_STATE, agent_id, final)
                    yield _sse({"type": "state", "state": _state_body(HANDLER_STATE)})
                    # Fire any due timeline events AFTER state mutation but
                    # BEFORE we yield 'done', so the client renders them in
                    # order and the next /options call sees them in HISTORIES.
                    if not HANDLER_STATE.ended:
                        for evt in fire_due_timeline_events(HANDLER_STATE, SCENARIO):
                            yield _sse({"type": "timeline", "event": _timeline_event_payload(evt)})
                    if HANDLER_STATE.ended:
                        await EVENT_BUS.publish({
                            "type": "game_ended",
                            "end_reason": HANDLER_STATE.end_reason,
                        })
                    ttfb_ms = int((t_first - t_start) * 1000) if t_first else -1
                    total_ms = int((time.monotonic() - t_start) * 1000)
                    usage = result.usage()
                    log.info(
                        "/action complete: agent=%s ttfb=%dms total=%dms response_chars=%d "
                        "input_tokens=%s output_tokens=%d action_count=%d history_msgs=%d "
                        "history_tokens=%s",
                        agent_id, ttfb_ms, total_ms, len(final.report_text or ""),
                        _pct_of_ctx(usage.input_tokens), usage.output_tokens,
                        HANDLER_STATE.action_count, len(HISTORIES[agent_id]),
                        _pct_of_ctx(_approx_tokens(HISTORIES[agent_id])),
                    )
            yield _sse({"type": "done", "agent_id": agent_id})
        except asyncio.CancelledError:
            log.info("/action cancelled by client (partial discarded) agent=%s", agent_id)
            raise
        except TimeoutError:
            log.warning(
                "/action stalled past %.1fs — likely context exhaustion (agent=%s)",
                ACTION_STREAM_TIMEOUT_SECONDS, agent_id,
            )
            yield _sse({
                "type": "error",
                "agent_id": agent_id,
                "message": (
                    f"model stalled after {ACTION_STREAM_TIMEOUT_SECONDS:.0f}s — "
                    "likely context exhaustion. Try /reset."
                ),
            })
        except ValidationError as e:
            log.warning("agent=%s response failed validation: %s", agent_id, e)
            yield _sse({"type": "error", "agent_id": agent_id, "message": f"validation_error: {e}"})
        except Exception as e:
            log.exception("/action stream failed agent=%s", agent_id)
            yield _sse({"type": "error", "agent_id": agent_id, "message": str(e)})


@app.post("/reset")
async def post_reset() -> JSONResponse:
    global HISTORIES, HANDLER_STATE
    if SCENARIO is None or EVENT_BUS is None:
        raise HTTPException(503, "service not ready")
    log.info("/reset")
    HISTORIES = {a.id: seed_history(a) for a in SCENARIO.agents}
    HANDLER_STATE = HandlerState.from_scenario(SCENARIO)
    await EVENT_BUS.publish({"type": "reset"})
    return JSONResponse(_state_body(HANDLER_STATE))


@app.get("/options")
async def get_options(agent_id: str) -> dict:
    """Generate 2-5 dialogue picks for the handler's next line in a specific channel.

    Read-only with respect to HISTORIES[agent_id]: the options agent's call is
    NOT written back via result.all_messages(), because doing so would leak the
    "Generate options for asset X." user prompt into the channel's history and
    confuse the agent on the next turn.

    Shares ACTION_LOCK with /action so this can't read HISTORIES mid-mutation.
    On any failure (validation, timeout, model error) returns
    `{"options": []}` so the Godot client falls into its existing
    "No options available." path instead of breaking the UX.
    """
    if SCENARIO is None or HANDLER_STATE is None or OPTIONS_AGENT is None:
        raise HTTPException(503, "service not ready")
    if agent_id not in HISTORIES:
        raise HTTPException(404, f"unknown agent_id: {agent_id}")
    if HANDLER_STATE.ended:
        return {"options": []}
    if HANDLER_STATE.agents[agent_id].lost:
        return {"options": []}
    action_count = HANDLER_STATE.action_count
    actions_remaining = HANDLER_STATE.actions_remaining()
    agent_def = next(a for a in SCENARIO.agents if a.id == agent_id)
    log.info("/options request agent=%s (action_count=%d)", agent_id, action_count)
    t_start = time.monotonic()
    # User prompt names the active asset only — the live action_count is
    # deliberately NOT included so the prefix-cache stays valid across calls
    # for the same channel. Pacing is already encoded in HISTORIES (timeline
    # scene-updates fire on the same clock).
    user_prompt = (
        f"Generate options for the handler's next line to {agent_def.callsign} "
        f"({agent_def.name}, {agent_def.city})."
    )
    with logfire.span(
        "options_request",
        agent_id=agent_id,
        action_count=action_count,
        actions_remaining=actions_remaining,
        history_msgs=len(HISTORIES[agent_id]),
    ) as span:
        try:
            async with ACTION_LOCK:
                async with asyncio.timeout(OPTIONS_TIMEOUT_SECONDS):
                    # Read-only: do NOT write result back to HISTORIES.
                    result = await OPTIONS_AGENT.run(user_prompt, message_history=HISTORIES[agent_id])
        except (TimeoutError, ValidationError) as e:
            span.set_attribute("error", type(e).__name__)
            log.warning("/options failed agent=%s (%s): %s", agent_id, type(e).__name__, e)
            return {"options": []}
        except Exception as e:
            span.set_attribute("error", type(e).__name__)
            log.exception("/options crashed agent=%s: %s", agent_id, e)
            return {"options": []}
        usage = result.usage()
        span.set_attribute("options_count", len(result.output.options))
        span.set_attribute("output_tokens", usage.output_tokens)
        log.info(
            "/options complete: agent=%s %dms options=%d input_tokens=%s output_tokens=%d",
            agent_id, int((time.monotonic() - t_start) * 1000), len(result.output.options),
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
