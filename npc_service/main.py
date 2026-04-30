from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
    Scenario,
    TimelineEvent,
    load_scenario,
)


logging.basicConfig(
    level=os.getenv("NPC_LOG_LEVEL", "info").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# httpx logs every request at INFO; at DEBUG it spams every health probe and
# chat completion. Pin it to WARNING so NPC_LOG_LEVEL=debug stays readable.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("npc_service")

# Logfire: instrument the captor agent and FastAPI. send_to_logfire='if-token-present'
# keeps the SDK silent when no project credentials / LOGFIRE_TOKEN are configured —
# safe for fresh checkouts and CI without credentials.
logfire.configure(
    service_name="npc_service",
    send_to_logfire="if-token-present",
    console=False,
)
logfire.instrument_pydantic_ai()
# httpx: capture every call to llama-server as a child span — surfaces LLM
# round-trip latency, status, and timeouts. Critical for diagnosing the cold-load
# / stream-stall / context-exhaustion failure modes called out in CLAUDE.md.
logfire.instrument_httpx()
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
# History compaction: when approx token count of message_history crosses this,
# summarize_when_large folds the oldest messages into a recap before the next
# captor turn. Heuristic: chars / 4. Set well below LLAMA_CTX so the summarizer
# has its own working room.
SUMMARY_TRIGGER_TOKENS = int(os.getenv("NPC_SUMMARY_TRIGGER_TOKENS", "5000"))
# Number of trailing messages to keep verbatim past the recap (≈ last 2 turns).
SUMMARY_KEEP_TAIL = int(os.getenv("NPC_SUMMARY_KEEP_TAIL", "4"))
# Hard ceiling: even if summarization didn't fire, never send more than this
# many messages to the captor. Last-resort safety net.
MAX_HISTORY_MSGS = int(os.getenv("NPC_MAX_HISTORY_MSGS", "20"))
# Whole-stream timeout for /action. If the model stalls past this, surface an
# explicit error rather than hanging the SSE response open indefinitely.
ACTION_STREAM_TIMEOUT_SECONDS = float(os.getenv("NPC_ACTION_TIMEOUT", "120"))


# ───────────────────────── globals (set in lifespan) ─────────────────────────

SCENARIO: Scenario | None = None
GAME_STATE: GameState | None = None
HISTORY: list = []  # pydantic_ai message history
CAPTOR_AGENT: Agent | None = None
SUMMARIZER_AGENT: Agent | None = None
LLAMA_PROC: subprocess.Popen | None = None
TIMELINE_TASK: asyncio.Task | None = None
EVENT_BUS: "EventBus | None" = None
ACTION_LOCK = asyncio.Lock()  # serialise /action so two players can't race


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
    if state.time_remaining_seconds() == 0:
        state.ended = True
        state.captor_outcome = state.captor_outcome or "breached"
        state.end_reason = "timer_expired"


# ───────────────────────── timeline scheduler ─────────────────────────


def synthetic_event_message(evt: TimelineEvent) -> ModelRequest:
    """Inject a scene-update as a user-role message the captor sees on their next turn."""
    return ModelRequest(parts=[UserPromptPart(content=f"[scene update: {evt.injection_text}]")])


async def timeline_scheduler(scenario: Scenario, state: GameState, bus: EventBus) -> None:
    try:
        for evt in scenario.timeline:
            target = state.scenario_start + timedelta(seconds=evt.at_seconds)
            wait = (target - datetime.now(timezone.utc)).total_seconds()
            if wait > 0:
                await asyncio.sleep(wait)
            if state.ended:
                return
            log.info("timeline event firing: id=%s at_seconds=%s", evt.id, evt.at_seconds)
            HISTORY.append(synthetic_event_message(evt))
            state.last_event_ids.append(evt.id)
            await bus.publish({
                "type": "timeline_event",
                "event": {
                    "id": evt.id,
                    "at_seconds": evt.at_seconds,
                    "kind": evt.kind,
                    "notification_text": evt.notification_text,
                },
            })
    except asyncio.CancelledError:
        log.info("timeline scheduler cancelled")
        raise


# ───────────────────────── history processors (pydantic-ai history_processors=) ─────────────────────────


def _approx_tokens(messages: list[ModelMessage]) -> int:
    """Cheap proxy for prompt token count. ~4 chars/token works as a threshold."""
    return sum(len(str(m)) for m in messages) // 4


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
        "history compaction: approx_tokens=%d threshold=%d folding %d msgs into recap, keeping %d",
        approx, SUMMARY_TRIGGER_TOKENS, len(head), len(tail),
    )
    t0 = time.monotonic()
    summary = await SUMMARIZER_AGENT.run(message_history=head)
    log.info(
        "history compaction: recap produced in %.2fs, recap_chars=%d",
        time.monotonic() - t0, len(summary.output),
    )
    recap_msg = ModelRequest(parts=[
        UserPromptPart(content=f"[recap of earlier negotiation]\n{summary.output}")
    ])
    return [recap_msg, *tail]


def hard_tail_cap(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Last-resort cap: drop oldest messages once we exceed MAX_HISTORY_MSGS."""
    if len(messages) <= MAX_HISTORY_MSGS:
        return messages
    log.info(
        "history compaction: hard cap dropping %d oldest msgs (had %d, cap %d)",
        len(messages) - MAX_HISTORY_MSGS, len(messages), MAX_HISTORY_MSGS,
    )
    return messages[-MAX_HISTORY_MSGS:]


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
    global SCENARIO, GAME_STATE, HISTORY, CAPTOR_AGENT
    global SUMMARIZER_AGENT, LLAMA_PROC, TIMELINE_TASK, EVENT_BUS

    scenario_path = Path(os.getenv("SCENARIO_PATH", str(Path(__file__).parent / "scenario.json")))
    log.info("loading scenario: %s", scenario_path)
    SCENARIO = load_scenario(scenario_path)

    if os.getenv("LAUNCH_LLAMA", "true").lower() in ("1", "true", "yes"):
        log.info("starting llama-server (LAUNCH_LLAMA=true)")
        cfg = resolve_config()
        try:
            LLAMA_PROC = start_llama(cfg)
            await wait_healthy(host=cfg.host, port=cfg.port)
        except LlamaStartupError:
            log.exception("llama-server failed to come up")
            stop_llama(LLAMA_PROC)
            raise
    else:
        log.info("LAUNCH_LLAMA=false — assuming llama-server is managed externally")
        cfg = resolve_config()
        # Quick non-blocking probe; warn but proceed even if unreachable.
        try:
            await wait_healthy(host=cfg.host, port=cfg.port, timeout=EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS)
        except LlamaStartupError as e:
            log.warning("llama-server not reachable yet: %s — sidecar will boot anyway", e)

    base_url = os.getenv("LLAMA_BASE_URL", f"http://{cfg.host}:{cfg.port}/v1")
    model_name = os.getenv("LLAMA_MODEL", "gemma-4-e4b")
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
    SUMMARIZER_AGENT = Agent(
        model,
        instructions=(
            "Summarize the negotiation so far in 4-6 short bullets. "
            "Cover: emotional state, demands raised, offers made, hostage status, "
            "and any commitments. Plain text, no preamble."
        ),
    )
    CAPTOR_AGENT = Agent(
        model,
        output_type=ToolOutput(CaptorResponse),
        instructions=build_system_prompt(SCENARIO.captor, SCENARIO.hostages),
        history_processors=[summarize_when_large, hard_tail_cap],
        model_settings={"max_tokens": 512},
    )

    HISTORY = seed_history(SCENARIO.captor)
    GAME_STATE = GameState.from_scenario(SCENARIO)
    EVENT_BUS = EventBus()
    TIMELINE_TASK = asyncio.create_task(
        timeline_scheduler(SCENARIO, GAME_STATE, EVENT_BUS),
        name="timeline_scheduler",
    )

    log.info("npc_service ready")
    try:
        yield
    finally:
        log.info("shutting down")
        if TIMELINE_TASK and not TIMELINE_TASK.done():
            TIMELINE_TASK.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await TIMELINE_TASK
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
        "duration_seconds": SCENARIO.duration_seconds,
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


@app.get("/state")
async def get_state() -> dict:
    if GAME_STATE is None:
        raise HTTPException(503, "game state not initialised")
    check_end_conditions(GAME_STATE)
    return GAME_STATE.public_view()


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
        completed = False
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
                    apply_response(GAME_STATE, final)
                    yield _sse({"type": "state", "state": GAME_STATE.public_view()})
                    if GAME_STATE.ended:
                        await EVENT_BUS.publish({
                            "type": "game_ended",
                            "end_reason": GAME_STATE.end_reason,
                            "captor_outcome": GAME_STATE.captor_outcome,
                        })
                    ttfb_ms = int((t_first - t_start) * 1000) if t_first else -1
                    total_ms = int((time.monotonic() - t_start) * 1000)
                    log.info(
                        "/action complete: ttfb=%dms total=%dms response_chars=%d history_msgs=%d approx_tokens=%d",
                        ttfb_ms, total_ms, len(final.response_text or ""),
                        len(HISTORY), _approx_tokens(HISTORY),
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


@app.post("/regenerate")
async def post_regenerate() -> StreamingResponse:
    """Debug-only: pop the last exchange and re-stream the captor's response."""
    if os.getenv("DEBUG_AFFORDANCES", "false").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "regenerate disabled (set DEBUG_AFFORDANCES=true)")
    if CAPTOR_AGENT is None or GAME_STATE is None:
        raise HTTPException(503, "service not ready")
    if GAME_STATE.ended:
        raise HTTPException(409, "scenario already ended")

    global HISTORY
    # find the last user prompt; pop it and any later messages, re-issue
    last_user_prompt = None
    new_history = list(HISTORY)
    while new_history:
        msg = new_history.pop()
        if isinstance(msg, ModelRequest):
            for part in reversed(msg.parts):
                if isinstance(part, UserPromptPart):
                    last_user_prompt = part.content
                    break
            if last_user_prompt is not None:
                break
    if last_user_prompt is None:
        raise HTTPException(400, "no prior user prompt to regenerate")
    HISTORY = new_history
    return StreamingResponse(
        _action_event_stream(str(last_user_prompt)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
async def post_reset() -> JSONResponse:
    global HISTORY, GAME_STATE, TIMELINE_TASK
    if SCENARIO is None or EVENT_BUS is None:
        raise HTTPException(503, "service not ready")
    log.info("/reset")
    if TIMELINE_TASK and not TIMELINE_TASK.done():
        TIMELINE_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await TIMELINE_TASK
    HISTORY = seed_history(SCENARIO.captor)
    GAME_STATE = GameState.from_scenario(SCENARIO)
    TIMELINE_TASK = asyncio.create_task(
        timeline_scheduler(SCENARIO, GAME_STATE, EVENT_BUS),
        name="timeline_scheduler",
    )
    await EVENT_BUS.publish({"type": "reset"})
    return JSONResponse(GAME_STATE.public_view())


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
