"""FastAPI surface for Inbox. See CLAUDE.md for the architecture and
load-bearing invariants.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import importlib.metadata
import json
import logging
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator, Literal

import logfire
from logfire import ConsoleOptions
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import replace
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

from .agents import (
    AgentBundle,
    ChatDeps,
    ExtractorOutput,
    build_agents,
    build_classifier_user_prompt,
    build_director_output_type,
    build_director_user_prompt,
    build_extractor_user_prompt,
    build_recent_classifier_history,
    commitments_snapshot,
    lexical_history_snapshot,
    make_llama_token_counter,
    player_canon_snapshot,
)
from pydantic_ai import NativeOutput
from .config import Settings
from .events import (
    BeatFired as BeatFiredEvent,
    CanonPromoted,
    ChatBodyRendered,
    ClassifierObserved,
    DirectorChose,
    EmailEmitted,
    EmailRead,
    Event,
    ExtractorObserved,
    SendRecorded,
    SessionEnded,
    ThreadHistoryAppended,
)
from .launcher import (
    LlamaStartupError,
    resolve_config,
    start_llama,
    stop_llama,
    wait_healthy,
)
from .models import (
    AxisResult,
    Beat,
    BeatEligibility,
    BeatFired,
    BeatSpec,
    CanonCandidate,
    CanonCategory,
    CanonDropRecord,
    Effect,
    Email,
    FactCheckAttempt,
    FactCheckTrace,
    InboxState,
    Not_,
    OpenRequest,
    Or_,
    Predicate,
    QualityDelta,
    Reply,
    Scenario,
    SendRequest,
    TickDecision,
    load_scenario,
)
from .session import Session, load_event_log, make_jsonl_persistor, make_session, replay_into_session
from .spine import (
    CANON_SALIENCE_THRESHOLD,
    derive_canon_promoted_events,
    derive_classify_effect_event,
    derive_literal_effect_event,
    evaluate_all_beats,
    evaluate_predicate,
    find_due_beats,
    scene_exit_satisfied,
)


_BOOT_SETTINGS = Settings()
# httpx is chatty at INFO; cap to WARNING. Third-party stdlib loggers still
# get bridged into Logfire below so request failures aren't lost.
logging.getLogger("httpx").setLevel(logging.WARNING)

# Logfire credentials live next to the npc_service project root.
_logfire_creds = Path(__file__).resolve().parents[2] / ".logfire" / "logfire_credentials.json"
_logfire_token = os.getenv("LOGFIRE_TOKEN")
if not _logfire_token and _logfire_creds.is_file():
    with contextlib.suppress(Exception):
        _logfire_token = json.loads(_logfire_creds.read_text()).get("token")
try:
    _service_version = importlib.metadata.version("npc-service")
except importlib.metadata.PackageNotFoundError:
    _service_version = "unknown"
logfire.configure(
    service_name="npc_service",
    service_version=_service_version,
    send_to_logfire="if-token-present",
    token=_logfire_token,
    console=ConsoleOptions() if _BOOT_SETTINGS.npc_logfire_console else False,
    # Single-player local game with no PII; default scrubbing would redact
    # email bodies and rationales — exactly the prose we want to inspect.
    scrubbing=False,
)
logfire.instrument_pydantic_ai()
logfire.instrument_pydantic()
logfire.instrument_httpx(capture_all=True)
# Bridge: third-party libraries (httpx, uvicorn-error, etc.) emit via stdlib
# logging. This handler ferries those records into Logfire so they're not
# lost when we don't use stdlib for our own call sites.
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())

# A pre-tagged Logfire instance for spans/events that wrap LLM work. Filtering
# by the `llm` tag in the Logfire UI surfaces every chat-agent / classifier
# span across all traces — beats drilling through the trace tree.
LLM_LOGFIRE = logfire.with_tags("llm")


class ChatAgentFailure(Exception):
    """Raised by `_render_email` (called from the reply / outbound /
    epilogue render entrypoints) when the LLM call (or its history
    processors) fails. `_send_stream` catches this and surfaces a
    structured `error` SSE event so the client can recover without seeing
    a torn HTTP stream. Classifier failures use a separate degraded path
    (skip the quality write) and don't raise."""


# ───────────────────────── globals (set in lifespan) ─────────────────────────
#
# Process-scope singletons: one scenario, one model + agent bundle, one
# llama subprocess. These are loaded once at lifespan boot and never
# mutate per-request.
#
# Per-game state lives in the SESSIONS registry — one `Session` per logical
# game, each holding its own InboxState, event log, and asyncio.Lock.
# Endpoints resolve a session via `Depends(get_session)`. The default
# single-player case maps to the entry under `DEFAULT_SESSION_ID`.


SCENARIO: Scenario | None = None
AGENTS: AgentBundle | None = None
SETTINGS: Settings | None = None  # populated at lifespan boot; read by hot-path endpoints
LLAMA_PROC: subprocess.Popen | None = None
EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS = 2.0  # tests set this to 0.1 for fast boot

DEFAULT_SESSION_ID = "default"
SESSIONS: dict[str, Session] = {}


def get_session(session_id: str = DEFAULT_SESSION_ID) -> Session:
    """FastAPI dependency: resolve a Session by id. Lazily creates a fresh
    session if one doesn't exist for the requested id (so the single-user
    `default` case Just Works without explicit session creation).
    """
    if session_id not in SESSIONS:
        assert SCENARIO is not None and AGENTS is not None, (
            "get_session called before lifespan boot"
        )
        SESSIONS[session_id] = make_session(SCENARIO, AGENTS)
    return SESSIONS[session_id]


# Back-compat shims for tests and the playtester that previously read these
# as module attributes. Only `default`-session reads are supported here —
# multi-session tests use the registry directly via `SESSIONS[...]`.
def _default_session() -> Session | None:
    return SESSIONS.get(DEFAULT_SESSION_ID)


def __getattr__(name: str):
    """Lazy module-level attribute resolver — keeps `main.STATE` working
    for tests that read it directly. Returns the default session's state /
    lock, or None if the session hasn't been booted yet."""
    if name == "STATE":
        s = _default_session()
        return s.state if s is not None else None
    if name == "ACTION_LOCK":
        s = _default_session()
        return s.lock if s is not None else None
    raise AttributeError(name)


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global SCENARIO, AGENTS, SETTINGS, LLAMA_PROC
    global _DEBUG_SCENARIO_CACHE

    settings = Settings()
    SETTINGS = settings
    scenario_path = settings.scenario_path or Path(__file__).resolve().parents[2] / "scenario.yaml"
    logfire.info("loading scenario from {path}", path=str(scenario_path))
    SCENARIO = load_scenario(scenario_path)
    # Invalidate the /debug/scenario cache so a different scenario loaded
    # in tests (or after a restart with a swapped YAML) doesn't serve stale
    # AST from the previous boot.
    _DEBUG_SCENARIO_CACHE = None

    cfg = resolve_config(settings)
    # Wrap the supervised period in one span so start, healthy, requests, and
    # stop all live under the same parent in the Logfire UI. `model_path` is
    # None when llama is downloading from HF instead of using a local file.
    with logfire.span(
        "llama_lifecycle",
        launch=settings.launch_llama,
        port=cfg.port,
        ctx=cfg.ctx,
        model_path=str(cfg.model_path) if cfg.model_path else None,
        hf_repo=cfg.hf_repo,
    ) as llama_span:
        if settings.launch_llama:
            logfire.info("starting llama-server (managed)")
            try:
                LLAMA_PROC = start_llama(cfg)
                llama_span.set_attribute("pid", LLAMA_PROC.pid)
                await wait_healthy(host=cfg.host, port=cfg.port)
                llama_span.set_attribute("healthy", True)
            except LlamaStartupError:
                logfire.exception("llama-server failed to come up")
                stop_llama(LLAMA_PROC)
                raise
        else:
            logfire.info("llama-server externally managed (LAUNCH_LLAMA=false)")
            try:
                await wait_healthy(
                    host=cfg.host, port=cfg.port,
                    timeout=EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS,
                )
            except LlamaStartupError as e:
                logfire.warn(
                    "llama-server not reachable yet — sidecar will boot anyway",
                    error=str(e),
                )

        api_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
        # /tokenize is a llama-server-specific endpoint at the root path,
        # NOT under /v1/. Strip /v1 if present.
        tokenize_url = api_url.removesuffix("/v1") + "/tokenize"
        model_name = "gemma-4-e4b"
        logfire.info(
            "building agents",
            model=model_name, base_url=api_url,
        )
        # Local-Gemma via llama-server: don't round-trip prior-turn thoughts.
        # Default 'auto' re-serialises ThinkingParts as <think>…</think> in
        # historical assistant content, violating Google's Gemma 4 multi-turn
        # rule (https://ai.google.dev/gemma/docs/capabilities/thinking) and
        # inflating input tokens by hundreds per turn. Replace on the default
        # profile (which carries supports_json_schema_output=True, etc.); a
        # bare OpenAIModelProfile() drops those flags and breaks NativeOutput.
        profile = replace(
            openai_model_profile(model_name),
            openai_chat_send_back_thinking_parts=False,
        )
        model = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=api_url, api_key="not-needed"),
            profile=profile,
        )
        token_budget = int(settings.llama_ctx * settings.history_compact_frac)
        AGENTS = build_agents(
            model,
            token_counter=make_llama_token_counter(tokenize_url),
            token_budget=token_budget,
            history_keep_tail=settings.history_compact_keep_tail,
        )
        logfire.info(
            "history compaction configured",
            budget_tokens=token_budget,
            ctx_tokens=settings.llama_ctx,
            budget_frac=settings.history_compact_frac,
            keep_tail=settings.history_compact_keep_tail,
        )

        SESSIONS.clear()
        await reset_session()
        boot_session = SESSIONS[DEFAULT_SESSION_ID]
        logfire.info(
            "npc_service ready",
            boot_emails=len(boot_session.state.emails),
            pending_beats=len(boot_session.state.pending_beats),
        )
        try:
            yield
        finally:
            logfire.info("shutting down")
            stop_llama(LLAMA_PROC)
            if LLAMA_PROC is not None and LLAMA_PROC.returncode is not None:
                llama_span.set_attribute("exit_code", LLAMA_PROC.returncode)


app = FastAPI(lifespan=lifespan, title="npc_service (Inbox)")
logfire.instrument_fastapi(app, capture_headers=False)


@app.exception_handler(HTTPException)
async def _log_http_exception(request: Request, exc: HTTPException):
    """Log every HTTPException once with structured fields. The FastAPI
    instrumentation tags the request span with the response status, but
    silent guard-rail trips (404 unknown email, 409 session ended, etc.)
    are easier to spot as a flat log line than by drilling into the span."""
    logfire.info(
        "http error response",
        status=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )
    return await http_exception_handler(request, exc)


# ───────────────────────── beat firing ─────────────────────────


async def _fire_beat(
    session: Session,
    beat: Beat,
    *,
    player_body: str | None,
    in_response_to: str | None,
    reply_thread_id: str | None,
) -> tuple[Email | None, list[QualityDelta], int | None, FactCheckTrace | None]:
    """Fire a beat and produce its email plus tick-decision side info.

    Returns `(email, write_deltas, scene_turn, fact_check)`:
      - `email`: the rendered email, or None on silent-complete.
      - `write_deltas`: any quality changes from literal effects (empty if
        the beat has none, or if writes were no-ops).
      - `scene_turn`: 1-indexed fire count for scene beats (computed
        after the email is appended), None for one-shots.
      - `fact_check`: per-attempt validator trace; None unless the beat
        declared `must_include_facts` or `must_not` AND wasn't
        pre-authored.

    Render paths:
    - `spec.pre_authored_prose` set → ship verbatim (no LLM call). The
      rendered email seeds `thread_messages` so the chat agent's next run
      sees the prose as a prior assistant turn.
    - chat-driven, beat.thread == reply_thread_id → reply-mode render: the
      chat agent receives `player_body` as user_prompt and the exchange is
      appended to thread history.
    - chat-driven, beat.thread != reply_thread_id (or reply_thread_id is
      None — boot/autonomous tick) → outbound-mode render: the chat agent
      receives a synthetic trigger and ChatDeps.outbound_mode=True so it
      composes an unprompted email rather than respond to the (other-
      thread) player_body it would otherwise see. Without this split, an
      off-thread NPC reads the player's other-thread body as if it were
      addressed to it (Bug #1) and pollutes its own thread history with
      messages the player never sent there.
    - terminals (`ends_session`) route through the epilogue renderer in
      either reply or outbound flavor depending on the same off-thread
      check.

    Scene-vs-one_shot post-fire bookkeeping:
    - one_shot → enters `fired_beat_ids` immediately (fire-once contract).
    - scene → enters `fired_beat_ids` only when `scene_exit_satisfied`
      passes after this fire's literal effects + email append. Otherwise
      it stays pending and may continue firing on subsequent ticks.
    """
    assert SCENARIO is not None and AGENTS is not None
    if beat.id in session.state.fired_beat_ids:
        return None, [], None, None

    # Literal effects apply BEFORE rendering so scene re-fires record
    # the new value even if rendering fails. Classify-driven effects on
    # this beat schedule the classifier on the next player reply — they
    # don't fire here.
    write_deltas: list[QualityDelta] = []
    for effect in beat.spec.effects:
        ev = derive_literal_effect_event(
            session.state, SCENARIO, effect, beat_id=beat.id,
        )
        if ev is not None:
            session.apply(ev)
            write_deltas.append(ev.delta)

    # No span wrapper here: pre-authored fires are sub-millisecond (a memcpy
    # plus a log line) and the chat-agent paths each carry their own
    # llm-tagged span. Adding a `fire_beat` layer here only deepened the
    # trace tree without filtering value. `beat_id` is threaded onto each
    # render via `beat=`, so the chat-driven fire is still searchable by
    # beat id.
    is_scene = beat.kind == "scene"
    is_outbound = (
        reply_thread_id is not None and beat.thread != reply_thread_id
    )
    email: Email | None
    fact_check: FactCheckTrace | None = None
    if beat.spec.pre_authored_prose is not None:
        npc = SCENARIO.npc_by_id[beat.npc]
        email = Email(
            id=f"email_{uuid.uuid4().hex[:8]}",
            beat_id=beat.id,
            thread_id=beat.thread,
            folder="Inbox",
            sender=npc.email_address,
            recipient=resolve_player_email(SCENARIO, session.state),
            body=beat.spec.pre_authored_prose.strip(),
            sent_at_label="just now",
        )
        session.apply(EmailEmitted(email=email))
        _seed_thread_history(session, beat.thread, email.body)
        logfire.info(
            "fire pre-authored beat {beat_id}",
            beat_id=beat.id, thread=beat.thread,
        )
    else:
        # Outbound chat-driven fires don't use player_body (it was
        # addressed to a different thread). Reply-mode fires require it.
        if not is_outbound:
            assert player_body is not None, (
                "reply-mode chat-driven beats require a player body — they trigger only on /send ticks"
            )
        prior_fires = _count_beat_email_fires(session, beat.id) if is_scene else 0
        if is_scene:
            logfire.info(
                "continue scene {beat_id} (turn {turn}, mode={mode})",
                beat_id=beat.id, turn=prior_fires + 1,
                mode="outbound" if is_outbound else "reply",
            )
        else:
            logfire.info(
                "fire plot beat {beat_id} (mode={mode})",
                beat_id=beat.id,
                mode="outbound" if is_outbound else "reply",
            )
        # None when the beat declares no must_include_facts / must_not — the
        # validator's append-when-non-None check is the single mechanism for
        # keeping un-validated beats trace-free.
        fact_check_log: list[FactCheckAttempt] | None = (
            [] if (beat.spec.must_include_facts or beat.spec.must_not) else None
        )
        if beat.ends_session:
            email = await _render_via_epilogue_agent(
                session,
                npc_id=beat.npc,
                thread_id=beat.thread,
                player_body=player_body or "",
                in_reply_to_email_id=in_response_to,
                plot_directive=beat.spec,
                beat_id=beat.id,
                outbound_mode=is_outbound,
                fact_check_log=fact_check_log,
            )
        else:
            email = await _render_via_chat_agent(
                session,
                npc_id=beat.npc,
                thread_id=beat.thread,
                plot_directive=beat.spec,
                beat_id=beat.id,
                player_body=player_body,
                outbound_mode=is_outbound,
                fact_check_log=fact_check_log,
            )
        if fact_check_log is not None:
            # Empty list means the validator no-op'd (shouldn't happen — we
            # only allocate fact_check_log when must_include_facts / must_not
            # is non-empty) — leave fact_check None in that case rather than
            # ship an attempts=[] payload that's semantically misleading.
            fact_check = FactCheckTrace(attempts=fact_check_log) if fact_check_log else None
        session.apply(EmailEmitted(email=email))

    scene_turn: int | None = prior_fires + 1 if is_scene else None

    # Emit the BeatFired event. one_shot: completed=True (always adds to
    # fired_beat_ids). scene: completed iff scene_exit_satisfied passes
    # after the email + literal effects landed.
    completed = (not is_scene) or scene_exit_satisfied(beat, session.state)
    session.apply(BeatFiredEvent(
        beat_id=beat.id,
        beat_kind=beat.kind,
        scene_turn=scene_turn,
        completed=completed,
    ))
    if is_scene and completed:
        logfire.info(
            "scene completed {beat_id} on turn {turn}",
            beat_id=beat.id, turn=scene_turn,
        )
    if beat.ends_session:
        session.apply(SessionEnded(beat_id=beat.id))
    return email, write_deltas, scene_turn, fact_check


def _seed_thread_history(session: Session, thread_id: str, body: str) -> None:
    """Anchored emails on NPC threads land in `thread_messages` as
    `ModelResponse(TextPart)` so the chat agent's next run sees them as
    prior assistant turns. System threads (no NPC) skip this — there's no
    chat agent for them."""
    assert SCENARIO is not None
    thread = SCENARIO.thread_by_id.get(thread_id)
    if thread is None or thread.npc is None:
        return
    session.apply(ThreadHistoryAppended(
        thread_id=thread_id,
        messages=[ModelResponse(parts=[TextPart(content=body)])],
    ))


# Synthetic user-prompt for outbound-mode renders. Pydantic AI's `Agent.run`
# expects a non-empty user_prompt; the OUTBOUND_MODE_FRAMING instructions
# block tells the chat agent that this trigger is system-shaped, not a
# player message, and that the agent must compose an unprompted outbound
# email per the directive instead of responding to it.
_OUTBOUND_TRIGGER_PROMPT = "[system: compose an outbound email per directive]"


async def _render_email(
    session: Session,
    *,
    npc_id: str,
    thread_id: str,
    user_prompt: str,
    plot_directive: BeatSpec | None,
    beat_id: str | None,
    message_history: list | None = None,
    append_history: bool = False,
    epilogue_digest: str | None = None,
    outbound_mode: bool = False,
    fact_check_log: list[FactCheckAttempt] | None = None,
    span_label: str = "render to {npc} on {thread}",
    extra_span_attrs: dict | None = None,
) -> Email:
    """Single LLM-render path. Builds ChatDeps, runs the chat agent, wraps
    failures as ChatAgentFailure, returns the assembled Email. If
    `append_history=True`, extends `session.state.thread_messages[thread_id]`
    via a ThreadHistoryAppended event with `result.new_messages()` — except
    in outbound mode, where only the assistant response is appended (the
    synthetic trigger is not a real player turn and would pollute the
    thread's history if persisted).
    When `epilogue_digest` is set, ChatDeps switches the agent into
    closing-email mode. When `outbound_mode` is set, ChatDeps activates
    the OUTBOUND framing block.

    `fact_check_log`, when supplied, is a mutable list the validator
    appends to per attempt; `_fire_beat` packages it into a
    `FactCheckTrace` after the render returns."""
    assert SCENARIO is not None and AGENTS is not None
    npc = SCENARIO.npc_by_id[npc_id]
    # Lexical-history snapshot — counts per-token over the last N
    # NPC-authored messages on this thread. Empty when the dossier
    # authors no `lexical_budget`. Surfaces in the chat agent's
    # dossier block when usage is at-or-over the per-window cap.
    thread_msgs = session.state.thread_messages.get(thread_id, [])
    deps = ChatDeps(
        npc=npc,
        plot_directive=plot_directive,
        commitments=commitments_snapshot(
            session.state.qualities,
            SCENARIO.quality_defaults,
            SCENARIO.quality_descriptions,
            SCENARIO.surface_to_chat_quality_ids,
        ),
        player_canon=player_canon_snapshot(
            session.state.player_canon,
            current_turn=len(session.state.replies),
        ),
        lexical_history=lexical_history_snapshot(thread_msgs, npc),
        epilogue_digest=epilogue_digest,
        outbound_mode=outbound_mode,
        fact_check_log=fact_check_log,
    )

    with LLM_LOGFIRE.span(
        span_label,
        npc=npc_id,
        thread=thread_id,
        beat=beat_id,
        **(extra_span_attrs or {}),
    ) as span:
        try:
            result = await AGENTS.chat_agent.run(
                user_prompt,
                message_history=message_history,
                deps=deps,
            )
        except Exception as e:
            # Llama hiccup, schema/tool error, or history-processor failure
            # (e.g. /tokenize unreachable). Re-raise as ChatAgentFailure so
            # the SSE stream emits a structured `error` event instead of
            # tearing the connection.
            logfire.exception(
                "render agent run failed",
                npc=npc_id, thread=thread_id, beat=beat_id,
            )
            raise ChatAgentFailure(str(e)) from e
        if append_history:
            new_messages = list(result.new_messages())
            if outbound_mode:
                # Drop the synthetic-trigger user-turn from what gets
                # persisted. ModelRequest carries the user prompt;
                # ModelResponse carries the assistant's reply. We only
                # want the assistant response in thread history so a
                # later reply-mode render doesn't see the trigger as if
                # the player had typed it.
                new_messages = [m for m in new_messages if isinstance(m, ModelResponse)]
            if new_messages:
                session.apply(ThreadHistoryAppended(
                    thread_id=thread_id, messages=new_messages,
                ))
        usage = result.usage()
        span.set_attribute("input_tokens", usage.input_tokens)
        span.set_attribute("output_tokens", usage.output_tokens)
        span.set_attribute("response_preview", result.output.strip()[:200])

    body = result.output.strip()
    session.apply(ChatBodyRendered(
        beat_id=beat_id,
        npc_id=npc_id,
        thread_id=thread_id,
        body=body,
        epilogue=epilogue_digest is not None,
        fact_check=(
            FactCheckTrace(attempts=list(fact_check_log))
            if fact_check_log else None
        ),
    ))
    return Email(
        id=f"email_{uuid.uuid4().hex[:8]}",
        beat_id=beat_id,
        thread_id=thread_id,
        folder="Inbox",
        sender=npc.email_address,
        recipient=resolve_player_email(SCENARIO, session.state),
        body=body,
        sent_at_label="just now",
    )


async def _render_via_chat_agent(
    session: Session,
    *,
    npc_id: str,
    thread_id: str,
    plot_directive: BeatSpec | None,
    beat_id: str | None,
    player_body: str | None = None,
    outbound_mode: bool = False,
    fact_check_log: list[FactCheckAttempt] | None = None,
) -> Email:
    """Render path for chat-turn fallbacks and non-terminal beat fires.
    Modes:

    - reply (`outbound_mode=False`, default): `player_body` is the
      user_prompt; the resulting exchange (player + assistant turn) is
      appended to `state.thread_messages[thread_id]`.
    - outbound (`outbound_mode=True`): the chat agent receives the fixed
      synthetic trigger and `ChatDeps.outbound_mode=True`; the
      OUTBOUND_MODE_FRAMING instructions block tells it to compose an
      unprompted email rather than respond to the trigger. Only the
      assistant turn is persisted (the synthetic trigger isn't a real
      player message), so a subsequent reply-mode render on this thread
      doesn't see the trigger as if the player had typed it.

    Thread history is passed in either mode — an in-flight scene's prior
    turns are valid context for continuation regardless of which side
    drove this fire."""
    if outbound_mode:
        user_prompt = _OUTBOUND_TRIGGER_PROMPT
        span_label = "render outbound to {npc} on {thread}"
    else:
        assert player_body is not None, (
            "reply-mode render requires player_body"
        )
        user_prompt = player_body
        span_label = "render chat to {npc} on {thread}"
    history = list(session.state.thread_messages.get(thread_id, []))
    return await _render_email(
        session,
        npc_id=npc_id,
        thread_id=thread_id,
        user_prompt=user_prompt,
        plot_directive=plot_directive,
        beat_id=beat_id,
        message_history=history,
        append_history=True,
        outbound_mode=outbound_mode,
        fact_check_log=fact_check_log,
        span_label=span_label,
        extra_span_attrs={
            "plot": plot_directive is not None,
            "history_len": len(history),
            "outbound_mode": outbound_mode,
            "prompt_preview": user_prompt[:200],
        },
    )


async def _render_via_epilogue_agent(
    session: Session,
    *,
    npc_id: str,
    thread_id: str,
    player_body: str,
    in_reply_to_email_id: str | None,
    plot_directive: BeatSpec | None,
    beat_id: str | None,
    outbound_mode: bool = False,
    fact_check_log: list[FactCheckAttempt] | None = None,
) -> Email:
    """Render path for terminal beats. Same chat agent as ordinary turns,
    but ChatDeps carries an `epilogue_digest` that flips it into
    closing-email mode (the `_epilogue_framing_block` and
    `_epilogue_digest_block` instruction decorators in `agents.py` only
    activate when the digest is set). No message history — the digest is
    the run summary.

    `outbound_mode=True` is set when an off-thread terminal fires (rare in
    the current scenario, but propagated for correctness): the
    OUTBOUND_MODE_FRAMING composes alongside epilogue framing and the
    user_prompt is swapped to the synthetic trigger."""
    digest = _build_epilogue_digest(session)
    user_prompt = _OUTBOUND_TRIGGER_PROMPT if outbound_mode else player_body
    return await _render_email(
        session,
        npc_id=npc_id,
        thread_id=thread_id,
        user_prompt=user_prompt,
        plot_directive=plot_directive,
        beat_id=beat_id,
        message_history=None,
        append_history=False,
        epilogue_digest=digest,
        outbound_mode=outbound_mode,
        fact_check_log=fact_check_log,
        span_label="render epilogue to {npc} on {thread}",
        extra_span_attrs={
            "digest_preview": digest[:300],
            "outbound_mode": outbound_mode,
        },
    )


def _build_epilogue_digest(session: Session) -> str:
    """Per-run digest passed to the epilogue agent in the user prompt.

    Sections: commitments (delta from defaults), per-axis classifier
    history (most-recent-first), fired beats in order, last 3 player
    replies verbatim. The dossier and plot directive are already injected
    via @instructions; this digest is the run-tailored payload.
    """
    assert SCENARIO is not None
    state = session.state
    # Commitments: filter to surface_to_chat qualities ≠ default.
    commits_dict = commitments_snapshot(
        state.qualities,
        SCENARIO.quality_defaults,
        SCENARIO.quality_descriptions,
        SCENARIO.surface_to_chat_quality_ids,
    )
    commit_lines = [
        f"  {k} = {v!r}" if isinstance(v, str) else f"  {k} = {v:g}"
        for k, (v, _desc) in commits_dict.items()
    ] or ["  (no commitments yet)"]

    history = build_recent_classifier_history(state.replies, axes=None, per_axis=5)
    history_lines = [f"  {axis}: {', '.join(values)}" for axis, values in history.items()]
    history_block = "\n".join(history_lines) or "  (no replies classified)"

    fired_in_order = _fired_beat_ids_in_order(session)
    fired_block = ", ".join(fired_in_order) if fired_in_order else "(none)"

    # most recent last for natural reading order; trim defensively in case of
    # very long player pastes.
    recent_replies = list(state.replies)[-3:]
    reply_block_parts = [f"  - {r.body.strip()[:1000]}" for r in recent_replies]
    reply_block = "\n".join(reply_block_parts) or "  (no replies)"

    return (
        "RUN DIGEST:\n"
        f"Commitments:\n" + "\n".join(commit_lines) + "\n"
        f"Classifier history (last 5 per axis, most-recent-first):\n{history_block}\n"
        f"Fired beats (oldest → newest): {fired_block}\n"
        f"Recent player replies verbatim:\n{reply_block}"
    )


def _count_beat_email_fires(session: Session, beat_id: str) -> int:
    """How many times this beat has rendered an email so far. Used by
    `_fire_beat` for the scene_turn count and by `_select_beat`'s
    scene-continuity short-circuit (any prior fire → continue). Walks
    `state.emails` directly because scenes don't enter `fired_beat_ids`
    until exit_when fires, so the set lookup is incomplete for in-flight
    scenes."""
    return sum(1 for e in session.state.emails if e.beat_id == beat_id)


def _fired_beat_ids_in_order(session: Session, *, last: int | None = None) -> list[str]:
    """Fired-beat ids in emit order — read off Email.beat_id where set,
    since state.fired_beat_ids is a set with no order guarantee. `last`
    slices the tail (e.g. for the Director's recent-fire history)."""
    state = session.state
    ids = [
        e.beat_id for e in state.emails
        if e.beat_id is not None and e.beat_id in state.fired_beat_ids
    ]
    return ids[-last:] if last is not None else ids


# ───────────────────────── scheduler tick ─────────────────────────


async def _select_beat(
    session: Session,
    due: list[Beat],
    *,
    reply_thread_id: str | None = None,
) -> tuple[Beat | None, Literal["none", "sole", "continue", "director"], str | None]:
    """Pick one beat from the due list and report (beat, method, rationale).

    Method is one of:
      - "none"     — no beats due, or Director chose "none"
      - "sole"     — exactly one beat due, auto-selected without an LLM call
      - "continue" — the player replied on a thread where a scene beat with
                     a prior fire is still eligible. The same scene
                     continues, no Director call. Saves one LLM round-trip
                     per scene-continuation tick AND prevents the Director
                     from interrupting a scene mid-flow.
      - "director" — ≥2 due (and no scene-continuity short-circuit fired),
                     the Director made the call (rationale carries
                     its one-sentence justification)

    The rationale is None for "none", "sole", and "continue" cases. The
    tuple shape is consumed by `_tick` to populate the per-tick
    `TickDecision` payload.
    """
    assert AGENTS is not None
    if not due:
        logfire.info("no beats eligible this tick")
        return None, "none", None

    # Scene continuity: if the player replied on a thread where a scene
    # is already in motion, keep firing it without consulting the
    # Director. "In motion" = same-thread + kind=scene + at least one
    # prior fire. Multiple matching scenes on one thread (rare —
    # usually an authoring smell) fall through to the normal selector.
    if reply_thread_id is not None:
        scene_matches = [
            b for b in due
            if b.kind == "scene"
            and b.thread == reply_thread_id
            and _count_beat_email_fires(session, b.id) > 0
        ]
        if len(scene_matches) == 1:
            logfire.info(
                "scene-continuity: continuing {beat_id} on {thread}",
                beat_id=scene_matches[0].id, thread=reply_thread_id,
            )
            return scene_matches[0], "continue", None

    if len(due) == 1:
        logfire.info(
            "single beat eligible — auto-selected {beat_id}",
            beat_id=due[0].id,
        )
        return due[0], "sole", None

    by_id = {b.id: b for b in due}
    user_prompt = build_director_user_prompt(
        qualities=dict(session.state.qualities),
        recent_classifier_history=build_recent_classifier_history(
            session.state.replies, axes=None, per_axis=5,
        ),
        recent_fire_history=_fired_beat_ids_in_order(session, last=6),
        eligible=[
            {"id": b.id, "intent_summary": _intent_summary_for(b)} for b in due
        ],
    )
    output_type = NativeOutput(build_director_output_type([b.id for b in due]))

    with LLM_LOGFIRE.span(
        "director picks among {n} eligible",
        n=len(due),
        eligible=[b.id for b in due],
    ) as span:
        try:
            result = await AGENTS.director_agent.run(user_prompt, output_type=output_type)
        except Exception:
            logfire.exception(
                "director_agent failed; defaulting to first eligible",
                eligible=[b.id for b in due],
            )
            return due[0], "director", "(director failed; fell back to first eligible)"
        choice = result.output
        chosen_id = getattr(choice, "chosen_beat_id", None)
        rationale = getattr(choice, "rationale", "")
        span.set_attribute("chosen_beat_id", chosen_id)
        span.set_attribute("rationale", rationale)
        logfire.info(
            "director picked {chosen}",
            chosen=chosen_id, rationale=rationale,
            eligible=[b.id for b in due],
        )
        if chosen_id == "none":
            return None, "none", rationale
        if chosen_id not in by_id:
            logfire.warn(
                "director returned an id not in eligible set; defaulting to first",
                returned=chosen_id, eligible=[b.id for b in due],
            )
            return due[0], "director", rationale or "(director returned invalid id; fell back to first eligible)"
        return by_id[chosen_id], "director", rationale


def _intent_summary_for(beat: Beat) -> str:
    """One-line summary used in the Director's eligible-beat list.
    Pre-authored beats surface as a preview of the prose; chat-driven
    beats use the first line of `spec.intent` (truncated)."""
    if beat.spec.pre_authored_prose is not None:
        prose_lines = [
            ln for ln in beat.spec.pre_authored_prose.strip().splitlines()
            if ln.strip()
        ]
        first_line = prose_lines[0] if prose_lines else beat.id
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        return f"(pre-authored email: {first_line})"
    first_line = beat.spec.intent.strip().splitlines()[0] if beat.spec.intent.strip() else "(no intent)"
    if len(first_line) > 200:
        first_line = first_line[:197] + "..."
    suffix = " [TERMINAL]" if beat.ends_session else (" [SCENE]" if beat.kind == "scene" else "")
    return f"{first_line}{suffix}"


async def _tick(
    session: Session,
    *,
    player_body: str | None = None,
    in_response_to: str | None = None,
    reply_thread_id: str | None = None,
    classifier_deltas: list[QualityDelta] | None = None,
    canon_writes: list[CanonCandidate] | None = None,
    canon_drops: list[CanonDropRecord] | None = None,
) -> tuple[list[Email], set[str], TickDecision]:
    """Walk the beat scheduler. Returns
    `(newly-emitted emails, threads where a chat-driven beat fired, decision)`.

    The thread-id set lets the caller decide whether a chat-turn fallback
    is still needed. The `TickDecision` bundles every iteration-relevant
    decision the engine made on this tick — eligibility, selection, quality
    deltas, what fired — and is shipped to the in-browser debug overlay
    via the `tick_decision` SSE event AND attached to the tick span as
    queryable Logfire attributes.

    `classifier_deltas`, `canon_writes`, and `canon_drops` carry the
    upstream pre-tick state mutations from `_send_stream` (classifier and
    extractor run before `_tick` so terminals can read post-classifier
    qualities and canon-gated predicates). They're folded into the
    TickDecision payload so the overlay surfaces "what changed this tick"
    in one place across all sources.
    """
    assert SCENARIO is not None
    state = session.state
    new_emails: list[Email] = []
    plot_fired_threads: set[str] = set()
    quality_deltas: list[QualityDelta] = list(classifier_deltas or [])

    with logfire.span("tick (in response to {in_response_to})", in_response_to=in_response_to) as span:
        eligible, eligibility_report = find_due_beats(state, SCENARIO)

        # Retire scenes whose `exit_when` is already satisfied at selection
        # time. Without this, such a scene would fire once just to exit —
        # e.g. when `*calibration_done` flips during the welcome reply
        # (extractor canonizes the player's name) calibration_scene would
        # render one calibration question before the post-fire exit check
        # retires it. The chat-turn fallback in `_send_stream` still covers
        # the reply thread with a free-conversation reply, so retiring
        # without firing doesn't leave the player on silence.
        remaining: list[Beat] = []
        for beat in eligible:
            if beat.kind == "scene" and scene_exit_satisfied(beat, state):
                session.apply(BeatFiredEvent(
                    beat_id=beat.id, beat_kind="scene",
                    scene_turn=None, completed=True,
                ))
                logfire.info(
                    "retire scene {beat_id} (exit_when satisfied on entry)",
                    beat_id=beat.id,
                )
                for be in eligibility_report:
                    if be.beat_id == beat.id and be.status == "eligible":
                        be.status = "retired"
                        break
            else:
                remaining.append(beat)
        eligible = remaining

        # Defer in-flight off-thread scenes. A scene that has already had
        # at least one fire is "in motion" — and once in motion, advancing
        # it requires the player to actually engage on its own thread.
        # Without this filter, a scene whose `exit_when` keys off
        # `player_replies_to_thread_gte <its-thread>` would re-fire on every
        # consecutive same-thread tick on a different thread (the Director
        # picking it cross-thread because no same-thread scene-continuity
        # match exists), monologuing while the player is mid-conversation
        # elsewhere. First fires are still allowed cross-thread so an NPC
        # can introduce themselves on a tick driven by another NPC's reply.
        if reply_thread_id is not None:
            # Single linear scan of state.emails to count fires per beat,
            # then O(1) per-beat lookups in the filter loop. Without this,
            # the loop would scan state.emails once per eligible beat.
            fire_count: dict[str, int] = {}
            for em in state.emails:
                if em.beat_id is not None:
                    fire_count[em.beat_id] = fire_count.get(em.beat_id, 0) + 1
            report_by_id = {be.beat_id: be for be in eligibility_report}
            after_defer: list[Beat] = []
            for beat in eligible:
                if (
                    beat.kind == "scene"
                    and beat.thread != reply_thread_id
                    and fire_count.get(beat.id, 0) > 0
                ):
                    note = f"awaiting reply on {beat.thread}"
                    logfire.info(
                        "defer in-flight scene {beat_id}: {reason}",
                        beat_id=beat.id, reason=note,
                    )
                    be = report_by_id.get(beat.id)
                    if be is not None and be.status == "eligible":
                        be.status = "deferred"
                        be.blocked_by = note
                else:
                    after_defer.append(beat)
            eligible = after_defer

        # One log line per value-blocked beat, so the "why didn't this fire?"
        # answer reads inline in the terminal alongside selection + writes.
        # Beats blocked on a structural prereq (BeatFired not yet) are
        # already filtered out of the report by find_due_beats.
        for be in eligibility_report:
            if be.status == "blocked" and be.blocked_by:
                logfire.info(
                    "beat {beat_id} blocked: {reason}",
                    beat_id=be.beat_id, reason=be.blocked_by,
                )

        chosen, selection, rationale = await _select_beat(
            session, eligible, reply_thread_id=reply_thread_id,
        )
        chosen_beat_id = chosen.id if chosen is not None else None
        # Record the Director's decision (or short-circuit) as an event for
        # the audit log; fold is a no-op for this observation event.
        session.apply(DirectorChose(
            selection_method=selection,
            chosen_beat_id=chosen_beat_id,
            rationale=rationale,
            eligible=[b.id for b in eligible],
        ))
        fired_beat_id: str | None = None
        scene_turn: int | None = None
        fact_check: FactCheckTrace | None = None
        if chosen is not None:
            e, write_deltas, turn, trace = await _fire_beat(
                session, chosen,
                player_body=player_body,
                in_response_to=in_response_to,
                reply_thread_id=reply_thread_id,
            )
            quality_deltas.extend(write_deltas)
            if e is not None:
                new_emails.append(e)
                fired_beat_id = chosen.id
                scene_turn = turn
                fact_check = trace
                # Only chat-driven (non-pre-authored) fires count as covering
                # the thread for the chat-turn fallback purposes.
                if chosen.spec.pre_authored_prose is None:
                    plot_fired_threads.add(chosen.thread)

        decision = TickDecision(
            in_response_to=in_response_to,
            eligible=eligibility_report,
            selection=selection,
            chosen_beat_id=chosen_beat_id,
            chosen_rationale=rationale,
            quality_deltas=quality_deltas,
            fired_beat_id=fired_beat_id,
            scene_turn=scene_turn,
            canon_writes=list(canon_writes or []),
            canon_drops=list(canon_drops or []),
            player_canon={
                k: {
                    "summary": v.summary,
                    "literal_text_quote": v.literal_text_quote,
                    "salience": v.salience,
                    "category": v.category.value,
                    "turn_of_origin": v.turn_of_origin,
                }
                for k, v in state.player_canon.items()
            },
            fact_check=fact_check,
            # Per-axis history snapshot — most-recent-first values per axis,
            # capped at 10 (matches roughly two screens of pill chips).
            classifier_history=build_recent_classifier_history(
                state.replies, axes=None, per_axis=10,
            ),
            # Snapshot AFTER the fire — so the graph view and post-load
            # recovery see the up-to-date fired set without per-tick
            # accumulation logic on the frontend.
            fired_beat_ids=sorted(state.fired_beat_ids),
            # Full per-beat predicate trace against post-tick state. Drives
            # the debug Beats inspector. Cost is O(beats × atoms); ~12
            # beats in production scenario, trivial.
            beats_evaluation=evaluate_all_beats(state, SCENARIO),
        )
        # Span attributes give Logfire UI users a queryable overview without
        # parsing message strings. The full decision lands as `decision`;
        # individual rollups land alongside for filtering ("ticks with
        # quality_delta_count > 0", etc.).
        span.set_attribute("decision", decision.model_dump())
        span.set_attribute("eligible_count", len(eligible))
        span.set_attribute("selection", selection)
        span.set_attribute("fired_beat_id", fired_beat_id)
        span.set_attribute("quality_delta_count", len(quality_deltas))
        span.set_attribute("new_email_count", len(new_emails))
        return new_emails, plot_fired_threads, decision


# ───────────────────────── endpoints ─────────────────────────


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "scenario_loaded": SCENARIO is not None,
        "state_initialised": SESSIONS.get(DEFAULT_SESSION_ID) is not None,
    }


@app.get("/scenario")
async def get_scenario() -> dict:
    if SCENARIO is None:
        raise HTTPException(503, "scenario not loaded")
    # Per-NPC payload includes the canonical thread (each NPC owns at most
    # one in current scenarios; takes the first matching thread). The
    # frontend uses this to render multiple chat rows dynamically without
    # any per-id conditionals.
    npcs_payload: list[dict] = []
    for npc in SCENARIO.npcs:
        thread_id = next(
            (t.id for t in SCENARIO.threads if t.npc == npc.id),
            None,
        )
        npcs_payload.append({
            "id": npc.id,
            "display_name": npc.display_name,
            "email_address": npc.email_address,
            "thread_id": thread_id,
            "replies_enabled": npc.replies_enabled,
        })
    return {
        "title": SCENARIO.title,
        "premise": SCENARIO.premise,
        "npcs": npcs_payload,
    }


@app.get("/state")
async def get_state(session: Session = Depends(get_session)) -> dict:
    return session.state.public_view()


@app.get("/scenario_meta")
async def get_scenario_meta() -> dict:
    """Static scenario metadata used by the DEV-only debug overlay.

    Returns three sections:
      - `qualities`: per-quality {description, type, min, max, default,
        surface_to_chat, allowed_values}. Drives hover tooltips on quality
        names anywhere they appear in the panel.
      - `beats`: per-beat metadata (npc, thread, flags, intent/prose
        summary). Drives hover tooltips on beat ids and the scenario-graph
        node detail panel.
      - `beat_graph`: nodes (beat ids in author order) + edges. Edges
        come in two kinds — `structural` (positive BeatFired prereq:
        upstream MUST fire first) and `mutex` (Not_(BeatFired): upstream
        firing BLOCKS this one). Frontend renders both with different
        glyphs.

    Always-on (not DEV-gated server-side) since the data is harmless and
    the production frontend doesn't hit it. Computed per request — the
    scenario has <30 beats so caching is overkill.
    """
    if SCENARIO is None:
        raise HTTPException(503, "scenario not loaded")
    qualities_payload = {
        q.id: {
            "description": q.description,
            "type": q.type,
            "min": q.min,
            "max": q.max,
            "default": q.default,
            "surface_to_chat": q.surface_to_chat,
            "allowed_values": q.allowed_values,
        }
        for q in SCENARIO.qualities
    }
    beats_payload = {b.id: _beat_meta(b) for b in SCENARIO.beats}
    return {
        "qualities": qualities_payload,
        "beats": beats_payload,
        "beat_graph": _build_beat_graph(SCENARIO),
    }


_DEBUG_SCENARIO_CACHE: dict | None = None


@app.get("/debug/scenario")
async def get_debug_scenario() -> dict:
    """Per-beat predicate AST + authored effects, for the DEV debug Beats
    inspector. Pairs with `TickDecision.beats_evaluation` (which carries
    the per-tick pass/fail trace per atom) — this endpoint ships the
    structural shape, the trace ships the verdicts.

    Returned once at boot; static for the session (scenario YAML doesn't
    hot-reload). Per-beat shape:

      {
        id, kind, npc, thread, ends_session,
        preconditions: list[Predicate],   # the entry_when AST
        exit_when:     list[Predicate],   # scenes only; empty for one_shots
        effects:       list[Effect],      # authored mutations
      }

    Cached after first request — SCENARIO is a process-level singleton, so
    serializing the same Pydantic models on every page reload is wasted
    work. Always-on (not DEV-gated server-side); production frontend
    doesn't hit it.
    """
    global _DEBUG_SCENARIO_CACHE
    if SCENARIO is None:
        raise HTTPException(503, "scenario not loaded")
    if _DEBUG_SCENARIO_CACHE is None:
        _DEBUG_SCENARIO_CACHE = {
            "beats": {
                b.id: {
                    "id": b.id,
                    "kind": b.kind,
                    "npc": b.npc,
                    "thread": b.thread,
                    "ends_session": b.ends_session,
                    "preconditions": [p.model_dump() for p in b.preconditions],
                    "exit_when": [p.model_dump() for p in b.spec.exit_when],
                    "effects": [e.model_dump() for e in b.spec.effects],
                }
                for b in SCENARIO.beats
            }
        }
    return _DEBUG_SCENARIO_CACHE


def _state_debug_view(state: InboxState) -> dict:
    """Richer-than-public_view snapshot used by the debug timeline. Includes
    qualities, fired_beat_ids, player_canon, all emails (full body), end
    flags, and unread counts. Per-tick decisions are NOT included — the
    timeline reconstructs them by replaying the corresponding event slice.
    """
    return {
        "ended": state.ended,
        "end_reason": state.end_reason,
        "qualities": dict(state.qualities),
        "fired_beat_ids": sorted(state.fired_beat_ids),
        "player_canon": {
            k: {
                "summary": v.summary,
                "literal_text_quote": v.literal_text_quote,
                "salience": v.salience,
                "category": v.category.value,
                "turn_of_origin": v.turn_of_origin,
            }
            for k, v in state.player_canon.items()
        },
        "emails": [_email_full(e) for e in state.emails],
        "reply_count": len(state.replies),
        "unread_counts": state._unread_counts(),
        "unread_counts_by_thread": state._unread_counts_by_thread(),
    }


@app.get("/debug/session/events")
async def get_debug_session_events(session: Session = Depends(get_session)) -> dict:
    """Full append-only event log for the active session. Drives the
    DEV-only debug Timeline tab. Read-only; `/send` always applies at head.
    Each entry is the JSON-serialized Event with its `kind` discriminator.
    """
    from pydantic import TypeAdapter
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    return {
        "session_id": session.id,
        "event_count": len(session.event_log),
        "events": [adapter.dump_python(ev, mode="json") for ev in session.event_log],
    }


@app.get("/debug/session/state_at")
async def get_debug_session_state_at(
    index: int,
    session: Session = Depends(get_session),
) -> dict:
    """Re-fold the first `index` events on a fresh state and return the
    snapshot. `index=0` returns the freshly-bootstrapped state (qualities
    at scenario defaults, no emails); `index=len(event_log)` matches the
    live state. Read-only — does not affect the session."""
    try:
        snapshot = session.state_at(index)
    except IndexError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "session_id": session.id,
        "index": index,
        "event_count": len(session.event_log),
        "state": _state_debug_view(snapshot),
    }


@app.post("/debug/session/save")
async def post_debug_session_save(
    path: str,
    session: Session = Depends(get_session),
) -> dict:
    """Dump the active session's full event log to a JSONL file at `path`.
    One event per line, JSON-encoded. Overwrites any existing file at the
    target. Independent of `Settings.session_save_path` autosave — this is
    a one-shot snapshot for sharing or backing up.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    from pydantic import TypeAdapter
    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    with target.open("w", encoding="utf-8") as f:
        for ev in session.event_log:
            f.write(adapter.dump_json(ev).decode("utf-8"))
            f.write("\n")
    return {
        "path": str(target),
        "event_count": len(session.event_log),
        "session_id": session.id,
    }


@app.post("/debug/session/load")
async def post_debug_session_load(
    path: str,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict:
    """Load a saved JSONL into a fresh session, replacing the entry under
    `session_id` (defaults to the active session). Bypasses the autosave
    hook during the replay so the load doesn't double-write, then re-arms
    the hook for subsequent events if `session_save_path` is configured.
    """
    assert SCENARIO is not None and AGENTS is not None
    src = Path(path)
    if not src.exists():
        raise HTTPException(404, f"no such file: {path}")
    events = load_event_log(src)

    persist = None
    if (
        SETTINGS is not None
        and SETTINGS.session_save_path is not None
        and session_id == DEFAULT_SESSION_ID
    ):
        persist = make_jsonl_persistor(SETTINGS.session_save_path)

    session = make_session(SCENARIO, AGENTS, persist=persist)
    replay_into_session(session, events)
    SESSIONS[session_id] = session
    return {
        "session_id": session.id,
        "event_count": len(session.event_log),
        "loaded_from": str(src),
    }


def _beat_meta(beat: Beat) -> dict:
    """One beat's meta payload for /scenario_meta. Same `kind` flags as the
    Director's intent_summary uses — frontend chooses icons from these."""
    if beat.spec.pre_authored_prose is not None:
        # First non-empty line, truncated; matches the Director's preview.
        prose_lines = [
            ln for ln in beat.spec.pre_authored_prose.strip().splitlines()
            if ln.strip()
        ]
        summary = (prose_lines[0] if prose_lines else "")[:160]
    else:
        first_intent_line = beat.spec.intent.strip().splitlines()[0] if beat.spec.intent.strip() else ""
        summary = first_intent_line[:160]
    return {
        "id": beat.id,
        "npc": beat.npc,
        "thread": beat.thread,
        "is_scene": beat.kind == "scene",
        "ends_session": beat.ends_session,
        "pre_authored": beat.spec.pre_authored_prose is not None,
        "fact_checked": bool(beat.spec.must_include_facts or beat.spec.must_not),
        "summary": summary,
    }


def _build_beat_graph(scenario: Scenario) -> dict:
    """Build the beat adjacency for the scenario-graph view.

    Walk each beat's preconditions and emit one edge per `BeatFired`
    reference encountered. Positive form (`BeatFired` directly under
    preconditions / inside `Or_`) is structural; `Not_(BeatFired ...)`
    is the mutex idiom — same beat reference, different visual glyph.

    Recurses through `Or_`/`Not_` so deeply-nested gates get every
    underlying BeatFired surfaced. Edges may duplicate if the same
    upstream is referenced more than once on a beat; the frontend
    de-duplicates by (from, to, kind) at render.
    """
    nodes = [
        {"id": b.id, "thread": b.thread, "npc": b.npc}
        for b in scenario.beats
    ]
    edges: list[dict] = []

    def walk(pred: Predicate, target: str, in_negation: bool) -> None:
        if isinstance(pred, BeatFired):
            kind = "mutex" if in_negation else "structural"
            edges.append({"from": pred.beat, "to": target, "kind": kind})
            return
        if isinstance(pred, Not_):
            walk(pred.of, target, not in_negation)
            return
        if isinstance(pred, Or_):
            for sub in pred.of:
                walk(sub, target, in_negation)
            return
        # Other predicate kinds reference qualities / threads / canon — not
        # beat-to-beat edges. Ignored at the graph layer.

    for beat in scenario.beats:
        for pred in beat.preconditions:
            walk(pred, beat.id, in_negation=False)

    return {"nodes": nodes, "edges": edges}


@app.get("/inbox")
async def get_inbox(session: Session = Depends(get_session)) -> dict:
    """Folder summary. The client uses this for the inbox list view."""
    state = session.state
    folders: dict[str, list[dict]] = {"Inbox": [], "Sent": []}
    for e in state.emails:
        folders.setdefault(e.folder, []).append(_email_summary(e))
    # Newest first within each folder. Append order is emit order; reverse it.
    for k in folders:
        folders[k].reverse()
    return {"folders": folders, "state": state.public_view()}


@app.get("/thread/{thread_id}")
async def get_thread(thread_id: str, session: Session = Depends(get_session)) -> dict:
    """All emails in a thread, in emit order."""
    if SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if thread_id not in SCENARIO.thread_by_id:
        raise HTTPException(404, f"unknown thread: {thread_id}")
    emails = [_email_full(e) for e in session.state.emails if e.thread_id == thread_id]
    return {"thread_id": thread_id, "emails": emails}


@app.post("/open")
async def post_open(req: OpenRequest, session: Session = Depends(get_session)) -> dict:
    """Mark email read. /open does not mutate qualities — reading isn't a
    communication act, so it can't move engagement or fire any beat."""
    if SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if session.state.ended:
        raise HTTPException(409, f"session ended ({session.state.end_reason})")

    async with session.lock:
        email = next((e for e in session.state.emails if e.id == req.email_id), None)
        if email is None:
            raise HTTPException(404, f"unknown email_id: {req.email_id}")
        session.apply(EmailRead(email_id=req.email_id))
        return {
            "state": session.state.public_view(),
            "new_emails": [],
        }


@app.post("/send")
async def post_send(req: SendRequest, session: Session = Depends(get_session)) -> StreamingResponse:
    if AGENTS is None or SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if session.state.ended:
        raise HTTPException(409, f"session ended ({session.state.end_reason})")
    if not req.body.strip():
        raise HTTPException(400, "body is required")
    return StreamingResponse(
        _send_stream(session, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@dataclass
class RecordedSend:
    """The state-recorded result of a player /send before any LLM work runs.

    Bridges `record_send` → `process_send` so the FastAPI layer can emit the
    early `send_recorded` SSE event before the (slow) classifier+tick
    pipeline starts. The playtester calls the same two functions back-to-back
    and ignores the gap.
    """
    req: SendRequest
    body: str
    send_email: Email
    reply: Reply
    thread_id: str | None
    in_reply_email: Email | None
    recipient_npc_id: str | None
    replied_beat: Beat | None


@dataclass
class SendResult:
    """Full output of `process_send`. The FastAPI layer emits SSE events
    from this shape; the playtester reads it directly.

    `process_send` may raise `ChatAgentFailure` after state has already
    been mutated (player email, classifier writes, canon writes, tick).
    Callers catch and decide how to surface; SendResult is only returned
    on success.
    """
    classifier_results: list[AxisResult] | None
    classifier_deltas: list[QualityDelta]
    canon_writes: list[CanonCandidate]
    canon_drops: list[CanonDropRecord]
    new_emails: list[Email]
    plot_fired_threads: set[str]
    tick_decision: TickDecision


def record_send(session: Session, req: SendRequest) -> RecordedSend:
    """Append the player's email + reply to session.state via a SendRecorded
    event and bundle the per-send derived fields (thread, recipient NPC,
    replied beat) into a RecordedSend.

    Synchronous and fast — no LLM, no tick. Caller is expected to hold
    `session.lock` if running concurrently with other mutators (the FastAPI
    /send path does; the playtester does not need to).
    """
    assert SCENARIO is not None
    body = req.body.strip()
    in_reply_email = (
        next((e for e in session.state.emails if e.id == req.in_reply_to), None)
        if req.in_reply_to else None
    )
    thread_id = in_reply_email.thread_id if in_reply_email else None
    send_email = Email(
        id=f"email_{uuid.uuid4().hex[:8]}",
        beat_id=None,
        thread_id=thread_id,
        folder="Sent",
        sender="you",
        recipient=req.recipient,
        body=body,
        sent_at_label="just now",
        is_read=True,
    )
    reply = Reply(
        id=f"reply_{uuid.uuid4().hex[:8]}",
        thread_id=thread_id,
        beat_responded_to=req.in_reply_to,
        recipient=req.recipient,
        body=body,
    )
    session.apply(SendRecorded(email=send_email, reply=reply))
    recipient_npc = req.recipient_npc_id or _resolve_recipient_npc_id(req.recipient)
    replied_beat = (
        SCENARIO.beat_by_id.get(in_reply_email.beat_id)
        if in_reply_email and in_reply_email.beat_id else None
    )
    return RecordedSend(
        req=req,
        body=body,
        send_email=send_email,
        reply=reply,
        thread_id=thread_id,
        in_reply_email=in_reply_email,
        recipient_npc_id=recipient_npc,
        replied_beat=replied_beat,
    )


async def process_send(session: Session, rec: RecordedSend) -> SendResult:
    """Run the classifier + extractor in parallel, fold their writes into
    session.state via events, walk one tick, and (if no chat-driven beat
    covered the thread) fall back to a chat-agent reply. Returns a
    `SendResult` on success; raises `ChatAgentFailure` if the chat-agent
    fallback fails.

    Classifier and extractor run in parallel via asyncio.gather (latency-
    neutral). Both must complete before `_tick` because `find_due_beats`
    reads state mutated by both: classifier writes qualities (gates
    terminal decisions) and extractor writes player_canon
    (PlayerCanonExists predicates gate calibration).
    """
    assert SCENARIO is not None and AGENTS is not None
    state = session.state

    # Classify-typed effects come from two sources: the beat the player
    # just replied to (per-beat axes) and scenario.effects (always-on,
    # gated by their own `when:`). Both pour into one batched classifier
    # call. Beat-scope effects come first so a scenario-scope effect on
    # the same quality (rare, probably an authoring mistake) doesn't
    # shadow the beat-scope verdict.
    beat_classify_effects: list[Effect] = (
        [e for e in rec.replied_beat.spec.effects if e.is_classify]
        if rec.replied_beat else []
    )
    scenario_classify_effects: list[Effect] = [
        e for e in SCENARIO.effects
        if e.is_classify
        and (e.when is None or evaluate_predicate(e.when, state))
        and (e.thread is None or e.thread == rec.thread_id)
    ]
    classify_effects = beat_classify_effects + scenario_classify_effects
    recipient_blurb = _persona_blurb_for_recipient(rec.req.recipient, rec.recipient_npc_id)
    prior_body = rec.in_reply_email.body if rec.in_reply_email else ""

    classifier_coro = _classify_player_reply(
        recipient_blurb=recipient_blurb,
        prior_email_body=prior_body,
        reply_body=rec.body,
        effects=classify_effects,
    ) if classify_effects else _none_coro()
    extractor_coro = _extract_player_canon(
        recipient_blurb=recipient_blurb,
        prior_email_body=prior_body,
        reply_body=rec.body,
    )
    classifier_results, extractor_output = await asyncio.gather(
        classifier_coro, extractor_coro,
    )

    classifier_deltas: list[QualityDelta] = []
    if classify_effects and classifier_results is not None:
        session.apply(ClassifierObserved(
            reply_id=rec.reply.id, verdicts=classifier_results,
        ))
        _apply_classify_results(
            session,
            effects=classify_effects,
            results=classifier_results,
            replied_beat_id=rec.replied_beat.id if rec.replied_beat else None,
            reply=rec.reply,
            out_deltas=classifier_deltas,
        )

    canon_writes: list[CanonCandidate] = []
    canon_drops: list[CanonDropRecord] = []
    if extractor_output is not None and extractor_output.candidates:
        # Categorize every candidate so the debug overlay shows BOTH what
        # was promoted AND what the extractor proposed but the engine
        # rejected. Two rejection reasons:
        #   - anchor_fail: literal_text_quote isn't a verbatim substring
        #     of the reply (anti-hallucination floor)
        #   - below_salience: anchored, but extractor confidence below
        #     the promotion threshold
        # The salience floor lives in derive_canon_promoted_events too,
        # so promoted candidates ALWAYS clear it; we categorize here so
        # canon_drops carries the dropped ones for surfacing.
        promoted: list[CanonCandidate] = []
        for c in extractor_output.candidates:
            if c.literal_text_quote not in rec.body:
                canon_drops.append(CanonDropRecord(candidate=c, reason="anchor_fail"))
            elif c.salience < CANON_SALIENCE_THRESHOLD:
                canon_drops.append(CanonDropRecord(candidate=c, reason="below_salience"))
            else:
                promoted.append(c)
        session.apply(ExtractorObserved(
            reply_id=rec.reply.id,
            candidates=list(extractor_output.candidates),
            drops=list(canon_drops),
        ))
        if canon_drops:
            logfire.warn(
                "extractor candidates dropped",
                anchor_fail=sum(1 for d in canon_drops if d.reason == "anchor_fail"),
                below_salience=sum(1 for d in canon_drops if d.reason == "below_salience"),
                kept=len(promoted),
            )
        if promoted:
            for ev in derive_canon_promoted_events(
                promoted, turn_of_origin=len(state.replies),
            ):
                session.apply(ev)
            canon_writes = promoted

    new_emails, plot_fired_threads, decision = await _tick(
        session,
        player_body=rec.body,
        in_response_to=rec.send_email.id,
        reply_thread_id=rec.thread_id,
        classifier_deltas=classifier_deltas,
        canon_writes=canon_writes,
        canon_drops=canon_drops,
    )

    # Chat-turn fallback fires when no chat-driven beat covered this
    # thread. With scene-as-primitive, scenes that match the player's
    # reply thread short-circuit the Director and fire here; this branch
    # only matters for threads without a covering scene OR plot beat this
    # tick.
    if (
        rec.recipient_npc_id is not None
        and rec.thread_id is not None
        and rec.thread_id not in plot_fired_threads
    ):
        chat_email = await _render_via_chat_agent(
            session,
            npc_id=rec.recipient_npc_id,
            thread_id=rec.thread_id,
            plot_directive=None,
            beat_id=None,
            player_body=rec.body,
        )
        session.apply(EmailEmitted(email=chat_email))
        new_emails.append(chat_email)

    return SendResult(
        classifier_results=classifier_results,
        classifier_deltas=classifier_deltas,
        canon_writes=canon_writes,
        canon_drops=canon_drops,
        new_emails=new_emails,
        plot_fired_threads=plot_fired_threads,
        tick_decision=decision,
    )


async def _send_stream(session: Session, req: SendRequest) -> AsyncIterator[bytes]:
    assert SCENARIO is not None and AGENTS is not None
    state = session.state
    # Pre-compute thread_id for the span name. record_send recomputes the
    # same value (and re-resolves in_reply_email) once inside the lock —
    # cheap dict-lookup-shaped work; preferable to plumbing it back out.
    thread_id = (
        next((e.thread_id for e in state.emails if e.id == req.in_reply_to), None)
        if req.in_reply_to else None
    )

    with logfire.span("send to {recipient} on {thread_id}", recipient=req.recipient, thread_id=thread_id):
        async with session.lock:
            rec = record_send(session, req)
            yield _sse({"type": "send_recorded", "email": _email_full(rec.send_email)})
            try:
                result = await process_send(session, rec)
            except ChatAgentFailure as err:
                # Player email and any classifier write are already persisted
                # — surface a structured error so the client can display it
                # without losing the player's send. State + done still flush.
                logfire.warn(
                    "send_stream aborted: chat agent failed",
                    thread_id=rec.thread_id, message=str(err),
                )
                yield _sse({
                    "type": "error",
                    "message": (
                        "Aria couldn't compose a reply just now. "
                        "Please try again."
                    ),
                })
                yield _sse({"type": "state", "state": state.public_view()})
                yield _sse({"type": "done"})
                return

            for ne in result.new_emails:
                yield _sse({"type": "new_email", "email": _email_full(ne)})

            yield _sse({"type": "tick_decision", "decision": result.tick_decision.model_dump()})
            yield _sse({"type": "state", "state": state.public_view()})
            yield _sse({"type": "done"})


async def _none_coro():
    """Awaitable yielding None — paired with `_extract_player_canon` in
    `asyncio.gather` so we can run the extractor solo on beats that
    authored no classify-typed effects."""
    return None


async def _extract_player_canon(
    *,
    recipient_blurb: str,
    prior_email_body: str,
    reply_body: str,
) -> ExtractorOutput | None:
    """Run the extractor agent against the player's reply. Returns the
    structured output on success, or None on failure (the call is
    fail-soft — canon stays unchanged on a failed extraction; the scenario
    continues without missing data).

    Wrapped in an llm-tagged span so the span row in the trace tree
    carries reply preview + the count of extracted candidates without
    requiring a drill-into.
    """
    assert AGENTS is not None
    with LLM_LOGFIRE.span(
        "extract player canon",
        reply_preview=reply_body[:200],
    ) as span:
        try:
            result = await AGENTS.extractor_agent.run(
                build_extractor_user_prompt(
                    recipient_blurb, prior_email_body, reply_body,
                )
            )
        except Exception:
            logfire.exception("extractor failed; player_canon unchanged")
            return None
        output = result.output
        span.set_attribute("candidate_count", len(output.candidates))
        span.set_attribute("candidate_keys", [c.key for c in output.candidates])
        return output


def _slugify_name(name: str) -> str:
    """Email local-part slug from a free-form display name.

    Rules: lowercase; whitespace runs become a single dot; non-alphanumeric
    (and non-dot) characters dropped. So `"Marcus Chen"` → `"marcus.chen"`,
    `"O'Brien"` → `"obrien"`, `"Marcus L. Chen"` → `"marcus.l.chen"`.
    Empty string in → empty string out (caller should fall back).
    """
    out: list[str] = []
    in_ws = False
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
            in_ws = False
        elif ch.isspace():
            if not in_ws and out and out[-1] != ".":
                out.append(".")
            in_ws = True
        else:
            in_ws = False
    while out and out[-1] == ".":
        out.pop()
    return "".join(out)


def resolve_player_email(scenario: Scenario, state: InboxState) -> str:
    """Recipient address used for every NPC→player email.

    Resolution order:
    1. If the player has a canonized name in `state.player_canon["name"]`,
       slugify it and use `<slug>@<email_domain>`. This is the steady state
       once calibration has run.
    2. Else if the scenario authored an explicit `player_email`, use it as
       a back-compat override (test fixtures, legacy scenarios).
    3. Else fall back to `<fallback_player_local>@<email_domain>` (the
       pre-canonization default — used during the welcome and any time
       calibration cap-exited without a name).

    Slugification of an empty name string drops back to the fallback.
    """
    name_entry = state.player_canon.get(CanonCategory.NAME.value)
    if name_entry is not None:
        slug = _slugify_name(name_entry.summary)
        if slug:
            return f"{slug}@{scenario.email_domain}"
    if scenario.player_email:
        return scenario.player_email
    return f"{scenario.fallback_player_local}@{scenario.email_domain}"


async def _classify_player_reply(
    *,
    recipient_blurb: str,
    prior_email_body: str,
    reply_body: str,
    effects: list[Effect],
) -> list[AxisResult] | None:
    """Run the classifier against the player's reply across all classify-typed
    effects in a single batched call. Returns None on failure so the caller
    skips every quality write.

    Wrapped in an llm-tagged span so the auto-instrumented Pydantic AI run
    has a business-context parent — and so `reply_preview` and the per-quality
    targets ride on the span row in the trace tree, no drilling.
    """
    assert AGENTS is not None
    with LLM_LOGFIRE.span(
        "classify reply for {qualities}",
        qualities=[e.quality for e in effects],
        reply_preview=reply_body[:200],
    ) as span:
        try:
            result = await AGENTS.classifier_agent.run(
                build_classifier_user_prompt(
                    recipient_blurb, prior_email_body, reply_body, effects,
                )
            )
        except Exception:
            logfire.exception("classifier failed; qualities unchanged")
            return None
        results = result.output
        span.set_attribute("classifier_results", [
            {"quality": r.quality, "value": r.value} for r in results
        ])
        return results


def _apply_classify_results(
    session: Session,
    *,
    effects: list[Effect],
    results: list[AxisResult],
    replied_beat_id: str | None,
    reply: Reply,
    out_deltas: list[QualityDelta],
) -> None:
    """Apply each classify effect in order against the classifier's
    returned results. Defense-in-depth: single pass over `results` to
    categorize, then a pass over `effects` to apply:

    - Unknown quality (not targeted by any effect in scope) → logfire-warn, drop.
    - Duplicate (LLM returned multiple results for the same quality)
      → keep the first, logfire-warn the rest.
    - Missing (LLM omitted an authored quality) → logfire-warn, skip.
    - Off-spec value (not in `allowed_values`) → derive_classify_effect_event
      returns None, logfire-warn, no append onto Reply.

    Each effect produces a ClassifyEffectApplied event, which the fold
    appends to reply.classifier_results AND (if delta is set) writes to
    state.qualities. `out_deltas` collects the deltas so `_tick` can fold
    them into `TickDecision.quality_deltas`.
    """
    assert SCENARIO is not None
    target_qualities = {e.quality for e in effects}
    by_quality: dict[str, AxisResult] = {}
    for r in results:
        if r.quality not in target_qualities:
            logfire.warn(
                "classifier returned unauthored quality; dropped",
                beat_id=replied_beat_id, quality=r.quality, value=r.value,
            )
            continue
        if r.quality in by_quality:
            logfire.warn(
                "classifier returned duplicate quality result; first kept",
                beat_id=replied_beat_id, quality=r.quality,
            )
            continue
        by_quality[r.quality] = r

    for effect in effects:
        r = by_quality.get(effect.quality)
        if r is None:
            logfire.warn(
                "classifier missing quality result; quality unchanged",
                beat_id=replied_beat_id, quality=effect.quality,
            )
            continue
        ev = derive_classify_effect_event(
            session.state, SCENARIO, effect, r.value,
            reply_id=reply.id, rationale=r.rationale,
        )
        if ev is None:
            assert effect.classify is not None
            logfire.warn(
                "classifier returned off-spec value; quality unchanged",
                beat_id=replied_beat_id, quality=effect.quality,
                value=r.value, allowed=effect.classify.allowed_values,
            )
            continue
        session.apply(ev)
        if ev.delta is not None:
            out_deltas.append(ev.delta)


def _persona_blurb_for_recipient(recipient: str, npc_id: str | None) -> str:
    if SCENARIO is None or npc_id is None:
        return f"a recipient: {recipient}"
    n = SCENARIO.npc_by_id[npc_id]
    return f"{n.display_name} — {n.relationship_to_owner}"


def _resolve_recipient_npc_id(recipient: str) -> str | None:
    """Fallback when the client doesn't pass `recipient_npc_id` on /send —
    fuzzy-match the recipient string against dossier display name / id.
    Newer clients should always pass the id directly."""
    if SCENARIO is None:
        return None
    needle = recipient.lower()
    for n in SCENARIO.npcs:
        if n.display_name.lower() in needle or n.id in needle:
            return n.id
    return None


async def reset_session(session_id: str = DEFAULT_SESSION_ID) -> Session:
    """Build a fresh Session for `session_id` (replacing any prior entry),
    then fire any beats already due — pre-authored beats with empty
    preconditions land here on the first tick (e.g. aria.welcome).

    When `Settings.session_save_path` is set:
    - For the default session: if the JSONL file exists, REPLAY it instead
      of running the boot tick. New events written subsequently are
      autosaved to the same file. This is how a sidecar restart resumes
      mid-game.
    - For non-default sessions: autosave is NOT enabled (the path is a
      single-session file).

    Used at lifespan boot AND by the playtest harness to clear state between
    synthetic sessions. The TickDecision from boot is discarded — there's
    no SSE consumer at boot, and the playtester treats boot as setup, not
    a measured step.

    Returns the fresh Session for the caller's convenience.
    """
    assert SCENARIO is not None and AGENTS is not None
    persist = None
    save_path: Path | None = (
        SETTINGS.session_save_path
        if SETTINGS is not None and session_id == DEFAULT_SESSION_ID
        else None
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        persist = make_jsonl_persistor(save_path)

    session = make_session(SCENARIO, AGENTS, persist=persist)
    SESSIONS[session_id] = session

    if save_path is not None and save_path.exists() and save_path.stat().st_size > 0:
        # Resume: replay the saved log instead of booting fresh. The
        # replay bypasses the persist hook so we don't double-write.
        events = load_event_log(save_path)
        logfire.info(
            "resuming session from JSONL",
            path=str(save_path), events=len(events),
        )
        replay_into_session(session, events)
        return session

    # Boot tick fires pre-authored beats with empty preconditions. Chat-
    # driven beats are auto-pended too but require a player /send to satisfy
    # `player_replies_to_thread_gte`, so they can't fire here.
    await _tick(session, player_body=None, in_response_to=None)
    return session


# ───────────────────────── DTO helpers ─────────────────────────


def _email_summary(email: Email) -> dict:
    """Compact email row for inbox list views."""
    return {
        "id": email.id,
        "from": email.sender,
        "to": email.recipient,
        "sent_at": email.sent_at_label,
        "is_read": email.is_read,
        "preview": email.body[:140] + ("…" if len(email.body) > 140 else ""),
        "folder": email.folder,
        "thread_id": email.thread_id,
    }


def _email_full(email: Email) -> dict:
    return {
        **_email_summary(email),
        "body": email.body,
        "beat_id": email.beat_id,
    }


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
