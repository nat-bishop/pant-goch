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
from pathlib import Path
from typing import AsyncIterator, Iterator

import logfire
from logfire import ConsoleOptions
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIChatModel
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
    make_llama_token_counter,
    player_canon_snapshot,
)
from pydantic_ai import NativeOutput
from .config import Settings
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
from .spine import (
    CANON_SALIENCE_THRESHOLD,
    apply_canon_extraction,
    apply_classify_effect,
    apply_literal_effects,
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
    """Raised by `_render_via_chat_agent` when the LLM call (or its history
    processors) fails. `_send_stream` catches this and surfaces a structured
    `error` SSE event so the client can recover without seeing a torn HTTP
    stream. Classifier failures use a separate degraded path (skip the
    quality write) and don't raise."""


# ───────────────────────── globals (set in lifespan) ─────────────────────────


SCENARIO: Scenario | None = None
STATE: InboxState | None = None
AGENTS: AgentBundle | None = None
SETTINGS: Settings | None = None  # populated at lifespan boot; read by hot-path endpoints
LLAMA_PROC: subprocess.Popen | None = None
ACTION_LOCK = asyncio.Lock()  # serialises mutating endpoints
EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS = 2.0  # tests set this to 0.1 for fast boot
# Most recent tick decision, updated from `_tick` after every scheduler
# walk. Read by `/playtest_note` so a one-click annotation captures the
# engine state at the moment the player flagged something. Held globally
# rather than threaded through ChatDeps because note submission is
# completely out-of-band from the /send pipeline.
LAST_TICK_DECISION: TickDecision | None = None
# Boot-stamped per-process id used to bucket playtest notes. One JSONL file
# per server boot under `playtest_notes/{SESSION_ID}.jsonl`. Solo-dev
# workflow: a fresh process is "this playthrough"; no rotation, no retention.
SESSION_ID: str | None = None


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global SCENARIO, STATE, AGENTS, SETTINGS, LLAMA_PROC, SESSION_ID

    settings = Settings()
    SETTINGS = settings
    # Per-boot id used to bucket playtest notes; uuid suffix disambiguates
    # boots that happen within the same second.
    SESSION_ID = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    scenario_path = settings.scenario_path or Path(__file__).resolve().parents[2] / "scenario.yaml"
    logfire.info("loading scenario from {path}", path=str(scenario_path))
    SCENARIO = load_scenario(scenario_path)

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
        model = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=api_url, api_key="not-needed"),
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

        await _bootstrap_state()
        logfire.info(
            "npc_service ready",
            boot_emails=len(STATE.emails),
            pending_beats=len(STATE.pending_beats),
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
    beat: Beat,
    *,
    player_body: str | None,
    in_response_to: str | None,
) -> tuple[Email | None, list[QualityDelta], int | None, FactCheckTrace | None]:
    """Fire a beat and produce its email plus tick-decision side info.

    Returns `(email, write_deltas, scene_turn, fact_check)`:
      - `email`: the rendered email, or None on silent-complete.
      - `write_deltas`: any quality changes from literal effects (empty if
        the beat has none, or if writes were no-ops).
      - `scene_turn`: 1-indexed fire count for scene beats (computed
        after the email is appended), None for one-shots.
      - `fact_check`: per-attempt validator trace; None unless the beat
        had `validate_facts: true` AND wasn't pre-authored.

    Two render paths:
    - `spec.pre_authored_prose` set → ship verbatim (no LLM call). The
      rendered email seeds `thread_messages` so the chat agent's next run
      sees the prose as a prior assistant turn.
    - otherwise → run the chat agent with `plot_directive=beat.spec`, using
      the current player_body as user_prompt. Terminals (`ends_session`)
      additionally inject the per-run epilogue digest into ChatDeps.
      Caller must pass player_body for chat-driven beats — they're
      triggered by /send ticks where a body always exists.

    Scene-vs-one_shot post-fire bookkeeping:
    - one_shot → enters `fired_beat_ids` immediately (fire-once contract).
    - scene → enters `fired_beat_ids` only when `scene_exit_satisfied`
      passes after this fire's literal effects + email append. Otherwise
      it stays pending and may continue firing on subsequent ticks.
    """
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    if beat.id in STATE.fired_beat_ids:
        return None, [], None, None

    # Literal effects apply BEFORE rendering so scene re-fires record
    # the new value even if rendering fails. `apply_literal_effects` logs
    # each before→after change and returns the resulting deltas (no-op
    # writes are filtered). Classify-driven effects on this beat schedule
    # the classifier on the next player reply — they don't fire here.
    write_deltas: list[QualityDelta] = apply_literal_effects(
        STATE, SCENARIO, beat.spec.effects, beat_id=beat.id,
    )

    # No span wrapper here: pre-authored fires are sub-millisecond (a memcpy
    # plus a log line) and the chat-agent path is 1:1 with
    # `_render_via_chat_agent`, which carries its own llm-tagged span.
    # Adding a `fire_beat` layer here only deepened the trace tree without
    # filtering value. `beat_id` is threaded onto `render_chat` via `beat=`,
    # so the chat-driven fire is still searchable by beat id.
    is_scene = beat.kind == "scene"
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
            recipient=resolve_player_email(SCENARIO, STATE),
            body=beat.spec.pre_authored_prose.strip(),
            sent_at_label="just now",
        )
        _seed_thread_history(beat.thread, email.body)
        logfire.info(
            "fire pre-authored beat {beat_id}",
            beat_id=beat.id, thread=beat.thread,
        )
    else:
        assert player_body is not None, (
            "chat-driven beats require a player body — they trigger only on /send ticks"
        )
        prior_fires = _count_beat_email_fires(beat.id) if is_scene else 0
        if is_scene:
            logfire.info(
                "continue scene {beat_id} (turn {turn})",
                beat_id=beat.id, turn=prior_fires + 1,
            )
        else:
            logfire.info("fire plot beat {beat_id}", beat_id=beat.id)
        # Terminals run the chat agent in epilogue mode (digest in ChatDeps);
        # everything else runs an ordinary chat turn.
        renderer = _render_via_epilogue_agent if beat.ends_session else _render_via_chat_agent
        # None when the beat didn't opt in — the validator's append-when-non-None
        # check is the single mechanism for keeping unflagged beats trace-free.
        fact_check_log: list[FactCheckAttempt] | None = (
            [] if beat.spec.validate_facts else None
        )
        email = await renderer(
            npc_id=beat.npc,
            thread_id=beat.thread,
            player_body=player_body,
            in_reply_to_email_id=in_response_to,
            plot_directive=beat.spec,
            beat_id=beat.id,
            fact_check_log=fact_check_log,
        )
        if fact_check_log is not None:
            # Empty list means the validator no-op'd (shouldn't happen since
            # validate_facts requires must_include_facts/must_not at load
            # time) — leave fact_check None in that case rather than ship
            # an attempts=[] payload that's semantically misleading.
            fact_check = FactCheckTrace(attempts=fact_check_log) if fact_check_log else None

    STATE.emails.append(email)
    if beat.ends_session:
        STATE.ended = True
        STATE.end_reason = beat.id
    scene_turn: int | None = prior_fires + 1 if is_scene else None

    # one_shot: enters fired_beat_ids immediately. scene: enters
    # fired_beat_ids only when its exit_when predicates all pass after
    # this fire — otherwise it stays pending for next tick.
    if not is_scene:
        STATE.fired_beat_ids.add(beat.id)
    elif scene_exit_satisfied(beat, STATE):
        STATE.fired_beat_ids.add(beat.id)
        logfire.info(
            "scene completed {beat_id} on turn {turn}",
            beat_id=beat.id, turn=scene_turn,
        )
    return email, write_deltas, scene_turn, fact_check


def _seed_thread_history(thread_id: str, body: str) -> None:
    """Anchored emails on NPC threads land in `thread_messages` as
    `ModelResponse(TextPart)` so the chat agent's next run sees them as
    prior assistant turns. System threads (no NPC) skip this — there's no
    chat agent for them."""
    assert STATE is not None and SCENARIO is not None
    thread = SCENARIO.thread_by_id.get(thread_id)
    if thread is None or thread.npc is None:
        return
    history = STATE.thread_messages.setdefault(thread_id, [])
    history.append(ModelResponse(parts=[TextPart(content=body)]))


async def _render_email(
    *,
    npc_id: str,
    thread_id: str,
    user_prompt: str,
    plot_directive: BeatSpec | None,
    beat_id: str | None,
    message_history: list | None = None,
    append_history: bool = False,
    epilogue_digest: str | None = None,
    fact_check_log: list[FactCheckAttempt] | None = None,
    span_label: str = "render to {npc} on {thread}",
    extra_span_attrs: dict | None = None,
) -> Email:
    """Single LLM-render path. Builds ChatDeps, runs the chat agent, wraps
    failures as ChatAgentFailure, returns the assembled Email. If
    `append_history=True`, extends `STATE.thread_messages[thread_id]` with
    `result.new_messages()`. When `epilogue_digest` is set, ChatDeps
    switches the agent into closing-email mode (different system framing,
    digest surfaced via `@instructions`).

    `fact_check_log`, when supplied, is a mutable list the validator
    appends to per attempt; `_fire_beat` packages it into a
    `FactCheckTrace` after the render returns."""
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    npc = SCENARIO.npc_by_id[npc_id]
    deps = ChatDeps(
        npc=npc,
        plot_directive=plot_directive,
        commitments=commitments_snapshot(
            STATE.qualities,
            SCENARIO.quality_defaults,
            SCENARIO.quality_descriptions,
            SCENARIO.surface_to_chat_quality_ids,
        ),
        player_canon=player_canon_snapshot(
            STATE.player_canon,
            current_turn=len(STATE.replies),
        ),
        epilogue_digest=epilogue_digest,
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
            history = STATE.thread_messages.setdefault(thread_id, [])
            history.extend(result.new_messages())
        usage = result.usage()
        span.set_attribute("input_tokens", usage.input_tokens)
        span.set_attribute("output_tokens", usage.output_tokens)
        span.set_attribute("response_preview", result.output.strip()[:200])

    body = result.output.strip()
    return Email(
        id=f"email_{uuid.uuid4().hex[:8]}",
        beat_id=beat_id,
        thread_id=thread_id,
        folder="Inbox",
        sender=npc.email_address,
        recipient=resolve_player_email(SCENARIO, STATE),
        body=body,
        sent_at_label="just now",
    )


async def _render_via_chat_agent(
    *,
    npc_id: str,
    thread_id: str,
    player_body: str,
    in_reply_to_email_id: str | None,
    plot_directive: BeatSpec | None,
    beat_id: str | None,
    fact_check_log: list[FactCheckAttempt] | None = None,
) -> Email:
    """Render path for chat turns and non-terminal beat fires. Appends to
    thread_messages so the next turn sees this exchange in conversation history."""
    assert STATE is not None
    history = STATE.thread_messages.setdefault(thread_id, [])
    return await _render_email(
        npc_id=npc_id,
        thread_id=thread_id,
        user_prompt=player_body,
        plot_directive=plot_directive,
        beat_id=beat_id,
        message_history=list(history),
        append_history=True,
        fact_check_log=fact_check_log,
        span_label="render chat to {npc} on {thread}",
        extra_span_attrs={
            "plot": plot_directive is not None,
            "history_len": len(history),
            "prompt_preview": player_body[:200],
        },
    )


async def _render_via_epilogue_agent(
    *,
    npc_id: str,
    thread_id: str,
    player_body: str,
    in_reply_to_email_id: str | None,
    plot_directive: BeatSpec | None,
    beat_id: str | None,
    fact_check_log: list[FactCheckAttempt] | None = None,
) -> Email:
    """Render path for terminal beats. Same chat agent as ordinary turns,
    but ChatDeps carries an `epilogue_digest` that flips it into
    closing-email mode (the `_epilogue_framing_block` and
    `_epilogue_digest_block` instruction decorators in `agents.py` only
    activate when the digest is set). No message history — the digest is
    the run summary."""
    digest = _build_epilogue_digest()
    return await _render_email(
        npc_id=npc_id,
        thread_id=thread_id,
        user_prompt=player_body,
        plot_directive=plot_directive,
        beat_id=beat_id,
        message_history=None,
        append_history=False,
        epilogue_digest=digest,
        fact_check_log=fact_check_log,
        span_label="render epilogue to {npc} on {thread}",
        extra_span_attrs={"digest_preview": digest[:300]},
    )


def _build_epilogue_digest() -> str:
    """Per-run digest passed to the epilogue agent in the user prompt.

    Sections: commitments (delta from defaults), per-axis classifier
    history (most-recent-first), fired beats in order, last 3 player
    replies verbatim. The dossier and plot directive are already injected
    via @instructions; this digest is the run-tailored payload.
    """
    assert STATE is not None and SCENARIO is not None
    # Commitments: filter to surface_to_chat qualities ≠ default.
    commits_dict = commitments_snapshot(
        STATE.qualities,
        SCENARIO.quality_defaults,
        SCENARIO.quality_descriptions,
        SCENARIO.surface_to_chat_quality_ids,
    )
    commit_lines = [
        f"  {k} = {v!r}" if isinstance(v, str) else f"  {k} = {v:g}"
        for k, (v, _desc) in commits_dict.items()
    ] or ["  (no commitments yet)"]

    history = build_recent_classifier_history(STATE.replies, axes=None, per_axis=5)
    history_lines = [f"  {axis}: {', '.join(values)}" for axis, values in history.items()]
    history_block = "\n".join(history_lines) or "  (no replies classified)"

    fired_in_order = _fired_beat_ids_in_order()
    fired_block = ", ".join(fired_in_order) if fired_in_order else "(none)"

    # most recent last for natural reading order; trim defensively in case of
    # very long player pastes.
    recent_replies = list(STATE.replies)[-3:]
    reply_block_parts = [f"  - {r.body.strip()[:1000]}" for r in recent_replies]
    reply_block = "\n".join(reply_block_parts) or "  (no replies)"

    return (
        "RUN DIGEST:\n"
        f"Commitments:\n" + "\n".join(commit_lines) + "\n"
        f"Classifier history (last 5 per axis, most-recent-first):\n{history_block}\n"
        f"Fired beats (oldest → newest): {fired_block}\n"
        f"Recent player replies verbatim:\n{reply_block}"
    )


def _count_beat_email_fires(beat_id: str) -> int:
    """How many times this beat has rendered an email so far. Used by
    `_fire_beat` for the scene_turn count and by `_select_beat`'s
    scene-continuity short-circuit (any prior fire → continue). Walks
    `STATE.emails` directly because scenes don't enter `fired_beat_ids`
    until exit_when fires, so the set lookup is incomplete for in-flight
    scenes."""
    assert STATE is not None
    return sum(1 for e in STATE.emails if e.beat_id == beat_id)


def _fired_beat_ids_in_order(*, last: int | None = None) -> list[str]:
    """Fired-beat ids in emit order — read off Email.beat_id where set,
    since state.fired_beat_ids is a set with no order guarantee. `last`
    slices the tail (e.g. for the Director's recent-fire history)."""
    assert STATE is not None
    ids = [
        e.beat_id for e in STATE.emails
        if e.beat_id is not None and e.beat_id in STATE.fired_beat_ids
    ]
    return ids[-last:] if last is not None else ids


# ───────────────────────── scheduler tick ─────────────────────────


async def _select_beat(
    due: list[Beat],
    *,
    reply_thread_id: str | None = None,
) -> tuple[Beat | None, str, str | None]:
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
    assert STATE is not None and AGENTS is not None
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
            and _count_beat_email_fires(b.id) > 0
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
        qualities=dict(STATE.qualities),
        recent_classifier_history=build_recent_classifier_history(
            STATE.replies, axes=None, per_axis=5,
        ),
        recent_fire_history=_fired_beat_ids_in_order(last=6),
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
    assert STATE is not None and SCENARIO is not None
    new_emails: list[Email] = []
    plot_fired_threads: set[str] = set()
    quality_deltas: list[QualityDelta] = list(classifier_deltas or [])

    with logfire.span("tick (in response to {in_response_to})", in_response_to=in_response_to) as span:
        eligible, eligibility_report = find_due_beats(STATE, SCENARIO)
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
            eligible, reply_thread_id=reply_thread_id,
        )
        chosen_beat_id = chosen.id if chosen is not None else None
        fired_beat_id: str | None = None
        scene_turn: int | None = None
        fact_check: FactCheckTrace | None = None
        if chosen is not None:
            e, write_deltas, turn, trace = await _fire_beat(
                chosen,
                player_body=player_body,
                in_response_to=in_response_to,
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
                for k, v in STATE.player_canon.items()
            },
            fact_check=fact_check,
            # Per-axis history snapshot — most-recent-first values per axis,
            # capped at 10 (matches roughly two screens of pill chips).
            classifier_history=build_recent_classifier_history(
                STATE.replies, axes=None, per_axis=10,
            ),
            # Snapshot AFTER the fire — so the graph view and post-load
            # recovery see the up-to-date fired set without per-tick
            # accumulation logic on the frontend.
            fired_beat_ids=sorted(STATE.fired_beat_ids),
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
        # Stash for /playtest_note so out-of-band annotations can snapshot
        # the engine state at the moment of capture without reaching back
        # through SSE history.
        global LAST_TICK_DECISION
        LAST_TICK_DECISION = decision
        return new_emails, plot_fired_threads, decision


# ───────────────────────── endpoints ─────────────────────────


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "scenario_loaded": SCENARIO is not None,
        "state_initialised": STATE is not None,
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
async def get_state() -> dict:
    if STATE is None:
        raise HTTPException(503, "state not initialised")
    return STATE.public_view()


class PlaytestNoteRequest(BaseModel):
    note: str = Field(min_length=1)


@app.post("/playtest_note")
async def post_playtest_note(req: PlaytestNoteRequest) -> dict:
    """Append one playtest annotation to a per-boot JSONL file.

    Snapshots {timestamp, note, last_tick_decision, qualities, canon,
    recent_emails (last 10 with body)} so a "wait what was happening when I
    noticed this" question is answerable from the file alone — no need to
    re-correlate with logs.

    Returns the count of notes written this session so the panel can show
    a session counter ("3 notes saved").
    """
    if STATE is None or SESSION_ID is None:
        raise HTTPException(503, "state not initialised")
    notes_dir = Path("playtest_notes")
    notes_dir.mkdir(exist_ok=True)
    notes_path = notes_dir / f"{SESSION_ID}.jsonl"
    payload = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "note": req.note,
        "session_id": SESSION_ID,
        "last_tick_decision": (
            LAST_TICK_DECISION.model_dump() if LAST_TICK_DECISION else None
        ),
        "qualities": dict(STATE.qualities),
        "player_canon": {
            k: {
                "summary": v.summary,
                "literal_text_quote": v.literal_text_quote,
                "salience": v.salience,
                "category": v.category.value,
                "turn_of_origin": v.turn_of_origin,
            }
            for k, v in STATE.player_canon.items()
        },
        "recent_emails": [
            {
                "id": e.id,
                "beat_id": e.beat_id,
                "thread_id": e.thread_id,
                "sender": e.sender,
                "body": e.body,
            }
            for e in STATE.emails[-10:]
        ],
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    # Append-mode write is atomic on Linux/macOS for sub-PIPE_BUF lines and
    # sufficient on Windows for our scale. /send is serialized via
    # ACTION_LOCK; /playtest_note runs concurrently with /send but the OS
    # append-write is the right primitive for one-line records.
    with notes_path.open("a", encoding="utf-8") as f:
        f.write(line)
    # Cheap count for the panel — line count of the file. Capped at the
    # current write so we don't re-walk it on every call.
    return {
        "session_id": SESSION_ID,
        "path": str(notes_path),
        "note_count": _count_lines(notes_path),
    }


def _count_lines(path: Path) -> int:
    """Plain line-count for the playtest-note counter. Sub-millisecond at
    realistic note volumes — no need to maintain a counter in memory."""
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


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
        "validate_facts": beat.spec.validate_facts,
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
async def get_inbox() -> dict:
    """Folder summary. The client uses this for the inbox list view."""
    if STATE is None:
        raise HTTPException(503, "state not initialised")
    folders: dict[str, list[dict]] = {"Inbox": [], "Sent": []}
    for e in STATE.emails:
        folders.setdefault(e.folder, []).append(_email_summary(e))
    # Newest first within each folder. Append order is emit order; reverse it.
    for k in folders:
        folders[k].reverse()
    return {"folders": folders, "state": STATE.public_view()}


@app.get("/thread/{thread_id}")
async def get_thread(thread_id: str) -> dict:
    """All emails in a thread, in emit order."""
    if STATE is None or SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if thread_id not in SCENARIO.thread_by_id:
        raise HTTPException(404, f"unknown thread: {thread_id}")
    emails = [_email_full(e) for e in STATE.emails if e.thread_id == thread_id]
    return {"thread_id": thread_id, "emails": emails}


@app.post("/open")
async def post_open(req: OpenRequest) -> dict:
    """Mark email read. /open does not mutate qualities — reading isn't a
    communication act, so it can't move engagement or fire any beat."""
    if STATE is None or SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if STATE.ended:
        raise HTTPException(409, f"session ended ({STATE.end_reason})")

    async with ACTION_LOCK:
        email = next((e for e in STATE.emails if e.id == req.email_id), None)
        if email is None:
            raise HTTPException(404, f"unknown email_id: {req.email_id}")
        email.is_read = True
        return {
            "state": STATE.public_view(),
            "new_emails": [],
        }


@app.post("/send")
async def post_send(req: SendRequest) -> StreamingResponse:
    if STATE is None or AGENTS is None or SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if STATE.ended:
        raise HTTPException(409, f"session ended ({STATE.end_reason})")
    if not req.body.strip():
        raise HTTPException(400, "body is required")
    return StreamingResponse(
        _send_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _send_stream(req: SendRequest) -> AsyncIterator[bytes]:
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    body = req.body.strip()
    in_reply_email = (
        next((e for e in STATE.emails if e.id == req.in_reply_to), None)
        if req.in_reply_to else None
    )
    thread_id = in_reply_email.thread_id if in_reply_email else None

    with logfire.span("send to {recipient} on {thread_id}", recipient=req.recipient, thread_id=thread_id):
        async with ACTION_LOCK:
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
            STATE.emails.append(send_email)
            reply = Reply(
                id=f"reply_{uuid.uuid4().hex[:8]}",
                thread_id=thread_id,
                beat_responded_to=req.in_reply_to,
                recipient=req.recipient,
                body=body,
            )
            STATE.replies.append(reply)
            yield _sse({"type": "send_recorded", "email": _email_full(send_email)})

            # Classifier and extractor run in parallel before _tick (via
            # asyncio.gather, latency-neutral). Both must complete first
            # because find_due_beats reads state mutated by both: classifier
            # writes qualities (gates terminal decisions) and extractor
            # writes player_canon (PlayerCanonExists predicates gate calibration).
            recipient_npc = req.recipient_npc_id or _resolve_recipient_npc_id(req.recipient)
            replied_beat = (
                SCENARIO.beat_by_id.get(in_reply_email.beat_id)
                if in_reply_email and in_reply_email.beat_id else None
            )
            # Classify-typed effects come from two sources: the beat the
            # player just replied to (per-beat axes) and scenario.effects
            # (always-on, gated by their own `when:`). Both pour into one
            # batched classifier call. Beat-scope effects come first so a
            # scenario-scope effect on the same quality (rare and probably
            # an authoring mistake) doesn't shadow the beat-scope verdict.
            beat_classify_effects: list[Effect] = (
                [e for e in replied_beat.spec.effects if e.is_classify]
                if replied_beat else []
            )
            scenario_classify_effects: list[Effect] = [
                e for e in SCENARIO.effects
                if e.is_classify
                and (e.when is None or evaluate_predicate(e.when, STATE))
            ]
            classify_effects = beat_classify_effects + scenario_classify_effects
            recipient_blurb = _persona_blurb_for_recipient(req.recipient, recipient_npc)
            prior_body = in_reply_email.body if in_reply_email else ""

            classifier_coro = _classify_player_reply(
                recipient_blurb=recipient_blurb,
                prior_email_body=prior_body,
                reply_body=body,
                effects=classify_effects,
            ) if classify_effects else _none_coro()
            extractor_coro = _extract_player_canon(
                recipient_blurb=recipient_blurb,
                prior_email_body=prior_body,
                reply_body=body,
            )
            classifier_results, extractor_output = await asyncio.gather(
                classifier_coro, extractor_coro,
            )

            classifier_deltas: list[QualityDelta] = []
            if classify_effects and classifier_results is not None:
                _apply_classify_results(
                    effects=classify_effects,
                    results=classifier_results,
                    replied_beat_id=replied_beat.id if replied_beat else None,
                    reply=reply,
                    out_deltas=classifier_deltas,
                )

            canon_writes: list[CanonCandidate] = []
            canon_drops: list[CanonDropRecord] = []
            if extractor_output is not None and extractor_output.candidates:
                # Categorize every candidate so the debug overlay shows BOTH
                # what was promoted AND what the extractor proposed but the
                # engine rejected. Two rejection reasons:
                #   - anchor_fail: literal_text_quote isn't a verbatim
                #     substring of the reply (anti-hallucination floor)
                #   - below_salience: anchored, but extractor confidence
                #     below the promotion threshold
                # apply_canon_extraction also enforces the salience floor
                # internally, so promoted candidates ALWAYS clear it; we
                # categorize here so canon_drops carries the dropped ones
                # for surfacing without re-running any logic.
                promoted: list[CanonCandidate] = []
                for c in extractor_output.candidates:
                    if c.literal_text_quote not in body:
                        canon_drops.append(CanonDropRecord(candidate=c, reason="anchor_fail"))
                    elif c.salience < CANON_SALIENCE_THRESHOLD:
                        canon_drops.append(CanonDropRecord(candidate=c, reason="below_salience"))
                    else:
                        promoted.append(c)
                if canon_drops:
                    logfire.warn(
                        "extractor candidates dropped",
                        anchor_fail=sum(1 for d in canon_drops if d.reason == "anchor_fail"),
                        below_salience=sum(1 for d in canon_drops if d.reason == "below_salience"),
                        kept=len(promoted),
                    )
                if promoted:
                    apply_canon_extraction(
                        STATE,
                        promoted,
                        turn_of_origin=len(STATE.replies),
                    )
                    canon_writes = promoted

            try:
                new_emails, plot_fired_threads, decision = await _tick(
                    player_body=body,
                    in_response_to=send_email.id,
                    reply_thread_id=thread_id,
                    classifier_deltas=classifier_deltas,
                    canon_writes=canon_writes,
                    canon_drops=canon_drops,
                )

                # Chat-turn fallback fires when no chat-driven beat covered
                # this thread. With scene-as-primitive, scenes that match
                # the player's reply thread short-circuit the Director and
                # fire here; this branch only matters for threads without
                # a covering scene OR plot beat this tick.
                if (
                    recipient_npc is not None
                    and thread_id is not None
                    and thread_id not in plot_fired_threads
                ):
                    chat_email = await _render_via_chat_agent(
                        npc_id=recipient_npc,
                        thread_id=thread_id,
                        player_body=body,
                        in_reply_to_email_id=send_email.id,
                        plot_directive=None,
                        beat_id=None,
                    )
                    STATE.emails.append(chat_email)
                    new_emails.append(chat_email)
            except ChatAgentFailure as err:
                # Player email and any classifier write are already persisted
                # — surface a structured error so the client can display it
                # without losing the player's send. State + done still flush.
                logfire.warn(
                    "send_stream aborted: chat agent failed",
                    thread_id=thread_id, message=str(err),
                )
                yield _sse({
                    "type": "error",
                    "message": (
                        "Aria couldn't compose a reply just now. "
                        "Please try again."
                    ),
                })
                yield _sse({"type": "state", "state": STATE.public_view()})
                yield _sse({"type": "done"})
                return

            for ne in new_emails:
                yield _sse({"type": "new_email", "email": _email_full(ne)})

            yield _sse({"type": "tick_decision", "decision": decision.model_dump()})
            yield _sse({"type": "state", "state": STATE.public_view()})
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
    - Off-spec value (not in `allowed_values`) → `apply_classify_effect`
      returns None, logfire-warn, no append onto Reply.

    Appends a `QualityDelta` to `out_deltas` for each effect whose value
    actually changed; `_tick` folds those into the
    `TickDecision.quality_deltas` so the debug overlay shows "what
    changed this tick" in one place.
    """
    assert STATE is not None and SCENARIO is not None
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
        applied, delta = apply_classify_effect(
            STATE, SCENARIO, effect, r.value, r.rationale,
        )
        if applied is None:
            assert effect.classify is not None
            logfire.warn(
                "classifier returned off-spec value; quality unchanged",
                beat_id=replied_beat_id, quality=effect.quality,
                value=r.value, allowed=effect.classify.allowed_values,
            )
            continue
        reply.classifier_results.append(applied)
        if delta is not None:
            out_deltas.append(delta)


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


async def _bootstrap_state() -> None:
    """Init state from scenario; fire any beats already due (pre-authored
    beats with empty preconditions land here on the first tick)."""
    global STATE
    assert SCENARIO is not None
    STATE = InboxState.from_scenario(SCENARIO)
    # Boot tick fires pre-authored beats with empty preconditions (e.g.
    # aria.welcome). Chat-driven beats are auto-pended too but require a player /send
    # to satisfy `player_replies_to_thread_gte`, so they can't fire here.
    # The TickDecision from boot is discarded — there's no SSE consumer.
    await _tick(player_body=None, in_response_to=None)


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
