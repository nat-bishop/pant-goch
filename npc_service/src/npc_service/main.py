"""FastAPI surface for Inbox. See CLAUDE.md for the architecture and
load-bearing invariants.
"""
from __future__ import annotations

import asyncio
import contextlib
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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agents import (
    AgentBundle,
    ChatDeps,
    build_agents,
    build_classifier_user_prompt,
    build_director_output_type,
    build_director_user_prompt,
    build_recent_classifier_history,
    commitments_snapshot,
    make_llama_token_counter,
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
    AnchoredBeat,
    AxisResult,
    Beat,
    ClassifyReply,
    Email,
    InboxState,
    OpenRequest,
    PlotBeat,
    Reply,
    Scenario,
    SendRequest,
    load_scenario,
)
from .spine import (
    apply_classification,
    apply_writes,
    evaluate_completes_when,
    find_due_beats,
    reply_subject_for_email,
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


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global SCENARIO, STATE, AGENTS, SETTINGS, LLAMA_PROC

    settings = Settings()
    SETTINGS = settings
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
) -> Email | None:
    """Fire a beat and produce its email.

    AnchoredBeats ship verbatim; the rendered email seeds `thread_messages`
    so the chat agent's next run sees the anchored prose as a prior turn.

    PlotBeats run the chat agent with `plot_directive=beat.spec`, using the
    current player_body as user_prompt. Caller must pass player_body for
    PlotBeats — they're triggered by /send ticks where a body always exists.

    Multi-turn lifecycle: if the beat has a non-empty `completes_when` AND
    the gate is already met at fire-time, mark complete WITHOUT rendering
    (returns None). Otherwise render. After rendering, the beat is added
    to `fired_beat_ids` only if `completes_when` is empty (legacy fire-once)
    OR has just been met. Multi-turn beats with unmet gates stay out of
    `fired_beat_ids` so `find_due_beats` re-fires them on subsequent ticks.
    """
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    if beat.id in STATE.fired_beat_ids:
        return None

    # Writes apply *before* the silent-complete check so the fact of
    # completion is recorded even on the no-render path.
    if isinstance(beat, PlotBeat) and beat.spec.writes:
        apply_writes(STATE, SCENARIO, beat.spec.writes)

    # If a multi-turn beat's gate is already satisfied (e.g. the
    # just-applied classifier bumped its quality past the threshold),
    # complete silently — no render — and let the chain advance.
    if beat.completes_when and evaluate_completes_when(STATE, beat):
        STATE.fired_beat_ids.add(beat.id)
        logfire.info("beat completed without render", beat_id=beat.id)
        return None

    # No span wrapper here: AnchoredBeat fires are sub-millisecond (a memcpy
    # plus a log line) and the PlotBeat path is 1:1 with `_render_via_chat_agent`,
    # which carries its own llm-tagged span. Adding a `fire_beat` layer here
    # only deepened the trace tree without filtering value. `beat_id` is
    # threaded onto `render_chat` via `beat=`, so the PlotBeat fire is still
    # searchable by beat id.
    email: Email | None = None
    if isinstance(beat, AnchoredBeat):
        email = Email(
            id=f"email_{uuid.uuid4().hex[:8]}",
            beat_id=beat.id,
            thread_id=beat.thread,
            folder="Inbox",
            sender=beat.from_,
            recipient=SCENARIO.player_email,
            subject=beat.subject,
            body=beat.authored_prose.strip(),
            sent_at_label=beat.sent_at or "just now",
        )
        _seed_thread_history(beat.thread, email.body)
        logfire.info(
            "fire anchored beat {beat_id}",
            beat_id=beat.id, thread=beat.thread, subject=beat.subject,
        )
    elif isinstance(beat, PlotBeat):
        assert player_body is not None, (
            "PlotBeat fires require a player body — they trigger only on /send ticks"
        )
        # Terminals are the epilogue: they go through the epilogue agent
        # with a richer run digest. Everything else uses the chat agent.
        renderer = _render_via_epilogue_agent if beat.ends_session else _render_via_chat_agent
        email = await renderer(
            npc_id=beat.npc,
            thread_id=beat.thread,
            player_body=player_body,
            in_reply_to_email_id=in_response_to,
            plot_directive=beat.spec,
            beat_id=beat.id,
            subject=beat.subject,
        )

    is_plot = isinstance(beat, PlotBeat)
    if email is not None:
        STATE.emails.append(email)
        if is_plot and beat.ends_session:
            STATE.ended = True
            STATE.end_reason = beat.id

    # multi_turn beats never enter fired_beat_ids; they stop firing
    # only when preconditions stop holding (typical pattern: mutex via
    # Not_(BeatFired …) against terminals that supersede them).
    if not (is_plot and beat.multi_turn) and not beat.completes_when:
        STATE.fired_beat_ids.add(beat.id)
    return email


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
    agent,
    npc_id: str,
    thread_id: str,
    user_prompt: str,
    in_reply_to_email_id: str | None,
    plot_directive,
    beat_id: str | None,
    subject: str | None,
    message_history: list | None = None,
    append_history: bool = False,
    span_label: str = "render to {npc} on {thread}",
    extra_span_attrs: dict | None = None,
) -> Email:
    """Single LLM-render path. Builds ChatDeps, runs `agent`, wraps failures as
    ChatAgentFailure, returns the assembled Email. If `append_history=True`,
    extends `STATE.thread_messages[thread_id]` with `result.new_messages()`.
    """
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    npc = SCENARIO.npc_by_id[npc_id]
    deps = ChatDeps(
        npc=npc,
        plot_directive=plot_directive,
        commitments=commitments_snapshot(
            STATE.qualities,
            SCENARIO.quality_defaults,
            SCENARIO.quality_descriptions,
        ),
    )

    with LLM_LOGFIRE.span(
        span_label,
        npc=npc_id,
        thread=thread_id,
        beat=beat_id,
        **(extra_span_attrs or {}),
    ) as span:
        try:
            result = await agent.run(
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
    derived_subject = subject if subject is not None else reply_subject_for_email(
        STATE, in_reply_to_email_id
    )
    return Email(
        id=f"email_{uuid.uuid4().hex[:8]}",
        beat_id=beat_id,
        thread_id=thread_id,
        folder="Inbox",
        sender=npc.email_address,
        recipient=SCENARIO.player_email,
        subject=derived_subject,
        body=body,
        sent_at_label="just now",
        in_reply_to_beat=in_reply_to_email_id,
    )


async def _render_via_chat_agent(
    *,
    npc_id: str,
    thread_id: str,
    player_body: str,
    in_reply_to_email_id: str | None,
    plot_directive,
    beat_id: str | None,
    subject: str | None,
) -> Email:
    """Render path for chat turns and non-terminal PlotBeat fires. Appends to
    thread_messages so the next turn sees this exchange in conversation history."""
    assert STATE is not None
    history = STATE.thread_messages.setdefault(thread_id, [])
    return await _render_email(
        agent=AGENTS.chat_agent if AGENTS else None,
        npc_id=npc_id,
        thread_id=thread_id,
        user_prompt=player_body,
        in_reply_to_email_id=in_reply_to_email_id,
        plot_directive=plot_directive,
        beat_id=beat_id,
        subject=subject,
        message_history=list(history),
        append_history=True,
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
    plot_directive,
    beat_id: str | None,
    subject: str | None,
) -> Email:
    """Render path for terminal beats. Builds a per-run digest and runs the
    epilogue agent (separate system prompt, no message history)."""
    assert AGENTS is not None
    digest = _build_epilogue_digest()
    user_prompt = (
        f"{digest}\n\n"
        f"PLAYER'S MOST RECENT REPLY (the one that immediately precedes this terminal):\n"
        f"{player_body}"
    )
    return await _render_email(
        agent=AGENTS.epilogue_agent,
        npc_id=npc_id,
        thread_id=thread_id,
        user_prompt=user_prompt,
        in_reply_to_email_id=in_reply_to_email_id,
        plot_directive=plot_directive,
        beat_id=beat_id,
        subject=subject,
        message_history=None,
        append_history=False,
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
    # Commitments: filter to commitment.* qualities ≠ default; format with prefix stripped.
    commits_dict = commitments_snapshot(
        STATE.qualities,
        SCENARIO.quality_defaults,
        SCENARIO.quality_descriptions,
    )
    commit_lines = [
        f"  {k.split('.', 1)[1]} = {v:g}" for k, (v, _desc) in commits_dict.items()
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


async def _select_beat(due: list[Beat]) -> Beat | None:
    """Pick one beat from the due list. Short-circuits at 0 or 1 due (most
    ticks); calls the Director when ≥2 are eligible. Returns None when no
    beat should fire (empty due, or Director chose "none")."""
    assert STATE is not None and AGENTS is not None
    if not due:
        return None
    if len(due) == 1:
        return due[0]

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
            return due[0]
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
            return None
        if chosen_id not in by_id:
            logfire.warn(
                "director returned an id not in eligible set; defaulting to first",
                returned=chosen_id, eligible=[b.id for b in due],
            )
            return due[0]
        return by_id[chosen_id]


def _intent_summary_for(beat: Beat) -> str:
    """One-line summary used in the Director's eligible-beat list. For plot
    beats, the first line of `spec.intent` (truncated). For anchored beats,
    the subject."""
    if isinstance(beat, PlotBeat):
        first_line = beat.spec.intent.strip().splitlines()[0] if beat.spec.intent.strip() else "(no intent)"
        if len(first_line) > 200:
            first_line = first_line[:197] + "..."
        suffix = " [TERMINAL]" if beat.ends_session else (" [multi-turn]" if beat.multi_turn else "")
        return f"{first_line}{suffix}"
    if isinstance(beat, AnchoredBeat):
        return f"(authored email: '{beat.subject}')"
    return "(unknown beat type)"


async def _tick(
    *,
    player_body: str | None = None,
    in_response_to: str | None = None,
) -> tuple[list[Email], set[str]]:
    """Walk the beat scheduler. Returns (newly-emitted emails, set of thread
    ids where a PlotBeat fired this tick). The thread-id set lets the caller
    decide whether a chat-turn fallback is still needed.

    Selection: `find_due_beats` returns every eligible beat; `_select_beat`
    asks the Director to pick one (or short-circuits at 0/1 due). Exactly
    one beat fires per tick.
    """
    assert STATE is not None and SCENARIO is not None
    new_emails: list[Email] = []
    plot_fired_threads: set[str] = set()

    with logfire.span("tick (in response to {in_response_to})", in_response_to=in_response_to) as span:
        due = find_due_beats(STATE, SCENARIO)
        chosen = await _select_beat(due)
        if chosen is not None:
            e = await _fire_beat(
                chosen,
                player_body=player_body,
                in_response_to=in_response_to,
            )
            if e is not None:
                new_emails.append(e)
                if isinstance(chosen, PlotBeat):
                    plot_fired_threads.add(chosen.thread)
        span.set_attribute("new_email_count", len(new_emails))
        return new_emails, plot_fired_threads


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
                subject=reply_subject_for_email(STATE, req.in_reply_to),
                body=body,
                sent_at_label="just now",
                in_reply_to_beat=req.in_reply_to,
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

            # Classifier runs serially before _tick because the replied-to
            # beat may consume its own classifier output (e.g. aria.loop's
            # `accumulate=true` engagement axis whose value gates the
            # complete_warm vs complete_standard split). _tick must see the
            # post-classifier qualities to make terminal decisions.
            recipient_npc = _infer_recipient_npc_id(req.recipient)
            replied_beat = (
                SCENARIO.beat_by_id.get(in_reply_email.beat_id)
                if in_reply_email and in_reply_email.beat_id else None
            )
            axes = list(replied_beat.classify_reply_axes) if replied_beat else []
            if axes:
                results = await _classify_player_reply(
                    recipient_blurb=_persona_blurb_for_recipient(req.recipient, recipient_npc),
                    prior_email_body=in_reply_email.body if in_reply_email else "",
                    reply_body=body,
                    axes=axes,
                )
                if results is not None:
                    for event in _apply_axis_results(
                        axes=axes, results=results, replied_beat_id=replied_beat.id, reply=reply,
                    ):
                        yield event

            try:
                new_emails, plot_fired_threads = await _tick(
                    player_body=body,
                    in_response_to=send_email.id,
                )

                # Chat-turn fallback fires when no PlotBeat covered this
                # thread. In the steady state aria.loop is multi-turn and
                # fires on every tick after the welcome reply, so this branch
                # only matters for threads without a covering loop beat.
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
                        subject=None,
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

            yield _sse({"type": "state", "state": STATE.public_view()})
            yield _sse({"type": "done"})


async def _classify_player_reply(
    *,
    recipient_blurb: str,
    prior_email_body: str,
    reply_body: str,
    axes: list[ClassifyReply],
) -> list[AxisResult] | None:
    """Run the classifier against the player's reply across all authored
    axes in a single batched call. Returns None on failure so the caller
    skips every axis's quality write.

    Wrapped in an llm-tagged span so the auto-instrumented Pydantic AI run
    has a business-context parent — and so `reply_preview` and the per-axis
    qualities ride on the span row in the trace tree, no drilling.
    """
    assert AGENTS is not None
    with LLM_LOGFIRE.span(
        "classify reply for {qualities}",
        qualities=[a.identifier for a in axes],
        reply_preview=reply_body[:200],
    ) as span:
        try:
            result = await AGENTS.classifier_agent.run(
                build_classifier_user_prompt(
                    recipient_blurb, prior_email_body, reply_body, axes,
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


def _apply_axis_results(
    *,
    axes: list[ClassifyReply],
    results: list[AxisResult],
    replied_beat_id: str,
    reply: Reply,
) -> Iterator[bytes]:
    """Apply each authored axis in YAML order against the classifier's
    returned results. Defense-in-depth, single pass over `results` to
    categorize, then a pass over `axes` to apply:

    - Unknown axis (quality not authored on the beat) → logfire-warn, drop.
    - Duplicate axis (LLM returned multiple results for the same quality)
      → keep the first, logfire-warn the rest.
    - Missing axis (LLM omitted an authored quality) → logfire-warn, skip.
    - Off-spec value (not in `allowed_values`) → `apply_classification`
      returns None, logfire-warn, no SSE event, no append onto Reply.

    Emits one `classified` SSE event per applied axis, in YAML axis order.
    """
    assert STATE is not None and SCENARIO is not None
    authored_identifiers = {a.identifier for a in axes}
    by_identifier: dict[str, AxisResult] = {}
    for r in results:
        if r.quality not in authored_identifiers:
            logfire.warn(
                "classifier returned unauthored axis; dropped",
                beat_id=replied_beat_id, identifier=r.quality, value=r.value,
            )
            continue
        if r.quality in by_identifier:
            logfire.warn(
                "classifier returned duplicate axis result; first kept",
                beat_id=replied_beat_id, identifier=r.quality,
            )
            continue
        by_identifier[r.quality] = r

    for axis in axes:
        r = by_identifier.get(axis.identifier)
        if r is None:
            logfire.warn(
                "classifier missing axis result; quality unchanged",
                beat_id=replied_beat_id, identifier=axis.identifier,
            )
            continue
        applied = apply_classification(STATE, SCENARIO, axis, r.value, r.rationale)
        if applied is None:
            logfire.warn(
                "classifier returned off-spec value; quality unchanged",
                beat_id=replied_beat_id, identifier=axis.identifier,
                value=r.value, allowed=axis.allowed_values,
            )
            continue
        reply.classifier_results.append(applied)
        yield _sse({
            "type": "classified",
            "quality": axis.identifier,
            "value": r.value,
            "score": axis.map[r.value],
        })


def _persona_blurb_for_recipient(recipient: str, npc_id: str | None) -> str:
    if SCENARIO is None or npc_id is None:
        return f"a recipient: {recipient}"
    n = SCENARIO.npc_by_id[npc_id]
    return f"{n.display_name} — {n.relationship_to_owner}"


def _infer_recipient_npc_id(recipient: str) -> str | None:
    if SCENARIO is None:
        return None
    needle = recipient.lower()
    for n in SCENARIO.npcs:
        if n.display_name.lower() in needle or n.id in needle:
            return n.id
    return None


@app.post("/reset")
async def post_reset() -> JSONResponse:
    """Re-initialise state; useful for testing."""
    if SCENARIO is None:
        raise HTTPException(503, "scenario not loaded")
    logfire.info("/reset")
    await _bootstrap_state()
    assert STATE is not None
    return JSONResponse(STATE.public_view())


async def _bootstrap_state() -> None:
    """Init state from scenario; fire any beats already due (anchored beats
    with empty preconditions land here on the first tick)."""
    global STATE
    assert SCENARIO is not None
    STATE = InboxState.from_scenario(SCENARIO)
    # Boot tick fires AnchoredBeats with empty preconditions (e.g.
    # aria.welcome). PlotBeats are auto-pended too but require a player /send
    # to satisfy `player_replies_to_thread_gte`, so they can't fire here.
    await _tick(player_body=None, in_response_to=None)


# ───────────────────────── DTO helpers ─────────────────────────


def _email_summary(email: Email) -> dict:
    """Compact email row for inbox list views."""
    return {
        "id": email.id,
        "from": email.sender,
        "to": email.recipient,
        "subject": email.subject,
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
        "in_reply_to_beat": email.in_reply_to_beat,
    }


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
