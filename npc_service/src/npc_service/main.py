"""FastAPI surface for Inbox. See CLAUDE.md for the architecture and
load-bearing invariants.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import logfire
from logfire import ConsoleOptions
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agents import (
    AgentBundle,
    ChatDeps,
    build_agents,
    build_classifier_user_prompt,
)
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
    Beat,
    ClassificationResult,
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
    evaluate_completes_when,
    find_due_beats,
    reply_subject_for_email,
)


_BOOT_SETTINGS = Settings()
logging.basicConfig(
    level=_BOOT_SETTINGS.npc_log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("npc_service")

# Logfire credentials live next to the npc_service project root.
_logfire_creds = Path(__file__).resolve().parents[2] / ".logfire" / "logfire_credentials.json"
_logfire_token = os.getenv("LOGFIRE_TOKEN")
if not _logfire_token and _logfire_creds.is_file():
    with contextlib.suppress(Exception):
        _logfire_token = json.loads(_logfire_creds.read_text()).get("token")
logfire.configure(
    service_name="npc_service",
    send_to_logfire="if-token-present",
    token=_logfire_token,
    console=ConsoleOptions() if _BOOT_SETTINGS.npc_logfire_console else False,
    # Single-player local game with no PII; default scrubbing would redact
    # email bodies and rationales — exactly the prose we want to inspect.
    scrubbing=False,
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)
logging.getLogger().addHandler(logfire.LogfireLoggingHandler())


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
            await wait_healthy(
                host=cfg.host, port=cfg.port,
                timeout=EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS,
            )
        except LlamaStartupError as e:
            log.warning("llama-server not reachable yet: %s — sidecar will boot anyway", e)

    base_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
    model_name = "gemma-4-e4b"
    log.info("building agents: model=%s base_url=%s", model_name, base_url)
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key="not-needed"),
    )
    AGENTS = build_agents(model)

    await _bootstrap_state()
    log.info(
        "npc_service ready: %d boot emails, %d pending beats",
        len(STATE.emails), len(STATE.pending_beats),
    )
    try:
        yield
    finally:
        log.info("shutting down")
        stop_llama(LLAMA_PROC)


app = FastAPI(lifespan=lifespan, title="npc_service (Inbox)")
logfire.instrument_fastapi(app, capture_headers=False)


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

    # If a multi-turn beat's gate is already satisfied (e.g. the
    # just-applied classifier bumped its quality past the threshold),
    # complete silently — no render — and let the chain advance.
    if beat.completes_when and evaluate_completes_when(STATE, beat):
        STATE.fired_beat_ids.add(beat.id)
        logfire.info("beat completed without render", beat_id=beat.id)
        return None

    with logfire.span("fire_beat", beat_id=beat.id, beat_type=type(beat).__name__):
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
        elif isinstance(beat, PlotBeat):
            assert player_body is not None, (
                "PlotBeat fires require a player body — they trigger only on /send ticks"
            )
            email = await _render_via_chat_agent(
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
    """Single rendering path used by both chat turns (no plot directive) and
    PlotBeat fires (directive injected via deps).

    Appends `result.new_messages()` to thread_messages so future turns see
    the player's body and the NPC's reply as the next two messages in the
    Pydantic AI conversation.
    """
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    npc = SCENARIO.npc_by_id[npc_id]
    deps = ChatDeps(npc=npc, plot_directive=plot_directive)
    history = STATE.thread_messages.setdefault(thread_id, [])

    with logfire.span(
        "render_chat",
        npc=npc_id,
        thread=thread_id,
        beat=beat_id,
        plot=plot_directive is not None,
        history_len=len(history),
    ) as span:
        result = await AGENTS.chat_agent.run(
            player_body,
            message_history=list(history),
            deps=deps,
        )
        history.extend(result.new_messages())
        usage = result.usage()
        span.set_attribute("input_tokens", usage.input_tokens)
        span.set_attribute("output_tokens", usage.output_tokens)

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


# ───────────────────────── scheduler tick ─────────────────────────


async def _tick(
    *,
    player_body: str | None = None,
    in_response_to: str | None = None,
) -> tuple[list[Email], set[str]]:
    """Walk the beat scheduler. Returns (newly-emitted emails, set of thread
    ids where a PlotBeat fired this tick). The thread-id set lets the caller
    decide whether a chat-turn fallback is still needed.
    """
    assert STATE is not None and SCENARIO is not None
    new_emails: list[Email] = []
    plot_fired_threads: set[str] = set()

    with logfire.span("tick", in_response_to=in_response_to) as span:
        for beat in find_due_beats(STATE, SCENARIO):
            e = await _fire_beat(
                beat,
                player_body=player_body,
                in_response_to=in_response_to,
            )
            if e is not None:
                new_emails.append(e)
                if isinstance(beat, PlotBeat):
                    plot_fired_threads.add(beat.thread)
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
    return {
        "title": SCENARIO.title,
        "premise": SCENARIO.premise,
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

    with logfire.span("send_stream", recipient=req.recipient, thread_id=thread_id):
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
            STATE.replies.append(Reply(
                id=f"reply_{uuid.uuid4().hex[:8]}",
                thread_id=thread_id,
                beat_responded_to=req.in_reply_to,
                recipient=req.recipient,
                body=body,
            ))
            yield _sse({"type": "send_recorded", "email": _email_full(send_email)})

            # Classifier runs serially before _tick because the replied-to
            # beat may consume its own classifier output (e.g. aria.loop's
            # `accumulate=true` engagement classifier whose value gates the
            # complete_warm vs complete_standard split). _tick must see the
            # post-classifier qualities to make terminal decisions.
            recipient_npc = _infer_recipient_npc_id(req.recipient)
            replied_beat = (
                SCENARIO.beat_by_id.get(in_reply_email.beat_id)
                if in_reply_email and in_reply_email.beat_id else None
            )
            classify = replied_beat.classify_reply if replied_beat else None
            if classify is not None:
                classification = await _classify_player_reply(
                    recipient_blurb=_persona_blurb_for_recipient(req.recipient, recipient_npc),
                    prior_email_body=in_reply_email.body if in_reply_email else "",
                    reply_body=body,
                    classify=classify,
                )
                if classification is not None:
                    applied = apply_classification(STATE, SCENARIO, classify, classification.value)
                    if applied:
                        yield _sse({
                            "type": "classified",
                            "quality": classify.quality,
                            "value": classification.value,
                            "score": classify.map[classification.value],
                        })
                    else:
                        logfire.warning(
                            "classifier returned off-spec value; quality unchanged",
                            beat_id=replied_beat.id if replied_beat else None,
                            value=classification.value,
                            allowed=classify.allowed_values,
                        )

            new_emails, plot_fired_threads = await _tick(
                player_body=body,
                in_response_to=send_email.id,
            )

            # Chat-turn fallback fires when no PlotBeat covered this thread.
            # In the steady state aria.loop is multi-turn and fires on every
            # tick after the welcome reply, so this branch only matters for
            # threads without a covering loop beat.
            if recipient_npc is not None and thread_id is not None and thread_id not in plot_fired_threads:
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

            for ne in new_emails:
                yield _sse({"type": "new_email", "email": _email_full(ne)})

            yield _sse({"type": "state", "state": STATE.public_view()})
            yield _sse({"type": "done"})


async def _classify_player_reply(
    *,
    recipient_blurb: str,
    prior_email_body: str,
    reply_body: str,
    classify: ClassifyReply,
) -> ClassificationResult | None:
    """Run the classifier against the player's reply using the beat-authored
    question and allowed values. Returns None on failure so the caller skips
    the quality write."""
    assert AGENTS is not None
    try:
        result = await AGENTS.classifier_agent.run(
            build_classifier_user_prompt(
                recipient_blurb, prior_email_body, reply_body,
                classify.question, classify.allowed_values,
            )
        )
        out = result.output
        logfire.info(
            "classifier scored",
            value=out.value, rationale=out.rationale, quality=classify.quality,
        )
        return out
    except Exception:
        logfire.exception("classifier failed; quality unchanged")
        return None


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
    log.info("/reset")
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
