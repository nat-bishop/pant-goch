"""FastAPI surface for Inbox.

The two LLM agents (see agents.py) are constructed in lifespan against one
shared model handle. The deterministic spine (spine.py) owns all state
mutation; agents return prose / scores only.
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
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agents import (
    AgentBundle,
    build_agents,
    build_email_user_prompt,
    build_texture_user_prompt,
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
    ArchiveRequest,
    Beat,
    Email,
    InboxState,
    OpenRequest,
    RenderedBeat,
    Reply,
    Scenario,
    SendRequest,
    TextureScore,
    load_scenario,
)
from .spine import (
    advance_time,
    apply_effect,
    apply_texture_score,
    check_end_conditions,
    find_due_beats,
    initialize_boot_inbox,
    render_thread_transcript,
    schedule_npc_reply,
    select_voice_band,
    thread_id_from_beat_id,
)


# Time costs per verb (game-time hours). Reading is free; sending takes a
# realistic 15 minutes; waiting and sleeping are deliberate clock pushes.
SEND_HOURS = 0.25
WAIT_HOURS = 2.0
SLEEP_HOURS = 8.0


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
LLAMA_PROC: subprocess.Popen | None = None
ACTION_LOCK = asyncio.Lock()  # serialises mutating endpoints
EXTERNAL_LLAMA_PROBE_TIMEOUT_SECONDS = 2.0  # tests set this to 0.1 for fast boot


# ───────────────────────── lifespan ─────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global SCENARIO, STATE, AGENTS, LLAMA_PROC

    settings = Settings()
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


async def _fire_beat(beat: Beat, in_response_to: str | None) -> Email | None:
    # `in_response_to` chains rendered NPC replies onto the player's send so
    # the next beat reads the right transcript.
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    if beat.id in STATE.fired_beat_ids:
        return None

    with logfire.span("fire_beat", beat_id=beat.id, beat_type=type(beat).__name__):
        email: Email | None = None
        if isinstance(beat, AnchoredBeat):
            email = Email(
                id=f"email_{uuid.uuid4().hex[:8]}",
                beat_id=beat.id,
                thread_id=beat.thread,
                folder=beat.folder,
                sender=beat.from_,
                recipient=beat.to,
                subject=beat.subject,
                body=beat.authored_prose.strip(),
                sent_at_label=beat.sent_at or "just now",
                sent_at_hours=STATE.game_time_hours,
            )
        elif isinstance(beat, RenderedBeat):
            email = await _render_beat_email(beat, in_response_to)

        if email is not None:
            STATE.emails.append(email)

        for eff in beat.effects:
            apply_effect(eff, STATE, SCENARIO)
        STATE.fired_beat_ids.add(beat.id)
        return email


async def _render_beat_email(beat: RenderedBeat, in_response_to: str | None) -> Email:
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    npc = SCENARIO.npc_by_id[beat.npc]
    transcript = render_thread_transcript(STATE, beat.thread)
    user_prompt = build_email_user_prompt(beat, npc, STATE, transcript)
    band = select_voice_band(npc, STATE)
    band_name = band.name if band else None

    with logfire.span(
        "render_beat",
        beat_id=beat.id,
        npc=beat.npc,
        band=band_name,
    ) as span:
        result = await AGENTS.email_agent.run(user_prompt)
        usage = result.usage()
        span.set_attribute("input_tokens", usage.input_tokens)
        span.set_attribute("output_tokens", usage.output_tokens)

    return Email(
        id=f"email_{uuid.uuid4().hex[:8]}",
        beat_id=beat.id,
        thread_id=beat.thread,
        folder="Inbox",
        sender=beat.from_,
        recipient=beat.to,
        subject=beat.subject,
        body=result.output.prose.strip(),
        sent_at_label="just now",
        sent_at_hours=STATE.game_time_hours,
        in_reply_to_beat=in_response_to,
        voice_band_used=band_name,
    )


# ───────────────────────── scheduler tick ─────────────────────────


async def _tick(in_response_to: str | None = None) -> list[Email]:
    """Walk the beat scheduler. Returns the list of newly-emitted emails this
    tick (for SSE).
    """
    assert STATE is not None and SCENARIO is not None
    new_emails: list[Email] = []

    with logfire.span("tick", in_response_to=in_response_to) as span:
        for bf in find_due_beats(STATE, SCENARIO):
            e = await _fire_beat(bf.beat, in_response_to=in_response_to)
            if e is not None:
                new_emails.append(e)

        check_end_conditions(STATE)
        span.set_attribute("new_email_count", len(new_emails))
        return new_emails


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
        "max_hours": SCENARIO.max_hours,
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
    folders: dict[str, list[dict]] = {"Inbox": [], "Sent": [], "Archive": []}
    for e in STATE.emails:
        folders.setdefault(e.folder, []).append(_email_summary(e))
    # Sort within each folder by sent_at_hours desc (newest first).
    for k in folders:
        folders[k].sort(key=lambda x: -x["sent_at_hours"])
    return {"folders": folders, "state": STATE.public_view()}


@app.get("/thread/{thread_id}")
async def get_thread(thread_id: str) -> dict:
    """All emails in a thread, ordered by sent_at_hours."""
    if STATE is None or SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if thread_id not in SCENARIO.thread_by_id:
        raise HTTPException(404, f"unknown thread: {thread_id}")
    emails = [_email_full(e) for e in STATE.emails if e.thread_id == thread_id]
    emails.sort(key=lambda x: x["sent_at_hours"])
    return {"thread_id": thread_id, "emails": emails}


@app.post("/open")
async def post_open(req: OpenRequest) -> dict:
    """First-read marks the email as read. Free — does not advance the clock."""
    if STATE is None:
        raise HTTPException(503, "state not initialised")
    if STATE.ended:
        raise HTTPException(409, f"session ended ({STATE.end_reason})")

    async with ACTION_LOCK:
        email = next((e for e in STATE.emails if e.id == req.email_id), None)
        if email is None:
            raise HTTPException(404, f"unknown email_id: {req.email_id}")
        email.is_read = True
        return {"state": STATE.public_view(), "new_emails": []}


@app.post("/archive")
async def post_archive(req: ArchiveRequest) -> dict:
    """Move email to Archive. Free — does not advance the clock."""
    if STATE is None:
        raise HTTPException(503, "state not initialised")
    if STATE.ended:
        raise HTTPException(409, f"session ended ({STATE.end_reason})")

    async with ACTION_LOCK:
        email = next((e for e in STATE.emails if e.id == req.email_id), None)
        if email is None:
            raise HTTPException(404, f"unknown email_id: {req.email_id}")
        email.folder = "Archive"
        email.is_read = True
        return {"state": STATE.public_view(), "new_emails": []}


@app.post("/wait")
async def post_wait() -> dict:
    """Advance the world clock by WAIT_HOURS, tick the scheduler."""
    if STATE is None:
        raise HTTPException(503, "state not initialised")
    if STATE.ended:
        raise HTTPException(409, f"session ended ({STATE.end_reason})")

    async with ACTION_LOCK:
        advance_time(STATE, WAIT_HOURS)
        new_emails = await _tick(in_response_to=None)
        return {
            "state": STATE.public_view(),
            "new_emails": [_email_full(e) for e in new_emails],
        }


@app.post("/sleep")
async def post_sleep() -> dict:
    """Advance the world clock by SLEEP_HOURS (overnight), tick the scheduler."""
    if STATE is None:
        raise HTTPException(503, "state not initialised")
    if STATE.ended:
        raise HTTPException(409, f"session ended ({STATE.end_reason})")

    async with ACTION_LOCK:
        advance_time(STATE, SLEEP_HOURS)
        new_emails = await _tick(in_response_to=None)
        return {
            "state": STATE.public_view(),
            "new_emails": [_email_full(e) for e in new_emails],
        }


@app.post("/send")
async def post_send(req: SendRequest) -> StreamingResponse:
    if STATE is None or AGENTS is None or SCENARIO is None:
        raise HTTPException(503, "state not initialised")
    if STATE.ended:
        raise HTTPException(409, f"session ended ({STATE.end_reason})")
    return StreamingResponse(
        _send_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _send_stream(req: SendRequest) -> AsyncIterator[bytes]:
    assert STATE is not None and SCENARIO is not None and AGENTS is not None
    body = req.body if req.body.strip() else "got it."  # placeholder fallback

    thread_id = thread_id_from_beat_id(req.in_reply_to)

    with logfire.span("send_stream", recipient=req.recipient, thread_id=thread_id):
        async with ACTION_LOCK:
            advance_time(STATE, SEND_HOURS)

            send_email = Email(
                id=f"email_{uuid.uuid4().hex[:8]}",
                beat_id=None,
                thread_id=thread_id,
                folder="Sent",
                sender="you",
                recipient=req.recipient,
                subject=_reply_subject(req.in_reply_to),
                body=body,
                sent_at_label="just now",
                sent_at_hours=STATE.game_time_hours,
                in_reply_to_beat=req.in_reply_to,
                is_read=True,
            )
            STATE.emails.append(send_email)
            yield _sse({"type": "send_recorded", "email": _email_full(send_email)})

            recipient_npc = _infer_recipient_npc_id(req.recipient)
            recipient_blurb = _persona_blurb_for_recipient(req.recipient, recipient_npc)
            transcript = render_thread_transcript(STATE, thread_id)
            try:
                tex_result = await AGENTS.texture_agent.run(
                    build_texture_user_prompt(recipient_blurb, transcript, body)
                )
                tex = tex_result.output
                logfire.info(
                    "texture scored",
                    warmth=tex.warmth,
                    rationale=tex.rationale,
                )
            except Exception:
                logfire.exception("texture scoring failed; defaulting to neutral")
                tex = TextureScore(warmth=0.0, rationale="scorer-failed")

            STATE.replies.append(Reply(
                id=f"reply_{uuid.uuid4().hex[:8]}",
                thread_id=thread_id,
                beat_responded_to=req.in_reply_to,
                recipient=req.recipient,
                body=body,
                warmth=tex.warmth,
                rationale=tex.rationale,
                sent_at_hours=STATE.game_time_hours,
            ))
            apply_texture_score(STATE, SCENARIO, recipient_npc, tex.warmth)
            yield _sse({"type": "texture", "warmth": tex.warmth})

            if recipient_npc is not None:
                schedule_npc_reply(STATE, SCENARIO, recipient_npc)

            # Time has advanced by SEND_HOURS, so a time-gated bounce that
            # just crossed its threshold can fire on this tick.
            new_emails = await _tick(in_response_to=send_email.id)
            for ne in new_emails:
                yield _sse({"type": "new_email", "email": _email_full(ne)})
            yield _sse({"type": "state", "state": STATE.public_view()})
            yield _sse({"type": "done"})


def _reply_subject(in_reply_to: str | None) -> str:
    if STATE is None or in_reply_to is None:
        return "(no subject)"
    target = next((e for e in STATE.emails if e.beat_id == in_reply_to), None)
    if target is None:
        return "(no subject)"
    s = target.subject
    return s if s.lower().startswith("re:") else f"Re: {s}"


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
    """Init state from scenario, fire boot inbox, then any beats already due."""
    global STATE
    assert SCENARIO is not None
    STATE = InboxState.from_scenario(SCENARIO)
    for bf in initialize_boot_inbox(STATE, SCENARIO):
        await _fire_beat(bf.beat, in_response_to=None)
    for bf in find_due_beats(STATE, SCENARIO):
        await _fire_beat(bf.beat, in_response_to=None)


# ───────────────────────── DTO helpers ─────────────────────────


def _email_summary(email: Email) -> dict:
    """Compact email row for inbox list views."""
    return {
        "id": email.id,
        "from": email.sender,
        "to": email.recipient,
        "subject": email.subject,
        "sent_at": email.sent_at_label,
        "sent_at_hours": email.sent_at_hours,
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
        "voice_band_used": email.voice_band_used,
    }


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
