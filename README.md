# Inbox

Single-surface email correspondence game. The player is the protagonist; an AI counterpart (the only NPC) replies on every send. Authored "plot beats" (calibration milestones, terminal endings) inject one-shot directives onto the chat agent's next turn; the rest is free chat in character. How warmly the player writes shifts the counterpart's voice register and accelerates the calibration meter.

The current scenario, **"Continuity Program"**, is a corporate-satire pilot. You've been "enrolled" with an AI counterpart — Aria — who is calibrating on your communication style "to assume agreed-upon portions of your role." Severance-flat HR-bright tone. The horror is structural, never named.

## Architecture

```
┌──────────┐  HTTP/SSE  ┌──────────────┐  spawn  ┌──────────────┐
│ Browser  │ ─────────► │ npc_service  │ ──────► │ llama-server │
│ (Vite +  │            │ (FastAPI +   │  /v1/   │ (llama.cpp)  │
│  React)  │ ◄───────── │  Pydantic AI)│ ◄────── │              │
└──────────┘            └──────────────┘         └──────────────┘
```

- **npc_service** owns the deterministic spine (state, beats, scheduler) and runs two Pydantic AI agents:
  - `chat_agent` — `output_type=str`, `deps_type=ChatDeps`, runs once per NPC reply with per-thread `message_history`. Per-run `@instructions` decorators inject the dossier, the selected voice band, and (when set) an authored plot directive. Used for both default chat turns (`plot_directive=None`) and PlotBeat fires.
  - `texture_agent` — `ToolOutput`, scores player reply warmth on -3..3.
  - Spawns and manages llama-server's lifecycle.
- **Browser** is a thin Vite+React client over the FastAPI surface. Server is the source of truth; the Zustand store is a render cache and refreshes after every mutation.
- **Two player verbs**: `/open` (mark email read; does NOT mutate calibration) and `/send` (reply). `/send` advances the `calibration` quality (0–100, the player-visible "Counterpart fidelity %") by `base + coef·|warmth|` — strong signal in either direction calibrates fast, lukewarm sends barely move it.
- **Two reply paths**:
  - **Chat turn (default)** — fires immediately within the same `/send` SSE. Free conversation in character; no per-turn authoring.
  - **PlotBeat (authored milestone)** — when `find_eligible_plot_beat` matches, `schedule_thread_reply` queues it with a calibration-threshold runtime predicate. Above the post-send value, so the just-queued PlotBeat does NOT fire on its own scheduling tick. A subsequent /send (especially a warmth-charged one) ticks calibration past the threshold and the PlotBeat lands as the next chat-agent turn, with the spec's intent / must-include / must-not / length window injected as per-run instructions.

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Node 18+ with npm
- `llama-server` from llama.cpp (bundled at `vendor/llama.cpp/llama-server.exe`, or built from source with CUDA)
- ~8 GB free disk for the Q8_0 GGUF (auto-downloaded by `make run`)

## Run

```bash
make run     # boots sidecar (spawns llama) + web dev server in parallel
```

Browser opens at http://localhost:5173. First run downloads the model and takes ~90 s for cold load; subsequent runs are fast.

To run layers separately:

```bash
make llama       # terminal 1: llama-server
make sidecar     # terminal 2: Python sidecar (LAUNCH_LLAMA=false)
make web         # terminal 3: Vite dev server
```

The two flags `--jinja` and `--reasoning off` on llama-server are non-negotiable for Gemma 4 E-models — without them you get either an empty response or a 19K-token monologue inside a hidden `<think>` block. The bundled launcher sets both.

## Tests

```bash
make test                 # 51 pytest tests (no llama needed; uses TestModel)
make typecheck            # tsc on web/
```

`test_spine.py` covers predicate evaluation (including `BeatFired` and `Not_` for mutual exclusion), calibration delta + cap, `apply_calibration_from_warmth` (parametrized across the warmth range), the texture/register loop with clamping, beat scheduling with `runtime_preconditions`, `find_eligible_plot_beat` (returns first eligible PlotBeat, skips already-pending or already-fired), terminal mutual-exclusion via `Not_(BeatFired)`, `schedule_thread_reply` (calibration-threshold queueing), `reply_subject_for_email`, and `latest_npc_email_in_thread`. `test_app.py` boots the FastAPI app with TestModel-overridden agents and asserts: `/open` does NOT mutate calibration; `/send` delta == `base + coef·|warmth|`; first /send returns an immediate chat-turn fallback in the same SSE; `thread_messages` accumulates `ModelRequest`/`ModelResponse` across sends; rendered bodies contain no `Subject:`/`From:`/`To:`/JSON/markdown header leakage; midprogram PlotBeat lands one /send after the threshold-crossing send.

## Smoke-test the texture loop

After `make run`:

1. Inbox shows one email: Aria's "Welcome — Calibration Session 1 of 12". Header pill reads "Counterpart fidelity: 0%".
2. Open the email. Read it. **Fidelity stays at 0%** — reading isn't a communication act; only sends move the meter.
3. Reply with something curt ("noted." or "fine."). Texture warmth ≈ -1 to -2; calibration ticks up by ~3.5 to 5.0. Aria's reply arrives **inside the same SSE** (chat-turn fallback). Voice band is `standard`.
4. Reply curt again, twice. Warmth accumulates negative; `register.aria` drifts negative. Aria's third or fourth reply lands in the **corrective** band — clinical, references "engagement comfort" and the "Wellness portal."
5. `curl -X POST http://localhost:8000/reset` and refresh.
6. Reply with something warm and personal ("thank you so much, looking forward to working together"). Repeat warmly. Aria's next reply lands in the **affirming** band — clingy-bright, "our partnership", "wonderful."
7. Continue replying — after the third reply in the thread, the `aria.midprogram` PlotBeat is queued (with a calibration-threshold delay). Your next warmth-charged send ticks calibration past the threshold and you receive the milestone email — subject "Calibration Milestone — Counterpart Fidelity Update", referencing fidelity %, asking a more personal calibration prompt.

If chat turns arrive immediately, the meter only moves on /send (not /open), the voice band shifts visibly with warmth, and the midprogram PlotBeat lands one send AFTER its precondition is satisfied — the loop works.

## Authoring a new scenario

`npc_service/scenario.yaml` is the authoring surface. Schema is validated strictly at sidecar startup (see `npc_service/src/npc_service/models.py`):

- `title`, `premise`
- `qualities`: `calibration` (0–100, the doom counter), per-NPC `warmth.<id>` (cumulative, drives terminal-beat predicates) and `register.<id>` (range -3..3, drives voice band selection)
- `npcs`: each with `display_name`, `email_address` (sender address used for chat-turn emails), `relationship_to_owner`, `base_voice_register`, `will_say` / `will_not_say`, `voice_bands` (each with `register_min/max`, `vocabulary_examples`, `tone_examples`), `register_quality`, `reply_delay_calibration: <float>` (used only by PlotBeat queueing; chat turns ignore it). The delay must exceed the typical post-send calibration delta or PlotBeats will fire on their own scheduling tick and lose the deliberate-pacing feel.
- `threads`: each with optional `npc:` (system threads have `npc: null`). **Beat IDs must be prefixed with their thread ID** — the spine treats the prefix as the source of truth, and the loader validates the match.
- `beats`: tagged on `type:` (`anchored` or `plot`). Common fields: `id`, `thread`, `preconditions` (list of predicates).
  - **Anchored** beats add `from`, `to`, `subject`, `authored_prose`, `sent_at`. They ship verbatim. Empty preconditions → fires on the first scheduler tick at boot. Anchored beats on NPC threads also seed the chat agent's `message_history` so subsequent runs see the authored prose as a prior assistant turn.
  - **Plot** beats add `npc`, `subject`, `spec` (with `intent`, `must_include_facts`, `must_not`, `length_words`). They're queued only by `schedule_thread_reply` after the player /sends and `find_eligible_plot_beat` matches. When fired, the chat agent runs with the spec injected as a per-run instruction; the agent's prose IS the email body, and the authored `subject` overrides the chat-turn-default `Re:` derivation.
- `predicates` available: `quality_gte`, `quality_lt`, `player_replies_to_thread_gte`, `beat_fired`, and `not` (use `not` over `beat_fired` to mutually exclude two beats sharing a tick).

Want a thread that's pure free chat with no plot moments? Author the dossier + thread and write zero PlotBeats. Every player /send produces an immediate chat turn driven entirely by dossier + voice-band + accumulated `message_history`. The conversation can run as long as the player keeps writing.

See `CLAUDE.md` for architectural invariants ("things that look removable but aren't") that AI assistants editing this code should treat as load-bearing.

## Project layout

```
smallville/
├── README.md                 ← this file
├── CLAUDE.md                 ← architecture invariants for AI assistants
├── Makefile
├── npc_service/              ← Python sidecar (uv project)
│   ├── src/npc_service/
│   │   ├── main.py           ← FastAPI app + endpoints + SSE; chat-turn vs PlotBeat dispatch
│   │   ├── models.py         ← Pydantic schemas (authoring + runtime + DTOs); thread_messages on InboxState
│   │   ├── spine.py          ← deterministic state spine; find_eligible_plot_beat, schedule_thread_reply
│   │   ├── agents.py         ← chat_agent (str output, deps-driven instructions) + texture_agent
│   │   ├── launcher.py       ← llama-server subprocess + Job Object
│   │   └── config.py         ← pydantic-settings (calibration_base, calibration_warmth_coef)
│   ├── scenario.yaml         ← Continuity Program (1 NPC, 6 beats: 1 welcome + 3 PlotBeats + 2 system anchored)
│   ├── tests/
│   └── pyproject.toml
└── web/                      ← Vite + React frontend
    ├── src/
    │   ├── App.tsx, main.tsx
    │   ├── api.ts            ← fetch + SSE wrappers
    │   ├── store.ts          ← Zustand store
    │   ├── hooks/useStreamSend.ts
    │   ├── util/email.ts     ← stripAddress
    │   └── components/       ← InboxHeader, FolderTabs, InboxList, ThreadView, EmailComposer
    ├── public/fonts/         ← IBM Plex Mono
    ├── package.json
    └── vite.config.ts
```

## Troubleshooting

- **Empty agent responses or 19K-token monologues** — `--jinja` and/or `--reasoning off` missing on llama-server. The bundled launcher sets both; if you run llama-server yourself, include them.
- **`No module named npc_service`** — run `uv sync` from `npc_service/`.
- **Streaming arrives in big bursts** — verify your reverse proxy isn't buffering. The server sets `X-Accel-Buffering: no` on the `/send` SSE response.
- **Port 8000 already in use** — set `NPC_SERVICE_PORT` in `npc_service/.env` and update `web/vite.config.ts`'s proxy target.
- **Aria never replies** — chat-turn fallback should fire on every /send. If it doesn't, the player's /send body is empty (rejected with HTTP 400) or the recipient string doesn't match an NPC's display_name (`_infer_recipient_npc_id` returns None and no chat turn is rendered). Check the texture agent ran (look for the `texture` SSE event).
- **Plot beat lands on the same send as it's queued** — `reply_delay_calibration` is below the post-send delta. Increase it. Default 3.5; with `calibration_base=2.0` and `coef=1.5`, only sends with |warmth| ≥ 1 trip the threshold on the next send (warmth=0 → delta 2.0 < 3.5).
- **Chat turns leak `Subject:` / `From:` headers** — the chat agent's base instructions block this. Test `test_chat_turn_body_does_not_leak_email_headers` is the regression guard. If it fires under a real LLM, tighten `_BASE_CHAT_INSTRUCTIONS` in `agents.py`.
