# Inbox

Single-surface email correspondence game. The player is the protagonist; an AI counterpart (the only NPC) replies after the player's actions advance a calibration counter past a threshold; how warmly the player writes shifts that counterpart's voice register, which the email-rendering LLM picks up on the next response.

The current scenario, **"Continuity Program"**, is a corporate-satire pilot. You've been "enrolled" with an AI counterpart — Aria — who is calibrating on your communication style "to assume agreed-upon portions of your role." Severance-flat HR-bright tone. The horror is structural, never named.

## Architecture

```
┌──────────┐  HTTP/SSE  ┌──────────────┐  spawn  ┌──────────────┐
│ Browser  │ ─────────► │ npc_service  │ ──────► │ llama-server │
│ (Vite +  │            │ (FastAPI +   │  /v1/   │ (llama.cpp)  │
│  React)  │ ◄───────── │  Pydantic AI)│ ◄────── │              │
└──────────┘            └──────────────┘         └──────────────┘
```

- **npc_service** owns the deterministic spine (state, beats, scheduler) and runs two Pydantic AI agents: `EMAIL_AGENT` (renders NPC replies given dossier + voice band + transcript + beat spec) and `TEXTURE_AGENT` (scores player reply warmth on -3..3). Spawns and manages llama-server's lifecycle.
- **Browser** is a thin Vite+React client over the FastAPI surface. Server is the source of truth; the Zustand store is a render cache and refreshes after every mutation.
- **Two player verbs**: `/open` (mark email read) and `/send` (reply). Both advance a single `calibration` quality (0–100, the player-visible "Counterpart fidelity %") and tick the scheduler. There is no clock, no `/wait`, no `/sleep`, no archive. Beats fire when their authored predicates pass — typically `quality_gte` against `calibration` or `player_replies_to_thread_gte`. Aria's reply asynchrony falls out of authoring: each queued reply carries a runtime `quality_gte calibration` threshold above the post-send value, so a single send doesn't satisfy its own queued reply — a second send (or several opens) trips it.

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
make test                 # 38 pytest tests (no llama needed; uses TestModel)
make typecheck            # tsc on web/
```

`test_spine.py` covers predicate evaluation (including `BeatFired` and `Not` for mutual exclusion), effect application, calibration delta and cap, beat scheduling with `runtime_preconditions`, terminal mutual-exclusion ordering, voice-band selection, and `schedule_npc_reply`. `test_app.py` boots the FastAPI app with TestModel-overridden agents and exercises `/scenario`, `/inbox`, `/open` (advances calibration), `/send` (rejects empty body, runs texture, queues reply), and end-to-end reply landing after a second send.

## Smoke-test the texture loop

After `make run`:

1. Inbox shows one email: Aria's "Welcome — Calibration Session 1 of 12". Header pill reads "Counterpart fidelity: 0%".
2. Open the email — fidelity ticks to 1%. Read it. The premise is presented as a perk.
3. Reply with something curt ("noted." or "fine."). Fidelity jumps by `SEND_CALIBRATION_DELTA` (5 by default → 6%). No reply yet — the queued threshold sits above current fidelity.
4. Reply again with something else short. Fidelity ticks to 11%, past the queued threshold. Aria's reply lands in the **corrective** band — clinical, references "engagement comfort" and the "Wellness portal."
5. Hit the back arrow, `/reset` via `curl -X POST http://localhost:8000/reset`, refresh.
6. This time reply with something warm and personal ("thank you so much, looking forward to working together"). Repeat. Aria's next reply lands in the **affirming** band — clingy-bright, talks about "our partnership" and how "wonderful" the engagement is.

That's the verb. If the two paths produce visibly different prose, the loop works.

## Authoring a new scenario

`npc_service/scenario.yaml` is the authoring surface. Schema is validated strictly at sidecar startup (see `npc_service/src/npc_service/models.py`):

- `title`, `premise` (no `max_hours` — the time mechanic is gone)
- `qualities`: `calibration` (0–100, the doom counter), per-NPC `warmth.<id>` (cumulative, drives terminal-beat predicates) and `register.<id>` (range -3..3, drives voice band selection)
- `npcs`: each with `display_name`, `relationship_to_owner`, `base_voice_register`, `will_say` / `will_not_say`, `voice_bands` (each with `register_min/max`, `vocabulary_examples`, `tone_examples`), `register_quality`, `reply_delay_calibration: [min, max]` (must exceed `SEND_CALIBRATION_DELTA`)
- `threads`: each with optional `npc:` (system threads have `npc: null`). **Beat IDs must be prefixed with their thread ID** — the spine treats the prefix as the source of truth, and the loader validates the match.
- `beats`: tagged on `type:` (`anchored` or `rendered`). Common fields: `id`, `thread`, `preconditions` (list of predicates), `in_boot_inbox`, `effects`. Anchored beats add `from`, `to`, `subject`, `authored_prose`, `sent_at`. Rendered beats add `npc`, `from`, `to`, `subject`, `spec` (with `intent`, `must_include_facts`, `must_not`, `length_words`).
- `predicates` available: `quality_gte`, `quality_lt`, `player_replied_to`, `player_replies_to_thread_gte`, `beat_fired` (use under `not` for mutual exclusion of beats sharing a tick), and the logical compositions `all`, `any`, `not`.

See `CLAUDE.md` for architectural invariants ("things that look removable but aren't") that AI assistants editing this code should treat as load-bearing.

## Project layout

```
smallville/
├── README.md                 ← this file
├── CLAUDE.md                 ← architecture invariants for AI assistants
├── Makefile
├── npc_service/              ← Python sidecar (uv project)
│   ├── src/npc_service/
│   │   ├── main.py           ← FastAPI app + endpoints + SSE
│   │   ├── models.py         ← Pydantic schemas (authoring + runtime + DTOs)
│   │   ├── spine.py          ← deterministic state spine
│   │   ├── agents.py         ← EMAIL_AGENT + TEXTURE_AGENT
│   │   ├── launcher.py       ← llama-server subprocess + Job Object
│   │   └── config.py         ← pydantic-settings (env + .env)
│   ├── scenario.yaml         ← Continuity Program (1 NPC, 7 beats, calibration-driven)
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
- **Aria never replies** — confirm `reply_delay_calibration` (in `scenario.yaml`'s NPC dossier) is configured. If the lower bound is below `SEND_CALIBRATION_DELTA`, a single send may already satisfy the queued threshold and asynchrony collapses; if it's far above, the player needs many actions to trigger a reply. Defaults: send +5, reply delay [2, 5].
