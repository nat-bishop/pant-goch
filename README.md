# Inbox

Single-surface email correspondence game. The player is the protagonist; NPCs reply after a human-feeling delay; how warmly the player writes shifts each NPC's voice register, which the email-rendering LLM picks up on the next response.

The current scenario, **"Threads"**, opens on a delivery-failure bounce showing your account sent an email to a stranger you don't recognize. Someone is sending email as you. Sam (tech-savvy friend, fast replier) and Alex (sibling, slow replier) get pulled in. You triangulate.

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
- **The player verb that matters** is `/send`. It costs 0.25 hours of game time, runs TEXTURE_AGENT to score warmth, drifts the NPC's `register` quality halfway toward the new score, and schedules that NPC's next eligible beat for `now + midpoint(latency_range)`. `/open` and `/archive` are free; `/wait` advances 2 hours and `/sleep` advances 8 hours, both ticking the scheduler so any due replies surface.

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
make test                 # 37 pytest tests (no llama needed; uses TestModel)
make typecheck            # tsc on web/
```

`test_spine.py` covers predicate evaluation, effect application, beat scheduling, voice band selection, time advancement, and `schedule_npc_reply`. `test_app.py` boots the FastAPI app with TestModel-overridden agents and exercises `/scenario`, `/inbox`, `/open`, `/archive`, `/send`, `/wait`, `/sleep` end-to-end.

## Smoke-test the texture loop

After `make run`:

1. Inbox shows two emails: a delivery bounce and Sam's "uhh did u just email d.cuevas".
2. Open Sam's email and reply with a curt one-liner ("idk").
3. Hit **Wait 2h** in the header. Sam's reply lands with `voice_band_used: "terse"` — short, punctuated, formal.
4. Reply again with something specific and warm ("yeah weird, can you check the spf record? want me to forward the headers?"). Hit **Wait 2h**.
5. Sam's next reply arrives in the `enthusiastic` band — lowercase, fragments, `"lmao"`s.

That's the verb. If steps 3 and 5 produce visibly different prose, the loop works.

## Authoring a new scenario

`npc_service/scenario.yaml` is the authoring surface. Schema is validated strictly at sidecar startup (see `npc_service/src/npc_service/models.py`):

- `title`, `premise`, `max_hours`
- `qualities`: per-NPC `warmth.<id>` (cumulative, drives terminal-beat predicates) and `register.<id>` (range -3..3, drives voice band selection)
- `npcs`: each with `display_name`, `relationship_to_owner`, `base_voice_register`, `will_say` / `will_not_say`, `voice_bands` (each with `register_min/max`, `vocabulary_examples`, `tone_examples`), `register_quality`, `reply_latency_hours: [min, max]`
- `threads`: each with optional `npc:` (system threads like `bounces` have `npc: null`)
- `beats`: tagged on `type:` (`anchored` or `rendered`). Common fields: `id`, `thread`, `preconditions` (list of predicates), `arrives_after_hours`, `in_boot_inbox`, `effects`. Anchored beats add `from`, `to`, `subject`, `authored_prose`. Rendered beats add `npc`, `from`, `to`, `subject`, `spec` (with `intent`, `must_include_facts`, `must_not`, `length_words`).

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
│   ├── scenario.yaml         ← Threads (2 NPCs, 14 beats, 96-hour session)
│   ├── tests/
│   └── pyproject.toml
└── web/                      ← Vite + React frontend
    ├── src/
    │   ├── App.tsx, main.tsx
    │   ├── api.ts            ← fetch + SSE wrappers
    │   ├── store.ts          ← Zustand store
    │   ├── hooks/useStreamSend.ts
    │   ├── util/email.ts     ← stripAddress + formatGameTime
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
