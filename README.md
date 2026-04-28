# Smallville — Hostage Negotiation MVP

Single-scenario hostage negotiation game. Player negotiates with one
LLM-driven captor holding two hostages. ~30-minute scenario; resolution
computed from real game state, not LLM-narrated.

## Architecture

```
┌──────────┐  spawn   ┌──────────────┐  spawn   ┌──────────────┐
│ Godot 4  │ ───────► │ npc_service  │ ───────► │ llama-server │
│ (C#)     │  HTTP/   │ (FastAPI +   │  /v1/    │ (llama.cpp)  │
│          │ ◄──SSE── │  Pydantic AI)│ ◄────────│              │
└──────────┘          └──────────────┘          └──────────────┘
```

- **Godot** spawns one child (Python). On exit, kills the entire process
  tree (Python + llama-server cascade).
- **npc_service** owns llama-server lifecycle. One Pydantic AI `Agent`
  with `output_type=CaptorResponse` (structured output enforced).
- **Hidden state** lives server-side: stress / trust / desperation
  meters (0–100). Player never sees numbers — only the captor's
  behaviour.
- **CaptorResponse** (text + meter deltas + optional action) is the
  single bridge from LLM to game state. Even if `response_text` is
  gibberish, the game progresses coherently.
- **Timeline events** (asyncio task in lifespan) inject scripted scene
  updates at fixed timestamps so the world moves whether or not the
  player acts.

## Prerequisites

- Python 3.11+ (`uv` for dependency management — install from
  https://docs.astral.sh/uv/).
- .NET 8 SDK (Godot 4.6 mono ships with it; check `dotnet --version`).
- Godot 4.6 mono (works with 4.4+; .NET edition required).
- `llama-server` from llama.cpp ([releases](https://github.com/ggerganov/llama.cpp/releases))
  or built locally with CUDA support.
- ~10 GB free disk for the Q8_0 GGUF (auto-downloaded on first run).

## Run end-to-end (dev)

Three terminals make iteration fastest. The single-bootstrap "double-click
the game and everything starts" path also works (see below).

### Terminal 1 — llama-server

```bash
llama-server \
  -m models/gemma-4-E4B-it-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 8192 --jinja --reasoning off -ngl 99
```

Two flags are non-negotiable for Gemma 4:

- `--jinja` — Gemma 4's chat template must be applied via Jinja.
- `--reasoning off` — Gemma 4 E-models default to thinking-on; without
  this flag, the model burns its entire generation budget inside a
  hidden `<think>` block and returns empty content.

Download the GGUF directly (the `-hf` shorthand 404s for this repo on
recent llama.cpp builds):

```bash
mkdir -p models
curl -L -o models/gemma-4-E4B-it-Q8_0.gguf \
  https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
```

(~8 GB.)

Verify with:

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"Say only the word PONG"}],"max_tokens":4}'
```

### Terminal 2 — Python sidecar

```bash
cd npc_service
uv sync                                  # one-time
PYTHONPATH=.. LAUNCH_LLAMA=false uv run python -m npc_service
```

Smoke-test:

```bash
curl -s http://127.0.0.1:8000/health
curl -N -X POST http://127.0.0.1:8000/action \
  -H "Content-Type: application/json" \
  -d '{"type":"empathize","detail":""}'
```

Expect a stream of `data: {"type":"text","text":"..."}` lines arriving
smoothly, then `{"type":"state",...}`, then `{"type":"done"}`. If text
arrives in big bursts, something is buffering.

### Terminal 3 — Godot

Open `godot_project/` in Godot 4.6 mono and press F5, OR from CLI:

```bash
godot --path godot_project --no-sidecar
```

`--no-sidecar` tells Godot to skip auto-spawning Python (since you're
running it yourself in Terminal 2).

### Single-bootstrap mode (the ship path)

```bash
godot --path godot_project
```

Godot spawns Python; Python spawns llama-server. Watch the Godot console
for `[npc] ...` log lines (Python forwarded), then `[npc:err] ...` from
llama-server. First run takes 90 s+ for cold model load; subsequent
runs are fast.

## Verification checklist

After llama-server, sidecar, and Godot are all up:

- [ ] **Streaming** — click Empathize. First captor character appears
      within ~2 s; subsequent characters arrive smoothly (not in
      multi-second bursts). Bursty = SSE buffering somewhere.
- [ ] **State mutation** — repeatedly Empathize + Offer → Vehicle. The
      hostage status panel should eventually flip one hostage to
      "RELEASED" (green badge) when trust accumulates.
- [ ] **Timeline event** — wait 5 minutes (or edit `at_seconds` in
      `scenario.json` to 10). The "Comms" panel on the left shows the
      tactical-commander notification AND Marlon's next response
      references the perimeter activity.
- [ ] **Action cancellation** — click Empathize, then immediately click
      Threaten while the response is streaming. The Empathize response
      cuts off with `[interrupted]`; the Threaten response starts
      cleanly. Marlon's next turn does not reference the cut-off line.
- [ ] **Stop** — click Empathize, then Stop while streaming. Same
      cancellation behaviour.
- [ ] **Auto-breach** — set `starting_meters.desperation: 95` in
      `scenario.json`, restart, then Threaten once. Resolution screen
      fires within seconds with reason "auto_breach".
- [ ] **Reset** — click Restart on the resolution screen. Hostages
      back to held, time reset, fresh history.

## Project layout

```
smallville/
├── README.md                  ← this file
├── npc_service/               ← Python sidecar
│   ├── main.py                ← FastAPI app, agent, SSE, timeline scheduler
│   ├── models.py              ← Pydantic DTOs + GameState
│   ├── launcher.py            ← llama-server subprocess management
│   ├── scenario.json          ← authored scenario data
│   ├── pyproject.toml
│   ├── README.md              ← endpoints + env vars + troubleshooting
│   └── ...
└── godot_project/             ← Godot 4 / C# UI
    ├── project.godot
    ├── game.tscn              ← single scene
    ├── Smallville.csproj
    └── scripts/
        ├── Game.cs            ← scene controller
        ├── LLMClient.cs       ← HTTP + SSE
        ├── SidecarHost.cs     ← spawns + kills python
        ├── ActionPanel.cs     ← 4-button action UI
        ├── HostageStatusPanel.cs
        ├── TimelineFeed.cs
        └── (DTOs) Scenario, Captor, Hostage, GameState, TimelineEvent, ChatMessage
```

See `npc_service/README.md` for env var reference, full endpoint table,
and ship-time bundling notes.

## Authoring a new scenario

`npc_service/scenario.json` is the only file to edit. Schema (validated
strictly at sidecar startup):

- `title`, `setup`, `duration_seconds`
- `captor`: `name`, `persona`, `deflection_style`, `initial_demands[]`,
  `known_facts[]` (the *only* facts the captor may reference),
  `seed_exchanges[]` (used as in-prompt example dialogue, *not* injected
  into history — see `npc_service/README.md` for why),
  `starting_meters`
- `hostages`: exactly 2, each with `name` + `description`
- `timeline`: 3-4 events, each with `at_seconds`, `kind`,
  `notification_text` (shown in UI feed), `injection_text` (injected as
  scene update into captor history)

The placeholder scenario is a 2:14 AM convenience-store holdup with
Marlon as the captor and a deliberately-tied medical event for Janet
that gives the "medical" offer-detail real weight.

## Troubleshooting

- **Captor outputs prose, not JSON** — Gemma not following the
  structured-output protocol. See `npc_service/README.md` →
  Troubleshooting; consider `PromptedOutput` mode or smaller-context
  retries.
- **Streaming chunky in Godot** — verify `HttpCompletionOption.ResponseHeadersRead`
  is on the SSE call (it is — `LLMClient.cs:170`).
- **`No module named npc_service`** — set `PYTHONPATH=..` when running
  uv from the `npc_service/` directory, OR run uv from the parent dir:
  `uv --directory npc_service run python -m npc_service`. The Godot
  side does this automatically.
- **Port 8000 already in use** — set `NPC_SERVICE_PORT=8042` (or any
  free port) on the sidecar AND `NPC_SERVICE_URL=http://127.0.0.1:8042`
  on the Godot side.
