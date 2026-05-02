# Smallville — spy-handler MVP

Single-scenario spy-handler game (*Operation Iron Threshold*). The
player runs three field assets — MAGPIE in Riga, CICADA in Vienna,
DOVE in Lyon — over a shared 18-decision operation. One LLM persona
per asset; channel switching is free; only sending an order spends a
decision. Scripted timeline events fire on the global op clock and
inject scene-updates into the addressed channels.

## Architecture

```
┌──────────┐  spawn   ┌──────────────┐  spawn   ┌──────────────┐
│ Godot 4  │ ───────► │ npc_service  │ ───────► │ llama-server │
│ (C#)     │  HTTP/   │ (FastAPI +   │  /v1/    │ (llama.cpp)  │
│          │ ◄──SSE── │  Pydantic AI)│ ◄────────│              │
└──────────┘          └──────────────┘          └──────────────┘
```

- **Godot** spawns one child (Python). On exit, the entire process
  tree (Python + llama-server) is killed via Windows Job Objects.
- **npc_service** owns llama-server lifecycle and runs one Pydantic AI
  `Agent` per asset persona (plus one shared dialogue-options agent).
  Each agent has its own message HISTORY; per-channel SSE streams the
  active agent's reply.
- **Timeline events** fire when the global `action_count` crosses
  authored thresholds. Each event can poke any subset of channels,
  appending a scene-update to those agents' HISTORIES so the next
  reply on those channels reflects the world having moved.

## Prerequisites

- Python 3.11+ (`uv` for dependency management — install from
  https://docs.astral.sh/uv/).
- .NET 8 SDK (Godot 4.6 mono ships with it; check `dotnet --version`).
- Godot 4.6 mono (works with 4.4+; .NET edition required).
- `llama-server` from llama.cpp ([releases](https://github.com/ggerganov/llama.cpp/releases))
  or built locally with CUDA support.
- ~8 GB free disk for the Q8_0 GGUF (auto-downloaded on first run).

## Run end-to-end (dev)

`make run` boots all three layers; `make llama` / `make sidecar` /
`make godot` run them separately for iteration.

### Terminal 1 — llama-server

```bash
llama-server \
  -m models/gemma-4-E4B-it-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 32768 --jinja --reasoning off -ngl 99
```

Two flags are non-negotiable for Gemma 4:

- `--jinja` — Gemma 4's chat template must be applied via Jinja.
- `--reasoning off` — Gemma 4 E-models default to thinking-on; without
  this flag, the model burns its entire generation budget inside a
  hidden `<think>` block and returns empty content.

If you don't have the GGUF, the bundled launcher (`make llama` / the
auto-spawn path) downloads it from Hugging Face on first run. To grab
it manually:

```bash
mkdir -p models
curl -L -o models/gemma-4-E4B-it-Q8_0.gguf \
  https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf
```

Smoke test:

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"Say only the word PONG"}],"max_tokens":4}'
```

### Terminal 2 — Python sidecar

```bash
cd npc_service
uv sync                                   # one-time
LAUNCH_LLAMA=false uv run python -m npc_service
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/scenario | jq .
curl -N -X POST http://127.0.0.1:8000/action \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"magpie","line":"Status?"}'
```

Expect `data: {"type":"text","agent_id":"magpie","text":"..."}` lines
arriving smoothly, then `{"type":"state",...}`, optional
`{"type":"timeline",...}` events, then `{"type":"done"}`. If text
arrives in big bursts, something is buffering.

### Terminal 3 — Godot

Open `godot_project/` in Godot 4.6 mono and press F5, OR from CLI:

```bash
godot --path godot_project --no-sidecar
```

`--no-sidecar` tells Godot to skip auto-spawning Python (since you're
running it yourself).

### Single-bootstrap mode (the ship path)

```bash
godot --path godot_project
```

Godot spawns Python; Python spawns llama-server. Watch the Godot
console for `[npc] ...` log lines. First run takes 90 s+ for cold
model load; subsequent runs are fast.

## Tests

```bash
cd npc_service && uv run pytest
```

32 tests: scenario loading, end-condition logic, per-agent history
isolation, timeline injection routing, FastAPI smoke + per-channel
SSE. Uses Pydantic-AI `TestModel`, so no llama-server needed.

## Verification checklist

After llama-server, sidecar, and Godot are all up:

- [ ] **Boot** — three dossier tabs (MAGPIE, CICADA, DOVE) render in
      the left column. Each tab's chat opens with that agent's
      `opening_transmission` rendered as their first line. The action
      panel shows 2–5 decision cards for the active channel.
- [ ] **Streaming** — pick a card. First agent character appears
      within ~2 s; subsequent characters arrive smoothly. Bursty =
      SSE buffering somewhere.
- [ ] **Channel switch** — click another dossier tab. Cached
      transcript renders instantly; fresh `/options` call fetches
      that channel's picks. No action budget cost.
- [ ] **Timeline event** — play through to action 3 (`truck_in_motion`
      fires). The TimelineFeed shows FLASH TRAFFIC; MAGPIE and CICADA
      receive different scene-updates; their `●NEW` badges appear if
      they're not the active channel.
- [ ] **Action cancellation** — click a card, then immediately switch
      tabs while the response is streaming. The interrupted line gets
      `[interrupted]`; the global `action_count` does NOT advance
      (partial turns are discarded by design).
- [ ] **Budget exhaustion** — play to 18 actions. Resolution overlay
      shows `WINDOW CLOSED`, decisions used, and per-asset last-contact.
- [ ] **Reset** — click `REOPEN FILE` on the resolution screen. Fresh
      HISTORIES; tabs reset; budget back to 18.

## Project layout

```
smallville/
├── README.md                 ← this file
├── CLAUDE.md                 ← architecture invariants for AI assistants
├── npc_service/              ← Python sidecar (uv project)
│   ├── src/npc_service/
│   │   ├── main.py           ← FastAPI app + agents + SSE + timeline firing
│   │   ├── models.py         ← Pydantic scenario schema + DTOs + HandlerState
│   │   ├── launcher.py       ← llama-server subprocess + Job Object
│   │   └── config.py         ← pydantic-settings (env + .env)
│   ├── scenario.json         ← Operation Iron Threshold (3 agents, 4 events, 18 actions)
│   ├── tests/                ← pytest suite
│   ├── pyproject.toml
│   └── README.md             ← endpoints + env vars + troubleshooting
└── godot_project/
    ├── project.godot
    ├── scripts/
    │   ├── Game.cs           ← scene controller, channel routing, end overlay
    │   ├── LLMClient.cs      ← HTTP + SSE client (autoload)
    │   ├── SidecarHost.cs    ← spawns Python (autoload)
    │   ├── DossierTabs.cs    ← left-edge channel tabs
    │   ├── ActionPanel.cs    ← decision cards (hover-reveal hidden_line)
    │   ├── TimelineFeed.cs   ← right-column FLASH TRAFFIC log
    │   ├── BudgetPunchcard.cs← top-bar ●/○ remaining-decisions strip
    │   └── (DTOs) Scenario.cs / AgentDef.cs / Operation.cs / HandlerState.cs /
    │              TimelineEvent.cs / DialogueOption.cs / ChatMessage.cs
    └── assets/fonts/         ← IBM Plex Mono + Stardos Stencil (OFL)
```

See `npc_service/README.md` for env var reference, full endpoint
table, and ship-time bundling notes. See `CLAUDE.md` for the
architecture invariants ("things that look removable but aren't")
that AI assistants editing this code should treat as load-bearing.

## Authoring a new scenario

`npc_service/scenario.json` is the only file to edit. Schema
(validated strictly at sidecar startup — see `models.py`):

- `title`, `setup`, `max_actions` (global decision budget)
- `operation`: `codename`, `objective`, optional `world_state` (author
  crib sheet — never sent to any model)
- `agents`: exactly 3, each with `id`, `callsign`, `name`, `city`,
  `role`, `persona`, `deflection_style`, `known_facts[]`,
  `opening_transmission`, optional `seed_exchanges[]` (used as
  in-prompt examples in the system prompt; *not* injected into
  history — see `npc_service/README.md` for why)
- `timeline`: 3–4 events, each with `id`, `after_actions` (threshold
  on the global counter), `kind` (`world_event` /
  `flash_traffic` / `directive`), `notification_text` (shown in the
  TimelineFeed), and `injections: {agent_id: text}` (per-channel
  scene-update strings appended to the addressed agents' HISTORIES).

## Troubleshooting

- **Agent outputs prose, not structured output** — usually llama-server
  was started without `--jinja`. The bundled launcher sets it; if you
  run llama-server yourself, include it.
- **Empty agent responses / 18K-token "responses"** — `--reasoning off`
  is missing. Gemma 4 E-models default to thinking-on and burn their
  budget inside a hidden `<think>` block.
- **Streaming chunky in Godot** — verify `HttpCompletionOption.ResponseHeadersRead`
  on the C# `HttpClient.SendAsync` (it is — `LLMClient.cs`).
- **`No module named npc_service`** — run `uv sync` from `npc_service/`
  to install the package editable. The src layout means
  `python -m npc_service` only resolves via the venv, not via CWD.
- **Port 8000 already in use** — set `NPC_SERVICE_PORT=8042` (or any
  free port) on the sidecar AND `NPC_SERVICE_URL=http://127.0.0.1:8042`
  on the Godot side.
