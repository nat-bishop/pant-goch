# HANDLER

Single-player spy game. One verb: **authorize one of 3–4 options per cycle**.

The player sits at a desk in a windowless room. Operations Desk drafts options each cycle; the player picks one; an agent in the field carries it out and files a dispatch. The campaign closes when a mission resolves, succeeds, or fails. The MVP mission is **Op. LYNX-EXFIL**: get the deep-cover agent LYNX out before the surveillance window closes, with a second agent (KESTREL) who can be sacrificed to clear her lanes.

The game is LLM-native — the Operations Desk's drafting and the agents' dispatches are situated prose conditioned on world state. Mechanics, however, are fully deterministic and live in YAML. The LLM never decides outcomes.

Full design: `HANDLER_SPEC_V2.md`. Visual reference: `_mockups/handler-authorize-desk.html`.

## Architecture

```
┌──────────┐  HTTP/SSE  ┌─────────────┐  spawn  ┌──────────────┐
│ Browser  │ ─────────► │   handler   │ ──────► │ llama-server │
│ (Vite +  │            │  (FastAPI + │  /v1/   │ (llama.cpp)  │
│  React)  │ ◄───────── │  Pydantic AI)│ ◄────── │              │
└──────────┘            └─────────────┘         └──────────────┘
```

- **handler** owns the deterministic engine (state, archetypes, moments, beats) and three stateless one-shot Pydantic AI agents per cycle:
  - `SOP_AGENT` — `output_type=str`, streamed. Synthesises the State-of-Play memo at cycle open, in the Operations Desk voice (terse, declarative, third-party handler).
  - `DRAFTER_AGENT` — `NativeOutput(OptionSet)`. Produces 3–4 candidate courses of action from the engine-computed valid catalog. Validated server-side; retries 3× with corrective feedback then falls back to an engine-authored option set so the player never sees a broken cycle.
  - `RENDERER_AGENT` — `output_type=str`, streamed. Renders the field dispatch in the acting agent's voice (or Operations for HOLD).

  All three are **stateless** — no `message_history` is passed between cycles. Continuity is preserved by explicitly including recent dispatches in user-side context (with same-agent preference for the renderer, per spec §8.3).

- **Word-pacing is server-side.** The SSE event stream emits word-by-word at ≥80 ms/word. Frontend renders each word with an 80 ms CSS fade-in; clicking the streaming text container flushes the rest instantly.

- **Browser** is a Vite + React client styled as a paper document on a dark institutional desk (Courier Prime monospace, classified-red stamps, paper textures). Server is the source of truth; the Zustand store is a render cache.

- **One verb**: `POST /mission/{id}/authorize { option_index: N }`. Per cycle the player picks one of 3–4 options. No free-text input.

## Prerequisites

- Python 3.11 or 3.12 + [uv](https://docs.astral.sh/uv/).
- Node 18+ for the web client.
- Windows for the bundled `llama-server.exe`. Linux/macOS works with a hand-built `llama-server` (point `LLAMA_SERVER_BIN` at it).
- ~10 GB free for the Gemma 4 E4B Q8 model (auto-downloaded via Hugging Face on first run).

## Run

Three terminals (or `make run` if `concurrently` is happy on your shell):

```
# Terminal 1: llama-server (spawned by the sidecar by default, OR boot manually)
make llama

# Terminal 2: the FastAPI sidecar
make sidecar       # LAUNCH_LLAMA=false — uses the externally-managed llama from terminal 1
# OR: make sidecar with LAUNCH_LLAMA unset to spawn its own llama subprocess.

# Terminal 3: the Vite dev server
make web
```

Open <http://localhost:5173>. The mission boots immediately; cycle 1's SoP memo streams in word-by-word; options appear; pick one; authorize; the dispatch streams in. Repeat to terminal.

## Test

```
make test                           # full unit suite (~3s)
cd handler && uv run pytest         # same
```

The suite covers the engine primitives (predicates, effects, archetypes, moments, beats, for_each_agent expansion), event union round-trip + fold equivalence with the live mutation path, mission YAML loading, and FastAPI endpoint shape. Live-llama integration tests go behind the `integration` marker; none are authored yet — add `test_cycle_integration.py` when you need one (use the `live_llama` fixture in `conftest.py`).

## Authoring a mission

Missions live in `handler/missions/*.yaml`. The schema is `Mission` in `handler/src/handler/models.py`. The MVP example is `lynx_exfil.yaml` (7 archetypes, 1 moment, 5 beats, 2 dossiers).

Universal qualities are auto-injected from the schema (`mission.cycles_remaining` from `max_cycles`, `agent[<id>].cover` from each `AgentDossier.starting_cover`). Authored qualities go under `initial_qualities`.

Predicates and effects use quality refs as opaque dotted strings — `mission.extraction_stage`, `agent[LYNX].cover`. `$agent` is a placeholder substituted at evaluation time for `for_each_agent` beats and `params:[agent]` archetypes.

Terminal beats need a `terminal: true` + `outcome_key:` that exists in the mission's `outcomes:` map; the validator catches mismatches at load time.

Point the sidecar at a non-default mission with `MISSION_PATH=/path/to/mission.yaml make sidecar`.

## Settings

`handler_port`, `handler_log_level`, `handler_logfire_console`, `launch_llama`, `mission_path`, `llama_*` — see `handler/src/handler/config.py`. Settings are loaded via pydantic-settings; env vars override defaults (case-insensitive).

`LOGFIRE_TOKEN` is read directly via `os.getenv` and is also picked up from `handler/.logfire/logfire_credentials.json` if present. Logfire console output is on by default in dev; set `HANDLER_LOGFIRE_CONSOLE=false` to silence.

## Project layout

```
handler/
├── src/handler/        # engine + agents + API
├── missions/           # authored mission YAMLs (lynx_exfil.yaml is MVP)
├── scripts/
│   └── playtest/       # option-picker harness + personas + evaluators
└── tests/              # unit suite (49 tests as of Phase 6)

web/                    # Vite + React + Tailwind. Single dark theme.
├── src/
│   ├── App.tsx
│   ├── components/     # Topbar, StateStrip, Paper, OptionsList, Inbox, ...
│   ├── hooks/useStream.ts
│   ├── store.ts
│   ├── api.ts
│   └── index.css

_mockups/               # design references (handler-authorize-desk.html is current)
HANDLER_SPEC_V2.md      # authoritative design
```

## Troubleshooting

- **`llama-server failed to come up`** — check `handler/logs/llama-*.log` for the underlying stderr. Common causes: port `:8080` in use (`make kill-zombies`), GPU OOM (drop `LLAMA_NGL`), corrupt GGUF (delete `models/` and re-run `make download-model`).
- **`engine sidecar unreachable`** (from playtest) — sidecar isn't running on `:8000`. Boot via `make sidecar`.
- **Browser shows "booting…" forever** — sidecar is up but the SSE stream is silent. Usually means llama-server isn't responding to /v1/chat/completions. Check the sidecar's stdout for the first agent call.

## License + attribution

Mockup design references in `_mockups/` are evolving concept art. Courier Prime via Google Fonts. Gemma 4 E4B via Hugging Face.
