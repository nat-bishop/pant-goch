# npc_service

FastAPI sidecar that drives the captor NPC for the hostage-negotiation game.

- Pydantic AI agent with `output_type=CaptorResponse` (structured output).
- Manages llama-server (llama.cpp) lifecycle.
- Streams text deltas + state updates over SSE.
- Pure-Python timeline scheduler injects scripted events.

## Quick start (dev)

```bash
cd npc_service
uv sync

# Option A — let the sidecar manage llama-server
uv run python -m npc_service

# Option B — run llama-server yourself in another terminal
LAUNCH_LLAMA=false uv run python -m npc_service
```

Option B is easier for iteration: leave llama-server running, restart only
the sidecar. Reference command for llama-server (matches what the sidecar
spawns):

```bash
llama-server \
  -m ../models/gemma-4-E4B-it-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 8192 --jinja --reasoning off -ngl 99
```

`--jinja` is required for Gemma's chat template; `--reasoning off`
disables the default thinking-mode for E-models (without it, the model
emits all its tokens inside a hidden `<think>` block and returns empty
content). The bundled `launcher.py` sets both flags automatically.

## Verify

```bash
# llama-server
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"Say only the word PONG"}],"max_tokens":4}'

# sidecar
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/scenario | jq .
curl -s http://127.0.0.1:8000/state | jq .

# action with live SSE (-N disables curl buffering — important)
curl -N -X POST http://127.0.0.1:8000/action \
  -H "Content-Type: application/json" \
  -d '{"type":"empathize","detail":""}'
```

Expected `/action` output: a sequence of `data: {"type":"text","text":"..."}`
events arriving smoothly, then `data: {"type":"state",...}`, then
`data: {"type":"done"}`.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness probe (used by Godot to gate boot) |
| GET | `/scenario` | scenario data minus `known_facts`, `seed_exchanges`, meters |
| GET | `/state` | live public game state (hostage statuses, time, ended flags) |
| POST | `/action` | SSE — text deltas, then state event, then done |
| POST | `/regenerate` | SSE — debug only, gated by `DEBUG_AFFORDANCES=true` |
| POST | `/reset` | wipes game state, restarts timeline scheduler |
| GET | `/events` | SSE — timeline events + game-end notifications |

## Env vars

| Var | Default | Purpose |
|---|---|---|
| `LLAMA_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible base URL |
| `LLAMA_MODEL` | `gemma-4-e4b` | Cosmetic model name (llama-server ignores) |
| `LLAMA_SERVER_BIN` | `<exeDir>/llama-server[.exe]` | llama-server binary path |
| `LLAMA_MODEL_PATH` | `<exeDir>/models/gemma-4-E4B-it-Q8_0.gguf` | GGUF file path |
| `LLAMA_HF_REPO` | `unsloth/gemma-4-E4B-it-GGUF:Q8_0` | Fallback HF source if no local model |
| `LLAMA_HOST` | `127.0.0.1` | llama-server bind host |
| `LLAMA_PORT` | `8080` | llama-server port |
| `LLAMA_CTX` | `8192` | Context window |
| `LLAMA_NGL` | `99` | GPU layers (-ngl) |
| `LLAMA_EXTRA_ARGS` | (unset) | Whitespace-split extra args appended verbatim |
| `LAUNCH_LLAMA` | `true` | Set `false` to skip auto-launch |
| `LLAMA_HEALTH_TIMEOUT` | `600` | Seconds to wait for `/health` (cold load can be 90+s) |
| `NPC_SERVICE_HOST` | `127.0.0.1` | sidecar bind host |
| `NPC_SERVICE_PORT` | `8000` | sidecar port |
| `SCENARIO_PATH` | `<sidecarDir>/scenario.json` | Scenario data file |
| `DEBUG_AFFORDANCES` | `false` | Enables `/regenerate` |
| `NPC_LOG_LEVEL` | `info` | Python log level |

## Architecture notes

- **Where llama-server lifecycle lives**: Python lifespan owns it. The
  Godot side spawns *one* child (Python); Python spawns llama-server as
  its own child. Single bootstrap point at the game exe; single shutdown
  signal cascades. All LLM lifecycle code lives next to the code that
  uses it.
- **Why structured output is load-bearing**: the captor is constrained to
  emit a `CaptorResponse` (text + meter deltas + optional action). Game
  state mutates atomically from the validated structure — never from the
  text. The LLM does character work; state changes do game work. Even if
  `response_text` is gibberish, the game still progresses coherently.
- **Why `seed_exchanges` are NOT seeded into history**: with
  `output_type=CaptorResponse`, the agent is configured to enforce a JSON
  schema. Pre-populating history with plain-text `ModelResponse` would
  teach the model to emit text instead of JSON. We embed `seed_exchanges`
  as in-prompt examples in the system prompt instead.
- **Why we discard partial captor turns on cancel**: if the player
  switches actions mid-stream, the in-flight response is incomplete and
  state mutations would be unsafe. Persistence of `HISTORY` and the call
  to `apply_response` only happen after the stream-output loop completes.
  Cancellation jumps straight to the `finally` block via `CancelledError`,
  bypassing both.

## Ship-time bundling (out of MVP scope, but path-resolution is ship-ready)

Steam install layout target:

```
<steam>/<game>/
├── smallville.exe
├── llama-server.exe
├── models/gemma-4-E4B-it-Q8_0.gguf
├── python/python.exe       # python-embed-amd64
├── lib/                    # `uv pip install --target lib --no-deps -r ...`
└── npc_service/
```

`launcher.py:_exe_dir()` resolves to `<steam>/<game>/` whether dev or
ship; `LLAMA_SERVER_BIN` and `LLAMA_MODEL_PATH` default off that.

## Troubleshooting

- **Sidecar starts but `/health` reports `agent_ready: false`**: lifespan
  failed midway. Check sidecar logs for the traceback.
- **`llama-server failed to come up`**: check `npc_service/logs/llama-*.log`
  for the underlying llama-server stderr. Common causes: port in use,
  GPU OOM (drop `LLAMA_NGL`), missing `--jinja` (we set it; if you've
  overridden `LLAMA_EXTRA_ARGS`, make sure it's still there).
- **Captor turns are slow / runaway generation / `/action stalled past
  120s`**: with an unbounded `response_text` field on `CaptorResponse`,
  the model can fill it indefinitely and each turn's bloated reply seeds
  the next turn into even more verbosity. The shipped sidecar uses
  `ToolOutput(CaptorResponse)` plus `model_settings={"max_tokens": 512}`,
  which keeps turns bounded. Earlier builds shipped `NativeOutput` —
  reliable JSON via GBNF grammar at the sampler, but no per-field length
  cap, so it hit the runaway pattern on long sessions. If you switch back
  to `NativeOutput`, also add `Field(max_length=...)` on `response_text`.
- **Empty captor responses / 18K-token "responses"**: `--reasoning off`
  is missing from llama-server. Gemma 4 E-models default to thinking-on,
  spending all their tokens inside a hidden `<think>` block. The
  bundled launcher passes this flag automatically; if you run llama-server
  yourself, include it.
- **Streaming feels chunky**: confirm `X-Accel-Buffering: no` is on the
  response and curl was run with `-N`. On the C# side, confirm
  `HttpCompletionOption.ResponseHeadersRead`.
