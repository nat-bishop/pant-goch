# npc_service

FastAPI sidecar that drives the field-asset agents for the spy-handler
game. One Pydantic AI `Agent` per asset persona, plus one shared
dialogue-options agent. Manages the `llama-server` (llama.cpp)
lifecycle. Streams agent text deltas + state + timeline events over
SSE.

## Quick start (dev)

```bash
cd npc_service
uv sync

# Option A — let the sidecar manage llama-server
uv run python -m npc_service

# Option B — run llama-server yourself in another terminal
LAUNCH_LLAMA=false uv run python -m npc_service
```

Option B is easier for iteration: leave llama-server running, restart
only the sidecar. Reference command for llama-server (matches what
the sidecar spawns):

```bash
llama-server \
  -m ../models/gemma-4-E4B-it-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 32768 --jinja --reasoning off -ngl 99
```

`--jinja` is required for Gemma's chat template; `--reasoning off`
disables the default thinking-mode for E-models (without it, the
model emits all its tokens inside a hidden `<think>` block and
returns empty content). The bundled `launcher.py` sets both flags
automatically.

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
curl -s 'http://127.0.0.1:8000/options?agent_id=magpie' | jq .

# action with live SSE (-N disables curl buffering — important)
curl -N -X POST http://127.0.0.1:8000/action \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"magpie","line":"Status?"}'
```

Expected `/action` output: a sequence of
`data: {"type":"text","agent_id":"magpie","text":"..."}` events
arriving smoothly, then `data: {"type":"state",...}`, zero or more
`data: {"type":"timeline",...}` events if a threshold was crossed,
then `data: {"type":"done"}`.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | liveness probe (used by Godot to gate boot) |
| GET | `/scenario` | scenario data minus `known_facts`, `seed_exchanges`, `deflection_style`, `world_state` |
| GET | `/state` | live public game state (per-agent last-contact, action budget, ended flag) |
| POST | `/action` | SSE — text deltas (per-channel), state event, timeline events, done. Body: `{agent_id, line}` |
| GET | `/options?agent_id=X` | 2–5 LLM-generated dialogue picks for the named channel; read-only over `HISTORIES[X]` |
| GET | `/events` | SSE — game-end notifications + reset signals (timeline events flow inline through `/action`) |

## Env vars

Loaded via `pydantic-settings` (see `src/npc_service/config.py`) from
the environment and an optional `.env` at the repo root.

| Var | Default | Purpose |
|---|---|---|
| `LAUNCH_LLAMA` | `true` | Set `false` to skip auto-launch and run llama-server yourself |
| `LLAMA_PORT` | `8080` | llama-server port |
| `LLAMA_CTX` | `32768` | Context window |
| `LLAMA_NGL` | `99` | GPU layers (-ngl); drop on GPU OOM |
| `LLAMA_EXTRA_ARGS` | (unset) | Whitespace-split extra args appended verbatim |
| `LLAMA_SERVER_BIN` | `<repo>/vendor/llama.cpp/llama-server[.exe]` | llama-server binary path |
| `LLAMA_MODEL_PATH` | `<repo>/models/gemma-4-E4B-it-Q8_0.gguf` | Local GGUF; falls back to HF download |
| `LLAMA_HF_REPO` | `unsloth/gemma-4-E4B-it-GGUF:Q8_0` | HF source when no local model |
| `LLAMA_BASE_URL` | `http://127.0.0.1:${LLAMA_PORT}/v1` | Override only when pointing at a remote llama |
| `NPC_SERVICE_PORT` | `8000` | sidecar port |
| `NPC_LOG_LEVEL` | `info` | Python log level |
| `SCENARIO_PATH` | `<sidecarDir>/scenario.json` | Scenario data file |
| `SUMMARY_TRIGGER_FRAC` | `0.8` | Compaction kicks in when a HISTORY would exceed this fraction of `LLAMA_CTX` |
| `SUMMARY_TRIGGER_TOKENS` | `0` | Absolute compaction trigger override (0 = compute from frac) |
| `SUMMARY_KEEP_TAIL` | `4` | Trailing messages kept verbatim past the recap |
| `ACTION_STREAM_TIMEOUT_SECONDS` | `120.0` | Whole-stream timeout for `/action` |
| `OPTIONS_TIMEOUT_SECONDS` | `30.0` | Whole-call timeout for `/options` |
| `LOGFIRE_TOKEN` | (unset) | Logfire write token; the sidecar also reads `npc_service/.logfire/logfire_credentials.json` |

## Architecture notes

- **Where llama-server lifecycle lives**: Python lifespan owns it.
  The Godot side spawns *one* child (Python); Python spawns
  llama-server as its own child. Single bootstrap point at the game
  exe; single shutdown signal cascades. All LLM lifecycle code lives
  next to the code that uses it.
- **One `Agent` per asset persona**: each agent gets its own
  `summarize_when_large` history processor over its own
  `HISTORIES[id]`, so per-channel compaction is independent. A single
  shared `OPTIONS_AGENT` is parameterised via the per-call user
  prompt — keeps the system-prompt prefix cache-warm across channel
  switches.
- **`ToolOutput(AgentReport)` + `max_tokens=512`**: each agent emits
  a structured `AgentReport` (currently a single `report_text`
  field). Tool-output mode plus the per-turn token cap keeps replies
  bounded and parses cleanly first try; an earlier `NativeOutput`
  variant with no per-field cap blew turns up to 19K tokens.
  Validation against the schema is the wire contract — keeps the
  option open to grow more typed fields later without re-plumbing.
- **`OPTIONS_AGENT` uses `PromptedOutput`, not `ToolOutput`**:
  llama.cpp's `peg-gemma4` tool-call grammar generator emits a
  malformed `thought ::=` rule on the `OptionsResponse` schema and
  crashes GBNF install. `PromptedOutput` sends no `tools=[...]` so
  `peg-gemma4` is never activated; the schema lives in the system
  prompt and Pydantic AI parses the JSON client-side.
  `output_retries=2` covers the rare bad-JSON turn. The per-asset
  `AGENTS` keep `ToolOutput` — the flatter `AgentReport` schema
  doesn't trip the bug.
- **Why `seed_exchanges` are NOT seeded into history**: the agent is
  configured with `output_type=ToolOutput(AgentReport)`, enforcing a
  tool-call schema. Pre-populating history with a plain-text
  `ModelResponse` would teach the model to emit text instead of
  tool-calling. We embed `seed_exchanges` as in-prompt examples in
  the system prompt instead.
- **`opening_transmission` is rendered, not generated**: the string
  lives in `scenario.json`. The server seeds it into `HISTORIES[id]`
  as a scene-update so the agent treats it as their own prior
  context; the *client* GETs `/scenario` and renders the string into
  the chat at boot. The agent never speaks the text aloud (no LLM
  call) — it's the player's first message in that channel.
- **Why we discard partial agent turns on cancel**: if the player
  switches channels or hits ABORT mid-stream, the in-flight response
  is incomplete and state mutations would be unsafe. Persistence of
  `HISTORIES[id]` and the `action_count` increment only happen after
  the stream-output loop completes. Cancellation jumps straight to
  the `finally` block via `CancelledError`, bypassing both.
- **`/options` is read-only over the channel's history**: the call's
  `result.all_messages()` is *discarded*. Writing it back would leak
  the "Generate options for asset X." user prompt into the channel's
  HISTORY and confuse the agent on the next turn. `ACTION_LOCK` is
  global across channels so `/options` never reads `HISTORIES[id]`
  mid-mutation.
- **Timeline events fire per-channel**: each `TimelineEvent` has an
  `injections: dict[agent_id, str]` map. When `action_count` crosses
  the event's `after_actions` threshold, scene-update messages are
  appended to *only* the addressed channels' HISTORIES — not all of
  them. Validators at scenario load enforce that every key resolves
  to a known agent_id.

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

- **Sidecar starts but `/health` reports `agents_ready: []`**: lifespan
  failed midway. Check sidecar logs for the traceback.
- **`llama-server failed to come up`**: check `npc_service/logs/llama-*.log`
  for the underlying llama-server stderr. Common causes: port in use,
  GPU OOM (drop `LLAMA_NGL`), missing `--jinja` (we set it; if you've
  overridden `LLAMA_EXTRA_ARGS`, make sure it's still there).
- **Agent turns are slow / runaway generation / `/action` stalls past
  120s**: an unbounded `report_text` field plus `NativeOutput` mode
  used to fill replies indefinitely and seed each turn into more
  verbosity. The shipped sidecar uses `ToolOutput(AgentReport)` plus
  `model_settings={"max_tokens": 512}`, which keeps turns bounded. If
  you switch back to `NativeOutput`, also add `Field(max_length=...)`
  on `report_text`.
- **Empty agent responses / 18K-token "responses"**: `--reasoning off`
  is missing from llama-server. Gemma 4 E-models default to
  thinking-on, spending all their tokens inside a hidden `<think>`
  block. The bundled launcher passes this flag automatically; if you
  run llama-server yourself, include it.
- **`/options` returns `{"options": []}` repeatedly**: usually a
  validation error in the LLM's response (PromptedOutput JSON failed
  the `OptionsResponse` schema, e.g. fewer than 2 picks or duplicate
  tags). Logged at WARNING. The Godot client falls into its
  "No options available." path.
- **Streaming feels chunky**: confirm `X-Accel-Buffering: no` is on
  the response and curl was run with `-N`. On the C# side, confirm
  `HttpCompletionOption.ResponseHeadersRead`.
