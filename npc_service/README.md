# npc_service

FastAPI sidecar for Inbox. Owns the deterministic spine (`spine.py`), runs six Pydantic AI agents (`agents.py`) on a single managed `llama-server` (Gemma 4 E4B Q8_0), and streams the reply path to the React frontend over SSE. See the top-level `README.md` for the game design + project layout, and `CLAUDE.md` for engine invariants.

## Quick start (dev)

```bash
cd npc_service
uv sync

# Option A — sidecar manages llama-server itself (default)
uv run python -m npc_service

# Option B — run llama-server in another terminal, sidecar attaches
LAUNCH_LLAMA=false uv run python -m npc_service     # in this terminal
uv run python -m npc_service.run_llama              # in a sibling terminal
```

Option B is the iterate-fast path: leave llama running, restart only the sidecar. `run_llama.py` shares `launcher.build_command` with the lifespan path so the argv can't drift between Makefile-launched and sidecar-launched llama.

The flag `--jinja` on llama-server is non-negotiable on Gemma 4 E-models — without it the chat template doesn't load. The launcher also passes `--flash-attn on`, `--batch-size 512 --ubatch-size 512`, `--cache-type-k q8_0 --cache-type-v q8_0` (KV-quant requires flash-attn), `-ngl 99`, and `-np 4 --no-kv-unified` for single-user interactive use. `make llama-playtest` overrides `LLAMA_N_PARALLEL=64` for the playtester (Makefile). Gemma 4 thinking is OFF by default; set `LLAMA_REASONING_MODE=on` to restore `--reasoning on --reasoning-format deepseek` on chat + director (classifier/extractor/fact-checker/summary stay no-think via per-agent overrides).

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
curl -s http://127.0.0.1:8000/inbox | jq .

# Send a player reply (SSE; -N disables curl buffering)
curl -N -X POST http://127.0.0.1:8000/send \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"aria.continuity","reply_to_email_id":"<some-id>","body":"Hi."}'
```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET  | `/health` | liveness |
| GET  | `/scenario` | scenario data (sanitised — author-only fields stripped) |
| GET  | `/scenario_meta` | display_name + email_address per NPC; cheap probe for the chat header |
| GET  | `/state` | live `InboxState` — qualities, action counter, fired beats, ended flag |
| GET  | `/inbox` | rendered email list (chronological, with read flags) |
| GET  | `/thread/{thread_id}` | per-thread email + reply log |
| POST | `/open` | mark email read; mutates no qualities |
| POST | `/send` | SSE — player reply triggers a tick (extractor + classifier in parallel, beat selection, chat-agent render). Streams `email_emitted`, `tick_decision`, `error` (on chat failure), `done`. |
| GET  | `/debug/scenario` | (DEV) scenario AST for the debug panel |
| GET  | `/debug/session/events` | (DEV) committed event log for the active session |
| GET  | `/debug/session/state_at` | (DEV) state replayed up to event N |
| POST | `/debug/session/save` | (DEV) snapshot session events to JSONL |
| POST | `/debug/session/load` | (DEV) replay JSONL into a fresh session |

## Configuration

Everything lives on a `Settings(BaseSettings)` class in `src/npc_service/config.py` — pydantic-settings handles typed parsing, env-file loading, and case-insensitive snake_case → uppercase name coercion. `Settings()` is re-instantiated at each entry point (lifespan, uvicorn bootstrap, playtest boot) so test fixtures' `monkeypatch.setenv(...)` calls bind on the next construction.

In practice four things actually get toggled — by tests, by `make sidecar`, or by the opt-in persistence feature:

| Var | Default | Purpose |
|---|---|---|
| `LAUNCH_LLAMA` | `true` | Set `false` to skip auto-launch (used by `make sidecar` and tests with a stubbed health probe) |
| `SCENARIO_PATH` | (unset) | Override scenario YAML path; default falls back to `npc_service/scenario.yaml`. Tests set this per-fixture; the playtester sets it from CLI |
| `NPC_LOGFIRE_CONSOLE` | `true` | `false` silences Logfire console output (conftest sets this for clean pytest captures) |
| `SESSION_SAVE_PATH` | (unset) | When set, append committed events as JSONL and replay on boot. Unset → in-memory only |

The rest — `LLAMA_PORT`, `LLAMA_CTX`, `LLAMA_NGL`, `LLAMA_N_PARALLEL`, `LLAMA_KV_UNIFIED`, `LLAMA_HF_REPO`, `LLAMA_BASE_URL`, `EVAL_LLAMA_PORT`, `EVAL_LLAMA_BASE_URL`, `LLAMA_SERVER_BIN`, `LLAMA_MODEL_PATH`, `LLAMA_EXTRA_ARGS`, `NPC_SERVICE_PORT`, `NPC_LOG_LEVEL`, `HISTORY_COMPACT_FRAC`, `HISTORY_COMPACT_KEEP_TAIL` — are all readable from env via the same `Settings` class but tuned once and otherwise left alone. See `config.py` for the full list, defaults, and per-field rationale.

`LOGFIRE_TOKEN` is read directly via `os.getenv` and is also picked up from `npc_service/.logfire/logfire_credentials.json` if not set.

## Architecture

- **Lifespan owns llama-server.** `launcher.start_llama` spawns it as a child process, enrolls it in a Windows Job Object with `KILL_ON_JOB_CLOSE` so it dies if the sidecar crashes, then waits on `/health` before building agents. `LAUNCH_LLAMA=false` skips the spawn and just probes — used by `make sidecar` and tests. The argv is rendered from `launcher.build_command(LlamaConfig)`; both the sidecar and `make llama` go through it so flags can't drift.
- **One model handle, six agents.** A single `OpenAIChatModel` (Gemma 4 E4B via llama-server's OpenAI-compatible endpoint) is shared across `chat_agent`, `classifier_agent`, `extractor_agent`, `summary_agent`, `fact_checker_agent`, `director_agent`. Profile is built with `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` — without it pydantic-ai re-serialises parsed `ThinkingPart`s as `<think>…</think>` in next-turn assistant content, violating Google's Gemma 4 multi-turn rule and inflating prompt tokens.
- **The chat agent does double duty.** `ChatDeps.epilogue_digest` flips it into closing-email mode via conditional `@instructions` decorators (`_epilogue_framing_block`, `_epilogue_digest_block`) — terminal beats with `ends_session: true` route through `_render_via_epilogue_agent` instead of a second agent definition. Per-run instructions inject the dossier, optional plot directive, current commitments, and a top-K player-canon snapshot.
- **`/send` runs the extractor and classifier in parallel.** `asyncio.gather` makes them latency-neutral. Extractor harvests open-ended specifics into `state.player_canon` (substring-anchor + salience-threshold); classifier judges the player's reply against the replied-to beat's `classify_reply_axes`. Director runs only when `find_due_beats` returns ≥2 eligible — short-circuited otherwise.
- **State writes are owned by `spine.py`.** Three entry points: `apply_literal_effects` (beat-fire-time `value:` writes), `apply_classify_effect` (classifier verdict → quality, at next-reply time), `apply_canon_extraction` (extractor candidates → `state.player_canon`, REPLACE on collision, salience-thresholded). The chat agent never writes state — its only side effect is the fact-checker `ModelRetry` loop, which can refuse an output but not mutate.
- **History compaction lives on the chat agent's history processor.** Token budget = `LLAMA_CTX × HISTORY_COMPACT_FRAC` (32K × 0.8 = 26,214 tokens at boot). When a per-thread `message_history` would exceed it, the oldest messages are summarised by `summary_agent` and replaced with a single recap; `HISTORY_COMPACT_KEEP_TAIL` recent messages stay verbatim past the recap. The token counter calls llama-server's `/tokenize` endpoint (root path, NOT under `/v1/`); tests stub it to zero so the unit suite doesn't need a live llama.
- **Per-tick audit trail.** Every scheduler tick produces a `TickDecision` (eligibility report, selection method `none`/`sole`/`continue`/`director`, Director rationale when applicable, all quality deltas, what fired). It surfaces three ways: as Logfire span attributes, as a console log line, and as a `tick_decision` SSE event consumed by the DEV debug panel.
- **Sessions are per-game; the engine is process-singleton.** `SCENARIO`, `AGENTS`, `LLAMA_PROC`, `SETTINGS` are loaded once in lifespan and never mutate. Per-game state lives in `SESSIONS[session_id]` — each carries its own `InboxState`, event log, and `asyncio.Lock`. Endpoints resolve a session via `Depends(get_session)`. The single-player default is `SESSIONS["default"]`, lazily created on first access.

## Ship-time bundling

Steam install layout target:

```
<steam>/<game>/
├── smallville.exe
├── vendor/llama.cpp/llama-server.exe
├── models/ggml-org-gemma-4-E4B-it-Q8_0.gguf
├── python/python.exe       # python-embed-amd64
├── lib/                    # `uv pip install --target lib --no-deps -r ...`
└── npc_service/
```

`launcher._exe_dir()` resolves to `<steam>/<game>/` whether dev or ship; `resolve_config()` derives the binary and model paths from that base, so no env override is needed.

## Troubleshooting

- **Sidecar boots but `/health` is unhappy** — lifespan failed midway. Check sidecar logs for the traceback.
- **`llama-server failed to come up`** — check `npc_service/logs/llama-*.log` for the underlying stderr. Common causes: port `:8080` in use (`make kill-zombies` from the repo root), GPU OOM (drop `LLAMA_NGL` in config.py), corrupt GGUF (delete and re-download with `make download-model`).
- **Empty agent responses or 19K-token monologues** — `--jinja` missing on llama-server (chat template not loaded). The bundled launcher always sets it. If 19K-token reasoning monologues appear in bodies after flipping `LLAMA_REASONING_MODE=on`, also verify `--reasoning-format deepseek` is on the argv — without it Gemma 4 inlines `<think>` blocks into `content`.
- **Prior-turn `<think>` blocks leaking into request bodies** — `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` is missing on the `OpenAIChatModel` constructor. Verify in Logfire by inspecting any chat-agent POST span's `http.request.body.text`.
- **`request (N tokens) exceeds the available context size (M)`** — at single-user startup the launcher passes `-np 4 --no-kv-unified` with `-c` auto-sized to `n_parallel × llama_ctx = 4 × 32k = 128k`, so each slot gets the full 32k. Under playtest, `LLAMA_N_PARALLEL=64` is set in the Makefile and `-c` auto-scales to `64 × 32k = 2M`. If you see this error you're running llama-server with stale flags — restart via `python -m npc_service.run_llama`.
- **`No module named npc_service`** — run `uv sync` from `npc_service/`.
- **Streaming arrives in big bursts** — verify your reverse proxy isn't buffering. The server sets `X-Accel-Buffering: no` on the `/send` SSE response.
