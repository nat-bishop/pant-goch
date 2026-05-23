# HANDLER v2 — Pant Goch

LLM-driven investigation game set in the Welsh Marches village of Pant Goch. You're a freelance journalist who arrives to look into a death the local police closed too quickly. The world — ten locations, nine villagers, the truth, the major beats — is hand-authored. The scenes inside it are LLM-improvised. The map is the primary navigation surface: walk the village by clicking locations.

The architectural bet, in one sentence: **the LLM writes prose, the engine writes state**. State changes flow through a discriminated-union type with grammar-sampled enums; the model literally cannot invent a beat or location it doesn't have permission for.

Design doc: `SPEC.md`.

## Architecture

```
┌──────────┐  HTTP/SSE  ┌───────────────┐  spawn  ┌──────────────┐
│ Browser  │ ─────────► │    handler    │ ──────► │ llama-server │
│ (Vite +  │            │  (FastAPI +   │  /v1/   │ (llama.cpp)  │
│  React)  │ ◄───────── │  Pydantic AI) │ ◄────── │              │
└──────────┘            └───────────────┘         └──────────────┘
```

Two pydantic-ai agents per session:

- **Scene runner.** Two-stage emission per turn against one stable system prompt. Phase 1 emits a `StateChangeBundle` (`unlock_beat`, `move_npc`, ...) validated against `requires_beats` / `blocked_by_beats`; `ModelRetry` on failure, budget 2. Phase 2 emits a `ProsePart` (free-form prose + ≤4 options including ≥1 "Leave for…"). Both calls share KV cache because the system prompt is byte-identical.
- **Log writer.** Conversation-based; one user+assistant pair appended per scene-end. System prompt carries the world bible + journalist's voice rules. The accumulating history is the case file.

Word-pacing is server-side at ≥150 ms cadence (`POST /skip` flushes the remainder at full speed). The frontend's `<StreamingText>` only animates fade-in.

## Prerequisites

- Python 3.11 or 3.12 + [uv](https://docs.astral.sh/uv/).
- Node 18+ for the web client.
- Windows for the bundled `llama-server.exe` at `vendor/llama.cpp/`. Linux/macOS works with a hand-built `llama-server` (point `LLAMA_SERVER_BIN` at it).
- ~10 GB free for the Gemma 4 E4B Q8 model (auto-downloaded via Hugging Face on first run, or fetch ahead of time with `make download-model`).

## Run

Two terminals:

```
# Terminal 1 — FastAPI sidecar (spawns llama-server on :8080)
make sidecar

# Terminal 2 — Vite dev server
make web
```

Open <http://localhost:5173>. Click "Begin investigation". Pick options, walk to adjacent locations, watch your case file fill up in the sidebar.

For a text-only playthrough in one terminal (no web, uses the inferno gateway on the Blackwell box):

```
make cli
```

## Test

```
make test                  # fast unit suite (~3s, no live llama)
make test-integration      # includes the live-llama integration test
```

## Evals

One CLI, three subcommands (run from the repo root; the engine defaults to the
inferno gateway). Every setting is a flag — run any command with `--help`:

```
uv run python -m evals --help          # lists iso / play / judge
uv run python -m evals play --help     # every flag for that subcommand
```

| Command | What it does |
|---|---|
| `uv run python -m evals iso` | Fast deterministic per-agent gate (frozen turn contexts, K samples, CI floor). `--suite deltas,attention,…` to scope. |
| `uv run python -m evals play` | Full simulated-player corpus + invariant report + `summary.json`. |
| `uv run python -m evals play --experiment <name>` | **A/B**: runs the two arms in `evals/experiments/<name>.yaml`, diffs the metrics, and (with `--judge`) pairwise-judges blinded transcripts and clusters the rationales into common themes. |
| `uv run python -m evals judge <run-dir>` | Re-judge a recorded corpus (no engine needed). |

`make eval ARGS="…"` is a thin pass-through, e.g. `make eval ARGS="iso --suite deltas"`.

**A/B human rating** (optional). After `play --experiment <name>` writes blinded
`pairs.jsonl`, rate them side-by-side in the browser, then re-run with `--human`
to merge your verdicts (human win-rate + agreement with the LLM judge):

```
uv run python scripts/ab_rater.py --run evals/artifacts/ab/<ts>   # opens :8766
uv run python -m evals play --experiment leave_mode --human       # merges human_ratings.jsonl
```

### Shorter command (optional shell alias)

`uv run python -m evals …` is the portable form. To type just `evals …` instead,
add a shell function (it must be run from inside the repo, like `uv run` itself):

```powershell
# PowerShell — add to $PROFILE
function evals { uv run python -m evals @args }
```

```bash
# bash / zsh — add to ~/.bashrc or ~/.zshrc
evals() { uv run python -m evals "$@"; }
```

Then `evals iso`, `evals play --experiment leave_mode --judge`, etc. Tab-completion
is available via `uv run python -m evals --install-completion`.

## Authoring

World content lives in `world/`:

- `locations/*.yaml` — one file per location. `connects_to` enumerates the adjacency graph.
- `cast.yaml` — all NPCs in one file. Each has `themes`, `knowledge`, `attitude`.
- `truth.yaml` — beats and endings. Beats are gated by `requires_beats` / `blocked_by_beats`. Endings gate on player-driven decision beats authored as mutually-exclusive.
- `brief.yaml` — opening location + editor brief + framing letter.

Schema lives in `src/handler/models.py`. Loader invariants in `src/handler/world.py`.

Point the sidecar at a non-default world with `WORLD_PATH=/path/to/world make sidecar` (once authoring stabilizes — currently world path is fixed to `world/`).

## Inferno gateway (dev-mode)

`make sidecar-inferno` skips the bundled llama-server and routes pydantic-ai at the inferno gateway on `:7770` instead (byte-identical chat template + kernel params; VRAM co-residency with other models on the same box). Requires inferno running. See `CLAUDE.md` → "Inferno gateway" for env vars.

## Project layout

```
SPEC.md                        ← authoritative design doc
pyproject.toml, uv.lock
Makefile

src/handler/
├── main.py, __main__.py
├── launcher.py                ← bundled llama-server launcher
├── models.py                  ← Location, NPC, Beat, Ending, Brief
├── world.py                   ← loader + invariants
├── state.py                   ← State, TurnRecord, LogEntry, GameOver
├── schemas.py                 ← StateChangeBundle, ProsePart, Option
├── engine.py                  ← apply, validate, reachable beats
├── agents.py                  ← scene runner + log writer
├── prompts.py                 ← KV-cache-friendly prompt builders
├── scene_loop.py              ← play_scene, play_session, event emission
└── sse.py                     ← paced word emitter + event builders

tests/                         ← unit suite + live integration
world/                         ← Pant Goch authored content
│   ├── brief.yaml, cast.yaml, truth.yaml
│   └── locations/             ← 10 location YAMLs
web/                           ← React + Vite + TypeScript
scripts/                       ← Gemini backdrop generator + proofing UI
design/                        ← planning docs, UI mockup, mood board

vendor/llama.cpp/              ← bundled llama-server.exe
models/                        ← Gemma 4 E4B Q8 GGUF
```

## Troubleshooting

- **`llama-server failed to come up`** — port `:8080` in use (`make kill`), GPU OOM (drop `LLAMA_NGL`), corrupt GGUF (delete `models/` and re-run `make download-model`).
- **Browser shows "booting…" forever** — sidecar is up but the SSE stream is silent. Check the sidecar's stdout for the first agent call. Usually means llama-server isn't responding to `/v1/chat/completions`.
- **Inferno mode fails immediately** — inferno isn't reachable at `INFERNO_URL` (default `http://127.0.0.1:7770`). The lifespan probe fails loud by design; no silent fallback. Start inferno or drop `SMALLVILLE_USE_INFERNO`.

## License + attribution

Courier Prime via Google Fonts. Gemma 4 E4B via Hugging Face.
