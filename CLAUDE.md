# CLAUDE.md

HANDLER v2 is an LLM-driven investigation game set in the Welsh Marches village of Pant Goch. The player is a freelance journalist looking into a death the local police closed too quickly. The world (locations, cast, the truth) is hand-authored YAML; the LLM improvises scenes inside it via two-stage structured emission.

## Repo

```
SPEC.md           # authoritative v2 design — source of truth
src/handler/      # Python engine + pydantic-ai agents
world/            # hand-authored bible (player content)
web/              # Vite + React frontend
tests/            # pytest
scripts/          # Gemini backdrop generator + proofing UI
design/
  mockup-stage/   # static HTML/React reference for the final UX
Makefile          # all dev commands
```

### Engine modules (`src/handler/`)

One-line orientation; read the module docstring for detail.

- `models.py` — Pydantic types for the world bible.
- `world.py` — loads + validates the bible (10 invariants from SPEC §2.6).
- `state.py` — `State` dataclass; session-persistent source of truth.
- `schemas.py` — LLM output schemas. State-**in**dependent structural rules live here as validators.
- `agents.py` — pydantic-ai agents + per-call dynamic `Literal[...]` narrowing for ids. State-**dependent** rules live here.
- `prompts.py` — Phase 1 / Phase 2 prompt builders (phase-1 classifier, prose writer, choice author, free-text gate, caption author); every per-agent system prompt is byte-stable across the session for KV-cache hits, so per-turn/per-location detail (e.g. the choice author's location-named leave guidance) lives in the user message.
- `engine.py` — pure helpers (reachability, state application). No LLM, no SSE.
- `scene_loop.py` — `play_session(...)`, the single driver shared by CLI, SSE backend, and tests.
- `main.py` / `sse.py` — FastAPI sidecar + SSE event surface.
- `__main__.py` — text-only CLI playthrough.
- `launcher.py` / `config.py` — bundled llama-server spawn + env-driven config.

## Authoring (`world/`)

The whole player-facing content surface is ~15 YAML files. The author writes prose; the LLM improvises scenes inside it.

```
world/
  locations/*.yaml   # one per location: description, notes, adjacent_to, short_name, exit_lines (1-3 hand-written departure lines, no destination named)
  cast.yaml          # NPCs: themes, knowledge, attitude, voice_sample, default_location
  truth.yaml         # crime + beats (requires_beats / blocked_by_beats) + endings
  brief.yaml         # opening_seed + player_starts_at
  few_shots.yaml     # optional Phase 2 example exchanges (NPC voice)
  map.yaml / map.png # frontend map geometry + backdrop
```

## Load-bearing invariants

- **Grammar-sampled enums over the catalog.** `beat_id`, `ending_id`, `npc_id`, `location_id` are narrowed per-call to `Literal[*valid_ids]`. llama.cpp's grammar sampler rejects invalid ids at decode time — the model literally cannot emit one it doesn't have access to. State-dependent constraints (reachable beats, satisfied endings) are grammar-enforced too. The diegetic exit is a scalar `leave: bool` flag on a Continue option (NOT a grammar-narrowed id and NOT a union variant — a per-option continue/leave union collapsed under the grammar sampler, the "leaves_for scar"); the engine routes a picked `leave` option to scene close.
- **Don't mix structural and state-dependent rules.** State-independent (prose length, ≤1 leave option) → validators on canonical schemas in `schemas.py`. State-dependent (which beats are reachable, which endings have prereqs met) → per-call Literal narrowing in `agents.py`.
- **Player-driven endings.** Endings gate on decision beats authored as mutually-exclusive via pairwise `blocked_by_beats`. The engine cannot summon a climax; the player commits in fiction.
- **Gemma 4 sampling.** temp=1.0 / top_p=0.95 / top_k=64 for all agents. Do not drop temp for "judgment" turns — the model needs creative latitude inside the bible's constraints.

## Commands

- `make dev` — sidecar + Vite together (inferno-routed). Standard dev flow. → http://localhost:5173
- `make cli` — text-only playthrough in the terminal.
- `make test` — fast unit suite, no live llama.
- `make test-integration` — includes the live-llama integration test (`tests/test_live_integration.py`).
- `make kill` — force-kill orphans on :8000 / :8080 / :5173-5176.
- Single test: `uv run pytest tests/test_engine.py::test_name -q`
- `make help` — full list.
