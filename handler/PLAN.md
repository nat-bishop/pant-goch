# HANDLER v2 — remaining work plan

Copied from `~/.claude/plans/you-re-picking-up-a-purring-finch.md` and
updated with current state as of 2026-05-14. Use this as the in-repo
source of truth for the remaining build. The original plan file may
get archived once we land further; this one travels with the branch.

---

## Status at a glance

| Milestone | Status | Tests / artifact |
|-----------|--------|-------------------|
| M0–M3 (skeleton, world loader, engine, scene runner) | ✅ committed at `7a5396e` | 90 unit + 1 live |
| M7 — full Pant Goch world authoring | ✅ done | world/ has 10 locations, 9 NPCs, 18 beats, 6 endings; loader green |
| M6 — log writer | ✅ done | tests/test_log_writer.py (8 tests) |
| **Eval/sim refactor** | ✅ done (added in flight) | `play_session(...)` + sync/async callbacks |
| M4 — text-only frontend + SSE backend | ✅ done | 10 SSE tests; FastAPI routes wired; React/Vite frontend builds (152 kB / 49 kB gz) |
| **Phase 1/2 prompt split + conversation Phase 2** | ✅ done | split system prompts; NPC.voice_sample; world/few_shots.yaml; SceneRunnerRuntime; scene-bounded message_history; 129 unit tests green |
| M3.5 — beat-firing rubric gate | 🟡 **awaits human run** | rubric template at `playtest/m3.5-gate-run-1.md` |
| M5 — clickable map | ⏳ pending | needs author-supplied map image (Gemini prompt at end of this file) |
| M8 — full playtest in browser | ⏳ pending | runs once M3.5 + M5 are settled |
| handler/CLAUDE.md | ⏳ pending | write after M8 |

**Test totals:** 129 unit tests passing (110 baseline + 19 new across prompt-split, conversation history, voice_sample, few_shots). 1 live integration test against inferno's Gemma 4 E4B carried over from M3.

## Prose-voice refactor (2026-05-14)

Split the scene-runner's single shared system prompt into two:
- **Phase 1** (classifier) — keeps the full bible: locations + cast + beats + notes_to_llm + endings.
- **Phase 2** (narrator) — slim, voice-focused: locations + cast-with-voice-samples + crime overview + few-shot examples. NO beats, NO notes_to_llm, NO endings, NO per-NPC `knowledge`.

Phase 2 is also now scene-bounded conversation: `SceneRunnerRuntime.phase2_message_history` accumulates within a scene, resets at `enter_scene`. Failed retries don't pollute history. Trade-off: lose the inter-phase KV cache hit per turn; gain Phase 2 attention focus on prose, plus intra-scene voice continuity. New data: `NPC.voice_sample` field in `cast.yaml`, optional `world/few_shots.yaml` (one or two short example exchanges). Anti-slop list deferred (mine outputs first).

---

## Context

V1 (the spy-handler game in `handler/`) failed for three reasons: too many
entities to model from text, no time to attach to characters, and the
desk-handler framing kept the player emotionally cold. V2 inverts the
bargain — author a **world bible** (locations, cast, truth, brief), let
the LLM **improvise scenes inside it** via structured output. Map is the
primary navigation surface.

The architectural bet: Gemma 4 E4B can judge beat moments competently
inside the constraint of `requires_beats` / `blocked_by_beats`,
grammar-sampled id enums, ≤2 unlock_beat cap, and `ModelRetry`-backed
application validation. M3.5 is the gate that proves or kills that bet.
**Per the in-flight pivot, we dropped the stub world** — the rubric runs
against the real Pant Goch authoring (M7 below).

V1 stays as reference-only in `handler/`. V2 is a clean greenfield in
`handler/`.

---

## What's done (this session)

### M7 — Full Pant Goch world

Authored 13 YAML files in `handler/world/`:

- **Locations** (10): `the_old_vicarage`, `the_post_office`,
  `the_slaughtered_lamb`, `st_caradocs`, `the_lane`, `knighton`,
  `the_black_dog_bend`, `tan_y_coed`, `lower_hafod`, `bryn_iago_wood`.
  Adjacency per SPEC §6.2 sketch; 4–8 sentence atmospheric descriptions
  + 4–5 specific notes each.
- **`cast.yaml`** (9 NPCs): tom, maud, iwan, wendy, eluned, dylan,
  gareth, helen, owain — each with themes, knowledge, attitude.
- **`truth.yaml`** (18 beats + 6 endings):
  - 9 positive investigative beats (tom_named_a_vehicle,
    tom_admits_full_silence, maud_confirms_the_affair,
    dylan_admits_the_relationship, dylan_admits_he_knows,
    helen_admits_she_suspected, iwan_grants_you_intimacy, eluned_breaks,
    tom_decides_to_speak_to_police).
  - 5 negative beats (tom_closes_off, dylan_refuses_to_speak_again,
    helen_wont_take_your_call, eluned_throws_you_out,
    village_runs_you_out).
  - 4 decision beats (decided_to_file, decided_to_walk_away,
    decided_to_confront_eluned, decided_to_tell_iwan_privately) —
    mutually exclusive via pairwise `blocked_by_beats`.
  - 6 endings: published_story, buried_the_story,
    eluned_arrested_after_confrontation, iwan_finds_peace,
    run_out_of_the_village, tom_takes_responsibility — each gated on
    its decision or conditional-fire beat.
- **`brief.yaml`** — opening at the_old_vicarage, full editor brief +
  Iwan's letter framing.

Loader green: `uv run python -c "from handler.world import load_world; load_world(Path('world'))"` → 10 / 9 / 18 / 6.

### M6 — Log writer

Added the second pydantic-ai agent. Conversation-based; one
user+assistant pair appended to `LogWriterRuntime.message_history` per
scene-end. System prompt carries the full bible + voice rules (first-
person past-tense journalist's field notes, 80–120 words).

Files touched:
- `state.py` — added `LogEntry` dataclass (scene_id, location_id,
  npcs_present, text — no `action` field, v2 logs are scene-keyed).
- `prompts.py` — added `build_log_writer_system_prompt` +
  `build_log_writer_user_prompt`.
- `agents.py` — added `LogWriterRuntime`, `build_log_writer_agent`,
  `make_log_writer_runtime`, `emit_log_entry`. Imports `ModelMessage`
  from `pydantic_ai.messages`.
- `scene_loop.py` — `play_session` invokes the log writer between
  scenes and after EndGame; appends text to caller-managed
  `case_file_entries`.
- `__main__.py` — CLI prints each entry indented after the closing
  prose.
- Tests: `tests/test_log_writer.py` (8 tests covering prompt
  assembly, runtime construction, anchor extraction, history growth,
  empty-dialogue defensive path).

### Eval/sim refactor (in flight)

Pulled the session-loop machinery into a single composable function so
CLI / web / eval drivers share one surface:

- New `play_session(bundle, log_writer, state, world, *,
  pick_option, pick_walk, ...)` in `scene_loop.py` — runs the full
  scenes-until-game-over loop with pluggable callbacks. Returns the
  terminal ProsePart and sets `state.game_over` on EndGame.
- Three callback hooks: `pick_option` (returns int), `pick_walk`
  (returns location id), `on_log_entry` + `on_scene_close` (driver
  side-effects).
- Both sync and async callbacks accepted via `_maybe_await(...)`.
  Tests use sync lambdas; SSE backend uses async future-resolvers.
- CLI in `__main__.py` shrunk to thin driver: stdin `_prompt_pick` /
  `_prompt_walk` / `_print_log_entry` callbacks → `play_session(...)`
  → print epilogue.
- Tests: 3 added `play_session` tests in `test_scene_loop.py`.

### M4 — SSE backend + text-only frontend

**Backend (`handler/src/handler/`):**

- `sse.py` (new) — paced_word_emitter + `_words_from_text` +
  `_words_from_stream` lifted verbatim from v1
  `handler/src/handler/main.py:312-389` (80 ms cadence floor
  non-negotiable). Plus typed event-builder helpers
  (scene_start_event, state_change_event, options_event,
  scene_exit_event, location_change_event, game_over_event,
  error_event).
- `scene_loop.py` — `play_scene` + `play_session` accept optional
  `emit_event` callback. When supplied, scene_loop fires
  scene_start → state_change* → prose_word* (paced) →
  options/scene_exit/game_over → location_change (between scenes)
  → log_entry_word* (paced log entry). Async pick/walk callbacks
  awaited at the right boundaries.
- `main.py` — added:
  - `build_model(api_url)` factory (shared with __main__.py); sets
    `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)`.
  - Lifespan extended to load world + build agent bundle once.
  - `SessionRuntime` dataclass holding bundle, log_writer, state,
    world, event_queue, pick_future, walk_future, play_task.
  - Routes: `POST /start` (kicks off play_session in background
    task, emits events to per-session queue), `POST /pick`,
    `POST /walk` (with defensive adjacency check), `GET /events`
    (long-lived SSE stream draining the queue), `GET /world`
    (static world metadata for the frontend).

**Frontend (`handler/web/`):**

- React 18 + Vite 6 + TypeScript. Plain CSS (no Tailwind for v2 MVP).
- Verbatim lifts:
  - `web/src/hooks/useStream.ts` — `useStream` hook with the
    setTimeout(0) StrictMode-deferred-fetch trick + the 14-line
    comment explaining why.
  - `web/src/api.ts` — `readSseStream` async generator.
  - `web/vite.config.ts` — `/api → :8000` proxy.
- New code:
  - `src/store.ts` — Zustand store with `handleEvent` dispatcher
    for the v2 SSE event vocabulary. Tracks phase
    (idle/in_scene/awaiting_pick/between_scenes/game_over/stranded),
    scenes (with per-turn prose words + options + player_pick),
    case_file (per-scene streaming entries).
  - `src/App.tsx` — main UI. Header + scene panel + case-file
    sidebar. SceneHistory renders all scenes; current scene bright,
    prior scenes dim. StreamingText animates word fade-in.
    OptionsPanel + WalkPanel + GameOverPanel + StrandedPanel for
    the four interactive states.
  - `src/index.css` — single dark theme with sepia accents
    (parchment palette). Courier Prime via Google Fonts CDN
    preloaded in `index.html`.
- Tests:
  - `tests/test_sse.py` (10) — `_words_from_text` /
    `_words_from_stream` correctness, paced emitter timing
    (≥80 ms cadence at the configured interval, idx + sentinel
    sentinel), `play_scene` emits scene_start/state_change/
    prose_word/options/scene_exit/game_over at right boundaries,
    `play_session` emits location_change + log_entry_word in
    cross-scene flow.
- Build verified: `cd web && npm install && npm run build` → 33
  modules → 152 kB JS / 3.6 kB CSS (49 kB gz).

---

## What's left

### Step A — M3.5 gate (human-run)

**Artifact ready:** `handler/playtest/m3.5-gate-run-1.md` has:
- Pre-defined intended-beat-per-scene table for the default
  exploration route (10 scenes from the vicarage).
- 4 binary scoring criteria (right-beat-right-moment, no spurious
  beat, no premature ending, no invented NPC).
- Score sheet template.
- Failure-mode catalog mapping observation → cheapest escalation rung.

**To run:**
```
cd handler
SMALLVILLE_USE_INFERNO=1 uv run python -m handler
```
Play 10 scenes; fill in the score table; commit the marked-up file.

**Threshold:** ≥8/10 scenes scoring 4/4 → M5. Else escalation per Step B.

### Step B — M3.5 escalation (conditional)

If <8/10, add per-beat floors per the cheapest rung that addresses the
observed failure mode. Each rung is a small engine extension:

- **(a) `min_co_presence: int` on Beat** — `models.py` adds field,
  `engine.validate_state_change_bundle` checks against
  `state.npc_scenes_co_present[<primary_npc>]`, `engine.reachable_beat_ids`
  filters unreachable, `prompts._render_reachable_beats` shows the floor.
  Authoring: add `min_co_presence: 2` + `primary_npc: tom` to the beat.
- **(b) `min_scene_turns: int` on Beat** — same shape, checks
  `len(state.dialogue_history)` at firing time.
- **(c) Third-stage classifier** — new short LLM call after phase 2;
  structured `Literal[*unfired_beat_ids, "NONE"]` output. Use only if
  phase 1 reliably under-fires.
- **(d) Soft-prereq DSL** — last resort; reintroduces v1's dropped
  predicate language. Consider model swap (Sonnet via inferno) first.

Each floor lands as an inline `# Floor: min_co_presence=2 (run-1, tom_named_a_vehicle 3/10 too early)` annotation in `truth.yaml` AND a
note in `handler/CLAUDE.md` (written at Step D).

### Step C — M5 clickable map

- **Map image asset.** Author-supplied via Gemini. Prompt below.
  Drop the file at `handler/web/public/map.png`.
- **Coordinate authoring.** New `handler/world/map.yaml` keyed by
  location id, each entry `{ x: <0–1>, y: <0–1> }`. Fractional
  coordinates make the map responsive without re-authoring on resize.
  `world.py::load_world` extended to read it (optional for tests).
- **Frontend.** New `web/src/components/Map.tsx` with hotspots
  positioned over the background image. Two states:
  - Active (`phase=between_scenes`): adjacent locations bright +
    hover-glow + clickable (POST `/walk`). Non-adjacent dim/inert.
  - Inactive (`phase=in_scene` / `awaiting_pick`): all dim, cursor
    `not-allowed`, hover tooltip "Pick a 'Leave for…' option to walk
    away." Click suppressed (backend rejects defensively too).

The map replaces the current text-list `WalkPanel`. The "Leave for…"
options in scenes stay schema-enforced (>=1 per ContinueScene), making
the bargain honest.

### Step D — M8 full playtest + handler/CLAUDE.md

- Author plays Pant Goch to at least 2 endings
  (`published_story` + `buried_the_story` minimum) in the browser.
- Tune `notes_to_llm` on beats/endings; adjust NPC attitudes if
  behaviour feels off.
- Drift smoke test: scene at pub at game start → visit other
  locations → return to pub. Verify Tom reflects accumulated state
  (case file + unlocked beats), not fresh acquaintance. No invented
  NPCs (grep prose vs cast.yaml).
- KV-cache hit-rate spot-check via pydantic-ai `Hooks.on.before_model_request`.
- Write `handler/CLAUDE.md` capturing v2 invariants: two-stage
  emission, ≤2 unlock_beat cap, leave-pick semantics,
  `OpenAIModelProfile` flag requirement, KV-cache layout contract, any
  M3.5 floors that landed in `truth.yaml`.

---

## Map-image generation prompt (for Step C)

Paste to Gemini and tweak:

```
Hand-drawn village map of "Pant Goch" — a small Welsh Marches village in
a hollow between two streams, with surrounding hillside. Watercolour
and ink, parchment-toned background (warm cream, slight texture, lightly
aged but legible), brown-black hand-lettered place names in a casual
serif. Top-down 3/4 perspective. No grid; soft contour lines for hills.

Exactly 10 named locations, laid out so the adjacency graph reads
spatially:

- Bryn Iago Wood — wooded ridge at the top (north). Connects down to
  Tan-y-coed.
- Tan-y-coed — stone farmhouse + barns, upper-left under the wood.
  Connects to Bryn Iago Wood, The Lane, and Lower Hafod.
- Lower Hafod — small cottage + polytunnel + tin-roofed shed,
  upper-right of Tan-y-coed, end of a track. Connects only to
  Tan-y-coed.
- The Lane — a winding road in the middle of the map, connecting four
  things. Connects to Tan-y-coed (up), St Caradoc's (down-right),
  Knighton (down-left toward the edge), and an unlabeled continuation
  out of frame.
- Knighton — a small market town silhouette at the bottom-left edge,
  with a Costa shopfront and a police-station block tucked behind.
  Connects to The Lane and The Black Dog Bend.
- The Black Dog Bend — a sharp curve on the road south of Knighton,
  with a single bundle of flowers against the embankment. Connects
  only to Knighton.
- St Caradoc's — small parish church with an ancient yew in the
  churchyard, mid-right of the lane. Connects to The Lane and The
  Slaughtered Lamb.
- The Slaughtered Lamb — village pub with a hanging sign, below St
  Caradoc's. Connects to St Caradoc's and The Post Office.
- The Post Office — shop with a small bench outside, below the pub.
  Connects to The Slaughtered Lamb and The Old Vicarage.
- The Old Vicarage — stone-and-slate house with a yew hedge, at the
  bottom-right of the map (the player's lodgings). Connects only to
  The Post Office.

Show paths between connected locations as dotted lines. Streams curve
through the hollow. No characters in the image. No compass rose; subtle
"N" arrow only. 1600×1200 px.
```

Drop the result at `handler/web/public/map.png` and author the
fractional coordinates into `handler/world/map.yaml`.

---

## File layout (current)

```
handler/
├── PLAN.md                                ← this file
├── SPEC.md                                ← v2 spec (auth-of-record design)
├── pyproject.toml
├── uv.lock
├── playtest/
│   └── m3.5-gate-run-1.md                ← rubric template, awaits run
├── src/handler/
│   ├── __init__.py
│   ├── __main__.py                       ← CLI, thin driver over play_session
│   ├── main.py                           ← FastAPI: lifespan + routes
│   ├── launcher.py                       ← llama-server + inferno probe (lifted from v1)
│   ├── config.py
│   ├── models.py                         ← Location, NPC, Beat, Ending, Brief
│   ├── world.py                          ← loader + 10 §2.6 invariants
│   ├── state.py                          ← State, TurnRecord, LogEntry, GameOver
│   ├── schemas.py                        ← StateChangeBundle, ProsePart, Option, etc.
│   ├── engine.py                         ← apply, validate, reachable beats
│   ├── agents.py                         ← scene runner bundle + log writer runtime
│   ├── prompts.py                        ← KV-cache-friendly prompt builders
│   ├── scene_loop.py                     ← play_scene, play_session, event emission
│   └── sse.py                            ← paced word emitter + event builders
├── tests/                                 ← 111 unit tests + 1 live integration
├── world/                                 ← Pant Goch authored content
│   ├── brief.yaml
│   ├── cast.yaml
│   ├── truth.yaml
│   └── locations/                        ← 10 location YAMLs
└── web/                                   ← React + Vite + TS frontend
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api.ts                        ← /api/* HTTP + readSseStream
        ├── store.ts                      ← Zustand + handleEvent dispatcher
        ├── hooks/useStream.ts            ← lifted from v1
        ├── index.css
        └── vite-env.d.ts
```

---

## Verification (where we are right now)

- `cd handler && uv run pytest -q --ignore=tests/test_live_integration.py`
  → **109 passed**.
- `cd handler && uv run python -c "from pathlib import Path; from handler.world import load_world; w = load_world(Path('world')); print(len(w.locations), len(w.npcs), len(w.beats), len(w.endings))"`
  → `10 9 18 6`.
- `cd handler/web && npm run typecheck` → clean.
- `cd handler/web && npm run build` → 152 kB / 49 kB gz.
- Live integration test (`tests/test_live_integration.py`) — not re-run
  this session; was green at `7a5396e`. Worth a confidence check
  before the M3.5 playthrough.

End-to-end smoke (NOT YET RUN — recommend doing before commit):
1. `SMALLVILLE_USE_INFERNO=1 uv run uvicorn handler.main:app --reload --port 8000`
2. `cd web && npm run dev` (port 5173)
3. Open http://localhost:5173, click "Begin investigation", play one
   scene through, click an option, walk to an adjacent location,
   verify case-file entry appears in the sidebar.
