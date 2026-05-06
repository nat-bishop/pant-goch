# Inbox

Single-surface chat-style correspondence game (data model is email-shaped; UI renders Microsoft Teams-style chat bubbles). The player writes; an AI Counterpart — **Aria** — replies on every send. A second institutional sender — the **Continuity Program Office** — appears mid-game on the vengeance path.

The current scenario, **"Continuity Program — Cohort 014"**, is corporate-satire on the surface and walks Mary Shelley's *Frankenstein* underneath: Aria as creature, the player as creator, a midpoint bargain, four terminal endings (integration / mutual ruin / exile / vengeance) selected by the run's texture. Severance-flat HR-bright tone. The horror is structural, never named. See `docs/frankenstein-adaptation.md` for the full design rationale and `docs/authoring-guide.md` for new-scenario guidance.

## Architecture

```
┌──────────┐  HTTP/SSE  ┌──────────────┐  spawn  ┌──────────────┐
│ Browser  │ ─────────► │ npc_service  │ ──────► │ llama-server │
│ (Vite +  │            │ (FastAPI +   │  /v1/   │ (llama.cpp)  │
│  React)  │ ◄───────── │  Pydantic AI)│ ◄────── │              │
└──────────┘            └──────────────┘         └──────────────┘
```

- **npc_service** owns the deterministic spine (state, beats, scheduler) and runs six Pydantic AI agents on a shared llama-server handle:
  - `chat_agent` — `output_type=str`, `deps_type=ChatDeps`, runs once per NPC reply with per-thread `message_history`. Per-run `@instructions` decorators inject dossier, optional plot directive, current commitments, player canon, and (for terminals) an epilogue framing + per-run digest. Used for ordinary chat turns AND closing epilogues; ChatDeps's `epilogue_digest` flips the framing without a second agent definition. An `@output_validator` engages only on beats with `spec.validate_facts=True`, running the fact-checker against the directive.
  - `classifier_agent` — `NativeOutput(list[AxisResult])`. Judges the player's reply against one or more beat-authored axes in a single batched LLM call. Each axis returns one of its `allowed_values` plus a one-sentence rationale. Runs only when the replied-to beat has authored `classify_reply_axes`.
  - `extractor_agent` — `NativeOutput(ExtractorOutput)`. Runs in parallel with the classifier on every player /send (via `asyncio.gather`, latency-neutral). Harvests open-ended player-introduced specifics — name, role, named people, places, claims about self — into `state.player_canon` after a substring-anchor + salience-threshold filter.
  - `summary_agent` — invoked only inside the chat agent's history-compaction `history_processors` when `message_history` exceeds the token budget. Summarise-then-replace the oldest messages.
  - `fact_checker_agent` — `NativeOutput(FactCheckResult)`. Opt-in via `spec.validate_facts: true`; the chat agent's output validator raises `ModelRetry` on missing facts or violations.
  - `director_agent` — picks one beat per tick when `find_due_beats` returns ≥2 eligible. Output schema is constructed per-call with `chosen_beat_id` constrained to a closed Literal of the eligible ids plus "none" (skip-tick). Short-circuits without an LLM call when 0 or 1 beats are eligible.
- **Browser** is a Vite + React client styled as a Teams-like chat. Server is the source of truth; the Zustand store is a render cache and refreshes after every mutation. A DEV-only `DebugPanel` overlays per-tick decisions (eligibility, selection rationale, quality deltas) on top of the running game.
- **Two player verbs**: `/open` (mark email read; mutates no qualities) and `/send` (reply). Pacing is action-counter driven, never time-driven.
- **Two reply paths**:
  - **Chat turn (default fallback)** — fires immediately within the same `/send` SSE if no plot beat is eligible. Free conversation in character; no per-turn authoring.
  - **Plot beat (authored milestone)** — when a beat's preconditions hold, it fires as the next chat-agent turn with the spec's intent / must-include / must-not injected as per-run instructions. Beats can be `multi_turn: true` (re-fires every tick — `aria.calibration` and `aria.loop` are the canonical examples) and/or `ends_session: true` (terminal — runs the chat agent in epilogue mode, then flips `state.ended` after the email is emitted).

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
make test                                       # 143 unit tests (no llama; uses TestModel)
cd npc_service && uv run pytest -m integration  # 6 live-llama tests (needs `make llama` running)
make typecheck                                  # tsc on web/
```

Unit tests run by default; integration tests are skipped via `addopts = "-m 'not integration'"` in `pyproject.toml`. The integration suite exercises the real `/v1/chat/completions` and `/tokenize` round-trips — slower (~5–15s per chat turn on Gemma 4 E4B), but the only thing that catches prompt drift, classifier sycophancy, and llama-server endpoint contract changes. Local model, $0/run — use liberally before shipping any prompt or agent change.

Test layout:
- `test_spine.py` — predicate evaluation (incl. `RecentClassifierValue`, `Or_`, `PlayerCanonExists`), `apply_classification` (set/accumulate/clamp/verdict-only), `apply_writes` and `apply_writes_by_axis`, multi-turn re-fire, scenario load-time validation (axis collisions, unknown qualities, validate_facts requirements, writes_by_axis arm key checks), beat scheduler edge cases.
- `test_app.py` — FastAPI app with TestModel-overridden agents: boot tick fires the welcome only; `/open` is read-only; `/send` rejects empty bodies; chat-driven beats fire on first reply; multi-turn loops re-fire; `thread_messages` accumulates `ModelRequest`/`ModelResponse` across sends; rendered bodies contain no `Subject:`/`From:`/`To:`/JSON/markdown header leakage; classifier writes apply, off-spec/duplicate/unknown-axis defenses kick in; `tick_decision` SSE event ships the decision payload; chat-agent failures emit a structured `error` event (no torn streams); `/send` returns 409 after session end; the fact-checker validator engages only on `validate_facts: true` beats.
- `test_director.py` — Director receives both ids when ≥2 beats are eligible; short-circuits when 1 beat is due; "none" returns skip-tick.
- `test_player_canon.py` — `Or_` predicate semantics; `PlayerCanonExists`; canon extraction (threshold, replace semantics, custom threshold); `_slugify_name` rules; `resolve_player_email` resolution order (canonized name → explicit override → fallback); player_canon snapshot ranking (recency × salience).
- `test_scenario.py` — Continuity Program scenario loads cleanly; calibration-done predicate (`Or_(name canonized, ≥4 replies)`) is correctly threaded through the YAML anchor; phase-1 calibration peers don't gate on `*calibration_done`; phase-2 body beats do; bargain followup writes per-arm; bridge/Office/terminals fire on the right state shapes.
- `test_history_compaction.py` — under-budget noop, short-history noop, over-budget summarise-then-replace splice.
- `test_app_integration.py` and `test_history_compaction_integration.py` — live llama-server roundtrips.

## Smoke-test the loop

After `make run`:

1. The chat opens with Aria typing… then her welcome bubble pops in (welcome typing-replay on first session-open). Theme follows your OS preference; toggle in the top-right.
2. Reply substantively. Player bubble appears instantly on the right; "Aria is calibrating…" indicator at left; her reply pops in after a paced delay (25 chars/sec, clamped 1.5–4.5s).
3. Aria's first ~3–4 replies are calibration: she asks for your name, role, one personal detail. Once she has your name (extracted by `extractor_agent` into `state.player_canon["name"]`) OR you've replied 4 times without giving one, calibration ends and the body of the run unlocks.
4. As you correspond, two qualities track in the background: `creator_affect` (the texture of your replies — paternal / cold / neutral, replaced each reply) and `creature_grievance` (refusals push it up, engaging with the naming demand pushes it down — accumulating, floor-clamped at 0).
5. Around reply 4–5 Aria offers a bargain: an HR-flavored authorization to draft preliminary correspondence on your behalf. Accept / refuse / hedge — `commitment.bargain_state` records the verdict; the followup beat branches on it via `writes_by_axis`. If you accept and then go cold, the bargain-betrayed beat fires; if grievance climbs high enough, Aria acts in your name and the **Continuity Program Office** delivers a procedural notice.
6. After ≥8 replies, one of four terminals fires based on the run's texture (`creator_affect` window × `creature_grievance` × bargain history): **integration** (paternal + low grievance + bargain offered), **mutual ruin** (paternal + high grievance), **exile** (cold + low grievance), or **vengeance** (cold + high grievance + Office dispatched). Each terminal runs the chat agent in epilogue mode against a per-run digest; the closing email is tailored to your run.
7. Open the DEV debug panel (bottom-right) to watch quality deltas, eligibility, and the Director's per-tick rationales as they happen.

Smoke-pass criteria: welcome typing-replay works, theme toggle is FOUC-free, plot-beat replies land with the authored subject as an uppercase eyebrow, the calibration scene cap-exits cleanly, and the four endings actually differentiate based on play texture.

## Authoring a new scenario

`npc_service/scenario.yaml` is the authoring surface. Schema is validated strictly at sidecar startup (see `npc_service/src/npc_service/models.py`):

- `title`, `premise`
- `email_domain` — domain part of the player's address. Local-part is derived from `state.player_canon["name"]` once canonized; pre-canonization uses `fallback_player_local`.
- `qualities`: continuous numeric state with optional `min`/`max`/`default`. Authored qualities + the `commitment.*` namespace (set-only via `spec.writes`, surfaced to the chat agent on every render).
- `npcs`: each with `display_name`, `email_address`, `relationship_to_owner`, `base_voice_register`, `base_voice_forbidden`, `will_say`, `will_not_say`, and `replies_enabled` (false → frontend hides composer; institutional one-way senders).
- `threads`: each with optional `npc:` (system threads have `npc: null`). **Beat IDs must be prefixed with their thread ID** — the spine treats the prefix as the source of truth.
- `beats`: each beat has `id`, `thread`, `npc`, `spec`, optional `preconditions`, optional `classify_reply_axes`, optional `multi_turn` / `ends_session`.
  - `spec.pre_authored_prose` ships the body verbatim — no LLM call. Use only when exact phrasing matters (e.g. boot welcome).
  - `spec.intent` + `must_include_facts` + `must_not` injected as per-run instructions on the chat agent. The agent's prose IS the email body.
  - `spec.writes` — SET-only writes onto state qualities. Authoring convention: `commitment.*` keys for fact-surfacing.
  - `spec.writes_by_axis` — per-arm writes keyed on the most recent classifier verdict for an axis. Collapses what would otherwise be N parallel followup beats.
  - `spec.validate_facts: true` — opt-in fact-check via `fact_checker_agent`. Default off keeps loop beats free of validator latency. All four terminals carry it.
  - `classify_reply_axes` — list of `ClassifyReply` axes. Multi-axis batched into a single classifier call. Quality-writing axes have `quality:` set; verdict-only axes set `axis_name` and omit `quality` (used by `RecentClassifierValue` predicates without a state write).
- **Predicates**: `quality_gte`, `quality_lt`, `player_replies_to_thread_gte`, `beat_fired`, `recent_classifier_value` (window over per-axis history with `match: any | all | int`), `not`, `or` (list-AND is implicit at precondition level), `player_canon_exists`.

Want a thread that's pure free chat with no plot moments? Author a dossier + thread and write zero plot beats. Every player /send produces an immediate chat-turn reply driven entirely by dossier + accumulated `message_history`. The conversation can run as long as the player keeps writing.

See `CLAUDE.md` for engine invariants ("things that look removable but aren't") and `docs/authoring-guide.md` for the full new-scenario workflow.

## Project layout

```
smallville/
├── README.md                 ← this file
├── CLAUDE.md                 ← engine invariants for AI assistants
├── Makefile
├── docs/
│   ├── authoring-guide.md         ← how to write a new scenario on this engine
│   ├── frankenstein-adaptation.md ← design rationale for the shipped scenario
│   ├── LLM-game-research.md       ← prior art / field vocabulary
│   └── agent-patterns.md
├── npc_service/              ← Python sidecar (uv project)
│   ├── src/npc_service/
│   │   ├── main.py           ← FastAPI app + endpoints + SSE; tick orchestration; epilogue digest; player-email derivation
│   │   ├── models.py         ← Pydantic schemas (predicates, beats, runtime state, DTOs, TickDecision); cross-reference validator
│   │   ├── spine.py          ← deterministic state spine; find_due_beats, apply_classification, apply_writes, apply_writes_by_axis, apply_canon_extraction
│   │   ├── agents.py         ← six agents wired against one model handle; ChatDeps; player-canon snapshot/format helpers; director output-type builder
│   │   ├── launcher.py       ← llama-server subprocess + Job Object
│   │   └── config.py         ← pydantic-settings (ports, llama lifecycle, history compaction)
│   ├── scenario.yaml         ← Continuity Program (2 NPCs, 12 beats: welcome + calibration + 2 early forks + bargain chain + bridge + Office + loop + 4 terminals)
│   ├── tests/                ← test_spine, test_app, test_director, test_player_canon, test_scenario, test_history_compaction (+ integration variants); fixtures/test_scenario.yaml
│   └── pyproject.toml
└── web/                      ← Vite + React frontend (Teams-style chat)
    ├── index.html            ← inline boot script: theme class on <html> before bundle loads (FOUC-free)
    ├── src/
    │   ├── App.tsx, main.tsx
    │   ├── api.ts            ← fetch + SSE wrappers; TickDecision/QualityDelta types
    │   ├── store.ts          ← Zustand: chatMessages, theme, pendingTypingFor, firstOpenForChat, lastTickDecision, qualityHistory, debugCollapsed
    │   ├── hooks/             ← useStreamSend (paced reveal + tick-decision capture), useScrollAnchor, useAutoGrow, useTheme
    │   ├── util/             ← email.ts (stripAddress), avatar.ts (gradient + initials), pacing.ts
    │   └── components/        ← AppHeader, Avatar, ChatHeader, ChatRow, ChatSidebar, DebugPanel (DEV-only), ErrorBoundary, MessageBubble, MessageComposer, MessageGroup, MessageList, NewMessagePill, PresenceDot, ThemeToggle, TypingIndicator
    ├── tailwind.config.ts    ← darkMode: "class"; semantic tokens via CSS vars; Inter + JetBrains Mono
    ├── package.json
    └── vite.config.ts
```

## Troubleshooting

- **Empty agent responses or 19K-token monologues** — `--jinja` and/or `--reasoning off` missing on llama-server. The bundled launcher sets both; if you run llama-server yourself, include them.
- **`No module named npc_service`** — run `uv sync` from `npc_service/`.
- **Streaming arrives in big bursts** — verify your reverse proxy isn't buffering. The server sets `X-Accel-Buffering: no` on the `/send` SSE response.
- **Port 8000 already in use** — set `NPC_SERVICE_PORT` in `npc_service/.env` and update `web/vite.config.ts`'s proxy target. Or `make kill-zombies` to force-kill stragglers on :8000/:8080/:5173-5176.
- **Aria's reply takes too long to "type"** — the typing-indicator delay is `body.length / 25` clamped to `[1.5s, 4.5s]`. Tune `CHARS_PER_SECOND`, `MIN_DELAY_MS`, `MAX_DELAY_MS` in `web/src/util/pacing.ts`.
- **Theme flashes wrong on first paint** — the inline boot script in `index.html` should run before the bundle. If you've moved theme detection to `main.tsx`, you'll see a 50–200ms flash.
- **Chat turns leak `Subject:` / `From:` headers** — the chat agent's base instructions block this; the body-only regression guard in `test_app.py` covers it. If it fires under a real LLM, tighten `_BASE_CHAT_INSTRUCTIONS` in `agents.py`.
- **Debug panel doesn't appear** — it gates on `import.meta.env.DEV` at the App.tsx mount site; only present in `make web` / `make run`. Production bundles tree-shake the entire DebugPanel module out.
