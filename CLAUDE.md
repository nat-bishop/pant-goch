# CLAUDE.md

Inbox — single-surface chat-style correspondence game (data model is email-shaped; UI renders Microsoft Teams-style chat bubbles). The player writes; **Aria** (the AI Counterpart) replies. A second institutional sender — the **Continuity Program Office** — appears mid-game on the vengeance path. Tone is Severance-flat / corporate-satire — the horror is structural, never named. The current scenario walks Mary Shelley's *Frankenstein* under an unchanged corporate-satire surface (see `docs/frankenstein-adaptation.md`); the YAML lives at `npc_service/scenario.yaml`.

See **`README.md`** for stack, run/test commands, scenario authoring, project layout, and troubleshooting. Don't duplicate that here.

## Where the engine lives

- `npc_service/src/npc_service/spine.py` — deterministic state machine. No LLM calls.
- `npc_service/src/npc_service/agents.py` — six agents (chat, classifier, **extractor**, summary, fact-checker, director) + history-compaction processor wired onto the chat agent. The chat agent handles both ordinary turns and terminal epilogues; passing `epilogue_digest` through `ChatDeps` flips it into closing-email mode via conditional `@instructions` decorators. The extractor runs in parallel with the classifier on every player /send (via `asyncio.gather` in `main._send_stream`, so it's latency-neutral) and harvests open-ended specifics into `state.player_canon`. The module docstring + inline comments cover non-obvious choices; read them before "improving."
- `npc_service/src/npc_service/main.py` — FastAPI endpoints, SSE streaming, beat dispatch (Director picks one beat per tick).
- `npc_service/src/npc_service/models.py` — Pydantic schemas (authoring + runtime + DTOs).
- `npc_service/scenario.yaml` — content layer. The engine is content-agnostic.
- `npc_service/tests/fixtures/test_scenario.yaml` — minimal Continuity-Program-shaped scenario used by `test_app.py` so app-wiring tests stay decoupled from production story content.
- `web/src/store.ts`, `web/src/hooks/useStreamSend.ts`, `web/src/App.tsx` — bulk of the frontend.

## Authoring vocabulary

- **Multi-turn beats are the default for scenes-of-many-exchanges.** A storylet whose `multi_turn: true` re-fires on every reply with one authored intent + an evolving classifier picture is enough to carry a long scene without a beat-per-exchange. `aria.loop` is the existing example.
- **`spec.pre_authored_prose` is the exception**, used only when exact phrasing matters: the boot welcome (first-impression voice), or any beat that needs to quote a specific source. The beat ships verbatim — no LLM call, no fact-checker. Authoring every ordinary turn this way reproduces the linearity trap the engine is trying to avoid.
- **Multi-axis `classify_reply_axes`** lets one storylet score multiple qualities per reply at the same per-send LLM cost as one. Reach for a second axis (or `RecentClassifierValue` predicates over recent classifier history) before authoring a parallel beat that just exists to score a different quality.
- **Verdict-only axes** (`axis_name` set, `quality` omitted): the classifier still runs and the result lands on `Reply.classifier_results` for `RecentClassifierValue` predicates to read, but no quality is written. Use when a classifier verdict gates downstream beats without needing scalar state (e.g. `bargain_response` → accepted/refused/hedged routing).
- **`spec.writes`** is the cheap way to record a fact ("the bargain has been offered") without a classifier round-trip. Author convention is the `commitment.*` namespace; the chat agent's commitments-block surfaces those values to every subsequent render.
- **`spec.validate_facts: true`** is opt-in fact-check on critical beats (terminals, pivotal forks). Costs one extra LLM call per render attempt; never enable on `multi_turn` loop beats. All four terminals in the current scenario carry it — they enforce the dossier's parallel-naming bans.
- **`NPCDossier.replies_enabled: false`** marks an NPC the player cannot reply to (institutional senders like the Continuity Program Office). Frontend hides the composer; backend does not actively reject /sends to such NPCs (defensive only).

## Invariants worth knowing

- **Pacing is action-counter / quality driven, never time.** No clocks, no decorative meters. If you're tempted to add a Wait verb or a real-time gate, look for an action-counter alternative first.
- **The chat agent never writes state.** State writes have four code-owned entry points, all in `spine.py`: `apply_classification` (driven by classifier output), `apply_writes` (SET-only writes from `spec.writes`), `apply_writes_by_axis` (per-arm writes keyed on the most recent classifier verdict for an axis), and `apply_canon_extraction` (extractor candidates → `state.player_canon`, REPLACE semantics on key collision, salience-thresholded). Don't give the chat agent side effects beyond the opt-in `validate_facts` retry — the validator only refuses an output, never mutates state.
- **Beat-id prefix must equal the beat's `thread` field.** The spine treats the prefix as the source of truth (`thread_id_from_beat_id`); `Scenario._cross_references` enforces this at load.
- **Exactly one beat fires per tick.** When `find_due_beats` returns multiple eligible beats, the **Director agent** picks one (or "none" to skip the tick). Authoring order is not load-bearing for selection — same-tick mutex via `Not_(BeatFired …)` only catches the post-fire state, not within-tick. Tests stub `_select_beat` to "first eligible" for determinism.
- **Email metadata is derived, not authored** — sender from the beat's `npc` dossier, recipient from `resolve_player_email(scenario, state)`. Recipient resolution order: (1) if `state.player_canon["name"]` is set, slugify it via `_slugify_name` and use `<slug>@<scenario.email_domain>`; (2) else if the scenario authored an explicit `player_email`, use it (back-compat for tests / legacy fixtures); (3) else fall back to `<scenario.fallback_player_local>@<scenario.email_domain>`. So early emails (pre-canonization) go to e.g. `candidate@halcyon.com`; once Aria has elicited a name, subsequent emails go to `marcus.chen@halcyon.com`. There is no authored `player_email` in the production scenario.
- **`spec.writes` collision contract on multi-turn beats.** If a `writes` key on a multi-turn beat is also written by another beat, that other beat's mutex (typically `Not_(BeatFired …)` on the multi-turn beat) must take effect before the colliding beat fires — otherwise the multi-turn beat re-fires next tick and clobbers the later write back to its own value. Validation does not enforce this; author it correctly.
- **Terminal beats run the chat agent in epilogue mode.** `_fire_beat` routes `ends_session=True` chat-driven beats through `_render_via_epilogue_agent`, which calls the same chat agent with `ChatDeps.epilogue_digest` populated. The agent's `_epilogue_framing_block` and `_epilogue_digest_block` instructions decorators only emit content when the digest is set, so closing emails get a different framing (commitments + classifier history + fired-beat order + recent player replies) without a second agent definition.
- **Player canon is extractor-only, REPLACE on collision, anti-hallucination guarded.** `state.player_canon` is written *only* by `apply_canon_extraction` (driven by `extractor_agent` output). Each candidate carries a `literal_text_quote` that MUST be a verbatim substring of the player's reply — `_send_stream` drops any candidate whose quote isn't anchored before promoting. Salience threshold (default 0.5) filters speculation; later mentions of the same key replace earlier (so "actually call me Robert" canonically replaces "Bob"). Predicates read canon via `PlayerCanonExists`; the chat agent reads a top-K snapshot via `_player_canon_block` (ranked by `(decay ** turns_since_origin) × salience`); the epilogue digest gets the full ledger.
- **`TickDecision` is the per-tick audit trail.** `_tick` builds one `TickDecision` per scheduler invocation containing eligibility report, selection method (none/sole/director), Director rationale (when ≥2 eligible), all quality deltas this tick (classifier writes + `spec.writes` + `spec.writes_by_axis`), and what fired. It's used three ways: span attributes on the tick logfire span (queryable in the Logfire UI), per-tick console log, and the `tick_decision` SSE event consumed by the DEV-only `DebugPanel`. The boot tick discards its decision (no SSE consumer).
- **Calibration scene self-stops via `Not_(*calibration_done)`, not `completes_when`.** The `*calibration_done` YAML anchor (`Or_(player_canon_exists name, player_replies_to_thread_gte 4)`) is referenced under `Not_` on the calibration beat AND as a positive precondition on every Phase-2 body beat, so the same flip unlocks the rest of the scenario. Phase-1 calibration peers (`aria.early_emergence`, `aria.naming_demand`) deliberately don't gate on it — they fire *during* the calibration window alongside `aria.calibration` and the Director picks among them per tick.

## Running tests

- `make test` — fast, no llama, runs by default.
- `cd npc_service && uv run pytest -m integration` — live llama (boot one with `make llama`). Local model, so $0/run — use liberally before shipping any prompt or agent change; they're the only tests that catch prompt drift and llama-server endpoint contract changes.

## Don't "clean up" without reading the comment first

Each is anchored by an inline comment at the call site explaining why:

- `launcher.py` — `--jinja` and `--reasoning off` on llama-server.
- `agents.py` — chat agent's `output_type=str`; classifier's `max_tokens=2048`; `_BASE_CHAT_INSTRUCTIONS` body-only rules (regression-tested in `tests/test_app.py`).
- `web/index.html` — inline pre-bundle theme script (FOUC fix).
- `web/src/store.ts` — `bootStarted` (StrictMode guard); `firstOpenForChat` (welcome typing-replay).
