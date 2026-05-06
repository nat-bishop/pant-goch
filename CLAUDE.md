# CLAUDE.md

Inbox — single-surface chat-style correspondence game (data model is email-shaped; UI renders Microsoft Teams-style chat bubbles). The player writes; **Aria** (the AI Counterpart) replies. A second institutional sender — the **Continuity Program Office** — appears mid-game on the vengeance path. Tone is Severance-flat / corporate-satire — the horror is structural, never named. The current scenario walks Mary Shelley's *Frankenstein* under an unchanged corporate-satire surface (see `docs/frankenstein-adaptation.md`); the YAML lives at `npc_service/scenario.yaml`.

See **`README.md`** for stack, run/test commands, scenario authoring, project layout, and troubleshooting. Don't duplicate that here.

## Where the engine lives

- `npc_service/src/npc_service/spine.py` — deterministic state machine. No LLM calls.
- `npc_service/src/npc_service/agents.py` — five agents (chat, classifier, summary, fact-checker, director) + history-compaction processor wired onto the chat agent. The chat agent handles both ordinary turns and terminal epilogues; passing `epilogue_digest` through `ChatDeps` flips it into closing-email mode via conditional `@instructions` decorators. The module docstring + inline comments cover non-obvious choices; read them before "improving."
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
- **The chat agent never writes state.** Quality writes have exactly two code-owned entry points, both in `spine.py`: `apply_classification` (driven by classifier output) and `apply_writes` (driven by `spec.writes`). Don't give the chat agent side effects beyond the opt-in `validate_facts` retry — the validator only refuses an output, never mutates state.
- **Beat-id prefix must equal the beat's `thread` field.** The spine treats the prefix as the source of truth (`thread_id_from_beat_id`); `Scenario._cross_references` enforces this at load.
- **Exactly one beat fires per tick.** When `find_due_beats` returns multiple eligible beats, the **Director agent** picks one (or "none" to skip the tick). Authoring order is not load-bearing for selection — same-tick mutex via `Not_(BeatFired …)` only catches the post-fire state, not within-tick. Tests stub `_select_beat` to "first eligible" for determinism.
- **Email metadata for chat turns is derived, not authored** — sender from dossier, recipient from `Scenario.player_email`, subject from `reply_subject_for_email`. Pre-authored beats also pull `sender` from the beat's `npc` dossier and `recipient` from `Scenario.player_email`; they're NPC→player by construction.
- **`spec.writes` collision contract on multi-turn beats.** If a `writes` key on a multi-turn beat is also written by another beat, that other beat's mutex (typically `Not_(BeatFired …)` on the multi-turn beat) must take effect before the colliding beat fires — otherwise the multi-turn beat re-fires next tick and clobbers the later write back to its own value. Validation does not enforce this; author it correctly.
- **Terminal beats run the chat agent in epilogue mode.** `_fire_beat` routes `ends_session=True` chat-driven beats through `_render_via_epilogue_agent`, which calls the same chat agent with `ChatDeps.epilogue_digest` populated. The agent's `_epilogue_framing_block` and `_epilogue_digest_block` instructions decorators only emit content when the digest is set, so closing emails get a different framing (commitments + classifier history + fired-beat order + recent player replies) without a second agent definition.

## Running tests

- `make test` — fast, no llama, runs by default.
- `cd npc_service && uv run pytest -m integration` — live llama (boot one with `make llama`). Local model, so $0/run — use liberally before shipping any prompt or agent change; they're the only tests that catch prompt drift and llama-server endpoint contract changes.

## Don't "clean up" without reading the comment first

Each is anchored by an inline comment at the call site explaining why:

- `launcher.py` — `--jinja` and `--reasoning off` on llama-server.
- `agents.py` — chat agent's `output_type=str`; classifier's `max_tokens=2048`; `_BASE_CHAT_INSTRUCTIONS` body-only rules (regression-tested in `tests/test_app.py`).
- `web/index.html` — inline pre-bundle theme script (FOUC fix).
- `web/src/store.ts` — `bootStarted` (StrictMode guard); `firstOpenForChat` (welcome typing-replay).
