# CLAUDE.md

Inbox — single-surface email correspondence game. The player writes; **Aria** (the only NPC) replies. Tone is Severance-flat / corporate-satire — the horror is structural, never named. Current scenario "Continuity Program" is in `npc_service/scenario.yaml`.

See **`README.md`** for stack, run/test commands, scenario authoring, project layout, and troubleshooting. Don't duplicate that here.

## Where the engine lives

- `npc_service/src/npc_service/spine.py` — deterministic state machine. No LLM calls.
- `npc_service/src/npc_service/agents.py` — four agents (chat, classifier, summary, fact-checker) + history-compaction processor wired onto the chat agent. The module docstring + inline comments cover non-obvious choices; read them before "improving."
- `npc_service/src/npc_service/main.py` — FastAPI endpoints, SSE streaming, beat dispatch.
- `npc_service/src/npc_service/models.py` — Pydantic schemas (authoring + runtime + DTOs).
- `npc_service/scenario.yaml` — content layer. The engine is content-agnostic.
- `web/src/store.ts`, `web/src/hooks/useStreamSend.ts`, `web/src/App.tsx` — bulk of the frontend.

## Authoring vocabulary

- **Multi-turn `PlotBeat`s are the default for scenes-of-many-exchanges.** A storylet whose `multi_turn: true` re-fires on every reply with one authored intent + an evolving classifier picture is enough to carry a long scene without a beat-per-exchange. `aria.loop` is the existing example.
- **`AnchoredBeat`s are the exception**, used only when exact phrasing matters: the boot welcome (first-impression voice), or a terminal that needs to quote a specific source. Authoring each ordinary turn as an anchored beat reproduces the linearity trap the engine is trying to avoid.
- **Multi-axis `classify_reply_axes`** lets one storylet score multiple qualities per reply at the same per-send LLM cost as one. Reach for a second axis (or `RecentClassifierValue` predicates over recent classifier history) before authoring a parallel beat that just exists to score a different quality.
- **`spec.writes`** is the cheap way to record a fact ("the bargain has been offered") without a classifier round-trip. Author convention is the `commitment.*` namespace; the chat agent's commitments-block surfaces those values to every subsequent render.
- **`spec.validate_facts: true`** is opt-in fact-check on critical beats (terminals, pivotal forks). Costs one extra LLM call per render attempt; never enable on `multi_turn` loop beats.

## Invariants worth knowing

- **Pacing is action-counter / quality driven, never time.** No clocks, no decorative meters. If you're tempted to add a Wait verb or a real-time gate, look for an action-counter alternative first.
- **The chat agent never writes state.** Quality writes have exactly two code-owned entry points, both in `spine.py`: `apply_classification` (driven by classifier output) and `apply_writes` (driven by `spec.writes`). Don't give the chat agent side effects beyond the opt-in `validate_facts` retry — the validator only refuses an output, never mutates state.
- **Beat-id prefix must equal the beat's `thread` field.** The spine treats the prefix as the source of truth (`thread_id_from_beat_id`); `Scenario._cross_references` enforces this at load.
- **Same-tick beat mutex depends on YAML authoring order.** Terminals must appear before any loop they suppress, so they enter `extra_fired` first. Validation doesn't check order.
- **Email metadata for chat turns is derived, not authored** — sender from dossier, recipient from `Scenario.player_email`, subject from `reply_subject_for_email`. Anchored beats also pull `recipient` from `Scenario.player_email`; they're NPC→player by construction.
- **`spec.writes` collision contract on multi-turn beats.** If a `writes` key on a multi-turn beat is also written by another beat, that other beat's mutex (typically `Not_(BeatFired …)` on the multi-turn beat) must take effect before the colliding beat fires — otherwise the multi-turn beat re-fires next tick and clobbers the later write back to its own value. Validation does not enforce this; author it correctly.

## Running tests

- `make test` — fast, no llama, runs by default.
- `cd npc_service && uv run pytest -m integration` — live llama (boot one with `make llama`). Local model, so $0/run — use liberally before shipping any prompt or agent change; they're the only tests that catch prompt drift and llama-server endpoint contract changes.

## Don't "clean up" without reading the comment first

Each is anchored by an inline comment at the call site explaining why:

- `launcher.py` — `--jinja` and `--reasoning off` on llama-server.
- `agents.py` — chat agent's `output_type=str`; classifier's `max_tokens=2048`; `_BASE_CHAT_INSTRUCTIONS` body-only rules (regression-tested in `tests/test_app.py`).
- `web/index.html` — inline pre-bundle theme script (FOUC fix).
- `web/src/store.ts` — `bootStarted` (StrictMode guard); `firstOpenForChat` (welcome typing-replay).
