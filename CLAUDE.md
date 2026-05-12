# CLAUDE.md

HANDLER — single-player spy game built around one verb: **authorize one of 3–4 options per cycle**. Three stateless LLM agents per cycle (Operations Desk drafts the SoP memo + options; the acting agent's voice renders the field dispatch). Mechanics are deterministic and authored in YAML; the LLM never decides outcomes. The MVP mission is **Op. LYNX-EXFIL** — six cycles, two agents, one authored moment (`kestrel_sacrifice`). The design lives in `HANDLER_SPEC_V2.md`; the visual reference is `_mockups/handler-authorize-desk.html`.

See **`README.md`** for stack, run/test commands, mission authoring, project layout, and troubleshooting. Don't duplicate that here.

## Where the engine lives

- `handler/src/handler/models.py` — Pydantic schemas. `Predicate` union (5 quality leaves + `BeatFired` + `MomentFired` + `Not_`/`And_`/`Or_` composites), `Effect`, `Archetype`, `Moment`, `Beat`, `Quality` (initial value + descriptor co-located), `DramaticQuestion`, `Mission`, `Option`/`OptionSet`, `State` (qualities dict + fired ledgers + outcome). `substitute_predicate`/`substitute_effect` take a `dict[str, str]` placeholder map and resolve `$name` refs (the `$agent` placeholder is the original use; arbitrary names are accepted).
- `handler/src/handler/spine.py` — pure deterministic engine. `evaluate_predicate`, `valid_archetype_combinations`, `due_moments`, `apply_archetype_effects`, `apply_moment_effects`, `fire_due_beats`, `tick_cycle_close`, `option_progresses_mission`. Plus four narrative-shape derivations consumed by the prompt builders: `derive_act_phase`, `derive_tension_band`, `select_dramatic_question`, `invalid_archetype_combinations`. No I/O, no LLM calls.
- `handler/src/handler/dsl.py` — author-facing string shorthand for predicates (`"agent[LYNX].cover > 0 and mission.stage in 1..2"`) and effects (`"agent[$agent].cover -= 2"`). `mission_loader.py` runs `normalise_mission_dsl` before pydantic; the engine sees only the canonical tagged-union dicts.
- `handler/src/handler/cycle.py` — `run_cycle` orchestration. The ONLY place LLM calls and engine mutations interleave: SoP → drafter → await_authorization → effects → moment → beats → dispatch → cycle_close. Calls into spine + agents; emits events via `session.apply`.
- `handler/src/handler/agents.py` — three pydantic-ai agents: `SOP_AGENT` (str output), `DRAFTER_AGENT` (`NativeOutput(OptionSet)`), `RENDERER_AGENT` (str output). All stateless — no `message_history` between cycles. Built per mission at lifespan boot via `build_agent_bundle(model, mission)`. `draft_with_retries` runs up to 3 retries with corrective feedback then falls back to `engine_fallback_options`.
- `handler/src/handler/main.py` — FastAPI endpoints (`/mission/start`, `/mission/{id}/stream` SSE, `/mission/{id}/authorize`, `/mission/{id}/state`). Owns `SessionRuntime` (engine session + SSE queue + auth_future + DispatchRecord buffer + cycle task), the paced word-stream emitter, and the per-cycle SSE event vocabulary.
- `handler/src/handler/events.py` — `Event` discriminated union (10 mutation/observation classes) + `fold()`. Live path: spine mutates state in place + returns delta; cycle.py emits event carrying that delta. Replay path (`session.state_at`): only fold runs, reconstructs identical state. `test_fold_replay_matches_live_mutation` pins the equivalence.
- `handler/src/handler/session.py` — `Session` dataclass (mission, agents, state, event_log, lock). `apply(event)` is the only state-write entry point. Optional JSONL persistor for replay.
- `handler/src/handler/mission_loader.py` — `load_mission(path)`. Pydantic validators catch terminal-beat outcome-key mismatches, id uniqueness, `for_each_agent`-vs-`once_per_mission` incompatibility.
- `handler/missions/lynx_exfil.yaml` — the MVP mission. 7 archetypes, 1 moment, 5 beats, 2 dossiers.
- `handler/scripts/playtest/` — heuristic option-picker playtest. Personas are weighted archetype priorities; sessions drive `/mission/start` + SSE + `/authorize` end-to-end; programmatic evaluators check terminal/budget/forced-moment/tonal-spread/no-engine-fallback.
- `web/src/App.tsx` + `components/*` + `store.ts` + `hooks/useStream.ts` — the desk UI. `<Paper>` is the document (classification stamp, memo header, SoP, options divider, options list, paper footer); `<StreamingText>` is the single word-by-word reveal primitive; `<Authorize>` is the button below paper; `<Inbox>` + `<DispatchModal>` surface past dispatches; `<OutcomeCard>` renders the terminal screen.

## Invariants worth knowing

- **Engine decides outcomes; LLM writes prose.** State writes happen only through three named spine functions (`apply_archetype_effects`, `apply_moment_effects`, `fire_due_beats`) plus `tick_cycle_close`. The LLM never mutates state, never decides whether a beat fires, never decides what an effect resolves to. Dispatches that contradict the resolved delta are prompt bugs, not alternate outcomes.
- **All three agents are stateless one-shot** per HANDLER_SPEC_V2 §8.4. No `message_history` is passed between cycles. Each cycle is a fresh prompt; continuity is preserved by explicitly including the last 1–2 dispatches as `recent_dispatches` (with same-agent preference for the renderer per §8.3) in user-side context. Don't reintroduce conversation history threading — it forces the model to defend prior assistant turns against state that mutated between calls.
- **Prompt structure is `[stable prefix] + [dynamic suffix]`.** The system prompt for each agent is baked once at `build_agent_bundle(model, mission)` time — it carries the voice register, dossiers, mission framing, and rules (everything immutable within a mission). The user prompt carries current state, recent dispatches, this-cycle context. This shape is required for KV-cache hits on llama-server's prefix cache. Don't flatten the two prompts together.
- **Word-pacing is server-side.** `paced_word_emitter` in `main.py` throttles word emission to ≥80 ms between events at three call sites: SoP/dispatch (paced as the LLM streams) and drafter options (paced over already-completed prose). Frontend is a dumb renderer that appends words as they arrive; `<StreamingText>` only animates the visual fade-in. Don't pace on the client — it forfeits the server's flow-control over slow LLM moments and breaks the "one source of timing truth" property.
- **Drafter retries 3× then engine fallback.** `validate_options` rejects bad archetype tags, agent mismatches, missing forced moments, and wrong option counts. Each failure feeds the validator's complaints back as `retry_feedback` in the next prompt. After 4 total attempts, `engine_fallback_options` synthesises a deterministic 3-option set (HOLD + one per available agent) so the player never sees a broken cycle. The fallback fire is visible to evaluators via `engine_fallback=True`.
- **Cycles_remaining decrements at cycle close**, after effects and beats apply. The `timeout` beat fires on the second beat pass that runs after `tick_cycle_close`. Don't decrement at cycle open — passive-pressure beats need to read the pre-decrement cycle count.
- **Passive beats gate on `only_when_no_progress_this_cycle`.** `option_progresses_mission(delta, mission)` returns True iff any non-cover, non-cycle `mission.*` quality moved forward. Cover changes alone don't count. This is what makes HOLD costly — the action does nothing, no progress, so the world-attention passive beat fires.
- **`for_each_agent` beats fire one instance per matching agent.** `$agent` in the beat's `when:` and `effects` is substituted per agent at evaluation time. Pairs with `fires_at_most: once_per_agent` and a `fired_per_agent` ledger keyed by `f"{beat_id}:{agent_id}"`.
- **Terminal beats short-circuit the cycle's beat phase.** When a terminal fires, subsequent beats are skipped that cycle. The terminal's outcome_key sets `state.outcome`, the dispatch renders one final time, `MissionResolved` is emitted.
- **`MissionStarted` carries an independent `initial_state` instance, not a `model_copy` of session.state.** `start_mission` calls `make_initial_state(mission)` twice — once for `session.state`, once for the event — so fold() can re-seed during replay without aliasing the live qualities dict.
- **The Drafter is the only structured-output agent.** `output_type=NativeOutput(OptionSet)`. SoP and Renderer use `output_type=str` and stream via `agent.run_stream`. `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` is set in main.py's lifespan — required for Gemma 4 multi-turn correctness even though we don't multi-turn. Don't remove it.

## Adding a new `Predicate` variant — coordinated edits

1. Define the class in `models.py`, inheriting `_PredicateLeaf` (evaluated directly against state) or `_PredicateComposite` (must implement `children()`).
2. Add the class to the `Predicate` Union.
3. Extend `evaluate_predicate` in `spine.py` (ends in `assert_never(pred)` — fires loudly on a missed leaf).
4. Extend `substitute_predicate` in `models.py` if the variant carries a quality ref that should be substituted under `$agent` expansion.
5. Add a unit test in `tests/test_models.py` (round-trip) and `tests/test_spine.py` (evaluation).

The single composite walker `iter_predicate_leaves` covers analysis-side consumers automatically — new variants don't need extra plumbing.

## Running tests

- `make test` — fast unit suite. 49 tests as of Phase 6. No live llama needed.
- `cd handler && uv run pytest -m integration` — placeholder for live-llama integration tests. The current `conftest.py` exposes the `live_llama` fixture and the `consume_sse` SSE helper. Add `test_cycle_integration.py` here to walk a full mission with a deterministic option picker.

## Inferno gateway (dev-mode opt-in)

The shipped game spawns its own bundled `llama-server` subprocess on `:8080` loading Gemma 4 E4B GGUF — single-player offline, no external dependencies. **This path is sacred — don't change it.**

For dev sessions on the Blackwell box, the sidecar can instead point at the [inferno gateway](`C:\Users\Nat\Documents\dev\inferno`) which serves the same GGUF via llama-cpp-python (byte-identical chat template + kernel params; parity is established by construction in inferno's `LlamaCppModelConfig` defaults). The win is VRAM co-residency — Gemma shares the budget with FLUX / TRELLIS / Qwen-Image instead of competing through orphaned subprocesses.

Env vars (both must be set; either alone is ignored):

- `SMALLVILLE_USE_INFERNO=1` — skip the bundled `llama-server` subprocess; probe inferno's `/health` instead and route the pydantic-ai client at inferno's `/v1`. Unset (default): shipping behavior, bundled subprocess on `:8080`.
- `INFERNO_URL` — inferno's root URL (default `http://127.0.0.1:7770`).

When `SMALLVILLE_USE_INFERNO=1`, lifespan's `_probe_inferno_health` fails loud if inferno isn't reachable — there's no silent fallback. Rollback is `unset SMALLVILLE_USE_INFERNO` and restart.

Playtester / eval LLM: `eval_llama_base_url` (Settings field at `config.py:131`) now defaults to `f"{INFERNO_URL}/v1"` when `INFERNO_URL` is set, otherwise None (caller falls back to the legacy `EVAL_LLAMA_BASE_URL` constant pointing at `make llama-eval` on `:8081`). No consumer wires this today — programmatic evaluators only — but the plumbing is there for future LLM judges.

## Don't "clean up" without reading the comment first

Each invariant is anchored by an inline comment at the call site explaining why:

- `launcher.py` — `--jinja` always required (Gemma chat template). `--reasoning` / `--reasoning-format` are setting-driven (`LLAMA_REASONING_MODE`, default `off`). Perf flags `--flash-attn on`, `--batch-size 512 --ubatch-size 512` are non-negotiable. KV cache stays fp16 on the engine (`bench_kv_compare` showed q8_0 is ~2.5× slower for short outputs).
- `main.py` — `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)`. Without this, pydantic-ai's default 'auto' round-trips parsed `ThinkingPart`s back into prior assistant `content` as `<think>...</think>` blocks, violating Gemma 4's multi-turn rule.
- `agents.py` — the three `system_prompt`s are the stable-prefix half of the KV-friendly prompt structure. User prompts (`_sop_user_prompt`, `_drafter_user_prompt`, `_renderer_user_prompt`) carry only dynamic content. Don't merge them.
- `web/index.html` — Courier Prime via Google Fonts CDN. Single dark theme — no FOUC theme script.
