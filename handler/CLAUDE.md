# HANDLER v2 — engine invariants

This file supersedes the provisional notes in `../CLAUDE.md`. v2 is an LLM-driven investigation game set in the Welsh Marches village of Pant Goch. The player is a freelance journalist looking into a death the police closed too quickly. The world (locations, cast, beats, endings) is hand-authored in `world/`; the LLM improvises scenes inside it via structured output. The map (`web/public/`-served from `world/map.png`) is the primary navigation surface.

Spec: `SPEC.md`. Remaining work: `PLAN.md`. Read both before changing engine-shaped code.

## The two-stage emission split

Per scene turn, the engine makes **two LLM calls with separate system prompts**:

**Phase 1 — `StateChangeBundle` (classifier).** System prompt: `build_phase1_system_prompt` in `prompts.py`. Carries the FULL bible — locations, cast with `knowledge`, the truth (crime + beats with `requires_beats`/`blocked_by_beats`/`notes_to_llm`/`prerequisites`, endings). Stateless per turn (no `message_history`). Output validated; `ModelRetry` on `requires_beats`/`blocked_by_beats`/≤2-cap violations.

**Phase 2 — `ProsePart` (narrator).** System prompt: `build_phase2_system_prompt`. SLIM — locations + cast-with-voice-samples + crime overview + few-shot style examples. Explicitly **drops** beat catalog, `notes_to_llm`, endings, and per-NPC `knowledge`. Phase 2 is also **scene-bounded conversation**: `SceneRunnerRuntime.phase2_message_history` accumulates within a scene (the model sees its own prior turns as assistant messages); `play_scene` clears it at scene start. On `UnexpectedModelBehavior` (retries exhausted) the runtime is NOT mutated and engine-authored fallback prose is surfaced to the player but never persisted — voice continuity stays clean.

**Do not merge the prompts back together.** The split was deliberate: Phase 2 attention is finite on a 4B model, and beat structure in Phase 2's context produced beat-aware prose (telegraphed reveals, NPCs serving structure). The trade-off cost was the inter-phase KV cache hit per turn; the gain was Phase 2 focus on writing and intra-scene voice continuity. If a future change tempts you to consolidate "for cache efficiency," read this paragraph again and the prose-voice refactor entry in `PLAN.md`. Each phase's system prompt is byte-stable across the session → inter-turn cache hits remain on both. Phase 2 additionally hits the accumulating message_history prefix within a scene.

## Three agents (and their runtimes)

| Agent | System prompt | History scope | Failure policy |
|---|---|---|---|
| Phase 1 classifier | `build_phase1_system_prompt` (full bible) | none (stateless per turn) | retries exhausted → drop bundle (empty `StateChangeBundle`) |
| Phase 2 narrator | `build_phase2_system_prompt` (slim, voice) | `SceneRunnerRuntime.phase2_message_history`, scene-bounded | retries exhausted → engine `_fallback_prose_part`, NOT persisted to history |
| Log writer | `build_log_writer_system_prompt` (full bible + journalist's voice rules) | `LogWriterRuntime.message_history`, session-bounded | no fallback authored; failures bubble |

`AgentBundle` (world-scoped, immutable) holds `phase1` + N `phase2_by_location` agents — one Phase 2 agent per location so `Literal[*adjacent_ids]` for `Option.leaves_for` is baked into the per-location output schema. All N+1 agents are constructed at lifespan boot in `build_scene_runner_bundle`. `SceneRunnerRuntime` and `LogWriterRuntime` are session-scoped, mutable, threaded through `play_session(bundle, scene_runner, log_writer, state, world, ...)`.

## No tool-call loop

State changes are typed fields in `StateChange` (a discriminated union: currently just `UnlockBeat`). They are NOT pydantic-ai tool calls. There is no mid-turn tool-result feedback. If you want "a tool that…", either it's a new `StateChange` variant or it's out of scope.

## Grammar-sampled enums + ≤2 cap

Per-turn dynamic schema builders narrow `UnlockBeat.beat_id` to `Literal[*beat_ids]` and `Option.leaves_for` to `Literal[*adjacent_ids] | None`. llama.cpp's grammar sampler rejects invalid ids at decode time — the model literally cannot emit an undefined id. Application-layer rules (`requires_beats`, `blocked_by_beats`, ≤2 `UnlockBeat` per bundle) are checked in the `@agent.output_validator` and raise `ModelRetry` on failure. `output_retries=3`.

`UNLOCK_BEAT_CAP_PER_BUNDLE = 2` in `engine.py` is non-negotiable — prevents cascade-unlocking the truth in one beat. Naturally-paired beats are still allowed.

## Leave-pick semantics

Every `ContinueScene` emission must include ≥1 option with `leaves_for` set (engine-validated, `ProsePartError` → `ModelRetry`). When the player picks that option, the next turn's Phase 2 is invoked with `leave_pick=<that option>` and MUST emit `EndScene` (validated). Phase 1 of that turn may still fire a beat — the act of leaving CAN be the moment. The map's clickable hotspots are an alternative entry point to walking, but only fire between scenes; during an active scene the map is dim and the player must use a Leave-for option in the text. The backend defensively rejects non-adjacent walks (`POST /walk`).

## Player-driven endings

Endings gate on **decision beats** authored as mutually exclusive (pairwise `blocked_by_beats`) — see `truth.yaml` decision beats `decided_to_file` / `decided_to_walk_away` / `decided_to_confront_eluned` / `decided_to_tell_iwan_privately`. The engine cannot force an ending; the player commits to one by triggering a decision beat in fiction, after which Phase 2 may emit `EndGame(ending_id=...)`. The output validator rejects `EndGame` unless the ending's `requires_beats` are satisfied.

## Authored data layout

```
world/
├── brief.yaml              # player_starts_at + opening_seed
├── cast.yaml               # 9 NPCs: id, name, default_location, themes,
│                           #  knowledge, attitude, voice_sample
├── truth.yaml              # crime + 18 beats + 6 endings
├── few_shots.yaml          # OPTIONAL: 1-2 short example exchanges
│                           #  (setup/player/narrator) for Phase 2 voice
├── map.yaml                # OPTIONAL: { location_id: { x, y } }
│                           #  fractional (0-1) hotspot coords for the map
├── map.png                 # OPTIONAL: the drawn village map background
│                           #  (1408×768 ish; aspect 1.83:1 in the current
│                           #  asset; CSS uses aspect-ratio so any matching
│                           #  ratio works)
└── locations/              # one YAML per location, 10 total
```

- **`NPC.voice_sample: list[str]`** — 1-2 representative dialogue lines per NPC. Phase 2 only; Phase 1 never sees them. Author from the NPC's existing `knowledge`/`attitude` prose — don't invent new facts.
- **`few_shots.yaml`** — surfaces in Phase 2 system prompt as `[STYLE EXAMPLES]`. Few-shot examples beat instructions for small models; this is the single biggest prose-quality lever. Keep examples short (a setup line, a player option, 60-180 words of narrator response).
- **`map.yaml` + `map.png`** — `world.py::load_world` reads them as optional; missing/empty → frontend falls back to text walk panel (`WalkPanel` in `App.tsx`). Coords are fractional so the map is responsive without re-authoring on resize.
- **Anti-slop banned-phrase list** — NOT yet authored. Mine outputs from a playthrough before adding.

## OpenAIModelProfile flag is load-bearing

`build_model` in `main.py` sets `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)`. Without it, pydantic-ai's default `'auto'` round-trips `ThinkingPart`s back into prior assistant content as `<think>…</think>` blocks, which Gemma 4 cannot tolerate in multi-turn. Both Phase 2's accumulating scene history AND the log writer's accumulating session history depend on this; drift compounds every turn without it.

## Word-pacing is server-side, ≥80 ms cadence

`handler.sse.paced_word_emitter` enforces an 80 ms floor on prose word emission. The frontend `<StreamingText>` only animates fade-in — never paces. Don't slow or remove the floor; don't move pacing client-side.

## Launcher invariants (lift from v1)

`handler/src/handler/launcher.py` — `--jinja` always required (Gemma chat template). `--flash-attn on`, `--batch-size 512 --ubatch-size 512` non-negotiable. KV cache stays fp16 (q8_0 KV measured ~2.5× slower on short outputs). `--no-kv-unified` (v1 measured `--kv-unified` ~2.7× slower at 64-concurrent on this stack; v2 is single-session so the gap is even wider). Bundled `llama-server.exe` lives at `vendor/llama.cpp/`.

## Inferno gateway (dev-mode opt-in)

The shipped game spawns its own bundled `llama-server` on `:8080` loading Gemma 4 E4B GGUF — single-player offline, no external deps. **Don't change that path.**

For dev sessions on the Blackwell box, set BOTH env vars to point the sidecar at inferno (byte-identical chat template + kernel params; the win is VRAM co-residency with FLUX / TRELLIS / Qwen-Image):

- `SMALLVILLE_USE_INFERNO=1` — skip bundled `llama-server`, probe inferno's `/health`, route pydantic-ai at inferno's `/v1`.
- `INFERNO_URL` — inferno's root URL (default `http://127.0.0.1:7770`).

When enabled, lifespan's inferno health probe fails loud if inferno isn't reachable — no silent fallback. Rollback: `unset SMALLVILLE_USE_INFERNO` and restart.

## Don't "clean up" without reading the call-site comment

These each carry an inline `# Why:` at the call site — preserve them:

- `prompts.py::build_phase1_system_prompt` / `build_phase2_system_prompt` — the split, with `[THE TRUTH]` deliberately absent from Phase 2.
- `agents.py::SceneRunnerRuntime` — scene-bounded history; reset in `play_scene` after `enter_scene`; fallback prose intentionally NOT persisted.
- `agents.py::emit_prose` — `is_first_turn = not runtime.phase2_message_history` computes from runtime, not state. The runtime is the source of truth for scene-conversation lifecycle.
- `scene_loop.py::play_session` — single composable driver shared by CLI, SSE backend, and tests. Async/sync callbacks both supported via `_maybe_await`. Threads `bundle, scene_runner, log_writer, state, world` in that order.
- `main.py::build_model` — `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)`. See above.
- `world/few_shots.yaml`, `world/map.yaml` — optional files; loader tolerates missing. Don't make them required without thinking through the test fixtures (`tests/fixtures/world_valid/` does NOT author them).

## Tests

- `make test` — fast unit suite, no live llama needed.
- `make test-integration` — includes the live-llama integration test (`tests/test_live_integration.py`). Hits Gemma 4 E4B via inferno (or the bundled server). Run before shipping engine-shaped changes.

## What v2 is NOT

- v1 (the spy-handler / Continuity-Program game) has been deleted. Don't reintroduce v1 concepts: forced Actions, the selector/prose-writer split, `state.fired` ledger, Mission YAML, the static-analysis pipeline, action counters as gating, the encoder/engine drift suite. The shape of v2 is structurally different — a world bible + LLM improvisation, not a scripted-storylet engine.
- v2 has no flags, no qualities, no effects, no predicates, no DSL. Beats are unlock units identified by id. If you find a soft-prereq that the LLM can't judge from `prerequisites` prose alone, the cheapest escalation rungs are `min_co_presence` / `min_scene_turns` on Beat (see `PLAN.md` Step B); the DSL is a last resort and probably means swapping models first.
