# CLAUDE.md

HANDLER v2 — LLM-driven investigation game set in the Welsh Marches village of Pant Goch. The player is a journalist looking into a death the local police closed too quickly. Locations, cast, and the truth are hand-authored in `handler/world/`; the LLM improvises scenes inside that world via two-stage structured emission (`StateChangeBundle` then `ProsePart`). The map is the primary navigation surface.

- **Grammar-sampled enums over the catalog.** `unlock_beat` / `move_npc` use `Literal[*reachable_beat_ids]` / `Literal[*location_ids]` built per-turn. llama.cpp's grammar sampler rejects invalid ids at decode time — the model literally cannot emit a beat it doesn't have access to.

- **Three runtimes.** `AgentBundle` (world-scoped, immutable: phase1 + N phase2 agents). `SceneRunnerRuntime` (session-scoped, holds Phase 2's `phase2_message_history`; cleared per scene). `LogWriterRuntime` (session-scoped, accumulating one user+assistant pair per scene-end — the conversation is the case file). All three threaded through `play_session(bundle, scene_runner, log_writer, state, world, ...)`.

- **NPC voice samples + few-shot examples are authored data, not engine constants.** `NPC.voice_sample: list[str]` lives in `cast.yaml` per NPC (1-2 representative lines). `world/few_shots.yaml` (optional file) holds 1-2 example exchanges. Both surface ONLY in the Phase 2 system prompt; Phase 1 never sees them.

- **Player-driven endings.** Endings gate on decision beats authored as mutually-exclusive (via pairwise `blocked_by_beats`), not on engine-summoned climaxes. The engine cannot force an ending; the player commits to one by triggering a decision beat in fiction.

`SceneRunnerRuntime.phase2_message_history` is the scene-bounded conversation; resetting it lives in `play_scene` (after `enter_scene`). Don't merge phase 1 and phase 2 prompts — the split is the whole point.
- `handler/src/handler/scene_loop.py` — `play_session(...)` is the single composable driver shared by CLI, SSE backend, and tests. Async/sync callbacks both supported via `_maybe_await`.

- `make test` — fast unit suite, no live llama needed.
- `make test-integration` — includes the live-llama integration test (`tests/test_live_integration.py`).
- `make kill` — force kill old/orphaned proccesses
- `make dev` — runs the game, starts the sidecar (inferno) and web server together