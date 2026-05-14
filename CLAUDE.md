# CLAUDE.md

HANDLER v2 — LLM-driven investigation game set in the Welsh Marches village of Pant Goch. The player is a journalist looking into a death the local police closed too quickly. Locations, cast, and the truth are hand-authored in `handler/world/`; the LLM improvises scenes inside that world via two-stage structured emission (`StateChangeBundle` then `ProsePart`). The map is the primary navigation surface.

**v2 lives at `handler/`.** Spec: `handler/SPEC.md`. Remaining work plan: `handler/PLAN.md`.

**A v2-specific CLAUDE.md is a Step D / M8 deliverable** — it will be authored at `handler/CLAUDE.md` once the full playtest pins down which engine extensions (per-beat floors, etc.) landed. Until then this file is the only standing engine-invariant guidance; treat it as load-bearing notes rather than a complete reference.

**v1 is gone.** The spy-handler game has been deleted (commit history preserves it). Don't reintroduce v1 concepts: forced Actions, the selector/prose-writer split, `state.fired` ledger, Mission YAML, etc. v2 is structurally different.

## v2 invariants (provisional — supersede with handler/CLAUDE.md when written)

- **Two-stage emission per scene turn.** Scene-runner is one pydantic-ai agent with one stable system prompt; per turn it makes two LLM calls: (1) `StateChangeBundle` (validated against `requires_beats` / `blocked_by_beats`; `ModelRetry` on failure, budget 2), then (2) `ProsePart` (free-form prose + options, conditioned on the just-applied state). Both calls share the KV cache because the system prompt is byte-identical.

- **No tool-call loop.** State changes are typed fields in `StateChange` (a discriminated union: `unlock_beat`, `move_npc`, ...). They are NOT pydantic-ai tool calls. If you find yourself wanting "a tool that...", either it's a new `StateChange` variant or it's out of scope.

- **Grammar-sampled enums over the catalog.** `unlock_beat` / `move_npc` use `Literal[*reachable_beat_ids]` / `Literal[*location_ids]` built per-turn. llama.cpp's grammar sampler rejects invalid ids at decode time — the model literally cannot emit a beat it doesn't have access to.

- **`unlock_beat` capped at ≤2 per turn.** Schema enforces; engine validates. Prevents cascade-unlocking the truth in one beat.

- **Two agents.** Scene runner (stateless, fresh `message_history` per turn) and log writer (conversation-based, one user+assistant pair appended per scene-end, system prompt carries the bible + journalist's voice rules). Log writer's accumulating history is the case file.

- **`OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)`** is set when building the model (see `handler.main.build_model`). Without it, pydantic-ai's default round-trips `ThinkingPart`s back into prior assistant content as `<think>...</think>` blocks, violating Gemma 4's multi-turn rule. The log writer's accumulating `message_history` makes this load-bearing.

- **Word-pacing is server-side, ≥80 ms cadence.** `handler.sse.paced_word_emitter` lifted verbatim from v1; do not slow or remove the floor. Frontend `<StreamingText>` only animates fade-in.

- **Leave-pick semantics.** Every `ContinueScene` emission must include ≥1 "Leave for…" option (schema-enforced). The map's clickable hotspots (M5, pending) replace the text walk panel for adjacent locations; non-adjacent locations are visually dim and the backend rejects walks to them defensively.

- **Player-driven endings.** Endings gate on decision beats authored as mutually-exclusive (via pairwise `blocked_by_beats`), not on engine-summoned climaxes. The engine cannot force an ending; the player commits to one by triggering a decision beat in fiction.

## Don't "clean up" without reading the call-site comment

These each carry an inline `# Why:` at the call site — preserve them:

- `handler/src/handler/launcher.py` — `--jinja` always required (Gemma chat template). `--flash-attn on`, `--batch-size 512 --ubatch-size 512` non-negotiable. KV cache stays fp16. Bundled `llama-server.exe` lives at `vendor/llama.cpp/` (resolved from `handler/../vendor/llama.cpp/`).
- `handler/src/handler/agents.py` — `build_scene_runner_system_prompt`, `build_log_writer_system_prompt` are the stable-prefix half of the KV-friendly structure. Per-turn `Literal` builders for `unlock_beat` / `move_npc` are dynamic. Don't merge system and user surfaces.
- `handler/src/handler/scene_loop.py` — `play_session(...)` is the single composable driver shared by CLI, SSE backend, and tests. Async/sync callbacks both supported via `_maybe_await`.
- `handler/web/index.html` — Courier Prime via Google Fonts CDN. Single dark theme with sepia/parchment accents — no FOUC theme script.

## Inferno gateway (dev-mode opt-in)

The shipped game spawns its own bundled `llama-server` on `:8080` loading Gemma 4 E4B GGUF — single-player offline, no external deps. **Don't change that path.**

For dev sessions on the Blackwell box, set BOTH env vars to point the sidecar at the [inferno gateway](`C:\Users\Nat\Documents\dev\inferno`) instead (byte-identical chat template + kernel params; the win is VRAM co-residency with FLUX / TRELLIS / Qwen-Image):

- `SMALLVILLE_USE_INFERNO=1` — skip bundled `llama-server`, probe inferno's `/health`, route pydantic-ai at inferno's `/v1`.
- `INFERNO_URL` — inferno's root URL (default `http://127.0.0.1:7770`).

When enabled, lifespan's inferno health probe fails loud if inferno isn't reachable — no silent fallback. Rollback: `unset SMALLVILLE_USE_INFERNO` and restart.

## Tests

- `make test` — fast unit suite, no live llama needed.
- `make test-integration` — includes the live-llama integration test (`tests/test_live_integration.py`).
