# CLAUDE.md

HANDLER — single-player spy game. One verb: **authorize one of 3–4 options per cycle**. Three stateless LLM agents per cycle (SoP memo, drafter, dispatch renderer). Mechanics are deterministic and authored in YAML; the LLM never decides outcomes.

**Design doc:** `HANDLER_SPEC_V2.md`. **Stack / run / test:** `README.md`. **Visual ref:** `_mockups/handler-authorize-desk.html`.

## Non-obvious invariants

These are things you wouldn't discover quickly by reading the code. Honour them.

- **Engine writes state; LLM writes prose.** All state writes happen through `apply_archetype_effects`, `apply_forced_event_effects`, `fire_due_passive_events`, `tick_cycle_close`. The LLM never decides whether an event fires or what an effect resolves to. Dispatches that contradict the resolved delta are prompt bugs, not alternate outcomes.

- **One `Event` model, shape-selected by flags.** `force_into_drafter=True` = drafter-surfaced decision point with prose constraints (intent/must_communicate/must_not/tone). Otherwise passive: `terminal`, `only_when_no_progress_this_cycle`, or a plain reactive predicate. `for_each_agent=True` is passive-only and expands `$agent`. `Event._consistency` enforces the mutual exclusions — read it once.

- **Agents are stateless one-shot — no `message_history` between cycles.** State mutates between calls; multi-turn would force the model to defend prior assistant turns against state that has since changed. Continuity is carried by explicitly passing the last 1–2 dispatches in user-side context (same-agent preference for the renderer per spec §8.3). Don't reintroduce conversation history threading.

- **Prompt structure is `[stable system prefix] + [dynamic user suffix]`.** The system prompt is baked once at `build_agent_bundle(model, mission)` — voice, dossiers, mission framing, rules. The user prompt carries this-cycle context. Required for KV-cache hits on llama-server's prefix cache. Don't flatten the two together.

- **Word-pacing is server-side.** `paced_word_emitter` in `main.py` throttles to ≥80 ms between emissions. Frontend appends words as they arrive; `<StreamingText>` only animates fade-in. Pacing on the client forfeits server flow-control over slow LLM moments and breaks "one source of timing truth."

- **Cycles_remaining decrements at cycle close**, after archetype + forced + passive events apply. The `timeout` terminal fires on the second passive-event pass after `tick_cycle_close`. Don't decrement at cycle open — passive-pressure events need to read the pre-decrement count.

- **`state.fired` is the single fired-event ledger.** All event fires append `event.id` (duplicates allowed for `fires_at_most="unlimited"`). The `EventFired(event_id)` predicate is a membership check. `state.fired_per_agent` holds per-(event, agent) keys via `per_agent_key(event_id, agent_id)`. No split `fired_beats` / `fired_moments`.

- **`MissionStarted` carries a fresh `make_initial_state` snapshot, not a `model_copy` of `session.state`.** Otherwise fold() during replay aliases the live qualities dict.

- **`OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)`** is set in `main.py`'s lifespan. Without it, pydantic-ai's default 'auto' round-trips `ThinkingPart`s back into prior assistant `content` as `<think>...</think>` blocks, violating Gemma 4's multi-turn rule.

## Don't "clean up" without reading the call-site comment

Each of these has an inline `# Why:` at the call site:

- `launcher.py` — `--jinja` always required (Gemma chat template). `--flash-attn on`, `--batch-size 512 --ubatch-size 512` are non-negotiable. KV cache stays fp16 (q8_0 is ~2.5× slower for short outputs per `bench_kv_compare`).
- `agents.py` — the three `system_prompt`s are the stable-prefix half of the KV-friendly structure. User prompts (`_sop_user_prompt`, `_drafter_user_prompt`, `_renderer_user_prompt`) carry only dynamic content. Don't merge them.
- `web/index.html` — Courier Prime via Google Fonts CDN. Single dark theme — no FOUC theme script.

## Inferno gateway (dev-mode opt-in)

The shipped game spawns its own bundled `llama-server` on `:8080` loading Gemma 4 E4B GGUF — single-player offline, no external deps. **Don't change that path.**

For dev sessions on the Blackwell box, set BOTH env vars to point the sidecar at the [inferno gateway](`C:\Users\Nat\Documents\dev\inferno`) instead (byte-identical chat template + kernel params; the win is VRAM co-residency with FLUX / TRELLIS / Qwen-Image):

- `SMALLVILLE_USE_INFERNO=1` — skip bundled `llama-server`, probe inferno's `/health`, route pydantic-ai at inferno's `/v1`.
- `INFERNO_URL` — inferno's root URL (default `http://127.0.0.1:7770`).

When enabled, lifespan's `_probe_inferno_health` fails loud if inferno isn't reachable — no silent fallback. Rollback: `unset SMALLVILLE_USE_INFERNO` and restart.

## Tests

- `make test` — fast unit suite, no live llama needed.
- `cd handler && uv run pytest -m integration` — live-llama integration tests. `test_cycle_integration.py` uses stubbed agents end-to-end.
