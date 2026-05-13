# CLAUDE.md

HANDLER — single-player spy game. One verb: **authorize one of 3–4 options per day**. Three LLM agents per session: a stateless one-shot selector (picks N catalog targets via a grammar-sampled `Literal` over the valid catalog), a stateless one-shot prose writer (writes `(title, prose)` for each selector slot), and a conversation-based Operations-voice log writer (one entry per day, post-resolution, appended to a growing case file). Mechanics are deterministic and authored in YAML; the LLM never decides outcomes.

**Stack / run / test:** `README.md`. **Visual ref:** `_mockups/handler-authorize-desk.html`.

## Authored types

The mission YAML has three authored types — keep them separate:

- **`Action`** = player-authorizable. Carries `when:`, `effects:` OR `outcomes:` (first-match-wins on sibling `when:`, mutually-exclusive at load), optional `for_each_agent: bool` (expands `$agent` over agents with cover > 0), optional `intent:`. A forced Action sets `forced: true` + prose constraints (`intent`/`must_communicate`/`must_not`/`tone`); when its `when:` holds and its id isn't yet in `state.fired`, it MUST appear as one of the cycle's options. The forced Action IS the option — no carrier coupling.
- **`Reaction`** = engine-applied consequence. Carries `when:`, `effects:`, `priority` (lower fires first), `fires_at_most`, optional `terminal: bool` + `outcome_key`, optional `only_when_no_progress_this_day`, optional `for_each_agent: bool`. Fires after the player's Action.
- **`Beat`** = pure prose framing fragment, fires-once-per-mission. No mechanical effects.

`Outcome.leans` and `Reaction.lean_on` thread renderer directives into the log writer's LEAN-ON block. `Action`'s opposed-check shape uses `outcomes:` with `leans` on each outcome; flat actions have no leans.

## Non-obvious invariants

These are things you wouldn't discover quickly by reading the code. Honour them.

- **Engine writes state; LLM writes prose.** All state writes happen through `apply_action_effects`, `fire_due_reactions`, `tick_day_close`. The LLM never decides whether a reaction fires or what an outcome resolves to. Dispatches that contradict the resolved delta are prompt bugs, not alternate outcomes.

- **`Action.outcomes` is first-match-wins.** Sibling outcomes' `when:` predicates are validated at mission load to be mutually exclusive (via `_predicates_may_overlap` on AND-of-comparisons). The matched outcome's `effects` AND `leans` fire — no parallel branches.

- **Forced Actions fire pre-effects via drafter selection, not post-effects via engine scan.** `due_forced_actions(state, mission)` enumerates them at cycle open; the selector includes one as a target; on authorization, `apply_action_effects` runs its effects and appends `action.id` to `state.fired`. There is no separate `apply_forced_event_effects` — forced and non-forced Actions use the same path.

- **Selector + prose-writer split is load-bearing.** The selector emits a per-day `Literal`-typed token under `strict=True`, which compiles to a llama.cpp grammar-sampler that rejects invalid catalog picks at decode time. The prose writer emits free-form title+prose. Gemma E4B can't satisfy both schema constraints in one structured emission — leave the split alone.

- **Log writer is conversation-based.** It accumulates one user+assistant pair per day; the conversation IS the case file. The selector and prose writer are stateless one-shots — continuity for them is carried by explicitly passing the last 2 log entries in user-side context.

- **Prompt structure is `[stable system prefix] + [dynamic user suffix]` for selector + prose writer.** System prompts are baked once at `build_agent_bundle(model, mission)`. Required for KV-cache hits. The log writer has a stable system prefix too, but its dynamic suffix is delivered as pydantic-ai `message_history`. Don't flatten any of these.

- **Word-pacing is server-side.** `paced_word_emitter` in `main.py` throttles to ≥80 ms between emissions. Frontend appends words as they arrive; `<StreamingText>` only animates fade-in.

- **`days_remaining` decrements at day close**, after action + reaction passes apply. The `timeout` terminal Reaction fires on the second reaction pass after `tick_day_close`. Don't decrement at day open — passive-pressure Reactions need to read the pre-decrement count.

- **`state.fired` is the single fired-id ledger** across Beats, Actions (forced only), and Reactions. The Mission validator enforces id-uniqueness across all three. `EventFired(event_id)` predicate is a membership check. `state.fired_per_agent` holds per-(reaction, agent) keys via `per_agent_key(reaction_id, agent_id)`. Non-forced Actions don't write to `state.fired` — they're freely re-authorizable.

- **One-day flags use the explicit-reset pattern.** A quality like `agent[LYNX].cover_protected` is set by an Action (e.g. DIVERSION_BY_AGENT) and cleared by a dedicated high-priority Reaction (`clear_cover_protected`) that fires after the consumers (priority > consumer's priority). The engine has no `one_cycle_flags` channel — authoring discipline owns the lifecycle.

- **`MissionStarted` carries a fresh `make_initial_state` snapshot, not a `model_copy` of `session.state`.** Otherwise fold() during replay aliases the live qualities dict.

- **`OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)`** is set in `main.py`'s lifespan. Without it, pydantic-ai's default 'auto' round-trips `ThinkingPart`s back into prior assistant `content` as `<think>...</think>` blocks, violating Gemma 4's multi-turn rule. The log writer's accumulating `message_history` makes this load-bearing — without the profile, drift compounds every cycle.

## Don't "clean up" without reading the call-site comment

Each of these has an inline `# Why:` at the call site:

- `launcher.py` — `--jinja` always required (Gemma chat template). `--flash-attn on`, `--batch-size 512 --ubatch-size 512` are non-negotiable. KV cache stays fp16 (q8_0 is ~2.5× slower for short outputs per `bench_kv_compare`).
- `agents.py` — `_selector_system_prompt`, `_prose_writer_system_prompt`, `_log_writer_system_prompt` are the stable-prefix half of the KV-friendly structure. The selector pairs with `_selector_user_prompt` + a per-day `Literal`-typed schema built by `_make_selector_output_type`. The prose writer pairs with `_prose_writer_user_prompt` + a fixed-length `OptionProse[]` schema. The log writer pairs with `_log_writer_user_prompt` (day inputs as the user side of one turn in an accumulating `message_history`). Don't merge any system and user surfaces.
- `web/index.html` — Courier Prime via Google Fonts CDN. Single dark theme — no FOUC theme script.

## Inferno gateway (dev-mode opt-in)

The shipped game spawns its own bundled `llama-server` on `:8080` loading Gemma 4 E4B GGUF — single-player offline, no external deps. **Don't change that path.**

For dev sessions on the Blackwell box, set BOTH env vars to point the sidecar at the [inferno gateway](`C:\Users\Nat\Documents\dev\inferno`) instead (byte-identical chat template + kernel params; the win is VRAM co-residency with FLUX / TRELLIS / Qwen-Image):

- `SMALLVILLE_USE_INFERNO=1` — skip bundled `llama-server`, probe inferno's `/health`, route pydantic-ai at inferno's `/v1`.
- `INFERNO_URL` — inferno's root URL (default `http://127.0.0.1:7770`).

When enabled, lifespan's `_probe_inferno_health` fails loud if inferno isn't reachable — no silent fallback. Rollback: `unset SMALLVILLE_USE_INFERNO` and restart.

## Tests

- `make test` — fast unit suite, no live llama needed.
- `cd handler && uv run pytest -m integration` — live-llama integration tests. `test_day_integration.py` uses stubbed agents end-to-end.
