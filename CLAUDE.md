# CLAUDE.md

Inbox — single-surface email correspondence game. The player is the protagonist; an AI counterpart (the only NPC) replies on every send. Authored "plot beats" (calibration milestones, terminal endings) inject one-shot directives onto the chat agent's next turn; the rest is free chat in character. How warmly the player writes shifts the NPC's voice register and accelerates the calibration meter. Three-process tree:

```
Browser (Vite + React + TS) ──HTTP/SSE──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
                            ◄──────────                                     ◄──OpenAI /v1──
```

`make run` boots sidecar + web in parallel (sidecar spawns llama). `make sidecar` / `make web` / `make llama` run layers separately. Tests: `cd npc_service && uv run pytest` (51 tests).

## Game premise

Current scenario is **"Continuity Program"**: a corporate-satire pilot in which the player has been "enrolled" with an AI counterpart, **Aria**, who is calibrating on the player's communication style "to assume agreed-upon portions of your role." Boots into a single thread with a single welcome email. Severance-flat HR-bright tone — the horror is structural, never named.

The mechanic the engine is built to test: **"Reply to Aria. Your warmth shifts her voice register and paces the calibration meter. Authored plot moments arrive as the meter advances."**

There is no clock. The single progression axis is the `calibration` quality (0–100) — the player-visible doom counter, diegetically "counterpart fidelity %." Calibration advances per /send by `base + coef·|warmth|` — strong signal calibrates fast, lukewarm sends barely move it. /open does NOT mutate calibration. Reaching 100 ends the session.

## Architecture in one diagram

```
                        ┌───────────────────────────┐
   player /send body ──►│ texture_agent (ToolOutput)│── warmth ──► register.<npc>, warmth.<npc>
                        └───────────────────────────┘                                      │
                                       │                                                   │
                                       └──── warmth ──► apply_calibration_from_warmth      │
                                                                                           │
                                                                                           ▼
                                                                              find_eligible_plot_beat
                                                                                /                  \
                                                                          PlotBeat               (none)
                                                                          eligible                  │
                                                                            │                       ▼
                                                                            ▼              chat_agent.run(body, msg_history, deps=ChatDeps(npc, band))
                                                                schedule_thread_reply              │ — fires immediately
                                                                (queues with                       │
                                                                 quality_gte threshold;            ▼
                                                                 lands on a future tick     thread_messages.extend(new_messages)
                                                                 once threshold crossed)         emit Email
                                                                            │
                                                                            ▼
                                                                  next /send ticks past
                                                                  threshold → chat_agent.run
                                                                  with plot_directive=spec
                                                                  injected via @instructions
```

## File map

```
npc_service/                  uv project root — pyproject.toml, scenario.yaml, .logfire/
  src/npc_service/
    main.py        FastAPI app: lifespan, agent wiring, all endpoints
                   (/scenario, /state, /inbox, /thread, /open, /send,
                   /reset, /health). Owns SSE streaming for /send and the
                   chat-turn-vs-PlotBeat dispatch. Caches the Settings
                   instance at lifespan boot.
    models.py      Pydantic schemas. Authoring (QualityDef, NPCDossier with
                   email_address + voice_bands + reply_delay_calibration,
                   AnchoredBeat, PlotBeat with BeatSpec, Predicate tagged
                   union: QualityGte/Lt, PlayerRepliesToThreadGte,
                   BeatFired, Not_, Scenario); runtime (Email, Reply,
                   PendingBeat, InboxState with thread_messages dict of
                   list[ModelMessage]); HTTP DTOs.
    spine.py       Deterministic spine. Predicate evaluation, beat scheduler
                   (find_due_beats with tick-local extra_fired for mutual
                   exclusion), find_eligible_plot_beat (next un-fired
                   PlotBeat in a thread whose authored preconditions pass),
                   schedule_thread_reply (queues a resolved PlotBeat with
                   QualityGte calibration threshold), apply_calibration_delta,
                   apply_calibration_from_warmth (base + coef·|warmth|),
                   apply_texture_score, voice-band selection,
                   reply_subject_for_email, latest_npc_email_in_thread.
                   No LLM calls — pure code-owned state mutation.
    agents.py      Two LLM agents.
                   chat_agent: output_type=str, deps_type=ChatDeps; per-run
                   @instructions decorators inject dossier, voice band, and
                   (when set) plot directive. Used for both chat turns
                   (plot_directive=None) and PlotBeat fires.
                   texture_agent: ToolOutput, scores warmth on player reply.
                   _BASE_CHAT_INSTRUCTIONS enforces body-only output —
                   no Subject:/From: headers, no JSON, no markdown fences,
                   no signature unless the dossier specifies one.
    launcher.py    llama-server subprocess + Windows Job Object for cleanup.
    config.py      pydantic-settings — calibration_base, calibration_warmth_coef.
  scenario.yaml    Inbox content layer. Qualities (calibration + warmth.aria
                   + register.aria), one NPC dossier (Aria with
                   email_address + three voice bands), threads (aria +
                   system), beats (6: 1 anchored welcome, 3 PlotBeats
                   for Aria's authored milestones, 2 anchored
                   calibration-gated system notices).
  tests/
    test_spine.py  37 tests — predicate eval, calibration delta + cap,
                   apply_calibration_from_warmth (parametrized),
                   texture/register loop with clamping, beat scheduler with
                   runtime_preconditions, find_eligible_plot_beat (returns
                   first eligible, skips pending/fired), terminal mutual
                   exclusion via Not_(BeatFired), schedule_thread_reply
                   (threshold delay), reply_subject_for_email,
                   latest_npc_email_in_thread.
    test_app.py    14 FastAPI tests — lifespan boot, /open does NOT mutate
                   calibration, /send delta = base + coef·|warmth|,
                   first /send returns immediate chat-turn fallback,
                   thread_messages accumulates ModelRequest/Response across
                   sends, body-only regression guard (no Subject:/From:
                   leakage), midprogram PlotBeat lands after threshold-
                   crossing /send, anchored system beat fires via
                   calibration cross.

web/                          Vite + React + TypeScript at workspace root
  src/
    main.tsx, App.tsx
    api.ts                    fetch + native SSE wrapper
    store.ts                  single Zustand store (server is source of truth;
                              store is render cache, refresh after mutations)
    hooks/useStreamSend.ts    shared SSE-drive helper used by EmailComposer;
                              refreshes inbox + reopens thread on done
    util/email.ts             stripAddress
    fonts.css, index.css      IBM Plex Mono via @font-face; Tailwind base
    components/
      InboxHeader.tsx         title + "Counterpart fidelity: N%" pill +
                              optional "transition complete" badge
      FolderTabs.tsx          Inbox / Sent (no Archive)
      InboxList.tsx           folder content
      ThreadView.tsx          email chain + composer (composer's inReplyTo
                              is the email's id, not its beat_id, so
                              chat-turn replies are also replyable)
      EmailComposer.tsx       free-form reply; Send disabled when empty
  public/fonts/               IBMPlexMono Regular + Bold
```

## Game loop

```
Boot:  scenario.yaml → InboxState.from_scenario (auto-pends every
       anchored beat; PlotBeats stay out until schedule_thread_reply
       queues them) → find_due_beats fires aria.welcome at tick 0
       (no preconditions). Welcome's body is appended to
       thread_messages["aria"] as ModelResponse(TextPart) — the chat
       agent's first run sees the welcome as Aria's prior turn.

Player /open → mark email read. No calibration mutation. (Read-only
       w.r.t. state mutations; still acquires ACTION_LOCK to serialise
       with /send.)

Player /send → record player Email (Sent folder) → texture_agent scores
       warmth → apply_texture_score (warmth.<npc>, register.<npc>) →
       apply_calibration_from_warmth (base + coef·|warmth|) → find_due_beats
       fires AnchoredBeats whose preconditions just passed AND PlotBeats
       whose calibration threshold was just crossed (those PlotBeats run
       chat_agent with plot_directive=spec). If no PlotBeat replied for
       the player's NPC thread → fire chat-turn fallback immediately
       (chat_agent.run with no plot directive, plain str output, body
       appended to thread_messages alongside the agent's response).
       Then find_eligible_plot_beat for the next PlotBeat to schedule
       (queue with QualityGte threshold = current_cal +
       reply_delay_calibration; lands on a future /send tick).

Asynchrony for plot beats falls out of authoring, not waiting. The
PlotBeat's threshold sits above the post-send calibration value, so
the just-queued PlotBeat doesn't fire on its own scheduling tick.
A subsequent /send (especially a warmth-charged one — delta scales with
|warmth|) ticks calibration past the threshold and the PlotBeat lands.
Chat turns have no threshold and fire immediately within the same /send
SSE — async lives in LLM render time + streaming, not in calibration math.

End:   when calibration reaches its cap (100), state.ended = True with
       reason "calibration_complete". Terminal PlotBeats are authored to
       fire at calibration ≥ 80 + reply count gates, so they normally
       land before the cap.
```

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel. Don't switch to `--reasoning-budget 0` — different PEG parse error per llama.cpp PR #21697.
- **`output_type=str` on `chat_agent`, NOT `ToolOutput`.** With `ToolOutput`, the model's reply lands in `message_history` as a `ToolCallPart`, not a `TextPart` — subsequent turns re-running with that history see a tool-call sequence instead of natural assistant turns. Plain text with `max_tokens=800` keeps generation bounded without the JSON-schema-driven cascade that previously produced 19K-token Gemma outputs (that was a NativeOutput problem, different code path from plain str). The texture agent keeps `ToolOutput` because it's one-shot and never feeds into a conversation.
- **`_BASE_CHAT_INSTRUCTIONS` explicitly forbids `Subject:` / `From:` / `To:` / `Date:` headers, JSON, markdown code fences, and unsanctioned signatures.** With `output_type=str` there's no schema enforcing structure, so the rule lives in the prompt. There's a pytest regression guard that asserts the rendered body contains none of these tokens.
- **Chat agent uses `@instructions` (per-run dynamic), not user-prompt formatting.** Pydantic AI's `@instructions` are appended to the system prompt every run, even when `message_history` is non-empty. This lets the per-turn dossier/voice-band/plot-directive layer sit in instructions while the conversation history (player+assistant turns) flows naturally through `message_history`. Mixing these paths (e.g. embedding the dossier in the user prompt) would pollute history.
- **The chat agent never writes state.** The agent returns prose only. Only TEXTURE_AGENT writes state, and only to `warmth.<npc>` and `register.<npc>`. The spine itself is the only thing that writes `calibration` (via `apply_calibration_from_warmth → apply_calibration_delta`). Eliminates the LLM-as-state-actor risk class.
- **`register.<npc>` drifts halfway toward each reply's warmth on every send.** This is the per-reply visible feedback loop — the NPC's *next* email comes back in a different voice band. Without this, the texture mechanic is invisible.
- **Calibration is `base + coef·|warmth|`, NOT a flat per-action delta.** Strong signal in either direction calibrates fast (the Program calibrates fastest on unmistakable voice). Lukewarm sends barely move the meter. /open does not mutate calibration at all — reading isn't a deliberate communication act, and removing it keeps the meter exclusively driven by the texture-scored player voice.
- **`schedule_thread_reply` queues PlotBeats with a QualityGte calibration threshold above the current value.** This is how plot moments *feel asynchronous* without a wait verb. The `reply_delay_calibration` scalar on the dossier MUST exceed the typical send delta (otherwise a single send satisfies its own queued reply and the asynchrony collapses). Chat turns do NOT use this delay — they fire inside the same /send.
- **`find_eligible_plot_beat` walks `scenario.beats` in authoring order and skips beats that are already pending or fired.** Authoring order is the tiebreaker for mutual exclusion (e.g. `terminal_warm` is authored before `terminal_standard`). When `terminal_warm` fires, `terminal_standard`'s `Not_(BeatFired terminal_warm)` precondition keeps it suppressed on subsequent ticks.
- **`/open` no longer mutates calibration but still acquires `ACTION_LOCK`.** Keeps it serialised with `/send` so the `email.is_read` flip and any concurrent state inspection by `/send` stay coherent.
- **`InboxState.from_scenario` auto-pends every anchored beat.** Their authored preconditions become runtime predicates evaluated by `find_due_beats`; an anchored beat with empty preconditions fires on the first scheduler tick at boot. PlotBeats are NOT auto-pended — those are added only by `schedule_thread_reply` on player send. Don't merge these two paths; the PlotBeat chain depends on player verbs to advance.
- **Anchored beats on NPC threads seed `thread_messages` with a `ModelResponse(TextPart)` when they fire** (`_seed_thread_history` in `main.py`). Without this, the chat agent's first run after boot would have no context — Aria would be replying to the player's first send without "remembering" her welcome email. System-thread anchored beats skip seeding (no chat agent on the system thread).
- **The beat-id prefix MUST equal the beat's `thread` field.** Validated at scenario load (`Scenario._cross_references`). The spine treats the prefix as the source of truth via `thread_id_from_beat_id`; silent mismatch would break reply threading and reply counting.
- **Email metadata for chat turns is *derived*, not authored.** `sender` = `NPCDossier.email_address`; `recipient` = `latest_npc_email_in_thread(...).recipient` (the player's address, read off the most recent NPC-originated email's `to:` field — the boot welcome guarantees this is populated); `subject` = `reply_subject_for_email(state, in_reply_to_email_id)`. PlotBeat fires override `subject` with the authored value but inherit the rest.
- **`SendRequest.in_reply_to` is an email id, not a beat id.** Frontend `ThreadView` passes `composeTarget.id` (not `composeTarget.beat_id`) so chat-turn replies (which have `beat_id=None`) can be replied to. `_reply_subject` looks up by email id. The on-Email field is named `in_reply_to_beat` for backward compat with the frontend type, but its semantic is "email id being replied to."
- **`apply_calibration_delta` mutates `state.qualities` directly** and uses `_clamp_quality` (shared with `apply_texture_score`) for bounds. The terminal-end-condition (set `state.ended = True` when the cap is reached) is the only calibration-specific behavior. `apply_calibration_from_warmth` is a thin wrapper that computes the delta and delegates — single clamping path.
- **Action-Lock around mutating endpoints** (`ACTION_LOCK` global mutex). Serializes `/open` and `/send` so spine state mutations stay coherent. Read-only endpoints don't acquire it.
- **`X-Accel-Buffering: no`** on `/send` SSE response. Prevents reverse proxies from buffering the stream.
- **`/send` rejects empty body with HTTP 400; the composer disables Send when the textarea is empty.** Defense-in-depth: a server-side placeholder used to substitute for empty bodies and stamped warmth-charged content into the transcript without the player choosing it.
- **Predicates are tagged-union Pydantic models, not strings.** YAML authoring is slightly more verbose, but every predicate is type-checked at load time and cross-references (quality ids, beat ids, thread ids) are validated by `Scenario._cross_references`.
- **`/scenario` exposes only `title`, `premise`** to the client. Dossiers, beats, qualities stay server-side.
- **`SETTINGS` is cached at lifespan boot.** Hot-path endpoints read `calibration_base` / `calibration_warmth_coef` from the cached instance. The cache is set inside `lifespan` after `Settings()` runs, so test fixtures using `monkeypatch.setenv(...)` before `TestClient(main.app)` still take effect.
