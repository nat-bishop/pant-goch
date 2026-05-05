# CLAUDE.md

Inbox — single-surface email correspondence game. The player is the protagonist; an AI counterpart (the only NPC) replies on every send. Authored "plot beats" (calibration milestones, terminal endings) inject one-shot directives onto the chat agent's next turn; the rest is free chat in character. The chain advances on reply counts and on classifier-driven qualities written when the player replies to specific authored beats. Three-process tree:

```
Browser (Vite + React + TS) ──HTTP/SSE──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
                            ◄──────────                                     ◄──OpenAI /v1──
```

`make run` boots sidecar + web in parallel (sidecar spawns llama). `make sidecar` / `make web` / `make llama` run layers separately. Tests: `cd npc_service && uv run pytest`.

## Game premise

Current scenario is **"Continuity Program"**: a corporate-satire pilot in which the player has been "enrolled" with an AI counterpart, **Aria**, who is calibrating on the player's communication style "to assume agreed-upon portions of your role." Boots into a single thread with a single welcome email. Severance-flat HR-bright tone — the horror is structural, never named.

The mechanic the engine is built to test: **"Reply to Aria. Authored plot moments arrive as the chain advances; the warmth check at midprogram routes you to the warm or standard ending."**

There is no clock. The player-visible doom counter is `calibration` (0–100, "counterpart fidelity %"); it advances by a flat per-send delta and is otherwise decorative — no beat gates on its value. Reaching 100 ends the session (safety net; terminals are authored to fire well before).

## Architecture in one diagram

```
                                    player /send body
                                            │
                                            ▼
                                 record send (Sent folder)
                                            │
                                            ▼
                              apply_calibration_per_send (+base)
                                            │
                                            ▼
                       replied_beat.classify_reply is not None?
                                /                      \
                              yes                        no
                               │                         │
                               ▼                         │
                  classifier_agent.run(question,         │
                  allowed_values) → write quality        │
                               │                         │
                               └─────────────┬───────────┘
                                             ▼
                                      find_due_beats
                                       /            \
                                PlotBeat eligible    (none)
                                       │              │
                                       ▼              ▼
                            chat_agent.run(body,  chat-turn fallback:
                            history, deps with    chat_agent.run with
                            plot_directive=spec)  no plot directive
                                       │              │
                                       ▼              ▼
                       thread_messages.extend(new_messages); emit Email
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
                   email_address, AnchoredBeat, PlotBeat with BeatSpec,
                   ClassifyReply, Predicate tagged union: QualityGte/Lt,
                   PlayerRepliesToThreadGte, BeatFired, Not_, Scenario);
                   runtime (Email, Reply, PendingBeat, InboxState with
                   thread_messages dict of list[ModelMessage]); HTTP DTOs.
    spine.py       Deterministic spine. Predicate evaluation, beat scheduler
                   (find_due_beats with tick-local extra_fired for mutual
                   exclusion), apply_calibration_delta,
                   apply_calibration_per_send (flat delta),
                   apply_classification (LLM-judge → named quality),
                   reply_subject_for_email, latest_npc_email_in_thread.
                   No LLM calls — pure code-owned state mutation.
    agents.py      Two LLM agents.
                   chat_agent: output_type=str, deps_type=ChatDeps; per-run
                   @instructions decorators inject dossier and (when set)
                   plot directive. Used for both chat turns
                   (plot_directive=None) and PlotBeat fires.
                   classifier_agent: ToolOutput, scores the player's reply
                   against a beat-authored question + allowed_values list,
                   returns a single value. Runs only when the replied-to
                   beat authored a `classify_reply`.
                   _BASE_CHAT_INSTRUCTIONS enforces body-only output —
                   no Subject:/From: headers, no JSON, no markdown fences,
                   no signature unless the dossier specifies one.
    launcher.py    llama-server subprocess + Windows Job Object for cleanup.
    config.py      pydantic-settings — calibration_base (flat per-send delta).
  scenario.yaml    Inbox content layer. Qualities (calibration +
                   engaged_initial + partnership_warmth), one NPC dossier
                   (Aria with email_address), threads (aria + system),
                   beats (7: 1 anchored welcome, 3 PlotBeats for Aria's
                   authored milestones with classify_reply on
                   first_calibration and midprogram, 1 PlotBeat terminal
                   pair, 2 anchored midpoint/transition system notices).
  tests/
    test_spine.py  Predicate eval, calibration delta + cap,
                   apply_calibration_per_send, beat scheduler with
                   runtime_preconditions, multi-turn beat re-fire and
                   completion, apply_classification, terminal mutual
                   exclusion via Not_(BeatFired) gated on
                   partnership_warmth, reply_subject_for_email,
                   latest_npc_email_in_thread.
    test_app.py    FastAPI tests — lifespan boot, /open does NOT mutate
                   calibration, /send delta is flat (=calibration_base),
                   first /send returns immediate chat-turn fallback,
                   thread_messages accumulates ModelRequest/Response across
                   sends, body-only regression guard (no Subject:/From:
                   leakage), classifier runs on reply to a beat with
                   classify_reply (first_calibration → engaged_initial,
                   midprogram → partnership_warmth), anchored system beat
                   fires via beat_fired chain.

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
Boot:  scenario.yaml → InboxState.from_scenario (auto-pends every beat —
       anchored AND plot — with their authored preconditions evaluated
       each tick) → find_due_beats fires aria.welcome at tick 0 (no
       preconditions). Welcome's body is appended to thread_messages["aria"]
       as ModelResponse(TextPart) — the chat agent's first run sees the
       welcome as Aria's prior turn.

Player /open → mark email read. No calibration mutation. (Read-only
       w.r.t. state mutations; still acquires ACTION_LOCK to serialise
       with /send.)

Player /send → record player Email (Sent folder) → apply_calibration_per_send
       (+base, decorative meter advance) → if the replied-to beat has a
       `classify_reply` block, run classifier_agent and apply_classification
       to write the named quality → find_due_beats fires AnchoredBeats whose
       preconditions just passed AND PlotBeats whose preconditions just
       passed (those PlotBeats run chat_agent with plot_directive=spec).
       If no PlotBeat replied for the player's NPC thread → fire chat-turn
       fallback immediately (chat_agent.run with no plot directive, plain
       str output, body appended to thread_messages alongside the agent's
       response).

Asynchrony in chat-turn replies lives in LLM render time + SSE streaming.
Plot beats fire when the chain reaches them (beat_fired predicates,
reply-count predicates, classifier-driven qualities) — not on a wait verb.

End:   when calibration reaches its cap (100), state.ended = True with
       reason "calibration_complete". Terminal PlotBeats are authored to
       fire on reply count + partnership_warmth gates, so they normally
       land before the cap.
```

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel. Don't switch to `--reasoning-budget 0` — different PEG parse error per llama.cpp PR #21697.
- **`output_type=str` on `chat_agent`, NOT `ToolOutput`.** With `ToolOutput`, the model's reply lands in `message_history` as a `ToolCallPart`, not a `TextPart` — subsequent turns re-running with that history see a tool-call sequence instead of natural assistant turns. Plain text with `max_tokens=800` keeps generation bounded without the JSON-schema-driven cascade that previously produced 19K-token Gemma outputs (that was a NativeOutput problem, different code path from plain str). The classifier agent keeps `ToolOutput` because it's one-shot and never feeds into a conversation.
- **`_BASE_CHAT_INSTRUCTIONS` explicitly forbids `Subject:` / `From:` / `To:` / `Date:` headers, JSON, markdown code fences, and unsanctioned signatures.** With `output_type=str` there's no schema enforcing structure, so the rule lives in the prompt. There's a pytest regression guard that asserts the rendered body contains none of these tokens.
- **Chat agent uses `@instructions` (per-run dynamic), not user-prompt formatting.** Pydantic AI's `@instructions` are appended to the system prompt every run, even when `message_history` is non-empty. This lets the per-turn dossier and plot-directive layer sit in instructions while the conversation history (player+assistant turns) flows naturally through `message_history`. Mixing these paths (e.g. embedding the dossier in the user prompt) would pollute history.
- **The chat agent never writes state.** The agent returns prose only. Only the classifier writes state, and only to the single named quality the replied-to beat authored. The spine itself is the only thing that writes `calibration` (via `apply_calibration_per_send → apply_calibration_delta`). Eliminates the LLM-as-state-actor risk class.
- **Calibration is a flat per-send delta, not a heuristic or LLM-derived value.** No scenario beat gates on `calibration` — it's the visible "Counterpart fidelity %" doom counter and a session-end safety net at 100. Cheap deterministic climb is exactly what's wanted; coupling to LLM output would just add latency without changing the gate set.
- **Classifier runs only on replies to beats that author `classify_reply`.** Per-send default path skips the classifier entirely — no warmth scoring on every send. The chain advances on reply counts and on classifier-written qualities at specific authored moments (`first_calibration` → `engaged_initial`, `midprogram` → `partnership_warmth`). Adding a classifier to a beat is purely YAML-side; no code change.
- **`/open` no longer mutates calibration but still acquires `ACTION_LOCK`.** Keeps it serialised with `/send` so the `email.is_read` flip and any concurrent state inspection by `/send` stay coherent.
- **`InboxState.from_scenario` auto-pends every beat (anchored AND plot).** Their authored preconditions become runtime predicates evaluated by `find_due_beats`; an anchored beat with empty preconditions fires on the first scheduler tick at boot. PlotBeats sit in pending until their preconditions cascade (typically `beat_fired` of an upstream beat plus a reply-count threshold).
- **Anchored beats on NPC threads seed `thread_messages` with a `ModelResponse(TextPart)` when they fire** (`_seed_thread_history` in `main.py`). Without this, the chat agent's first run after boot would have no context — Aria would be replying to the player's first send without "remembering" her welcome email. System-thread anchored beats skip seeding (no chat agent on the system thread).
- **The beat-id prefix MUST equal the beat's `thread` field.** Validated at scenario load (`Scenario._cross_references`). The spine treats the prefix as the source of truth via `thread_id_from_beat_id`; silent mismatch would break reply threading and reply counting.
- **Email metadata for chat turns is *derived*, not authored.** `sender` = `NPCDossier.email_address`; `recipient` = `latest_npc_email_in_thread(...).recipient` (the player's address, read off the most recent NPC-originated email's `to:` field — the boot welcome guarantees this is populated); `subject` = `reply_subject_for_email(state, in_reply_to_email_id)`. PlotBeat fires override `subject` with the authored value but inherit the rest.
- **`SendRequest.in_reply_to` is an email id, not a beat id.** Frontend `ThreadView` passes `composeTarget.id` (not `composeTarget.beat_id`) so chat-turn replies (which have `beat_id=None`) can be replied to. `_reply_subject` looks up by email id. The on-Email field is named `in_reply_to_beat` for backward compat with the frontend type, but its semantic is "email id being replied to."
- **`apply_calibration_delta` mutates `state.qualities` directly** and uses `_clamp_quality` (shared with `apply_classification`) for bounds. The terminal-end-condition (set `state.ended = True` when the cap is reached) is the only calibration-specific behavior. `apply_calibration_per_send` is a thin wrapper that applies a flat `base` and delegates — single clamping path.
- **Action-Lock around mutating endpoints** (`ACTION_LOCK` global mutex). Serializes `/open` and `/send` so spine state mutations stay coherent. Read-only endpoints don't acquire it.
- **`X-Accel-Buffering: no`** on `/send` SSE response. Prevents reverse proxies from buffering the stream.
- **`/send` rejects empty body with HTTP 400; the composer disables Send when the textarea is empty.**
- **Predicates are tagged-union Pydantic models, not strings.** YAML authoring is slightly more verbose, but every predicate is type-checked at load time and cross-references (quality ids, beat ids, thread ids) are validated by `Scenario._cross_references`.
- **`/scenario` exposes only `title`, `premise`** to the client. Dossiers, beats, qualities stay server-side.
- **`SETTINGS` is cached at lifespan boot.** Hot-path endpoints read `calibration_base` from the cached instance. The cache is set inside `lifespan` after `Settings()` runs, so test fixtures using `monkeypatch.setenv(...)` before `TestClient(main.app)` still take effect.
