# CLAUDE.md

Inbox — single-surface email correspondence game. The player is the protagonist; an AI counterpart (the only NPC) replies after the player advances calibration past a threshold; how warmly the player writes shifts the NPC's voice register, which the EMAIL_AGENT picks up on the next render. Three-process tree:

```
Browser (Vite + React + TS) ──HTTP/SSE──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
                            ◄──────────                                     ◄──OpenAI /v1──
```

`make run` boots sidecar + web in parallel (sidecar spawns llama). `make sidecar` / `make web` / `make llama` run layers separately. Tests: `cd npc_service && uv run pytest` (38 tests).

## Game premise

Current scenario is **"Continuity Program"**: a corporate-satire pilot in which the player has been "enrolled" with an AI counterpart, **Aria**, who is calibrating on the player's communication style "to assume agreed-upon portions of your role." Boots into a single thread with a single welcome email. Severance-flat HR-bright tone — the horror is structural, never named.

The mechanic the engine is built to test: **"Reply to Aria. Your warmth shifts her voice register. Her next reply, after enough calibration ticks, comes back in a different voice band."**

There is no clock. The single progression axis is the `calibration` quality (0–100) — the player-visible doom counter, diegetically "counterpart fidelity %." Every player verb advances calibration; reaching the cap ends the session.

## File map

```
npc_service/                  uv project root — pyproject.toml, scenario.yaml, .logfire/
  src/npc_service/
    main.py        FastAPI app: lifespan, agent wiring, all endpoints
                   (/scenario, /state, /inbox, /thread, /open, /send,
                   /reset, /health). Owns SSE streaming for /send.
                   Caches the Settings instance at lifespan boot so
                   hot-path endpoints don't re-parse env per-request.
    models.py      Pydantic schemas. Authoring (QualityDef, NPCDossier with
                   VoiceBands and reply_delay_calibration, Beat tagged-union,
                   Predicate tagged union including BeatFired,
                   Effect = SetQuality, Scenario); runtime (Email, Reply,
                   PendingBeat with runtime_preconditions, InboxState);
                   HTTP DTOs.
    spine.py       Deterministic spine. Predicate evaluation (with tick-local
                   `extra_fired` set for mutual exclusion), effect
                   application, beat scheduler (find_due_beats — single
                   pass over scenario.beats × pending_by_id),
                   schedule_npc_reply (queues with QualityGte calibration
                   threshold), apply_calibration_delta, voice-band
                   selection. No LLM calls — pure code-owned state mutation.
    agents.py      The two LLM agents (EMAIL_AGENT, TEXTURE_AGENT), their
                   system prompts, and per-call user-prompt builders.
    launcher.py    llama-server subprocess + Windows Job Object for cleanup.
    config.py      pydantic-settings — including SEND_CALIBRATION_DELTA and
                   OPEN_CALIBRATION_DELTA tunables.
  scenario.yaml    Inbox content layer. Qualities (calibration + warmth.aria
                   + register.aria), one NPC dossier (Aria with three voice
                   bands: affirming/standard/corrective), threads (aria +
                   system), beats (7: 1 anchored boot, 2 anchored
                   calibration-gated system notices, 4 rendered Aria reply
                   chain).
  tests/
    test_spine.py  27 tests — predicate eval (incl. BeatFired and Not),
                   effect application, calibration delta + cap,
                   beat scheduler with runtime_preconditions, terminal
                   mutual-exclusion ordering, voice-band selection,
                   schedule_npc_reply.
    test_app.py    11 FastAPI tests — lifespan boot, all endpoints,
                   /open advances calibration, /send rejects empty body,
                   send→texture→reply scheduling end-to-end with TestModel.

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
      ThreadView.tsx          email chain + composer
      EmailComposer.tsx       free-form reply; Send disabled when empty
  public/fonts/               IBMPlexMono Regular + Bold
```

## Game loop

```
Boot:  scenario.yaml → InboxState.from_scenario (auto-pends every
       non-boot anchored beat; rendered Aria-reply beats stay out until
       schedule_npc_reply queues them) → fire in_boot_inbox beats →
       client GET /inbox shows Aria's welcome email.

Player /open → mark email read → apply_calibration_delta(+OPEN_DELTA)
       → tick scheduler → may surface system anchored beats whose
       quality_gte calibration threshold the player just crossed.

Player /send → apply_calibration_delta(+SEND_DELTA) → append player
       email → TEXTURE_AGENT scores warmth → spine writes warmth.aria
       (cumulative) and register.aria (drift halfway) → append Reply →
       schedule_npc_reply(aria): walks scenario.beats, finds first
       un-fired beat in aria's thread whose authored preconditions pass,
       queues with runtime predicate
         quality_gte(calibration, current + midpoint(reply_delay))
       → tick scheduler.

Asynchrony falls out of authoring, not waiting. The reply's threshold is
above the post-send calibration value (because reply_delay > 0), so a
single /send doesn't satisfy its own queued reply. A second /send (or
several /opens) ticks calibration past the threshold and the reply lands.

End:   when calibration reaches its cap (100), state.ended = True with
       reason "calibration_complete". Terminal beats are authored to fire
       at calibration ≥ 80 + reply count gates, so they normally land
       before the cap.
```

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel. Don't switch to `--reasoning-budget 0` — different PEG parse error per llama.cpp PR #21697.
- **`ToolOutput(...)` with `max_tokens` cap on every agent.** Tool-calling parses cleanly on Gemma 4 E4B; turns are bounded. NativeOutput unbounded text fields previously cascaded into 19K-token generations.
- **EMAIL_AGENT outputs `{prose}` only — no `claimed_facts`, no `requested_state_changes`.** The prose-rendering LLM never touches state. Only TEXTURE_AGENT (the *judging* agent) writes state, and only to the cumulative `warmth.<npc>` and the `register.<npc>` quality. SetQuality mutations come exclusively from beat.effects. Eliminates the LLM-as-state-actor risk class.
- **`register.<npc>` drifts halfway toward each reply's warmth on every send.** This is the per-reply visible feedback loop — the NPC's *next* email comes back in a different voice band. Without this, the texture mechanic is invisible.
- **`schedule_npc_reply` queues with a QualityGte calibration threshold above the current value.** This is how Aria's reply *feels asynchronous* without a wait verb. The `reply_delay_calibration` range on the dossier MUST exceed `SEND_CALIBRATION_DELTA`, otherwise a single send satisfies its own queued reply and the asynchrony collapses.
- **`schedule_npc_reply` does not reschedule when a reply is already pending for that NPC's thread.** If the player double-sends, the existing pending reply still wins; when it renders it sees the full transcript including all subsequent sends. Authors get batched-response behavior for free.
- **`/open` advances calibration too — not just `/send`.** Reading is calibration-positive in the fiction (the Program reads what you read). This is the only reason `/open` is a mutating endpoint and acquires `ACTION_LOCK`.
- **`InboxState.from_scenario` auto-pends every anchored beat that isn't `in_boot_inbox`.** Their authored preconditions become runtime predicates evaluated by `find_due_beats`. Rendered per-NPC reply beats are NOT auto-pended — those are added only by `schedule_npc_reply` on player send. Don't merge these two paths; the rendered chain depends on player verbs to advance.
- **`find_due_beats` walks pending beats in scenario authoring order and tracks tick-local `just_due_ids`.** Predicate evaluation receives `extra_fired = just_due_ids`, so a `Not(BeatFired(other))` precondition can express mutual exclusion between two beats that become eligible the same tick (e.g., `aria.terminal_warm` vs `aria.terminal_standard`). Authoring order breaks ties — author the higher-priority beat first.
- **The beat-id prefix MUST equal the beat's `thread` field.** Validated at scenario load (`Scenario._cross_references`). The spine treats the prefix as the source of truth via `thread_id_from_beat_id`; silent mismatch would break reply threading and reply counting.
- **`apply_calibration_delta` delegates clamping to `apply_effect(SetQuality)`** rather than re-implementing it. Single source of truth for quality mutation. The terminal-end-condition is the only calibration-specific behavior.
- **Action-Lock around mutating endpoints** (`ACTION_LOCK` global mutex). Serializes `/open` and `/send` so spine state mutations stay coherent. Read-only endpoints don't acquire it.
- **`X-Accel-Buffering: no`** on `/send` SSE response. Prevents reverse proxies from buffering the stream.
- **`/send` rejects empty body with HTTP 400; the composer disables Send when the textarea is empty.** A server-side `"got it."` placeholder used to substitute for empty bodies — that stamped warmth-charged content into the transcript and into Aria's training data without the player choosing it. The fix is enforced at both layers (defense-in-depth) so a curl bypass also rejects.
- **Effects and predicates are tagged-union Pydantic models, not strings.** YAML authoring is slightly more verbose, but every effect and every predicate is type-checked at load time and cross-references (quality ids, beat ids, thread ids) are validated by `Scenario._cross_references`.
- **`/scenario` exposes only `title`, `premise`** to the client. Dossiers, beats, qualities stay server-side — the client doesn't need them and surfacing them would risk leaking authoring metadata.
- **`SETTINGS` is cached at lifespan boot, not re-instantiated per request.** Hot-path endpoints (`/open`, `/send`) read deltas from the cached instance. The cache is set inside `lifespan` after `Settings()` runs, so test fixtures using `monkeypatch.setenv(...)` before `TestClient(main.app)` still work — the lifespan reads the patched env.
