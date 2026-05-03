# CLAUDE.md

Inbox — single-surface email correspondence game. The player is the protagonist; NPCs reply to the player's emails after a human-feeling delay; how warmly the player writes shifts the NPC's voice register, which the EMAIL_AGENT picks up on the next render. Three-process tree:

```
Browser (Vite + React + TS) ──HTTP/SSE──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
                            ◄──────────                                     ◄──OpenAI /v1──
```

`make run` boots sidecar + web in parallel (sidecar spawns llama). `make sidecar` / `make web` / `make llama` run layers separately. Tests: `cd npc_service && uv run pytest` (37 tests).

## Game premise

Current scenario is "Threads": you wake up to a delivery-failure bounce showing your account sent an email to a stranger you don't recognize. Someone is sending email as you. Sam (tech-savvy friend, fast replier) and Alex (sibling, slow replier) both got pulled into the situation. You triangulate.

The mechanic the engine is built to test: **"Reply to an NPC. Your warmth shifts their voice register. Their next reply, after a human-feeling delay, comes back in a different voice band."**

## File map

```
npc_service/                  uv project root — pyproject.toml, scenario.yaml, .logfire/
  src/npc_service/
    main.py        FastAPI app: lifespan, agent wiring, all endpoints
                   (/scenario, /inbox, /thread, /open, /archive, /send,
                   /wait, /sleep). Owns SSE streaming for /send.
    models.py      Pydantic schemas. Authoring (QualityDef, NPCDossier with
                   VoiceBands, Beat tagged-union with type discriminator,
                   Predicate/Effect tagged unions, Scenario); runtime
                   (Email, Reply, PendingBeat, InboxState); HTTP DTOs.
    spine.py       Deterministic spine. Predicate evaluation, effect
                   application, beat scheduler (find_due_beats),
                   schedule_npc_reply, voice-band selection, time helpers
                   (advance_time, sleep_overnight). No LLM calls — pure
                   code-owned state mutation.
    agents.py      The two LLM agents (EMAIL_AGENT, TEXTURE_AGENT), their
                   system prompts, and per-call user-prompt builders.
    launcher.py    llama-server subprocess + Windows Job Object for cleanup.
    config.py      pydantic-settings.
  scenario.yaml    Inbox content layer. Qualities, NPC dossiers (Sam and
                   Alex with three voice bands each, register-conditioned),
                   threads (bounces + sam + alex), beats (14: 2 anchored
                   boot, 4 anchored time-gated bounces+alex, 8 rendered
                   per-NPC reply chains).
  tests/
    test_spine.py  25 tests — predicate/effect, beat scheduler, voice band
                   selection, time advancement, schedule_npc_reply.
    test_app.py    12 FastAPI tests — lifespan boot, all endpoints,
                   send→texture→reply scheduling end-to-end with TestModel.

web/                          Vite + React + TypeScript at workspace root
  src/
    main.tsx, App.tsx
    api.ts                    fetch + native SSE wrapper
    store.ts                  single Zustand store (server is source of truth;
                              store is render cache, refresh after mutations)
    hooks/useStreamSend.ts    shared SSE-drive helper used by EmailComposer;
                              refreshes inbox + reopens thread on done
    util/email.ts             stripAddress + formatGameTime
    fonts.css, index.css      IBM Plex Mono via @font-face; Tailwind base
    components/
      InboxHeader.tsx         title, current game time display, Wait/Sleep buttons
      FolderTabs.tsx          Inbox/Sent/Archive
      InboxList.tsx           folder content
      ThreadView.tsx          email chain + composer
      EmailComposer.tsx       free-form reply with 'got it.' placeholder
  public/fonts/               IBMPlexMono Regular + Bold
```

## Game loop

```
Boot:  scenario.yaml → InboxState → fire in_boot_inbox beats →
       client GET /inbox shows the inciting bounce + Sam's opening message

Player /open / /archive: free; clock does not advance.

Player /send → TEXTURE_AGENT scores warmth → spine writes warmth.<npc>
       (cumulative) and register.<npc> (drift halfway) → append Reply →
       advance_time(SEND_HOURS=0.25) → schedule_npc_reply(npc): walks
       scenario.beats, finds first un-fired beat for npc's thread whose
       preconditions pass, schedules it for now + midpoint(latency_range)
       → tick scheduler → may surface time-gated bounces

Player /wait → advance_time(WAIT_HOURS=2.0) → tick scheduler.
Player /sleep → advance_time(SLEEP_HOURS=8.0) → tick scheduler.
       Both surface any due NPC replies and time-gated anchored beats.

End:   when game_time_hours >= max_hours (96.0), state.ended = True with
       reason "time_exhausted". Terminal beats fire from precondition gates
       (time_hours_gte + warmth thresholds) before the budget runs out.
```

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel. Don't switch to `--reasoning-budget 0` — different PEG parse error per llama.cpp PR #21697.
- **`ToolOutput(...)` with `max_tokens` cap on every agent.** Tool-calling parses cleanly on Gemma 4 E4B; turns are bounded. NativeOutput unbounded text fields previously cascaded into 19K-token generations.
- **EMAIL_AGENT outputs `{prose}` only — no `claimed_facts`, no `requested_state_changes`.** The prose-rendering LLM never touches state. Only TEXTURE_AGENT (the *judging* agent) writes state, and only to the cumulative `warmth.<npc>` and the `register.<npc>` quality. Atom-style mutations come exclusively from beat.effects. Eliminates the LLM-as-state-actor risk class.
- **`register.<npc>` drifts halfway toward each reply's warmth on every send.** This is the per-reply visible feedback loop — the NPC's *next* email comes back in a different voice band. Without this, the texture mechanic is invisible.
- **Beats with no preconditions are excluded from `find_due_beats`'s Pass B** (`spine.find_due_beats`). Otherwise they'd fire on every tick forever. Per-NPC reply beats with no preconditions are scheduled exclusively via `schedule_npc_reply` on player send.
- **`schedule_npc_reply` does not reschedule when a reply is already pending.** If the player double-sends to an NPC, the existing pending reply still wins the latency clock; when it renders it sees the full transcript including all subsequent sends. Authors get batched-response behavior for free.
- **`/open` and `/archive` do not advance the clock.** Reading existing material can never be punished — that's how the player explores at their own pace without losing time budget. Only `/send`, `/wait`, `/sleep` move the clock.
- **Action-Lock around mutating endpoints** (`ACTION_LOCK` global mutex). Serializes mutating calls so spine state mutations stay coherent. Read-only endpoints don't acquire it.
- **`X-Accel-Buffering: no`** on `/send` SSE response. Prevents reverse proxies from buffering the stream.
- **Effects and predicates are tagged-union Pydantic models, not strings.** YAML authoring is slightly more verbose, but every effect and every predicate is type-checked at load time and cross-references (quality ids, beat ids, thread ids) are validated by `Scenario._cross_references`.
- **`game_time_hours` starts at 0.0.** The `formatGameTime` helper in the frontend adds an 8-hour offset for display (so 0.0 = "Day 1, 8:00 AM"). The spine never sees the offset.
- **`/scenario` exposes only `title`, `premise`, `max_hours`** to the client. Dossiers, beats, qualities stay server-side — the client doesn't need them and surfacing them would risk leaking authoring metadata.
