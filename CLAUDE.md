# CLAUDE.md

Inbox — single-surface epistolary game. The player inhabits their sister Tess's email account while she's on a 72-hour psychiatric hold. Three-process tree:

```
Browser (Vite + React + TS) ──HTTP/SSE──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
                            ◄──────────                                     ◄──OpenAI /v1──
```

`make run` boots sidecar + web in parallel (sidecar spawns llama). `make sidecar` / `make web` / `make llama` run layers separately. Tests: `cd npc_service && uv run pytest` (47 tests; spine + FastAPI smoke with TestModel).

## Game premise

Player has been given access to Tess's inbox by hospital social worker Janet Rivera. Tess asked Janet to ask the player (her older brother) to triage anything urgent until she's back. **Scope: triage, not speak for her.** That bounding gives the unsent-draft mechanic its weight.

The arc moves from grief and quiet dread toward earned hopefulness. The reveal is *recontextualization*, not a horror twist — Tess had quietly built a life (Thursday writing group, food-pantry shifts, a friendship with Maya, a class she registered for) that the brother didn't know about.

## File map

```
npc_service/                  uv project root — pyproject.toml, scenario.yaml, .logfire/
  src/npc_service/
    main.py        FastAPI app: lifespan, agent wiring, all endpoints (boot,
                   inbox, thread, open, send, archive, draft, search, coda).
                   Owns the SSE streaming for /send, /draft, /coda.
    models.py      Pydantic schemas. Authoring (KnowledgeAtomDef, QualityDef,
                   NPCDossier with VoiceBands, Beat tagged-union with type
                   discriminator, Thread state machine, Predicate/Effect
                   tagged unions); runtime (Email, Reply, InboxState); HTTP DTOs.
    spine.py       Deterministic spine. Predicate evaluation, effect
                   application, beat scheduler, thread state machine evaluator,
                   voice-band selection, full-text search, coda fragment
                   selector. No LLM calls — pure code-owned state mutation.
    agents.py      The four LLM agents (EMAIL, TEXTURE, VOICE_FIDELITY, CODA),
                   their system prompts, and per-call user-prompt builders.
    launcher.py    llama-server subprocess + Windows Job Object for cleanup.
    config.py      pydantic-settings.
  scenario.yaml    Inbox content layer. Atoms, qualities, NPC dossiers (Maya
                   with three voice bands, Tess voice-only), threads (6),
                   beats (14: 3 anchored boot, 3 anchored arrivals, 6 rendered
                   Maya beats, the unsent draft, the coda), archive corpus
                   (8 emails for search). All anchored prose lives here.
  tests/
    test_spine.py  31 tests — predicate/effect, beat scheduler, thread state
                   machine (warm/ambivalent/cold trajectories), voice band
                   selection, search, coda fragment selector, end conditions.
    test_app.py    16 FastAPI tests — lifespan boot, all endpoints, three
                   draft branches end-to-end with TestModel.

web/                          Vite + React + TypeScript at workspace root
  src/
    main.tsx, App.tsx
    api.ts                    fetch + native EventSource SSE wrapper
    store.ts                  single Zustand store (server is source of truth;
                              store is render cache, refresh after mutations)
    hooks/useStreamSend.ts    shared SSE-drive helper used by EmailComposer and
                              DraftEditor; refreshes inbox + reopens thread on done
    fonts.css, index.css      IBM Plex Mono via @font-face; Tailwind base
    components/
      InboxHeader.tsx         persistent search bar + close-session button
      FolderTabs.tsx          Inbox/Drafts/Sent/Archive with unread badges
      InboxList.tsx           folder content
      ThreadView.tsx          email chain + composer or DraftEditor
      EmailComposer.tsx       free-form reply with 'got it.' placeholder
      DraftEditor.tsx         three equal-weight buttons; reveal editor on edit
      SearchResults.tsx       free; doesn't advance actions.elapsed
      Coda.tsx                interleaves SSE fragments + framing lines
  public/fonts/               IBMPlexMono Regular + Bold
```

## Game loop

```
Boot:  scenario.yaml → InboxState → fire 3 in_boot_inbox beats →
       client GET /inbox shows Janet, class-reg, old Maya email

Player action (open / archive / send): action_count++ → tick scheduler →
       fire due beats (anchored ship verbatim; rendered → EMAIL_AGENT) →
       evaluate thread transitions → check end conditions → SSE state

Reply path:  POST /send → TEXTURE_AGENT scores warmth+honesty → spine
             writes warmth.maya, register.maya (drift halfway), honesty.global →
             append Reply → action_count++ → tick → may fire Maya rendered
             beats with the new voice band

Draft path:  POST /draft/{id}/decision (send_unchanged | edit_and_send | hold)
             edit_and_send → VOICE_FIDELITY_AGENT scores against Tess's dossier
             → spine writes voice_fidelity, voice_fidelity_rationale →
             flow into maya.response_to_draft's spec
             hold → no Maya beat fires; thread continues prior trajectory

End:    coda fires when (action_count >= 16 AND maya in {warm,ambivalent,cold})
        OR action_count >= 18. Coda fragments selected deterministically;
        CODA_AGENT only writes 6-10 framing lines.
```

## Things that look removable but aren't

- **`when:` not `on:`** in ThreadTransition. PyYAML 1.1 parses bare `on:` as the boolean `True`, silently swallowing transition predicates. Don't rename back without switching to a YAML 1.2 loader.
- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel. Don't switch to `--reasoning-budget 0` — different PEG parse error per llama.cpp PR #21697.
- **`ToolOutput(...)` with `max_tokens` cap on every agent.** Tool-calling parses cleanly on Gemma 4 E4B; turns are bounded. NativeOutput unbounded text fields previously cascaded into 19K-token generations.
- **EMAIL_AGENT outputs `{prose}` only — no `claimed_facts`, no `requested_state_changes`.** The prose-rendering LLM never touches state. Only TEXTURE_AGENT and VOICE_FIDELITY_AGENT (the *judging* agents) write state, and only to continuous quality fields. Atom mutations come exclusively from beat.effects. Eliminates the LLM-as-state-actor risk class.
- **The thread state machine is invisible to the LLM.** EMAIL_AGENT sees the dossier, voice band, transcript, and beat spec — it does not know which transition just fired or which beat is queued next. Prevents the model from "playing along" with the spine and drifting.
- **Trajectory predicates use averages over `state.replies`, not last-reply scores.** `avg_quality_gte` reads every reply in the thread and averages — that's how a cold trajectory survives a late warmth rush. Removing this re-enables the cheap "be cold then nice at the end" tactic.
- **`register.maya` drifts halfway toward each reply's warmth on every send.** This is the per-reply visible feedback loop — Maya's *next* email comes back in a different voice band. Without this, the texture mechanic is invisible until the coda.
- **All three DraftEditor buttons are styled equally.** Holding must read as a real, respectful option in the UI, not a cancel affordance. Janet's permission scope ("triage anything urgent") makes holding the in-bounds choice; sending is the one that exceeds scope.
- **Tess never speaks live.** Her dossier seeds VOICE_FIDELITY_AGENT and authors the unsent draft prose. The archive emails (search corpus) are her secondary voice exhibit. No rendered beat ever invokes EMAIL_AGENT against Tess's dossier.
- **Action counter increments only on opens, archives, and sends.** Search and re-reads are free. Reading existing material can never be punished — that would defeat the search-as-recontextualization mechanic.
- **Boot inbox sits in scenario.yaml as `in_boot_inbox: true` beats.** They fire on the first scheduler tick at action_count=0. Three of them: Janet (newest), class registration auto-reminder (yesterday), Maya's old "hey are you ok" (3 days ago, sent before the hold).
- **`maya.draft_to_her_visible` is gated on `player_replied_to: maya.first_email`.** The draft surfaces in the Drafts folder only after the player has engaged with Maya at least once. Until then, Drafts is empty.
- **`maya.response_to_draft` only fires on `unsent_draft_disposition in [sent_unchanged, sent_edited]`.** Held → Maya never sees the draft, thread continues on its prior trajectory.
- **`HttpCompletionOption.ResponseHeadersRead`** equivalent for fetch is automatic in browsers; SSE works as long as the server emits proper `text/event-stream` with no Content-Length and no chunked-buffering proxies.
- **`X-Accel-Buffering: no`** on `/send`, `/draft/.../decision`, and `/coda` responses. Same reason as before — prevents reverse proxies from buffering the stream.
- **`ACTION_LOCK` global mutex** serializes mutating endpoints so the spine's state mutations stay coherent. Search and read endpoints don't acquire it.
- **`/scenario` exposes only `title`, `premise`, `max_actions`** to the client. Dossiers, beats, atoms, thread state machines stay server-side — the client doesn't need them and surfacing them would risk leaking authoring metadata.
- **Effects and predicates are tagged-union Pydantic models, not strings.** YAML authoring is slightly more verbose, but every effect and every predicate is type-checked at load time and cross-references (atom ids, quality ids, beat ids, thread ids) are validated by `Scenario._cross_references`.
