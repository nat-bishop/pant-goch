# CLAUDE.md

Inbox — single-surface email correspondence game. The player is the protagonist; an AI counterpart (the only NPC) replies on every send. One ongoing "calibration loop" PlotBeat fires on every player /send and classifies the reply into a cumulative `engagement` quality; two ending PlotBeats fire when reply count + engagement thresholds are met. Three-process tree:

```
Browser (Vite + React + TS) ──HTTP/SSE──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
                            ◄──────────                                     ◄──OpenAI /v1──
```

`make run` boots sidecar + web in parallel (sidecar spawns llama). `make sidecar` / `make web` / `make llama` run layers separately. Tests: `cd npc_service && uv run pytest`.

## Game premise

Current scenario is **"Continuity Program"**: a corporate-satire pilot in which the player has been "enrolled" with an AI counterpart, **Aria**, who is calibrating on the player's communication style "to assume agreed-upon portions of your role." Boots into a single thread with a single welcome email. Severance-flat HR-bright tone — the horror is structural, never named.

The mechanic the engine is built to test: **"Reply to Aria. Every reply gets classified for engagement; that score accumulates and routes you to the warm or standard ending after five replies."**

There is no clock and no decorative meter. Pacing is action-counter driven: the warm ending requires five replies AND `engagement ≥ 4.0`; the standard ending requires five replies and the warm ending hasn't fired (mutex). Either ending sets `state.ended = True` via the beat's `ends_session: true` flag.

## Architecture in one diagram

```
                                    player /send body
                                            │
                                            ▼
                                 record send (Sent folder),
                                 append Reply to state.replies
                                            │
                                            ▼
                       replied_beat.classify_reply is not None?
                                /                      \
                              yes                        no
                               │                         │
                               ▼                         │
                  classifier_agent.run(question,         │
                  allowed_values) →                      │
                  apply_classification (set OR           │
                  accumulate, depending on the           │
                  beat's classify_reply.accumulate)      │
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
                                       │
                                       ▼
                           emit Email; if PlotBeat.ends_session
                           → state.ended=True; if PlotBeat.multi_turn
                           → never enters fired_beat_ids
```

## File map

```
npc_service/                  uv project root — pyproject.toml, scenario.yaml, .logfire/
  src/npc_service/
    main.py        FastAPI app: lifespan, agent wiring, all endpoints
                   (/scenario, /state, /inbox, /thread, /open, /send,
                   /reset, /health). Owns SSE streaming for /send and the
                   chat-turn-vs-PlotBeat dispatch. _fire_beat honors the
                   PlotBeat flags (multi_turn, ends_session).
    models.py      Pydantic schemas. Authoring (QualityDef, NPCDossier with
                   email_address, AnchoredBeat, PlotBeat with BeatSpec +
                   multi_turn + ends_session, ClassifyReply with accumulate,
                   Predicate tagged union: QualityGte/Lt,
                   PlayerRepliesToThreadGte, BeatFired, Not_, Scenario);
                   runtime (Email, Reply, PendingBeat, InboxState with
                   thread_messages dict of list[ModelMessage]); HTTP DTOs.
    spine.py       Deterministic spine. Predicate evaluation, beat scheduler
                   (find_due_beats with tick-local extra_fired for mutual
                   exclusion), apply_classification (LLM-judge → named
                   quality; set OR accumulate),
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
                   beat authored a `classify_reply`. max_tokens=2048 so
                   Gemma 4 E4B has headroom to emit the tool call.
                   _BASE_CHAT_INSTRUCTIONS enforces body-only output —
                   no Subject:/From: headers, no JSON, no markdown fences,
                   no signature unless the dossier specifies one.
    launcher.py    llama-server subprocess + Windows Job Object for cleanup.
    config.py      pydantic-settings — service ports, llama lifecycle,
                   history compaction knobs.
  scenario.yaml    Inbox content layer. One quality (`engagement`,
                   0–10, accumulating). One NPC dossier (Aria with
                   email_address). One thread (aria). Four beats:
                   1 anchored welcome + 1 multi_turn loop PlotBeat
                   (with classify_reply accumulate=true) + 2 ends_session
                   ending PlotBeats (complete_warm gated on engagement≥4
                   + replies≥5; complete_standard gated on replies≥5 +
                   Not_(BeatFired complete_warm)).
  tests/
    test_spine.py  Predicate eval, apply_classification (set + accumulate
                   + clamp + off-spec), beat scheduler with
                   runtime_preconditions, loop scheduling and mutex against
                   terminals, complete_warm/standard threshold + reply-count
                   gating, same-tick mutex via Not_(BeatFired) +
                   extra_fired, reply_subject_for_email,
                   latest_npc_email_in_thread.
    test_app.py    FastAPI tests — lifespan boot, /open is read-only,
                   /send rejects empty bodies, first /send fires the
                   loop beat, multi_turn loop re-fires across sends,
                   thread_messages accumulates ModelRequest/Response,
                   body-only regression guard (no Subject:/From: leakage),
                   classifier accumulates engagement on reply to loop,
                   complete_warm fires + state.ended at engagement≥4 +
                   replies≥5, complete_standard fires when warm blocked,
                   /send returns 409 after session end.

web/                          Vite + React + TypeScript; Tailwind 3 with
                              CSS-variable theme tokens (light/dark)
  index.html                  inline boot script reads localStorage/
                              prefers-color-scheme and sets class="dark" on
                              <html> BEFORE the JS bundle loads — FOUC-free
  src/
    main.tsx, App.tsx         App.tsx renders the single Aria chat inside
                              AppShell (no router; one chat, one composer)
    api.ts                    fetch + native SSE wrapper. SseEvent union:
                              send_recorded, classified, new_email, state,
                              done, error.
    store.ts                  Zustand store. Per-thread chatMessages cache,
                              pendingTypingFor (drives the typing indicator),
                              firstOpenForChat Set (welcome-replay tracking),
                              theme, bootStarted (StrictMode-safe boot).
                              openChat replays unread Aria messages through
                              the typing indicator on first session-open.
    hooks/useStreamSend.ts    SSE consumer. Buffers new_email events in a
                              local queue, holds the typing indicator, then
                              pops each bubble after typingDelayMs(body) so
                              Aria appears to be composing rather than
                              spitting an LLM response.
    hooks/useScrollAnchor.ts  tracks whether the message list is pinned to
                              the bottom; powers the "↓ New message" pill
    hooks/useAutoGrow.ts      composer textarea auto-grow (capped at 6 lines)
    hooks/useTheme.ts         light/dark; persists toggle to localStorage,
                              syncs class="dark" on <html>
    util/email.ts             stripAddress
    util/avatar.ts            deterministic HSL gradient + initials from a
                              seed string (Aria's identity is stable across
                              reloads without any asset pipeline)
    util/npc.ts               ARIA_SEED, ARIA_DISPLAY_NAME constants — the
                              single source of truth for Aria's avatar seed
    util/pacing.ts            typingDelayMs = clamp(body.length / 25, 1.5s,
                              4.5s) — slower than reading speed, deliberate
    index.css                 :root + .dark CSS-variable theme blocks,
                              custom scrollbar, focus ring, keyframes
                              (bubble-in, dot-bounce, presence-pulse,
                              pill-slide, send-slide). Honors
                              prefers-reduced-motion.
    components/
      AppShell.tsx            top-level layout: AppHeader on top, sidebar
                              left (both bg-bg-sidebar). Floating main card
                              with rounded corners + shadow over the
                              L-shaped chrome.
      AppHeader.tsx           wordmark + optional "Transition Complete" pill
                              (state.ended) + ThemeToggle
      ChatSidebar.tsx         single "Direct" section with the Aria row
      ChatRow.tsx             sidebar row primitive (active state animates a
                              left accent bar; unread badge from state)
      Avatar.tsx              gradient+initials avatar
      PresenceDot.tsx         green dot with a slow outer pulse — Aria is
                              always Active
      ChatHeader.tsx          per-chat header: avatar + name + presence
      MessageList.tsx         scrollable bubble container; groups
                              consecutive same-sender messages, holds
                              scroll-anchor logic, mounts the typing
                              indicator when pendingTypingFor matches
      MessageGroup.tsx        same-sender consecutive bubbles share one
                              avatar + one trailing timestamp
      MessageBubble.tsx       single bubble (me/them variants, asymmetric
                              corner radii by group position, optional
                              uppercase eyebrow for plot beats — the
                              authored subject)
      TypingIndicator.tsx     three-dot bouncing animation in a bubble
      NewMessagePill.tsx      "↓ New message" pill that slides up when the
                              user is scrolled away from the bottom
      MessageComposer.tsx     auto-grow textarea, Enter to send,
                              Shift+Enter newline. Send slides in when text
                              is non-empty. Disabled while
                              pendingTypingFor === currentChatId.
      ThemeToggle.tsx         sun/moon crossfade
  tailwind.config.ts          darkMode: "class". Colors map to CSS vars
                              (bg, bg-sidebar, bg-bubble-them, text, accent,
                              etc.); fontFamily.sans → Inter, .mono →
                              JetBrains Mono. Animation keyframes registered.
```

## Game loop

```
Boot:  scenario.yaml → InboxState.from_scenario (auto-pends every beat —
       anchored AND plot — with their authored preconditions evaluated
       each tick) → find_due_beats fires aria.welcome at tick 0 (no
       preconditions). Welcome's body is appended to thread_messages["aria"]
       as ModelResponse(TextPart) — the chat agent's first run sees the
       welcome as Aria's prior turn.

Player /open → mark email read. Mutates no qualities; still acquires
       ACTION_LOCK to serialise the is_read flip with concurrent /send.

Player /send → record player Email (Sent folder) + Reply (drives
       player_replies_to_thread_gte) → if the replied-to beat has a
       `classify_reply` block, run classifier_agent and apply_classification
       (set or accumulate) to write the named quality → find_due_beats
       walks pending beats in authoring order. The first /send (replies≥1)
       lands on the loop, which is multi_turn so it never enters
       fired_beat_ids. After enough replies + engagement, the terminal
       beats become eligible; they're authored before the loop so on the
       crossing tick the terminal lands first in extra_fired and the loop's
       Not_(BeatFired complete_*) preconditions block it from firing
       alongside.

End:   when an ends_session PlotBeat fires, _fire_beat sets state.ended=True
       and state.end_reason = beat.id. /send and /open then return 409.
       In the standard arc complete_warm or complete_standard fires on the
       fifth reply; there is no other end condition.
```

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel. Don't switch to `--reasoning-budget 0` — different PEG parse error per llama.cpp PR #21697.
- **`output_type=str` on `chat_agent`, NOT `ToolOutput`.** With `ToolOutput`, the model's reply lands in `message_history` as a `ToolCallPart`, not a `TextPart` — subsequent turns re-running with that history see a tool-call sequence instead of natural assistant turns. Plain text with `max_tokens=800` keeps generation bounded without the JSON-schema-driven cascade that previously produced 19K-token Gemma outputs (that was a NativeOutput problem, different code path from plain str). The classifier agent keeps `ToolOutput` because it's one-shot and never feeds into a conversation.
- **`classifier_agent` runs at `max_tokens=2048`, not 256.** Gemma 4 E4B occasionally preambles before emitting the tool call; at 256 it regularly hit `finish_reasons=["length"]` and forced a Pydantic AI retry, doubling latency. 2048 leaves comfortable headroom.
- **`_BASE_CHAT_INSTRUCTIONS` explicitly forbids `Subject:` / `From:` / `To:` / `Date:` headers, JSON, markdown code fences, and unsanctioned signatures.** With `output_type=str` there's no schema enforcing structure, so the rule lives in the prompt. There's a pytest regression guard that asserts the rendered body contains none of these tokens.
- **Chat agent uses `@instructions` (per-run dynamic), not user-prompt formatting.** Pydantic AI's `@instructions` are appended to the system prompt every run, even when `message_history` is non-empty. This lets the per-turn dossier and plot-directive layer sit in instructions while the conversation history (player+assistant turns) flows naturally through `message_history`. Mixing these paths (e.g. embedding the dossier in the user prompt) would pollute history.
- **The chat agent never writes state.** The agent returns prose only. Only the classifier writes state, and only to the single named quality the replied-to beat authored. Eliminates the LLM-as-state-actor risk class.
- **`ClassifyReply.accumulate: bool`.** Default False (classifier output overwrites the quality — useful for one-shot decisions like a midpoint warmth check). True is what the loop uses: the mapped score is added to the existing value, clamped — so engagement accumulates across replies without any spine-side bookkeeping.
- **`PlotBeat.multi_turn: bool`.** When True, the beat re-fires on every tick because `_fire_beat` skips adding it to `fired_beat_ids`. Cleaner than abusing `completes_when` with a never-True predicate. Stop the loop firing via mutex preconditions (e.g. `Not_(BeatFired aria.complete_warm)`). Same-tick mutex resolution depends on the terminals being authored BEFORE the loop in scenario.yaml so they enter `extra_fired` before the loop is evaluated.
- **`PlotBeat.ends_session: bool`.** When True, `state.ended` flips immediately after the beat's email is emitted, with `end_reason = beat.id`. Keeps session-end logic out of `apply_classification` and out of any global "session over" predicate — it's just a property of the terminal beat.
- **Classifier runs only on replies to beats that author `classify_reply`.** Replying to the welcome (anchored, no classifier) doesn't classify; replying to a loop email does. The loop's classifier accumulates engagement; that's the only signal the warm/standard split reads.
- **`/open` mutates no qualities but still acquires `ACTION_LOCK`.** Keeps it serialised with `/send` so the `email.is_read` flip and any concurrent state inspection by `/send` stay coherent.
- **`InboxState.from_scenario` auto-pends every beat (anchored AND plot).** Their authored preconditions become runtime predicates evaluated by `find_due_beats`; an anchored beat with empty preconditions fires on the first scheduler tick at boot. PlotBeats sit in pending until preconditions hold.
- **Anchored beats on NPC threads seed `thread_messages` with a `ModelResponse(TextPart)` when they fire** (`_seed_thread_history` in `main.py`). Without this, the chat agent's first run after boot would have no context — Aria would be replying to the player's first send without "remembering" her welcome email.
- **The beat-id prefix MUST equal the beat's `thread` field.** Validated at scenario load (`Scenario._cross_references`). The spine treats the prefix as the source of truth via `thread_id_from_beat_id`; silent mismatch would break reply threading and reply counting.
- **Email metadata for chat turns is *derived*, not authored.** `sender` = `NPCDossier.email_address`; `recipient` = `latest_npc_email_in_thread(...).recipient` (the player's address, read off the most recent NPC-originated email's `to:` field — the boot welcome guarantees this is populated); `subject` = `reply_subject_for_email(state, in_reply_to_email_id)`. PlotBeat fires override `subject` with the authored value but inherit the rest.
- **`SendRequest.in_reply_to` is an email id, not a beat id.** Frontend `App.tsx::pickComposeTarget` finds the most recent NPC email and passes its `.id` (not `.beat_id`) so chat-turn replies (which have `beat_id=None`) can be replied to. `_reply_subject` looks up by email id. The on-Email field is named `in_reply_to_beat` for backward compat with the frontend type, but its semantic is "email id being replied to."
- **`apply_classification` mutates `state.qualities` directly** and routes both set and accumulate paths through `_clamp_quality` so quality bounds (e.g. engagement's 0..10 floor at 0) apply uniformly.
- **Action-Lock around mutating endpoints** (`ACTION_LOCK` global mutex). Serializes `/open` and `/send` so spine state mutations stay coherent. Read-only endpoints don't acquire it.
- **`X-Accel-Buffering: no`** on `/send` SSE response. Prevents reverse proxies from buffering the stream.
- **`/send` rejects empty body with HTTP 400; the composer disables Send when the textarea is empty.**
- **Predicates are tagged-union Pydantic models, not strings.** YAML authoring is slightly more verbose, but every predicate is type-checked at load time and cross-references (quality ids, beat ids, thread ids) are validated by `Scenario._cross_references`.
- **`/scenario` exposes only `title`, `premise`** to the client. Dossiers, beats, qualities stay server-side.
- **No token streaming on the LLM path.** Pydantic AI agents return a complete `ModelResponse`; `/send` emits the full email body in one `new_email` SSE event. The frontend buffers `new_email` events and reveals them after `typingDelayMs(body)` (25 cps clamped 1.5–4.5s) so Aria appears to be composing rather than spitting tokens. This is deliberate — token streaming would lose the "deliberate, calibrated" pacing that makes Aria feel like a presence rather than an autocomplete.
- **Welcome typing-replay on first session-open.** `store.openChat` checks `firstOpenForChat` and, if Aria's chat has unread messages on the first session-open, plays them through the typing indicator instead of dumping the welcome already-on-screen. The very first interaction begins with "Aria is typing…" — atmosphere win for the opening.
- **`store.bootStarted` flag.** React StrictMode double-mounts the App in dev, which double-fires `boot()` and would replay the welcome twice. The flag short-circuits a second concurrent boot. Set synchronously before any await so the second call sees it.
- **`index.html` inline boot script.** Reads `localStorage.theme` (or `prefers-color-scheme`) and toggles `class="dark"` on `<html>` BEFORE the JS bundle parses. Without it the page paints in the wrong theme for ~50–200ms and snaps. Don't move this into `main.tsx`.
- **Chat-bubble bodies pass through `flowParagraphs` in `MessageBubble`.** Authored prose (welcome, plot beats) often arrives with hard wraps from a YAML `|` block scalar — those look like a poem inside a chat bubble. The transform collapses single newlines to spaces while preserving paragraph breaks (`\n\n`). System notice cards (if reintroduced) would skip this since their structure depends on hard wraps.
- **Plot-beat email's authored `subject` becomes the bubble eyebrow.** `MessageGroup` renders any email with `beat_id` set as a single-bubble group (no grouping with adjacent bubbles) so the eyebrow reads cleanly. Chat-turn replies (`beat_id: null`) have no eyebrow. This is the visual signal that an authored milestone has landed without breaking the chat metaphor.
- **`unread_counts` is keyed by folder name, not thread id.** Backend response is `{ "Inbox": N }` — for our single-thread game Aria's unread count IS Inbox unread. `ChatSidebar` reads `state.unread_counts.Inbox`.
- **`ChatId = "aria"` is a one-element string union, not a constant.** Leaves shape for a future second chat (a #notices channel, another NPC) without rewriting types. The sidebar already has the `Section` primitive to hold a "Channels" group when needed.
- **Tailwind colors map to CSS variables via the config**, so the same `bg-bg`, `text-text`, `bg-accent` classes work in both themes — no `dark:` prefixes scattered through the components. The two `:root` / `.dark` variable blocks in `index.css` are the only place theme colors are defined.
