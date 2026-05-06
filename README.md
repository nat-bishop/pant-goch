# Inbox

Single-surface correspondence game styled as a Microsoft Teams–like chat. The player is the protagonist; an AI counterpart (the only NPC) replies on every send. One ongoing "calibration loop" PlotBeat fires on every player /send and classifies the reply into a cumulative `engagement` quality; two ending PlotBeats fire when reply count + engagement thresholds are met.

The current scenario, **"Continuity Program"**, is a corporate-satire pilot. You've been "enrolled" with an AI counterpart — Aria — who is calibrating on your communication style "to assume agreed-upon portions of your role." Severance-flat HR-bright tone. The horror is structural, never named.

## Architecture

```
┌──────────┐  HTTP/SSE  ┌──────────────┐  spawn  ┌──────────────┐
│ Browser  │ ─────────► │ npc_service  │ ──────► │ llama-server │
│ (Vite +  │            │ (FastAPI +   │  /v1/   │ (llama.cpp)  │
│  React)  │ ◄───────── │  Pydantic AI)│ ◄────── │              │
└──────────┘            └──────────────┘         └──────────────┘
```

- **npc_service** owns the deterministic spine (state, beats, scheduler) and runs two Pydantic AI agents:
  - `chat_agent` — `output_type=str`, `deps_type=ChatDeps`, runs once per NPC reply with per-thread `message_history`. Per-run `@instructions` decorators inject the dossier and (when set) an authored plot directive. Used for both default chat turns (`plot_directive=None`) and PlotBeat fires.
  - `classifier_agent` — `ToolOutput`, judges the player's reply against a beat-authored question with an `allowed_values` list, returns one value. Runs only when the replied-to beat authored a `classify_reply` block.
  - Spawns and manages llama-server's lifecycle.
- **Browser** is a Vite+React client styled as a Teams-like chat. Server is the source of truth; the Zustand store is a render cache and refreshes after every mutation.
- **Two player verbs**: `/open` (mark email read; mutates no qualities) and `/send` (reply). Pacing is action-counter driven, not time-driven.
- **Two reply paths**:
  - **Chat turn (default fallback)** — fires immediately within the same `/send` SSE if no PlotBeat is eligible. Free conversation in character; no per-turn authoring.
  - **PlotBeat (authored milestone)** — when a PlotBeat's preconditions hold, it fires as the next chat-agent turn with the spec's intent / must-include / must-not / length window injected as per-run instructions. The `aria.loop` PlotBeat is `multi_turn` and re-fires on every reply (with a `classify_reply` writing the cumulative `engagement` quality); the two terminal PlotBeats (`aria.complete_warm`, `aria.complete_standard`) are `ends_session: true` and mutually exclusive via `Not_(BeatFired)`.

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- Node 18+ with npm
- `llama-server` from llama.cpp (bundled at `vendor/llama.cpp/llama-server.exe`, or built from source with CUDA)
- ~8 GB free disk for the Q8_0 GGUF (auto-downloaded by `make run`)

## Run

```bash
make run     # boots sidecar (spawns llama) + web dev server in parallel
```

Browser opens at http://localhost:5173. First run downloads the model and takes ~90 s for cold load; subsequent runs are fast.

To run layers separately:

```bash
make llama       # terminal 1: llama-server
make sidecar     # terminal 2: Python sidecar (LAUNCH_LLAMA=false)
make web         # terminal 3: Vite dev server
```

The two flags `--jinja` and `--reasoning off` on llama-server are non-negotiable for Gemma 4 E-models — without them you get either an empty response or a 19K-token monologue inside a hidden `<think>` block. The bundled launcher sets both.

## Tests

```bash
make test                              # 44 unit tests (no llama; uses TestModel)
cd npc_service && uv run pytest -m integration  # 6 live-llama tests (needs `make llama` running)
make typecheck                         # tsc on web/
```

Unit tests run by default; integration tests are skipped via `addopts = "-m 'not integration'"` in `pyproject.toml`. The integration suite exercises the real `/v1/chat/completions` and `/tokenize` round-trips — slower (~5-15s per chat turn on Gemma 4 E4B), but the only thing that catches prompt drift, classifier sycophancy, and llama-server endpoint contract changes.

`test_spine.py` covers predicate evaluation (including `BeatFired` and `Not_` for mutual exclusion), beat scheduling, multi-turn beat re-fire, `apply_classification` (LLM-judge → named quality with `set` and `accumulate` modes, clamping), terminal mutual-exclusion via `Not_(BeatFired)` gated on `engagement`, and `reply_subject_for_email`. `test_app.py` boots the FastAPI app with TestModel-overridden agents and asserts: `/open` is read-only; `/send` rejects empty bodies; first /send fires the loop beat; the multi_turn loop re-fires across sends; `thread_messages` accumulates `ModelRequest`/`ModelResponse` across sends; rendered bodies contain no `Subject:`/`From:`/`To:`/JSON/markdown header leakage; the classifier accumulates `engagement` on replies to the loop; `complete_warm` fires + `state.ended` flips at `engagement ≥ 4` and `replies ≥ 5`; `complete_standard` fires when warm is blocked; `/send` returns 409 after session end; chat-agent failures emit a structured `error` SSE event (no torn streams). `test_history_compaction.py` unit-tests the summarise-then-replace history processor in isolation (under-budget noop, short-history noop, over-budget summary splice).

## Smoke-test the loop

After `make run`:

1. The chat opens with Aria typing… then her welcome bubble pops in (welcome typing-replay on first session-open). Theme follows your OS preference; toggle in the top-right if you want to flip.
2. Reply to Aria with anything substantive. Player bubble appears instantly on the right; "Aria is calibrating…" indicator appears at left; her reply pops in after a paced delay (25 chars/sec, clamped 1.5–4.5s).
3. Aria's reply is rendered by the `aria.loop` PlotBeat (the chat-turn fallback never runs in this scenario — the loop's preconditions hold from the first reply). Each reply is classified by the `classifier_agent` and `engagement` accumulates: `engaged → +1.0`, `unclear → +0.5`, `deflected → -0.5`, floor-clamped at 0.
4. After 5 replies AND `engagement ≥ 4.0`, `aria.complete_warm` fires — a warm program-completion email with subject "Pilot Complete — Thank You for Your Partnership". `state.ended` flips to True; `/send` and `/open` then return 409. The chrome shows a "Transition Complete" pill where the wordmark sits.
5. If you've sent 5 replies but `engagement < 4.0`, `aria.complete_standard` fires instead — clinical, parameter-language sign-off.

If the welcome bubble arrives via typing replay, the theme toggle works without a flash, plot-beat replies land with the authored subject as an uppercase eyebrow above the bubble, and the warm vs standard ending matches your reply texture — the loop works.

## Authoring a new scenario

`npc_service/scenario.yaml` is the authoring surface. Schema is validated strictly at sidecar startup (see `npc_service/src/npc_service/models.py`):

- `title`, `premise`
- `qualities`: any classifier-target qualities authored on beats. The Continuity Program scenario uses one — `engagement` (0–10, accumulating) — written by `aria.loop`'s `classify_reply`.
- `npcs`: each with `display_name`, `email_address` (sender address used for chat-turn emails), `relationship_to_owner`, `base_voice_register`, `will_say` / `will_not_say`.
- `threads`: each with optional `npc:` (system threads have `npc: null`). **Beat IDs must be prefixed with their thread ID** — the spine treats the prefix as the source of truth, and the loader validates the match.
- `beats`: tagged on `type:` (`anchored` or `plot`). Common fields: `id`, `thread`, `preconditions` (list of predicates), optional `classify_reply` (one classifier question authored on the beat — runs when the player replies to that beat's email and writes a single named quality), optional `completes_when` (multi-turn beats re-fire until the gate passes).
  - **Anchored** beats add `from`, `to`, `subject`, `authored_prose`, `sent_at`. They ship verbatim. Empty preconditions → fires on the first scheduler tick at boot. Anchored beats on NPC threads also seed the chat agent's `message_history` so subsequent runs see the authored prose as a prior assistant turn.
  - **Plot** beats add `npc`, `subject`, `spec` (with `intent`, `must_include_facts`, `must_not`), and optional `multi_turn: true` (re-fires every tick) / `ends_session: true` (flips `state.ended` after the email is emitted). When fired, the chat agent runs with the spec injected as a per-run instruction; the agent's prose IS the email body, and the authored `subject` overrides the chat-turn-default `Re:` derivation. The frontend renders the subject as an uppercase eyebrow above the bubble.
- `predicates` available: `quality_gte`, `quality_lt`, `player_replies_to_thread_gte`, `beat_fired`, and `not` (use `not` over `beat_fired` to mutually exclude two beats sharing a tick — authoring order is the tiebreaker for same-tick resolution).

Want a thread that's pure free chat with no plot moments? Author the dossier + thread and write zero PlotBeats. Every player /send produces an immediate chat-turn reply driven entirely by dossier + accumulated `message_history`. The conversation can run as long as the player keeps writing.

See `CLAUDE.md` for architectural invariants ("things that look removable but aren't") that AI assistants editing this code should treat as load-bearing.

## Project layout

```
smallville/
├── README.md                 ← this file
├── CLAUDE.md                 ← architecture invariants for AI assistants
├── Makefile
├── npc_service/              ← Python sidecar (uv project)
│   ├── src/npc_service/
│   │   ├── main.py           ← FastAPI app + endpoints + SSE; chat-turn vs PlotBeat dispatch; _fire_beat honors multi_turn + ends_session
│   │   ├── models.py         ← Pydantic schemas (authoring + runtime + DTOs); thread_messages on InboxState; ClassifyReply.accumulate; PlotBeat.multi_turn / .ends_session
│   │   ├── spine.py          ← deterministic state spine; find_due_beats, apply_classification (set/accumulate)
│   │   ├── agents.py         ← chat_agent (str output, deps-driven instructions) + classifier_agent
│   │   ├── launcher.py       ← llama-server subprocess + Job Object
│   │   └── config.py         ← pydantic-settings (ports, llama lifecycle, history compaction)
│   ├── scenario.yaml         ← Continuity Program (1 NPC, 4 beats: welcome + loop + complete_warm + complete_standard)
│   ├── tests/
│   └── pyproject.toml
└── web/                      ← Vite + React frontend (Teams-style chat)
    ├── index.html            ← inline boot script: theme class on <html> before bundle loads (FOUC-free)
    ├── src/
    │   ├── App.tsx, main.tsx
    │   ├── api.ts            ← fetch + SSE wrappers
    │   ├── store.ts          ← Zustand: chatMessages, theme, pendingTypingFor, firstOpenForChat
    │   ├── hooks/             ← useStreamSend (paced reveal), useScrollAnchor, useAutoGrow, useTheme
    │   ├── util/             ← email.ts (stripAddress), avatar.ts (gradient + initials), npc.ts (Aria constants), pacing.ts
    │   └── components/        ← AppShell, AppHeader, ChatSidebar, ChatRow, ChatHeader, MessageList, MessageGroup, MessageBubble, TypingIndicator, NewMessagePill, MessageComposer, ThemeToggle, Avatar, PresenceDot
    ├── tailwind.config.ts    ← darkMode: "class"; semantic tokens via CSS vars; Inter + JetBrains Mono
    ├── package.json
    └── vite.config.ts
```

## Troubleshooting

- **Empty agent responses or 19K-token monologues** — `--jinja` and/or `--reasoning off` missing on llama-server. The bundled launcher sets both; if you run llama-server yourself, include them.
- **`No module named npc_service`** — run `uv sync` from `npc_service/`.
- **Streaming arrives in big bursts** — verify your reverse proxy isn't buffering. The server sets `X-Accel-Buffering: no` on the `/send` SSE response.
- **Port 8000 already in use** — set `NPC_SERVICE_PORT` in `npc_service/.env` and update `web/vite.config.ts`'s proxy target.
- **Aria's reply takes too long to "type"** — the typing-indicator delay is `body.length / 25` clamped to `[1.5s, 4.5s]`. Tune `CHARS_PER_SECOND`, `MIN_DELAY_MS`, `MAX_DELAY_MS` in `web/src/util/pacing.ts`.
- **Theme flashes wrong on first paint** — the inline boot script in `index.html` should run before the bundle. If you've moved theme detection to `main.tsx`, you'll see a 50–200ms flash.
- **Chat turns leak `Subject:` / `From:` headers** — the chat agent's base instructions block this. The body-only regression guard in `test_app.py` covers this. If it fires under a real LLM, tighten `_BASE_CHAT_INSTRUCTIONS` in `agents.py`.
