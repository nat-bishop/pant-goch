# Inbox

Single-surface email correspondence game. The player is the protagonist; an AI counterpart (the only NPC) replies on every send. Authored "plot beats" (calibration milestones, terminal endings) inject one-shot directives onto the chat agent's next turn; the rest is free chat in character. The chain advances on reply counts and on classifier-driven qualities written when the player replies to specific authored beats.

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
- **Browser** is a thin Vite+React client over the FastAPI surface. Server is the source of truth; the Zustand store is a render cache and refreshes after every mutation.
- **Two player verbs**: `/open` (mark email read; does NOT mutate calibration) and `/send` (reply). `/send` advances the `calibration` quality (0–100, the player-visible "Counterpart fidelity %") by a flat `calibration_base` per send. The meter is decorative — no scenario beat gates on its value; reaching 100 is a session-end safety net.
- **Two reply paths**:
  - **Chat turn (default)** — fires immediately within the same `/send` SSE. Free conversation in character; no per-turn authoring.
  - **PlotBeat (authored milestone)** — when a PlotBeat's preconditions cascade (typically `beat_fired` of an upstream beat plus a reply-count threshold), it fires as the next chat-agent turn with the spec's intent / must-include / must-not / length window injected as per-run instructions. Authored ordering is the tiebreaker for mutual exclusion (e.g. `terminal_warm` is authored before `terminal_standard`, gated on `partnership_warmth ≥ 0.7`).

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
make test                 # 46 pytest tests (no llama needed; uses TestModel)
make typecheck            # tsc on web/
```

`test_spine.py` covers predicate evaluation (including `BeatFired` and `Not_` for mutual exclusion), calibration delta + cap, `apply_calibration_per_send` (flat per-send delta), beat scheduling with `runtime_preconditions`, multi-turn beat re-fire and completion, `apply_classification` (LLM-judge → named quality with clamping), terminal mutual-exclusion via `Not_(BeatFired)` gated on `partnership_warmth`, `reply_subject_for_email`, and `latest_npc_email_in_thread`. `test_app.py` boots the FastAPI app with TestModel-overridden agents and asserts: `/open` does NOT mutate calibration; `/send` delta is flat (== `calibration_base`); first /send returns an immediate chat-turn fallback in the same SSE; `thread_messages` accumulates `ModelRequest`/`ModelResponse` across sends; rendered bodies contain no `Subject:`/`From:`/`To:`/JSON/markdown header leakage; the classifier runs on replies to beats that author `classify_reply` (`first_calibration` → `engaged_initial`, `midprogram` → `partnership_warmth`).

## Smoke-test the loop

After `make run`:

1. Inbox shows one email: Aria's "Welcome — Calibration Session 1 of 12". Header pill reads "Counterpart fidelity: 0%".
2. Open the email. Read it. **Fidelity stays at 0%** — reading isn't a communication act; only sends move the meter.
3. Reply with anything. Aria's reply arrives **inside the same SSE** (the chat-turn fallback or the `aria.first_calibration` PlotBeat, depending on how the chain has progressed).
4. `aria.first_calibration` is multi-turn: it re-renders until the classifier judges your reply as `engaged` (mapped to `engaged_initial = 1.0 ≥ 0.7`). Until then the beat keeps asking with rephrased prompts.
5. After first_calibration completes and you've sent ≥3 replies in the thread, `aria.midprogram` fires — subject "Calibration Milestone — Counterpart Fidelity Update", referencing fidelity %, asking a more personal calibration prompt.
6. Reply to the midprogram email. The classifier judges your reply as `warm`/`neutral`/`cold` and writes `partnership_warmth`. After ≥5 replies in the thread, one of the terminals fires: `terminal_warm` if `partnership_warmth ≥ 0.7`, otherwise `terminal_standard`.

If chat turns arrive immediately, the calibration meter only moves on /send (not /open), the chain progresses through `first_calibration → midprogram → terminal`, and the warm vs standard ending matches your reply texture at the midprogram milestone — the loop works.

## Authoring a new scenario

`npc_service/scenario.yaml` is the authoring surface. Schema is validated strictly at sidecar startup (see `npc_service/src/npc_service/models.py`):

- `title`, `premise`
- `qualities`: `calibration` (0–100, the visible "Counterpart fidelity %" doom counter; advances by a flat per-send delta) plus any classifier-target qualities authored on beats (the Continuity Program scenario uses `engaged_initial` and `partnership_warmth`).
- `npcs`: each with `display_name`, `email_address` (sender address used for chat-turn emails), `relationship_to_owner`, `base_voice_register`, `will_say` / `will_not_say`.
- `threads`: each with optional `npc:` (system threads have `npc: null`). **Beat IDs must be prefixed with their thread ID** — the spine treats the prefix as the source of truth, and the loader validates the match.
- `beats`: tagged on `type:` (`anchored` or `plot`). Common fields: `id`, `thread`, `preconditions` (list of predicates), optional `classify_reply` (one classifier question authored on the beat — runs when the player replies to that beat's email and writes a single named quality), optional `completes_when` (multi-turn beats re-fire until the gate passes).
  - **Anchored** beats add `from`, `to`, `subject`, `authored_prose`, `sent_at`. They ship verbatim. Empty preconditions → fires on the first scheduler tick at boot. Anchored beats on NPC threads also seed the chat agent's `message_history` so subsequent runs see the authored prose as a prior assistant turn.
  - **Plot** beats add `npc`, `subject`, `spec` (with `intent`, `must_include_facts`, `must_not`, `length_words`). When fired, the chat agent runs with the spec injected as a per-run instruction; the agent's prose IS the email body, and the authored `subject` overrides the chat-turn-default `Re:` derivation.
- `predicates` available: `quality_gte`, `quality_lt`, `player_replies_to_thread_gte`, `beat_fired`, and `not` (use `not` over `beat_fired` to mutually exclude two beats sharing a tick).

Want a thread that's pure free chat with no plot moments? Author the dossier + thread and write zero PlotBeats. Every player /send produces an immediate chat turn driven entirely by dossier + accumulated `message_history`. The conversation can run as long as the player keeps writing.

See `CLAUDE.md` for architectural invariants ("things that look removable but aren't") that AI assistants editing this code should treat as load-bearing.

## Project layout

```
smallville/
├── README.md                 ← this file
├── CLAUDE.md                 ← architecture invariants for AI assistants
├── Makefile
├── npc_service/              ← Python sidecar (uv project)
│   ├── src/npc_service/
│   │   ├── main.py           ← FastAPI app + endpoints + SSE; chat-turn vs PlotBeat dispatch
│   │   ├── models.py         ← Pydantic schemas (authoring + runtime + DTOs); thread_messages on InboxState
│   │   ├── spine.py          ← deterministic state spine; find_due_beats, apply_classification, apply_calibration_per_send
│   │   ├── agents.py         ← chat_agent (str output, deps-driven instructions) + classifier_agent
│   │   ├── launcher.py       ← llama-server subprocess + Job Object
│   │   └── config.py         ← pydantic-settings (calibration_base)
│   ├── scenario.yaml         ← Continuity Program (1 NPC, 7 beats: 1 welcome + 4 PlotBeats + 2 system anchored)
│   ├── tests/
│   └── pyproject.toml
└── web/                      ← Vite + React frontend
    ├── src/
    │   ├── App.tsx, main.tsx
    │   ├── api.ts            ← fetch + SSE wrappers
    │   ├── store.ts          ← Zustand store
    │   ├── hooks/useStreamSend.ts
    │   ├── util/email.ts     ← stripAddress
    │   └── components/       ← InboxHeader, FolderTabs, InboxList, ThreadView, EmailComposer
    ├── public/fonts/         ← IBM Plex Mono
    ├── package.json
    └── vite.config.ts
```

## Troubleshooting

- **Empty agent responses or 19K-token monologues** — `--jinja` and/or `--reasoning off` missing on llama-server. The bundled launcher sets both; if you run llama-server yourself, include them.
- **`No module named npc_service`** — run `uv sync` from `npc_service/`.
- **Streaming arrives in big bursts** — verify your reverse proxy isn't buffering. The server sets `X-Accel-Buffering: no` on the `/send` SSE response.
- **Port 8000 already in use** — set `NPC_SERVICE_PORT` in `npc_service/.env` and update `web/vite.config.ts`'s proxy target.
- **Aria never replies** — chat-turn fallback should fire on every /send. If it doesn't, the player's /send body is empty (rejected with HTTP 400) or the recipient string doesn't match an NPC's display_name (`_infer_recipient_npc_id` returns None and no chat turn is rendered). Check the calibration event arrived (look for the `calibration` SSE event).
- **Chat turns leak `Subject:` / `From:` headers** — the chat agent's base instructions block this. The body-only regression guard in `test_app.py` covers this. If it fires under a real LLM, tighten `_BASE_CHAT_INSTRUCTIONS` in `agents.py`.
