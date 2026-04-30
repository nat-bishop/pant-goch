# CLAUDE.md

Spy-handler game. Single fixed scenario (Operation Iron Threshold). Three field assets in different European cities, one shared op clock, one global decision budget. Three-process tree:

```
Godot 4.6 (C#) ──spawn──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
              ◄──HTTP/SSE──                                 ◄──OpenAI /v1──
```

`make run` boots the whole tree; `make llama` / `make sidecar` / `make godot` run the layers separately. Tests: `cd npc_service && uv run pytest` (state mutation, history processors, FastAPI smoke + per-channel SSE; uses Pydantic-AI `TestModel` instead of llama-server).

## Game premise

Player runs N=3 field agents (MAGPIE/Riga, CICADA/Vienna, DOVE/Lyon) over a shared operation. Reports arrive on the operation's clock — driven by the global `action_count`, not wall-clock. Each agent has a distinct persona; reports are LLM-generated. The handler picks one agent at a time and spends one decision; other channels stay silent that turn but may receive scripted scene-updates from the timeline. Pressure is *who do you spend attention on*. End state is per-agent final state + an after-action report.

## File map

```
npc_service/                  uv project root — pyproject.toml, uv.lock, .venv, scenario.json, .logfire/
  src/npc_service/            installed editable into .venv via `uv sync` (src layout)
    main.py        FastAPI app, N agent agents + options agent wiring, per-channel SSE,
                   action-counted timeline with per-agent injections, history processors
    models.py      Pydantic scenario schema (Operation, AgentDef, TimelineEvent.injections),
                   AgentReport (text + 2 deltas), HandlerState (per-agent dict)
    launcher.py    llama-server subprocess + Windows Job Object for cleanup
    config.py      pydantic-settings Settings (env + .env at workspace root)
  scenario.json    Operation Iron Threshold — 3 agents, 4 timeline events, 18 max_actions
  tests/           pytest suite (39 tests; channel isolation, injection routing, SSE smoke)
godot_project/scripts/
  Game.cs              scene controller — boot, channel switching, per-channel chat cache,
                       FLASH TRAFFIC banner on timeline, after-action overlay
  LLMClient.cs         HTTP + SSE client (autoload). /action and /options keyed by agent_id;
                       SSE events echo agent_id so client routes deltas to the right channel
  SidecarHost.cs       spawns Python + holds the parent Job Object (autoload)
  ActionPanel.cs       raised paper "decision cards" — hover reveals hidden_line,
                       *CONFIRMED* stamp on selection, RECEIVING TRANSMISSION cross-stamp
                       during streaming
  DossierTabs.cs       left-edge filing-cabinet tabs — one per agent. Click to switch
                       channel; emits ChannelSelected. Inactive tabs show ●NEW on unread
  BudgetPunchcard.cs   row of ●/○ dots in the top bar (custom-drawn _Draw)
  DebugPanel.cs        per-agent meters (active channel only), mounted only when
                       DEBUG_AFFORDANCES=true
  TimelineFeed.cs      right column FLASH TRAFFIC log
  HandlerFonts.cs      static font loader (IBM Plex Mono + Stardos Stencil from
                       assets/fonts/)
  Scenario.cs / AgentDef.cs / Operation.cs / HandlerState.cs (with AgentStateView,
    DebugView, AgentDebugView) / TimelineEvent.cs / DialogueOption.cs / ChatMessage.cs
godot_project/assets/fonts/   IBM Plex Mono Regular/Bold + Stardos Stencil Regular/Bold (OFL)
```

## Game loop (per turn)

```
client click in channel X ─► POST /action {agent_id: X, line: hidden_line}
                         ◄─ SSE: {text, agent_id} ×N → {state} → [{timeline} ×M] → {done, agent_id}
client                   ─► GET /options?agent_id=X    (read-only over HISTORIES[X])
                         ◄─ JSON: {options: [{tag, hidden_line}, ...]}   2-5 picks
```

Boot: client renders each agent's `opening_transmission` directly into that channel's chat (no LLM call). Server seeds each `HISTORIES[id]` with the same line as a scene-update so the agent treats it as their own prior context. Then the client calls `/options` for the first channel and presents picks.

Channel switching: zero-cost client-side render of the cached BBCode transcript + a fresh `/options?agent_id=X` for the new channel. No server roundtrip beyond `/options`.

Pacing: **action-counted** (`max_actions` global budget; `TimelineEvent.after_actions` against the same global counter). No wall-clock anywhere in game logic. Reading is free; only successful `/action` increments the counter.

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py:_build_command`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel. Don't switch to `--reasoning-budget 0` — different PEG parse error per llama.cpp PR #21697.
- **`ToolOutput(AgentReport)` with `max_tokens=512`** (per agent). With current llama.cpp tool calling works cleanly; turns are bounded and parse first try. `NativeOutput` was previously shipped on the captor with an unbounded response_text and cascaded turn-over-turn into 19K-token generations.
- **`seed_history(agent)` returns a single scene-update message, not a `ModelResponse`.** A `ModelResponse` with the opening text would teach the model to emit plain text instead of tool-calling structured output. The opening transmission is treated as scene context the agent "said previously." The *client* renders the actual prose into the chat at boot.
- **`HISTORIES[X] = …`, `action_count++`, `apply_report(…)`, and `fire_due_timeline_events(…)` only run after the stream loop completes** in `_action_event_stream`. `CancelledError` jumps to `finally`, so partial agent turns from interrupted streams are discarded by design — no global counter advance, no meter mutation.
- **`HttpCompletionOption.ResponseHeadersRead`** on the C# `HttpClient.SendAsync`. Without it, HttpClient buffers the entire body before returning, breaking SSE.
- **`X-Accel-Buffering: no`** on `/action` and `/events` responses. Same reason on the server side.
- **Windows Job Object cleanup** (`SidecarHost.cs`, `launcher.py`). Godot holds a `KILL_ON_JOB_CLOSE` job; Python holds a nested one for llama-server. Only reliable cleanup path when Godot dies abruptly and skips `_ExitTree()`. New subprocesses must inherit the job (`close_fds=True`) or be enrolled explicitly.
- **Meters never leave the server in normal play.** Per-agent stress/trust surface in `/state.debug.agents[]` only when `DEBUG_AFFORDANCES=true`; Godot mounts `DebugPanel` only on that signal. State mutates from the validated `AgentReport` struct, never from `report_text` — gibberish prose still advances the meters coherently.
- **`/options` discards `result.all_messages()`.** Writing them back would leak the "Generate options for asset X." user prompt into that channel's history and confuse the agent next turn. `ACTION_LOCK` is global across channels for read-coherence with HISTORIES mutations.
- **Action budget is global, not per-agent.** `HANDLER_STATE.action_count` increments by 1 on each successful `/action` regardless of which channel. Timeline events schedule against the same global counter — the operation moves in op-time, not in any single conversation's time.
- **Tags are UX-only.** `DialogueOption.tag` is never read server-side; `hidden_line` IS the handler input the agent sees. Tag → meter rules would compete with the agent LLM's judgment.
- **Timeline events fire per-channel via `TimelineEvent.injections: dict[agent_id, str]`.** Each event can address one or many agents; only addressed channels get scene-updates appended to their HISTORY. Validator at scenario-load enforces every key resolves to a known agent_id. Events fire AFTER the captor's N-th turn but BEFORE the next `/options` — same ordering as the original single-channel game, generalised across N HISTORIES.
- **One `Agent(name=f"agent_{id}", …)` per asset persona.** Each gets its own `summarize_when_large` history processor over its own HISTORY — independent compaction. Logfire spans show per-agent runs distinctly.
- **One shared `OPTIONS_AGENT`, parameterised via the user prompt.** The active agent's callsign + city goes in the per-call prompt (not the system prompt) so the prefix cache stays warm across channel switches. Sharing one agent across channels saves boot cost and keeps Logfire spans simple.
- **`opening_transmission` lives in scenario.json AND is rendered by the client.** The string lives once in the scenario; the server seeds it into HISTORIES as a scene-update; the client GETs `/scenario` and pulls the same string to render into the chat at boot. The agent never speaks the text aloud (no LLM call) — the player just sees it as their first message in that channel.
- **`HandlerState.from_scenario()` keys agents by `id` in a dict.** `for a in state.agents.values()` gives you all agents; `state.agents[agent_id]` indexes one. Don't iterate the scenario's `Scenario.agents` list and look up state by index — agent_id is the stable key across reset.
- **DossierTabs `MarkUnread(agentId)` no-ops on the active channel.** Active is read-by-construction; only inactive channels carry the `●NEW` mark. `SetActive(id)` clears unread for the new active.
