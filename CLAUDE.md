# CLAUDE.md

Single-scenario hostage-negotiation game. Three-process tree:

```
Godot 4.6 (C#) ──spawn──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
              ◄──HTTP/SSE──                                 ◄──OpenAI /v1──
```

`make run` boots the whole tree; `make llama` / `make sidecar` / `make godot` run the layers separately for iteration. Tests: `cd npc_service && uv run pytest` (state mutation, history processors, FastAPI smoke; uses Pydantic-AI `TestModel` instead of llama-server).

## File map

```
npc_service/                  uv project root — pyproject.toml, uv.lock, .venv, scenario.json, .logfire/
  src/npc_service/            installed editable into .venv via `uv sync` (src layout)
    main.py        FastAPI app, captor + options agent wiring, SSE, action-counted timeline, history processors
    models.py      Pydantic scenario schema + CaptorResponse + OptionsResponse + GameState (mutable)
    launcher.py    llama-server subprocess + Windows Job Object for cleanup
    config.py      pydantic-settings Settings (env + .env at workspace root)
  tests/           pytest suite
godot_project/scripts/
  Game.cs              scene controller — boot, action flow, end-of-game overlay
  LLMClient.cs         HTTP + SSE client (autoload)
  SidecarHost.cs       spawns Python + holds the parent Job Object (autoload)
  ActionPanel.cs       tag-button dialogue picker (LLM-generated picks per turn)
  DebugPanel.cs        meters + budget overlay, mounted only when DEBUG_AFFORDANCES=true
  HostageStatusPanel.cs / TimelineFeed.cs   right-side and left-side panels
  Scenario/Captor/Hostage/GameState/TimelineEvent/DialogueOption/ChatMessage.cs   DTOs
```

## Game loop (per turn)

```
client click ─► POST /action {line: hidden_line}
            ◄─ SSE: text deltas, state, [timeline ×N], done
client     ─► GET /options                       (read-only over HISTORY)
            ◄─ JSON: {options: [{tag, hidden_line}, ...]}   2-5 picks
```

First turn: client skips `/action` and goes straight to `GET /options` after boot. Tags are UX-only labels; the chosen option's `hidden_line` is the player input the captor sees. Pacing is **action-counted** (`max_actions` budget; `TimelineEvent.after_actions`) — no wall-clock anywhere in game logic.

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py:_build_command`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside the thinking channel and return empty content. Don't switch to `--reasoning-budget 0` — it produces a different PEG parse error per llama.cpp PR #21697 and doesn't suppress the `thought` rule emission anyway.
- **`OPTIONS_AGENT` uses `PromptedOutput`, not `ToolOutput`.** llama.cpp build b8953's peg-gemma4 tool-call grammar generator emits a duplicated `thought ::= "<|channel>thought" … "<channel|>"` rule (the second copy with its LHS stripped) and crashes GBNF install. The captor's flat schema slips through; OPTIONS's nested-list schema doesn't, regardless of `Field(...)` constraints. PromptedOutput sends no `tools=[…]` → peg-gemma4 isn't activated → schema goes in the prompt and Pydantic AI parses the JSON response client-side. `output_retries=2` covers rare bad-JSON turns.
- **`OptionsResponse` bounds live in `_validate_options`, not in `Field(...)`.** With `PromptedOutput` the model isn't constrained at the sampler — the validator IS the enforcement. Raised `ValueError`s become corrective retries.
- **`ToolOutput(CaptorResponse)` with `max_tokens=512`.** `NativeOutput` (`response_format=json_schema`) was previously shipped but the unbounded `response_text` field cascaded turn-over-turn into 19K-token generations. The `max_tokens` cap also catches non-tool-call prefix prose that Pydantic AI silently discards.
- **`seed_history()` returns `[]`.** A plain-text seeded `ModelResponse` would teach the model to emit text instead of a tool_call. Seed exchanges go in the system prompt as in-prompt examples instead.
- **`HISTORY = …`, `action_count++`, and `apply_response(...)` only run after the stream loop completes** in `_action_event_stream`. `CancelledError` jumps to `finally`, so partial captor turns from interrupted streams are discarded by design.
- **`HttpCompletionOption.ResponseHeadersRead`** on `HttpClient.SendAsync` — without it, the body buffers and SSE breaks.
- **`X-Accel-Buffering: no`** on `/action` and `/events` — same reason, server side.
- **Windows Job Object cleanup** (`SidecarHost.cs`, `launcher.py`). Godot holds a `KILL_ON_JOB_CLOSE` job; Python holds a nested one for llama-server. Only reliable cleanup when Godot is killed abruptly and skips `_ExitTree()`. New subprocesses must inherit the job (`close_fds=True`) or be enrolled, else they orphan.
- **Meters never leave the server in normal play.** Stress/trust/desperation surface in `/state.debug` only when `DEBUG_AFFORDANCES=true`; Godot mounts `DebugPanel` only on that signal. State mutates from the validated `CaptorResponse` struct, never from `response_text` — gibberish prose still progresses the game coherently.
- **`/options` discards `result.all_messages()`.** Writing them back would leak the "Generate options." user-prompt into the captor's history. Shares `ACTION_LOCK` with `/action` for read-coherence with HISTORY mutations.
- **Tags are UX-only.** `DialogueOption.tag` is never read server-side; `hidden_line` IS the player input. Tag → meter rules would compete with the captor LLM's judgment; the captor's reaction *is* the mechanic.
- **Timeline events fire AFTER the captor's N-th turn but BEFORE the next `/options`.** Inline in `_action_event_stream` after `apply_response`; skipped if the turn ended the game. The next options agent sees the synthetic scene-update in HISTORY. Don't move firing into `/options` — that would make a GET have side effects.
- **`OPTIONS_AGENT` sees foreign tool-call shadows in `HISTORY`.** Every captor turn's `ToolCallPart` references `final_result_CaptorResponse`, which isn't in OPTIONS_AGENT's registry. Pydantic AI's hand-off pattern trims the trailing tool call (`messages[:-1]`); we can't, because every captor turn is foreign. Empirically Gemma handles it. Escape hatch if quality regresses: render `HISTORY` as plain `"Negotiator: …"` / `"Captor: …"` text in the user prompt with no `message_history`.

