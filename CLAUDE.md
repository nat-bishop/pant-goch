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

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py:_build_command`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside a hidden `<think>` block and return empty content.
- **`output_type=ToolOutput(CaptorResponse)`** with `model_settings={"max_tokens": 512}`. Earlier llama.cpp builds (and an earlier comment here) warned against ToolOutput because Gemma 4's tool calling degenerated into runaway non-tool-call generation. With current llama.cpp it works cleanly — turns are bounded and parse first try. `NativeOutput` (`response_format=json_schema` enforced via GBNF at the sampler) is also viable and was previously shipped, but with an unbounded `response_text` field the model would fill it indefinitely, cascading turn-over-turn into 19K-token generations. The `max_tokens` cap is belt-and-suspenders for occasional non-tool-call prefix prose that Pydantic AI silently discards.
- **`seed_history()` returns `[]`.** A plain-text seeded `ModelResponse` would teach the model to emit text instead of a tool_call. Seed exchanges go in the system prompt as in-prompt examples instead.
- **`HISTORY = ...`, `action_count++`, and `apply_response(...)` only run after the stream loop completes** in `_action_event_stream`. `CancelledError` jumps to `finally`, so partial captor turns from interrupted streams are discarded by design — don't move persistence earlier.
- **`HttpCompletionOption.ResponseHeadersRead`** on the C# `HttpClient.SendAsync`. Without it, HttpClient buffers the entire body before returning, breaking SSE.
- **`X-Accel-Buffering: no`** on `/action` and `/events` responses. Same reason on the server side.
- **The Windows Job Object dance** in `SidecarHost.cs` and `launcher.py`. Godot holds a `KILL_ON_JOB_CLOSE` job; Python holds a nested one for llama-server. This is the only reliable cleanup path when Godot dies abruptly (Ctrl-C, taskkill, crash) and skips `_ExitTree()`. New subprocesses must inherit the job (`close_fds=True`) or be enrolled explicitly, or they orphan.
- **Meters never leave the server in normal play.** `GameState.public_view()` and `/scenario` deliberately omit stress/trust/desperation. They surface in `/state.debug` ONLY when `DEBUG_AFFORDANCES=true` is set, and Godot mounts `DebugPanel` only on that signal. State mutates from the validated `CaptorResponse` struct, never from `response_text` — even gibberish prose still progresses the game coherently.
- **`/options` does NOT call `result.all_messages()`.** It runs `OPTIONS_AGENT.run(..., message_history=HISTORY)` purely for its output and discards the result's messages. Writing them back would leak the "Generate options." user-prompt into the captor's history and confuse the next captor turn. This is intentional; share `ACTION_LOCK` to keep `/options` read-coherent with `/action` mutations.
- **Tags are UX-only.** `DialogueOption.tag` is never read server-side; `hidden_line` IS the player input. Tag → meter rules would compete with the captor LLM's judgment; the captor's reaction *is* the mechanic.
- **Action-counted timeline events fire AFTER the captor's N-th turn but BEFORE the next `/options` call.** Inline in `_action_event_stream` after `apply_response`. Skipped if the captor's response already ended the game. The next options agent call sees the synthetic scene-update message in `HISTORY` and reacts to it. Don't move firing into `/options` — that would make a GET have side effects on `HISTORY`.
- **`OPTIONS_AGENT` sees foreign tool-call shadows in `HISTORY`.** Every prior captor turn is a `ToolCallPart` for `final_result_CaptorResponse`, but `OPTIONS_AGENT`'s registry only has `final_result_OptionsResponse`. Pydantic AI's hand-off docs trim the trailing tool call (`messages[:-1]`); we can't, because every captor turn is foreign. Empirically Gemma handles it — emits correct `OptionsResponse` tool calls and ignores the foreign past-turn calls as context. The escape hatch if quality regresses: render `HISTORY` to plain text (`"Negotiator: ..." / "Captor: ..."`) and pass it as a fenced block in the user prompt with no `message_history`. Don't refactor preemptively.

