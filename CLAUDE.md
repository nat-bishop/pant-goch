# CLAUDE.md

Single-scenario hostage-negotiation game. Three-process tree:

```
Godot 4.6 (C#) ──spawn──► npc_service (FastAPI/Pydantic AI) ──spawn──► llama-server (llama.cpp)
              ◄──HTTP/SSE──                                 ◄──OpenAI /v1──
```

`make run` boots the whole tree; `make llama` / `make sidecar` / `make godot` run the layers separately for iteration. No tests yet.

## File map

```
npc_service/
  main.py        FastAPI app, agent wiring, SSE, timeline scheduler, history processors
  models.py      Pydantic scenario schema + CaptorResponse + GameState (mutable)
  launcher.py    llama-server subprocess + Windows Job Object for cleanup
  scenario.json  the one scenario (validated strictly at startup)
godot_project/scripts/
  Game.cs              scene controller — boot, action flow, end-of-game overlay
  LLMClient.cs         HTTP + SSE client (autoload)
  SidecarHost.cs       spawns Python + holds the parent Job Object (autoload)
  ActionPanel.cs       free-text input row
  HostageStatusPanel.cs / TimelineFeed.cs   right-side and left-side panels
  Scenario/Captor/Hostage/GameState/TimelineEvent/ChatMessage.cs   DTOs
```

## Things that look removable but aren't

- **`--jinja` and `--reasoning off`** on llama-server (`launcher.py:_build_command`). Without `--jinja`, Gemma's chat template is wrong; without `--reasoning off`, Gemma 4 E-models burn the entire generation budget inside a hidden `<think>` block and return empty content.
- **`output_type=NativeOutput(CaptorResponse)`** (not `ToolOutput`). NativeOutput → `response_format=json_schema` → llama-server enforces it via GBNF at the sampler. Pydantic AI's default `ToolOutput` requires Gemma 4 to emit OpenAI tool_calls and degenerates into runaway non-tool-call generation.
- **`seed_history()` returns `[]`.** With a JSON-schema output_type, a plain-text seeded `ModelResponse` would teach the model to emit text instead of JSON. Seed exchanges go in the system prompt as in-prompt examples instead.
- **`HISTORY = ...` and `apply_response(...)` only run after the stream loop completes** in `_action_event_stream`. `CancelledError` jumps to `finally`, so partial captor turns from interrupted streams are discarded by design — don't move persistence earlier.
- **`HttpCompletionOption.ResponseHeadersRead`** on the C# `HttpClient.SendAsync` (`LLMClient.cs:201`). Without it, HttpClient buffers the entire body before returning, breaking SSE.
- **`X-Accel-Buffering: no`** on `/action` and `/events` responses. Same reason on the server side.
- **The Windows Job Object dance** in `SidecarHost.cs` and `launcher.py`. Godot holds a `KILL_ON_JOB_CLOSE` job; Python holds a nested one for llama-server. This is the only reliable cleanup path when Godot dies abruptly (Ctrl-C, taskkill, crash) and skips `_ExitTree()`. New subprocesses must inherit the job (`close_fds=True`) or be enrolled explicitly, or they orphan.
- **Meters never leave the server.** `GameState.public_view()` and `/scenario` deliberately omit stress/trust/desperation. State mutates from the validated `CaptorResponse` struct, never from `response_text` — even gibberish prose still progresses the game coherently.

## Invocation gotcha

`uv run python -m npc_service` from inside `npc_service/` fails with `No module named npc_service` — the package resolves from the parent. Set `PYTHONPATH=..` (the Makefile and `SidecarHost.cs` do this).
