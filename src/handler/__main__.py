"""CLI runner for the text-only playthrough.

Loads the world from `world/` (or `WORLD_PATH`), spawns/probes
llama, builds the scene-runner bundle, and runs scenes interactively in
the terminal until the player ends the game.

Usage:
    uv run python -m handler

Prerequisites: a populated world/ AND either bundled llama-server binary
at vendor/llama.cpp/llama-server OR SMALLVILLE_USE_INFERNO=1 with inferno
running on :7770.

The FastAPI sidecar + SSE surface for the web frontend lives in `main.py`.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel

from .agents import (
    build_scene_runner_bundle,
    make_scene_runner_runtime,
)
from .config import Settings
from .launcher import resolve_config, start_llama, stop_llama, wait_healthy
from .main import _probe_inferno_health, build_model
from .schemas import (
    ChronicleEntry,
    Continue,
    EndGame,
    EndScene,
    FreeText,
    ProsePart,
)
from .scene_loop import (
    PickOption,
    PlayerAction,
    TypeText,
    WalkTo,
    play_session,
)
from .state import State, make_initial_state
from .world import World, load_world


async def _build_model(settings: Settings) -> tuple[OpenAIChatModel, object]:
    """Boot llama or probe inferno; build the OpenAIChatModel pointed at
    the resulting endpoint via the shared `main.build_model` factory.
    Returns (model, llama_proc-or-None) so the caller can stop_llama on
    shutdown."""
    cfg = resolve_config(settings)
    llama_proc = None
    if settings.use_inferno:
        await _probe_inferno_health(settings.inferno_url)
        api_url = f"{settings.inferno_url.rstrip('/')}/v1"
    else:
        llama_proc = start_llama(cfg)
        await wait_healthy(host=cfg.host, port=cfg.port)
        api_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
    return build_model(api_url), llama_proc


def _print_prose_part(prose_part: ProsePart) -> None:
    print()
    print(prose_part.prose.strip())
    print()
    if isinstance(prose_part.next, Continue):
        for i, opt in enumerate(prose_part.next.options, start=1):
            tag = "  (leave)" if opt.leave else ""
            print(f"  {i}) {opt.text}{tag}")
    elif isinstance(prose_part.next, FreeText):
        print(f"[type your reply: {prose_part.next.prompt}]")
        if prose_part.next.suggested:
            print(f"  (suggested: {prose_part.next.suggested})")
    elif isinstance(prose_part.next, EndScene):
        print("[scene ends]")
    elif isinstance(prose_part.next, EndGame):
        print(f"[game ends — {prose_part.next.ending_id}]")


def _request_input(prose_part: ProsePart, state: State, world: World) -> PlayerAction:
    """Single stdin input handler — the CLI's `request_player_input`. Prints
    the turn (prose + options or free-text prompt), then parses one line into a
    `PlayerAction`. A scene is left exactly two ways, mirroring the web UI — no
    always-available walk:

      - the player picks an option the author marked `(leave)`, or the scene
        soft-closes (EndScene); either way we then ask where they go and walk;
      - otherwise a bare number picks an option (Continue) or any text is a typed
        move (FreeText). The adjacency list / destination prompt appear only when
        a walk is actually live.
    """
    _print_prose_part(prose_part)
    adj = world.locations[state.player_location].adjacent_to

    def _parse_walk(raw: str) -> WalkTo | None:
        parts = raw.split()
        if len(parts) == 2 and parts[0].lower() in ("go", "g") and parts[1].isdigit():
            gi = int(parts[1]) - 1
            if 0 <= gi < len(adj):
                return WalkTo(location_id=adj[gi])
        return None

    nxt = prose_part.next
    if isinstance(nxt, Continue):
        n = len(nxt.options)
        while True:
            raw = input(f"Pick 1-{n}: ").strip()
            try:
                idx = int(raw) - 1
            except ValueError:
                print(f"Enter an option number 1-{n}.")
                continue
            if 0 <= idx < n:
                if nxt.options[idx].leave:
                    # Diegetic exit — fall through to the destination prompt so
                    # the player picks where they go (the close resolves there).
                    break
                return PickOption(index=idx)
            print(f"Out of range. Pick 1-{n}.")

    if isinstance(nxt, FreeText):
        label = nxt.prompt.strip() or "Type your reply"
        while True:
            raw = input(f"{label}: ").strip()
            if raw:
                return TypeText(text=raw)
            # Empty line falls back to the suggested move, mirroring the web
            # box's pre-loaded, submittable suggestion.
            if nxt.suggested:
                return TypeText(text=nxt.suggested)
            print("(non-empty)")

    # The player is leaving — an authored `(leave)` pick (fell through above) or
    # a soft-close (EndScene). Surface the adjacents and ask for a destination;
    # a bare number or 'go <n>' both walk there.
    print()
    for i, a in enumerate(adj, start=1):
        print(f"  go {i}) → {world.locations[a].name}")
    while True:
        raw = input(f"Where do you go? 1-{len(adj)} (or 'go <n>'): ").strip()
        walk = _parse_walk(raw)
        if walk is not None:
            return walk
        if raw.isdigit():
            gi = int(raw) - 1
            if 0 <= gi < len(adj):
                return WalkTo(location_id=adj[gi])
        print(f"Enter 1-{len(adj)}.")


def _print_chronicle_entry(state: State, output: ChronicleEntry) -> None:
    print(f"\n--- case-file entry (scene {state.scene_id}) ---")
    print(f"  {output.case_file_entry}")


async def _amain() -> int:
    settings = Settings()
    world_path = settings.world_path or Path(__file__).resolve().parents[2] / "world"
    print(f"Loading world from {world_path}")
    world = load_world(world_path)
    print(f"  {len(world.locations)} locations, {len(world.npcs)} npcs, "
          f"{len(world.beats)} beats, {len(world.endings)} endings")

    print("Booting llama subsystem...")
    model, llama_proc = await _build_model(settings)
    try:
        bundle = build_scene_runner_bundle(
            model, world, decision_max_tokens=settings.decision_agent_max_tokens
        )
        scene_runner = make_scene_runner_runtime()
        state = make_initial_state(world)

        print(f"\n=== Scene at {state.player_location} ===")
        terminal = await play_session(
            bundle, scene_runner, state, world,
            request_player_input=_request_input,
            on_scene_close=_print_prose_part,
            on_chronicle_entry=_print_chronicle_entry,
        )
        if isinstance(terminal.next, EndGame):
            ending = world.endings[terminal.next.ending_id]
            print()
            print(ending.epilogue.strip())
        return 0
    finally:
        stop_llama(llama_proc)


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
