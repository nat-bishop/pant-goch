"""CLI runner for the text-only playthrough.

Loads the world from `handler/world/` (or `WORLD_PATH`), spawns/probes
llama, builds the scene-runner bundle, and runs scenes interactively in
the terminal until the player ends the game.

Usage:
    cd handler
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
    make_log_writer_runtime,
)
from .config import Settings
from .launcher import resolve_config, start_llama, stop_llama, wait_healthy
from .main import _probe_inferno_health, build_model
from .schemas import ContinueScene, EndGame, EndScene, ProsePart
from .scene_loop import play_session
from .state import LogEntry, State, make_initial_state
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
    if isinstance(prose_part.next, ContinueScene):
        for i, opt in enumerate(prose_part.next.options, start=1):
            tag = (
                f" [→ {opt.leaves_for}]" if opt.leaves_for else ""
            )
            print(f"  {i}) {opt.text}{tag}")
    elif isinstance(prose_part.next, EndScene):
        print("[scene ends]")
    elif isinstance(prose_part.next, EndGame):
        print(f"[game ends — {prose_part.next.ending_id}]")


def _prompt_pick(prose_part: ProsePart) -> int:
    assert isinstance(prose_part.next, ContinueScene)
    _print_prose_part(prose_part)
    n = len(prose_part.next.options)
    while True:
        raw = input(f"Pick 1-{n}: ").strip()
        try:
            idx = int(raw) - 1
        except ValueError:
            print("Enter a number.")
            continue
        if 0 <= idx < n:
            return idx
        print(f"Out of range. Pick 1-{n}.")


def _prompt_walk(state: State, world: World) -> str:
    """Between-scenes adjacent picker for the CLI. The web/eval drivers
    supply their own pick_walk callback to play_session; this one reads
    stdin."""
    location = world.locations[state.player_location]
    print()
    for i, adj in enumerate(location.adjacent_to, start=1):
        print(f"  {i}) Walk to {world.locations[adj].name}")
    while True:
        raw = input(f"Pick adjacent 1-{len(location.adjacent_to)}: ").strip()
        try:
            idx = int(raw) - 1
        except ValueError:
            print("Enter a number.")
            continue
        if 0 <= idx < len(location.adjacent_to):
            destination = location.adjacent_to[idx]
            print(f"\n=== Scene at {destination} ===")
            return destination
        print(f"Out of range. Pick 1-{len(location.adjacent_to)}.")


def _print_log_entry(entry: LogEntry) -> None:
    print("\n--- case-file entry ---")
    print(f"  {entry.text}")


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
        bundle = build_scene_runner_bundle(model, world)
        log_writer = make_log_writer_runtime(model, world)
        state = make_initial_state(world)

        print(f"\n=== Scene at {state.player_location} ===")
        terminal = await play_session(
            bundle, log_writer, state, world,
            pick_option=_prompt_pick,
            pick_walk=_prompt_walk,
            opening_seed=world.brief.opening_seed,
            on_scene_close=_print_prose_part,
            on_log_entry=_print_log_entry,
        )
        if isinstance(terminal.next, EndGame):
            ending = world.endings[terminal.next.ending_id]
            print()
            print(ending.epilogue.strip())
        else:
            # Stranded — only EndScene reachable, no adjacents.
            print("[stranded — no adjacents]")
        return 0
    finally:
        stop_llama(llama_proc)


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
