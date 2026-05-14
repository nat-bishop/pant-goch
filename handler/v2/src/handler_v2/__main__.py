"""CLI runner for end-to-end M3 verification.

Loads the world from `handler/v2/world/` (or HANDLER_V2_WORLD_PATH),
spawns/probes llama, builds the scene-runner bundle, and runs scenes
interactively in the terminal until the player ends the game.

Usage:
    cd handler/v2
    uv run python -m handler_v2

Prerequisites: a populated world/ (M7) AND either bundled llama-server
binary at vendor/llama.cpp/llama-server OR SMALLVILLE_USE_INFERNO=1 with
inferno running on :7770.

This is a manual-verification harness, not the production entry point —
the FastAPI app + SSE surface lands in M4-M6.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

from .agents import build_scene_runner_bundle
from .config import Settings
from .launcher import resolve_config, start_llama, stop_llama, wait_healthy
from .main import _probe_inferno_health
from .schemas import ContinueScene, EndGame, EndScene, ProsePart
from .scene_loop import play_scene
from .state import make_initial_state
from .world import load_world


async def _build_model(settings: Settings) -> tuple[OpenAIChatModel, object]:
    """Boot llama or probe inferno; build the OpenAIChatModel pointed at
    the resulting endpoint. Returns (model, llama_proc-or-None) so the
    caller can stop_llama on shutdown."""
    cfg = resolve_config(settings)
    llama_proc = None
    if settings.use_inferno:
        await _probe_inferno_health(settings.inferno_url)
        api_url = f"{settings.inferno_url.rstrip('/')}/v1"
    else:
        llama_proc = start_llama(cfg)
        await wait_healthy(host=cfg.host, port=cfg.port)
        api_url = settings.llama_base_url or f"http://{cfg.host}:{cfg.port}/v1"
    model_name = "gemma-4-e4b"
    base_profile = openai_model_profile(model_name)
    # Load-bearing for the conversation-based log writer (M6); set here
    # too so the scene-runner agent matches main.py behavior.
    profile = replace(base_profile, openai_chat_send_back_thinking_parts=False)
    provider = OpenAIProvider(base_url=api_url, api_key="local")
    model = OpenAIChatModel(model_name, provider=provider, profile=profile)
    return model, llama_proc


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
        state = make_initial_state(world)

        print(f"\n=== Scene at {state.player_location} ===")
        opening_seed = world.brief.opening_seed
        while state.game_over is None:
            terminal = await play_scene(
                bundle, state,
                opening_seed=opening_seed,
                pick_option=_prompt_pick,
            )
            opening_seed = None
            _print_prose_part(terminal)
            if isinstance(terminal.next, EndGame):
                ending = world.endings[terminal.next.ending_id]
                print()
                print(ending.epilogue.strip())
                break
            # EndScene — between-scenes; player picks adjacent via map.
            # CLI surrogate: pick first adjacent.
            location = world.locations[state.player_location]
            if not location.adjacent_to:
                print("[stranded — no adjacents]")
                break
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
                    state.player_location = location.adjacent_to[idx]
                    break
            print(f"\n=== Scene at {state.player_location} ===")
        return 0
    finally:
        stop_llama(llama_proc)


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
