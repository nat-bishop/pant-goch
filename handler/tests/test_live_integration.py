"""Live integration test against inferno (SMALLVILLE_USE_INFERNO=1) or
bundled llama-server. Skipped by default; run with:

    SMALLVILLE_USE_INFERNO=1 uv run pytest -m integration -v -s

Verifies the M3 scene-runner round-trips against an actual LLM:
  - phase 1 + phase 2 emit valid schemas
  - output_validators don't reject obviously-valid output
  - prose is non-empty + options include a leave option
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

from handler.agents import build_scene_runner_bundle, make_scene_runner_runtime
from handler.config import Settings
from handler.engine import enter_scene
from handler.schemas import ContinueScene, EndGame, EndScene
from handler.scene_loop import run_turn
from handler.state import make_initial_state
from handler.world import load_world


FIXTURE = Path(__file__).parent / "fixtures" / "world_valid"


def _make_model() -> OpenAIChatModel:
    settings = Settings()
    if settings.use_inferno:
        api_url = f"{settings.inferno_url.rstrip('/')}/v1"
    else:
        api_url = (
            settings.llama_base_url
            or f"http://{settings.llama_host}:{settings.llama_port}/v1"
        )
    model_name = "gemma-4-e4b"
    base_profile = openai_model_profile(model_name)
    profile = replace(base_profile, openai_chat_send_back_thinking_parts=False)
    provider = OpenAIProvider(base_url=api_url, api_key="local")
    return OpenAIChatModel(model_name, provider=provider, profile=profile)


@pytest.mark.integration
async def test_live_one_turn_at_pub():
    """Single turn at the pub with Tom present. Asserts prose is non-empty
    and options include at least one leave option (schema-enforced)."""
    world = load_world(FIXTURE)
    bundle = build_scene_runner_bundle(_make_model(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")

    prose_part = await run_turn(
        bundle, runtime, state,
        opening_seed=(
            "The journalist has just walked in. Tom is at the bar, "
            "nursing his bitter."
        ),
    )

    print(f"\n=== PROSE ===\n{prose_part.prose}\n")
    print(f"=== NEXT === {type(prose_part.next).__name__}")
    if isinstance(prose_part.next, ContinueScene):
        for i, opt in enumerate(prose_part.next.options, start=1):
            tag = f" [-> {opt.leaves_for}]" if opt.leaves_for else ""
            print(f"  {i}. {opt.text}{tag}")

    assert prose_part.prose.strip(), "prose should be non-empty"
    assert len(state.dialogue_history) == 1
    if isinstance(prose_part.next, ContinueScene):
        leave_opts = [o for o in prose_part.next.options if o.leaves_for]
        assert leave_opts, "ContinueScene must include ≥1 leave option (schema-enforced)"
