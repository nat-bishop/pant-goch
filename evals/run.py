"""Playtest harness entrypoint: run many simulated playthroughs in parallel and
score them against the invariant suite.

    uv run python -m evals.run --runs-per-persona 2 --parallel 4

pydantic-evals is the orchestration layer: `Dataset.evaluate(..., max_concurrency,
repeat)` runs the playthroughs concurrently (set max_concurrency to the engine
server's `-np` so demand matches its KV slots) and produces the invariant report
in one pass. The corpus reachability sweep is computed from the dumped artifacts
afterward.

Requires a reachable Gemma engine (inferno or bundled llama, per Settings) and
`GOOGLE_API_KEY` in the environment for the Gemini player.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

# Tag eval spans env:eval before importing handler.main (which configures
# Logfire at import time), so harness runs are filterable out of real plays.
os.environ.setdefault("HANDLER_ENV", "eval")

import yaml
from pydantic_evals import Case, Dataset

from handler.config import Settings
from handler.main import build_model
from handler.world import load_world
from handler.agents import build_scene_runner_bundle

from .evaluators import coverage_report, invariant_evaluators
from .session import SessionArtifact, run_session

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE_MODEL_NAME = "gemma-4-e4b"


def resolve_api_url(settings: Settings) -> str:
    """OpenAI-compatible endpoint for the engine model — mirrors main.lifespan
    and test_live_integration: inferno when opted in, else the bundled llama."""
    if settings.use_inferno:
        return f"{settings.inferno_url.rstrip('/')}/v1"
    return settings.llama_base_url or f"http://{settings.llama_host}:{settings.llama_port}/v1"


def load_personas(subset: list[str] | None = None) -> dict[str, str]:
    data = yaml.safe_load((Path(__file__).parent / "personas.yaml").read_text(encoding="utf-8"))
    if subset:
        missing = [p for p in subset if p not in data]
        if missing:
            raise SystemExit(f"unknown personas: {missing}. available: {sorted(data)}")
        return {p: data[p] for p in subset}
    return data


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HANDLER playtest harness")
    ap.add_argument("--world", type=Path, default=_REPO_ROOT / "world")
    ap.add_argument("--runs-per-persona", type=int, default=4)
    ap.add_argument("--parallel", type=int, default=None,
                    help="max concurrent sessions (default: llama_n_parallel)")
    ap.add_argument("--max-scenes", type=int, default=60,
                    help="scene budget per session before truncation")
    ap.add_argument("--personas", type=str, default=None,
                    help="comma-separated subset of persona names")
    ap.add_argument("--out", type=Path, default=None,
                    help="artifact dir (default: evals/artifacts/<timestamp>)")
    return ap.parse_args()


async def main() -> None:
    args = _parse_args()
    settings = Settings()
    if not settings.gemini_api_key:
        raise SystemExit(
            "no Gemini API key — set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env "
            "or the environment."
        )
    subset = args.personas.split(",") if args.personas else None
    personas = load_personas(subset)
    # Default to running every case at once (personas x repeats); the engine
    # must have enough KV slots (-np) to serve them without queuing.
    parallel = args.parallel or (len(personas) * args.runs_per_persona)
    out_dir = args.out or (_REPO_ROOT / "evals" / "artifacts" / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = resolve_api_url(settings)
    world = load_world(args.world)
    model = build_model(api_url)
    bundle = build_scene_runner_bundle(model, world)  # immutable, shared across sessions

    print(f"playtest: {len(personas)} personas x {args.runs_per_persona} runs "
          f"@ parallel={parallel}; engine={api_url}; player={settings.playtest_player_model}")
    print(f"artifacts -> {out_dir}")

    async def task(persona: str) -> SessionArtifact:
        return await run_session(
            bundle, world,
            persona=persona,
            persona_prompt=personas[persona],
            max_scenes=args.max_scenes,
            out_dir=out_dir,
            model_engine=_ENGINE_MODEL_NAME,
            model_player=settings.playtest_player_model,
            player_api_key=settings.gemini_api_key,
        )

    dataset = Dataset(
        name="handler_playtest",
        cases=[Case(name=p, inputs=p) for p in personas],
        evaluators=invariant_evaluators(world),
    )
    report = await dataset.evaluate(
        task, max_concurrency=parallel, repeat=args.runs_per_persona,
    )
    report.print(include_input=False, include_output=False)

    # Corpus reachability sweep from the dumped artifacts (decoupled from the
    # pydantic-evals report internals).
    artifacts = [SessionArtifact.load(p) for p in sorted(out_dir.glob("*.json"))]
    coverage = coverage_report(artifacts, world)
    print("\ncoverage:")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
