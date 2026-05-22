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
import contextlib
import os
import sys
import time
from pathlib import Path

# Tag eval spans env:eval before importing handler.main (which configures
# Logfire at import time), so harness runs are filterable out of real plays.
os.environ.setdefault("HANDLER_ENV", "eval")

# pydantic_evals' rich summary table prints ✔/✗ glyphs; a Windows console
# defaults to cp1252 and aborts the whole report with UnicodeEncodeError. Force
# UTF-8 on our streams so the report (and coverage JSON) always renders.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8")

import yaml
from pydantic_evals import Case, Dataset

from handler.config import Settings
from handler.main import build_model
from handler.world import load_world
from handler.agents import build_scene_runner_bundle

from . import report as run_report
from .evaluators import invariant_evaluators
from .metrics import corpus_metrics
from .retry_model import RetryModel
from .session import SessionArtifact, load_run, run_session

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
                    help="max concurrent sessions = engine KV slots "
                         "(default: settings.playtest_parallel, 16 for inferno); "
                         "total sessions = personas × runs_per_persona may exceed it")
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
    # Concurrency is the engine's KV-slot count, NOT the total session count.
    # Total sessions (personas × runs_per_persona) may exceed this; pydantic-evals
    # queues the surplus and backfills a fresh session as each one finishes, so the
    # engine stays saturated to the end instead of decaying to a straggler tail as
    # short sessions drop out. Raise runs_per_persona for a bigger corpus; raise
    # --parallel / SMALLVILLE_PLAYTEST_PARALLEL only if the engine has more slots.
    parallel = args.parallel or settings.playtest_parallel
    out_dir = args.out or (_REPO_ROOT / "evals" / "artifacts" / time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    api_url = resolve_api_url(settings)
    world = load_world(args.world)
    # RetryModel absorbs transient engine 500s (KV/VRAM pressure under concurrent
    # load) so a single blip doesn't drop a whole multi-scene session — eval-only.
    model = RetryModel(build_model(api_url))
    bundle = build_scene_runner_bundle(  # immutable, shared across sessions
        model, world, decision_max_tokens=settings.decision_agent_max_tokens
    )

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
    invariant_report = await dataset.evaluate(
        task, max_concurrency=parallel, repeat=args.runs_per_persona,
    )
    invariant_report.print(include_input=False, include_output=False)

    # Corpus distributional metrics from the dumped artifacts (decoupled from
    # the pydantic-evals report internals), then the diffed run report +
    # summary.json — the iteration loop.
    # load_run skips the sidecars (summary.json / judges.json) the report and
    # judge runs drop into the dir — a raw glob would feed those non-artifact
    # payloads to SessionArtifact.load and crash, e.g. when --out points at an
    # existing run dir or a prior judge run left judges.json behind.
    artifacts = load_run(out_dir)

    # Corpus-integrity guard: a session that hits an unhandled engine error
    # (e.g. a transient gemma 500 under load) is dropped by pydantic-evals and
    # never dumped, so the metrics quietly compute over the survivors. A run
    # that lost a chunk of its sessions is NOT a corpus — flag it loudly so its
    # numbers aren't mistaken for signal (one run silently shrank to 6/16 and
    # 2/16 this way). Below the floor we still write the report (for forensics)
    # but exit non-zero so `make playtest` surfaces the failure.
    expected = len(personas) * args.runs_per_persona
    survivors = len(artifacts)
    if survivors < expected:
        print(f"\n⚠️  {survivors}/{expected} sessions survived — "
              f"{expected - survivors} dropped (unhandled engine/player error). "
              f"Lower --parallel or check the engine.")

    metrics = corpus_metrics(artifacts, world)
    run_report.emit(out_dir, metrics, invariant_report)

    if survivors < 0.8 * expected:
        raise SystemExit(
            f"corpus too small to trust: {survivors}/{expected} sessions "
            f"(<80%). Metrics above are over survivors only."
        )


if __name__ == "__main__":
    asyncio.run(main())
