"""Playtest harness: run many simulated playthroughs in parallel and score them
against the invariant suite. Driven by the unified `evals` CLI (`evals play`).

pydantic-evals is the orchestration layer: `Dataset.evaluate(..., max_concurrency,
repeat)` runs the playthroughs concurrently (set max_concurrency to the engine
server's `-np` so demand matches its KV slots) and produces the invariant report
in one pass. The corpus reachability sweep is computed from the dumped artifacts
afterward.

`run_corpus` is the reusable core (one persona × runs sweep into one out dir under
one `RunConfig`); `main` wraps it with the single-arm report + integrity guard. The
A/B harness (`evals.ab`) calls `run_corpus` once per arm.

Requires a reachable Gemma engine (inferno or bundled llama, per Settings) and
`GOOGLE_API_KEY` in the environment for the Gemini player.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Tag eval spans env=eval before importing handler.main (which configures Logfire
# at import). Idempotent with the same setdefault in evals.cli — whichever import
# fires first wins, so the env tag is correct no matter the entry path.
os.environ.setdefault("HANDLER_ENV", "eval")

import yaml
from pydantic_evals import Case, Dataset

from handler.config import Settings
from handler.main import build_model
from handler.scene_loop import RunConfig
from handler.world import World, load_world
from handler.agents import AgentBundle, build_scene_runner_bundle

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


async def run_corpus(
    bundle: AgentBundle,
    world: World,
    *,
    personas: dict[str, str],
    runs_per_persona: int,
    parallel: int,
    max_scenes: int,
    out_dir: Path,
    player_model: str,
    player_api_key: str | None,
    model_engine: str = _ENGINE_MODEL_NAME,
    run_config: RunConfig = RunConfig(),
    arm: str | None = None,
):
    """Run one persona × runs_per_persona sweep into `out_dir` under one
    `RunConfig`, scored against the invariant suite. Returns the pydantic-evals
    invariant report; session artifacts are dumped to `out_dir` (tagged `arm`).

    The reusable core shared by single-arm `main` and the A/B harness — vary
    `run_config` / `arm` / `out_dir` per arm and call once each.

    Concurrency is the engine's KV-slot count, NOT the total session count.
    Total sessions (personas × runs_per_persona) may exceed it; pydantic-evals
    queues the surplus and backfills a fresh session as each finishes, so the
    engine stays saturated instead of decaying to a straggler tail.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    async def task(persona: str) -> SessionArtifact:
        return await run_session(
            bundle, world,
            persona=persona,
            persona_prompt=personas[persona],
            max_scenes=max_scenes,
            out_dir=out_dir,
            model_engine=model_engine,
            model_player=player_model,
            player_api_key=player_api_key,
            run_config=run_config,
            arm=arm,
        )

    dataset = Dataset(
        name="handler_playtest" + (f"_{arm}" if arm else ""),
        cases=[Case(name=p, inputs=p) for p in personas],
        evaluators=invariant_evaluators(world),
    )
    return await dataset.evaluate(
        task, max_concurrency=parallel, repeat=runs_per_persona,
    )


async def main(
    *,
    world_path: Path | None = None,
    runs_per_persona: int | None = None,
    parallel: int | None = None,
    max_scenes: int = 60,
    personas: str | None = None,
    out: Path | None = None,
) -> Path:
    """Single-arm playtest: run the corpus, print the invariant report, write the
    diffed `summary.json`, and guard corpus integrity. Returns the artifact dir (so
    a follow-on judge pass can target it). Settings supply defaults for any None
    argument (so the env-driven defaults still apply)."""
    settings = Settings()
    if not settings.gemini_api_key:
        raise SystemExit(
            "no Gemini API key — set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env "
            "or the environment."
        )
    subset = personas.split(",") if personas else None
    persona_map = load_personas(subset)
    parallel = parallel or settings.playtest_parallel
    runs_per_persona = runs_per_persona or settings.playtest_runs_per_persona
    out_dir = out or (_REPO_ROOT / "evals" / "artifacts" / time.strftime("%Y%m%d-%H%M%S"))

    api_url = resolve_api_url(settings)
    world = load_world(world_path or (_REPO_ROOT / "world"))
    # RetryModel absorbs transient engine 500s (KV/VRAM pressure under concurrent
    # load) so a single blip doesn't drop a whole multi-scene session — eval-only.
    model = RetryModel(build_model(api_url))
    bundle = build_scene_runner_bundle(  # immutable, shared across sessions
        model, world, decision_max_tokens=settings.decision_agent_max_tokens
    )

    print(f"playtest: {len(persona_map)} personas x {runs_per_persona} runs "
          f"@ parallel={parallel}; engine={api_url}; player={settings.playtest_player_model}")
    print(f"artifacts -> {out_dir}")

    invariant_report = await run_corpus(
        bundle, world,
        personas=persona_map,
        runs_per_persona=runs_per_persona,
        parallel=parallel,
        max_scenes=max_scenes,
        out_dir=out_dir,
        player_model=settings.playtest_player_model,
        player_api_key=settings.gemini_api_key,
    )
    invariant_report.print(include_input=False, include_output=False)

    # Corpus distributional metrics from the dumped artifacts (decoupled from
    # the pydantic-evals report internals), then the diffed run report +
    # summary.json — the iteration loop. load_run skips the sidecars
    # (summary.json / judges.json) the report and judge runs drop into the dir.
    artifacts = load_run(out_dir)

    # Corpus-integrity guard: a session that hits an unhandled engine error is
    # dropped by pydantic-evals and never dumped, so metrics quietly compute over
    # survivors. A run that lost a chunk of its sessions is NOT a corpus — flag it
    # loudly. Below the floor we still write the report (forensics) but exit
    # non-zero so `evals play` surfaces the failure.
    expected = len(persona_map) * runs_per_persona
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
    return out_dir
