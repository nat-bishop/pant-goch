"""Isolation-eval lane — driven by the unified `evals` CLI (`evals iso`).

Builds the engine bundle once (and the judge model only if a behavioural suite is
selected), runs each suite K times per case, merges each suite's pass-rates / judge
means into `<out>/summary.json`, prints the diffed report, and exits non-zero on any
floor breach (unless gate is off). Default out dir is `evals/artifacts/isolation/<ts>`
so the run diffs against prior isolation runs (a trend root separate from the playtest
corpus). Requires a reachable Gemma engine (inferno or bundled llama, per Settings).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Tag spans env=eval before importing handler.main (idempotent with evals.cli).
os.environ.setdefault("HANDLER_ENV", "eval")

from handler.agents import build_scene_runner_bundle
from handler.config import Settings
from handler.main import build_model
from handler.world import load_world

from ..judges.model import build_judge_model, warm_judge
from ..retry_model import RetryModel
from . import attention, behavioural, deltas, free_text_gate
from .harness import run_and_report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALL_SUITES = ("deltas", "attention", "prose", "choices", "chronicle", "free_text_gate")
_NEEDS_JUDGE = frozenset({"prose", "choices"})


def resolve_api_url(settings: Settings) -> str:
    """OpenAI-compatible endpoint for the engine model — mirrors evals.run."""
    if settings.use_inferno:
        return f"{settings.inferno_url.rstrip('/')}/v1"
    return settings.llama_base_url or f"http://{settings.llama_host}:{settings.llama_port}/v1"


async def main(
    *,
    suite: str | None = None,
    k: int = 16,
    concurrency: int = 16,
    out: Path | None = None,
    judge_model: str | None = None,
    gate: bool = True,
) -> None:
    """Run the selected isolation suites (default: all) K times per case, merge
    pass-rates / judge means into `<out>/summary.json`, print the diffed report,
    and (when `gate`) exit non-zero on a floor breach. `suite` is a comma-
    separated subset of `deltas,attention,prose,choices,chronicle,free_text_gate`."""
    selected = [s.strip() for s in suite.split(",")] if suite else list(_ALL_SUITES)
    unknown = [s for s in selected if s not in _ALL_SUITES]
    if unknown:
        raise SystemExit(f"unknown suites: {unknown}. available: {list(_ALL_SUITES)}")

    settings = Settings()
    world = load_world(settings.world_path or (_REPO_ROOT / "world"))
    api_url = resolve_api_url(settings)
    # RetryModel absorbs transient engine 500s (KV/VRAM pressure under concurrent
    # load) so a blip doesn't drop a whole case-run — eval-only, same as evals.run.
    model = RetryModel(build_model(api_url))
    bundle = build_scene_runner_bundle(
        model, world, decision_max_tokens=settings.decision_agent_max_tokens
    )

    judge = None
    if _NEEDS_JUDGE.intersection(selected):
        judge = build_judge_model(settings, override=judge_model)
        warm_judge(settings, override=judge_model)

    factories = {
        "deltas": lambda: deltas.suite(bundle, world),
        "attention": lambda: attention.suite(bundle, world),
        "prose": lambda: behavioural.prose_suite(bundle, world, judge),
        "choices": lambda: behavioural.choices_suite(bundle, world, judge),
        "chronicle": lambda: behavioural.chronicle_suite(bundle, world),
        "free_text_gate": lambda: free_text_gate.suite(bundle, world),
    }
    suites = [factories[s]() for s in selected]

    out_dir = out or (_REPO_ROOT / "evals" / "artifacts" / "isolation"
                      / time.strftime("%Y%m%d-%H%M%S"))
    print(f"isolation: suites={selected} k={k} @ concurrency={concurrency}; "
          f"engine={api_url}"
          + (f"; judge={judge_model or settings.judge_model}" if judge else ""))
    print(f"artifacts -> {out_dir}")

    await run_and_report(
        suites, out_dir=out_dir, k=k, concurrency=concurrency, gate=gate,
    )
