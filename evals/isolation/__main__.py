"""Isolation-eval CLI.

    uv run python -m evals.isolation [--suite deltas,prose,choices,chronicle]
                                     [--k 16] [--concurrency 16] [--out DIR] [--no-gate]

Builds the engine bundle once (and the judge model only if a behavioural suite is
selected), runs each suite K times per case, merges each suite's pass-rates / judge
means into `<out>/summary.json`, prints the diffed report, and exits non-zero on any
floor breach (unless `--no-gate`). Default out dir is `evals/artifacts/isolation/<ts>`
so the run diffs against prior isolation runs (a trend root separate from the playtest
corpus). Requires a reachable Gemma engine (inferno or bundled llama, per Settings).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
from pathlib import Path

# Tag spans env=eval before importing handler.main (which configures Logfire at
# import), so isolation runs are filterable out of real plays.
os.environ.setdefault("HANDLER_ENV", "eval")

# Windows consoles default to cp1252 and abort the report on any non-ascii glyph.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8")

from handler.agents import build_scene_runner_bundle
from handler.config import Settings
from handler.main import build_model
from handler.world import load_world

from ..judges.model import build_judge_model, warm_judge
from ..retry_model import RetryModel
from . import attention, behavioural, deltas
from .harness import run_and_report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALL_SUITES = ("deltas", "attention", "prose", "choices", "chronicle")
_NEEDS_JUDGE = frozenset({"prose", "choices"})


def resolve_api_url(settings: Settings) -> str:
    """OpenAI-compatible endpoint for the engine model — mirrors evals.run."""
    if settings.use_inferno:
        return f"{settings.inferno_url.rstrip('/')}/v1"
    return settings.llama_base_url or f"http://{settings.llama_host}:{settings.llama_port}/v1"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HANDLER isolation-eval lane")
    ap.add_argument("--suite", type=str, default=None,
                    help=f"comma-separated subset of {','.join(_ALL_SUITES)} (default: all)")
    ap.add_argument("--k", type=int, default=16,
                    help="samples per case (temp=1.0). 16 keeps the 0.8 floor stable.")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="max engine/judge calls in flight (= engine KV slots)")
    ap.add_argument("--out", type=Path, default=None,
                    help="artifact dir (default: evals/artifacts/isolation/<timestamp>)")
    ap.add_argument("--judge-model", type=str, default=None,
                    help="override Settings.judge_model for the behavioural suites")
    ap.add_argument("--no-gate", action="store_true",
                    help="report + diff only; never exit non-zero on a floor breach")
    return ap.parse_args()


async def main() -> None:
    args = _parse_args()
    selected = [s.strip() for s in args.suite.split(",")] if args.suite else list(_ALL_SUITES)
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

    judge_model = None
    if _NEEDS_JUDGE.intersection(selected):
        judge_model = build_judge_model(settings, override=args.judge_model)
        warm_judge(settings, override=args.judge_model)

    factories = {
        "deltas": lambda: deltas.suite(bundle, world),
        "attention": lambda: attention.suite(bundle, world),
        "prose": lambda: behavioural.prose_suite(bundle, world, judge_model),
        "choices": lambda: behavioural.choices_suite(bundle, world, judge_model),
        "chronicle": lambda: behavioural.chronicle_suite(bundle, world),
    }
    suites = [factories[s]() for s in selected]

    out_dir = args.out or (_REPO_ROOT / "evals" / "artifacts" / "isolation"
                           / time.strftime("%Y%m%d-%H%M%S"))
    print(f"isolation: suites={selected} k={args.k} @ concurrency={args.concurrency}; "
          f"engine={api_url}"
          + (f"; judge={args.judge_model or settings.judge_model}" if judge_model else ""))
    print(f"artifacts -> {out_dir}")

    await run_and_report(
        suites, out_dir=out_dir, k=args.k, concurrency=args.concurrency,
        gate=not args.no_gate,
    )


if __name__ == "__main__":
    asyncio.run(main())
