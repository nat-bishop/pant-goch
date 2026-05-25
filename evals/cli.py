"""Unified HANDLER eval CLI — one entrypoint for every eval lane.

    uv run python -m evals iso   [--suite deltas,attention] [--k 16]   # fast deterministic gate
    uv run python -m evals play  [--experiment NAME] [--judge]         # full-session corpus / A-B
    uv run python -m evals judge <run-dir>                             # re-judge a recorded corpus

Two real lanes: `iso` (frozen-context per-agent gate, seconds) and `play`
(full simulated sessions, minutes). `judge` is a phase of `play` exposed
standalone for re-judging without re-playing. A/B is not a separate command —
it is `play --experiment <name>`, the multi-arm case of the corpus run.

Eval settings are flags here, defaulting to `Settings` (env / .env), so the
defaults that used to live in `SMALLVILLE_*` env vars are now discoverable via
`--help`. `--inferno/--no-inferno` replaces the `SMALLVILLE_USE_INFERNO=1`
prefix the old make targets carried.
"""
import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import Optional

import typer

# Tag eval spans env=eval BEFORE any handler import (handler.main configures
# Logfire at import). Done here and (idempotently) in each lane module, so the
# tag is correct no matter which is imported first.
os.environ.setdefault("HANDLER_ENV", "eval")

# Windows consoles default to cp1252 and abort the rich report on a non-ascii
# glyph; force UTF-8 on our streams so the report always renders.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8")

app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    help="HANDLER evals — the isolation gate, the playtest corpus, and A/B experiments.",
)


def _set_inferno(inferno: bool) -> None:
    """Select the engine endpoint for this invocation. Settings reads
    SMALLVILLE_USE_INFERNO at construction (inside each lane's main), and an
    explicit env var beats any .env value — so set it before dispatching."""
    os.environ["SMALLVILLE_USE_INFERNO"] = "1" if inferno else "0"


@app.command()
def iso(
    suite: Optional[str] = typer.Option(
        None, help="comma-separated subset of deltas,attention,prose,choices,chronicle (default: all)"),
    k: int = typer.Option(16, help="samples per case (temp=1.0); 16 keeps the 0.8 floor stable"),
    concurrency: int = typer.Option(16, help="max engine/judge calls in flight (= engine KV slots)"),
    out: Optional[Path] = typer.Option(None, help="artifact dir (default: evals/artifacts/isolation/<ts>)"),
    judge_model: Optional[str] = typer.Option(None, help="override Settings.judge_model for the behavioural suites"),
    gate: bool = typer.Option(True, help="exit non-zero on a floor breach (--no-gate to report only)"),
    inferno: bool = typer.Option(True, help="use the inferno gateway endpoint"),
) -> None:
    """Fast deterministic per-agent gate — frozen turn contexts, K samples, CI floor."""
    _set_inferno(inferno)
    from .isolation import __main__ as iso_lane

    asyncio.run(iso_lane.main(
        suite=suite, k=k, concurrency=concurrency, out=out,
        judge_model=judge_model, gate=gate,
    ))


@app.command()
def play(
    experiment: Optional[str] = typer.Option(
        None, help="A/B experiment name (evals/experiments/<name>.yaml); omit for a single-arm playtest"),
    runs: Optional[int] = typer.Option(None, "--runs", "--runs-per-persona",
        help="repeat depth over the persona set (default: Settings.playtest_runs_per_persona)"),
    parallel: Optional[int] = typer.Option(None,
        help="max concurrent sessions = engine KV slots (default: Settings.playtest_parallel)"),
    max_scenes: Optional[int] = typer.Option(None,
        help="scene budget per session before truncation (default: 60, or the experiment's shared.max_scenes)"),
    personas: Optional[str] = typer.Option(None, help="comma-separated subset of persona names"),
    judge: bool = typer.Option(False, help="run the LLM-judge pass inline after the corpus"),
    human: bool = typer.Option(False, help="(A/B only) write blinded pairs for human side-by-side rating"),
    judge_model: Optional[str] = typer.Option(None, help="override Settings.judge_model"),
    out: Optional[Path] = typer.Option(None, help="artifact dir (default: evals/artifacts[/ab]/<ts>)"),
    inferno: bool = typer.Option(True, help="use the inferno gateway endpoint"),
) -> None:
    """Full-session corpus. With --experiment, runs an A/B of its arms + a diff,
    pairwise judging, and common-theme extraction; otherwise a single-arm playtest."""
    _set_inferno(inferno)
    if experiment:
        from . import ab

        asyncio.run(ab.main(
            experiment=experiment, runs_per_persona=runs, parallel=parallel,
            max_scenes=max_scenes, judge=judge, human=human,
            judge_model=judge_model, out=out,
        ))
        return

    if human:
        raise typer.BadParameter("--human applies only with --experiment (needs two arms to compare)")

    from . import judge_run, run

    out_dir = asyncio.run(run.main(
        runs_per_persona=runs, parallel=parallel,
        max_scenes=60 if max_scenes is None else max_scenes,
        personas=personas, out=out,
    ))
    if judge:
        asyncio.run(judge_run.main(run=out_dir, judge_model=judge_model))


@app.command()
def judge(
    run: Path = typer.Argument(..., help="artifact dir to judge (a recorded corpus)"),
    judge_model: Optional[str] = typer.Option(None, help="override Settings.judge_model"),
    max_cases_per_scope: int = typer.Option(400, help="seeded cap on the transition scope; 0 = no cap"),
    seed: int = typer.Option(0, help="RNG seed for sampling (reproducible re-judge)"),
    concurrency: int = typer.Option(64, help="max judge calls in flight per scope"),
    inferno: bool = typer.Option(True, help="use the inferno gateway endpoint (local Qwen judge)"),
) -> None:
    """Re-judge an already-recorded corpus dir — reads dumped artifacts only, no engine."""
    _set_inferno(inferno)
    from . import judge_run

    asyncio.run(judge_run.main(
        run=run, judge_model=judge_model, max_cases_per_scope=max_cases_per_scope,
        seed=seed, concurrency=concurrency,
    ))


@app.command()
def static(
    out: Optional[Path] = typer.Option(None, help="write findings to <dir>/static.json"),
    gate: bool = typer.Option(False, help="exit non-zero if any location is unreachable from start"),
) -> None:
    """Static topology lint — no engine, no LLM. BFS reachability, cut-vertex
    chokepoints (what each gates), and per-beat reach distance over the authored
    map. Catches the 'content gated behind a chokepoint' class at authoring time."""
    from . import static as static_lane

    static_lane.main(out=out, gate=gate)


if __name__ == "__main__":
    app()
