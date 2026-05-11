"""CLI entry: `uv run python -m scripts.playtest [--seeds N] [--max-sends N]`.

Boots the engine against the local llama, runs every persona × seed
combination through `run_session`, evaluates with pydantic-evals, and
writes a JSON dump to `handler/_analysis/playtest_<timestamp>.json`.

Errors out with a clear message if either llama-server isn't reachable
(engine on :8080, eval on :8081 — see README.md for the two-server setup).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

# Mirrors analyze_scenario.py: ad-hoc invocations need the package on sys.path.
_PKG_SRC = Path(__file__).resolve().parents[2] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from handler.config import EVAL_LLAMA_BASE_URL, LLAMA_BASE_URL

from .evaluators import (
    AriaRoleAdherence,
    BeatCoverage,
    CanonHarvested,
    ChoiceMeaningfulness,
    ClimaxLanding,
    Closure,
    DirectorAmbiguity,
    EmailBodyShape,
    OfficeToneAdherence,
    OnboardingClarity,
    PacingMilestones,
    PydanticRetryCount,
    SessionCompleted,
    StakesEscalation,
    TerminalReached,
    TransitionMotivation,
    _require_evals,
)
from .llm_playtest import (
    boot_engine_for_playtest,
    probe_llama,
    run_session,
    _build_player_agent,
)
from .personas import PersonaSpec, load_personas
from .report_io import build_payload, print_stdout_summary, write_dump
from .session import SessionRecord


def _kill_engine_llama(engine_port: int = 8080) -> None:
    """Best-effort: kill the engine llama-server holding `engine_port`.
    Run between phase 1 and phase 2 to free the ~8 GB of VRAM (Gemma
    weights + per-slot KV pool) so the eval llama isn't starved when the
    LLM judges fire. Uses the same PowerShell pattern as
    `make kill-zombies` so behaviour is consistent across both shutdown
    paths. Silent if nothing's bound — phase 2 doesn't actually need the
    engine, so a missing engine is fine."""
    try:
        subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"Get-NetTCPConnection -LocalPort {engine_port} -ErrorAction "
                "SilentlyContinue | ForEach-Object { Stop-Process -Id "
                "$_.OwningProcess -Force -ErrorAction SilentlyContinue }",
            ],
            check=False, capture_output=True, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        # Non-Windows / PowerShell missing → just skip; engine staying up
        # is a perf hit, not a correctness issue.
        pass


# Per-session wall-clock cap. Prior runs hung indefinitely when one
# session's engine call entered a degenerate-slow state (191 tokens at
# 339ms/token = 65s/call, compounded by fact-checker retries × 3).
# httpx/pydantic-ai don't enforce a read deadline by default, so a stuck
# session would hold the semaphore forever and starve the whole batch.
# 15 min is enough headroom for the slowest plausible non-pathological
# session (25 sends × ~30s/turn under heavy load) and short enough that
# a hung session surfaces before the operator gives up on the run.
_SESSION_TIMEOUT_S = 15 * 60


async def _phase1_sessions(
    case_specs: list[tuple[PersonaSpec, int]],
    *,
    player_agent,
    max_sends: int,
    parallel: int,
) -> list[SessionRecord]:
    """Phase 1: run every session concurrently (≤parallel at a time).
    Both engine and eval llamas are hot during this phase — the engine
    handles chat/classifier/extractor/director/render, the eval handles
    the synthetic player. Each session uses its own per-call `Session`
    object so concurrency is safe."""
    sem = asyncio.Semaphore(parallel)

    async def _one(persona: PersonaSpec, seed_idx: int) -> SessionRecord:
        async with sem:
            try:
                return await asyncio.wait_for(
                    run_session(
                        persona, player_agent=player_agent, max_sends=max_sends,
                        seed=hash((persona.name, seed_idx)) & 0x7FFFFFFF,
                    ),
                    timeout=_SESSION_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                # Synthesize a record so the rest of the pipeline (evaluators,
                # JSON dump, summary) can still consume the case. Importing here
                # to keep the module-level deps minimal.
                from handler.session import make_session
                import handler.main as engine_main
                assert engine_main.SCENARIO is not None and engine_main.AGENTS is not None
                session = make_session(engine_main.SCENARIO, engine_main.AGENTS)
                return SessionRecord(
                    persona_name=persona.name,
                    transcript=[],
                    tick_decisions=[],
                    final_state=session.state,
                    error=f"session timeout after {_SESSION_TIMEOUT_S}s",
                )

    return list(await asyncio.gather(*(_one(p, s) for p, s in case_specs)))


async def _phase2_evaluators(
    case_specs: list[tuple[PersonaSpec, int]],
    records: list[SessionRecord],
    *,
    eval_model,
    parallel: int,
):
    """Phase 2: score the pre-computed SessionRecords. Wraps the records
    in a pydantic-evals Dataset whose `task` is identity — the actual work
    is in the evaluators. Engine llama is idle (or killed) by this point
    so the LLM judges get the full GPU.

    Single source of truth for evaluator wiring lives here. To add a new
    evaluator, drop it into the `evaluators=[...]` list and re-import."""
    _require_evals()
    from pydantic_evals import Case, Dataset  # type: ignore[import-not-found]

    cases = [
        Case(
            name=f"{persona.archetype}/{persona.name}/seed{seed_idx}",
            inputs=(persona, seed_idx),
            metadata={"archetype": persona.archetype, "seed": seed_idx},
        )
        for persona, seed_idx in case_specs
    ]
    dataset = Dataset(
        name="continuity_program_playtest",
        cases=cases,
        evaluators=[
            SessionCompleted(),
            TerminalReached(),
            BeatCoverage(),
            CanonHarvested(),
            PydanticRetryCount(),
            PacingMilestones(),
            DirectorAmbiguity(),
            EmailBodyShape(),
            AriaRoleAdherence(model=eval_model),
            OfficeToneAdherence(model=eval_model),
            TransitionMotivation(model=eval_model),
            OnboardingClarity(model=eval_model),
            StakesEscalation(model=eval_model),
            ClimaxLanding(model=eval_model),
            Closure(model=eval_model),
            ChoiceMeaningfulness(model=eval_model),
        ],
    )

    # Identity task: look up the pre-computed record. Keyed on
    # (persona.name, seed_idx) — persona objects round-trip cleanly
    # through the Dataset/case input pipeline so name+seed is stable.
    records_by_key = {
        (persona.name, seed_idx): record
        for (persona, seed_idx), record in zip(case_specs, records)
    }

    async def _identity_task(case_input: tuple[PersonaSpec, int]) -> SessionRecord:
        persona, seed_idx = case_input
        return records_by_key[(persona.name, seed_idx)]

    return await dataset.evaluate(_identity_task, max_concurrency=parallel)


async def _run_all(
    personas: list[PersonaSpec],
    seeds: int,
    max_sends: int,
    use_evals: bool,
    parallel: int,
    keep_engine: bool,
) -> tuple[list[SessionRecord], object | None]:
    """Two-phase run: sessions → engine kill → evaluators.

    Phase 1 saturates both llamas (engine + eval-side synthetic player).
    Phase 2 runs pure LLM-judge calls against the eval llama — the engine
    just sits in VRAM doing nothing if `keep_engine=True`, or is killed
    (PowerShell, port-based) to free ~8 GB. The eval llama doesn't
    auto-scale its `-np` after the free, but its compute is no longer
    shared, so per-call throughput rises.
    """
    _engine_model, eval_model = boot_engine_for_playtest()
    player_agent = _build_player_agent(eval_model)

    case_specs: list[tuple[PersonaSpec, int]] = [
        (persona, seed_idx)
        for persona in personas
        for seed_idx in range(seeds)
    ]

    print(f"\n=== Phase 1: {len(case_specs)} sessions, "
          f"≤{parallel}-way concurrent ===", flush=True)
    t0 = time.perf_counter()
    records = await _phase1_sessions(
        case_specs, player_agent=player_agent,
        max_sends=max_sends, parallel=parallel,
    )
    print(f"=== Phase 1 done in {time.perf_counter() - t0:.1f}s ===", flush=True)

    if not use_evals:
        return records, None

    if not keep_engine:
        print("Killing engine llama to free VRAM for phase 2...", flush=True)
        _kill_engine_llama()

    print(f"\n=== Phase 2: evaluators on {len(records)} records, "
          f"≤{parallel}-way concurrent ===", flush=True)
    t0 = time.perf_counter()
    report = await _phase2_evaluators(
        case_specs, records, eval_model=eval_model, parallel=parallel,
    )
    print(f"=== Phase 2 done in {time.perf_counter() - t0:.1f}s ===", flush=True)
    return records, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM playtester for smallville scenarios.",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of sessions per persona (default: 3). Each (persona, "
             "seed_idx) pair maps to a stable llama seed forwarded via "
             "extra_body, so re-running a case produces the same player "
             "inputs (engine agents stay stochastic). 3 is the dev-loop "
             "default; bump to 8+ for ship-readiness gates where variance "
             "coverage matters.",
    )
    parser.add_argument(
        "--max-sends", type=int, default=25,
        help="Cap per-session player /sends (default: 25).",
    )
    parser.add_argument(
        "--no-evals", action="store_true",
        help="Skip pydantic-evals; just run sessions and dump JSON.",
    )
    parser.add_argument(
        "--engine-url", default=LLAMA_BASE_URL.removesuffix("/v1"),
        help="Engine llama-server base URL (default from config: Gemma on :8080).",
    )
    parser.add_argument(
        "--eval-url", default=EVAL_LLAMA_BASE_URL.removesuffix("/v1"),
        help="Eval llama-server base URL (default from config: Qwen on :8081). "
             "Drives the synthetic player and every LLM judge.",
    )
    parser.add_argument(
        "--parallel", type=int, default=16,
        help="Max sessions to run concurrently (default: 16). BOTH llama-servers "
             "must be booted with at least this many parallel slots — the engine "
             "default LLAMA_N_PARALLEL=128 covers it; the eval llama defaults to "
             "EVAL_LLAMA_N_PARALLEL=16 (see config.py for why 16 and not 64). "
             "Raising this above 16 without raising `eval_llama_n_parallel` will "
             "queue requests rather than parallelise them.",
    )
    parser.add_argument(
        "--keep-engine", action="store_true",
        help="Don't kill the engine llama between phase 1 (sessions) and "
             "phase 2 (evaluators). Default behaviour kills it to free "
             "~8 GB of VRAM so the eval llama (Qwen + judges) gets the "
             "full GPU for the LLM-judge pass. Keep alive for debugging "
             "or when you plan to re-run sessions without rebooting llama.",
    )
    args = parser.parse_args(argv)

    # Probe both servers before starting; clear error names which one is missing.
    if not probe_llama(args.engine_url):
        print(
            f"ERROR: engine llama-server not reachable at {args.engine_url}. "
            "Start it with `make llama` from the repo root.",
            file=sys.stderr,
        )
        return 2
    if not probe_llama(args.eval_url):
        print(
            f"ERROR: eval llama-server not reachable at {args.eval_url}. "
            "Start it with `make llama-eval` from the repo root (boots Qwen "
            "3.6 35B-A3B at :8081).",
            file=sys.stderr,
        )
        return 2

    personas = load_personas()
    if not personas:
        print(
            "ERROR: no personas loaded. Check scripts/playtest/personas/.",
            file=sys.stderr,
        )
        return 2
    print(f"Loaded {len(personas)} personas: {[p.name for p in personas]}")
    print(f"Running {len(personas)} × {args.seeds} = "
          f"{len(personas) * args.seeds} sessions "
          f"(max_sends={args.max_sends}, parallel={args.parallel})")

    started_at = dt.datetime.now()
    records, report = asyncio.run(_run_all(
        personas=personas,
        seeds=args.seeds,
        max_sends=args.max_sends,
        use_evals=not args.no_evals,
        parallel=args.parallel,
        keep_engine=args.keep_engine,
    ))
    finished_at = dt.datetime.now()

    # pydantic-evals' rich per-case table first — biggest signal density,
    # and printing it before the JSON write means a slow disk doesn't
    # delay the operator seeing it.
    if report is not None:
        print("\n" + "=" * 70)
        report.print(include_input=True, include_output=False)  # type: ignore[attr-defined]

    payload = build_payload(
        records=records,
        report=report,
        run_args={
            "seeds": args.seeds,
            "max_sends": args.max_sends,
            "parallel": args.parallel,
            "no_evals": args.no_evals,
        },
        # Stable model identifiers — match what `boot_engine_for_playtest`
        # wires into the OpenAIChatModel handles (gemma-4-e4b / qwen3.6-35b-a3b).
        engine_model="gemma-4-e4b",
        eval_model="qwen3.6-35b-a3b",
        scenario_path=str(
            Path(__file__).resolve().parents[2] / "scenario.yaml"
        ),
        started_at=started_at,
        finished_at=finished_at,
    )

    out_dir = Path(__file__).resolve().parents[2] / "_analysis"
    json_path = write_dump(payload, out_dir)

    print_stdout_summary(payload)
    print(f"Playtest JSON: {json_path}")

    errored = [r for r in records if r.error is not None]
    if errored:
        print(f"\n{len(errored)} session(s) errored:")
        for r in errored:
            print(f"  {r.persona_name}: {r.error}")

    return 0 if not errored else 1


if __name__ == "__main__":
    sys.exit(main())
