"""CLI entry: `uv run python -m scripts.playtest [--seeds N] [--max-sends N]`.

Boots the engine against the local llama, runs every persona × seed
combination through `run_session`, evaluates with pydantic-evals, and
writes a JSON dump to `npc_service/_analysis/playtest_<timestamp>.json`.

Errors out with a clear message if llama-server isn't reachable.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

# Mirrors analyze_scenario.py: ad-hoc invocations need the package on sys.path.
_PKG_SRC = Path(__file__).resolve().parents[2] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from .evaluators import (
    ARIA_VOICE_RUBRIC,
    AriaVoiceJudge,
    BeatCoverage,
    CanonHarvested,
    SessionCompleted,
    TerminalReached,
    _require_evals,
)
from .llm_playtest import (
    boot_engine_for_playtest,
    probe_llama,
    run_session,
    _build_player_agent,
)
from .personas import PersonaSpec, load_personas
from .session import SessionRecord


async def _run_all(
    personas: list[PersonaSpec],
    seeds: int,
    max_sends: int,
    use_evals: bool,
    parallel: int,
) -> tuple[list[SessionRecord], object | None]:
    """Run `seeds` sessions per persona, up to `parallel` at a time. Each
    session uses its own per-call `Session` object — no shared module
    state — so concurrency is safe; the only shared resource is llama-server,
    which must be booted with at least `parallel` concurrent slots
    (`-np N`). If `use_evals`, also build a pydantic-evals Dataset and
    run it (returns the report); otherwise just return raw SessionRecords.
    """
    model = boot_engine_for_playtest()
    player_agent = _build_player_agent(model)

    if not use_evals:
        sem = asyncio.Semaphore(parallel)

        async def _one(persona: PersonaSpec) -> SessionRecord:
            async with sem:
                return await run_session(
                    persona, player_agent=player_agent, max_sends=max_sends,
                )

        cases = [persona for persona in personas for _ in range(seeds)]
        records = await asyncio.gather(*(_one(p) for p in cases))
        return list(records), None

    _require_evals()
    from pydantic_evals import Case, Dataset  # type: ignore[import-not-found]

    cases: list[Case] = []
    for persona in personas:
        for seed in range(seeds):
            cases.append(Case(
                name=f"{persona.archetype}/{persona.name}/seed{seed}",
                inputs=persona,
                metadata={"archetype": persona.archetype, "seed": seed},
            ))

    dataset = Dataset(
        name="continuity_program_playtest",
        cases=cases,
        evaluators=[
            SessionCompleted(),
            TerminalReached(),
            BeatCoverage(),
            CanonHarvested(),
            # Custom judge instead of bare LLMJudge — bare LLMJudge json-dumps
            # the entire SessionRecord (full transcript + tick decisions +
            # final state) into the prompt, which blew the context window.
            AriaVoiceJudge(rubric=ARIA_VOICE_RUBRIC, model=model),
        ],
    )

    async def _eval(persona: PersonaSpec) -> SessionRecord:
        return await run_session(
            persona, player_agent=player_agent, max_sends=max_sends,
        )

    # `run_session` constructs its own Session — no shared module state —
    # so cases can run concurrently up to llama-server's parallel-slot
    # budget (configure with `-np N`).
    report = await dataset.evaluate(_eval, max_concurrency=parallel)
    # Records are inside report; we still want a flat list for the JSON dump.
    records = [c.output for c in report.cases]  # type: ignore[attr-defined]
    return records, report


def _write_json_dump(records: list[SessionRecord], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"playtest_{ts}.json"
    payload = {
        "generated_at": ts,
        "session_count": len(records),
        "sessions": [r.to_json() for r in records],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM playtester for smallville scenarios.",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of sessions per persona (default: 3).",
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
        "--llama-url", default="http://127.0.0.1:8080",
        help="Llama-server base URL for the health probe (default: localhost:8080).",
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="Max sessions to run concurrently (default: 4). Llama-server "
             "must be booted with at least this many parallel slots (`-np N`).",
    )
    args = parser.parse_args(argv)

    if not probe_llama(args.llama_url):
        print(
            f"ERROR: llama-server not reachable at {args.llama_url}. "
            "Start it with `make llama` from the repo root.",
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
          f"{len(personas) * args.seeds} sessions (max_sends={args.max_sends})")

    records, report = asyncio.run(_run_all(
        personas=personas,
        seeds=args.seeds,
        max_sends=args.max_sends,
        use_evals=not args.no_evals,
        parallel=args.parallel,
    ))

    out_dir = Path(__file__).resolve().parents[2] / "_analysis"
    json_path = _write_json_dump(records, out_dir)
    print(f"\nPlaytest JSON: {json_path}")

    errored = [r for r in records if r.error is not None]
    if errored:
        print(f"\n{len(errored)} session(s) errored:")
        for r in errored:
            print(f"  {r.persona_name}: {r.error}")

    if report is not None:
        print("\n" + "=" * 70)
        report.print(include_input=True, include_output=False)  # type: ignore[attr-defined]

    return 0 if not errored else 1


if __name__ == "__main__":
    sys.exit(main())
