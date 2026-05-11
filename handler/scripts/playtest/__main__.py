"""CLI: `uv run python -m scripts.playtest [--seeds N]`.

Drives every persona × seed combination through `run_session` against an
already-running HANDLER sidecar (default http://127.0.0.1:8000), evaluates
the resulting `SessionRecord`s with the programmatic evaluator suite, and
writes a JSON dump to `_analysis/playtest_<timestamp>.json`.

Caller is responsible for booting llama-server + sidecar before running.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path

# Allow ad-hoc `python -m scripts.playtest` without `uv sync`.
_PKG_SRC = Path(__file__).resolve().parents[2] / "src"
if _PKG_SRC.is_dir() and str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

import httpx

from handler.mission_loader import load_mission

from .evaluators import EvalResult, run_all
from .personas import PersonaSpec, load_personas
from .report_io import build_payload, print_stdout_summary, write_dump
from .session import SessionRecord, run_session


DEFAULT_ENGINE_URL = "http://127.0.0.1:8000"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HANDLER playtest harness")
    p.add_argument("--seeds", type=int, default=2, help="seeds per persona (default 2)")
    p.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    p.add_argument("--parallel", type=int, default=1, help="max concurrent sessions")
    p.add_argument("--out-dir", type=Path, default=None, help="dump dir (default _analysis/)")
    return p.parse_args()


async def _probe(engine_url: str) -> None:
    async with httpx.AsyncClient(base_url=engine_url, timeout=5.0) as client:
        try:
            r = await client.get("/health")
            if r.status_code != 200:
                raise SystemExit(f"engine /health returned {r.status_code}")
        except httpx.HTTPError as e:
            raise SystemExit(f"engine sidecar unreachable at {engine_url}: {e}")


async def _run_case(
    persona: PersonaSpec, seed: int, engine_url: str, sem: asyncio.Semaphore
) -> SessionRecord:
    async with sem:
        return await run_session(persona, seed=seed, engine_base_url=engine_url)


async def amain(args: argparse.Namespace) -> int:
    await _probe(args.engine_url)
    personas = load_personas()
    if not personas:
        print("no personas found in personas/", file=sys.stderr)
        return 2
    print(f"loaded {len(personas)} persona(s): {[p.name for p in personas]}")

    sem = asyncio.Semaphore(args.parallel)
    tasks = [
        _run_case(p, seed, args.engine_url, sem)
        for p in personas
        for seed in range(args.seeds)
    ]
    started_at = dt.datetime.now()
    print(f"running {len(tasks)} session(s) at parallelism={args.parallel}…")
    records: list[SessionRecord] = await asyncio.gather(*tasks)
    ended_at = dt.datetime.now()

    # Look up authored max_cycles to bound the budget evaluator.
    mission_path = Path(__file__).resolve().parents[2] / "missions" / "lynx_exfil.yaml"
    mission = load_mission(mission_path)

    cases: list[tuple[SessionRecord, list[EvalResult]]] = []
    by_persona = {p.name: p for p in personas}
    for rec in records:
        persona = by_persona[rec.persona_name]
        evals = run_all(
            rec,
            expected_outcome=persona.expected_outcome,
            max_cycles=mission.max_cycles,
        )
        cases.append((rec, evals))

    payload = build_payload(
        cases=cases,
        started_at=started_at,
        ended_at=ended_at,
        cli_args=vars(args),
    )
    out_dir = args.out_dir or (Path(__file__).resolve().parents[2] / "_analysis")
    out_path = out_dir / f"playtest_{started_at:%Y%m%dT%H%M%S}.json"
    write_dump(payload, out_path)
    print_stdout_summary(payload)
    print(f"\nFull dump: {out_path}")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
