"""CLI: `uv run python -m scripts.playtest [--seeds N] [--spawn-engine]`.

Drives every persona × seed combination through `run_session` against a
HANDLER sidecar (default http://127.0.0.1:8000), evaluates the resulting
`SessionRecord`s with the programmatic evaluator suite, and writes a JSON
dump to `_analysis/playtest_<timestamp>.json`.

Two engine-lifecycle modes:
- Default (no flag): expects sidecar + llama already running (the two-terminal
  workflow — `make llama-playtest` + `make sidecar` — for fast re-runs while
  iterating).
- `--spawn-engine`: boot a sidecar (which boots llama via its launcher) sized
  for the playtest (LLAMA_N_PARALLEL=16, LLAMA_TOTAL_CTX=524288), wait for
  /health, run, and tear it down on exit. One-terminal workflow used by
  `make playtest`. Reuses an already-healthy engine if one is up.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import subprocess
import sys
import time
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
# Sidecar lifespan blocks on launcher.wait_healthy (cold Q8_0 model load is
# ~90s; cap at 240s to absorb slow-disk worst case). Once /health on :8000
# answers, both llama and the FastAPI surface are ready.
ENGINE_STARTUP_TIMEOUT_SECONDS = 240.0
ENGINE_POLL_INTERVAL_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HANDLER playtest harness")
    p.add_argument("--seeds", type=int, default=2, help="seeds per persona (default 2)")
    p.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    # Default 16 matches `make llama-playtest`'s `-np 16` slot count, so the
    # harness can saturate the engine without queueing. Run `make llama` (`-np 4`)
    # instead and you'll queue past slot 4.
    p.add_argument(
        "--parallel", type=int, default=16,
        help="max concurrent sessions (default 16; pair with `make llama-playtest`)",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="dump dir (default _analysis/)")
    p.add_argument(
        "--spawn-engine", action="store_true",
        help="boot sidecar (+ llama) sized for the playtest, then tear down on exit",
    )
    return p.parse_args()


async def _probe(engine_url: str) -> None:
    async with httpx.AsyncClient(base_url=engine_url, timeout=5.0) as client:
        try:
            r = await client.get("/health")
            if r.status_code != 200:
                raise SystemExit(f"engine /health returned {r.status_code}")
        except httpx.HTTPError as e:
            raise SystemExit(f"engine sidecar unreachable at {engine_url}: {e}")


async def _wait_healthy(
    engine_url: str,
    timeout: float,
    poll: float,
    proc: subprocess.Popen | None = None,
) -> bool:
    """Poll /health until 200, until `timeout`, or until `proc` exits early."""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(base_url=engine_url, timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            if proc is not None and proc.poll() is not None:
                return False  # sidecar died before becoming healthy
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(poll)
    return False


def _spawn_sidecar(log_path: Path) -> subprocess.Popen:
    """Spawn `python -m handler` with playtest-sized llama env.

    LAUNCH_LLAMA stays at its default (True) so the sidecar's lifespan boots
    llama via launcher.start_llama — which enrolls llama in a KILL_ON_JOB_CLOSE
    Job Object, guaranteeing llama dies when this sidecar dies.
    """
    handler_dir = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    # Mirror `make llama-playtest` env so the spawned llama has 16 slots / 512k
    # pool. setdefault so a caller's explicit override (e.g. testing a smaller
    # config) still wins.
    env.setdefault("LLAMA_N_PARALLEL", "16")
    env.setdefault("LLAMA_TOTAL_CTX", "524288")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "handler"],
        cwd=str(handler_dir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
    except Exception:
        pass


async def _ensure_engine(engine_url: str) -> subprocess.Popen | None:
    """Reuse an already-healthy engine, otherwise spawn one. Returns the
    spawned Popen for the caller to terminate on exit, or None if reusing."""
    if await _wait_healthy(engine_url, timeout=1.0, poll=0.5):
        print(f"engine already healthy at {engine_url} — reusing")
        return None

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"playtest-sidecar-{int(time.time())}.log"
    print(f"booting sidecar (llama + handler); sidecar log: {log_path}")

    proc = _spawn_sidecar(log_path)
    try:
        healthy = await _wait_healthy(
            engine_url,
            timeout=ENGINE_STARTUP_TIMEOUT_SECONDS,
            poll=ENGINE_POLL_INTERVAL_SECONDS,
            proc=proc,
        )
        if not healthy:
            died = proc.poll() is not None
            reason = (
                f"sidecar exited early (returncode={proc.returncode})"
                if died else
                f"sidecar did not become healthy within {ENGINE_STARTUP_TIMEOUT_SECONDS:.0f}s"
            )
            raise SystemExit(f"engine boot failed at {engine_url}: {reason}. Check log: {log_path}")
        print(f"engine ready at {engine_url}")
        return proc
    except BaseException:
        _terminate(proc)
        raise


async def _run_case(
    persona: PersonaSpec, seed: int, engine_url: str, sem: asyncio.Semaphore
) -> SessionRecord:
    async with sem:
        return await run_session(persona, seed=seed, engine_base_url=engine_url)


async def amain(args: argparse.Namespace) -> int:
    sidecar_proc: subprocess.Popen | None = None
    if args.spawn_engine:
        sidecar_proc = await _ensure_engine(args.engine_url)
    else:
        await _probe(args.engine_url)
    try:
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

        # Look up authored max_days to bound the budget evaluator.
        mission_path = Path(__file__).resolve().parents[2] / "missions" / "lynx_exfil.yaml"
        mission = load_mission(mission_path)

        cases: list[tuple[SessionRecord, list[EvalResult]]] = []
        by_persona = {p.name: p for p in personas}
        for rec in records:
            persona = by_persona[rec.persona_name]
            evals = run_all(
                rec,
                expected_outcome=persona.expected_outcome,
                max_days=mission.max_days,
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
    finally:
        if sidecar_proc is not None:
            print("shutting down sidecar…")
            _terminate(sidecar_proc)


def main() -> int:
    args = parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
