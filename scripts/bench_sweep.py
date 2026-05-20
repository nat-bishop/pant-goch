"""Sweep llama-server --parallel and benchmark concurrent throughput.

For each PARALLEL value: spawn a standalone llama-server (flags matched to
smallville's production launcher), wait healthy, run a fixed concurrency
sweep, tear down. Finds the compute-bound ceiling for gemma-4-E4B on this
GPU so we can pick INFERNO_GEMMA_N_PARALLEL.

Throwaway diagnostic. Runs on a spare port; does not touch inferno.
"""
from __future__ import annotations

import asyncio
import subprocess
import time

import httpx

BIN = r"C:\Users\Nat\Documents\dev\smallville\vendor\llama.cpp\llama-server.exe"
GGUF = r"C:\Users\Nat\Documents\dev\smallville\models\ggml-org-gemma-4-E4B-it-Q8_0.gguf"
PORT = 8090
HOST = "127.0.0.1"
N_CTX = 32768  # per-slot budget, matches production

PARALLEL_VALUES = [4, 8, 16, 32]
CONC_LEVELS = [1, 4, 8, 16, 32, 48]
MAX_TOKENS = 200

SYS = (
    "You are the narrator of an investigation game set in a Welsh Marches "
    "village. Write atmospheric, grounded prose. Keep NPC voices distinct."
)
USER = (
    "The journalist steps into the rain-soaked vicarage yard at dusk. "
    "Describe what they see and the mood of the place in a single vivid paragraph."
)


def spawn(parallel: int) -> subprocess.Popen:
    total_ctx = N_CTX * parallel
    argv = [
        BIN, "-m", GGUF,
        "--host", HOST, "--port", str(PORT),
        "-c", str(total_ctx),
        "-ngl", "99",
        "--flash-attn", "on",
        "--batch-size", "512", "--ubatch-size", "512",
        "--parallel", str(parallel),
        "--jinja",
        "--reasoning", "off",
        "--no-kv-unified",
    ]
    return subprocess.Popen(
        argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


async def wait_healthy(timeout: float = 300.0) -> None:
    url = f"http://{HOST}:{PORT}/health"
    deadline = time.perf_counter() + timeout
    async with httpx.AsyncClient(timeout=2.0) as c:
        while time.perf_counter() < deadline:
            try:
                r = await c.get(url)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("server did not become healthy")


async def one_call(client: httpx.AsyncClient) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = await client.post(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": USER},
            ],
            "max_tokens": MAX_TOKENS, "temperature": 1.0, "top_p": 0.95,
        },
        timeout=180.0,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    out = r.json().get("usage", {}).get("completion_tokens", 0)
    return dt, out


async def bench_level(client: httpx.AsyncClient, conc: int) -> str:
    t0 = time.perf_counter()
    res = await asyncio.gather(*(one_call(client) for _ in range(conc)))
    wall = time.perf_counter() - t0
    lats = [d for d, _ in res]
    tot = sum(n for _, n in res)
    return (
        f"  conc={conc:>2} | wall={wall:6.2f}s | out_tok={tot:>5} | "
        f"agg_tok/s={tot / wall:7.1f} | "
        f"lat min/mean/max={min(lats):.2f}/{sum(lats)/len(lats):.2f}/{max(lats):.2f}s"
    )


async def main() -> None:
    for parallel in PARALLEL_VALUES:
        print(f"\n{'='*72}\n--parallel {parallel}  (-c {N_CTX*parallel})\n{'='*72}", flush=True)
        proc = spawn(parallel)
        try:
            t_load = time.perf_counter()
            await wait_healthy()
            print(f"  [healthy in {time.perf_counter()-t_load:.1f}s]", flush=True)
            async with httpx.AsyncClient() as client:
                await one_call(client)  # warm
                for conc in CONC_LEVELS:
                    print(await bench_level(client, conc), flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            await asyncio.sleep(3)  # let VRAM free before next spawn


if __name__ == "__main__":
    asyncio.run(main())
