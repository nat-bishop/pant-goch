"""Quick concurrent-load benchmark against the resident gemma-4-e4b.

Fires N simultaneous chat completions at a fixed output budget and reports
aggregate completion tokens/sec + per-request latency at each concurrency
level. Throwaway diagnostic — not part of the eval harness.
"""
from __future__ import annotations

import asyncio
import time

import httpx

BASE = "http://127.0.0.1:7770/v1"
MODEL = "gemma-4-e4b"
MAX_TOKENS = 200  # ~ a prose-turn's worth of output

SYS = (
    "You are the narrator of an investigation game set in a Welsh Marches "
    "village. Write atmospheric, grounded prose. Keep NPC voices distinct."
)
USER = (
    "The journalist steps into the rain-soaked vicarage yard at dusk. "
    "Describe what they see and the mood of the place in a single vivid paragraph."
)


async def one_call(client: httpx.AsyncClient) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = await client.post(
        f"{BASE}/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": USER},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 1.0,
            "top_p": 0.95,
        },
        timeout=120.0,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    out_tok = body.get("usage", {}).get("completion_tokens", 0)
    return dt, out_tok


async def run_level(client: httpx.AsyncClient, conc: int) -> None:
    t0 = time.perf_counter()
    results = await asyncio.gather(*(one_call(client) for _ in range(conc)))
    wall = time.perf_counter() - t0
    lats = [dt for dt, _ in results]
    toks = [n for _, n in results]
    total_tok = sum(toks)
    print(
        f"conc={conc:>2} | wall={wall:6.2f}s | "
        f"out_tok={total_tok:>5} | "
        f"agg_tok/s={total_tok / wall:7.1f} | "
        f"per-req lat: min={min(lats):.2f} max={max(lats):.2f} mean={sum(lats)/len(lats):.2f}s"
    )


async def main() -> None:
    async with httpx.AsyncClient() as client:
        # warm-up (model is resident, but prime the path)
        await one_call(client)
        print(f"model={MODEL}  max_tokens={MAX_TOKENS}\n")
        for conc in (1, 2, 4, 8, 16):
            await run_level(client, conc)


if __name__ == "__main__":
    asyncio.run(main())
