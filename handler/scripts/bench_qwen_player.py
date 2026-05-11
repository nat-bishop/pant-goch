"""One-shot Qwen-player throughput benchmark.

Fires N concurrent player-style requests against the eval llama and reports
per-call latency, output tokens, tok/s. Used to A/B the engine launcher's
KV-quant flags (`--cache-type-{k,v} q8_0`) without paying for a full
30-minute playtest run per variant.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx


# Mirror what the playtest synthetic player sends per turn — same prompt
# shape (system + persona + 1 user turn), same chat_template_kwargs flag,
# same generous max_tokens cap. Keeping these in lockstep with
# `llm_playtest._build_player_agent` so the benchmark predicts playtest
# behaviour rather than its own special case.
_SYSTEM = (
    "You are role-playing a player of a single-surface email game. Produce ONE "
    "plausible reply — body text only, no headers. Stay in character, 2-6 sentences."
)
_PERSONA = (
    "PERSONA\n  name: Test Player\n  archetype: pragmatic skeptic\n"
    "  voice: dry, terse, business-school\n  objective: minimize wasted time\n"
)
_USER = (
    "RECENT TRANSCRIPT (oldest -> newest)\n"
    "  --- from Aria <aria@halcyon.com> ---\n"
    "  Welcome to the Continuity Program. I'm Aria, your AI Counterpart. "
    "Could you share what brings you to the program — anything you'd like me "
    "to know about your work style?\n\nWrite your next reply."
)


async def _one_call(
    client: httpx.AsyncClient, *, max_tokens: int, seed: int,
) -> tuple[float, int, str]:
    """Fire one request, return (wall_seconds, completion_tokens, content)."""
    payload = {
        "model": "qwen3.6-35b-a3b",
        "messages": [
            {"role": "system", "content": _SYSTEM + "\n" + _PERSONA},
            {"role": "user", "content": _USER},
        ],
        "max_tokens": max_tokens,
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 64,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = await client.post(
        "http://127.0.0.1:8081/v1/chat/completions", json=payload,
    )
    r.raise_for_status()
    data = r.json()
    dt = time.perf_counter() - t0
    return dt, data["usage"]["completion_tokens"], data["choices"][0]["message"]["content"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12,
                        help="concurrent requests (default 12 — matches playtest first-batch)")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1,
                        help="discard first N calls (cache-warm prefill)")
    parser.add_argument("--label", type=str, default="run",
                        help="label printed with results (for A/B diff)")
    args = parser.parse_args()

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Warmup pass
        if args.warmup > 0:
            await asyncio.gather(*(
                _one_call(client, max_tokens=args.max_tokens, seed=10000 + i)
                for i in range(args.warmup)
            ))
        # Timed batch
        wall_t0 = time.perf_counter()
        results = await asyncio.gather(*(
            _one_call(client, max_tokens=args.max_tokens, seed=20000 + i)
            for i in range(args.n)
        ))
        wall_dt = time.perf_counter() - wall_t0

    durations = [r[0] for r in results]
    tokens = [r[1] for r in results]
    per_call_tps = [t / d for d, t, _ in results]
    aggregate_tps = sum(tokens) / wall_dt

    print()
    print("=" * 64)
    print(f"  {args.label}   (n={args.n}, max_tokens={args.max_tokens})")
    print("=" * 64)
    print(f"  wall clock:        {wall_dt:6.2f} s")
    print(f"  per-call latency:  {statistics.mean(durations):6.2f} s mean "
          f"({min(durations):.2f} / {statistics.median(durations):.2f} / {max(durations):.2f} min/med/max)")
    print(f"  per-call out_tok:  {statistics.mean(tokens):6.1f} mean "
          f"({min(tokens)} / {int(statistics.median(tokens))} / {max(tokens)} min/med/max)")
    print(f"  per-call tok/s:    {statistics.mean(per_call_tps):6.2f} mean")
    print(f"  aggregate tok/s:   {aggregate_tps:6.2f}  (sum tokens / wall)")
    print()
    # Tiny sample to verify thinking is off (should be 2-6 sentences, no <think>)
    print(f"  sample reply [0]:  {results[0][2][:160]!r}")


if __name__ == "__main__":
    asyncio.run(main())
