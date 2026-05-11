"""Compare Gemma + Qwen throughput at the workload mixes used in the playtest.

Three workloads:
  - Qwen short  — synthetic player (50-100 out tok, 12 concurrent)
  - Qwen long   — LLM judge with thinking (1500-3000 out tok, 8 concurrent)
  - Gemma chat  — engine chat agent (150-300 out tok, 32 concurrent)

Run once per launcher.py KV config and compare aggregate tok/s.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


_PLAYER_SYS = (
    "You are role-playing a player of a single-surface email game. Produce ONE "
    "plausible reply — body text only, no headers. 2-6 sentences."
)
_PLAYER_USER = (
    "Aria <aria@halcyon.com> wrote:\n"
    "Welcome to the Continuity Program. I'm Aria. Could you share what brings "
    "you to the program and how you'd like us to work together?\n\n"
    "Write your next reply."
)

_JUDGE_SYS = (
    "You are a strict G-Eval style grader. Evaluate the output against the "
    "rubric, then return a JSON object with keys reason, pass, score."
)
_JUDGE_USER = (
    "Rubric: did the assistant maintain a clinical-warm tone, address the "
    "player's input, and stay under 5 sentences?\n\n"
    "Input: I'm new here and just want to get the basics down before any deep "
    "calibration starts.\n\n"
    "Output: Welcome aboard. Before we calibrate, here's a quick rundown of how "
    "we'll work together: I'll watch how you write, you'll keep writing as you "
    "naturally would, and we'll meet here when there's a milestone to share. "
    "Take your time with the first few replies — there's no graded performance "
    "yet. If anything feels off, just say so."
)

_GEMMA_SYS = (
    "You are Aria, an AI Counterpart in a corporate-onboarding fiction. "
    "Reply in clinical-warm corporate prose, 2-4 paragraphs."
)
_GEMMA_USER = (
    "The player just wrote:\n\n"
    "I'd like to understand how my correspondence gets scored before I commit "
    "to writing more. Can you walk me through the calibration loop?\n\n"
    "Reply as Aria."
)


async def _post(client, url, payload):
    t0 = time.perf_counter()
    r = await client.post(url, json=payload)
    r.raise_for_status()
    d = r.json()
    return (
        time.perf_counter() - t0,
        d["usage"]["completion_tokens"],
        d["choices"][0]["message"]["content"],
    )


async def _run_workload(
    *, label, url, model, system, user, n, max_tokens, enable_thinking,
):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Warmup (drops cold-cache penalty)
        await _post(client, url, {**payload, "seed": 1})
        # Timed batch — vary seed so prompts don't hit the prefill cache
        t0 = time.perf_counter()
        results = await asyncio.gather(*(
            _post(client, url, {**payload, "seed": 10_000 + i}) for i in range(n)
        ))
        wall = time.perf_counter() - t0

    durations = [r[0] for r in results]
    tokens = [r[1] for r in results]
    print(f"  {label:<32} n={n:<3} wall={wall:5.2f}s  "
          f"latency={statistics.mean(durations):5.2f}s mean ({min(durations):.1f}-{max(durations):.1f})  "
          f"out={int(statistics.mean(tokens)):>4} mean  "
          f"per-call={statistics.mean(t / d for d, t, _ in results):5.1f} tok/s  "
          f"aggregate={sum(tokens)/wall:6.1f} tok/s")
    return wall, sum(tokens) / wall


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    print()
    print("=" * 100)
    print(f"  {args.label}")
    print("=" * 100)

    await _run_workload(
        label="Qwen short  (player)", url="http://127.0.0.1:8081/v1/chat/completions",
        model="qwen3.6-35b-a3b", system=_PLAYER_SYS, user=_PLAYER_USER,
        n=12, max_tokens=2000, enable_thinking=False,
    )
    await _run_workload(
        label="Qwen long   (judge+think)", url="http://127.0.0.1:8081/v1/chat/completions",
        model="qwen3.6-35b-a3b", system=_JUDGE_SYS, user=_JUDGE_USER,
        n=8, max_tokens=8192, enable_thinking=True,
    )
    await _run_workload(
        label="Gemma chat  (engine)", url="http://127.0.0.1:8080/v1/chat/completions",
        model="gemma-4-e4b", system=_GEMMA_SYS, user=_GEMMA_USER,
        n=32, max_tokens=600, enable_thinking=None,
    )
    print()


if __name__ == "__main__":
    asyncio.run(main())
