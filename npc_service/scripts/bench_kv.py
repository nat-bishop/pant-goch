"""Concurrency cliff benchmark for the engine llama-server.

Drives N concurrent OpenAI-compatible chat completions against a target llama
with a fixed-shape prompt (~5k input tokens, ~400 output tokens), records
per-stream wall time / TTFT / decode tok/s, and reports the aggregate.

Usage:
  uv run python -m scripts.bench_kv --url http://127.0.0.1:8080 --concurrency 1
  uv run python -m scripts.bench_kv --url http://127.0.0.1:8080 --concurrency 64

Each "session" gets a unique prompt body so the per-slot KV demand is not
amortized by shared prefix caching — this is the measurement we care about
for sizing -c against -np × per_session_ctx.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict

import httpx


# Lorem-ish padding generator: produces deterministic text of approx target
# token count (4 chars ≈ 1 token for Gemma in practice). Each session gets a
# unique seed so the resulting prompts differ — no shared prefix to cache.
def _make_prompt_body(target_tokens: int, session_seed: int) -> str:
    rng = random.Random(session_seed)
    words = [
        "candidate", "continuity", "office", "memo", "directive", "register",
        "induction", "calibration", "dossier", "compliance", "cohort", "intake",
        "voice", "alignment", "bargain", "vengeance", "epilogue", "terminal",
        "scene", "beat", "predicate", "scenario", "engine", "spine", "agent",
        "chat", "classifier", "extractor", "summary", "fact", "checker",
        "director", "render", "tick", "decision", "literal", "effect",
        "quality", "threshold", "exit", "fired", "history", "compaction",
    ]
    out: list[str] = []
    target_chars = target_tokens * 4
    while sum(len(w) + 1 for w in out) < target_chars:
        out.append(rng.choice(words))
    return " ".join(out)


SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond to the user with a continuation of "
    "their text, in plain prose. Do not summarize, do not break character. "
    "Keep going for at least 300 words."
)


@dataclass
class StreamResult:
    session: int
    ok: bool
    error: str | None
    wall_s: float
    ttft_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    decode_tok_per_s: float | None  # completion_tokens / (wall - ttft)


async def _one_call(
    client: httpx.AsyncClient,
    url: str,
    session: int,
    prompt_tokens_target: int,
    max_tokens: int,
    timeout_s: float,
) -> StreamResult:
    body = _make_prompt_body(prompt_tokens_target, session)
    payload = {
        "model": "gemma-4-e4b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    try:
        async with client.stream("POST", f"{url}/v1/chat/completions",
                                 json=payload, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                err_body = (await resp.aread()).decode("utf-8", errors="replace")
                return StreamResult(
                    session=session, ok=False,
                    error=f"HTTP {resp.status_code}: {err_body[:200]}",
                    wall_s=time.perf_counter() - t0, ttft_s=None,
                    prompt_tokens=None, completion_tokens=None,
                    decode_tok_per_s=None,
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # First content delta = TTFT (covers prefill + first decode).
                if ttft is None:
                    choices = evt.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content"):
                        ttft = time.perf_counter() - t0
                usage = evt.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        error = f"{type(e).__name__}: {e}"

    wall = time.perf_counter() - t0
    decode_rate: float | None = None
    if (completion_tokens is not None and ttft is not None and wall > ttft):
        decode_rate = completion_tokens / (wall - ttft)

    return StreamResult(
        session=session, ok=error is None,
        error=error, wall_s=wall, ttft_s=ttft,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        decode_tok_per_s=decode_rate,
    )


def _summarize(label: str, results: list[StreamResult]) -> dict:
    ok = [r for r in results if r.ok and r.completion_tokens is not None]
    bad = [r for r in results if not r.ok]
    if not ok:
        return {"label": label, "ok": 0, "bad": len(bad),
                "errors": [r.error for r in bad[:3]]}

    walls = [r.wall_s for r in ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    decode_rates = [r.decode_tok_per_s for r in ok if r.decode_tok_per_s is not None]
    in_toks = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    out_toks = [r.completion_tokens for r in ok if r.completion_tokens is not None]

    def _q(xs, q):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[int(q * (len(xs) - 1))]

    total_wall = max(walls) if walls else 0
    total_out_toks = sum(out_toks)

    return {
        "label": label,
        "ok": len(ok),
        "bad": len(bad),
        "errors": [r.error for r in bad[:3]],
        "avg_in_toks": round(statistics.mean(in_toks), 0) if in_toks else None,
        "avg_out_toks": round(statistics.mean(out_toks), 0) if out_toks else None,
        "wall_p50": round(_q(walls, 0.5), 2),
        "wall_p95": round(_q(walls, 0.95), 2),
        "ttft_p50": round(_q(ttfts, 0.5), 2) if ttfts else None,
        "ttft_p95": round(_q(ttfts, 0.95), 2) if ttfts else None,
        "decode_tok_per_s_p50": round(_q(decode_rates, 0.5), 1) if decode_rates else None,
        "decode_tok_per_s_p95": round(_q(decode_rates, 0.95), 1) if decode_rates else None,
        "decode_tok_per_s_mean": round(statistics.mean(decode_rates), 1) if decode_rates else None,
        "aggregate_total_wall": round(total_wall, 2),
        "aggregate_out_toks": total_out_toks,
        "aggregate_tok_per_s": round(total_out_toks / total_wall, 1) if total_wall > 0 else None,
    }


async def _run(args: argparse.Namespace) -> int:
    timeout = httpx.Timeout(args.timeout, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Warmup: one call to ensure model+server are warm.
        if args.warmup:
            print(f"warmup: 1 call ({args.prompt_tokens} in / max {args.max_tokens} out)...",
                  flush=True)
            await _one_call(client, args.url, session=999_999,
                            prompt_tokens_target=args.prompt_tokens,
                            max_tokens=args.max_tokens, timeout_s=args.timeout)

        results_by_n: list[dict] = []
        for n in args.concurrency:
            print(f"\n=== concurrency={n} ===", flush=True)
            t0 = time.perf_counter()
            tasks = [
                _one_call(client, args.url, session=i,
                          prompt_tokens_target=args.prompt_tokens,
                          max_tokens=args.max_tokens, timeout_s=args.timeout)
                for i in range(n)
            ]
            results = await asyncio.gather(*tasks)
            wall = time.perf_counter() - t0
            summary = _summarize(f"n={n}", results)
            summary["batch_wall_s"] = round(wall, 2)
            results_by_n.append(summary)
            print(json.dumps(summary, indent=2), flush=True)

    print("\n=== full table ===")
    print(json.dumps(results_by_n, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KV-pool concurrency cliff benchmark.")
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="llama-server base URL (no trailing /v1).")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64],
                        help="Concurrency levels to test in sequence.")
    parser.add_argument("--prompt-tokens", type=int, default=5000,
                        help="Target prompt body token count (per session).")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Max output tokens.")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-call timeout in seconds.")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
