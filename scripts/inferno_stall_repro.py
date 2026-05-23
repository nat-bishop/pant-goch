#!/usr/bin/env python
"""Standalone reproduction harness for the inferno mid-stream streaming stall.

Cuts the game engine out entirely and hammers inferno's OpenAI-compatible
`/v1/chat/completions` endpoint directly, mirroring the prose writer's real
request shape (large-ish system prompt, streaming, Gemma sampling
temp=1.0/top_p=0.95/top_k=64, enable_thinking:false). N concurrent workers each
loop streaming completions back-to-back to build sustained load on the KV slots,
exactly like a `make playtest` with `--parallel N`.

The failure we are hunting: a stream that starts emitting tokens and then goes
SILENT mid-response — the connection stays open, no `[DONE]`, no error, no
finish_reason — until something times it out. We detect it directly: if no new
token arrives within --stall-threshold seconds AFTER the first token, the request
is recorded as STALLED (with how many tokens it got and at what wall-clock time),
then abandoned so the worker continues.

Bisection knobs (run these to localise the cause):
  --concurrency K     sweep 1,2,4,8,16,24 to find the threshold (np=16 slots)
  --no-stream         confirm non-streaming completions NEVER stall (isolates
                      the bug to the streaming path)
  --model qwen-3.6-35b-a3b
                      test the vllm backend vs gemma's llama.cpp backend
  --prompt-pad N      pad the system prompt to ~N tokens (prose writer is ~3-4k)
  --max-tokens M      longer generations spend more time mid-stream
  --duration S        sustained run (catches stalls that only appear after the
                      slots have been churning for a while — KV fragmentation /
                      slot leak hypotheses)

Examples:
  # Baseline reproduction at playtest-like load:
  uv run python scripts/inferno_stall_repro.py --concurrency 8 --duration 300

  # Is it streaming-specific? (expect zero stalls)
  uv run python scripts/inferno_stall_repro.py --concurrency 16 --duration 180 --no-stream

  # Concurrency sweep (run each, note where stalls begin):
  for k in 1 2 4 8 16 24; do uv run python scripts/inferno_stall_repro.py --concurrency $k --requests $((k*6)); done

  # Is it the llama.cpp (gemma) backend or vllm (qwen)?
  uv run python scripts/inferno_stall_repro.py --concurrency 8 --duration 180 --model qwen-3.6-35b-a3b

While running, watch the engine side in another terminal:
  docker logs -f inferno-test           # gateway + backend chatter at stall time
  docker stats inferno-test             # GPU/CPU/mem pressure
  curl -s http://127.0.0.1:7770/slots   # llama-server slot state, if proxied
  curl -s http://127.0.0.1:7770/metrics # prometheus metrics, if exposed
Correlate the wall-clock timestamps this script prints for each STALL against the
docker logs to see what the backend was doing at that instant.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_URL = "http://127.0.0.1:7770/v1/chat/completions"

# A chunk of system prompt to approximate the prose writer's context size. Padded
# at runtime to --prompt-pad tokens (~4 chars/token) so we can test whether large
# prompts make the stall worse.
_SYS_BASE = (
    "You are the scene narrator for a single-player investigation game set in a "
    "Welsh Marches village. Write 30-100 words of terse second-person present-tense "
    "prose. Match each character's authored voice; do not invent props or accept "
    "false premises. Atmosphere and small physical detail beat exposition. "
)
_SCENES = [
    "a guarded landlord nursing a pint in a low-ceilinged pub at dusk",
    "a churchyard under an ancient yew, fresh flowers on a small grave",
    "a farm yard with two collies and a watchful woman at the door",
    "a damp vicarage hallway, a sealed letter on the hall table",
    "a post office counter, the postmistress polishing the glass",
    "a single-track lane between high hedges in failing light",
]


@dataclass
class Result:
    worker: int
    ok: bool = False
    stalled: bool = False
    errored: bool = False
    err: str = ""
    ttft: float | None = None          # time to first token (s)
    tokens: int = 0
    max_gap: float = 0.0               # largest inter-token gap (s)
    dur: float = 0.0
    finish: str | None = None
    stalled_at_wall: str = ""          # human clock time the stall was detected


def _sys_prompt(pad_tokens: int) -> str:
    if pad_tokens <= 0:
        return _SYS_BASE
    filler = ("The village remembers what it does not say; the slate-grey light "
              "holds the valley still. ")
    target_chars = pad_tokens * 4
    s = _SYS_BASE
    while len(s) < target_chars:
        s += filler
    return s


def _payload(model: str, sys_prompt: str, max_tokens: int, stream: bool,
             unique: bool = False) -> dict:
    # A random nonce defeats inferno/llama.cpp prefix caching so every request
    # pays full prefill — the real game's contexts are unique and growing, so
    # without this the hammer gets cache hits the game never sees (TTFT ~0.15s
    # even on an 8k prompt is the tell that caching is masking the real cost).
    nonce = f"[ref {random.randrange(10**12)}] " if unique else ""
    sys = nonce + sys_prompt if unique else sys_prompt
    return {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        # mirror the engine's thinking-suppression body (config.THINKING_OFF_EXTRA_BODY)
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user",
             "content": f"{nonce}Write the scene: {random.choice(_SCENES)}."},
        ],
    }


async def _one_stream(
    client: httpx.AsyncClient, url: str, payload: dict, stall_threshold: float,
    worker: int,
) -> Result:
    r = Result(worker=worker)
    t0 = time.perf_counter()
    last = t0
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:200]
                r.errored = True
                r.err = f"HTTP {resp.status_code}: {body}"
                r.dur = time.perf_counter() - t0
                return r
            it = resp.aiter_lines()
            while True:
                try:
                    line = await asyncio.wait_for(it.__anext__(), timeout=stall_threshold)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    # The defining failure: silent mid-stream for > threshold.
                    r.stalled = True
                    r.stalled_at_wall = time.strftime("%H:%M:%S")
                    r.dur = time.perf_counter() - t0
                    return r
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content")
                fr = choices[0].get("finish_reason")
                if delta:
                    now = time.perf_counter()
                    if r.ttft is None:
                        r.ttft = now - t0
                    r.max_gap = max(r.max_gap, now - last)
                    last = now
                    r.tokens += 1
                if fr:
                    r.finish = fr
        r.ok = not r.stalled
        r.dur = time.perf_counter() - t0
        return r
    except Exception as e:  # noqa: BLE001 — diagnostic, report anything
        r.errored = True
        r.err = repr(e)
        r.dur = time.perf_counter() - t0
        return r


async def _one_nostream(
    client: httpx.AsyncClient, url: str, payload: dict, worker: int,
) -> Result:
    r = Result(worker=worker)
    t0 = time.perf_counter()
    try:
        resp = await asyncio.wait_for(client.post(url, json=payload), timeout=90.0)
        r.dur = time.perf_counter() - t0
        if resp.status_code != 200:
            r.errored = True
            r.err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return r
        obj = resp.json()
        ch = (obj.get("choices") or [{}])[0]
        r.finish = ch.get("finish_reason")
        r.tokens = len((ch.get("message") or {}).get("content") or "")
        r.ok = True
        return r
    except asyncio.TimeoutError:
        # No full response within 90s — prefill/scheduling starvation, the
        # same failure as the streaming stall but on the non-streaming path.
        r.stalled = True
        r.stalled_at_wall = time.strftime("%H:%M:%S")
        r.dur = time.perf_counter() - t0
        return r
    except Exception as e:  # noqa: BLE001
        r.errored = True
        r.err = repr(e)
        r.dur = time.perf_counter() - t0
        return r


async def _worker(
    wid: int, client: httpx.AsyncClient, args, sys_prompt: str,
    deadline: float | None, budget: list[int], results: list[Result],
) -> None:
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            return
        if deadline is None:
            if budget[0] <= 0:
                return
            budget[0] -= 1
        payload = _payload(args.model, sys_prompt, args.max_tokens,
                           not args.no_stream, unique=args.unique)
        if args.no_stream:
            res = await _one_nostream(client, args.url, payload, wid)
        else:
            res = await _one_stream(client, args.url, payload, args.stall_threshold, wid)
        results.append(res)
        tag = ("STALL" if res.stalled else "ERR " if res.errored else "ok   ")
        if res.stalled or res.errored:
            print(f"  [{tag}] w{wid:02d} t={res.stalled_at_wall or '-'} "
                  f"tokens={res.tokens} dur={res.dur:.1f}s "
                  f"ttft={res.ttft and round(res.ttft,2)} {res.err}")


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="gemma-4-e4b")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="concurrent workers (= concurrent streams). Sweep to find the threshold.")
    ap.add_argument("--requests", type=int, default=None,
                    help="total requests to send (split across workers). Mutually exclusive with --duration.")
    ap.add_argument("--duration", type=float, default=None,
                    help="run for this many seconds of sustained load instead of a fixed request count.")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--stall-threshold", type=float, default=30.0,
                    help="seconds of mid-stream silence (after first token) before declaring a stall.")
    ap.add_argument("--prompt-pad", type=int, default=900,
                    help="pad system prompt to ~this many tokens (prose writer is ~3-4k; try 0 vs 3500).")
    ap.add_argument("--no-stream", action="store_true",
                    help="non-streaming completions (control: should never stall).")
    ap.add_argument("--unique", action="store_true",
                    help="prepend a random nonce per request to defeat prefix caching "
                         "(mimics the game's unique, cache-missing contexts).")
    ap.add_argument("--dump", default=None, help="optional JSONL path for per-request results.")
    args = ap.parse_args()

    if args.requests is None and args.duration is None:
        args.requests = args.concurrency * 6  # default: ~6 rounds per worker

    sys_prompt = _sys_prompt(args.prompt_pad)
    print(f"== inferno stall repro ==")
    print(f"url={args.url} model={args.model} stream={not args.no_stream} "
          f"concurrency={args.concurrency} max_tokens={args.max_tokens} "
          f"prompt~={args.prompt_pad}tok stall_threshold={args.stall_threshold}s")
    print(f"mode={'duration %.0fs' % args.duration if args.duration else 'requests %d' % args.requests}")
    print("streaming stalls / errors print live below:\n")

    deadline = (time.perf_counter() + args.duration) if args.duration else None
    budget = [args.requests or 0]
    results: list[Result] = []
    limits = httpx.Limits(max_connections=args.concurrency + 4,
                          max_keepalive_connections=args.concurrency + 4)
    # read=None: we never want httpx to time out a read; our wait_for owns stall detection.
    timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=None)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        await asyncio.gather(*[
            _worker(w, client, args, sys_prompt, deadline, budget, results)
            for w in range(args.concurrency)
        ])
    wall = time.perf_counter() - t0

    n = len(results)
    stalled = [r for r in results if r.stalled]
    errored = [r for r in results if r.errored]
    oks = [r for r in results if r.ok]
    ttfts = [r.ttft for r in oks if r.ttft is not None]
    gaps = [r.max_gap for r in oks]
    print("\n===================== SUMMARY =====================")
    print(f"requests:    {n}  in {wall:.1f}s  ({n/wall:.2f} req/s)")
    print(f"  completed: {len(oks)}")
    print(f"  STALLED:   {len(stalled)}   <-- the bug" + (" *** REPRODUCED ***" if stalled else ""))
    print(f"  errored:   {len(errored)}")
    if not args.no_stream and oks:
        print(f"ttft   p50={_pct(ttfts,50):.2f}s p95={_pct(ttfts,95):.2f}s")
        print(f"maxgap p50={_pct(gaps,50):.2f}s p95={_pct(gaps,95):.2f}s  "
              f"(inter-token; near stall_threshold => approaching a stall)")
    fr: dict[str, int] = {}
    for r in oks:
        fr[r.finish or "none"] = fr.get(r.finish or "none", 0) + 1
    print(f"finish_reasons (completed): {fr}")
    if stalled:
        print("\nstalled requests (worker, clock, tokens-before-silence):")
        for r in stalled:
            print(f"  w{r.worker:02d} at {r.stalled_at_wall}  got {r.tokens} tokens "
                  f"then silent >{args.stall_threshold:.0f}s (ttft={r.ttft and round(r.ttft,2)})")
    if errored:
        print("\nerrors:")
        for r in errored[:10]:
            print(f"  w{r.worker:02d}: {r.err}")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.__dict__) + "\n")
        print(f"\nper-request results -> {args.dump}")


if __name__ == "__main__":
    asyncio.run(main())
