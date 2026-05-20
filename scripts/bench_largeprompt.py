"""Large-prompt throughput + VRAM probe at parallel=16.

The real game's Phase-1 system prompt is the whole world bible (multi-k
tokens), identical across parallel sessions, with a small varying user
suffix. Earlier benches used a ~50-token prompt and so ignored prefill.

This builds a ~7k-token system prompt, runs a concurrency sweep, and
captures nvidia-smi VRAM while the server is loaded. Runs the sweep twice:
once with --no-kv-unified (production default; per-slot KV, no cross-slot
prefix sharing) and once with --kv-unified (shared prefix KV — should
amortise the identical bible prefill across concurrent sessions).

Throwaway diagnostic. Spare port; does not touch inferno.
"""
from __future__ import annotations

import asyncio
import subprocess
import time

import httpx

BIN = r"C:\Users\Nat\Documents\dev\smallville\vendor\llama.cpp\llama-server.exe"
GGUF = r"C:\Users\Nat\Documents\dev\smallville\models\ggml-org-gemma-4-E4B-it-Q8_0.gguf"
PORT = 8091
HOST = "127.0.0.1"
N_CTX = 32768
PARALLEL = 16
CONC_LEVELS = [1, 4, 8, 16]
MAX_TOKENS = 200

# ~7k-token stand-in for the world bible. Repeated detailed authoring-style
# blocks; identical across all requests so the prefix is cacheable.
_BLOCK = (
    "LOCATION. The vicarage sits at the north edge of Pant Goch, a slate-roofed "
    "Victorian pile behind a yew hedge gone wild. Damp clings to the stone. The "
    "study window faces the churchyard. NOTES: the Reverend kept the side door "
    "unlocked; the gravel path shows two sets of prints the night of the death; "
    "a torn page from the parish register was found in the grate. ADJACENT: "
    "churchyard, lane, river path. CAST. Bryn Pritchard, sexton, sixties, terse, "
    "speaks in short Marches-inflected sentences, knows who walked the lane after "
    "dark, distrusts outsiders, will not name the family directly. Ffion Hughes, "
    "publican, warm but evasive, themes of debt and old loyalty, voice bright and "
    "quick, deflects with hospitality. TRUTH BEATS: investigated_letters requires "
    "the journalist to read the register page; dylan_lied_about_timeline is "
    "blocked_by confronted_bryn; the climax ending requires committed_to_print. "
)
SYS = (_BLOCK * 14).strip()  # ~7k tokens

USERS = [
    "The journalist steps into the rain-soaked vicarage yard at dusk. Describe it.",
    "The journalist presses Bryn about the prints on the gravel. Write his reply.",
    "The journalist reads the torn register page by candlelight. Describe it.",
    "Ffion pours a drink and changes the subject. Write the exchange.",
]


def gpu_mem() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
        return out
    except Exception as e:
        return f"(nvidia-smi failed: {e})"


def spawn(kv_unified: bool) -> subprocess.Popen:
    argv = [
        BIN, "-m", GGUF, "--host", HOST, "--port", str(PORT),
        "-c", str(N_CTX * PARALLEL), "-ngl", "99",
        "--flash-attn", "on", "--batch-size", "512", "--ubatch-size", "512",
        "--parallel", str(PARALLEL), "--jinja", "--reasoning", "off",
        "--kv-unified" if kv_unified else "--no-kv-unified",
    ]
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


async def wait_healthy(timeout: float = 300.0) -> None:
    url = f"http://{HOST}:{PORT}/health"
    deadline = time.perf_counter() + timeout
    async with httpx.AsyncClient(timeout=2.0) as c:
        while time.perf_counter() < deadline:
            try:
                if (await c.get(url)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError("not healthy")


async def one_call(client: httpx.AsyncClient, i: int) -> tuple[float, int, int]:
    t0 = time.perf_counter()
    r = await client.post(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": USERS[i % len(USERS)]},
            ],
            "max_tokens": MAX_TOKENS, "temperature": 1.0, "top_p": 0.95,
        },
        timeout=300.0,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    u = r.json().get("usage", {})
    return dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


async def bench_level(client: httpx.AsyncClient, conc: int) -> str:
    t0 = time.perf_counter()
    res = await asyncio.gather(*(one_call(client, i) for i in range(conc)))
    wall = time.perf_counter() - t0
    lats = [d for d, _, _ in res]
    p_tok = res[0][1]
    out = sum(c for _, _, c in res)
    return (
        f"  conc={conc:>2} | prompt_tok={p_tok:>5} | wall={wall:6.2f}s | "
        f"out_tok={out:>5} | agg_out_tok/s={out / wall:7.1f} | "
        f"lat min/mean/max={min(lats):.2f}/{sum(lats)/len(lats):.2f}/{max(lats):.2f}s"
    )


async def run(kv_unified: bool) -> None:
    tag = "--kv-unified (shared prefix)" if kv_unified else "--no-kv-unified (per-slot, production default)"
    print(f"\n{'='*78}\nparallel={PARALLEL}  {tag}\n{'='*78}", flush=True)
    print(f"  VRAM before spawn: {gpu_mem()}", flush=True)
    proc = spawn(kv_unified)
    try:
        await wait_healthy()
        await asyncio.sleep(1)
        print(f"  VRAM after model load: {gpu_mem()}", flush=True)
        async with httpx.AsyncClient() as client:
            await one_call(client, 0)  # warm + prime prefix cache
            print(f"  VRAM after warm call:  {gpu_mem()}", flush=True)
            for conc in CONC_LEVELS:
                print(await bench_level(client, conc), flush=True)
            print(f"  VRAM at end:           {gpu_mem()}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        await asyncio.sleep(3)


async def main() -> None:
    print(f"system prompt ~{len(SYS)} chars (~{len(SYS)//4} tok est), max_tokens={MAX_TOKENS}")
    await run(kv_unified=False)
    await run(kv_unified=True)


if __name__ == "__main__":
    asyncio.run(main())
