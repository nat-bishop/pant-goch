# Addendum — Truncation root cause, settled from Logfire + artifacts

**Date:** 2026-05-23 (follow-up to `FINDINGS-eval-judge-remediation.md` §4)
**Method:** Logfire traces (`smallville` project) + the truncated session artifacts themselves.
**Verdict:** The §4 framing ("intermittent engine stall / prefill-starvation **hang**") is **wrong**.
The truncation is **wall-clock starvation**: deep-exploration sessions produce 47–57 *real,
progressing* turns at ~16 s/turn under concurrent load and run out the fixed 900 s session
ceiling before committing to an ending. There is **no hang.**

---

## 1. What the evidence shows

### 1.1 Truncated sessions are full of distinct, progressing turns — not stuck, not hung
Every truncated session in all three "dirty" runs has a **high** turn count, never a low one:

| Run | Session | turns | scenes | dur | ending |
|---|---|---|---|---|---|
| 140146 | committed_closer | 51 | 5 | 900.0 | None |
| 140146 | diligent_journalist | 55 | 4 | 900.0 | None |
| 140146 | distracted_skimmer | 57 | 6 | 900.0 | None |
| 142042 | committed_closer | 56 | 5 | 900.0 | None |
| 142042 | distracted_skimmer | 56 | 5 | 900.0 | None |
| 144934 | aggressive_interrogator | 47 | 1 | 900.0 | None |
| 144934 | diligent_journalist | 56 | 4 | 900.0 | None |

A true single-turn hang would leave a session with **few** turns + 900 s (it hung early). **Zero**
truncated sessions look like that. The tail of `diligent_journalist` (144934) is distinct, coherent
prose moving across `st_caradocs → the_lane` — i.e. the player was *still actively exploring* the
back-half map at the 900 s cut. The session was producing turns the whole time.

### 1.2 The "clean" run finished *right at the cliff*
Run 153538 (NotTruncated 1.0): the three deep sessions ended at **863 / 873 / 887 s** — i.e. they
crossed the finish line with 13–37 s to spare under the 900 s wall. The budget is not comfortable;
it is **at the edge.** A few extra seconds/turn or a couple extra exploration turns is the whole
difference between "clean" and "truncated." That is the "intermittency" of §4.2 (0.25/0.5/0.5/1.0):
a **threshold race**, not a Bernoulli hang.

### 1.3 No span anywhere is consistent with a hang
Across **the entire day** in Logfire (`smallville`):
- max `turn` span = **28.5 s**; max single `chat gemma-4-e4b` call = **67.3 s**.
- The only multi-hundred-second spans are `GET /events` (the web UI's long-lived SSE stream — a
  red herring) and the LLM **judge** (`qwen-3.6-35b-a3b`, capped at 180 s by the runaway-generation
  guard). Neither is the playtest engine path.

If a prose stream wedged for hundreds of seconds, we'd see a `chat gemma-4-e4b` / `invoke_agent
prose_writer` / `turn` span of that length. **None exists.** Per-turn latency under load is
12.8–17.9 s avg (hourly), with a 67 s worst case — exactly "slow," not "stalled."

### 1.4 Why ~16 s/turn → 900 s: concurrency × fan-out × Gemma's slot count
- Gemma (`inferno/backends/llm_llamacpp.py`) serves **16 slots** (`n_parallel=16`). The registration
  comment claims 16 "lets inferno serve smallville's playtest concurrency without queueing" — but
  that assumes **1 request per session.**
- Playtest runs `--parallel 16` sessions, and **each session fans out per turn** (streaming
  prose holds a slot while `choice_author` + `free_text_gate` run via `asyncio.gather`, plus
  `phase1`, periodic `village_attention`, per-scene `chronicler`/`populator`). So 16 sessions issue
  **~32–48 concurrent** Gemma requests against **16 slots → 2–3× oversubscription → queueing.**
- That queueing *is* the ~16 s/turn (and the 67 s thundering-herd outlier). It is the same condition
  `scripts/inferno_stall_repro.py` reproduced — but note what the repro actually measured: TTFT
  **p95 16.3 s**, requests that **eventually returned**. That is latency inflation, **not** an
  infinite hang. §4 mislabeled "16 s TTFT" as a "stall."

### 1.5 The branch made it worse — by design
Workstreams A/B/C deliberately push deeper exploration (bypass edge, corroboration gate, decision
surfacing). They **work**: sessions now traverse the back-half map (`the_lane`, `st_caradocs`,
`knighton`) → **more scenes → 49–57 turns/session** instead of the ~24 turns the older,
pub-collapsed sessions ran. At ~16 s/turn, 56 turns ≈ 900 s. **The reachability win is what pushed
sessions over the wall-clock.** (The same back-half locations are the "thinly authored" ones whose
`choice_meaningfulness` dipped in §3.2 — nothing forces a conclusion there, so the player keeps
interacting and burns turns.)

### 1.6 Why §4 couldn't "capture a hung turn"
The three truncating runs are **not in Logfire at all** — the only two instrumented playtest runs
today (08:52 and 19:19 UTC) both completed with **0 truncations**, cases 372–520 s avg. There was
never a hang to capture because there was never a hang.

---

## 2. Corrected causal chain

1. Gemma = 16 KV slots.
2. Playtest = 16 parallel sessions × per-turn agent fan-out → ~2–3× slot oversubscription.
3. → per-turn latency ~16 s (67 s under thundering-herd), no individual call > ~67 s.
4. Branch reachability changes (A/B/C) → sessions explore the back half → 49–57 turns/session.
5. 49–57 turns × ~16 s ≈ 800–910 s, against a **fixed 900 s** `asyncio.wait_for` ceiling.
6. Sessions that don't trip an early forced ending (`run_out_of_the_village`) exhaust the budget →
   dumped `ending=None`, `trunc=True`. Variance near the 900 s edge → the 0.25/0.5/0.5/1.0 pattern.

No engine bug. No stream wedge. No grammar/thinking loop. A budget-vs-throughput mismatch.

---

## 3. Recommended fixes (ranked)

1. **Raise / scale the 900 s ceiling** (`evals/session.py:_SESSION_TIMEOUT_S`). It was sized for
   ~24-turn sessions; the branch routinely runs 55+. Make it a function of `max_scenes` (or
   per-turn budget × expected turns) rather than a flat 900 s. *Cheapest, highest-leverage.*
2. **Lower playtest `--parallel`** to ~6–8 so 16 Gemma slots aren't oversubscribed by per-turn
   fan-out. Cuts per-turn latency directly; sessions finish well inside any ceiling. (Confirms the
   mechanism: at lower concurrency the repro showed TTFT p95 2.6 s and 0 stalls.)
3. **Bound per-turn fan-out or cap turns/scene.** `aggressive_interrogator` (144934) did **47 turns
   in a single scene** — milling, never leaving. A turns-per-scene cap (or stronger leave pressure)
   both bounds runtime and is good for eval quality.
4. **First-token/stream timeout + retry on the prose path** (the `RetryModel` gap §4.6.2) — still
   worth doing for robustness, but it is **not** the cause here and will not change the truncation
   rate, because the turns aren't hanging.

## 4. What is NOT excluded
A *rare* true single-turn wedge (slot stuck, or client-side stream hang returning 200 but never
yielding) can't be 100 % ruled out from offline data. But it is **not the mechanism for these
truncations** — the artifact turn-counts (no low-turn truncations) and the latency distribution
(nothing > 67 s) both contradict it. If you want certainty per §4.2b, instrument the `wait_for`
timeout to dump the in-flight agent + token count; given this analysis, expect it to fire on a
session that has been steadily completing turns, not on one parked in a single call.
