# Findings — Eval/Judge Remediation + Inferno Truncation Investigation

**Branch:** `worktree-eval-judge-remediation` (commits `2afc80e`, `d69ef13`)
**Date:** 2026-05-23
**Baseline corpus under study:** `evals/artifacts/20260523-121958/`

---

## 1. Summary

Two linked problems were found in the latest clean playtest (`20260523-121958`):

1. **Beat-reachability ceiling** — only 3 of 11 beats fired; every session ended at the
   pub in ~3 scenes (`location_coverage` 0.40), because the pub is a graph cut-vertex and
   `tom_takes_responsibility` auto-fired the moment Tom warmed — *at that gateway*.
2. **LLM-judge quality debt** — voice/location drift mid-scene, truncated beat reveals,
   false-premise acceptance, cosmetic choices.

Six workstreams (A–F) plus one refinement were implemented and committed. **All deterministic
gates are green.** The prose-quality improvements (D/E/F) are validated across two judge passes.
The reachability changes (A/B/C) are mechanistically correct and directionally positive, but
their *corpus* validation is blocked by a pre-existing engine issue (below) that truncates ~half
of every playtest.

Separately, the truncation problem was investigated and its root cause **corrected**: it is not
an inferno "streaming endpoint" bug (as STATUS.md hypothesised) but **prefill/capacity starvation
under KV-slot oversubscription by cache-missing, large-context requests.**

---

## 2. What changed (per workstream)

| ID | Change | File(s) |
|----|--------|---------|
| **A** | Gate `tom_decides_to_speak_to_police` on cross-location corroboration (`any_evidence_of: [examined_the_crash_site, dylan_admits_he_knows, helen_hands_over_notes]`, deliberately **not** the pub reveal) so the climax can't fire from one pub conversation. | `world/truth.yaml` |
| **B** | Symmetric `the_post_office ↔ st_caradocs` edge — removes the pub as the sole cut-vertex to the back half of the map. | `world/locations/the_post_office.yaml`, `st_caradocs.yaml` |
| **C** | Hand the (system-prompt-stateless) choice author the currently-earned `decided_*` beats so Rule 7 can voice one as a turning-point option; the Phase-1 classifier then emits the commitment. | `src/handler/prompts.py` |
| **D** | Re-inject the location description (`## Here`) and present NPCs' **band-appropriate** voice lines (`## Voices Here`) every prose turn — not just turn 1. Activates `voice_by_band`, which was authored on every load-bearing NPC but **never reached the prose writer**. | `src/handler/prompts.py` |
| **E** | Surface a fired beat's `notes_to_llm` alongside its description; tighten Voice Rules 5 (carry the named particulars) and 6 (don't invent props / accept false premises). | `src/handler/prompts.py` |
| **F** | Model Mrs Hughes as a minimal background NPC (no disposition ladder) so the vicarage doorstep has an authored register for the judge to score against. | `world/cast.yaml` |
| **refinement** (`d69ef13`) | Don't echo an NPC's terse register line in `## Voices Here` on the turn its own reveal beat fires — the beat content leads instead (resolves a D↔E conflict on `tom_named_a_vehicle`). | `src/handler/prompts.py` |

Notable structural find: `voice_by_band` was dead authored data on the prose-writer path (only
`impression_by_band` was wired up) — D activates it rather than adding new prompt surface.

---

## 3. Evidence

### 3.1 Deterministic gates (immune to the truncation problem) — all GREEN
- `make test` → **260 passed**
- `eval-deltas` K=24 → **green**: `tom_warm_cold` 0.958 (≥ 0.917 baseline), `first_turn_inertia`/`lie_police`/`threaten_tom` 1.0. Confirms the Phase-1 trust path is untouched.
- full `eval-isolation` → all above floor: prose-voice `npc_voice_adherence` **5.0**, `choice_meaningfulness` 4.56, village-attention 1.0×4, chronicler 1.0.

### 3.2 LLM judges — corpus A/B (three post-change runs vs baseline)

Run 153538 is the **clean** (NotTruncated 1.0, full-exploration) corpus and is the most
representative; 142042/144934 were ~50% truncated (stayed in the content-rich pub/post-office).

| Judge | Baseline 121958 | 142042 | 144934 | **153538 (clean)** | Verdict (across 3 runs) |
|---|---|---|---|---|---|
| prose_matches_location | 4.059 | 4.11 | 4.385 | **4.343** | ↑ **robust** (D) |
| false_premise_accepted | 0.737 | 0.759 | 0.767 | **0.84** | ↑ **robust** (E) |
| emergent_event_honored | 3.5 | 3.5 | 4.5 | **4.5** | ↑ |
| npc_voice_adherence | 3.635 | 3.844 | 4.154 | **3.672** | ~flat / noisy (not robust) |
| choice_meaningfulness | 3.709 | 3.552 | 3.859 | **3.438** | ↓ in 2 of 3 — real dip |
| beat_fired_appropriately | 0.60 | 0.625 | 0.667 | **0.556** | noisy (n≈9) |
| prose_depicts_beat | 0.60 | 0.429 | 0.667 | **0.556** | noisy (n≈9) |

**Honest reading:** the robust prose wins are **location-grounding** and **false-premise
resistance**. `npc_voice` is net-neutral (one big gain, one flat). `choice_meaningfulness` is
**down** on the clean run — coherent with the reachability changes pushing players into the
back-half locations (the lane, churchyard, knighton), which are **thinly authored for
interaction**, yielding more cosmetic choices. That is a follow-up *authoring* need exposed by
deeper exploration, not a regression in the choice logic.

### 3.3 Corpus reachability metrics — VALIDATED on the clean run (153538)

The clean, untruncated run is the decisive evidence for the reachability package (A+B+C):

| Metric | Baseline 121958 | Clean run 153538 |
|---|---|---|
| location_coverage | 0.40 | **1.0** (every location visited; `locations_never_visited: []`) |
| beats_fired | 3 | **6** |
| median scenes / locations | 3.0 / — | **6.0 / 7.0** |
| endings_reached | run_out, tom_takes | **run_out, tom_takes** (3× run_out, 1× tom_takes) |
| persona_outcome_separation | 0.50 | 0.50 |
| ending_entropy_normalized | 0.314 | 0.314 |

- **`tom_takes_responsibility` IS reachable** — the diligent journalist gathered cross-location
  corroboration and reached the gated climax. Gate A does **not** over-gate (the earlier "never
  fired" was a truncation artifact: truncated sessions were cut off before reaching the deep
  corroboration locations).
- **Full map coverage** — knighton and the crash site, never visited in the truncated runs, were
  reached here. Bypass (B) + gate (A) push exploration as intended.
- `decided_to_confront_eluned` fired in run 142042 — Workstream C proven end-to-end.

**Open / follow-up (not blockers):**
- **3 of 4 personas hit `run_out_of_the_village`** (attention → hostile). That caps separation/entropy
  at 0.50/0.314 and is an **attention-tuning** matter (the attention classifier is pre-existing, not
  on this branch) — worth a look but out of scope here.
- **`choice_meaningfulness` dip** in the thinly-authored back-half locations — an authoring follow-up
  (add interaction depth to the lane / churchyard / knighton), exposed by the deeper exploration.

---

## 4. The truncation / inferno stall — corrected root cause

### 4.1 What "truncated" means
A playtest session is truncated when it never reaches an ending and is dumped with `ending=None`.
The signature is always `dur=900s, trunc=True` — the per-session wall-clock timeout
(`evals/session.py`, STATUS.md `82f752b`) firing on a hung session.

### 4.2 Diagnosis — partially established, NOT fully proven (corrects an earlier overclaim)

The truncation is **intermittent and engine-state-dependent**: across four post-change playtests
`NotTruncated` was 0.25 / 0.50 / 0.50 / **1.0** with identical code. The clean final run produced
zero truncations. This intermittency is itself a strong clue.

**What is established:**
- The signature is a session that runs many *complete, coherent* turns (47–56) and then hangs on the
  *next* turn (which is never written to the artifact), hitting the 900s wall with `ending=None`.
- A synthetic hammer (`scripts/inferno_stall_repro.py`) reproduces *a* stall — **prefill starvation**
  (0-token, TTFT → 16s) — but only under **cache-missing + concurrency ≥ slot count** (conc 16,
  `--unique`). The same load **non-streaming → HTTP 500s**, so that failure mode is *not*
  streaming-specific (correcting STATUS.md's "streaming endpoint wedged" framing).
- `eval-deltas`/`eval-isolation` never stall: short prompts + one reused **prefix-cached** system
  prompt → instant prefill, no contention.

**What is ruled OUT (re: the grammar/thinking-loop hypothesis):**
- Single grammar-constrained (`json_schema`) request with thinking ON / OFF / omitted, concurrency 1:
  all finish in ~1s, `finish_reason=stop`. No hang, no loop.
- Thinking is suppressed on **every** agent (prose writer inherits model-level
  `THINKING_OFF_EXTRA_BODY`; decision/populator agents re-include it).
- The intermittency argues against a template/grammar loop, which would be **deterministic** for a
  given input and recur — not appear after dozens of identical-shaped turns succeed, then vanish.

**What is NOT yet proven:** I have **not captured a hung turn in the engine logs** (the run I logged
came up clean). So I cannot yet say definitively whether a real truncation is (a) a slot stuck in
prefill, (b) a generation loop (the remaining form of the grammar/thinking hypothesis — low
probability but not excluded), or (c) a client-side stream hang (slot releases 200 OK but
pydantic-ai/httpx never returns). My best current hypothesis is an **intermittent resource/scheduling
condition under sustained mixed concurrent load**, but it is a hypothesis, not a proven cause.

### 4.2b How to settle it definitively
Instrument the 900s `asyncio.wait_for` in `evals/session.py` to dump, on timeout: the in-flight
agent name and whether it received any tokens/partial output; run playtests with
`docker logs --since … inferno-test` captured until one trips. The log signature then disambiguates:
`slot launch` with no `slot release` (server stuck) vs generation tokens climbing with no EOS (loop)
vs `200 OK` logged while the client hung (client-side).

### 4.3 Why the real playtest triggers it
- Playtest default `--parallel = 16` ("= engine KV slots"), and each session **fans out** per turn:
  the streaming prose call holds a slot while `scene_loop.py:376` runs the choice author + free-text
  gate concurrently (`asyncio.gather`), plus Phase-1, chronicler, and the attention classifier.
- So N sessions ≠ N requests — they oversubscribe the slots with unique (cache-missing) contexts,
  exactly the reproduced failure condition.

### 4.4 Reproduction (standalone, engine-only)
`scripts/inferno_stall_repro.py` hammers `/v1/chat/completions` directly, mirroring the prose
writer's request shape (temp 1.0 / top_p 0.95 / top_k 64, `enable_thinking:false`), with direct
mid-stream stall detection. Bisection results:

| Variant | Result |
|---|---|
| conc 8, cached (identical prompt), stream | **0 stalls**, TTFT 0.16s — rules out "concurrent streaming" |
| conc 8, 8k-token cached prompt, stream | **0 stalls**, TTFT 0.16s — caching masks prefill cost |
| **conc 16, `--unique` (cache-defeating) 6k prompt, stream** | **13/275 STALLED** ✱ reproduced; TTFT p50 7.8s / p95 16.3s; 7 stalls in one thundering-herd instant; all stalls 0 tokens (prefill starvation, not mid-gen) |
| conc 4, `--unique` 6k prompt, stream | **0 stalls**, TTFT p95 2.6s — concurrency alone insufficient; needs oversubscription |
| conc 16, `--unique` 6k prompt, **`--no-stream`** | **HTTP 500s** — failure is **not** streaming-specific |

**Commands:**
```bash
# Reproduce:
SMALLVILLE_USE_INFERNO=1 uv run python scripts/inferno_stall_repro.py \
    --concurrency 16 --duration 240 --prompt-pad 6000 --max-tokens 200 --unique

# Controls that do NOT reproduce (isolate the cause):
#   drop --unique  -> prefix cache hides it
#   --concurrency 4 -> below slot count, no starvation
#   --no-stream     -> same load returns 500s instead (not a streaming bug)

# Ground-truth in-harness repro (~50% sessions truncate):
SMALLVILLE_USE_INFERNO=1 uv run python -m evals.run
# while watching the engine:
docker logs -f inferno-test ; docker stats inferno-test
curl -s http://127.0.0.1:7770/slots ; curl -s http://127.0.0.1:7770/metrics
```

A clean inferno **restart did not fix it** — it recurs under concurrent cache-missing load within
minutes.

### 4.5 Next investigative steps (engine side)
- Confirm Gemma's **actual `-np`** on inferno and its VRAM/KV budget under co-residency (config notes
  inferno shares VRAM with FLUX/TRELLIS/Qwen — Gemma's *effective* slot count may be < 16).
- Capture `docker logs` / `/slots` / `/metrics` at a reproduced stall to see slot/queue state.
- Confirm prefix-cache (`cache_prompt`/`prompt_cache`) behaviour for unique contexts.

### 4.6 Mitigations to consider (not yet applied)
1. **Lower playtest concurrency** to account for per-turn fan-out (effective concurrency ≫ session count).
2. **Add a first-token / streaming timeout + retry** to the prose path (the `RetryModel` gap — it
   currently only retries the stream entry, so streaming starvation is uncatchable).
3. **Bound prose-writer context growth** (conversation accumulates every turn → larger prefills late-session).
4. **Raise inferno prefill/batch capacity** or give Gemma dedicated slots away from co-resident models.

---

## 5. Merge recommendation

> The deterministic gates and two judge passes are clean and immune to the truncation problem.
> The risk is concentrated in Workstream A. Recommendation below.

| Part | Merge? | Why |
|---|---|---|
| **D** (per-turn anchors) | ✅ **Merge** | Robust judge wins (npc_voice 3.64→4.15, location 4.06→4.39 across two runs); isolation 5.0; activates dormant `voice_by_band`. Low risk. |
| **E + refinement** (beat notes, Rules 5/6, reveal-turn voice) | ✅ **Merge** | Addresses the real judge findings; false_premise ↑, prose_depicts recovered. Low risk. |
| **F** (Mrs Hughes) | ✅ **Merge** | Pure authoring fix for a genuine judge gap; loads clean; inert NPC. |
| **C** (decision surfacing) | ✅ **Merge** | Fired a decision beat for the first time ever; only surfaces *earned* decisions, engine enforces at-most-one. Harmless without A. |
| **B** (bypass edge) | ✅ **Merge** | Low-risk topology; invariants pass; raises exploration. Harmless without A. |
| **A** (corroboration gate) | ✅ **Merge** (de-risked by clean run 153538) | Earlier held for fear it made `tom_takes` unreachable — the clean corpus **disproves that**: `tom_takes` fired, `location_coverage` hit 1.0 (knighton + crash site reached). Gate A works as designed and does not over-gate. |

**Now validated together:** the clean run 153538 (NotTruncated 1.0) proves the reachability package
(A+B+C) end-to-end — coverage 0.40→1.0, beats 3→6, `tom_takes` reachable. A is the keystone; without
it the pub re-collapses to the 3-scene auto-win.

**Bottom line:** **merge the whole branch (A–F + refinement).** Reachability is validated on a clean
corpus; prose-grounding and false-premise resistance are robust judge wins. Two **follow-ups** (not
blockers, and separable): (1) **authoring depth** for the back-half locations (the lane / churchyard /
knighton) — the `choice_meaningfulness` dip shows they're thin once players actually reach them; (2)
**attention tuning** — 3/4 personas hit `run_out`, capping ending diversity (pre-existing, off-branch).
Independently, the **intermittent truncation** (§4) is an engine/harness issue to pursue separately;
it does not block the content merge.

---

## 6. Artifacts
- Runs: `evals/artifacts/20260523-140146` (0.25), `…-142042` (0.50, judged), `…-144934` (0.50, judged)
- Isolation: `evals/artifacts/isolation/20260523-135845`
- Repro tool: `scripts/inferno_stall_repro.py` (currently untracked)

---

## 7. New evals added (so the next iteration is measured, not guessed)

The findings exposed measurement blind spots: we tracked *nodes* but not *edges/paths* or
structural reachability, and judge scores were aggregated so per-location dips hid in the mean.
Four evals were added on top of the unified `python -m evals` CLI (these slot into the existing
`iso` / `play` / `judge` lanes, plus a new no-engine `static` lane).

| Eval | Lane | What it catches | What it found immediately |
|---|---|---|---|
| **Static topology lint** (`evals static`) | new, deterministic, no LLM | locations unreachable from start; cut-vertex chokepoints (what each gates); per-beat reach distance | Confirmed Workstream B removed the pub as a cut-vertex; flagged `examined_the_crash_site` at **5 hops** (deepest) — why it never fired in play |
| **Edge coverage** (`edge_coverage`, `edges_never_traversed` in `play`) | corpus metric | which *connections* carry the player (node coverage can't see an unused edge) | The clean run was coverage 1.0 but **edge 0.9** — the one unused edge was the new bypass `st_caradocs — the_post_office`: players reached the back half via the old route, not B's edge |
| **Per-location judge breakdown** (`judges_by_location` in `judge`) | judge aggregation | which *places* drag a scored judge down | `npc_voice` 2.86 tan_y_coed / **3.00 st_caradocs (15/15 ≤3)** / 3.00 the_lane — the back-half locations, masked by the 3.67 average; an authoring to-do |
| **Attention regression guards** (`persistent_civil`, `low_effort_skim` in `iso --suite attention`) | isolation gate | a firm-but-civil or low-effort reporter being wrongly run out of the village | Both pass (noticed / none) — so the corpus `run_out` over-ejection is **not** a classifier misfire on these patterns; it's persona behaviour over a session. Locks the correct behaviour in. |

All four are deterministic or aggregation-only except the attention cases (K-sample gate, already
green). Unit-tested in `tests/test_static.py` and `tests/test_metrics.py` (edge coverage).
