# HANDLER — Eval Control Loop

How to **steer** the quality engine with evals, not just measure it. `EVALS.md`
catalogues what exists; this is the playbook for using it to get the outcomes we
want. Written from the 2026-05 quality-engine bring-up.

The engine is a thin spine: the LLM nudges per-NPC `trust`/`guard` + global
`village_attention`; the engine derives every reveal, the climax, and the ending
from arithmetic. So **outcome is controlled at two layers**, and each has its own
eval:

| Layer | What it controls | Eval (the gate) | Speed |
|---|---|---|---|
| **Per-turn delta** (phase1) | Did *this* move nudge standing correctly? | `evals iso --suite deltas` (deterministic, code-asserted) | seconds |
| **Corpus behaviour** (sessions) | Does *approach* fork *outcome*? | `evals play` → `summary.json` | minutes |

(All eval commands run via `uv run python -m evals <cmd>`; an A/B of a single
design knob is `evals play --experiment <name>`.)

Iterate the fast gate first; only run the corpus once the unit eval is green.

---

## What we want (the target outcomes)

**Approach forks outcome.** Different personas must reach different endings —
this is the north star (`persona_outcome_separation`, ending entropy,
disposition end-state divergence). Each ending is gated on a *different shape* of
disposition, so they should *not* converge:

| Ending | Disposition shape that earns it | "Owner" persona |
|---|---|---|
| `run_out_of_the_village` | `village_attention ≥ hostile` (alone) | aggressive |
| `buried_the_story` | walk away + real traction (≥3 thawed / noticed) | distracted |
| `published_story` | file + any evidence (detachment) | diligent |
| `tom_takes_responsibility` | **Tom at `at_peace`** (deepest band) + didn't file | patient step-back |
| `iwan_finds_peace` | tell Iwan privately + **Iwan `open`** + evidence | gentle |
| `eluned_arrested_after_confrontation` | confront + `eluned_breaks` + village onside | risky closer |

Reveals are **earned, not dumped or starved**: evidence trips when an NPC crosses
`open` (band 2 of cold→thawing→open→at_peace), reached over a few invested turns
— not in one turn, not never. A **burned relationship stays burned** (`guard`
BREACH latches). **Attention rises only on conspicuous, public acts** — never on
quiet conversation.

---

## Signals → diagnosis → knob

Read a signal that's off, name the layer, turn the matching knob, re-run its gate.

| Signal (where) | What it means | Knob to turn |
|---|---|---|
| One delta value emitted on *every* call regardless of input (delta eval ~0%) | Degenerate collapse — model latched a default | **Schema shape** in `agents._build_state_change_bundle_type`. Fix the wire schema, not the prompt (e.g. an optional signal → scalar default, not a union list member). |
| Warm move → no `trust`; gentle talk → spurious `attention`/`breach` (delta eval) | Per-turn nudge mis-signed | **phase1 prompt** (`build_phase1_system_prompt`). Gate every change on `evals iso --suite deltas`. |
| Personas converge on one ending; a shallow persona reaches a **deep-trust** ending (e.g. `distracted_skimmer` → `tom_takes_responsibility`) | Trust accrues **too fast** — `at_peace` reached incidentally | **Band width / pacing**: widen `_BAND_WIDTH_BY_WARMS` / raise `warms` to `slow`, bump `TRUST_MAX`. *And* check the delta eval: if neutral moves score `trust:+1` too often, tighten the prompt's neutral=0/0 rule first. |
| Every session collapses to `buried_the_story` / truncates with no evidence | Trust accrues **too slow** — nothing reaches `open` | Narrow bands / `warms: steady`; loosen the prompt's warmth threshold. |
| Patient sessions hit `run_out_of_the_village` | Attention over-emitted on benign play | phase1 prompt: attention `"0"` default, conspicuous-only; verify with the gentle delta cases. |
| A persona's signature ending never fires across the corpus | Ending gate unreachable / mis-authored | **Authoring**: ending `when`-gates + order in `truth.yaml`, `disposition.breach` lists. |

---

## The discipline (what keeps the evals honest)

- **Assert only what's *true*; measure the rest.** Absolute design rules (no
  attention on private talk, first-turn inertia, BREACH only on the authored
  provocation list) are hard assertions. Fuzzy magnitudes (is *this* remark +1 or
  0; guard+1 vs BREACH on a threat) are **reported, not asserted** — judging an
  opinion as ground truth poisons the gate. Pick case wording so the asserted
  axis is unambiguous; put the borderline version in a *measured* case.
- **Prefer programmatic over LLM-judge.** The delta the engine applies *is* the
  state diff — assert it in code with a known-correct answer. Reserve LLM judges
  (given the legible state + the diff) for genuinely subjective questions.
- **Sample, don't single-shot.** Gemma runs at temp 1.0; every delta case is K
  samples scored as a pass-*rate*. One run is noise (a breach rate swung
  83%→0% across runs); trust the rate and the corpus, not one playthrough.
- **Structural fix > prompt patch.** A collapse the prompt can't hold is a schema
  smell — change the shape.
- **Freeze every real failure as a case.** The bug you just fixed is the
  regression test that keeps it fixed (the deltas suite in `evals/isolation/`
  was seeded from the live attention-spam collapse: 18% → 99%).

---

## The loop, in one line

See a bad outcome → pick the layer (mechanism = delta eval; pacing = band widths;
divergence = ending gates) → turn one knob → re-run the *smallest* gating eval →
promote to the corpus only when green.
