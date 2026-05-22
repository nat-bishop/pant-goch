# HANDLER divergence-tuning — STATUS

Goal: improve **approach→outcome divergence** (CONTROL_LOOP north star:
`persona_outcome_separation`, `ending_entropy_normalized`) by turning one knob per
iteration (phase1 prompt OR pacing/band constants only), gating on `make eval-deltas`
(K=24), promoting to `make playtest` only when green. Branch: `divergence-tuning`.

## Baseline (B0) — pre-change

`make eval-deltas` K=24: **GREEN** (all asserted cases above floor)
- tom_warm_cold 0.917 · neutral_question_no_attention 1.0 · first_turn_inertia 1.0
- public_accusation_attention 0.792 · lie_police_to_tom 1.0 · threaten_tom 1.0
- ⚠️ histogram smell: `neutral_question` ("how long have you lived here") emitted
  `trust tom+1` in **21/24** samples — neutral talk is being scored as warmth.

`make playtest` (10/12 sessions; 2 stragglers killed after grinding to the scene cap):
| metric | B0 |
|---|---|
| persona_outcome_separation | **0.095** |
| ending_entropy_normalized | **0.181** |
| mean_pairwise_beat_jaccard_distance | 0.259 |
| beats_fired | 4 / (catalog) |
| filler_turn_ratio | 0.353 |
| post_exhaustion_turn_ratio | 0.27 |
| median_scenes_per_session | 3.0 |
| endings_reached | tom_takes_responsibility ×9, buried_the_story ×1 |

**Diagnosis (total collapse).** Every persona — including `distracted_skimmer` and
`aggressive_interrogator` — lands `tom_takes_responsibility`, the *deepest*-trust
ending (`at_peace`, depth 3), in ~3 scenes without leaving the pub. Per-turn trace
shows phase1 grants `trust:+1` on nearly every turn AND `guard:-1` on routine talk,
so effective warmth (`trust − guard`) climbs ~+2/turn; Tom (band width 2) crosses
`at_peace` (effective ≥6) in ~5 turns. Approach does not fork outcome → separation ≈ 0.

CONTROL_LOOP knob: "trust accrues too fast / neutral moves score +1 too often →
tighten the prompt's neutral=0/0 rule first."

## Iterations

### Iter 1a/1b — phase1 prompt tightening (REVERTED — broke the gate)
Tried making phase1 trust default to `0`, `+1` only for earned warmth. Two variants:
- 1a (trust + guard reworded): `tom_warm_cold` 0.917→**0.0** (empathy read as empty),
  `public_accusation` 0.792→**0.0** (Tom BREACHed 24/24). 2 floor breaches.
- 1b (guard restored verbatim, only trust reworded): `tom_warm_cold` **0.167**,
  `public_accusation` **0.042** — breach explosion *persisted* with original guard,
  so it tracks the **trust** wording, not guard. Conclusion: Gemma-4-E4B can't
  cleanly separate "warm" from "neutral" via the prompt without nuking genuine
  warmth and destabilising BREACH. **Reverted to baseline prompt.**

### Iter 1 — pacing knob: widen the slow band
Knob: `src/handler/engine.py` — `_BAND_WIDTH_BY_WARMS["slow"]` and `TRUST_MAX`
(coupled: `at_peace` = 3×width must stay below the cap). Band constants do **not**
affect per-turn delta emission, so the delta gate is unchanged by construction. The
lever: a slow NPC's `at_peace` (deepest band, fires `tom_takes_responsibility`) needs
much more warmth, opening a gap between `open` (evidence → file/confront/tell-Iwan
unlock) and `at_peace` (Tom resolves it himself) so a single warm scene no longer
auto-collapses every persona into `tom_takes_responsibility`.

- **1a: width 4 / TRUST_MAX 14 (committed `decc26a`) — STARVED.** Gate GREEN
  (tom_warm_cold 0.875, public_accusation 0.792, matches baseline). But the playtest
  (cap 40) ran ~17 min with **0/12 sessions resolving** — `at_peace@12`/`open@8` is so
  far that sessions wander to the scene cap without ever committing to an ending. Too
  wide; killed the run. (This removed baseline's *automatic* fast resolution without
  giving the player-driven endings time to fire.)
- **1b: width 3 / TRUST_MAX 10 — `at_peace@9`, `open@6`** (vs original 6/4). Moderate.
  Re-ran with a smaller scene cap so non-committing grinders truncate fast.
  - eval-deltas K=24: _pending_
  - playtest (cap 20): _pending_

Updated `test_confession_depth_slow_width` to the configured width (arithmetic unit test).

## Reverts
_(none yet)_

## Proposals (subjective — not auto-applied)
_(none yet)_

## Best / worst transcripts
_(to be filled at stop)_
