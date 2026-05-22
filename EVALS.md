# HANDLER — Evaluation Strategy

How we measure whether the game is *good*, not just whether it runs. Targets the
documented failure modes of LLM narrative games (incoherence over a session,
hollow/off-voice NPCs, choices without consequence, prompt-craft burden on the
player) plus the general qualities of good narrative games (pacing, mystery
payoff, prose, earned endings).

Authoritative engine orientation is `CLAUDE.md` (the v2 `SPEC.md` was retired).
This doc is the eval companion.

---

## Core principle: proactive infra, reactive judges

- **Infrastructure and invariants are built proactively** — they catch *silent*
  corruption and dead authored content that is otherwise invisible.
- **Graded LLM judges are built reactively** — each rubric is born from a real
  failure we saw and fixed in a prompt, with the failing playthrough frozen as
  its first case. Pre-writing rubrics for hypothetical failures yields judges
  that track nothing real and can't be calibrated.

The discipline is the regression-test loop applied to prose: fix the prompt,
capture the playthrough that broke it, add the check that would have caught it.

---

## The four buckets

Ordered by increasing cost and looseness.

### Bucket 1 — Deterministic programmatic checks
Pure functions over `State` + `dialogue_history` + transcripts. No LLM. Run on
every commit. **This is pytest, not pydantic-evals** — invariants must *hold*,
not score.

- State invariants: beat monotonicity, no duplicate `case_file_entries`,
  `npc_memory` length tracks scenes-present, every fired beat in the catalog.
- Reachability sweep: which beats/endings fire at all across N playthroughs?
  Dead authored content surfaces here and nowhere else.
- Repetition detector: n-gram overlap within/across scenes (the "AI Dungeon
  loop").
- Prose heuristics: sentence-length variance (burstiness), banned-phrase regex,
  lexical diversity. **Seed the banned list from slop our Gemma actually emits.**
- Voice fingerprint: per-NPC dialogue embedding distance (intra vs inter).

### Bucket 2 — LLM-as-judge over recorded artifacts
Where pydantic-evals earns its keep. Judge at the **smallest scope where the bug
manifests.**

- **2a Per-call** (fast, reproducible, isolates the agent at fault): prose-writer
  voice adherence, choice-author relevance, chronicler fact-capture, Phase 1 beat
  correctness (ground-truth labelled — easiest to start).
- **2b Whole-scene**: coherence, dramatic shape, payoff — needs scene context.
- **2c Whole-session**: solvability, consequence-of-choice, earned ending — needs
  the long arc and the alternative endings.

A great scene can sit in a wandering session; a fine-grained session can be made
of merely-OK scenes. The scopes diagnose different bugs.

### Bucket 3 — Simulated playthrough generation
Not an eval — the substrate the other buckets consume. A cloud model drives the
existing `play_session` callbacks as the player, across **personas** (diligent
journalist, aggressive interrogator, distracted/lazy, completionist). Personas >
random: random input doesn't cluster the way real players do, so it misses the
failure modes humans hit. See *Playtester architecture* below.

### Bucket 4 — Self-judging A/B (solo dev)
Cheap for a single human in the loop. Baseline build `A` vs candidate `B`, paired
comparable scenes side-by-side, **one specific question** ("which Bryn sounds
more like Bryn?"), three keys (A / tie / B). Aggregate to a per-dimension
win-rate. Pairwise is faster and lower-variance for humans than rating in
isolation — ~50 judgments in 20 minutes.

---

## What to build now vs. later

| Build now (proactive) | Build from real errors (reactive) |
|---|---|
| Sim-playthrough harness (Bucket 3) | LLM-judge rubrics (Bucket 2) |
| Deterministic invariants (Bucket 1) | Banned-phrase / prose heuristics |
| Beat/ending reachability sweep | Per-call classification evals |
| A/B harness *shell* (Bucket 4) | A/B *rubrics* |

Avoid the trap of a 12-rubric suite up front: you'll calibrate judges against
failures that never occur while the judges that matter don't exist yet.

---

## Judging scope: agent-as-whole vs. specific call

| Bug shape | Where to catch |
|---|---|
| Prose writer drifts off voice this turn | Per-call (2a), prose writer |
| Phase 1 fires the wrong beat | Per-call (2a), ground-truth labelled |
| Scene has no dramatic shape | Whole-scene (2b) |
| Choices feel inconsequential | Whole-session (2c) |
| Ending isn't earned | Whole-session (2c) |
| Voice drifts across a long session | Whole-session, per-NPC across scenes |

The judge must see **what the agent saw + the agent's output + ground truth.**
Anything less is scoring vibes.

---

## Representing the game as text for a judge

Three serialization layers, picked by eval scope:

- **State snapshot** (per-call): relevant `State` fields + in-scene
  `dialogue_history` + the bible fragments visible to that call. Reuse the
  prompt builder's serializer — it is the literal source of truth for "what did
  the agent know."
- **Scene-as-text** (~0.5–1.5k tok): location, NPCs present, beats fired
  entering, prose+picks per turn, exit reason, beats unlocked.
- **Session-as-text** (~3–6k tok): the case file (already prose) + location path
  + beats-fired log + ending reached + epilogue.

---

## Programmatic analysis: eval or separate?

**Separate.** Invariants ("no duplicate case-file entries") must hold on *every*
run — that is pytest with the sim-playthrough fixture, failing the build on
breach. pydantic-evals is for *graded* outputs where the score is the signal
(judges, burstiness, voice distance) and you watch a number trend over builds.

---

## A third category: deterministic rule over a stochastic subject

The pytest-vs-pydantic-evals binary above has no slot for the most useful engine
regression test: **a deterministic assertion about a single agent that runs at
temp=1.0.** "Phase 1 must not bump `attention` on a private warm exchange" is a
*hard rule*, but the agent is stochastic, so a single run is not a build-gateable
boolean — it is a **pass-rate over K samples**.

This is the **isolation lane** (`evals/isolation/`, `make eval-deltas` /
`make eval-isolation`). Freeze the input `State`, re-run the real agent K times,
score each sample, gate on the per-case pass-rate with a floor:

- It re-executes the agent, so unlike a frozen *judge* case (which freezes recorded
  *output*) it can test an engine-prompt change. It is the only lane that can.
- The floor catches **collapse, not drift** — set it well below a healthy rate so
  binomial noise doesn't trip it (the captured attention bug drove its case to ~0;
  a 0.5–0.8 floor separates "fixed" from "regressed" cleanly). Drift is the
  `summary.json` diff's job, not the gate's.
- **Structural rules may be asserted proactively** (no-op-on-first-turn, no dup
  case-file entry, non-empty prose) → bool assertions → they gate. **Behavioural
  rubrics stay reactive** — reuse an `evals.judges` class against the *fresh* output
  → Scored → it trends, it does not gate. Do not pre-write a voice/agency rubric for
  a failure you have not seen; grow those from real captured failures, same as the
  recorded-artifact judges.

To add the next agent: write 1–2 `IsoCase`s (a `build(world) -> State` + the `Expect`
or judge), wrap them in a `Suite`, register it in `evals/isolation/__main__.py`.

---

## Suggested layout

```
evals/
  datasets/        # per-call cases (hand-built) + generated sessions
  evaluators/      # deterministic + LLMJudge wrappers, composable
  harness/         # sim player, personas, session runner, A/B UI
```

---

## Playtester architecture

A cloud model (Gemini Flash) plays the game by supplying the same callbacks the
CLI supplies from stdin. The local Gemma server (system-under-test) keeps the
full GPU; the player runs in the cloud.

### Why a separate cloud model, not the same Gemma
- **GPU contention**: a local player competes for the same `-np` slots as the
  system-under-test, halving session throughput and polluting latency numbers.
- **Correlated blind spots**: same model as narrator *and* player never gets
  confused by what the narrator is also confused by — the failure modes we want
  to surface stay invisible.
- **Cost is negligible**: player turns emit a choice index or one short line.

Gemini Flash is the sweet spot: cheap, fast, high concurrency, independent of the
local stack. Strength isn't the issue — steering it into a *confused-novice*
persona (to surface prompt-craft burden) is, and that's a prompting problem.

### Player callbacks
`play_session` is already driver-agnostic (scene_loop.py:364): it takes
`pick_option`, `pick_walk`, `pick_free_text`. A `GeminiPlayer` implements all
three against one pydantic-ai `Agent` whose **message history persists across the
whole session**, so the player remembers its own playthrough.

- `pick_option(prose_part) -> int` — serialize prose + numbered options, return a
  validated index.
- `pick_free_text(req) -> str` — serialize prose + prompt, return one line in
  persona voice.
- `pick_walk(state, world) -> str` — serialize current location + adjacents,
  return a destination id from `adjacent_to`.

Persona lives in the player agent's system prompt. Structured output
(`output_type`) constrains the choice; validate against the live option count and
retry on out-of-range.

### Running many games in parallel
Each session is independent: `AgentBundle` is immutable and shareable; `State`,
`SceneRunnerRuntime`, `ChroniclerRuntime`, and the player's own agent history are
per-session. So a run is N `play_session` coroutines as asyncio tasks against one
shared bundle and one shared llama endpoint.

- Each `play_session` issues **one** llama call at a time (phase1 → prose →
  choice are sequential awaits). N concurrent sessions → up to N concurrent
  llama requests → set **N = `llama_n_parallel`** so demand matches the server's
  KV slots. Bound with an `asyncio.Semaphore(n_parallel)`.
- Player (Gemini) calls happen *between* llama calls, while a session waits for
  its pick — they add latency to that session but not load to the GPU, so other
  sessions keep the card busy. Natural pipelining.
- For an eval run, **don't use inferno** (it budgets VRAM for co-resident image
  models). Spawn a dedicated `llama-server` and raise `-np` (E4B Q8 is small —
  8–16 slots fit comfortably on Blackwell). KV VRAM scales linearly with `-np`;
  that, not compute, is the ceiling.
- Each task writes one `SessionArtifact` JSON (`asdict(state)` + per-turn prose +
  picks + beats-fired log + ending). That artifact is what Buckets 1, 2, and 4
  consume.

Reproducibility note: Gemma runs at temp=1.0, so runs aren't bit-reproducible.
Rely on the *corpus* of artifacts for trend signal, not exact replay; pass a
per-request seed only if a specific case needs pinning.

---

## What's built (and where)

Three lanes, scoped by where a property is decidable. The discipline: structural
facts are gates/metrics (pure-python, every run); behavioural judgments are LLM
judges (one thing each, run on demand).

| Lane | Scope | Module | Surface |
|---|---|---|---|
| **Gates** (must hold) | per session | `evals/evaluators.py` | `make playtest` |
| **Metrics** (trend) | corpus | `evals/metrics.py` → `report.py` | `make playtest` → `summary.json` + diffed report |
| **Judges** (score one thing) | transition / session | `evals/judges/` | `make judge RUN=<dir>` |
| **Isolation** (re-run one agent K×) | single agent | `evals/isolation/` | `make eval-deltas` / `make eval-isolation` |

- **Gates** — `NoPrematureEnding` is the live depth gate (an ending must follow ≥3
  scenes / ≥3 locations); `EndingEarned`/`ExploredBeyondStart` are superseded but
  kept for their evidence/prologue regressions.
- **Metrics** — five north-star numbers (`report.NORTH_STAR_KEYS`: ending entropy,
  beats fired, filler-turn ratio, beat-set Jaccard distance, persona→outcome
  separation) plus coverage, beats-per-location, depth, longest-common-beat-prefix,
  and ladder-compression. Each run writes `summary.json`; the report diffs it
  against the previous run — that diff is the iteration loop.
- **Judges** (`make judge RUN=<artifact dir>`) — default to local Qwen via inferno
  (no key; warmed once; fired concurrently). Each judge is a pydantic-evals
  `Evaluator` (`AgentJudge.evaluate` bridges its PassFail/Scored verdict onto a
  bool assertion / float score); `judge_run` wraps each scope's items in a `Dataset`
  over an identity task and calls `Dataset.evaluate`, so the judges report into
  Logfire experiments (`handler_judge_*`) alongside the invariants, merge their
  numbers into the run's `summary.json` under a `judges` key (diffed run-over-run
  like the metrics), and write the worst cases to `judges.md`. The high-volume
  Scored transition scope is sampled (`--max-cases-per-scope`, default 1000, seeded
  via `--seed`); the small PassFail bug-detector scopes run in full. **Single-transition**
  (before→after one turn): `beat_fired_appropriately` (the walk-away/Maud-dump/
  helen-without-meeting detector — *not* a regex), `prose_depicts_beat` (phoned-Iwan
  hallucination), `prose_matches_location`, `npc_voice_adherence`,
  `choice_meaningfulness` (the "look closer at the box" agency judge),
  `false_premise_accepted`. **Whole-session**: `ending_earned_narratively`,
  `emergent_event_honored` (freedom vs. hallucination), `choice_consequence_felt`.
  Freeze a real failing case under `evals/cases/` to grow the regression set
  (`evals/cases/README.md`).

### Scope rule of thumb
Engine correctness ("did the model do the right thing on *this* input?") →
single-transition. Authoring/pacing structure (counts, coverage, linearity) →
corpus metrics. Arc properties (earned ending, consequence, emergent continuity)
→ whole-session. Consequence-across-runs → persona→outcome separation (no forking).

### Engine affordances added for pacing
Two optional `Beat` fields close the gap the prompt alone couldn't hold against a
4B model: `requires_count` / `requires_count_beats` (a threshold gate — e.g.
`village_runs_you_out` needs ~3 negatives; `decided_to_walk_away` needs ≥2 engaged
threads) and `min_scenes_with_npc` (a pacing gate so reveals can't dump on the
first encounter / a whole ladder can't fire in one scene).
