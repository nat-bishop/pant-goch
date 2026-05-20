# HANDLER — Evaluation Strategy

How we measure whether the game is *good*, not just whether it runs. Targets the
documented failure modes of LLM narrative games (incoherence over a session,
hollow/off-voice NPCs, choices without consequence, prompt-craft burden on the
player) plus the general qualities of good narrative games (pacing, mystery
payoff, prose, earned endings).

Authoritative engine design is `SPEC.md`. This doc is the eval companion.

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
