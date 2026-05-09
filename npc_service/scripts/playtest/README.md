# LLM Playtester

Drives full smallville sessions against the production engine + a separate eval llama, with the player side synthesised by a `pydantic-ai` Agent given a persona spec. Evaluated with `pydantic-evals` — eight programmatic evaluators + eight focused LLM judges (G-Eval-style, voice & tone + game-design pacing).

## Two-server architecture

| Server | Model | Port | Binary | Purpose |
|---|---|---|---|---|
| Engine | Gemma 4 E4B Q8_0 | 8080 | `vendor/llama.cpp/llama-server.exe` | The SUT — chat/classifier/extractor/summary/fact-checker/director agents. Same model players will face in the actual game. |
| Eval | Qwen 3.6 35B-A3B Q8_0 (MoE) | 8081 | `vendor/ik_llama.cpp/llama-server.exe` | The synthetic player AND every LLM judge. Different family from the engine to mitigate same-family preference leakage in judging (arxiv 2502.01534). |

Both run concurrently for the duration of the playtest. The synthetic player and the SUT alternate turns within sessions; the LLM judges fire after sessions complete (Gemma idle then, but stays loaded — reload latency exceeds any saving).

### Why `ik_llama.cpp` for the eval server

Mainline `llama.cpp`'s MoE expert routing isn't optimized for A3B-style routing — at `--parallel 64` the throughput gap vs vLLM/SGLang is real. `ik_llama.cpp` is a llama.cpp fork (https://github.com/ikawrakow/ik_llama.cpp) with fused MoE kernels and hybrid CPU+GPU expert offload — closes most of the gap while keeping the GGUF + llama-server CLI shape, the OpenAI-compat endpoint, and Windows-native operation (no WSL2/Docker). vLLM/SGLang would be faster still but require WSL2 on Windows and a different lifecycle (no Job Object cleanup) — defer those for "if/when this becomes multi-tenant production."

You need to vendor the `ik_llama.cpp` binary at `vendor/ik_llama.cpp/llama-server.exe`. Either build from source (cmake; same shape as building mainline llama.cpp) or grab a release build from the upstream repo.

## Quickstart

```sh
# 1. Install the optional pydantic-evals dep
cd npc_service
uv sync --extra playtest

# 2. Download both models (one-time, ~8 GB + ~37 GB)
make download-model        # Gemma 4 E4B Q8_0
make download-eval-model   # Qwen 3.6 35B-A3B Q8_0

# 3. Boot both llama servers in separate terminals
make llama        # terminal 1: Gemma on :8080, -np 128
make llama-eval   # terminal 2: Qwen on :8081, -np 64 (ik_llama.cpp)

# 4. Run the playtester (terminal 3)
make playtest
# Or with overrides:
uv run python -m scripts.playtest --seeds 8 --max-sends 25 --parallel 64
```

Output:
- `npc_service/_analysis/playtest_<timestamp>.json` — full transcripts, tick decisions, final states per session
- pydantic-evals tabular report printed to stdout (per-case scores, aggregates)

## Hardware

The two-server setup at `--parallel 64` is sized for a **96 GB-VRAM GPU** (e.g., RTX PRO 6000 Blackwell). Approximate VRAM split with `--kv-unified` on both servers:

| Component | VRAM |
|---|---|
| Gemma 4 E4B Q8 weights + KV (128 slots × 32k, MQA) | ~15 GB |
| Qwen 3.6 35B-A3B Q8_0 weights + KV (64 slots × 32k, GQA) | ~65 GB |
| 2× CUDA contexts | ~1 GB |
| **Total** | **~80 GB** |

If you OOM at first long-context request (the documented failure mode for KV cache expansion), step down in this order:

1. Drop Qwen ctx-per-slot: edit `make llama-eval`'s `-c 32768` to `-c 16384` (16k per slot, plenty for condensed judge views).
2. Step Qwen Q8_0 → Q6_K via `EVAL_MODEL_URL` in the Makefile (~28 GB weights, small quality cost).
3. Drop `--parallel 64` to `--parallel 32` (halves Qwen's slot demand; sessions run in two batches).

If `ik_llama.cpp` doesn't support `--kv-unified` (it tracks mainline closely but verify on your build), each slot reserves its own ctx-sized KV — you'll need to drop ctx-per-slot or `-np` aggressively.

## What it tests

The playtester catches what static analysis can't:

- **Narrative coherence** — does Aria's voice stay on-register; do beat transitions feel motivated?
- **Game-design pacing** — does the game progress and become interesting at the right time? (`OnboardingClarity`, `StakesEscalation`, `ClimaxLanding`, `Closure`, `ChoiceMeaningfulness`)
- **Director bias** — beats the static analyzer says are reachable but the LLM Director consistently avoids; surfaced via `BeatCoverage` + cross-check against the analyzer.
- **Agent drift** — chat agent leaking email headers (regex-checked by `EmailBodyShape`), fact-checker false-positives (counted by `PydanticRetryCount`).
- **Pacing congestion** — `DirectorAmbiguity` counts ticks where the Director had to pick between ≥2 eligible beats; high counts indicate the predicate set is firing too many beats simultaneously.

What it does NOT test (use `analyze_scenario.py` instead):
- Whether a beat is reachable in principle
- Quality bound mismatches
- Dead-code effects / orphan predicates

## Evaluators (16 total)

**Programmatic (8 — cheap, run first):**
- `SessionCompleted` — engine ran without error
- `TerminalReached` — terminal id matches persona's `expected_terminal`
- `BeatCoverage` — count of fired beats
- `CanonHarvested` — count of player_canon keys promoted by extractor
- `PydanticRetryCount` — fact-checker retries (chat agent's only validator)
- `PacingMilestones` — calibration 3-5 turns; bargain entry @reply 4-7; terminal @reply 8-15
- `DirectorAmbiguity` — Director invocations (≥2 eligible)
- `EmailBodyShape` — regex: no leaked envelope, no chat artifacts

**LLM judges (8 — focused, G-Eval-style numbered checklists, anti-slop riders):**

*Voice & tone:*
- `AriaRoleAdherence` — Aria's clinical-warm register across the session
- `OfficeToneAdherence` — Continuity Program Office's neutral institutional tone
- `TransitionMotivation` — sliding-window per Aria reply: motivated continuation vs abrupt pivot

*Game-design pacing — vocabulary borrowed from non-LLM narrative-game critique:*
- `OnboardingClarity` — first 4 emails: voice established, no info-dump
- `StakesEscalation` — quality timeline: rising stakes arc, not random jumps
- `ClimaxLanding` — last 3 fired beats before terminal: did climax land at right moment?
- `Closure` — terminal email + classifier history: does the ending feel earned?
- `ChoiceMeaningfulness` — Director picks: was the chosen beat responsive to player text?

## Personas

Eight personas in `personas/`. Each is a YAML file with `name`, `voice`, `objective`, `archetype`, `quirks`, and an optional `expected_terminal`. The synthetic-player agent reads them and stays in character across the session.

Add new personas by dropping another YAML in `personas/` — the loader picks them up automatically.

## Judge model — different-family caveat

The Qwen 3.6 judge model is from a **different family** than the Gemma 4 SUT, which mitigates self-preference / preference-leakage (arxiv 2502.01534) — this is the primary reason for the two-server split. The remaining caveat: any same-family judging risk is small but not zero (both Qwen and Gemma are trained on overlapping web-scale corpora; judge biases on style/length are documented for all LLMs regardless of family). Mitigations baked into `evaluators.py`:

- Each rubric carries an explicit anti-slop checklist (LLM judges left to themselves PREFER AI-prose tells)
- Each section is truncated to 4000 chars (verbosity-bias mitigation, EQ-Bench precedent)
- Email bodies are stripped of envelope before reaching the judge (shortcut-bias mitigation)
- Judge rubrics use G-Eval style: numbered checklist + per-item reasoning before scoring

## Cross-checking against the static analyzer

After a playtest run, you can compare which beats fired in any session against the analyzer's reachability report. Beats the analyzer says are reachable but no playtest session actually reached are candidates for Director-bias investigation.

```sh
uv run python scripts/analyze_scenario.py        # writes _analysis/<scenario>.json
uv run python -m scripts.playtest                # writes _analysis/playtest_<ts>.json

# Compare in jq:
ANALYZER=$(ls -t npc_service/_analysis/*.json | grep -v playtest | head -1)
PLAYTEST=$(ls -t npc_service/_analysis/playtest_*.json | head -1)
jq -s '
  (.[0].beats | to_entries | map(select(.value.reachable) | .key)) as $reachable
  | (.[1].sessions | map(.final_state.fired_beat_ids) | flatten | unique) as $fired
  | $reachable - $fired
' "$ANALYZER" "$PLAYTEST"
```

Result is the list of "reachable but Director-avoided" beats.

## Performance

- 8 personas × 8 seeds = **64 sessions** at `--parallel 64`, ~25 sends each
- Sessions run concurrently (`max_concurrency=parallel` in `Dataset.evaluate`); each call uses its own per-call `Session` object, no shared mutable module state
- 8 LLM judges per case, mostly one call each (sliding-window judges cap at 6 calls/session) — total LLM-judge cost: ~8-13 calls × 64 = ~600-900 judge calls per run
- Fully local — $0/run with two GGUF servers
