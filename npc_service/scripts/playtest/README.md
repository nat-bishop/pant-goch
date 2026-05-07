# LLM Playtester

Drives full smallville sessions against the production engine + local llama, with the player side synthesised by another `pydantic-ai` Agent given a persona spec. Evaluated with `pydantic-evals` (programmatic evaluators primary; `LLMJudge` is directional only — see caveat below).

## Quickstart

```sh
# 1. Install the optional pydantic-evals dep
cd npc_service
uv sync --extra playtest

# 2. Boot llama-server in another terminal
make llama

# 3. Run the playtester
uv run python -m scripts.playtest --seeds 3 --max-sends 25
# or via the Makefile target:
make playtest
```

Output:
- `npc_service/_analysis/playtest_<timestamp>.json` — full transcripts, tick decisions, final states per session
- pydantic-evals tabular report printed to stdout (per-case scores, aggregates)

## What it tests

The playtester catches what static analysis can't:

- **Narrative coherence** — does Aria's voice stay on-register; do beat transitions feel motivated?
- **Director bias** — beats the static analyzer says are reachable but the LLM Director consistently avoids
- **Agent drift** — chat agent leaking email headers, classifier returning off-spec values, fact-checker false-positives

What it does NOT test (use `analyze_scenario.py` instead):
- Whether a beat is reachable in principle
- Quality bound mismatches
- Dead-code effects / orphan predicates

## Personas

Four starter personas in `personas/`. Each is a YAML file with `name`, `voice`, `objective`, `archetype`, `quirks`, and an optional `expected_terminal`. The synthetic-player agent reads them and stays in character across the session.

Add new personas by dropping another YAML in `personas/` — the loader picks them up automatically.

## LLMJudge caveat

Pydantic-evals best practice is to use a **stronger** model than the system-under-test for judging. The smallville project deliberately runs a single local llama (Gemma 4 E4B), so the same model judges its own outputs. This is **known to be unreliable** — Gemma will rate its own borderline outputs higher than a stronger judge would.

Consequences:
- Treat LLMJudge scores as **directional smoke**: "the rubric is grossly violated" registers; "the rubric is mostly met" doesn't tell you much
- The four programmatic evaluators (`SessionCompleted`, `TerminalReached`, `BeatCoverage`, `CanonHarvested`) are the actual quality bar
- If a stronger judge model becomes available, swap the `model=...` slot on `LLMJudge` in `__main__.py` — no other changes needed

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

- ~5–10 minutes for 4 personas × 3 seeds = 12 sessions, ~25 sends each on Gemma 4 E4B / 8 GB GPU
- Sessions are sequential (the engine uses module globals for state); parallel multi-session would require a stateless engine refactor
- LLMJudge adds ~1 extra LLM call per case; disable with `--no-evals` if you only want raw transcripts
