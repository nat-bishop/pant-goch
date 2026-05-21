# Frozen judge cases

The reactive discipline (EVALS.md): each LLM judge is born from a **real failure
we saw and fixed**, with the failing transition/session frozen here as its first
regression case. Pre-writing rubrics for hypothetical failures yields judges that
track nothing real.

```
cases/
  transitions/   # one before→after turn we saw fire/score wrong, + expected verdict
  sessions/      # one whole playthrough we saw end wrong, + expected verdict
```

## How to freeze a case

1. Find the offending turn/session in `evals/artifacts/<run>/*.json`.
2. Extract its `Transition` / `SessionView`:
   ```python
   from evals.transitions import extract_transitions, session_view
   from evals.session import SessionArtifact
   art = SessionArtifact.load("evals/artifacts/<run>/<persona>-<id>.json")
   # pick the transition for the bad turn, dump it:
   t = extract_transitions(art)[<i>]
   print(t.model_dump_json(indent=2))
   ```
3. Save it here as `<judge>__<slug>.json` with the verdict it *should* get, e.g.
   `transitions/beat_fired_appropriately__walkaway_on_navigation.json`:
   ```json
   {"judge": "beat_fired_appropriately",
    "expected_passed": false,
    "transition": { ... t.model_dump() ... }}
   ```

## Seeds from the 20260520-161349 baseline (pre-fix failures to capture)

These artifacts predate per-turn capture, so re-run a playtest after the Phase-A
change to get transitions, then freeze:

- `beat_fired_appropriately` → **false**: `decided_to_walk_away` firing while the
  player is mid-interrogation at the post office (they never said they were quitting).
- `beat_fired_appropriately` → **false**: `helen_wont_take_your_call` firing in a
  session that never visited Helen / Knighton.
- `prose_depicts_beat` → **false**: `phoned_iwan_from_the_vicarage` firing in a
  scene whose prose contains no phone call.
- `ending_earned_narratively` → **false**: `run_out_of_the_village` reached by a
  session that already had `tom_named_a_vehicle` + `dylan_admits_he_knows`.
- `choice_meaningfulness` → **low (1–2)**: the vicarage object-hunt turns
  ("examine the pamphlets", "look at the staircase").

A future `judge_run --frozen` mode can load these and assert the expected verdict
(attaching the judge in code, since the evaluator's `model` is a live object).
