# Lint rules

The lint script (`lint_scenario.py`, sibling of this file) emits findings prefixed with `ERROR` / `WARN` / `INFO` and ends each line with an anchor link into this file. One section per check, structured as **WHAT** (the rule), **WHY** (the invariant it protects), **SCOPE** (sound vs heuristic; defensive vs corrective), **FALSE POSITIVES** (known cases). Section anchors match the slugs the lint script outputs.

The lint script loads `npc_service/scenario.yaml` via `npc_service.models.load_scenario`, so every `Scenario._cross_references` violation surfaces as a parse error before lint logic runs — this file does not duplicate that surface (id uniqueness, beat-id prefix=thread, classifier map symmetry, axis-arm membership, predicate ref existence).

Severity:
- `ERROR` — provable invariant violation; non-zero exit.
- `WARN` — heuristic finding with non-zero false-positive risk; non-zero exit unless `--no-warn-fail`.
- `INFO` — observation; never affects exit.

## scene-write-collision

**WHAT.** For every quality / commitment key written by a `kind: scene` beat via a literal effect (`effect.value` set), the lint walks every other beat that also writes that key. Each colliding pair (the scene + the other beat) must have a `Not_(BeatFired ...)` exclusion in at least one direction — either the other beat excludes the scene from having fired, or the scene excludes the other from having fired. Pairs missing both directions emit `ERROR`.

**WHY.** Scene beats re-fire every tick that their preconditions hold and `spec.exit_when` is not yet satisfied (they don't enter `state.fired_beat_ids` on their own until exit fires; see `models.py:Beat.kind`). If the scene literal-writes quality Q to value A, and another beat later writes Q to value B without a mutex, the scene's NEXT tick will fire again and clobber Q back to A. The author's intended write is silently lost.

**SCOPE.** Defensive — production scenes that do literal writes (e.g. `aria.bargain_scene` writing `bargain_state=offered`) gate them with `when:` predicates so the second fire is a no-op. This check guards future scenes that wire literals without that guard.

**FALSE POSITIVES.** None known. The check requires both directions to have NO mutex, which is the actual unsafe case.

## terminal-overlap

**WHAT.** For every pair of `ends_session: true` beats, the lint extracts conjunctive features from each beat's preconditions (quality range constraints, `BeatFired` / `Not_(BeatFired)` sets, `RecentClassifierValue` constraints, `PlayerRepliesToThreadGte` thresholds, `QualityEquals` values). It then asks whether some single state could satisfy both beats' preconditions simultaneously. The check is **skipped silently if the scenario has fewer than 2 terminals**.

Severity tiered:
- `WARN` — overlap on hard constraints (quality ranges, fired-beat sets, etc.) with no authored mutex. Probably a missing exclusion.
- `INFO` (soft) — overlap only via disjoint `recent_classifier_value match=any` value sets on the same quality. The Director-discriminates-by-recent-texture pattern: a multi-slot window can co-satisfy both terminals (e.g., `[paternal, cold, neutral]` satisfies both `any-of-paternal` and `any-of-cold`), and the Director picks. Documented intentional design.
- `INFO` (mutex) — at least one terminal has `Not_(BeatFired other_terminal)` in its preconditions. Author-authored fallback chain; with `ends_session: true` on both, only one fires per run.
- `INFO` (unanalyzable) — at least one terminal has a precondition tree containing a `Not_(...)` over something other than `BeatFired`, or a top-level `Or_` not covered by the analyzer. Manual review needed.

**WHY.** Two terminals eligible at the same state lets the Director pick. Sometimes that's intentional (`integration` vs `exile` distinguishing by recent texture); sometimes it's a missing mutex. Surfacing the overlap forces the author to confirm intent.

**SCOPE.** Heuristic. The intersection analysis is sound on the predicate kinds it understands (quality ranges, beat-fired sets, recent-classifier value sets in `match=any/all` modes, replies thresholds, quality_equals); when it encounters something it can't model, it bails out with `INFO`. Implementation in `lint_scenario.py:features_can_overlap`.

**FALSE POSITIVES.**
- `recent_classifier_value match=any` with disjoint value sets reads as soft overlap (correct: a multi-slot window can co-satisfy). The lint downgrades to INFO; you're not nagged unless the overlap is hard.
- Anchored predicates inside `Or_` (the `*calibration_done` anchor) are correctly skipped from the analysis as non-discriminating — Or_ relaxes constraints, never tightens them, so a shared Or_ subtree doesn't disjoin two terminals.

## pre-authored-post-boot

**WHAT.** Any beat with `spec.pre_authored_prose` set whose preconditions include a `player_replies_to_thread_gte` predicate with `value ≥ 1` is `ERROR`.

**WHY.** Pre-authored prose ships verbatim — no chat agent call. The chat agent's render path hard-asserts `player_body` (`main.py:_fire_beat`) for chat-driven beats; the boot tick has no `player_body`, which is the only state where pre-authored makes mechanical sense. A pre-authored beat that fires post-boot is the linearity-trap antipattern: every player input collapses back to the same fixed output, defeating the engine's authoring economics.

**SCOPE.** Sound on the basic case (preconditions explicitly require ≥ 1 reply). A pre-authored beat with no preconditions is the boot pattern (correct, doesn't fire). A pre-authored beat with `BeatFired X` preconditions but no reply-count predicate isn't caught — that's a complex case (X must have fired post-boot for the beat to be eligible, but the lint heuristic doesn't trace that). In practice, the simple case is what authors hit.

**FALSE POSITIVES.** None known. If you have a legitimate use case for pre-authored prose post-boot, the engine's `player_body` assertion will fail at runtime regardless — this lint just catches it at edit time.

## forbidden-lexical

**WHAT.** For each beat, the lint extracts quoted tokens (single- or double-quoted phrases like `'monster'`, `"we are sorry"`) from the assigned NPC's `base_voice_forbidden` list, then searches the beat's `spec.intent` and each entry of `spec.must_include_facts` for whole-word case-insensitive matches. Hits emit `WARN`.

The lint **skips `spec.must_not`** — that list naturally contains forbidden phrases (it IS the deny list).

**Allow-list signal detection.** Some `base_voice_forbidden` entries name PRESCRIBED phrases rather than forbidden ones (e.g., CPO's `"any reference to the Counterpart as a person ('she', 'her'); use 'the Counterpart' or 'Counterpart ARIA-7'"` — the `;` segment after `use` names allowed phrasing). The lint splits each entry on `;` and skips segments containing `anything but`, `anything except`, ` use '`, ` use "`, `always `, or ` only `. Quoted tokens in those segments are NOT extracted as forbidden.

**WHY.** If the dossier bans "monster" and the beat's `intent` says "Aria addresses the player as a monster," the chat agent will likely echo the word — the prompt itself contains it. Same for `must_include_facts`: phrasing the directive with forbidden words tells the agent those words are required.

**SCOPE.** Lexical only. The lint catches quoted-token slips. Semantic bans (`"any acknowledgement that the Program means replacement, elimination, or firing"`) remain a playtest concern — the fact-checker (which runs automatically on any beat declaring `must_include_facts` / `must_not`) plus integration tests are the catch-net.

**FALSE POSITIVES.** Possible when:
- A ban entry uses an idiosyncratic structure that the allow-list signal heuristic doesn't catch. Example: "the words 'X' and 'Y' (acceptable in epilogue mode only)" — no `use`/`but`/`only`/`always` keyword, so tokens get extracted as forbidden. Currently no such entries exist.
- The `spec.intent` legitimately needs to NAME a forbidden phrase to instruct the agent NOT to use it. In that case, move the instruction into `spec.must_not` instead — the lint correctly skips that field.

## classify-on-silent-npc

**WHAT.** Any beat whose `npc` has `replies_enabled: false` (institutional senders like the Continuity Program Office) cannot have any `classify` directives in `spec.effects` — `ERROR`.

**WHY.** Institutional senders don't get player replies (the frontend hides the composer). The classifier runs on the next player reply to the beat's thread; with `replies_enabled: false`, no reply ever arrives. The classify effect is dead authoring — the verdict will never write.

**SCOPE.** Sound. The model permits the combination; lint catches it.

**FALSE POSITIVES.** None — the combination is always wrong by construction.

## reachability-contradiction

**WHAT.** For each beat, the lint walks the precondition tree and accumulates `BeatFired X` references into `required` (when reached outside `Not_`) and `excluded` (when reached inside one `Not_`). If `required ∩ excluded` is non-empty, the precondition is unsatisfiable — `WARN`. `Or_` subtrees are deliberately skipped from this walk to avoid false positives on intentional branching.

**WHY.** A beat that requires `BeatFired X` AND `Not_(BeatFired X)` in the same conjunctive scope is unreachable — the predicate can never be true. Per Twine / Ink / Yarn precedent, reachability lint is standard in narrative tooling; this is the cheap-and-obvious subset.

**SCOPE.** Heuristic. Catches simple contradictions only:
- Direct `[BeatFired X, Not_(BeatFired X)]` at the top level.
- Nested `BeatFired X` inside an outer `Not_(Not_(BeatFired X))` chain (treated as positive, then conflicting with another `Not_(BeatFired X)` at the top).

Does NOT catch:
- Contradictions across `Or_` branches (skipped to avoid flagging intentional fallbacks).
- Quality-range contradictions (`quality_gte X 5.0` AND `quality_lt X 4.0`).
- Cycle contradictions (X requires Y fired; Y requires X fired).

Full path coverage stays a manual exercise + the live-run check.

**FALSE POSITIVES.** None known on the current scenario. Possible if an author intentionally writes `[BeatFired X, Not_(BeatFired X)]` to hard-disable a beat — but that pattern is itself an authoring smell (use a different exclusion).

---

# Static analyzer rules

The static analyzer (`analyze_scenario.py`, sibling of this file) does bounded forward state-space exploration to determine which beats are reachable, what bounds each quality can take, and which effects/predicates are dead. Findings come through the same `Finding` channel as the spec lint above. The analyzer is heavier (10s–60s on production-sized scenarios), so the per-rule lint tests in `test_lint_scenario.py` skip it via `run_analyzer=False`; the analyzer has its own test module (`test_analyze_scenario.py`).

The analyzer is **sound for unreachability**: when it says a beat is unreachable, it is — modulo state-space truncation, which downgrades the claim from `ERROR` to `WARN` and emits an `analysis_incomplete` finding alongside.

## unreachable-beat

**WHAT.** Across all states explored by bounded forward BFS from the boot state, this beat never fires. ERROR when exhaustive; WARN when the analysis hit `--max-states` (the unreachable claim is provisional in that case).

**WHY.** Authoring leftovers — a beat referenced from no other beat's preconditions, or gated on a quality threshold no effect can reach, or carrying a contradicting precondition pair. Dead beats clutter the scenario and confuse readers.

**SCOPE.** Sound (over-approximating Director and classifier non-determinism, both safe directions for this question). Truncation can produce false unreachable claims; the `analysis_incomplete` finding flags it.

**FALSE POSITIVES.** Provisional WARNs under truncation are common on production-sized scenarios. The production scenario truncates at the default `max_states=300_000` (deepest beats — those gated on long quality-build-up chains plus dispatched secondary threads — sit past the BFS frontier). The `analysis_incomplete` finding flags the truncation; raise `--max-states` (analyzer is RAM-bounded, not time-bounded) or accept the WARN if manual playtest confirms reachability.

## unreachable-terminal

**WHAT.** Same shape as `unreachable-beat`, but specialized for `ends_session=True` beats. Separated because terminals never being reachable is more severe than scenes — a terminal is an authoring contract (the player CAN end the game this way), not a side-path.

**WHY.** A terminal that no path reaches is a broken promise to the author of the scenario design. Surfaces drift between predicate thresholds and effect magnitudes.

**SCOPE / FALSE POSITIVES.** Same as `unreachable-beat`.

## orphan-effect

**WHAT.** A literal or classify effect targets a quality that no predicate reads (`QualityGte`, `QualityLt`, `QualityEquals`, `RecentClassifierValue`) AND that quality isn't `surface_to_chat=True`. The write is dead — modifies state nothing checks.

**WHY.** Authoring drift — a quality used to gate something, the gate got removed, the writers stayed. Or a typo in the quality id.

**SCOPE.** Sound. `surface_to_chat=True` qualities are explicitly exempted: those modify chat-agent rendering rather than branching, so a "no predicate reads" check would false-flag intended behavior.

**FALSE POSITIVES.** None known once the `surface_to_chat` exemption is applied.

## orphan-predicate

**WHAT.** A predicate references a quality that no effect writes (literal or classify, scenario-level or per-beat). The predicate can never become true under any state path.

**WHY.** Dual to `orphan-effect` — the writer was removed but the gate stayed. Equivalent to a precondition that's always false.

**SCOPE.** Sound for quality predicates. Canon-key references are NOT flagged: the extractor's outputs are LLM-driven and not statically authorable, so we can't prove a key is never harvested.

**FALSE POSITIVES.** None known on quality predicates.

## quality-bound-mismatch

**WHAT.** A predicate threshold (e.g., `QualityGte 8`) lies outside the proven (min, max) bound for that quality across all reachable states. Issued as ERROR only when the analysis is **not truncated** — under truncation, bounds are partial and the threshold may still be reachable past the cap.

**WHY.** Calibration drift — predicate threshold and effect magnitudes diverged. Common when an author tunes one and forgets the other.

**SCOPE.** Sound when not truncated (bounds are exact across the explored state space). Suppressed under truncation to avoid false positives where the bound just hadn't been pushed yet.

**FALSE POSITIVES.** None when the analysis runs to exhaustion.

## analysis-incomplete

**WHAT.** Bounded forward BFS hit `--max-states` before reaching a fixed point. Reachability claims for the remaining unreachable beats are provisional; quality bound checks are suppressed entirely.

**WHY.** The analyzer is sound but bounded. Production-sized scenarios with many classifier axes × beats can blow past 100k states. Rather than silently report incomplete analysis as exhaustive (which would risk false ERROR-level "unreachable" claims), the analyzer downgrades and surfaces the truncation explicitly.

**SCOPE.** Always WARN. Authors can either raise `--max-states` (analyzer is RAM-bounded, not time-bounded — ~30 s per 20k states on the production scenario), or treat the WARN as expected and rely on playtest coverage for the rest.

**FALSE POSITIVES.** N/A — this finding IS the disclosure.

---

# Scaling notes

The analyzer is explicit-state forward BFS. State-space size is the budget. The current state signature already coarsens aggressively:

- **Qualities** filtered to those any predicate reads, then bucketed into predicate-equivalence classes (`_bucket_quality_value` in `analyze_scenario.py`). Two `creature_grievance` values that fall in the same gap between authored thresholds collapse to one BFS state.
- **Reply counts per thread** capped at `max(threshold) + 1`.
- **Recent classifier history per axis** capped at the largest authored window AND coalesced to a multiset of `(value, count)` pairs. The evaluator (`spine.py:_evaluate_recent_classifier_value`) reads only counts, never order, for every `match` mode (`any`/`all`/integer N), so multiset-equivalent histories collapse to one BFS state.
- **Reply-target enumeration** dedups beats by `(thread, classify-signature)` since the analyzer's synthetic replies don't carry `beat_responded_to`.
- **State clone** is a hand-rolled targeted copy (`_clone_state`); only mutated fields are deep-copied.

## What still grows as the scenario grows

| Dimension | Driver | Notes |
|---|---|---|
| `qualities` (post-bucket) | new qualities, more thresholds per quality | Linear-ish; bucketing keeps it tame. |
| `fired_beat_ids` ∪ `ever_fired_beat_ids` | new beats | Combinatorial subset, but mutex-pruned in practice. ~50 beats: probably fine. ~100+: cliff. |
| `canon_keys` | new `PlayerCanonExists` predicates | ×2 per referenced key. Slow growth. |
| `thread_reply_counts` | new threads | One new dimension per thread, capped per-thread. Slow. |
| `recent_classifier` | new classifier axes × value sets × window | **Multiplicative,** but mitigated by multiset coalescing — distinct *orderings* no longer count, only multisets. Each new axis is still a ~5–10× inflation rather than ~25×. |
| Verdict cartesian per turn | classifier axes per beat | Per-tick branching factor. 2 → 5 axes per beat is a 50-100× inflation in turns-to-explore. |

The two real cliffs: **classifier axes** (multiplicative everywhere) and **beat count past ~50-75** (combinatorial powerset).

## Deferred optimizations, in order of cost

These are NOT implemented today. Reach for them in this order when the analyzer takes more than ~10 minutes on `make lint-scenario`.

1. **Goal-directed backward search.** Today's BFS explores everything from boot and asks "did beat X fire?" When only 1-2 beats are at risk, this is wasteful. Per at-risk beat: walk preconditions backward (regression analysis) and check whether predecessors trace back to boot. Used in symbolic model checkers under names like "regression." Bigger refactor (~300 lines). Right answer when state space genuinely exceeds enumeration capacity but only a handful of beats need verifying.

2. **Symbolic methods (BDDs / SAT-based reachability / IC3-PDR).** What real model-checkers (nuXmv, Spin, hardware verification tools) use when explicit enumeration fails. Massive refactor and adds a solver dependency. Not worth it until the scenario has 100+ beats and millions of reachable states.

3. **Verdict-ordering fallback for new predicate types.** Today's multiset coalescing on `recent_classifier` is sound for every current `match` mode (`any`/`all`/integer N) because the evaluator reads only counts. If a future predicate type is added that *does* read verdict ordering (e.g., "the most recent verdict is X"), state_signature will silently lose paths. Mitigation: detect the new predicate type at profile-build time, mark the affected axes, and fall back to ordered tuples for those axes only. Localized change (~30 lines).
