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

## validate-facts-on-scene

**WHAT.** Any beat with both `kind: scene` and `spec.validate_facts: true` is `ERROR`.

**WHY.** `validate_facts: true` runs the fact-checker agent on the chat agent's output, requesting a retry on any miss. That's one extra LLM call per render attempt. On a scene, the beat re-fires every tick until `exit_when` is satisfied — so the fact-checker runs every tick. **Unbounded LLM cost per scenario run.** Reserve `validate_facts` for `kind: one_shot` beats (terminals, pivotal forks).

**SCOPE.** Sound. The Pydantic model allows the combination (`BeatSpec._validate_field_combinations` doesn't catch it); lint catches it.

**FALSE POSITIVES.** None — the combination is always wrong.

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

**SCOPE.** Lexical only. The lint catches quoted-token slips. Semantic bans (`"any acknowledgement that the Program means replacement, elimination, or firing"`) remain a playtest concern — `validate_facts: true` + the fact-checker + integration tests are the catch-net.

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
