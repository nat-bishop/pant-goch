# HANDLER — Playthrough Audit (session `eabcac5420ca`, 2026-05-12)

> **TL;DR.** The session played five cycles. The trace shows the engine working as designed — the engine isn't broken. The *game* is broken. Three structural problems combine to produce the "linear, telegraphed, decisions don't matter" feeling: the **option space is tiny** (3–4 options chosen from a 7-archetype catalog, mostly gated to one per cycle by `extraction_stage`); **forced events pre-narrate their own outcome** (the option's prose is the dispatch's punchline); and **resolution is fully deterministic with no hidden information** (you can read the option text and predict the dispatch word-for-word). The user's instinct is correct: hand-authored static branches would be a strict upgrade today, because there is no information the LLM is producing that the player can't already infer from the option title.

---

## 1. What was actually played

**Session.** `eabcac5420ca` — trace `019e1da7df549ffb62a318a4aa86b385`. Mission `LYNX-EXFIL`, 6-cycle limit. Played to cycle 5 (KESTREL sacrifice → did not see resolution in this trace; spans show it was still mid-pacing when telemetry ends).

**Player choices, cycle by cycle:**

| # | Authorized | Forced event present? | Cover delta (LYNX / KESTREL) |
|---|---|---|---|
| 1 | `HOLD` | yes — `burn_redfield` (declined) | -1 / -1 (passive pressure) |
| 2 | `DIVERSION_BY_AGENT(KESTREL)` | yes — `burn_redfield` (declined again) | 0 / -3 |
| 3 | `ACTIVATE_SLEEPER_FOR_PAPERS` (forced `burn_redfield`) | yes (accepted) | 0 / 0 (extraction_stage 0→2) |
| 4 | `LYNX_MOVES_TO_SAFEHOUSE` | none | -1 / 0 |
| 5 | `DIVERSION_BY_AGENT(KESTREL)` (forced `kestrel_sacrifice`) | yes (accepted) | +protected / -all |

The player resisted the forced `burn_redfield` for two cycles, ate the cover decay, then capitulated. Then in the late game accepted the `kestrel_sacrifice` immediately. Both forced events were accepted in the order the engine telegraphed them. There was no point at which the world *surprised* the player.

---

## 2. Trace-level observations

### 2a. The engine generated 14 LLM calls for 5 cycles. It needed ~10.

Per cycle the trace shows: SoP, drafter, renderer (post-auth), prefetch SoP, prefetch drafter. With a fully warm prefetch the live cycle should reuse SoP and drafter — only the renderer should be a fresh LLM call. In this session the prefetch raced the player and **lost twice** (cycles 4 and 5), so both the prefetch and the live drafter ran. Cycle 5 has **two `drafter_call` spans** (`80eb7e61753e8f78` from prefetch, `4a179f9acc82754e` from the live cycle); they generated nearly identical option sets — the second one just reordered the options. ~25 s of LLM compute thrown away.

> **Root cause.** `main.py` line 447 only consumes a prefetch result if `rt.prefetched_sop_text is not None` *at the moment of consumption*. If the player clicks Continue before the background prefetch task populates that field, the live SoP/drafter LLMs run anyway. The prefetch task isn't cancelled — it runs to completion in the background and writes to dead state.

### 2b. The same dramatic question fired in three consecutive cycles.

`dramatic_questions` rotation in `lynx_exfil.yaml` matches the first satisfying predicate. While `extraction_stage == 0` (cycles 1, 2, 3 in this session), the SoP repeatedly opened with **"Where do we surface the papers, and at what cost?"** Cycle 3's SoP was: *"Kestrel's cover integrity has degraded, complicating the immediate perimeter. Where do we surface the papers, and at what cost?"* — same question, prepended with one pivot line.

The player read the same dramatic prompt three times in a row before the state shape changed enough to unlock a different question. Two of the three stage-0 question variants in the YAML (`kestrel_warns_lynx`, `cycles_remaining <= 3`) never fired because their predicates didn't pass.

### 2c. Repetition phrases across dispatches

Pulled from the actual rendered text:

- *"perimeter remains tightly constrained"* — cycles 3, 4
- *"window for exfiltration continues to narrow"* — cycles 3, 4
- *"the eastern corridor remains clean / clear"* — cycles 3, 4, 5 (literally a `standing_orders` line so the model parrots it)
- *"sustained external pressure from the Romanian counter-intelligence directorate"* — cycles 3, 4
- *"protective cover"* (engine-isms leaking through the narrative voice) — cycles 1, 2, 3, 4, 5

The model has very little new information to chew on between cycles, so it repeats. The standing orders line *"The eastern train corridor is clean"* is being treated as a sentence to insert verbatim, not a fact to honor.

### 2d. The dispatch is the option's prose, restated.

Cycle 3 example, side-by-side:

| Option prose (drafter) | Dispatch (renderer) |
|---|---|
| "Redfield provides the papers; the asset is confirmed dead, and the safe-house is located." | "The activation of Redfield secured the target documentation. Asset termination confirmed; the papers are in hand. […] The safe house location has been identified, positioning LYNX for immediate exfiltration." |

The renderer added one sensory beat ("A tram passed the designated observation point") and otherwise transposed the same facts in identical order. The player **already saw** this outcome described in the option. The dispatch confirms — it doesn't reveal.

### 2e. Forced events telegraph their own resolution.

The drafter system prompt for `burn_redfield` includes a `must_communicate` block: *"REDFIELD is named, and named as dead by the dispatch's close. The papers are in LYNX's hand and the safe-house is confirmed."* The drafter is required to write an option whose prose contains those facts. The dispatch then does the same. The renderer's role here is decorative; the engine has already decided what the prose must say.

The same applies to `kestrel_sacrifice` ("KESTREL deliberately accepts exposure to create clear lanes for LYNX's exit"). The player reads that, picks it, and the dispatch describes KESTREL deliberately accepting exposure to create clear lanes. There is no information the player gains by clicking.

### 2f. Voice is inconsistently held

The `acting` value selects the voice. When `DIVERSION_BY_AGENT(KESTREL)` resolves, the renderer should write in KESTREL's first-person justifying voice. Cycle 2's dispatch did that well: *"The necessity of a decisive move became abundantly clear when I saw the specific cadence of their surveillance — it was tightening around the edges, requiring a kinetic response that was almost reckless…"* — that's recognizably KESTREL.

Cycle 3 (ACTIVATE_SLEEPER, acting = "operations") wrote in the desk register, correctly. But cycle 5 (KESTREL sacrifice) opened with *"The moment I initiated the overload, knowing what I had already paid for that old back channel in Vienna…"* — that's KESTREL, but the same prompt also produced cycle 4's dispatch *"LYNX executed the movement to the designated safe house. Repositioning was achieved under immediate threat from tightening local sweeps…"* which is **operations voice for a LYNX action**. Cycle 4's option was `LYNX_MOVES_TO_SAFEHOUSE` (no agent binding because the archetype params is `[]`), so `_acting_for` returned "operations" and the LYNX voice was lost. The mechanic produces a desk-narrated movement of a named agent — which reads as bureaucratic and removes the player from the agent's POV.

### 2g. Cover decay is monotonic and silent

The engine emits `effects_applied` and `cycle_closed` SSE events with `delta_descriptors`. The frontend turns those into the RESULT footer. But the *prose* never names the change in concrete terms — only in standing-vocabulary descriptors (`cover intact → cover degraded → cover compromised → cover blown`). Once a cover bucket flips, the next 3 cycles all describe LYNX as "cover degraded" even though the underlying number is sliding from 5 → 4 → 3 → 2. **The player can't feel the trend** unless they read the small footer text. The most important state in the game is invisible to the prose layer.

---

## 3. Structural problems with the design

### 3a. The option space is too small to surprise

`lynx_exfil.yaml` declares **7 archetypes**:

```
ACTIVATE_SLEEPER_FOR_PAPERS  → stage 0 only
RETRIEVE_PAPERS_VIA_AGENT    → stage 1 only, costs cover
DEAD_DROP_PAPERS             → stage 1 only, no effect
LYNX_MOVES_TO_SAFEHOUSE      → stages 1–2, -1 LYNX cover
RUN_EXFIL                    → stage 2 only, terminal
DIVERSION_BY_AGENT           → any stage, costs agent
HOLD                         → always available
```

In stage 0, the legal catalog is `{ACTIVATE_SLEEPER, DIVERSION(LYNX), DIVERSION(KESTREL), HOLD}`. **That's all 4 options every single stage-0 cycle**, plus `burn_redfield` is a forced event so one of them is engine-mandated. The drafter has no real authoring freedom — it picks the same 4 options with mildly varied prose. Once `burn_redfield` resolves, you skip stage 1 entirely and land in stage 2, where the catalog is `{LYNX_MOVES_TO_SAFEHOUSE, DIVERSION(LYNX), DIVERSION(KESTREL), RUN_EXFIL, HOLD}`. The progression is essentially a 3-state finite-state machine: papers? → moved? → exfilled?. The drafter is dressing up the FSM transitions.

### 3b. Forced events make decisions feel like clicking through a story

A forced event injects an option with a fixed effect AND a fixed prose template. The player can decline once or twice but the same forced event will requeue every cycle until accepted (its `when` predicate doesn't change quickly). So from the player's POV, the game is saying "you will burn Redfield eventually — the only question is whether you eat 2 cycles of cover decay first." There is no real choice: there's a delay timer on a story beat the engine has decided will happen.

This is the core problem. The forced-event system was built to ensure dramatic moments fire reliably, but in practice it converts dramatic moments into mandatory branches. The player isn't *choosing*; they're *consenting at their pace*.

### 3c. Resolution is fully deterministic with no hidden information

Every effect is written into the YAML. `DIVERSION_BY_AGENT` always costs 3 cover and always sets `cover_protected = true`. `ACTIVATE_SLEEPER` always sets `extraction_stage = 1` (or 2 via the forced event). There is no roll, no opposed check, no MUNTEANU AI, no chance the diversion fails, no chance the dead drop is intercepted. The state machine is a flowchart, and the player is reading the flowchart aloud to the LLM and asking it to pretty-print the result.

There is also no asymmetric information. Everything the engine knows, the option text states. There's no "you don't know if MUNTEANU has named her yet — you'll find out when REDFIELD's dispatch arrives." The world has no internal events the player learns of through dispatches.

### 3d. Standing orders become repetition, not constraints

`standing_orders` are appended to all three system prompts. Lines like *"The eastern train corridor is clean. That extraction route has not been compromised"* end up echoed verbatim in dispatches because the model is keeping context coherence. By cycle 5 the train has been called clean four times. A standing order should be a constraint on what *can* be true, not a sentence to surface.

### 3e. Passive pressure is silent

`world_attention_tightens` (the passive event that fires when no progress was made) decrements both agents' cover by 1. Its only player-visible signal is the cover descriptor flip. There's no event beat — no *"you held; MUNTEANU's people checked LYNX's tram pass at the Piața Unirii stop. He noticed the laminate was new."* The penalty for HOLD is real but invisible until it's catastrophic.

### 3f. The "memo + options + dispatch" rhythm is too slow for too little payload

A cycle takes ~45 seconds of wall-clock LLM time (SoP ~5 s + drafter ~20 s + renderer ~15 s + prefetch overhead). What the player gets:
- 90 words of SoP that re-states current state in fancier prose.
- 4 option titles (~8 words each) with one-line cost annotations.
- 90 words of dispatch that confirms the chosen option's prose.

That's ~270 words of LLM-generated text per cycle for one binary-ish decision. The same beat in *80 Days* or *Sunless Sea* would be ~80 words and offer a real branching reveal.

---

## 4. Why the user's instinct ("hand-authored would be better") is right

A hand-authored cycle could:
- Use the same 7 archetypes but **author 5–10 prose variants per (archetype, state-bucket)** with handcrafted reveals, twists, and consequences.
- Insert hand-tuned "you didn't expect this" moments — REDFIELD's wife appearing in an unrelated dispatch, MUNTEANU's photograph being smudged, the eastern train being subtly wrong.
- Avoid the LLM's tendency to copy the option text into the dispatch.
- Cost ~0 ms per cycle and feel snappier.

The LLM is currently **not generating content the engine couldn't author**. The voice differentiation between LYNX and KESTREL is real and good, but it's the only LLM-original contribution per cycle. Everything else is restatement.

The bar for "the LLM helps" is the LLM producing content the engine **cannot pre-author at scale**. Right now: state translations, option prose, and dispatch retellings are all derivable from the engine state. The LLM is decorating, not authoring.

---

## 5. Where the LLM should actually be working

If the engine continues with the current shape (deterministic state + LLM prose), the LLM should be doing things hand-authoring can't:

1. **React to the player's history with named callbacks.** The model should know "you held twice in a row; KESTREL's last cable was sharper than usual" and write it. Currently the user prompt only carries the last 1–2 dispatches and the current state. Add `recent_player_pattern: ["HOLD", "DIVERSION(KESTREL)", "ACTIVATE_SLEEPER"]` and let the model react to the *shape*, not the latest delta.
2. **Generate the contents of asymmetric reveals.** "REDFIELD's last brush pass note read: ___" — the engine knows REDFIELD is alive and a brush pass occurred; the LLM authors the *content* of the note. Each playthrough produces a different note. That's content the engine genuinely can't pre-author.
3. **Voice the absent characters.** MUNTEANU never appears in dispatches — but a transcript fragment of an intercepted Romanian-side cable, generated when the player has high enough Soviet-side intel, would be fresh content per playthrough.
4. **Write intel reports that are wrong.** Have the SoP occasionally produce a memo whose framing is incorrect (LYNX believes she's clean when she isn't) so that the dispatch reveals the discrepancy. The model is given an "intel confidence" field; it generates prose that overcommits when confidence is high. This is content that requires inference.

Today the LLM is not asked to do any of these.

---

## 6. Suggested mechanic injections

Numbered for traceability into the HTML artifact and prioritized: ★★★ = highest impact / lowest cost, ★ = speculative.

### 6.1 ★★★ Replace forced events with **opportunity windows** (not mandates)

Instead of "this option WILL appear and you must pick it eventually," forced events become **time-limited opportunities** that vanish if not taken within their window. `burn_redfield` is offered for cycles 1–2. If you don't take it, REDFIELD's next dispatch reveals he's been compromised by MUNTEANU's sweep and is no longer reachable. The choice now has a real cost on both sides.

> **Engine change.** Forced events get a `window_cycles: 2` field. After expiry, fire a *consequence* passive event (`redfield_compromised`) that closes the option and rewrites the world. The drafter no longer requeues an expired forced event.

### 6.2 ★★★ Add **opposed checks** with hidden state

MUNTEANU gets his own quality: `opposition.investigation_progress` (0–10). It ticks up on every cycle by 1 (faster if LYNX's cover dropped, slower if KESTREL drew attention this cycle). The player **never sees this number directly** — only the SoP describes it: *"the perimeter feels tighter than the file suggests"* at 4, *"a tram he should not have been on"* at 7, *"a knock at the door of an apartment LYNX has never visited"* at 9.

Most archetypes resolve against `investigation_progress`: `RETRIEVE_PAPERS_VIA_AGENT` succeeds if `investigation_progress < 6`, partial-success if 6–8, fails-and-burns-an-asset if 9+. The player has to read the SoP for clues and gamble.

> **Engine change.** Add a third effect class `effects_if_opposed_check_fails` that fires when an opposed check rolls below threshold. The drafter writes prose that names the check ("if MUNTEANU is closer than we think, this exposes LYNX") so the player knows they're rolling.

### 6.3 ★★★ Make HOLD interesting with **incoming reports**

When the player picks HOLD, instead of "no progress, cover decays," the engine fires a deterministic dispatch from a **non-acting agent or asset** with new information: a brush-pass from REDFIELD, an intercept from the Vienna network, a Treasury cable from MERIWETHER. HOLD becomes "I'm not moving — I'm listening." Its mechanical cost (cover -1) stays, but its prose payload becomes the most information-dense beat in the game.

> **Engine change.** Add an `incoming_dispatches` queue to State. HOLD pops the next queued dispatch and renders it. The queue is authored per scenario as a sequence of "tip-off" events.

### 6.4 ★★ Decision-revealing dispatches: **reveal vs. confirmation**

Forbid the renderer from restating the option's prose. The dispatch must contain at least one fact the option text did not have. Authored: every archetype gets a `reveal_pool: [...]` of small concrete details (a courier's coat smell, a tram conductor's nervous tic, a name in a phone ledger), and the renderer must surface one. The player learns about the world by acting on it.

> **Engine change.** `archetype.reveal_pool: List[str]` and inject a chosen sample into the renderer's user prompt as `MUST INCLUDE one of these details`. Pool entries can be flagged as `sets_quality: investigation_progress -= 1` so the reveal also matters mechanically.

### 6.5 ★★ Branching consequences from option **combinations**, not single picks

A choice's effect depends on the previous 1–2 choices. `RUN_EXFIL` after `ACTIVATE_SLEEPER + LYNX_MOVES_TO_SAFEHOUSE` is the clean path. `RUN_EXFIL` after two HOLDs is a panic exit and costs more cover. `DIVERSION_BY_AGENT(KESTREL)` after `kestrel_warns_lynx` triggers a paid-twice penalty (KESTREL's cover -5 not -3).

> **Engine change.** `archetype.effects` becomes context-sensitive: `effects: [{when: "fired:kestrel_warns_lynx", then: ...}, {when: "default", then: ...}]`.

### 6.6 ★★ Intel cards: a player-side **information economy**

The player accumulates intel cards through HOLD, DIVERSION, and forced events. Each card is a fact about MUNTEANU, REDFIELD, the Vienna network, or the train schedule. Spending an intel card before authorizing an option boosts its success probability or adds an effect ("redirect MUNTEANU's surveillance for one cycle"). Cards are LLM-generated lines of prose; spending them pulls the line into the next dispatch.

> **Engine change.** State gains `intel_cards: List[IntelCard]`. The drafter's option list adds a `spend_card?: card_id` slot per option. The renderer prompt receives the spent card and must work it into the dispatch.

### 6.7 ★★ Per-agent trauma / morale state

Agents accumulate trauma when their cover drops or they witness REDFIELD's death. High trauma changes the agent's voice prompt (KESTREL's sentences fragment more), changes which archetypes they'll perform, and can trigger `agent_refuses` events ("LYNX missed the brush pass — she didn't show"). Adds an interpersonal axis to the resource economy.

> **Engine change.** `agent[*].trauma: int` quality. Voice register block in the renderer prompt becomes `voice_register + trauma_modifier`.

### 6.8 ★ Multi-agent options: **coordinated archetypes** for the late game

Add archetypes that bind two agents at once: `KESTREL_DRAWS_FIRE_FOR_LYNX_RUN`, `LYNX_AND_KESTREL_SPLIT_THE_PAPERS`. These appear only when both agents have specific cover thresholds. The drafter has more axes to vary against.

> **Engine change.** `archetype.params: [agent, agent2]`. Catalog expansion handles 2-tuple bindings.

### 6.9 ★ A second opposition character with conflicting interests

Add a Treasury-side character (the deputy chief MERIWETHER warned about). MERIWETHER cables surface periodically with directives that conflict with field reality: "Treasury wants LYNX out by next cycle. Their reasons are not in the file." A cycle where Treasury is pushing creates a third-axis tradeoff: serving the field, serving the directive, or stalling.

> **Engine change.** Multiple `boss` cast entries with their own cable queues. Cable arrivals are authored as passive events with `force_into_drafter` into the next cycle.

### 6.10 ★ Replace `dramatic_questions` with **per-cycle authored beats**

Instead of one dramatic question per state shape, author 8–12 cycle-specific beats per scenario, each with a `when` predicate and a unique flavor. The SoP picks the highest-priority unfired beat. Beats are *mostly* once-per-mission so the player doesn't see the same opening twice. This is the cheapest variety upgrade in the entire system.

---

## 7. Prioritized fix list

For the next session of work on HANDLER, in order of impact:

1. **Fix the prefetch race.** Either await the in-flight prefetch task at consumption time, or cancel-and-fall-through more aggressively. Cycles 4 and 5 burned ~20 s each on duplicate drafter calls.
2. **Per-cycle authored SoP beats** (§6.10). One YAML change. Eliminates the "same dramatic question 3 times" problem immediately. No engine work.
3. **Forbid renderer restating option prose** (§6.4 minimal). Add a single line to the renderer system prompt: *"DO NOT restate the option's prose. The reader has already read it. Add one new concrete observation per dispatch — a smell, a face, a phrase overheard, an absence."* This is a prompt-only change that addresses the "everything is confirmation" feeling.
4. **HOLD becomes "incoming reports"** (§6.3). Authoring + small engine change. Converts the most boring option into the most informative.
5. **Hidden opposition state with opposed checks** (§6.2). Adds genuine surprise. Larger lift — needs new effect class, opposition cast role with own qualities, and SoP prompt revision.
6. **Forced events become opportunity windows** (§6.1). Removes the "consenting at your pace" problem. Needs `window_cycles` plumbing and consequence events.
7. **Intel cards** (§6.6) — authoring-heavy but high payoff for replayability.

Items 8–10 are scope-expansion and can wait.

---

## 8. Spy-genre research findings (full report: `spy_game_mechanics_research.md`)

The research surveyed Phantom Doctrine, Alpha Protocol, Invisible Inc, Papers Please, Republia Times, 80 Days, Sunless Sea/Skies, Cultist Simulator, NITE Team 4, This War of Mine, plus tradecraft from Le Carré / Slow Horses / The Americans / Burn Notice. Key transferable mechanics, ranked by fit with HANDLER's current shape:

1. **Hidden quality axes per asset** (Phantom Doctrine sleepers + Alpha Protocol dossiers + Le Carré burnout). Per-agent `loyalty`, `fatigue`, `cover_integrity` that the player can only sense indirectly. **One change → three problems addressed at once: paranoia, asset-as-liability, dossier-gating.**
2. **Roadwarden's choice tiers** (`fluff` / `spice` / `floodgate` / `burnt_bridge`) added to the YAML DSL. Burnt-bridge picks fire a *closure dispatch* within 2 cycles confirming the door is shut. This is what actually trains players to trust that choices matter.
3. **Per-cycle pulse events** (Crusader Kings, RimWorld AI Storyteller). Faction acts, asset has a crisis, news breaks — conditioned on world state, not on player choice. Fixes the "world doesn't react" feeling.
4. **Re-contextualization events** (Spec Ops: The Line, Inscryption pattern). 5–10 rare authored events whose only job is to retroactively reframe prior cycles ("the asset you trusted has been Bureau the whole time"). Breaks linearity *backward*.
5. **Lateral costs** (XCOM panic-spread, Darkest Dungeon stress contagion). Every option's effect block lists at least one *non-obvious* side cost on a different axis.
6. **Visible threat clocks** (Invisible Inc). 2–3 deterministic timers in the HUD. Tension comes from watching them tick, not from rolls.
7. **Briefing-vs-reality divergence** (Pathologic 2 unreliable narrators). Once or twice per mission, the SoP's predicted shape and the dispatch's resolved shape *meaningfully differ* — engine-flagged, deliberate, not LLM drift.
8. **Named-victim language enforcement** in the renderer prompt: "if this op cost someone, name them and reference one personal detail."

Direct overlap with §6 above: research §1 ↔ §6.4 (reveal pool), research §2 ↔ §6.2 (opposed checks), research §5 ↔ §6.7 (per-agent trauma/morale + §6.6 intel cards), research §6 ↔ §6.3 (HOLD as incoming reports), research §8 ↔ §6.10 (per-cycle authored beats). My §6 list converged on the same shape independently — the research adds names, sources, and a stronger moral framing (assets-as-people, not assets-as-tools).

The single highest-leverage insight from the research that I missed in §6: **the burnt-bridge / closure-dispatch pattern**. HANDLER's choices feel weightless partly because the world doesn't *show* the player which doors closed. Author-required follow-up dispatches within 2 cycles of every burnt-bridge choice would change the trust model fundamentally.

## 9. Cross-genre research findings (full report: `alternative_genre_mechanics.md`)

The cross-genre survey covered Failbetter QBN, CK3, Reigns, 80 Days, Suzerain, Citizen Sleeper, Disco/Roadwarden/Pentiment, XCOM/Battle Brothers, Hades/Inscryption, Blades in the Dark / PbtA / Wildermyth, heist mechanics, and investigation games. Three highest-leverage *additions* to the engine:

1. **First-class `Clock` type** (Blades in the Dark, Citizen Sleeper). A `Clock` has `name`, `segments`, `tick`, `fires_event_when_full`. Replaces several ad-hoc threshold patterns (e.g., MUNTEANU's investigation, REDFIELD's fourteen-year tenure, the eastern train schedule) with one named primitive that writers can reason about. The single cleanest type addition possible.
2. **Per-agent `traits` and `injuries`** (CK3, Battle Brothers, Wildermyth). Same agent, same option, *different* delta + *different* prose constraints depending on traits. Multiplies authored content by trait-cardinality without writing more options. KESTREL with `vienna_scarred` trait responds differently to a diversion order than KESTREL without.
3. **`meta_state` surviving mission boundaries** (Hades, Inscryption). The fired ledger and quality snapshots seed the next scenario's initial state. *"You lost"* becomes *"you continue, scarred"*. The engine already has the persistence layer — it just needs to **not** call `make_initial_state` between campaigns.

**Patterns that are already in the engine but unnamed:**
- HANDLER's `qualities` dict IS Failbetter QBN. Naming the patterns (Carousel, Grandfather Clock, Midnight Staircase) would give authors known shapes to think in.
- HANDLER's `passive_event` with `only_when_no_progress_this_cycle` is exactly Sunless Sea's Terror mechanic. Just unnamed.
- HANDLER's stable-prefix + dynamic-suffix prompt structure is exactly inkle's streaming-vs-sandbox model.
- The fired-event ledger + per-agent fired set already supports Reigns' "actor-reappearance" callbacks. It's an authoring discipline, not an engine gap.

**The richest pattern *not* in the engine:** Disco Elysium's parity rule. *Failure must produce a scene of equivalent or greater narrative density than success.* Right now HANDLER has no failure prose authored at all — every option is "succeed cleanly." Adding a `failure_lean_on` field to options would let scenarios author the prose hooks for the failure case, and then a future opposed-check system (§6.2) would have somewhere to land its lossy outcomes.

**Alternative settings that would unlock new mechanics for free:**

| Setting | The one thing the spy frame can't easily do |
|---|---|
| Hospital triage in a war zone | "Who do I abandon?" — agency *over which lives end*, not which targets to act on |
| Doomsday cult leader | The Faith Spiral — a quality you must *not* let drop AND *not* let max |
| Starship captain on long-haul | The imposed Distance Clock — pure ticking time the player can't influence |
| Saint hearing confessions in a plague | The Penitent's Lie — engine writes delta but LLM speaks the petitioner's lie without revealing the underlying truth |
| Film-noir DA in a corrupt city | Cases as parallel clocks that advance whether or not you push them |
| Startup CEO in collapse | The Pivot — a player-callable forced event that resets multiple qualities atomically |
| Immortal vampire managing a court | The Long Memory — fired ledger persists across *centuries* of campaigns; literal generational state |

The Saint setting is the most engine-aligned: it forces the LLM to genuinely author content the engine cannot pre-compute (the petitioner's lie is unique per playthrough), which is the bar I argued for in §5. The vampire-court setting is the most ambitious reuse of the existing engine's persistence layer with zero new types — *just don't reset state between campaigns*.

---

## Appendix A — Trace span tree (cycle 1)

```
POST /mission/start                                    19:26:41.492
└─ run_cycle (cycle 1)                                 19:26:41.493 — 45.13s
   ├─ agent run (SoP)                                  19:26:41.495 — 10.47s
   │  └─ chat gemma-4-e4b                              ─ 1988 in / 617 out
   ├─ drafter_call                                     19:26:51.872 — 4.71s
   │  └─ agent run (drafter)
   │     └─ chat gemma-4-e4b                           ─ 3020 in (2188 cached) / 751 out
   ├─ POST /mission/.../authorize                      19:27:19.238 — player picked HOLD
   ├─ agent run (renderer)                             19:27:19.242 — 7.38s
   │  └─ chat gemma-4-e4b
   └─ prefetch_next_cycle (background)                 19:27:26.527 — 29.60s
      ├─ agent run (SoP for cycle 2)                   19:27:26.528 — 4.17s
      └─ drafter_call (cycle 2 prefetched)             19:27:30.697 — 25.43s
```

## Appendix B — Cycle 5 wasted prefetch

```
[cycle 4 ends]
prefetch_next_cycle (cycle 5)              19:29:57.708 — 32.70s ───┐
├─ agent run (SoP cycle 5)                  ── 4-9s                  │
└─ drafter_call (cycle 5)                   ── 22.79s ──┐            │
                                                        │            │
[player clicks Continue at 19:30:00 — prefetch not yet  │            │
 done; live cycle starts and runs FRESH SoP + drafter] │            │
                                                        │            │
run_cycle (cycle 5)                         19:30:00.261 — 65.34s    │
├─ agent run (SoP cycle 5, FRESH)           19:30:00.263 — 15.56s ◀──┘ duplicate
└─ drafter_call (cycle 5, FRESH)            19:30:15.728 — 21.64s ◀── duplicate
   (only this one's options are used)
```

The first drafter_call (`80eb7e61753e8f78`) generated 3771 tokens of LLM output that was thrown away. The second (`4a179f9acc82754e`) ran on top of it, producing nearly identical options in a different order.
