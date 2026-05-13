# Alternative Genre Mechanics for the HANDLER Engine

> Engine baseline being evaluated against: deterministic state machine; LLM writes prose only; 3-4 options per cycle; ~6 cycles count down; agents are stateless one-shot; state = qualities (numbers/postures/flags), agent rosters, fired-event ledger; forced events can interrupt and demand specific options.

For each genre I extract concrete mechanics, then add a single-line **TRANSLATION** showing how to plug it into the engine without breaking the determinism contract. At the end, a setting catalogue: each suggestion names the *one* mechanic the spy frame can't easily produce.

---

## 1. Resource-management storylets (Failbetter / Weather Factory)

Three games share a lineage: **Fallen London**, **Sunless Sea / Skies**, **Cultist Simulator**. The structural innovation Failbetter named and shipped is **Quality-Based Narrative (QBN)**.

- **Storylets gated by qualities.** A storylet is a paragraph + 2-4 options. Each option may be (a) hard-locked behind a quality threshold, (b) a probabilistic *test* against a quality, or (c) a free pick that sets/raises another quality. Qualities are everything: inventory, skill, relationship, story-progress, posture (e.g. "Time of Day", "Suspicion in the Court").
- **Sixty named patterns, not one tree.** Failbetter catalogued 60+ recurring narrative shapes within their engine. Useful ones: **Carousel** (one quality cycles through fixed states; available storylets rotate), **Grandfather Clock** (build a quality up; on threshold it resets and advances a chain), **Midnight Staircase** (one storylet, many branches, all feeding the same progress quality).
- **Cultist Simulator's minimal layer.** Strips QBN to *card combinations*: drag a card onto a verb-slot, get a timed outcome, get new cards. The cards ARE the qualities, made tangible. Story is "scintillas of lore," never expository.
- **Sunless Sea's terror as a quality with thresholds.** Terror rises from environment; at 100 you trigger a likely-fatal forced event. It is *the* model of "passive pressure event reads a quality."

**TRANSLATION**
- HANDLER's `qualities` dict already IS QBN; what's missing is naming the *patterns*. Steal the pattern names — declare authoring intents like `pattern: carousel(quality=cover_identity)` so writers think in known shapes instead of ad-hoc graphs.
- Add a **Cultist-style "verb slot"** layer above options: each cycle the player drags a roster-agent onto a verb-slot ("Burn", "Recruit", "Sleep") and the engine selects from option templates that match (slot, agent traits). This re-uses your roster as a literal hand of cards.
- Sunless Sea's Terror = your `passive_event` with `only_when_quality_above`; you already have it. Name it.

Sources:
- [Beyond Branching: QBN, Salience, Waypoint — Emily Short](https://emshort.blog/2016/04/12/beyond-branching-quality-based-and-salience-based-narrative-structures/)
- [Failbetter: Fallen London Writer Guidelines I](https://www.failbettergames.com/news/fallen-london-writer-guidelines-part-i)
- [Why Cultist Simulator devs built on a house of cards — Game Developer](https://www.gamedeveloper.com/design/why-the-i-cultist-simulator-i-devs-built-their-lovecraftian-game-on-a-house-of-cards)
- [Sunless Skies redesigned Terror — community thread](https://community.failbettergames.com/t/feedback-focus-new-terror-mechanics/18825)

---

## 2. Crusader Kings 3 — event design

CK3's story-event is a pop-up with 2-4 options. The lessons:

- **Personality traits gate emotional outcomes.** Same event, different stress impact for `Vengeful` vs `Patient`. A `Sadistic` character *reduces* stress by picking the cruel option — explicit reward for in-character play. (Dev Diary 58.)
- **Weighted rather than scripted outcomes.** Each option's outcome branches on a list of weighted entries; weights are modified by character/world conditions. Final outcome is a roll, not a deterministic resolve.
- **Stress as a meta-pressure quality.** Acting against trait → stress; stress maxes → mental break → forced-event picks a Coping Mechanism *trait* → new decision unlocked. Stress is the engine's main mechanism for translating "you played against type" into mechanical narrative consequence.
- **Failure mode.** CK3 events get critiqued as "modifier soup": a million +0.1's that the player can't perceive. Lesson: keep modifiers visible AND few, or modifiers stop being legible choices.

**TRANSLATION**
- Add an `agent.traits` field; option-effects already gate on quality, just gate on traits too. The same option text resolves to *different* delta when the actor's traits clash.
- Add a `stress` quality per-agent-or-protagonist with a threshold-triggered forced event. On break, the forced event's options *write a new trait* (your fired-event ledger already supports this if you add a `set_trait` effect).
- Keep CK3's lesson on legibility: cap visible modifiers per option to ~2.

Sources:
- [CK3 Wiki — Events](https://ck3.paradoxwikis.com/index.php?title=Events)
- [Dev Diary #58 — Stretching the traits](https://forum.paradoxplaza.com/forum/developer-diary/ck3-dev-diary-58-stre-ss-tching-the-traits.1472092/)
- [CK3 Modding — Conditional logic & random events](https://www.oreateai.com/blog/ck3-mod-creation-guide-detailed-explanation-of-conditional-logic-and-random-event-systems/83eb67d75ce28a7bfa7382003946046f)

---

## 3. Reigns — making 2-option not feel binary

Reigns' verbs: swipe left = no, swipe right = yes. Four meters: Church, People, Army, Treasury. Any meter at 0 OR 100 = death. What makes it not boring:

- **Hidden inversion.** The "yes that pleases X" is often the "yes that secretly hurts Y." Players never know all four deltas before swiping; they read intent from the *speaker* (the card character) and infer.
- **Compound timing.** Today's "yes" funds a festival; three rulers later your treasury craters and a new card fires referencing the festival. Consequences are deferred and chained through *actor reappearance*, not "+1 quality" memory.
- **Constraints, not balance.** Designer Alliott (GDC): the rules were a creative *frame*, not a brake on writing. "What if Death visits?" is allowed even if it breaks the meter logic.

**TRANSLATION**
- Even with 3-4 options you can hide deltas: show *predicted* deltas only when an option's `intent` quality (see CK3 lean_on) is high enough on the actor. Adds an information game on top of the option game.
- **Recurring NPC cards.** An agent that fires this cycle's event "remembers" by setting a flag the writer references next cycle. You already have a fired-event ledger and per-agent fired set — Reigns calls this its main joke.
- The "any meter to 0 OR 100 = death" rule is your `terminal` event with `only_when_quality_outside_range`. Worth adding two-sided ranges as first-class.

Sources:
- [GDC — The Casual (but Regal) Swipe](https://www.gdcvault.com/play/1024278/The-Casual-(but-Regal)-Swipe)
- [Reigns Walkthrough — Resource Balance](https://www.reignsgames.com/blog/reigns-walkthrough-swipe-king)
- [Reigns Review — TheSixthAxis](https://www.thesixthaxis.com/2016/08/25/reigns-review/)

---

## 4. 80 Days (inkle)

Phileas Fogg's valet. ~150 cities. Resources: time, money, luggage, Fogg's health/mood. Why static-text feels kinetic:

- **The writer's job is to tempt you into bad decisions.** Lead writer Meg Jayanth: "near-misses, catastrophes, and daring escapes are what players remember." Options that sound exciting are *deliberately* the worse mechanical choice.
- **Streaming, not sandbox.** Engine streams story content on demand; the script never loads as a whole. Effect: writer can author a city as a tight self-contained unit and trust the global state to bridge.
- **Resource axes were the cheap part.** Time/money/luggage emerged because they're obvious for *travel*. Lesson: pick a setting where resources name themselves.

**TRANSLATION**
- Author option `flavor` (intent/tone/risk) explicitly, then have the renderer emphasize the dramatic option. The drafter prompt currently has `lean_on` — push it further: an explicit `temptation` field meaning "make this read as the cool choice even if delta is bad."
- Streaming = your stable-prefix + dynamic-suffix prompt structure. Already aligned.
- Quality of the resource axis matters more than quantity. Spy frame's `cover` / `suspicion` / `cycles_remaining` is exactly 80-Days-grade triangulation.

Sources:
- [Postmortem: Inkle's 80 Days — Game Developer](https://www.gamedeveloper.com/business/postmortem-inkle-s-i-80-days-i-)
- [Narrative insights from 80 Days' writing lead](https://www.gamedeveloper.com/design/narrative-and-design-insights-from-i-80-days-i-writing-lead)
- [80 Days — Emily Short review](https://emshort.blog/2014/08/05/80-days-meg-jayanth-inkle/)

---

## 5. Suzerain — chained decisions across acts

Single-player political thriller. President Anton Rayne navigates a four-year term. Key design moves:

- **Hidden vs visible stats.** Visible: Economy, Budget. Hidden: Alignment, Unrest, Opinion-of-X. Players see consequences; they don't see the full state. Re-readability of choice text changes as hidden state shifts.
- **Prologue choices have late-act payoff.** A flavor question in the prologue (your father's profession) gates a cabinet appointment in act 3. Choices feel weightier *because the writer doesn't tell you which ones mattered*.
- **Decision is the literal noun.** The wiki page calls them "Decisions" with capital D. Each is a 2-4 option modal that fires from chains, not always from time.

**TRANSLATION**
- You already split visible (`qualities`) from hidden (`fired` ledger). Be more deliberate: surface some qualities to player, hide others, and make the hidden ones drive option *availability* (not just outcome). Player learns hidden state by observing the option set itself.
- A cycle-1 option whose only effect is `set_flag(prologue_choice=X)` is cheap; a cycle-5 forced event whose `only_when` reads that flag is also cheap. The chain in between is FREE — pure storage. Use it.
- Suzerain's Acts ≈ your cycles. 6 cycles is enough for two-act callback if act 1 = cycles 1-3, act 2 = cycles 4-6.

Sources:
- [Suzerain — Articy showcase (engine details)](https://www.articy.com/en/showcase/suzerain/)
- [How early decisions affect Anton Rayne — ScreenRant](https://screenrant.com/suzerain-anton-rayne-prologue-decisions-choices-political-party/)
- [The Reign of Play: Suzerain and Practicing Politics — Cosmonaut](https://cosmonautmag.com/2024/10/the-reign-of-play-suzerain-and-practicing-politics/)

---

## 6. Citizen Sleeper — dice + clocks

- **Dice as a hand of options.** Each cycle you roll 1-5 dice (count = your condition quality). Each die has a *value*. Each action slot consumes one die; the value modifies success odds. You allocate the whole hand simultaneously.
- **Clocks tick whether you act or not.** Each NPC arc has a clock filling at its own rate. Ignore an NPC and their arc resolves *without you* — usually badly. Direct lift from Blades in the Dark.
- **Fail-forward via "success at a cost."** A low-die action that "succeeds" pushes its clock but also ticks an antagonist clock. No outcome is null.
- **Condition is the meta-resource.** The Sleeper's body decays; fewer dice each cycle → forced specialization → forced narrative shape (this run is about *one* arc).

**TRANSLATION**
- Replace "pick 1 of 4 options" with "spend 2-3 dice across 4 option slots" — same engine, but each option carries a `min_die` and a `quality_delta_scaled_by_die_value`. Multi-axis decision in one cycle without losing determinism.
- **Clocks that advance regardless** are passive events with `only_when_no_progress_this_cycle` inverted: `only_when_no_action_targeted_this_clock`. Add: `clock` as a first-class quality whose increment fires events at thresholds.
- Decreasing-dice meta = your `cycles_remaining` but per-resource. A `composure` quality where current_max decays creates the same "this run is about specialization" pressure.

Sources:
- [How Citizen Sleeper was inspired by tabletop RPGs and gig work — Game Developer](https://www.gamedeveloper.com/business/how-citizen-sleeper-was-inspired-by-tabletop-rpgs-and-gig-work)
- [Dice Roll Mechanic — UNCG video game blog](https://uncg237publications.wordpress.com/2024/12/20/the-dice-roll-mechanic-in-citizen-sleeper-an-in-depth-explanation/)
- [Case Study: Citizen Sleeper — How To Make An RPG](https://howtomakeanrpg.com/r/a/case-study-citizen-sleeper.html)

---

## 7. Disco Elysium / Roadwarden / Pentiment — failure as content

- **Disco's red checks are one-shot.** Pass: branch A. Fail: branch B, often more memorable. Iconic moments ("Raphaël Ambrosius Costeau", the hat speech, "I want to have intercourse with you") are FAIL outcomes.
- **The trick is parity.** A failure must produce a scene of equivalent or *greater* narrative density than success. If failure feels like "nothing happens," the player save-scums.
- **Roadwarden inverts.** Skill checks rarely lock content; they color *how* the same content is reached. Inclusivity over gating. Useful if you don't want to author parallel branches.
- **Pentiment uses backgrounds as colored lenses.** The same scene plays out, but the protagonist's chosen disciplines unlock different reactive lines. The branching is at the *prose* layer, not the *plot* layer.

**TRANSLATION**
- Disco's parity rule is the most important lesson for HANDLER: every option needs an *authored* failure delta and *authored* failure prose hook. Right now your engine writes the delta; the LLM has to invent the failure prose. Give it a `failure_lean_on` field.
- Roadwarden mode = let the engine resolve the same option even when a quality is too low, but the *prose constraints* on the drafter change ("you fumble the cover line, the room knows"). Single delta path, two prose modes.
- Pentiment mode = your forced-event `must_communicate` already supports this. Author the same Event with an `if_trait` prose-constraint variant.

Sources:
- [Disco Elysium and the Meaning of Failure](https://gameplayreflections.wordpress.com/disco-elysium-and-the-meaning-of-failure/)
- [Disco Elysium RPG System Analysis — Gabriel Chauri](https://www.gabrielchauri.com/disco-elysium-rpg-system-analysis/)
- [Deep Dive: Roadwarden impact and narrative variance](https://www.gamedeveloper.com/design/deep-dive-roadwarden)

---

## 8. XCOM / Phoenix Point / Battle Brothers — strategic-layer roster

Even ignoring tactical combat, the strategic layer is the part that maps onto HANDLER.

- **Roster of named units with traits.** Every brother has 0-2 randomly-rolled traits (Fat, Brave, Drunkard) that mod stats and gate dialogue/event reactions. Death is permanent; identity carries weight because the unit *was* the protagonist of past events.
- **Permanent vs temporary injuries.** Two-tier: this-cycle injury (heals) vs scar (forever). Injury location matters — wounded leg slows, wounded arm hurts hits. A forever-injury is a forever-quality on that agent.
- **Mission tradeoffs over a campaign clock.** Two contracts available; you can take one. The other still resolves (someone else takes it; consequences ripple). Time pressure forces opportunity-cost choices, not "do everything."
- **Equipment as inventory state.** Each agent has a slot loadout the engine can read into events ("the man with the heavy crossbow goes first").

**TRANSLATION**
- You already have a roster (`agents`). Add per-agent `traits: list[Trait]` and per-agent `injuries: list[Injury]` (with `temporary: bool`). Drafter prompt already lists agents; just thread these.
- "Mission you didn't take resolves anyway" = forced events that fire if a quality the player COULD have advanced went unadvanced. You have `only_when_no_progress_this_cycle` — generalize to `only_when_no_progress_on(quality)`.
- Permadeath of an agent removes them from the drafter's roster entirely; the LLM stops referencing them. Cheap, devastating consequence.

Sources:
- [Battle Brothers Wiki — Game Mechanics](https://battlebrothers.fandom.com/wiki/Game_Mechanics)
- [Dev Blog #79: Injury Mechanics](https://battlebrothersgame.com/dev-blog-79-progress-update-injury-mechanics/)
- [Battle Brothers Roster Building Guide](https://mygamingtutorials.com/2025/05/19/battle-brothers-tactical-series-part-2-building-a-roster-for-success/)

---

## 9. Roguelikes — Slay the Spire, Hades, Inscryption

Reframing failure as content:

- **Hades' codex.** Every NPC, enemy, room has a codex entry that fills in across runs. Death isn't reset — Achilles updates the book. The *between-runs cutscenes* are the actual game; combat is the delivery vehicle.
- **Hades' room-choice screen.** Just before each chamber: a small icon menu showing which boon/reward is in each next room. 2-4 options, no combat happening. This is *literally HANDLER's authorize-one-option screen*, repeating ~30 times per run.
- **Slay the Spire's map.** Branching paths with previews of node type (combat, elite, shop, mystery). Strategic choice between routes BEFORE engaging.
- **Inscryption's meta-twist.** The "frame story" is itself a state the engine tracks; what felt like a roguelike becomes a story-progression layer you didn't know was there.
- **Loss = content rule.** Players accept a 6-cycle game ending in loss IF the loss generates new narrative state for the next playthrough.

**TRANSLATION**
- Add a `meta_state` that survives mission_end and seeds the next mission's initial qualities. "Last time you betrayed Iva; this mission opens with Iva's brother running interpol's Eastern desk." Pure determinism, just persisting state.
- Hades' room-choice IS your engine; you can run multiple scenarios per "campaign" with `meta_state` carrying the dossier between them. Each scenario is one Hades run.
- Inscryption's meta-twist suggests: cycle 6 isn't always the end. A meta-quality `the_real_game` can flip on completion and unlock a longer second arc with the same engine. Re-uses authoring effort dramatically.

Sources:
- [Roguelites deconstructed — LinkedIn analysis](https://www.linkedin.com/pulse/roguelites-deconstructed-svyatoslav-torick-oofgf)
- [Hades Wiki — Codex](https://hades.fandom.com/wiki/Codex)
- [Slay the Spire 2 dev notes on Hades, Nightreign — GamesRadar](https://www.gamesradar.com/games/roguelike/slay-the-spire-2-dev-flattered-everyone-places-its-predecessor-among-the-best-roguelikes-ever-shouts-out-hades-elden-ring-nightreign-and-more-for-making-an-impact-on-them/)

---

## 10. Tabletop story games — Blades, PbtA, Wildermyth

The richest source of patterns for a determinist engine that delegates prose.

- **Blades in the Dark — Position × Effect × Consequence.** Before each roll: GM declares Position (Controlled / Risky / Desperate) and Effect (Limited / Standard / Great). After the roll: success-with-cost ticks a Consequence Clock (1-3 segments depending on severity). The grid is small (3×3), the outcome space is rich.
- **Blades' Progress Clocks.** A clock is a circle with N segments. Anything multi-step uses one: an investigation, a chase, an antagonist's plan. Clocks fill from PC actions AND from world ticks. Universally borrowed (Citizen Sleeper, ICON, Brindlewood Bay, half of indie TTRPG).
- **PbtA 2d6 — 10+ / 7-9 / 6-.** 7-9 is "yes, but" and is the most common roll. 6- isn't "nothing happens" — the GM "makes a move" (escalates, complicates, introduces NPC). No null outcomes.
- **PbtA Moves named by trigger.** "When you do X, roll +Y." Moves are addressable by player intent, not by skill list. Forces design where every common verb has a *named* outcome shape.
- **Wildermyth — story beats from procedural events.** Combat injury triggers a hand-authored beat where the character chooses how to react. The choice writes a permanent character trait ("now has a wooden arm; gains +X"). Mechanical scar = narrative scar = same datum.

**TRANSLATION**
- Blades' Position × Effect = two pre-roll qualities the engine sets, the writer reads. Implement as two enums per option: `position: enum`, `effect: enum`. Outcome delta scales by both. The drafter prompt gets this pair as constraints — "this option resolves at Risky/Standard."
- Progress Clocks = first-class `Clock` type. Author it once: `name`, `segments`, `fires_event_when_full`. Effects can `tick_clock(name, n)`. Cleanest single feature you could add.
- PbtA 7-9 = mandatory partial-success for at least one option per cycle. Authored as: option resolves to one of *three* delta sets selected by a single hidden roll, but one set is always "yes-but." Write all three; engine picks deterministically by quality threshold; LLM writes the prose for the picked one.
- Wildermyth's "injury → permanent trait choice" is your forced-event with options that each `set_trait` differently. Add this as a recipe: "scar event."

Sources:
- [Blades in the Dark — Effect](https://bladesinthedark.com/effect)
- [Blades — Progress Clocks](https://bladesinthedark.com/progress-clocks)
- [Apocalypse World — Wikipedia](https://en.wikipedia.org/wiki/Apocalypse_World)
- [PbtA — Lumpley Games](https://lumpley.games/2019/12/30/powered-by-the-apocalypse-part-1/)
- [Wildermyth — AIAS Game Maker's Notebook](https://gamemakersnotebook.libsyn.com/balancing-procedural-and-intimate-storytelling-in-wildermyth)

---

## 11. Heist-crew genre (Payday, Monaco, Blades-as-heist)

Spy fiction already does *covert ops*; what does *heist* add that spy doesn't?

- **Pre-mission planning as authored choice.** Before the heist: pick crew composition, pick approach (loud / quiet / social engineering), pick equipment loadout. Each pick changes the option *menu* of every subsequent cycle.
- **Specialist roles.** Each crew member has ONE thing they're the only one who can do (Hacker, Climber, Face). The roster isn't interchangeable — each name is a verb.
- **The Score / The Twist.** The heist always has a planned beat order; one beat ALWAYS goes wrong. Player's job is to react, not to plan-correct. (Blades: "the heist starts in flashback; the present is reaction to escalation.")
- **Take vs. trace.** Trade-off between *grabbing more loot* and *leaving evidence*. Spy frame has only "succeed cleanly"; heist frame has "succeed messily on purpose."

**TRANSLATION**
- A cycle-0 "planning" event with 3 sequential picks (crew / approach / loadout) sets three flags that gate the option pool of cycles 1-N. You already gate options on flags — just author with this in mind.
- Add a `specialist_for(verb)` field on each agent. Options that require that verb either need that agent in roster or fall back to a worse delta.
- A scripted "twist" forced-event in cycle 3 is structurally identical to your existing forced events; just always present, always interrupting plan.

Sources:
- [PAYDAY: The Heist — TV Tropes](https://tvtropes.org/pmwiki/pmwiki.php/VideoGame/PaydayTheHeist)
- [Blades in the Dark RPG — heist structure](https://bladesinthedark.com/effect)
- [Monaco 2 announcement — Thinglabs](https://thinglabs.io/monaco-2)

---

## 12. Investigation — Obra Dinn, Her Story, Disco

- **Obra Dinn's verification triple.** Player makes guesses; engine reveals correctness only when 3 are right *as a group*. Stops single-guess save-scumming, demands lateral confidence.
- **Lateral information.** Clue in scene 4 is needed to interpret scene 7 — but scene 7 doesn't tell you. Player must re-read.
- **Her Story's search.** Player types keywords into a corrupted database; matches surface video clips. The verb IS query; the world reveals itself through the player's vocabulary.
- **Truth as a hidden quality.** The world has a fixed answer. The player's beliefs are the modeled state. Score = belief alignment with truth.

**TRANSLATION**
- A `truth_state` immutable structure (set at scenario start) plus a `belief_state` that the player edits via authorized options. Forced event resolves at cycle-end by *comparing* the two.
- "Verification triple" → at cycle close, only commit beliefs that are internally consistent (no contradictions in the fired ledger). Determinism preserved; player gets no per-belief feedback.
- Her Story's search ports poorly to a 3-4 option engine — but a *cycle dedicated to choosing which dossier-line to interrogate next* would feel structurally similar.

Sources:
- [Return of the Obra Dinn — Wikipedia](https://en.wikipedia.org/wiki/Return_of_the_Obra_Dinn)
- [Obra Dinn & Lateral Information — Atomic Bob-Omb](https://atomicbobomb.home.blog/2020/03/21/return-of-the-obra-dinn-lateral-information/)
- [Obra Dinn: A Lesson on Hands-Off Detective Design](https://viciousundertow.wordpress.com/2018/11/08/return-of-the-obra-dinn-a-lesson-on-detective-games-and-hands-off-design/)

---

## 13. Alternative settings for this exact engine

Each setting names *one mechanic* the spy frame does not produce naturally. The engine stays as-is.

### A. Hospital triage in a war zone

- **Unique mechanic: the Triage Tag.** Every cycle, multiple patients arrive; you have N treatment slots and must *not* treat someone. The decision isn't "what to do" — it's "who to abandon." Spy fiction's options are about agency over targets; triage's options are about agency over which lives end.
- **Engine fit.** Patients = ephemeral agents with a `prognosis_clock` (Blades-style); ignored ones tick to death. The forced "MASCAL event" interrupts cycles. Qualities: `morale`, `supplies`, `chain_of_command_trust`.

### B. Doomsday cult leader (mid-collapse)

- **Unique mechanic: the Faith Spiral.** A quality (`fervor`) that you must *not* let drop AND must *not* let max. Max → mass-casualty event; min → schism. Spy frame has pressure to perform; cult frame has pressure to *also* believe what you sell.
- **Engine fit.** Per-follower `belief` quality; events fire on per-agent thresholds. "Inner circle" agents are roster; "the flock" is a single aggregated quality. Forced event: A federal raid, a defector, a vision.

### C. Starship captain on a long-haul voyage

- **Unique mechanic: the Distance Clock.** A monotonically-decreasing `distance_to_destination` that the player CANNOT influence — only react around. Spy fiction's `cycles_remaining` is similar but spy missions feel chosen; voyages feel imposed by physics.
- **Engine fit.** A pure ticking clock plus an authored "milestones at distance X" table. Crew morale, fuel, hull as qualities. Crew as roster with specialist roles (Engineer, Doctor, XO) — straight Battle Brothers fit.

### D. Saint hearing confessions during a plague

- **Unique mechanic: the Penitent's Lie.** Each confession is a forced event with 3-4 absolution options, but the petitioner has a hidden `truthfulness` quality. Pick the wrong absolution and a forced event two cycles later reveals you were complicit. Spy frame gives the player privileged info; this frame *withholds* it.
- **Engine fit.** Petitioners as ephemeral agents with hidden traits. Your engine writes the delta; the LLM speaks the petitioner's lie without revealing the underlying truth (perfect deterministic-engine + creative-prose split). Qualities: `holy_authority`, `worldly_compromise`, `plague_dread`.

### E. Film-noir DA in a corrupt city

- **Unique mechanic: the Case File.** Each cycle the player picks WHICH case to push (3-4 cases on the desk). Cases not pushed advance under their own momentum (Citizen Sleeper clock). Spy frame's options are tactical; DA's options are *which thread of justice gets bandwidth this week*.
- **Engine fit.** Cases as named clocks with passive ticks. "Take the bribe" is a recurring meta-option that adds resources at the cost of a `compromise` quality threshold. Roster: prosecutors, cops, witnesses, mob.

### F. Startup CEO in collapse

- **Unique mechanic: the Pivot.** A forced event you can call yourself once, that *replaces* multiple qualities atomically (burn rate, headcount, product flag). Spy frame doesn't reset; startups do. The pivot card gives the player an "authored gambit" inside an otherwise downward arc.
- **Engine fit.** Add a `player_callable_event` flag — events the player can voluntarily invoke (cost: a quality, a turn, a relationship). Most events still fire from engine; this is the exception. Qualities: `runway`, `team_morale`, `board_confidence`, `narrative_in_press`.

### G. Immortal vampire managing a court

- **Unique mechanic: the Long Memory.** State persists across *centuries* (multiple campaign instances). A grudge from cycle 3 of season 1 fires as a forced event in cycle 1 of season 4. Spy frame is single-mission; vampire frame is generational ledger.
- **Engine fit.** This is the Hades meta-state suggestion made literal: the fired-event ledger never wipes. Each "campaign" is one century. Roster: kindred (immortal, persist), ghouls (mortal, churn). Qualities: `masquerade_strain`, `prince_favor`, `hunger`. The engine you have already supports this — it's just a matter of NEVER calling `make_initial_state` after the first campaign.

---

## Summary: highest-leverage additions

If only three changes were made to HANDLER from this audit, in priority order:

1. **First-class Clocks (Blades / Citizen Sleeper).** A `Clock` type with `segments`, `tick`, `fires_event_when_full`. Replaces several ad-hoc threshold patterns and gives writers a known shape.
2. **Per-agent `traits` and `injuries` (CK3 / Battle Brothers / Wildermyth).** Same agent, same option, different delta and different prose constraints. Cheap; multiplies authored content by trait-cardinality.
3. **`meta_state` that survives mission boundaries (Hades / Inscryption).** The fired ledger and quality snapshots seed the next scenario's initial state. Turns "you lost" into "you continue, scarred."

The settings list (§13) shows the engine as written can already host most of them — the work is in **picking resource axes that the chosen setting names for you** (per 80 Days' lesson) and **choosing which existing engine feature to make load-bearing** for that setting.
