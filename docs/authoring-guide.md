# Scenario Authoring Guide

How to build new scenarios on the Inbox engine. Project-specific. For prior art, the Director/Performer split, and the AI-native primitive argument, see `docs/LLM-game-research.md`. For engine invariants and authoring vocabulary, see `CLAUDE.md`. For the design rationale of the shipped scenario, see `docs/frankenstein-adaptation.md`.

## What this game actually is

Inbox is a **two-sided free-form** narrative game: the player writes; an LLM-driven NPC reads what they wrote and writes back. Both sides in free prose is the property no narrative game before LLMs could afford. The leverage lives there. A storylet engine with hand-authored prose nodes is a worse Failbetter; the version that earns its keep uses the LLM for what only it can do — read texture, render directives, canonize player invention, judge against constraints — and leaves the spine deterministic.

The discipline: don't let the LLM do what it's bad at (counting, bookkeeping, hard rule enforcement, knowing when to stop), and don't waste it on what pre-LLM tech could already do.

## What scales and what doesn't

**Scales:**
- **Scenarios as anthology entries.** Each a self-contained 1–3 hour piece with its own dossiers, qualities, terminals. Different literary skeleton per scenario keeps the depth-read mechanic fresh without engine work. (Inkle shipped Sorcery! 1–4 this way; Citizen Sleeper ships DLC episodes.)
- **Storylets per arc.** 13–20 storylets across 1–2 deep NPCs ships better than 30 across 5 shallow ones. The dossier carries the voice; rendering is at runtime.
- **Multi-axis classifier reads.** Each non-trivial reply can read 2–3 texture axes at the same per-call cost as one. Reach for an extra axis before authoring a parallel beat.

**Doesn't scale:**
- **Quality count.** Past ~5–8 qualities per scenario, gating logic gets hard to keep coherent. Disco Elysium drove its endings on five.
- **Terminal categories.** Three or four attractors per scenario is the ceiling. Mutex matrix is N² in count.
- **Cross-scenario state.** Don't try to carry state between anthology entries. Authoring overhead is brutal; players don't notice.
- **NPC cast breadth.** 1–2 deep NPCs per scenario. 3+ at depth fights LLM context budgets and authoring time.

## Authoring loop per scenario

Order matters. The dossier is the leverage point; everything else is downstream of voice quality.

1. **Pick the literary skeleton.** A novel or play whose engine is *relational milestones gating one another* — Frankenstein, The Crucible, Heart of Darkness, Persuasion, The Trial. Plot-driven mysteries (Christie) fight the spine; relational drama fits naturally.
2. **Pick the surface fiction.** What corporate/institutional/domestic frame translates the depth. The Continuity Program (HR onboarding) is one example; embassy correspondence, therapy intake, customer-support escalation, monastic letters would all work the same way.
3. **Write the central NPC dossier.** 1500–3000 tokens. Voice register, will-say, will-not-say, three example quotes for tone, three forbidden topics with prepared deflections. This is ~60% of the authoring quality.
4. **Choose qualities.** 3–5 total. One cumulative (relational temperature). One "now-axis" surfaced via `RecentClassifierValue` (recent register). One or two `commitment.*` entries for fact-surfacing. Only add more when authoring a beat surfaces a felt gap.
5. **Choose terminal categories.** 3–4 attractors. Pick them by reading the literary skeleton — what readings does it support? Frankenstein supports integration / mutual ruin / exile / vengeance; Crucible supports confession / refusal / martyrdom.
6. **Author the spine beats.** 1–3 anchored beats (welcome, key reveals where exact phrasing matters). One multi-turn loop. 5–10 midpoint plot beats including 2+ "novel-not-taken" forks — recovery beats, paths the source didn't take. See `CLAUDE.md` "Authoring vocabulary" for the schema.
7. **Author terminals as directives, not prose.** Each terminal: `ends_session: true`, `validate_facts: true`, `intent` describing the genre + must-include facts + length window + voice constraints. The epilogue agent renders against the full ledger at runtime.
8. **Playtest 5+ runs cold.** Voice issues surface in the dossier, not the storylets. Iterate the dossier; storylets rarely need touching.

Realistic budget per scenario at this discipline: 7–14 days solo.

## Endings model: convergent-with-coda

Authored *categories*, generative *prose*.

- 3–4 terminal beats, each with `ends_session: true`. Categories mutex-gated by `Not_(BeatFired …)`.
- Each terminal's `intent` is a directive (genre + must-include facts + length window), not authored prose.
- The epilogue agent (see `CLAUDE.md` invariants) renders against the full run digest — commitments, classifier history, fired-beat order, recent player replies. Two players who hit the same terminal category get genuinely different prose because their digests are different.

This solves the "true ending" trap (every ending is true to *that* run) and lets few attractors carry many felt instances. Disco Elysium's structure: ~5 variables, 3 endings, felt diversity from a much richer downstream readout. Same shape applies here.

## The affordance checklist

For any new beat, ask: *am I using the LLM for what only it can do?* If the beat would work just as well in pre-LLM Twine, you're not collecting the dividend.

Affordances that justify being on this engine:

1. **Storylet-as-directive.** One authored intent, infinite renders conditioned on run state. (Already idiom; adopt universally.)
2. **Multi-axis per-storylet classifiers.** Reading texture cheaply.
3. **Generative epilogue.** Terminal prose conditioned on the full ledger.
4. **Player-quoting via commitments.** NPC references things the player said earlier. Felt-seenness at near-zero cost.
5. **Mid-scene reframing on midpoint fires.** When a pivotal beat fires, subsequent renders re-color earlier exchanges with the new lens.
6. **Voice-as-dossier, not voice-as-templates.** A new NPC costs one dossier, not 200 hand-authored lines.
7. **Texture-as-mechanic.** Register, hedge, length, formality as readable qualities.

Things that look LLM-native but aren't:

- **Voice fingerprinting / mimicry-as-mechanic.** Players don't reliably recognize their own statistical voice in another character's mouth. They notice *content* tells (specific phrases they used, names they invented), not style tells. Build "the AI is becoming you" effects on **content mimicry** (player-claim canonization, the NPC quoting facts the player invented) rather than on cadence/length/formality fingerprints. The Frankenstein-creature horror is *standing* (the creature has thoughts and acts), not *imitation*.

## What to defer until trigger

Pattern: don't pre-build features without a felt trigger from playtest. Each deferred item should be paired with a concrete trigger that sends you to build it.

Currently deferred:

- **Player-claim extraction.** A second structured-output agent on each player reply, extracting concrete specifics (named people, places, claims) with a `literal_text_quote` anchor and `salience` score; promotes to a parallel `state.player_canon` store, surfaced via the commitments-block path, gated by a new `PlayerCanonExists` predicate. *Trigger:* the next scenario authors a beat where the NPC should plausibly reference a specific the player invented and you feel the gap in playtest. Likely the next thing to land.
- **Calibration scene.** Multi-turn PlotBeat at boot whose intent elicits name + role + one personal detail in HR-bright register; `completes_when` over canon predicates; replaces static `player_email` config with a runtime-constructed address from canonized name + scenario-level `email_domain`. *Trigger:* land player-claim extraction first; this is its first canonical use.
- **`completes_when_judge`** (small PlotBeat extension for LLM-judged exit, with required `max_turns`). *Trigger:* a scenario authors a beat that needs an open-ended semantic exit — "the player admitted X," "Aria revealed Y" — that no deterministic predicate can express. Likely the bargain authored as real negotiation rather than 3-way classifier.
- **Salience scoring on storylet selection.** *Trigger:* beat count crosses ~20 with frequent overlapping eligibility and authoring-order tiebreaking starts producing stale sequences. The Director (one-beat-per-tick selection) handles current scale.
- **Scenes/phases as first-class architecture.** *Trigger:* a scenario with 3+ phases and ~30 storylets where gate-conjuncts on phase markers are climbing. Until then, a single phase quality (`player_canon.calibrated` or similar) is the right cost.
- **LiveAgentBeat as a new beat type.** Probably never; `completes_when_judge` covers the real use case at far smaller cost.

## Reading list

In order of how often you'll reach for them:

- **`CLAUDE.md`** — engine invariants, authoring vocabulary, the things that will silently break if violated.
- **`docs/frankenstein-adaptation.md`** — design rationale for the shipped scenario. Read for how a literary skeleton was chosen and translated; reuse the pattern.
- **`docs/LLM-game-research.md`** — field reference. Prior art, design vocabulary, Director/Performer split, AI-native primitive. Read once cover-to-cover; reach back for specific sections when stuck.
- **External canon** (a long weekend, if you haven't): Sam Kabo Ashwell, "Standard Patterns in Choice-Based Games" (2015); Emily Short, "Storylets: You Want Them" (2019) and "Beyond Branching" (2016); Jon Ingold's GDC 2017 "Narrative Sorcery" talk; Mateas & Stern's Façade AAAI 2005 paper; Hannah Nicklin's *Writing for Games* Ch. 6. Full bibliography in the research doc.
