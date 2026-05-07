# Designing an LLM-Driven Narrative Game: An Opinionated Field Guide

You are pivoting from agent engineering into a craft tradition that is older, weirder, and far more disciplined than most people in AI realize. The single best thing you can do before writing a line of code is to internalize that the interesting prior art for your project is not AI Dungeon or character.ai. It is Emily Short's blog, Jon Ingold's GDC talks, Sam Kabo Ashwell's pattern catalog, Failbetter's StoryNexus writing, Mateas & Stern's Façade postmortems, and the Kreminski/UCSC academic line on storylets. Those people have been solving — for two decades — most of the problems your prototype will hit in week three. The LLM is a new instrument, but the score has largely been written.

This document is long, opinionated, and assumes you already understand transformers, tool use, retrieval, judges, and structured output. Where I disagree with the user's framing (notably the "voice layer vs. spine" split), I will say so. Where evidence is thin, I will say so. Where the field disagrees, I will lay out the camps.

---

## 1. Narrative System Architecture

### The four canonical content-selection paradigms

Across two decades of indie and academic narrative games, four families dominate. Internalize them as a working vocabulary; almost every successful LLM-narrative architecture has been a hybrid built from these primitives.

**1. Branching graph / weave (ink, Twine, ChoiceScript, Yarn).** The script is a directed graph of nodes; choices are edges; state is a side-effect. Inkle's `ink`, the language behind 80 Days, Heaven's Vault, Sorcery!, Overboard!, Pendragon, Expelled! and TR-49, is the most refined member of this family. Jon Ingold's introduction to ink (Wireframe magazine; reproduced on Medium under D.S. Wadeson's "Game Writing Guide") describes its core construction as a "weave": text streams downward and branches and gathers (`-` syntax) without the author explicitly wiring nodes. The ink runtime exposes knots, stitches, diverts, weaves, list types, threads, and tunnels. The crucial property is that *most* choices in 80 Days are local weaves with implicit reconvergence; only a much smaller proportion are full diverts to alternate state-tracked content. This is what allows Inkle to ship hundreds of thousands of words with a small team — the authoring affordances reward dense local variation rather than global branching. Ingold has been explicit (his 2015 GDC talk "Adventures in Text," archive.org/details/GDC2015Ingold) that the right unit of variation is the line, not the chapter.

Failure modes: graphs scale combinatorially; "spaghetti node soup" once you exceed ~50 globally-relevant state flags; no good story for emergent recombination. Yarn (Night in the Woods, A Short Hike) and ChoiceScript (Choice of Games) sit in roughly the same family with stricter conventions; Twine sits closer to pure hypertext.

**2. Quality-Based / Salience-Based Narrative (Failbetter's Fallen London/StoryNexus, Emily Short's Versu, Cultist Simulator).** Coined by Alexis Kennedy and developed at Failbetter, with Emily Short's 2016 post "Beyond Branching: Quality-Based, Salience-Based, and Waypoint Narrative Structures" (emshort.blog/2016/04/12) the canonical articulation. The data model is:

```
Storylet := {
  id,
  preconditions: Predicate over Qualities (state),
  content: text/dialogue/effects,
  effects: Δ Qualities,
  weight / urgency / pinned?
}
```

The runtime is a filter: at any moment, find all storylets whose preconditions evaluate true on current state and surface them (or pick by salience score). Short's 2019 post "Storylets: You Want Them" (emshort.blog/2019/11/29) is the single best high-level explainer; her 2022 post "What does your narrative system need to do?" (emshort.blog/2022/04/09) sharpens the distinction between content selection, gating/filtering, and content generation as separable subsystems. Max Kreminski and Noah Wardrip-Fruin's "Sketching a Map of the Storylets Design Space" (ICIDS 2018) is the academic survey; it maps StoryNexus, StoryAssembler, The Ice-Bound Compendium, Reigns, and King of Chicago into a common design space.

The crucial property for an LLM-driven game: storylets are small, atomic, composable, and (critically) *do not require global graph authoring*. You write a storylet, declare its preconditions, and the engine handles where it appears. Kennedy's own retrospective writing on Sunless Sea (gamedeveloper.com/audio/postmortem-failbetter-games-i-sunless-sea-i-) describes how this enabled freelance writers to drop in islands without coordinating with the central plot — that is exactly the property you want for an indie LLM project where you are constantly adding content and your tooling cannot yet validate global consistency.

Failbetter themselves stopped using the term "quality-based narrative" because of brand confusion (Kennedy's 2017 post; the term "resource narrative" was floated). Short still uses "salience-based" to describe systems that pick the *most applicable* storylet rather than just any eligible one, which Façade also did (see §1, paradigm 4).

**3. Drama Manager (Façade, search-based DM in Anchorhead-style experiments).** Mateas & Stern's "Structuring Content in the Façade Interactive Drama Architecture" (AAAI AIIDE 2005, ojs.aaai.org/index.php/AIIDE/article/view/18722) and the longer "Façade: An Experiment in Building a Fully-Realized Interactive Drama" (GDC 2003, users.soe.ucsc.edu/~michaelm/publications/mateas-gdc2003.pdf) describe a three-tier system: ABL (A Behavior Language, a reactive planner built on Hap) controls character behaviors; an NLU template language maps player text into discourse acts (≈ intents); and a *drama manager* sequences "story beats" — clusters of joint dialogue behaviors — to bend tension toward an Aristotelian arc. A "beat" in Façade roughly corresponds to a McKee beat: a small exchange that meaningfully advances state.

The drama-manager idea — there is a separate authoring level *above* content selection that pursues *aesthetic goals* like rising tension, theme, character revelation — is the part most LLM projects skip and most regret skipping. Mark Riedl's surveys (e.g., the Riedl & Bulitko taxonomy of authorial intent vs. character autonomy, cited in the Drama Llama paper) formalize the same axis.

**4. Knowledge-tracked dialogue with autocut (Heaven's Vault).** Ingold's 2019 "Designing Investigate Conversations" (gamedeveloper.com/design/designing-investigate-conversations) describes the Heaven's Vault model in concrete detail. State is tracked at the level of knowledge atoms ("what does the player think happened to the robots after the Fall of the Empire?"). Conversation options are filtered by a knowledge-window: a question is only available if the player knows enough to ask it but not so much that it's irrelevant; once any NPC answers a question, it disappears from all NPCs' menus. This is a storylet-of-questions filter. Inkle's blog post on the "autocut" (inklestudios.com/blog) shows how, in a 3D world, conversations can be scripted with `autoCutZone` and `autoCutKnot` variables that gracefully terminate dialogue if the player walks away — a nice piece of practical authoring affordance worth stealing.

### Data schemas you can use today

For a small LLM project, I'd actively recommend a hybrid storylet + ink-style weave architecture, structured roughly like this:

```jsonc
// Quality (state variable)
{
  "id": "trust.detective.harriet",
  "type": "int",            // or "tag", "set", "bool", "enum"
  "min": -10, "max": 10,
  "default": 0,
  "surface_label": "Harriet's trust"   // human-readable, used for prompts
}

// Storylet
{
  "id": "harriet_offers_file",
  "preconditions": [
    "scene == 'precinct.night'",
    "trust.detective.harriet >= 3",
    "knowledge.case.victim_alibi == false",
    "!seen('harriet_offers_file')"
  ],
  "salience": "trust.detective.harriet * 2 + (urgency.case ?? 0)",
  "tags": ["scene", "investigation", "ally"],
  "content_ref": "scenes/harriet_offers_file.ink",
  "effects": [
    "knowledge.case.victim_alibi := true",
    "trust.detective.harriet += 1",
    "flags.harriet_compromised := true"
  ]
}

// Knowledge atom (separate from quality so LLM prompts can use them directly)
{
  "id": "knowledge.case.victim_alibi",
  "kind": "boolean",
  "summary_for_llm": "Whether the player has confirmed the victim was at the Maritime Club at 21:00.",
  "evidence_refs": ["clue.club.receipt", "testimony.bartender"]
}
```

The ink layer handles the *inside* of a storylet (the local weave: branching dialogue, rejoining, micro-variation). The storylet layer handles *which* scene you're in. The knowledge atoms are cleanly separable from numerical qualities and become the contract surface for LLM prompts and judges (see §5). This is essentially the architecture Drama Llama (Sun, Wang, Chung, Roemmele, Kim, Kreminski, arXiv 2501.09099) sketches — storylets with natural-language triggers — except I'd strongly argue you want preconditions in code, not natural language, because trigger evaluation by LLM is the slowest, most expensive, least debuggable part of their pipeline.

### What scales, what doesn't

- **ink alone** scales to ~80 Days size (≈500K words, modest state) with a single dedicated narrative architect. It does not scale to systemic emergent narrative.
- **Pure storylets** (Fallen London) scale to *millions* of words written by many freelancers but require a strong CMS and a lot of authoring discipline; the failure mode is "soup of disconnected encounters" which Failbetter mitigates with named storylet patterns: Carousel, Midnight Staircase (referenced in Short's "Mailbag: High-Agency Narrative Systems," emshort.blog/2017/05/25). Short notes that recent Failbetter content uses these less because grind-friendly patterns are no longer needed — which is itself useful intelligence.
- **Drama managers** require deep authoring; no shipped LLM-driven game has yet built a fully-realized drama manager and made it work, though Hidden Door's "story engine" with tens of thousands of human-written tropes (per their PC Gamer interview) is the closest commercial attempt.
- **Knowledge models** (Heaven's Vault) scale well for investigation/dialogue games but require disciplined authoring of the knowledge graph upfront. They're the natural fit for LLM games where you'll want to surface "what does the player know" to the prompt anyway.

---

## 2. Multiple Endings

### The "true ending" trap

Disco Elysium has three primary ending branches (Kim/Cuno/alone), each modulated by a small number of late-game state checks: success on the Authority check during the Mercenary Tribunal; whether Harry drank alcohol; whether the "Waste Land of Reality" thought has been internalized. The Disco Elysium Wiki and several walkthroughs (ginx.tv, gamersdecide.com, eneba.com) all describe the same skeleton: about *five* state variables drive the felt diversity of endings, plus around twelve game-overs that fire as failure-state newspaper headlines. The wide variation in epilogue text comes from a much larger set of downstream content that reads from many smaller flags (relationship to Kim, copotype, ideology stacking in the Thought Cabinet, side quest completion).

This is the lesson: **felt agency is dominated by the size of the readout, not the size of the branch.** A handful of decisive forks plus a richly-conditional epilogue beats twenty parallel branches.

Inkle leans on the same lesson. Heaven's Vault has no fail state and a huge variety of endings, but as Ingold explains in the GDC 2017 "Narrative Sorcery" talk and the gamedeveloper.com IGF Q&A, the architecture is "ad hoc" with "defensive logic." 80 Days has hundreds of routes but Ingold has said in multiple talks that the *level of drama is proportional to longitude* (the soup-too-hot example is in Sunless Sea, 80 Days and the rise of modular storytelling, gamedeveloper.com); the game leans on a *time gauntlet* (you must complete in 80 days) so that endings emerge from the combination of route, money, and gear rather than scripted forks. Citizen Sleeper's endings come from Drives — internalized goals you choose — and which faction-clocks complete first; the dice/clock system, inspired by Blades in the Dark per Gareth Damian Martin's gamedeveloper.com interview, means that any ending the player reaches *feels earned* because it was paid for in dice they personally rolled.

The trap to avoid: **the "true ending" gradient.** Disco Elysium fans argue endlessly about whether the Phasmid encounter is a "true ending"; the Insulindian phasmid talk in Stacey Henley's TheGamer piece is exactly this discourse. If your game telegraphs that one ending is the "real" one, players will reload to chase it and feel cheated by the others. Sorcery! 4 (Inkle, 2016) avoids this by giving genuinely incommensurable Crown of Kings outcomes. The cleanest design is convergent-with-coda: a small set of endings, each the "right" ending for the player who reached it, plus a per-playthrough generated coda that reflects accumulated state.

### How few variables can drive how much variation

Short's "Narrative States" post (emshort.blog/2019/11/23) is the best primer on this. Two to four well-chosen variables — placed on a ternary plot — can generate enough perceived variation that players will believe the game has dozens of endings. The trick is to *track derived state*, not raw state: instead of tracking `darcy_likes`, track `attention_share_darcy = darcy_likes / (darcy_likes + wickham_likes + collins_likes)`. The same observation applies to thought-trees: Disco Elysium's Thought Cabinet (Kurvitz's GameSpot Audio Logs interview is the primary source; Robert Kurvitz frames Thoughts as "loot from conversations") is a small set of slots that each carry rich semantic meaning — a derived, surfaced state that drives both gameplay and ending text.

For an LLM game specifically: an LLM is *spectacularly good* at generating personalized epilogues from a structured state ledger. Use this. Five variables and a 200-token state dump with the prompt "write Harry's farewell letter to Kim from the perspective of his copotype, factoring in the alcohol score and the internalized thoughts" will outperform forty hand-written branches. The architecture I'd advocate: variable-tracked at the spine, generative at the coda, with the LLM constrained by a schema and validated against a fact ledger (see §5).

### Surfacing eligibility without making it mechanical

The LLM era is the era where you can *finally* communicate consequence in prose without authoring 200 variants by hand. Disco Elysium's invisible-detour skill checks (analyzed in Ben Forrester's "Techniques of Thought" essay, parallelsuns.com) — where skills *interrupt* dialogue with their own perspectives — are an underused pattern that LLMs make trivially easy. Don't show the player a stat bar; show the player Logic muttering about it.

---

## 3. Sustaining Engagement Over 10–30 Hours

The LLM-narrative space has converged on a set of failure modes that read like the syllabus for a course you should take seriously.

### The known failure modes

- **Verb fatigue.** The player learns the input grammar in fifteen minutes and then plays the same fifteen minutes for hours. Cuckoo Network's writeup on "Negative Feedback on LLM-Powered Storytelling Apps" (cuckoo.network/blog/2025/04/17) compiles complaints from AI Dungeon, NovelAI and character.ai users about "looping" and "narrative drift."
- **State saturation.** As state accumulates, the model gets noisier: it loses the thread, contradicts itself, forgets the dossier. The SNAP paper (arXiv 2601.11529) measures continuity drops; LIGS at CHI 2025 (dl.acm.org/doi/10.1145/3706599.3720212) reports the same.
- **Plateau stakes.** No new external pressure; the world doesn't change.
- **World stasis.** No off-screen evolution. Ingold's Heaven's Vault explicitly counter-designs against this in the IGF Q&A: "Let's have worlds that move forward when you're not there."
- **Pattern recognition.** The player sees the seams of the LLM's preferred phrasing or the storylet system's idle behavior.

### Solutions from games that actually shipped

- **New verb introduction rate (Papers Please).** Lucas Pope's Papers Please introduces approximately one new mechanic per in-game day across 30 days; the development log on the TIGSource forum (preserved on the Papers Please wiki) is explicit about this. Each day adds a stamp, a rule, an ID check, an X-ray, an audio bug — never two at once. For a 10–30-hour LLM game, plan a verb-introduction curve and stick to it. New verbs mean new storylet preconditions, new judge rules, new ways state can change.
- **Subsystem layering (Disco Elysium, Citizen Sleeper).** Disco's Thought Cabinet, skill checks (active and passive), and white/red checks each layer over the dialogue base. Citizen Sleeper layers Drives on top of dice-on-clocks on top of NPC arcs.
- **Escalating consequence scope (80 Days).** As days pass, what you can no longer recover from grows. Inkle's "Problem of Failure" Medium post (medium.com/@inklestudios/the-problem-of-failure-518ec9c1c53e) is the canonical source.
- **Off-screen world evolution (Sorcery! 3, Heaven's Vault).** In Heaven's Vault, NPCs travel; in Sorcery! 3, day-night cycles change ports and characters. The 2017 GDC "Narrative Sorcery" talk is the technical writeup.
- **Midgame recontextualization (Roottrees Are Dead, Her Story).** Roottrees splits into two halves: in the second, the family tree gets retroactively complicated by claimants you have to disprove (per the IFDB review and walkthroughs at walkthroughking.com). Her Story has a similar mid-arc revelation about Eve. The pattern: at the midpoint, *re-read* prior content under a new lens.
- **Act structure.** Citizen Sleeper's dice/clock system runs short cycles inside larger arcs; Cultist Simulator runs Time-as-pressure (Kennedy's "Way through the Wood Part 2" Medium post) — every minute a Funds card vanishes; if you don't have one, Health is consumed.
- **Pattern-recognition counters.** Hypnospace Outlaw (Jay Tholen) holds 30+ hours of attention by *changing the entire texture of what the user-generated content looks like* every 1–2 in-game days; new in-fiction software updates, new aesthetic, new social dynamics. This is the thing LLM games most underweight: variation in *texture*, not just in plot beats.

A specific LLM-era observation: Kreminski and others (the FDG 2024 paper "Cheap and Easy Open-Ended Text Input for Interactive Emergent Narrative") argue that the failure mode is rarely the LLM's prose — it's the lack of *systemic surprise*. Players bail at the moment the system stops generating novel state, not at the moment the prose gets bad. Build for systemic novelty.

---

## 4. Story Craft for Branching Narrative

The IF/narrative-design community has a craft tradition that you should treat as canon. Here is the compressed version, with sources.

**Foreshadowing across branches.** Ingold's GDC 2017 "Narrative Sorcery" talk shows the technique he calls "defensive logic": every line is written assuming the player *might* know any subset of foreshadowed information. The trick is to write each scene as hard as you can while making the foreshadowing *robust to non-arrival*. Robert Kurvitz's interviews on Disco Elysium (his Rezzed 2018 GameAnalytics interview, his GameSpot Audio Logs episode) hit the same note: "We write every scene as hard as we can." Both of them treat foreshadowing as something you embed broadly so it lands somewhere, rather than narrowly so it lands precisely.

**Twists robust to player path.** Sam Barlow's Her Story is the cleanest example: there is no "twist sequence" — the twist is *whatever moment the player happens to discover Eve.* Barlow, in the AppUnwrapper interview, says: "I didn't really intend for there to be any red herrings… the actual writing itself was much more character-driven." Build the *character* deeply and the twist arrives wherever the player digs.

**Character voice consistency.** This is what LLMs are best at — and worst at. Best at: paraphrasing a character's voice across thousands of micro-prompts. Worst at: *staying* in voice when the player provokes them out. The remedy is dossier-driven prompting: for each NPC, a fixed dossier (history, vocabulary, register, what they will and will not say, three example quotes for tone, three forbidden topics with prepared deflections). Hidden Door's approach (per their PC Gamer interview and the Skywork.ai breakdown of their architecture) is the closest commercial precedent — characters as cards with attributes plus a small narrative trope library — though the press around their system is partially marketing-flavored, treat technical specifics with skepticism.

**Emotional escalation.** Ingold's narrative-design interview at narrativenews.substack.com puts it bluntly: "Don't overestimate the attention span of your audience." Cut hard. Hannah Nicklin's *Writing for Games: Theory & Practice* (CRC Press, 2022) — particularly Chapter 6, "Form-Led Design" — argues that escalation in games comes from *form changes*, not just content. The medium itself escalates: a phone game shifts from texts to emails to dating apps to forums (A Normal Lost Phone, Accidental Queens 2017); a detective game shifts from search to deduction to confrontation (Roottrees, Her Story). For a text-input LLM game, the escalation should also be *what the player can type*: in Act 1 they reply to texts; in Act 2 they compose; in Act 3 they have to lie convincingly to a system that knows them.

**Agency vs. authorial intent.** Every serious narrative designer points at this knot. The Riedl/Bulitko axis (cited in Drama Llama) is the academic version: how much character autonomy vs. how much authorial control? Inkle's answer is "high authorial control inside a fluid surface." Failbetter's is "high autonomy at the local scale, controlled drift at the global scale through Quality gating." Disco's is "the player can RP almost anything, but the central plot moves on rails." Avoid the "complete freedom" lie. Players want strong authorship in their stories; they want the *illusion* of freedom plus the felt sense that their input mattered. Ingold says this directly in his Browser interview (thebrowser.com/notes/jon-ingold/): "I don't see interactivity as that big a deal."

**Earned endings.** The principle: the player must have *paid* for the ending in the currency of the game. Citizen Sleeper's currency is dice and stress. Disco's is alcohol and Thought slots. 80 Days' is days and money. If the LLM is going to compose epilogues, the *inputs* to that composition should be states the player visibly accumulated.

---

## 5. LLM-Specific Architecture (the deepest section)

### What LLMs are actually good at

You are the audience that doesn't need this rehearsed, but it matters that the design philosophy be grounded:

- Stylistic paraphrase and register-shifting (turning structured state into prose in a chosen voice).
- In-the-moment improvisation within tight contextual scaffolds.
- Intent classification of free-form player text (especially with structured output and a closed schema).
- Sensitivity to *texture* in player input — register, formality, hedging, profanity, emoji density, sentence length. This is the most underused capability and the most novel design space.
- Personalized epilogue generation from structured state.
- One-shot variation: "give me 20 ways an exhausted detective greets a witness."

### What they are actually bad at

- Long-term consistency without an external memory.
- Refusing to invent. They will conjure entities, names, facts; "character hallucination" (RoleBreak, arXiv 2409.16727) is a distinct failure mode from generic factual hallucination.
- Counting and bookkeeping (resources, turns, days).
- Knowing when to stop a scene.
- Hard rule enforcement (will the gun fire? did the door lock?).
- Deterministic behavior across replays.

The Cuckoo Network compendium of complaints, the SNAP paper, and the LIGS CHI study converge: LLM stories drift, repeat, lose causality, and eventually break temporal/spatial consistency when player input goes off-script.

### Pushing back on "voice layer vs. spine"

You framed your intuition as a "voice layer vs. spine" split. I want to push back on this — not because it's wrong but because it's incomplete and, I think, dangerously load-bearing if you stop there.

The right mental model is **three layers, not two**:

1. **Spine** — deterministic state and content selection. Storylets, qualities, knowledge atoms, scene graph. No LLM here. Authored, debuggable, savable.
2. **Director** — a small LLM-or-symbolic layer that *chooses what is salient* given player input and current state. This is the layer that does intent classification, picks among eligible storylets when there are ties, and decides "is the scene over?". It is the equivalent of Façade's drama manager; it should be *constrained* to selecting from authored options, not generating new ones. It runs on every turn.
3. **Performer** — the LLM-as-actor layer that paraphrases and improvises within the bounded space the Director hands it. It is given the storylet content, the NPC dossier, the current knowledge ledger, and constraints; it produces the prose that reaches the player.

The "voice layer" concept conflates the Director and the Performer. They want different prompts, different temperatures, different validation, and very possibly different models. The Director is a classifier-with-judgment that needs to be *consistent* across replays and amenable to logging and testing; you want low temperature, structured output, and a tight allowed-action grammar. The Performer is a stylist; you want higher temperature, character-conditioned prompts, and validation against a fact ledger only on outputs that touch state-relevant claims.

This decomposition matches the architecture in the Drama Llama paper (Sun et al., arXiv 2501.09099) and in PANGeA (Buongiorno et al., arXiv 2404.19721): both keep generation behind a structural scaffold that decides *whether* a storylet should fire and *what* counts as canonical.

A second pushback: **the Performer should not "own voice" globally.** Voice belongs to the dossier, not to the model. If you let the LLM "be the voice layer," you will drift — Disco's Twitter-aggressive register (Kurvitz, GameSpot) won't survive a 5K-token context full of paraphrase. Treat each major NPC as a small fine-tunable persona with its own example bank, and re-inject a 200–400-token dossier on every Performer call. Cheap and effective.

### Validation patterns

This is where you, as an agent engineer, have a structural advantage. The toolkit:

- **Schema-constrained outputs.** Every Performer call returns JSON like `{prose: string, claimed_facts: [{atom_id, value}], requested_state_changes: [...]}` and the prose is rendered only after validation. Reject and regenerate on schema break. Standard tool-calling discipline.
- **Fact ledger / registry checks.** Maintain a ledger of canonical facts: NPC names, locations, items, relationships. Any prose that introduces a *new* proper noun is checked against the registry; if absent, either reject (strict) or auto-add with a flag and surface to the author (lenient). Inkle has spoken in talks about using "defensive logic" — this is the LLM-era equivalent.
- **Judge models on canonicity.** A second LLM call evaluates: "does this prose contradict any fact in the ledger? does it stay in voice for this character? does it advance any storylet effects not in the requested_state_changes list?" This is expensive; budget for it in scenes where state-bearing claims are likely and skip it for ambient text. Inworld and others have published similar pipelines but the academic primary source is the Drama Llama trigger-evaluation system.
- **Closed-world dossiers.** NPCs cannot reference characters outside their dossier without the Director explicitly granting it. The PANGeA paper explicitly invokes this constraint as the way they avoid LLM scope creep.
- **Determinism by hash.** For replay value, seed the Performer with a hash of (state, scene, player_input). Repeat playthroughs that re-hit the same state produce identical prose. This catches more bugs than you would believe and addresses one of the chief AI Dungeon complaints (capricious model behavior).
- **Soft retry with reward signals.** Latitude has reportedly used Direct Preference Optimization on retry-button data (skywork.ai's writeup of latitude.io's blog; this is secondary so caveat emptor). The signal "user pressed retry" is gold for training Performer-style models on what reads as good in your specific game's voice.

### What does and doesn't transfer from non-LLM design

**Transfers perfectly.** Storylet decomposition (storylets are even better with LLM rendering); knowledge atom tracking; act structure; opening design; failure-as-fiction (Inkle's whole thesis is more, not less, applicable when an LLM can paraphrase the failure into the fiction); dossier-driven NPC writing; quality-based gating.

**Transfers with friction.** Drama-manager-style global pacing (LLMs make local pacing easier but global pacing harder because the Performer encourages sprawl). Convergent endings — the LLM will happily generate "true ending energy" everywhere if you let it, drifting toward cliché.

**Doesn't transfer.** "Type anything" as a *full* commitment — Façade tried it and produced enormous authoring burden for marginal player power; modern LLMs reverse the equation but introduce the inverse problem (all input is "understood" but rarely consequential). The ChoiceScript-style tightly-curated stat readout doesn't transfer, because the LLM will produce slipstream prose that breaks the readout's discreteness; you have to choose one register or the other for any given scene.

### New design opportunities

This is where the genuinely new ground is, and where I think the user's current framing underweights things:

- **Style as mechanic.** Detect player register (formal, casual, hostile, hedged, profane) and use it as a state variable equal in standing to any other quality. The LLM is uniquely good at this; nothing else can do it cheaply. A therapy-game NPC who reacts not to *what* the player typed but *how* (e.g., word count, hedge frequency, emoji presence) is a mechanic only LLMs make tractable. Kreminski's "Throwing Bottles at God: Predictive Text as a Game Mechanic" (ICIDS 2018) is the prior art most aligned with this; it uses smartphone predictive text as gameplay material.
- **Personalized epilogue as default, not bonus.** Take the entire state ledger, dump it into a 4K-token prompt, generate a per-playthrough epilogue. This solves the "true ending" trap because every ending becomes a true ending of *that* playthrough.
- **Dossier-bounded improvisation.** The LLM can fill in NPC small-talk, scene description, ambient barks, *up to* the dossier's edge. This fills the "every NPC repeats the same line" hole that has plagued narrative games forever.
- **Texture-responsive systems.** The system reacts to *how* the player writes more than *what* they wrote: were they curt with the witness? did they use the suspect's first name without permission? did they swear? These are state changes that require zero authoring.
- **Free-form input as evidence.** Roottrees Are Dead lets you type any search; the game responds because the dataset is finite and curated. An LLM equivalent: type any question to a dossier-bounded NPC, who answers from their dossier. This is the *good* use case for LLM in games. (Hidden Door is essentially building this for IP-licensed worlds.)

### The retention problem

LLM-driven games — AI Dungeon, character.ai, Replika, NovelAI — have a real retention problem that is partly about content moderation drama and partly about a deeper design problem the field has not solved. The Cuckoo Network writeup catalogues complaints; the Latitude Voyage TechCrunch article (techcrunch.com/2026/04/21) describes Latitude's five-year "World Engine" effort to build NPCs that remember and react with consistency.

My read: the retention problem is structural. *Pure* LLM games with no spine have nothing to retain players past the toy phase. The systems that retain players for hundreds of hours — Fallen London, Disco Elysium, Citizen Sleeper, Cultist Simulator — all share the property that the *system* keeps generating problems for the player, independent of the prose. AI Dungeon does not. This implies: retention does not come from LLM novelty; retention comes from a systemic loop that LLM prose decorates.

This is also Kreminski's point in "Understanding Systems Suspense" (the Polaris Game Design Retreat report with Grinblat, Lennon, Manning, Pepers, Scott): players stay engaged with *systems* that produce suspense — clocks ticking down, factions about to collide, resources draining. LLM prose is *condiment*. If the spine has no suspense, no amount of beautiful prose retains anyone.

### Specific projects worth tracking and learning from

- **Drama Llama** (UCSC/Midjourney lab, arXiv 2501.09099) — natural-language storylet triggers; small authoring study suggests the approach works but trigger evaluation latency is the bottleneck.
- **Dramamancer** (UIST 2025) — successor; integrates LLM more deeply.
- **Loose Ends** (Kreminski, Dickinson, Mateas, Wardrip-Fruin, 2022) — addresses the "directionlessness" problem by tracking parallel plot threads explicitly.
- **Hidden Door** — production attempt; their architecture (manually-curated trope library + LLM blending) is the most disciplined commercial attempt; treat the press as press.
- **PANGeA** (Buongiorno et al., 2024) — RPG with input validation between player text and game rules. Useful template.
- **SNAP** (arXiv 2601.11529) — Cell-based segmentation and Plan-driven generation to constrain LLM narrative drift; their 18.1% improvement on Continuity is honest though small-scale.
- **Inworld** — production; their NPC dialogue system has shipped in commercial games but academic write-up is sparse.
- **The Riedl line** (Georgia Tech) — long history of plan-based and LLM-augmented narrative generation; the more recent work on commonsense reasoning and theory of mind is relevant for NPC modeling.
- **James Ryan's** dissertation work on character simulation and Talk of the Town remains essential reading for how to model offstage life cheaply.
- **Versu** (Richard Evans + Emily Short) — pre-LLM but the social practice modeling is aligned with what you want NPCs to do; partial reconstruction efforts (Praxish) exist.

I have not seen Jon Ingold publish a substantive piece specifically on Inkle's LLM experiments. He has commented on Mastodon (mastodon.gamedev.place/@joningold) and in the *Kaleidoscope* book interview at narrativenews.substack.com that LLMs sit uneasily with Inkle's craft model; he is, by design, conservative about generative tech for a studio whose value is the precision of its hand-written ink. I would not assume there is a hidden Inkle LLM stack to reverse-engineer; what they publish about ink and about defensive narrative architecture is the actual gold.

---

## 6. Free-Form Text Input

The single most important design question. Free-form text input is *the* feature the LLM era unlocks; it is also the feature whose history is most sobering.

### When does it actually work as a core mechanic?

Almost never, historically. Parser IF (Zork through Counterfeit Monkey) succeeded with a *constrained* free-form vocabulary plus enormous authoring labor. Façade was the high water mark for free-form input mapped via NLU templates to discourse acts; Mateas & Stern have been transparent that the authoring burden was vast and the player experience uneven.

The four conditions under which free-form text works as a core mechanic:

1. **The fiction absorbs failure.** The classic example is Inkle's *Fail-Safe* (2000), Ingold's parser game where the player commands a starship engineer over radio; if the parser misunderstands, the engineer simply mishears. The player's failure becomes character failure. The same trick is in Façade (Trip and Grace can plausibly be confused) and in Counterfeit Monkey's wordplay (the device just buzzes and refuses).
2. **The input axis is narrow and evaluable.** Her Story takes any English string; it evaluates it against a single axis (does this match clip transcripts?) and surfaces top-five-by-recency results. Roottrees Are Dead does the same with a search engine, and the design works because the dataset is curated and finite.
3. **Structured intent extraction sits between input and state.** The LLM-era pattern: free-form text in, structured intent out, deterministic state change downstream. This is exactly the agent pattern you already know, applied to fiction. The PANGeA architecture and Drama Llama's trigger system are both versions of this.
4. **Constraints as creative focus.** When the player's input is *pruned* by the fiction — a phone keyboard, a 280-char limit, an interrogation transcript — the input becomes more meaningful, not less. Mu Cartographer (Titouan Millet) and Cibele (Nina Freeman) both lean into this.

### The space between "menu of options" and "type anything"

This is where the design opportunity is. Some intermediate forms:

- **Fixed verbs + free-form objects** (parser IF). Type any noun phrase, choose from a small verb set. The LLM can generalize the noun parsing.
- **Free-form text classified into a hidden menu** (Façade). The player thinks they're typing freely; the system maps to discourse acts. This works when the fiction makes mismatches plausible.
- **Free-form text scored on multiple axes** (the LLM-novel design). The player's text is scored on (intent, register, hedge level, hostility, length); each axis becomes a state input. *No* axis needs to be a perfect classifier — the score is the mechanic.
- **Free-form text with constraint prompts.** The interface tells you *what kind* of text it wants ("write a three-line statement to the press"). The LLM then evaluates that text against rubrics that fit the prompt. This is the design space Hidden Door is mining.
- **Free-form text-as-evidence** (Roottrees, Her Story). The player's input retrieves; the game's body is the corpus.

### Prior art outside LLM games (what to actually steal)

- **Counterfeit Monkey** (Emily Short, 2012). A parser IF where the verbs are letter-removal devices. The player types `USE M-REMOVER ON MONKEYS` to turn monkeys into onkeys. Source available on GitHub (i7/counterfeit-monkey). The game's puzzle chart (emshort.blog/2020/01/20) is a beautiful example of constrained free-form input with deep authoring.
- **Galatea** (Emily Short, 2000). Conversation as the entire mechanic; topics as the verb space.
- **Photopia** (Adam Cadre, 1998). Tightly-narrated parser IF where free-form input does not branch the story; instead it lets the player *participate* in pre-authored beats.
- **Aisle** (Sam Barlow, 1999). One-move parser IF; entirety of the game is what the player chooses to do in one verb. Each input retrieves a different vignette.
- **1001 Nights** (Jason Nelson and others; an art-piece reference more than a game) — generative narrative as performance.
- **Façade** as documented in the Mateas & Stern AAAI papers: not for emulation but for the authoring lessons. NLU templates → discourse acts → ABL → drama manager.

The shape of all these games is the same: free-form text *contributes* to a richly-authored fiction; it does not *generate* the fiction.

---

## 7. Input Medium and Genre Fit

This is where I want to push back hardest on the implicit "LLM games are best at fantasy adventure / horror" assumption.

### The fiction has to absorb LLM imperfection

The genre-fit question is really a question about *whose voice the LLM is performing* and *what kind of error is in-fiction.* Reframe accordingly:

- **Email / correspondence.** Outstanding fit. People in real life are rambling, semi-coherent, tonally inconsistent; an LLM impersonating a correspondent has all the deniability it needs. *Digital: A Love Story* (Christine Love, 2010), *Emily is Away* (Kyle Seeley), *Cibele* (Nina Freeman), *Secret Little Haven* (Victoria Dominowski) are the canon. The LLM-era version: a years-long correspondence with a single character whose voice slowly shifts as the relationship deepens.
- **Phone OS / found media.** *A Normal Lost Phone* (Accidental Queens, 2017), *Sara Is Missing*, *Simulacra*, *Hypnospace Outlaw* (Jay Tholen, 2019). The fiction *is* the interface. Texts are short, emails long, forum posts ranty — variation in form is variation in pacing. Hypnospace Outlaw's 30+ hours of attention come from the *texture* changing across in-fiction software updates.
- **Interrogation / interview / therapy.** *Her Story*, *Telling Lies* (also Barlow), *The Hayseed Knight* (CrossSushi). The LLM is performing the interviewee, not the protagonist. The player asks; the model answers from a dossier; mistakes read as evasions, lies, or psychological complexity. This is the genre I would most strongly steer you toward if you want the LLM to do real work.
- **Journalism / investigation.** *Orwell* (Osmotic Studios), *Roottrees Are Dead* (Jeremy Johnston / Robin Ward), *Return of the Obra Dinn* (Lucas Pope). The player synthesizes; the LLM provides searchable, dossier-bounded, internally consistent witness statements and documents. Roottrees especially is the sleeper template here — its core design is "free-form search of a finite corpus" and that translates trivially to LLM-augmented worlds where the corpus is generated lazily but anchored to a registry.
- **Diplomacy / surveillance.** A Cold-War cable game, an embassy correspondence game. LLMs can do period register and political euphemism with surprising poise.
- **Historical fiction / epistolary period.** Letters home from the front, an exchange between sisters across the 1860s, monastic correspondence. The LLM's anachronisms read as character flaws.
- **Literary drama, comedy, sci-fi correspondence.** Vastly underexplored. Comedy especially: an LLM is funny in a way most game NPCs aren't, if you constrain it tightly enough. Inkle's *Overboard!* (Ingold's noir-femme-fatale design, per his narrativenews.substack.com interview) shows comedy in a non-LLM choice game; the LLM analog is wide open.

### Where horror is not actually the best fit (push back)

Horror is overrepresented in LLM-game pitches because of the AI Dungeon shadow and because horror tolerates incoherence. But horror is *also* the genre where LLM weaknesses (sudden tonal shifts, inappropriate warmth, refusals on harm) cause the most damage. Cosmic horror works (Cultist Simulator's text fragments would translate beautifully to LLM-rendered) but visceral horror tends to fight the model. You will spend more time fighting the safety layers and the model's tonal averaging than you will building tension.

Consider instead **literary drama** and **investigative correspondence** as the strongest genres for an LLM-driven text-input game. These are the two genres where (a) free-form text is *natural* in the fiction (people write to each other; people interrogate each other), (b) the LLM's improvisational mistakes are absorbed by character imperfection, (c) the medium constraints (short texts, formal letters) prevent runaway prose, and (d) there is almost no shipped competition.

---

## 8. Game Openings

The first 10–15 minutes are entirely a teaching problem disguised as a fiction problem. Here is what the canon does:

- **Her Story.** Sam Barlow pre-fills the search box with `MURDER`. Returns four clips, designed by Barlow to seed exactly the next four searches (per the Wikipedia article on Her Story and Barlow's AppUnwrapper interview). Lesson: the most powerful tutorial is a *pre-typed first input*. Steal this directly.
- **Roottrees Are Dead.** Opens with a fixed test: identify the three sisters from one photo and a search engine, with explicit hand-holding (the walkthrough at neoseeker.com lays out the exact cadence). Lesson: the first thirty minutes of a deduction game are explicit instruction inside a low-stakes test scene.
- **Papers Please.** Day 1 has only "check the passport, stamp it." Day 2 introduces date checks. Day 3 introduces nationality. Pope's Game Developer Q&A and his TIGSource devlog (preserved on the Papers Please wiki) are explicit about the curriculum. Lesson: introduce verbs at *the rate of one per session*.
- **A Normal Lost Phone.** Opens on the messaging app of a lost phone; the implicit verb is "snoop." Lesson: the medium *is* the tutorial.
- **Hypnospace Outlaw.** Opens with a fake employer onboarding video. Lesson: in-fiction tutorial-as-training-day.
- **Citizen Sleeper.** First cycle gives you five dice, a few verbs (work, eat, sleep, talk), and a cycle clock. Damian Martin's Game Developer interview says Blades in the Dark's economy of verbs informed everything. Lesson: the opening is a *single clean cycle* of the core loop.
- **Disco Elysium.** Wakes Harry up on the floor with no memory. The first verb is *deal with one's own mind*. The dialogue tree of "the necktie wants you to drink" and "the ceiling fan beckons death" trains you to read skill voices. Lesson: the opening is a calibration of *what kind of game* this is. Not a tutorial of mechanics, but of register.
- **Bioshock.** "Would you kindly" is foreshadowed before the player has any agency. Lesson: the opening is also where you plant your endgame thematic payload.
- **Papers Please** redux: the very first day's *very first* entrant is the inspector's brother-in-law. Pope plants the tension between rules and family in interaction one. Apply this principle: the *texture* of your endgame conflict should be present in minute one, even if the player can't yet name it.

The composite pattern, expressed for an LLM text-input game:
1. The first input box should have placeholder text that is also the first valid input.
2. The first NPC's first message should *carry the tone of the entire game*.
3. The first verb the player learns should be a verb they will use for the rest of the game.
4. By minute 10 the player has changed at least one piece of state and seen the change reflected in subsequent text.
5. By minute 15 the player has seen the first hint of the endgame's central tension, embedded in a low-stakes interaction.

---

## 9. MVP Scope

You are pivoting after a small amount of prior work, likely solo or in a team of two or three, with strong agent-engineering muscle. The mistake almost everyone in your position makes is to attempt a generative open world; the right move is the opposite.

### The questions an MVP must answer

A small-team narrative MVP is a vertical slice that exists to answer three questions, not to "be the game":

1. **Does free-form text input feel meaningful in this fiction?** Specifically: when the player types something they think matters, does the game register it in a way that feels seen — without breaking the fiction?
2. **Does the system retain the player past the first hour of novelty?** This is the AI Dungeon question. The honest answer is almost always *no, unless there is a systemic spine independent of the LLM*.
3. **Can a content author add a new scene in under a day without touching code?** This determines whether you can scale to 10–30 hours.

If the answer to all three is yes, you have a real game. If any is no, the game is not yet defensible.

### What is deliberately out of scope

For a vertical-slice MVP I would aggressively cut:

- Multiple acts. One act, ~45 minutes of play.
- Multiple NPCs with deep arcs. One central NPC with a deep dossier; two background NPCs with 200-token dossiers each.
- Branching endings. One scene with one consequential decision and three readouts.
- A drama manager. A single hand-coded scene clock is enough.
- Voice, art, music beyond placeholder. This is not where the risk lives.
- Save/load beyond a local JSON dump.

### What is on-scope

- A storylet engine with maybe 12–20 storylets, half of which gate on player-input-derived state.
- A knowledge atom ledger (5–10 atoms is enough).
- One central NPC dossier (1500–3000 tokens, with explicit "will say / won't say / will deflect" sections) plus an example bank of 10–20 in-voice quotes for tone.
- The Director (intent classification + storylet selection) implemented with structured output and a small allowed-action grammar.
- The Performer prompt with judge-validation against the ledger.
- A single piece of texture-as-mechanic — for example, the NPC tracks a register score that shifts their willingness to disclose.
- One generated epilogue at the end driven entirely by the ledger.
- A trivial authoring file format (YAML or JSON) that lets you add a new storylet without recompiling.

### How small narrative teams have built vertical slices

- **Lucas Pope** built the first eight days of Papers Please as a free public beta in Haxe/OpenFL (still on dukope.com); it tested the core loop on real players before expansion. Eight days, one mechanic, one location.
- **Sam Barlow** built Her Story by writing the script first as a spreadsheet that crashed his laptop, then filming over five days in Truro. The MVP was the search engine and a single set of clips; everything else was iteration on top.
- **Inkle's** prototype for 80 Days, per Joseph Humfrey's Game Developer post on open-sourcing ink, was a single trip around the world; once that ran, they layered in cities. The slice tested whether modular text-driven travel was even fun.
- **Failbetter's** prototype for Sunless Sea, per Alexis Kennedy's gamedeveloper.com postmortem, leveraged the existing StoryNexus engine; the MVP was islands in JSON pulled by the Unity client. They proved the core loop on infrastructure they already had.
- **Nerial's** Reigns prototype was reportedly built in a few weeks; the entire game's interaction is left/right swipes on a single card. The slice was the swipe + four stats + ten cards.
- **Gareth Damian Martin** prototyped Citizen Sleeper as pen, paper and physical dice before writing code (Game Developer, Origin Story podcast). The mechanical thrill of "rolling five ones and feeling despair" was found by literally rolling dice on a table.

The composite lesson: build the smallest rig that lets you *play the loop* with a real person, in any medium. Paper, spreadsheet, Discord-bot, command-line — whichever lets you get a friend to play in two days. Only after the loop survives unaided contact with a player do you build the system around it.

---

## What I would actually do, if I were you

1. **Pick a medium that absorbs LLM imperfection in-fiction.** Email, interrogation, therapy notes, embassy cables, journal-against-AI. Not a fantasy hero in a forest.
2. **Build a storylet engine first, in code, with no LLM.** Write 15 storylets by hand. Play it. Make sure the spine works as a game.
3. **Pick one central NPC and write a long dossier.** Voice them by hand for the first ten exchanges. Build a register-detection state variable around them. This is your single LLM mechanic.
4. **Add the Director and Performer separately.** Validate each. Log every Performer call with its prompt and ledger. Build the judge before you ship to a single playtester.
5. **Put a generated epilogue at the end.** Watch playtesters' faces when they see their playthrough reflected back to them. This is the moment the LLM earns its keep.
6. **Read, in this order:** Sam Kabo Ashwell's "Standard Patterns in Choice-Based Games" (2015); Emily Short's "Storylets: You Want Them" (2019), "Beyond Branching" (2016), and "What does your narrative system need to do?" (2022); Jon Ingold's "Designing Investigate Conversations" (Game Developer, 2019) and the GDC 2017 "Narrative Sorcery" talk; the Mateas & Stern AAAI 2005 Façade paper; Kreminski et al., "Sketching a Map of the Storylets Design Space" (ICIDS 2018) and the Drama Llama paper (arXiv 2501.09099); Hannah Nicklin's *Writing for Games* Chapter 6. That is a long weekend of reading and worth two months of design iteration.

The opportunity is real. Almost no one with serious agent-engineering chops has yet shipped a narrative game built on the storylet/QBN tradition with LLMs as the performer layer. The space between Hidden Door (overscoped, IP-licensed, multiplayer) and AI Dungeon (no spine) is wide, mostly empty, and genuinely undertheorized — and the discipline you bring from agent engineering (validation, schemas, judges, registries) is exactly what is missing from the field. The art is to not let the LLM do the parts of the job it cannot do, and to let it shine on the parts that no other technology can do at all.

---

## Postscript: patterns that emerged after writing

This guide was written from prior art before extended design work on a small inbox-genre LLM game. The argument the doc makes still holds. A handful of patterns surfaced in later design loops that this draft underweighted; they sit alongside, not above, the rest of the doc.

### Per-storylet framed salience instead of a global texture extractor

§5 recommends a global "register score" — a classifier that runs on every player input and writes to a continuous quality. There is a cleaner alternative: attach the salience question to the storylet that wants to consume it. When the player replies to a storylet, *that storylet's* classifier runs; when they reply to a storylet without a classifier, nothing extra runs. The classifier returns one of a closed `allowed_values` enum and writes the mapped score to a named quality.

This pays LLM cost only when relevant. It lets the author frame the salience question in the storylet's own context ("did the player engage with *this specific calibration prompt*?"). It composes naturally with the closed-enum + score-map idiom. Sun et al.'s *Dramamancer* (arXiv 2601.18785) describes a similar interpretation/instantiation split; the per-storylet version of interpretation lands at lower cost and with strictly less drift than a global one.

The natural extension when one axis stops being enough — and on any non-trivial scenario it will — is **multi-axis classification per storylet**: a list of question/allowed_values/map entries, batched into a single LLM call returning structured output. This recovers the global-texture-extractor's expressivity without the always-on cost.

### Recent-window predicates are absent from the QBN canon

QBN predicates evaluate cumulative state: trust ≥ 5, fear < 3. They do not capture *now*: "the last three replies have been hostile," "the most recent reply was a refusal." Galatea's topic salience handled the *now* axis with decay; the QBN tradition mostly does not. For relational stories — anything where the engine wants to register *the temperature has just dropped* rather than *the temperature is below threshold* — a window-shaped predicate (`QualityGteRecent(window=N, threshold=T)`) is the natural addition and is felt missing on the first pacing-driven scenario.

### Storylet body as generator, not branch

Pre-LLM storylets (Fallen London, StoryNexus) carry paragraphs of authored prose with effects. The LLM-era move is to make a storylet's content a **directive plus runtime render**: author writes intent, must-include facts, must-not constraints, length window; the runtime calls the LLM with the directive, the dossier, and the conversation history. One storylet that re-fires across a session — a multi-turn storylet — becomes the body of an entire scene. This is the move that makes LLM-storylet engines materially cheaper to author than their pre-LLM predecessors: not because the LLM produces better prose than a hand-author, but because the same authored intent can produce many in-character renders.

This is a vocabulary correction more than a technical one. The guide already implies it; calling it out explicitly is worth the page.

### Commitments as an authoring convention, not new machinery

§5 names a fact ledger as the contract surface for Performer prompts. The lightweight version is a **namespace of qualities** — `commitment.<fact>` — that storylets write to on fire and that the chat agent's instructions surface on every render. This is mostly a naming convention plus a tiny schema sugar (a `writes: dict[str, float]` field on a storylet's spec, plus an `@instructions` block that pulls qualities under the `commitment.*` namespace). It's worth adopting once any storylet's directive needs to reference a fact set up by an earlier storylet; over-engineering for a handful of beats whose ordering is already captured by `BeatFired` predicates and the LLM's natural attention to recent message history.

### What §5 would say if rewritten

The Director/Performer decomposition still holds. The implementation note that updates the recommendation: **make Director-shaped salience an authoring affordance attached to each storylet** (a per-storylet classifier with closed `allowed_values`), not a global pass that runs on every input. Multi-axis the storylet-local classifier when one axis stops being enough. Treat the texture-as-mechanic argument as something delivered through this affordance, not through a separate continuous register classifier.