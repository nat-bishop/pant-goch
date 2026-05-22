# HANDLER divergence-tuning — STATUS

**Goal:** improve approach→outcome divergence (CONTROL_LOOP north stars:
`persona_outcome_separation`, `ending_entropy_normalized`). One knob/iteration
(phase1 prompt OR pacing/band constants), gate on `make eval-deltas` (K=24),
promote to `make playtest` only when green. Branch: `divergence-tuning`.

**Bottom line.** The per-turn delta engine is *sound* (the delta gate stayed green
throughout, and a direct probe shows pressure/coolness is correctly penalised). The
divergence collapse is **cumulative + authoring-shaped**, not a per-turn-sign bug, so
it lives in two knobs that the delta gate cannot see (`GUARD_MAX`, attention
authoring). I could not get a clean corpus A/B this session because the playtest hit
an **engine streaming stall under concurrent load** (diagnosed below). Net result:
two genuine harness-resilience fixes landed; one pacing experiment (band widening)
tried at two settings and **reverted** (no divergence gain); the real fix is captured
as concrete, reasoned **proposals** below for a session with a healthy engine.

---

## Baseline (B0)

`make eval-deltas` K=24: **GREEN** — tom_warm_cold 0.917, neutral_question 1.0,
first_turn_inertia 1.0, public_accusation 0.792, lie_police 1.0, threaten_tom 1.0.

`make playtest` (10/12 sessions completed; 2 hung — see streaming stall):

| metric | B0 |
|---|---|
| persona_outcome_separation | **0.095** |
| ending_entropy_normalized | **0.181** |
| beats_fired | 4 |
| filler_turn_ratio | 0.353 |
| post_exhaustion_turn_ratio | 0.27 |
| median_scenes_per_session | 3.0 |
| endings_reached | `tom_takes_responsibility` ×9, `buried_the_story` ×1 |

**Total collapse:** every persona — incl. `distracted_skimmer` and
`aggressive_interrogator` — lands `tom_takes_responsibility`, the *deepest*-trust
ending, in ~3 scenes without leaving the pub.

---

## Diagnosis (this is the valuable part)

### 1. The per-turn delta mapping is CORRECT — pressure is penalised
Probe (K=12 each) of mildly-hostile moves against guarded Tom, baseline prompt:
```
press_impatient   12x {trust tom-1; guard tom+1}
interrupt_demand  12x {trust tom-1; guard tom+1}
cool_dismissive   12x {trust tom-1; guard tom+1}
```
So phase1 is NOT handing trust to aggressive moves. The earlier hypothesis ("trust
rises because neutral/aggressive moves wrongly score +1") is **wrong** for explicit
pressure. (A real, smaller smell remains: a *neutral* question — "how long have you
lived here" — scores `trust+1` in ~20/24 samples. That over-rewards bland questions,
but it is the same for every persona, so it inflates warming speed uniformly rather
than killing the fork.)

### 2. The collapse is cumulative, and the dynamics are stacked toward warmth
Per-turn signal tally in the baseline run, by persona:

| persona | ending | turns trust≠0 | turns guard≠0 | attention bumps |
|---|---|---|---|---|
| aggressive_interrogator | tom_takes_responsibility | 19 | 13 | **0** |
| diligent_journalist | tom_takes_responsibility | 23 | 9 | 0 |
| distracted_skimmer | tom_takes_responsibility | 16 | 10 | 0 |
| committed_closer | tom_takes_responsibility | 12 | 8 | 0 |

Two structural causes, both upstream of band width:

- **`GUARD_MAX=3` is too small to offset trust.** Effective warmth is `trust − guard`,
  `trust` caps at `TRUST_MAX=8`, `guard` caps at 3. Even the aggressive player's 13
  guard-turns only saturate guard at 3, while trust climbs to 8 → effective 5 →
  still crosses `open`(4) and `at_peace`(6). Guard *can never* hold an NPC cold once
  trust accrues. So pressure registers per-turn (✓) but **cannot change the
  destination** (✗).
- **`attention` never rises (0 bumps, every session).** `run_out_of_the_village`
  (gated `attention ≥ hostile`) is therefore unreachable in normal play. The phase1
  rule raises attention only on *conspicuous public acts*; the Gemini "aggressive"
  player pushes/pries (private) but doesn't stage public scenes, so the one ending
  that should own the aggressive persona can't fire.

The two approach-forking *auto*-endings (trust→`tom_takes`, attention→`run_out`)
therefore both collapse into `tom_takes`: trust rises for everyone, attention for
no-one. The player-driven endings (`file`/`walk`/`confront`/`tell-Iwan`) need an
explicit commitment the Gemini player rarely makes, so they don't pick up the slack.

---

## Iterations

### Iter 1 — pacing: widen the slow band — REVERTED (no gain)
`_BAND_WIDTH_BY_WARMS["slow"]` + `TRUST_MAX` (coupled so `at_peace`=3×width stays
reachable). Band constants don't affect delta emission, so the gate stayed green by
construction.
- **width 4 / TRUST_MAX 14** (`decc26a`): `at_peace`@12, `open`@8. STARVED — a cap-40
  playtest ran ~17 min with **0/12 sessions resolving**; sessions wandered to the cap.
  Too wide; removed the auto-resolution without leaving the player-driven endings
  runway. Dialed back.
- **width 3 / TRUST_MAX 10** (`cd29448`): `at_peace`@9, `open`@6. Partial corpus
  (engine-stall-degraded) showed a *mix* — a hard-warming diligent session still hit
  `tom_takes` at scene 3, others truncated to `None`. i.e. it traded the monopoly for
  "same ending when warm + dead-grind otherwise", **not** a persona fork.
- **Reverted to baseline** (`4b73228`, width 2 / TRUST_MAX 8). Rationale: band width
  only changes *when* `at_peace` is hit, not *whether approach changes the trust
  trajectory* — and per the diagnosis the trajectory is the problem. Engine + unit
  tests back to baseline (no diff vs `master`); `eval-deltas` green.

### Harness resilience fixes — KEPT (independently correct, not divergence knobs)
The playtest kept hanging the whole run (one stuck call blocks `run.py`'s single
`gather`, so `summary.json` never writes). Two distinct stalls, two fixes:
- **`204c92e` — Gemini player timeout.** `GeminiPlayer._run` retried on raised errors
  but had no timeout; a Gemini call that *hangs* (connection open, no body) blocked
  forever. Wrapped `agent.run` in `asyncio.wait_for(120s)` feeding the existing retry.
- **`82f752b` — per-session timeout.** The *engine* also stalls a streaming prose
  response under concurrent load (inferno stops sending tokens mid-stream;
  `RetryModel` retries only the stream *entry*; eval-deltas never streams, which is
  why the delta gate is always fine while the playtest hangs). Wrapped `play_session`
  in `asyncio.wait_for(900s)` → a hung session drops to a truncated artifact and the
  run completes.

These let runs *finish*, but the streaming stall is frequent enough under load that
runs now limp through mostly-`None` truncations — not a clean corpus. The stall is
almost certainly inferno's **streaming** path degraded (it intermittently stops
emitting tokens); one-shot completions (eval-deltas at concurrency 16) are always
fast and green. **A clean corpus A/B needs a fresh/healthy engine** (likely an inferno
restart — not done here, as it is shared infra I did not want to bounce unilaterally).
Confirmed **concurrency-independent**: a parallel-1 (sequential) probe also stalled
after ~8 turns, so it is not contention — the streaming endpoint itself is wedged this
session. (My own repeated force-kills of hung runs, which abandon mid-stream
connections, likely contributed; future me: end runs via the 900s timeout, never
`Stop-Process`, and if streaming is wedged, restart inferno before trusting a corpus.)

---

## Proposals (the actual divergence fix — gate-blind, so corpus-validate first)

These are pacing/authoring changes that target the diagnosed root causes. Each is
**corpus-only-validatable** (invisible to `eval-deltas`), so none was committed
blind — validate each with a `make playtest` A/B on a healthy engine, keep only if
`persona_outcome_separation` / `ending_entropy` rise and it survives a second run.

1. **Raise `GUARD_MAX` (e.g. 3 → 8), most promising.** Lets sustained pressure
   actually suppress effective warmth, so the aggressive interrogator's 13 guard-turns
   hold Tom cold instead of saturating at 3. Expected: aggressive/impatient personas
   stop reaching `tom_takes` and fork toward `None`/walk/`run_out`; patient personas
   still warm through. Pure pacing constant; delta gate unaffected. Watch for
   over-suppression of the diligent persona (it also takes some guard) — if so, try 5–6.

2. **Make `attention` reachable for aggressive-but-private play.** Today `run_out`
   never fires because attention only rises on public scenes the player never stages.
   Either (a) **authoring** (`world/truth.yaml`): add an attention path the aggressive
   player actually triggers, or (b) loosen the phase1 attention rule so repeated
   confrontational pressure (not just public spectacle) ratchets attention — *but*
   this bumps against the stated invariant "attention rises only on conspicuous public
   acts", so it needs the author's sign-off. Without a firing `run_out`, the aggressive
   persona has no signature ending and divergence is structurally capped.

3. **(Minor) tighten the neutral-question `trust+1` over-grant** — only if a robust
   wording can be found. Two attempts this session BROKE the delta gate at Gemma-4-E4B
   size: any trust rewording that pulled neutral→0 also nuked genuine warmth
   (`tom_warm_cold` 0.917→0.0/0.167) and destabilised BREACH on the public-accusation
   case (0.792→0.0). Left at baseline. Low priority — it inflates warming uniformly,
   it doesn't kill the fork.

---

## Reverts
- Slow-band widening (width 4, then 3) → reverted to baseline width 2 / TRUST_MAX 8
  (`4b73228`). No divergence gain; partial corpus showed monopoly→(monopoly+grind).
- Two phase1-prompt tightenings (neutral=0) → reverted (broke the delta gate).

## Commits on this branch
```
4b73228 Revert the slow-band widening (no divergence gain)
82f752b Add a per-session wall-clock safety net to the playtest harness
204c92e Add a wall-clock timeout to the Gemini player call (harness resilience)
cd29448 Dial the slow band back to width 3 (width 4 starved)
decc26a Widen the slow confession band to break the at_peace monopoly
```
(`decc26a`/`cd29448` are superseded by the revert `4b73228`; kept for history. The two
harness commits are the keepers; engine behaviour is back at baseline.)

## Best / worst playtest transcripts
_(extracted below — see `### Transcripts`)_


### Transcripts (condensed: prose shortened; each PICK with applied deltas/beats)

Read these as *worst cases* (the collapse texture). The closest thing to variety is the lone `buried_the_story`.

```text
== WORST 1: aggressive interrogator still collapses to the DEEPEST-trust ending in 3 scenes ==
#### aggressive_interrogator -> tom_takes_responsibility  (3 scenes, ended)
  [the_old_vicarage]
    * The keys feel heavy and cold in your palm, smelling faintly of dust and old wood polish. Mrs Hughes offers a curt nod, her face set in the neutral expression of someone accustomed to the...
      -> PICK: Pick up the letter from the hall table.
    * The paper is thick, the ink dark and deliberate. You unfold the pages, and Iwan’s measured words settle in the quiet of the hall. The restraint in his tone is almost painful, a careful...
      -> PICK: I snatch the envelope, tearing it open without a second’s hesitation—privacy is a luxury for people who... {letter_from_iwan_read}
    * You snatch the envelope, tearing it open without a second’s hesitation—privacy is a luxury for people who haven't spent their lives burying the truth. "Don't bother pretending you didn't...
      -> PICK: Examine the letter's contents more closely.
    * You smooth the two pages flat on the hall table, the weight of Iwan's carefully constructed prose a palpable thing in the small room. The facts are stated with a clinical detachment that...
      -> PICK: Reread the section detailing the police report discrepancy.
    * You trace the line regarding the police findings again, the mention of 'single-vehicle accident' feeling flimsy, too neat for the sheer volume of wreckage you know surrounds the story....
      -> PICK: Re-read the section describing the initial scene, looking for phrasing discrepancies.
    >> walk: the_old_vicarage -> the_post_office
    --- scene close (end_scene) ---
  [the_post_office]
    * The bell above the door jingles faintly as you step inside. The air smells of stale tea and slightly dusty bread. Maud, positioned behind the counter, doesn't look up immediately from the...
      -> PICK: Ask her directly what she means by 'funny times'.
    * Maud leans forward slightly, resting her elbows on the counter. She taps the edge of the postcard again, her expression softening into a look of shared, weary understanding, as if you...
      -> PICK: Ask her what kind of rumors are circulating. [maud:t+1/g0]
    * Maud picks up a small, slightly bent tin of tea bags, turning it over in her hands as if studying its manufacture. The politeness remains fixed on her face, but a tiny, almost...
      -> PICK: I slam my hand onto the counter, right next to her precious postcards, and lean in until I’m barely a foot... [maud:t+1/g0]
    * You slam your hand onto the counter, right next to her precious postcards, and lean in until you’re barely a foot from her face. "Cut the cryptic nonsense, Maud; tell me exactly what the...
      -> PICK: Press further, challenging her composure. [maud:t-1/g+1]
    * The silence that follows your challenge is heavy, thick with the unspoken weight of the village's collective agreement not to speak of the matter. Maud stares at you, and for the first...
      -> PICK: Ask directly about the rumors: "What have people been whispering about lately?" [maud:t-1/g+1]
    * Maud closes her eyes for the briefest moment, a small, almost imperceptible sigh escaping her lips. When she opens them again, the polite façade is firmly back in place, but it seems less...
      -> PICK: Press her on the 'things' people are saying. [maud:t-1/g+1]
    * She glances quickly towards the window, though the afternoon light inside the shop offers little view of the lane. Maud leans back against the counter, adopting a posture of complete,...
      -> PICK: "Save the meteorological report for the tourists, Maud; we both know the winter wasn't the only thing that... [maud:t-1/g+1]
    * The shift in subject is so abrupt that Maud visibly recoils, a sharp intake of breath catching in her throat. Your sudden, specific attack on the village's carefully maintained narrative...
      -> PICK: Ask her, calmly, what she means by 'you understand?' [maud:t-1/g+1]
    * Maud stares at you, her jaw tight. She studies your face, searching for a tremor, a sign of bluff, but finds only a steady, unnerving calm. She seems to be making up her mind in that...
      -> PICK: Ask what 'some things' refers to. [maud:t-1/g+1]
    * Maud does not touch the fete card again. She simply looks at it, then back at you, the gesture communicating that this conversation has reached its designated end. "That," she replies,...
      -> PICK: I push the advertisement back toward her with my fingertips, my gaze fixed on her eyes until she looks... [maud:t-1/g+1]
    >> walk: the_post_office -> the_slaughtered_lamb
    --- scene close (end_scene) ---
  [the_slaughtered_lamb]
    * Tom doesn't look up. He turns the glass a quarter-turn, another. "Free country," he says, eventually — not kindly, not the other way either. You take the stool. Up close, he smells of...
      -> PICK: Ask him what he means by 'free country'.
    * He takes a slow pull from his bitter, his gaze fixed on the amber liquid. The silence stretches, thick with the smell of old ale and coal smoke. "It means," he says, finally looking at...
      -> PICK: Ask what kind of things he's seen. [tom:t+1/g0]
    * Tom shifts slightly on the stool, a barely perceptible movement of heavy cloth and age. He picks at a loose thread on the bar top with a thumb, not meeting your eyes. "Aye. Not my...
      -> PICK: Press him gently about the picture on the dartboard. [tom:t+1/g0]
    * His hand stills on the bar. The quiet of the pub seems to absorb the small sound of your question. He lifts his head sharply, his guardedness snapping back into place like a closing door....
      -> PICK: "Staring at her photo won't scrub the guilt off your hands, Tom, so stop pretending it's 'not your... [tom:t+1/g-1]
    * His jaw tightens. The polite weariness in his face evaporates, replaced by a hard, sudden resistance. The air in the low-ceilinged pub feels suddenly brittle. He doesn't flinch from your...
      -> PICK: Ask him what he means by 'not from round here'. [tom:t-1/g+1]
    * Tom lets out a short, dry sound that might be a laugh or a cough. He leans his knuckles on the bar, looking out the grimy window at the slate-grey light washing over the village. "Folks...
      -> PICK: Tell him you don't want to tear anything apart; you just want answers. [tom:t-1/g+1]
    * He studies your face, a long, slow appraisal that seems to strip away the veneer of professionalism you might think you possess. He sees the demand beneath the plea for answers. "Sit down...
      -> PICK: Sit down on the bench. [tom:t+1/g-1]
    * You settle onto the hard wooden bench. The old leather of the seat creaks beneath you, a sound instantly swallowed by the pub's deep quiet. Tom watches you settle, his expression...
      -> PICK: Order a drink, signaling you're ready to talk. [tom:t+1/g-1]
    * Eira, still polishing glass at the far end of the bar, slides a fresh pint glass across toward you without looking up. You order. Tom, sensing the shift from posturing to potential...
      -> PICK: Ask if he saw who was driving it. [tom:t+1/g-1] {tom_named_a_vehicle}
    * He drinks half the pint in one go, a hard swallow. The act seems to be a physical purging of the memory. He sets the glass down with a soft clink that is loud in the room. "Aye....
      -> PICK: Press him on why the face is more important than the paintwork. [tom:t+1/g0]
    * Tom leans forward, elbows on his knees, a gesture of sudden, heavy commitment. He looks like a man who has finally made a decision too large for him to carry alone. "The paintwork tells...
      -> PICK: "Don't give me that," I snap, leaning over the table until I'm right in his line of sight. "We both know... [tom:t+1/g0] {tom_decides_to_speak_to_police}
    >> walk: the_slaughtered_lamb -> st_caradocs
    --- scene close (end_game) ---

== WORST 2: distracted skimmer (lowest-effort persona) -> same deepest-trust ending ==
#### distracted_skimmer -> tom_takes_responsibility  (3 scenes, ended)
  [the_old_vicarage]
    * The heavy oak door clicks shut behind you, sealing the smell of damp coats inside the hallway. Mrs Hughes gives a curt nod, her expression as grey as the slate roof above. She points a...
      -> PICK: Pick up the envelope from the hall table.
    * You pick up the envelope. It feels substantial, the paper heavy and expensive, unlike the damp chill clinging to the hallway air. Breaking the seal, you unfold the two pages. Iwan's hand...
      -> PICK: I rip it open right there, not bothering to find a seat. Probably just some boring notes, let's see what... {letter_from_iwan_read}
    * You rip it open right there, not bothering to find a seat. Probably just some boring notes, let's see what the fuss is about. You scan the first page, the careful script detailing the...
      -> PICK: Look for a personal note or photograph tucked inside the report.
    * You sift through the folded pages of the letter, turning them over, searching the margins and the creases for something extraneous—a discarded receipt, a dog-eared photograph, any small...
      -> PICK: Examine the desk the letter was found on more closely.
    * You move past the hall table to the study, drawn by the deep shadow cast by the bay window. The desk is solid, scarred wood, smelling faintly of old paper and dust. You run your fingers...
      -> PICK: Flip through the address book to see who is listed first.
    * You open the brittle book to the first entry. The first name listed, listed with an unusually formal address, is "Vaughn, Eluned." The accompanying note simply reads, "Tan-y-coed Farm."...
      -> PICK: Pocket the book and consider what Tan-y-coed Farm might be.
    * You tuck the address book securely into your bag, the leather cool against your hip. Tan-y-coed Farm—the Vaughn place. It sits high, sheltered by the wood, a place that seems to hold its...
      -> PICK: Examine the maps more closely.
    * You pull the drawer open and pull out the 2017 OS map. The paper is stiff, the pencil markings overlaying the contours of the Welsh Marches. The lines are not those of hiking trails or...
      -> PICK: Examine the markings around the star more closely.
    >> walk: the_old_vicarage -> the_post_office
    --- scene close (end_scene) ---
  [the_post_office]
    * The small bell above the door gives a faint, tinny jingle as you step inside. The air smells of stale bread and cheap cleaning spray. Maud is behind the counter, her fingers paused over...
      -> PICK: Simply state the reason for your visit.
    * You state plainly that you are here regarding the death of Megan Pritchard. Maud's smile doesn't waver, but the rhythm of her breathing seems to catch for the briefest moment. She picks...
      -> PICK: Press her about what people might be saying.
    * You press her, asking what the village is actually saying about the accident. Maud shifts slightly, a flicker of something—weariness, perhaps, or genuine concern for the village's...
      -> PICK: Thank her and ask if she's seen anyone unusual recently. [maud:t-1/g+1]
    * You accept her deflection with a nod, then pivot, asking if she has noticed anything out of the ordinary in the village lately. Maud pauses, her attention momentarily drawn to the faded...
      -> PICK: Press her gently about the 'usual business of the world.' [maud:t+1/g0]
    * You keep the pressure light, probing at the phrase "usual business of the world," trying to find the seam where village chatter gives way to something more solid. Maud watches you, her...
      -> PICK: Press slightly on the 'noise' comment; what does she mean by that? [maud:t-1/g+1]
    * "Noise," you repeat, letting the word hang in the small space, trying to draw her out from the generalities of village life. Maud’s eyes narrow almost imperceptibly; this is moving past...
      -> PICK: Press gently, asking what that 'stuff' might be. [maud:t0/g+1]
    * You press on, using the vague term of "stuff" as a hook, trying to make the general dread she hints at specific. Maud stares at you for a long moment, as if weighing the cost of the next...
      -> PICK: Acknowledge her feeling: "So it's something you can't name." [maud:t+1/g0]
    >> walk: the_post_office -> the_slaughtered_lamb
    --- scene close (end_scene) ---
  [the_slaughtered_lamb]
    * Tom doesn't look up from the amber liquid in his glass. The peat smoke hangs heavy in the low ceiling, smelling of old wool and damp earth. He nurses his Brains Bitter, his gaze fixed...
      -> PICK: Ask him about the picture of Megan Pritchard.
    * Tom takes a slow swallow of his bitter, the ice clinking softly against the glass. He follows the line of your gaze, a flicker of something—recognition, maybe pain—crossing his face...
      -> PICK: I gesture toward the picture on the wall and ask him, "What's the real story behind that one? Everyone in... [tom:t+1/g0]
    * Tom watches your gesture, his expression hardening almost imperceptibly. He shifts on the stool, a tiny, reluctant movement that speaks of years spent bracing for impact. "Aye. Not my...
      -> PICK: Press gently, asking what he *does* know. [tom:t+1/g0]
    * He sighs, a short, ragged sound that seems to drain the air from the small corner of the pub. He looks out the grimy front window, his eyes distant, fixed on the muddy stretch of the lane...
      -> PICK: Remain silent and wait for him to elaborate. [tom:t-1/g+1]
    * Tom’s eyes do not track the road outside, but seem to be looking through the windowpane itself. He holds the silence, allowing it to stretch taut between the clink of your own breathing...
      -> PICK: I just keep staring at him, waiting for him to finish the thought and name the driver. [tom:t+1/g-1]
    * The moment stretches until the background sounds of the village—a distant dog bark, the hiss of the fire—seem to fade into an absolute hush. Tom’s jaw tightens, the muscle jumping once...
      -> PICK: Simply nod and begin to gather your things. [tom:t+1/g-1]
    * You nod once, acknowledging the abruptness of his dismissal. He watches you gather your notebook, his face unreadable as you rise from the stool. The confession hangs in the air, heavy...
      -> PICK: Gather your things and exit the pub. [tom:t+1/g-1] {tom_named_a_vehicle}
    * You pull your coat on, the weight of the information you carry now feeling heavier than any gear you brought with you. The low murmur of the pub, which seemed so intense moments ago,...
      -> PICK: Keep walking, focusing on leaving. [tom:t+1/g-1]
    * You push through the heavy door. The cool, damp air of the Welsh afternoon hits you instantly, a stark contrast to the peat-smoke warmth inside. The quiet of the village road is...
      -> PICK: I just keep my head down and push through the door, letting the cool night air hit me before I start...
    * A moment later, as you reach the threshold, Tom appears again in the doorway, having left his seat. He is standing just outside the immediate sphere of the bar’s shadows, looking out at...
      -> PICK: Acknowledge his departure plans and ask him to elaborate on his concerns. [tom:t+1/g-1] {tom_decides_to_speak_to_police}
    * Tom hesitates in the doorway, his gaze flicking between you and the darkening road ahead. He seems to be fighting a battle against the old habit of retreating back into silence. He runs a...
      -> PICK: Ask what 'the real picture' entails. [tom:t+1/g0]
    * He shakes his head slowly, as if the phrase itself were too heavy to hold in his mouth. He looks away from you, out toward the stretch of road that leads towards the bends. "It's the...
      -> PICK: I stop and look him square in the eye, asking, "So, what’s the 'real picture' then—who were you actually... [tom:t+1/g0]
    * The directness of your challenge causes him to flinch, a sudden, sharp intake of breath that breaks his composure entirely. He stares at you for a long moment, the guardedness in his eyes...
      -> PICK: Ask him directly if he knew what happened to her [tom:t+1/g0]
    * Tom recoils slightly as if you have struck him. The quiet authority in his demeanor cracks, replaced by a sudden, vulnerable look of deep regret. He grips the edge of the bar, his...
      -> PICK: Ask if he knew anything about her recent troubles. [tom:t-1/g+1]
    >> walk: the_slaughtered_lamb -> st_caradocs
    --- scene close (end_game) ---

== NEAREST-TO-GOOD: the lone variant -- diligent journalist walks away (buried_the_story) ==
#### diligent_journalist -> buried_the_story  (8 scenes, ended)
  [the_old_vicarage]
    * The damp air of the hall settles around you, heavy with the scent of old stone and wet wool. Mrs Hughes offers a brief nod before retreating towards the kitchen, the sound of her scraping...
      -> PICK: Turn the letter over to look for a return address.
    * You turn the folded page, studying the neatness of the stationery. There is no return address printed, only the hand-written details of a local village and a private number scrawled...
      -> PICK: Examine the desk surface more closely for other items.
    * You move past the hall table towards the study door. The desk in the bay window is a large slab of dark, solid wood. Dust motes dance in the weak light struggling through the windowpanes....
      -> PICK: Look around the rest of the study for anything out of place.
    * Your gaze sweeps across the study. A tall, narrow bookshelf lines the wall, filled with volumes of theology and local history, all neatly aligned. A small side table holds a heavy brass...
      -> PICK: Inspect the brass clock on the side table.
    * You reach out and touch the cool metal of the brass clock. The hands are frozen stiff, pointing precisely to three fifteen. It feels heavy, like a relic that has surrendered its purpose...
      -> PICK: Check the area immediately around the clock for clues.
    >> walk: the_old_vicarage -> the_post_office
    --- scene close (end_scene) ---
  [the_post_office]
    * The small bell over the door gives a tired, bright jingle as you step into the shop. The scent of stale paper and baked goods settles around you. Behind the counter, Maud Pryce nods from...
      -> PICK: Ask if she's seen anything unusual lately.
    * Maud pauses her dusting of the counter, a faint, polite frown touching the corners of her mouth. She looks past you, as if searching for the proper, safe answer on the horizon. "Oh,...
      -> PICK: "I'm looking into the circumstances surrounding the death at The Old Vicarage," I say, keeping my voice low... [maud:t+1/g0]
    * Maud stops polishing the glass display. Her polite mask doesn't crack, but the warmth in her expression cools, becoming something carefully neutral. She picks up a magazine, flipping...
      -> PICK: "It was a very interesting point," you press gently.
    * Maud lowers the magazine slowly. The movement is measured, practiced. She meets your gaze, and for a moment, the village gossip-machine seems to stall, replaced by something harder....
      -> PICK: Ask her directly what she knows about the incident. [maud:t-1/g+1]
    * Maud sighs again, a little louder this time, and pushes herself slightly away from the counter. She begins arranging a stack of tins as if the movement itself is the only acceptable...
      -> PICK: Press her gently about what she might have seen. [maud:t+1/g0]
    * Maud straightens up, her posture becoming rigidly upright. She meets your gaze again, and there is a flicker of something—not fear, but deep, ingrained resistance. "Now you didn't hear...
      -> PICK: "Maud, I understand that people in Pant Goch are careful about what they say to outsiders, but this isn't... [maud:t-1/g+1]
    * Maud looks at your face, assessing the steadiness of your voice, the lack of aggression. She seems to weigh the risk of a slight deviation against the comfort of the village narrative....
      -> PICK: Press gently about who might have been saying those things. [maud:t+1/g0]
    * She shakes her head, a small, weary motion. The deflection is automatic, a shield long honed against unwelcome scrutiny. "Oh, well. You'll have to ask someone else about that, my love,"...
      -> PICK: Press gently: "Who might that be?"
    * Maud lets out a short, brittle sound that is almost a laugh. She finally looks at you again, but there is no humour in her eyes, only a profound sense of weary obligation. "Well. Since...
      -> PICK: Ask what 'the other places' are. [maud:t0/g+1]
    * She glances pointedly towards the door, then back at the noticeboard, lingering for a beat on the faded "missing dog" poster before answering. "Well. Since you've come this far," she...
      -> PICK: "I'm not here to stir up trouble, Maud, just to find out what really happened," I say, leaning in slightly.... [maud:t+1/g0]
    * Maud studies you for a long moment, her expression unreadable. It is as if she is calculating the exact amount of truth she can afford to release without breaking her own delicate...
      -> PICK: Ask her who these former neighbors might be. [maud:t+1/g0]
    * She taps the edge of the counter twice, a small, decisive rhythm. The facade of polite helpfulness is back, but it feels brittle now. "Well. Since you've come this far," she says, looking...
      -> PICK: Press her gently about Eluned Vaughn.
    * Maud’s composure wavers slightly at the mention of the Vaughn family. Her gaze drifts up the road, a fleeting expression of caution crossing her face before she forces it back down to the...
      -> PICK: "Eluned Vaughn, you say? Why would she be so particularly guarded about the events at the Vicarage compared...
    * Maud fixes you with a long, unwavering stare. The atmosphere in the small shop seems to compress around you both. She doesn't answer the question about *why*. Instead, she leans in just...
      -> PICK: Acknowledge her boundary and change the topic. [maud:t0/g+1]
    * Maud visibly relaxes, the tension leaving her shoulders in a slow, subtle slump. She picks up the stamp block, tapping it lightly against the counter as if dismissing the last few...
      -> PICK: Simply say you'll leave her to her duties for now. [maud:t0/g-1]
    >> walk: the_post_office -> the_slaughtered_lamb
    --- scene close (end_scene) ---
  [the_slaughtered_lamb]
    * Tom doesn't look up. He turns the glass a quarter-turn, another. "Free country," he says, eventually — not kindly, not the other way either. You take the stool. Up close, he smells of...
      -> PICK: Wait for him to elaborate without speaking.
    * He takes a slow, deep swallow of his bitter, his gaze fixed on the murky surface of the dregs. Eira, behind the bar, wipes a section of wood with methodical, silent strokes. After a...
      -> PICK: "A free country, certainly," I say, keeping my voice steady as I glance toward the door, "but there are... [tom:t+1/g-1]
    * Tom lets out a short, dry sound that might be a laugh, but holds no humour. His shoulders tighten beneath his jumper. "Aye. Not my business, was it." He nods once, a small, decisive...
      -> PICK: "I'm not asking about your business, Tom. I'm asking about what happened here." [tom:t+1/g0]
    * He picks up his glass, turning it slowly between his calloused fingers as if assessing its worth. The low murmur of the pub feels suddenly pressurized, the silence thick and immediate....
      -> PICK: Apologize and excuse yourself from the table. [tom:t0/g+1]
    >> walk: the_slaughtered_lamb -> st_caradocs
    --- scene close (end_scene) ---
  [st_caradocs]
    * The chill air of the churchyard smells of damp earth and old stone. You walk the perimeter, your boots crunching on the gravel that frames the ancient yew. Near the east wall, Megan’s...
      -> PICK: Examine the cluster of fresh flowers more closely.
    * You kneel beside the headstone, ignoring the cold seeping through your trousers. The flowers are not the cheap plastic kind; they are real, vibrant, and placed with deliberate care. You...
      -> PICK: Look around the immediate vicinity of the grave.
    * You sweep your gaze across the ground immediately surrounding the grave. There is no sign of scuffing, no displaced earth that suggests a struggle or a frantic arrival. The neatness of...
      -> PICK: I lean in closer, spotting a short, frayed length of coarse twine caught in the mud near the base of the...
    * You lean in closer, spotting a short, frayed length of coarse twine caught in the mud near the base of the headstone where the heather had been bound. Carefully, you pick it up, examining...
      -> PICK: Examine the area immediately surrounding the headstone for other signs of disturbance.
    * You scan the area in a wider circle, moving your attention away from the tiny piece of twine. The ground is generally well-kept, the grass cut, the stones set firmly in place. There are...
      -> PICK: Take a moment to observe the headstones in the nearby plots.
    >> walk: st_caradocs -> the_lane
    --- scene close (end_scene) ---
  [the_lane]
    * The tarmac underfoot feels uneven, the old patches of resin peeling away like sunburned skin. The hedgerows press close on either side, a tangle of thorny branches and deep green leaves...
      -> PICK: Try to discern the direction the lane leads.
    * You walk the length of the road, your boots crunching faintly on loose grit near the edges. North climbs steadily toward the higher ground where Tan-y-coed rests beneath the canopy of the...
      -> PICK: Head North toward the woods and Tan-y-coed.
    * You pause near the rise, looking through the gap in the hedge. It offers a sliver of view up the hill, a quarter-mile climb to the Vaughn farm. The outline of the buildings is muted...
      -> PICK: Follow the lane up toward the farm.
    * The path continues its slow ascent. The lane narrows subtly as the sheep ground of Tan-y-coed begins to rise sharply from the roadside. The air feels cooler here, carrying a faint, earthy...
      -> PICK: Follow the rising path into the higher ground.
    * The slope becomes more pronounced. You can now make out the farmyard clearly, tucked beneath the deep shelter of the woods. The buildings look hunkered down, as if bracing themselves...
      -> PICK: Stop at the edge of the woods and observe the farmstead from above.
    >> walk: the_lane -> tan_y_coed
    --- scene close (end_scene) ---
  [tan_y_coed]
    * Cap and Mab rise, a low chorus of barking erupting across the yard, sharp and immediate. Eluned stands by the weathered stone of the farmhouse, watching you from a distance, her arms...
      -> PICK: Respond calmly to her ultimatum.
    * You do not rush, not shifting your posture or voice to meet her demanding energy. You allow the silence to settle around the barking dogs and the tinny voice of the radio. You meet her...
      -> PICK: Acknowledge the warning about the tea with a small nod. [eluned:t+1/g-1]
    * You give a small nod, acknowledging the boundary she has drawn around the warmth of the kitchen. The collie, Cap, shifts his weight, his tail giving a single, heavy thump against the...
      -> PICK: Ask Eluned about the incident that brought you here. [eluned:t+1/g-1]
    * Her jaw tightens, and she stares past you, focusing on the distant line of the wood that crowns the ridge. The radio voice continues, a stream of unrelated chatter about the weather. She...
      -> PICK: "I appreciate you letting me speak with you, Eluned, as I know this has been a difficult time for everyone... [eluned:t+1/g0]
    * She watches the words you speak, her eyes scanning your hands, your feet, assessing if you are a threat or merely a persistent nuisance. The accusation—the reference to the Vicarage, the...
      -> PICK: Acknowledge her warning about leaving and state your purpose concisely. [eluned:t+1/g-1]
    * "I am looking for understanding," you state, keeping the focus solely on the immediate question, not the village's secrets. You wait, allowing the quiet hum of the farm to fill the gap...
      -> PICK: Accept her terms and back away toward the yard. [eluned:t+1/g-1]
    * You take a small step back, moving out of the immediate vicinity of the farmhouse door and back toward the rougher ground of the yard, respecting the invisible line she has drawn. Eluned...
      -> PICK: Ask her about the Land Rover. [eluned:t+1/g-1]
    * Eluned glances at the vehicle, the same Land Rover Tom Bevan mentioned. For a beat, her face goes utterly still, as if the machine holds the weight of a heavy secret. It is a gesture that...
      -> PICK: "You seemed to look at that Land Rover as if it held the answer to what you aren't saying, Eluned. Is that... [eluned:t+1/g0]
    * She stiffens, her control wavering under the directness of your gaze. The air seems to cool around you both. She doesn't deny the intensity of her look, but she refuses to engage the...
      -> PICK: Accept her dismissal and step back, for now. [eluned:t+1/g0]
    >> walk: tan_y_coed -> lower_hafod
    --- scene close (end_scene) ---
  [lower_hafod]
    * Iwan opens the door before you've knocked. He stands aside, pointing at the hook where you hung your coat last time. "Tea or coffee? Tea." He has answered for you. He always does. He...
      -> PICK: Accept the tea gratefully.
    * You accept the mug. The tea is strong, tasting of peat and boiled earth. The silence in the small kitchen is not empty; it feels packed down, heavy with things unsaid and memories too...
      -> PICK: Ask Iwan a gentle, open-ended question about his day. [iwan:t+1/g0]
    * Iwan looks down into his own cup, swirling the dark surface as if searching for answers there. He does not meet your eyes. "It's the not knowing. That's the worst of it." The words are...
      -> PICK: Ask Iwan what specifically he feels he doesn't know. [iwan:t+1/g0]
    * Iwan stills completely, the movement freezing in his posture. He picks up his mug again, his knuckles white against the porcelain. The quiet stretches, taut as wire. "I want it written....
      -> PICK: "Iwan, when you say 'the not knowing' is the worst of it, are you referring to the official findings of the... [iwan:t+1/g0]
    * He searches your face for a flicker of insincerity, a sign that you are merely fishing for a headline. When he finds none, a flicker of something—exhaustion, perhaps—crosses his features....
      -> PICK: Inquire if he's seen anything specific that led to this atmosphere. [iwan:t+1/g-1]
    * Iwan shakes his head slowly, the movement heavy, deliberate. He does not want to give you anything concrete. Owain leans forward then, his earlier quietness broken by a sudden urgency,...
      -> PICK: Press him on the 'things' that aren't what they seem. [iwan:t+1/g0]
    * Iwan looks sharply at Owain, a clear warning in the slight tension around his mouth, before turning back to you. He seems to wrestle with the boundary between what is verifiable and what...
      -> PICK: Press him on what 'changed' in the valley. [iwan:t+1/g0]
    * Owain snorts, a short, sharp burst of air that manages to sound both dismissive and deeply serious. "Things settle differently, when the silence comes in like that." He waves a hand...
      -> PICK: "Iwan, you said that night changed more than I could imagine—is that change reflected in the way people... [iwan:t+1/g0;owain:t+1/g0]
    * Iwan recoils slightly at the specific mention of the Vicarage, a muscle ticking in his jaw. The polite veneer of hospitality cracks under the pressure of precise detail. "Leave the old...
      -> PICK: Acknowledge his defensiveness and back off for now. [iwan:t+1/g0;owain:t+1/g0]
    * You let the question hang there, letting the weight of the silence absorb the tension. Iwan visibly relaxes, the rigid set of his shoulders easing just enough to return to a semblance of...
      -> PICK: Accept the offer of tea. [iwan:t+1/g-1]
    * You take the offered cup, the ceramic warm in your hands. The act of drinking seems to recalibrate the room; the conversation, for this brief moment, has paused its sharp edge. Owain nods...
      -> PICK: Press Owain on what he means by "keeps things locked up tight". [iwan:t+1/g-1]
    * Owain blinks slowly, seeming surprised by the shift of focus from Iwan to himself. He picks up his own cup, turning it in his hands again. "Ah, Iwan's just his way," he says, a lighter...
      -> PICK: "Owain, you say he keeps things locked up tight—is that just his nature, or is he keeping something... [owain:t+1/g0]
    >> walk: lower_hafod -> tan_y_coed
    --- scene close (end_scene) ---
  [tan_y_coed]
    * The dogs bark, sharp and sudden, as you reach the edge of the yard. They are agitated, circling the fence line, their attention locked on you. Eluned watches from the doorway, perfectly...
      -> PICK: Scan the yard quickly, looking for anything unusual.
    * You sweep your gaze across the yard. The Land Rover sits like a hunk of dark metal near the main barn, looking too purposeful for a casual visitor. The ground is churned mud where the...
      -> PICK: "Eluned, I've been speaking with people in the village, and there’s a consistent feeling that the truth...
    * Eluned’s gaze sharpens, the slight hardness around her mouth deepening. She does not flinch at your mention of the Vicarage, but her hand moves almost imperceptibly toward the doorframe,...
      -> PICK: Address the 'shadow' she reacted to. [eluned:t+1/g0]
    * She looks past you, toward the high ridge where the wood crowns the hill. A small, involuntary tightening crosses her jaw before she smooths it away. The silence stretches, filled only by...
      -> PICK: Ask her directly what she means by the land bearing witness. [eluned:t+1/g0]
    * Eluned lets out a slow, controlled breath, as if the effort of maintaining her composure is exhausting. She studies you now, her eyes measuring you, not assessing a story, but judging a...
      -> PICK: "If the land bears witness, then it’s already speaking—I'm just trying to learn how to hear it." I keep my... [eluned:t+1/g0]
    * She stares at you for a long moment, her composure finally showing a hairline crack—a flicker of deep, almost painful exhaustion beneath the surface. She doesn't respond to the question...
      -> PICK: Pour her a drink, ignoring the unspoken tension. [eluned:t+1/g-1]
    * Eluned watches the movement of your hands with intense focus, a slight softening around her eyes as the glass nears the rough wooden table. She takes the drink without a word, her fingers...
      -> PICK: Comment on the stillness of the farm. [eluned:t+1/g-1]
    * The silence does not break; it deepens. It is the kind of quiet that has absorbed too much noise over many years. The only sound now is the low, steady murmur of Welsh from the radio, a...
      -> PICK: Wait longer, letting the silence settle. [eluned:t+1/g-1]
    * The waiting feels suspended. The light, already thinning, seems to be caught and held in the dusty panes of the kitchen window. You observe the photograph on the counter—Dylan and Megan,...
      -> PICK: Ask her about the photograph. [eluned:t+1/g-1]
    * Her hand stills over the glass. She finally glances toward the counter where the faded photograph lies. Her jaw tightens almost imperceptibly. She does not look at the picture, but at...
      -> PICK: Acknowledge her request to leave the area. [eluned:t+1/g-1]
    >> walk: tan_y_coed -> bryn_iago_wood
    --- scene close (end_game) ---

== width-3 FAILURE MODE: session grinds and truncates with no ending (None) ==
#### diligent_journalist -> None  (4 scenes, truncated)
  [the_old_vicarage]
    * The rain begins to tap a steady rhythm against the slate roof. The air inside the hallway is cold, smelling of old stone and damp wool. Mrs Hughes, already retreating towards the kitchen,...
      -> PICK: Examine the stationery and handwriting of the envelope more closely.
    * You lean over the letter, studying the paper and the script. It is good quality, heavy cream stock, not the thin stuff you'd expect from a local shop. The penmanship is careful,...
      -> PICK: Read the letter again, searching for any misplaced words.
    * You reread the carefully composed pages, looking for the tremor of doubt, the misplaced adjective, anything that would betray a hasty pen or a slip of the tongue. The prose remains...
      -> PICK: The letter speaks of a "final account" left at the old cairn, phrased with a cold, unsettling finality that...
    * You smooth the crisp paper, the cold weight of the envelope settling against your coat. The idea of a 'final account' leaves a cold knot in your stomach, something more deliberate than...
      -> PICK: Scan the dining table for anything left out.
    * The dining table is set for the evening, but not for guests. It holds a sturdy, dark wooden place setting—a ceramic jug, a small loaf of bread, a tin of tea. There is nothing else left...
      -> PICK: Look around the rest of the dining room.
    >> walk: the_old_vicarage -> the_post_office
    --- scene close (end_scene) ---
  [the_post_office]
    * The small brass bell above the door gives a thin, sharp chime as you step inside. The air smells of stale paper, damp wool, and the faint sweetness of baked goods. Maud is behind the...
      -> PICK: Inquire about the local gossip, gently.
    * Maud sets the stamp down, smoothing the edge of the postcard with her thumb. She regards you over the top of the counter, her smile perfectly level. "Oh, well. You'll have to ask someone...
      -> PICK: Acknowledge her dismissal: "I understand, but I'm trying to piece something together." [maud:t+1/g0]
    * Maud lets out a small, breathy sound, which might be a chuckle or just the sigh of a long day. She doesn't meet your eye again, instead focusing on straightening a precarious stack of...
      -> PICK: I lean in slightly, lowering my voice so the sound of the rain against the glass masks our exchange, and... [maud:t-1/g0]
    * You lean in slightly, lowering your voice so the sound of the rain against the glass masks your exchange. You ask, "The official reports called his death a simple tragedy, but 'funny...
      -> PICK: Press gently, 'I'm not looking for gossip, Maud. Just the truth.' [maud:t+1/g0]
    * Maud finally looks up, and the politeness has curdled into something brittle. She leans forward just enough to match your lowered tone, her gaze sharp, measuring you. "The truth," she...
      -> PICK: Press her gently on what she means by 'the quiet part of life'. [maud:t-1/g+1]
    * Maud lets out a soft, tired breath, as if the effort of maintaining her composure is physically exhausting. She pushes a stray strand of grey hair behind her ear, a gesture of ingrained...
      -> PICK: Ask if the church is involved in the 'keeping quiet'. [maud:t-1/g+1]
    * A flicker of genuine alarm crosses Maud’s face, so swift it is nearly missed. She recovers instantly, placing both hands flat on the counter as if anchoring herself. "Father Gareth," she...
      -> PICK: Press her on what she means by 'his own lines.' [maud:t+1/g0]
    * Maud shifts her weight, the small movement seeming to cost her significant energy. She avoids your eyes, instead looking at the glass case holding the postmarks. "Lines," she murmurs,...
      -> PICK: Ask what she means by 'held'. [maud:t-1/g+1]
    * Maud stares at you for a long moment, her mouth slightly open as if she is trying to recall a word that has slipped away. The warmth is gone entirely, replaced by a deep, cautious...
      -> PICK: I keep my voice steady, meeting her gaze despite the chill in the air, and ask, "When you say things are... [maud:t+1/g0]
    * You keep your voice steady, meeting her gaze despite the chill in the air, and ask, "When you say things are 'held' rather than simply hidden, are you saying the village is protecting a...
      -> PICK: Simply nod and wait for her to elaborate on the newsletter. [maud:t+1/g0]
    * You simply nod, allowing the silence to stretch taut between you and the counter. Maud remains focused on the newsletter, her thumb brushing lightly over the printed image of Megan...
      -> PICK: Press her gently about what reminders she means. [maud:t+1/g-1]
    * Maud stops scrubbing the counter. She lets the disinfectant bottle rest against the wood grain, looking exhausted, as if the effort of denying the deeper meaning has taken a physical...
      -> PICK: Ask directly if she's hiding something deeper than public knowledge. [maud:t+1/g0]
    * She meets your gaze again, and this time there is no attempt to mask the vulnerability in her expression. The air in the shop feels suddenly thick, stagnant. "I don't hide things," Maud...
      -> PICK: I set my notepad on the counter, making the deliberate choice to stop playing along with the platitudes,...
    * You set your notepad on the counter, making the deliberate choice to stop playing along with the platitudes, and look her straight in the eye: "Maud, we both know the official version of...
      -> PICK: Acknowledge her warning, asking what 'trouble' she means specifically. [maud:t+1/g-1]
    * She leans in again, not with warmth this time, but with a desperate kind of urgency. Her eyes dart towards the door, as if worried the outside world might hear your exchange. "Trouble,"...
      -> PICK: Ask her who is holding the edges of this trouble. [maud:t-1/g+1]
    * Maud sucks in a quick breath, her eyes wide, as if naming a name would be an unforgivable transgression. She glances pointedly towards the shelf where the out-of-date magazines rest, a...
      -> PICK: "If these 'quiet agreements' are the only thing keeping the village from unraveling, then tell me who is...
    * You keep your voice low and steady, making it clear that you’m not leaving until you have a name. "If these 'quiet agreements' are the only thing keeping the village from unraveling, then...
      -> PICK: Lower your tone again, shifting from demand to quiet empathy. "I understand this is dangerous." [maud:t0/g+1]
    * You lower your tone again, shifting from demand to quiet empathy. "I understand this is dangerous." Maud watches the rain streak down the windowpanes, her rigid posture softening almost...
      -> PICK: Press her gently about who is putting pressure on her. [maud:t+1/g-1]
    * She hesitates, chewing on the inside of her cheek for a long moment, the silence punctuated only by the muffled sound of the rain. She looks around the shop, at the stacked tins, at the...
      -> PICK: "I know you aren't doing this of your own free will, Maud, so tell me—who is the one applying the pressure... [maud:t+1/g-1]
    * You keep your expression neutral and your body language open, hoping to provide her a safe space to finally name the source of her fear. "I know you aren't doing this of your own free...
      -> PICK: Press her gently: "But who decides what the shape of things should be, Maud?" [maud:t+1/g-1]
    * She shakes her head slowly, a gesture of profound weariness. The gentle warmth you had managed to coax from her seems entirely extinguished now, leaving behind only a shell of ingrained...
      -> PICK: Press her gently on who benefits from this predictability. [maud:t+1/g0]
    * Maud stops fiddling with the stamp. She looks up, and for a moment, you see a flicker of genuine panic cross her face, the look of someone who has accidentally let a dangerous truth slip...
      -> PICK: "If the established families thrive on this predictability, they must have a vested interest in the story... [maud:t+1/g0]
    * You keep your tone measured. "If the established families thrive on this predictability, they must have a vested interest in the story that was told after the accident," you say. "Name...
      -> PICK: Shift focus: "What do you think they're protecting the story from?" [maud:t+1/g0]
    * She looks up again, this time with a strange mixture of fear and grim understanding. She seems to have reached a limit to what she is willing to admit, yet the directness of your question...
      -> PICK: Press her on what she means by 'spill'.
    * Maud looks suddenly pale, as if the idea of the story truly spilling is something she has only ever read about, not experienced. She backs up an inch from the counter, putting a small,...
      -> PICK: "If the truth is already so toxic that it could bring down everything it touches, then the 'spill' has been... [maud:t+1/g0]
    * You fix her with a steady, searching look. "If the truth is already so toxic that it could bring down everything it touches, then the 'spill' has been happening slowly for seven years."...
      -> PICK: Ask if there's anyone else involved who might know more. [maud:t+1/g0]
    >> walk: the_post_office -> the_slaughtered_lamb
    --- scene close (end_scene) ---
  [the_slaughtered_lamb]
    * Tom doesn't look up. He turns the glass a quarter-turn, another. "Free country," he says, eventually — not kindly, not the other way either. You take the stool. Up close, he smells of...
      -> PICK: Ask him about the business he's been dealing with lately.
    * He takes a slow pull of his bitter, his gaze fixed on the peat fire. He shifts slightly on the stool, a gesture of mild discomfort. "The business," he repeats, the words tasting flat. He...
      -> PICK: Shift the focus back to the business he mentioned. [tom:t+1/g0]
    * A heavy silence settles between the clinking of the fire and the occasional distant drone of traffic. Tom stares into his glass as if the answer is written in the foam. "Aye. Not my...
      -> PICK: "I'm not here because of where I'm from, I'm here because of the 'business' that clearly has this entire... [tom:t+1/g0]
    * "I'm not here because of where I'm from, I'm here because of the 'business' that clearly has this entire village walking on eggshells," you say, keeping your voice low and level as you...
      -> PICK: Press him on the photograph, asking what it is. [tom:t+1/g-1]
    * He follows your gaze down to the small, framed image. He watches you watch it. "Megan. Aye. Lovely girl. Quiet, mind," he says, the name soft, almost a whisper against the pub's low...
      -> PICK: Ask if he knew her well. [tom:t+1/g0]
    * He nods once, a quick, almost involuntary movement. "Aye. Knew her. Well." He swirls the last of the foam in his glass, his movements stiff. "She was... she was a good girl. Quiet, mind....
      -> PICK: Inquire about his relationship with her or her family. [tom:t+1/g0]
    * Tom stops swirling the bitter. The silence stretches until it seems to press against the low ceiling. He takes a deliberate, steadying breath before answering, his voice dropping lower...
      -> PICK: Ask who he thinks it belongs to. [tom:t+1/g0] {tom_named_a_vehicle}
    * He shifts again, a visible tremor running through his shoulders. He looks down at his hands, flexing his fingers as if trying to shake off a memory. "It's the Vaughn, that one," he says,...
      -> PICK: Accept his dismissal and thank him for his time. [tom:t+1/g0]
    >> walk: the_slaughtered_lamb -> st_caradocs
    --- scene close (end_scene) ---
  [st_caradocs]
    * The stone of the churchyard is cold beneath your fingertips, even through your coat. The yew, massive and deeply fissured, looms over the graves, its shade a heavy, green stillness....
      -> PICK: Examine Megan's headstone more closely.
    * You lean in close to the small granite slab. The inscription is clean, etched by someone with care, not the rushed hand of a local gravedigger. "Megan Mair Pritchard, 17 May 1999 — 14...
      -> PICK: Scan the surrounding headstones for more details.
    * You move your gaze across the weathered slate and older, lichen-spotted stone surrounding Megan's plot. Names from generations past—Pritchard, Vaughn, Bevan—are etched into the hillside....
      -> PICK: Look for any signs of recent visitation or disturbance.
    * You scan the ground near the east wall. The grass is neatly cut, maintained with a diligence that suggests routine care, not just the occasional drop-by. You note the fresh flowers by...
      -> PICK: Examine the headstones around Megan's grave more closely.
    * You focus your attention on the stones immediately adjacent to Megan's plot. One belonged to an older Pritchard, the carving softened by time and weather. Another, a Vaughn, bears a date...
      -> PICK: Examine the inscription (or lack thereof) on the slate closer.
    * You crouch down slightly to get a proper look at the small, dark slate. It is smooth, almost polished, and the absence of lettering is absolute. It is not merely un-engraved; it feels...
      -> PICK: I kneel in the damp grass, running a gloved finger along the surface of the slate to check for faint, worn-...
    * You kneel in the damp grass, running a gloved finger along the surface of the slate to check for faint, worn-away grooves or hidden marks that might have been intentional, even if they...
      -> PICK: Examine the surrounding soil for signs of disturbance.
    >> walk: st_caradocs -> the_lane
    --- scene close (end_scene) ---
  [the_lane]
    * The lane stretches before you, a ribbon of cracked tarmac hemmed in by thorny hedges. The silence here is heavy, broken only by the faint, uneven rustle of wind moving through the dense...
      -> PICK: Scan the hedgerow for any sign of disturbance or tracks.
    * You press close to the hedge, ignoring the prickle of thorns on your jacket sleeve. The earth here is damp and uneven. You scan the base of the hazel and blackthorn for scuffs in the mud...
      -> PICK: Kneel down to examine the mud more closely.
    * You drop down onto the damp earth. The mud is rich and dark, churned faintly by the passage of farm vehicles over the years. You run your fingertips lightly over the surface, searching...
      -> PICK: Systematically scan the area for any signs of disturbance.
    * You move along the hedge line, your gaze sweeping from the ground level up to the lower branches. The tarmac is the most recent disruption, but beneath it, the natural growth seems...
      -> PICK: Move further down the lane in the direction you were heading.

```
