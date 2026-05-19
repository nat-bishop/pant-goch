# HANDLER V2 — Pant Goch

A grounded LLM-driven investigation game in a Welsh Marches village. The player is a freelance journalist who arrives in Pant Goch to investigate a death the local police closed too quickly. The world is hand-authored — locations, cast, the truth — but the scenes themselves are LLM-improvised within that world. The map is the primary navigation: the player walks the village by clicking locations, and the LLM runs whatever happens when they arrive.

This document is the v1 spec — what gets built and what gets authored.

---

## 0. Why v2 exists

V1 (the spy-handler game that previously occupied this repo) failed for three reasons: too many entities for the player to model from text, no time to develop attachment to any character, and the handler-from-the-desk framing kept the player emotionally cold. The deeper authoring problem was that V1 wrote *every option* by hand — a pre-LLM CYOA discipline that doesn't play to what LLMs are actually good at.

V2 inverts the bargain. Instead of authoring every scene and every option, the author writes a **world bible** — a grounded description of the place, the people, the truth, and the major beats that constitute uncovering it. The LLM runs scenes inside that world, improvising prose and options. Tools let it advance state when major beats happen. The map keeps the player grounded geographically at all times.

This is closer to "playing tabletop with a GM who's read the campaign book" than "navigating a CYOA tree." The risk is AI-Dungeon drift; the mitigation is heavy world grounding (the LLM cannot invent locations or characters; the truth is fixed; state changes only flow through authored beats).

V1 is not preserved. V2 is a clean greenfield rebuild. No infrastructure is carried over unless it works perfectly for the new system.

---

## 0.4 Vocabulary

The engine and the LLM communicate through **structured output**, not pydantic-ai tool calls. Mental-model hygiene up front saves architecture mistakes later.

- **State change** — a typed field in the LLM's per-turn structured emission. The discriminated union `StateChange` (see §3.3) has kinds like `unlock_beat` and `move_npc`. NOT a tool call.
- **`unlock_beat`, `move_npc`** — names of state-change kinds, NOT tool function names. The model emits them inside a `StateChange[]` field; the engine applies them after validation.
- **No tool-call loop.** There is no pydantic-ai tool execution layer. There is no mid-turn tool-result feedback to the model. One LLM emission per turn-phase (see two-stage emission below).
- **ModelRetry** — pydantic-ai's mechanism for re-prompting the model when its structured output fails application-layer validation (e.g., `unlock_beat` for a beat whose `requires_beats` aren't yet satisfied, or whose `blocked_by_beats` already are). This is the load-bearing safety mechanism that prevents AI-Dungeon-style drift; it's announced here so it isn't surprising in §3. Retry budget capped at N=2 per phase.
- **Two-stage emission, separate system prompts per phase** — each turn is two LLM calls with DIFFERENT system prompts: (1) `StateChangeBundle` against a classifier prompt (full bible: locations + cast-with-knowledge + beats + notes_to_llm + endings), validated, ModelRetry on failure; (2) `ProsePart` against a narrator prompt (locations + cast-with-voice-samples + crime overview + few-shot examples — NO beats, NO notes_to_llm, NO endings, NO per-NPC knowledge), prose + next + options conditioned on the just-applied state. The split costs the inter-phase KV cache hit; it buys Phase 2 attention focus on writing (vs. beat-aware prose). Each phase's system prompt is byte-stable across the session → cache hits across every turn. Phase 2 is additionally scene-bounded conversation: assistant messages accumulate within a scene and reset at scene boundary. Detail in §3.3 / §4.1.

If you find yourself wanting "a tool that...", stop and reconsider. Either it's a new state-change kind (typed field in `StateChange`), or it requires a real tool-call layer that's currently out of scope.

---

## 0.5 Genre fit

This engine is built for a specific story shape. Documenting the constraint up front prevents wasted weeks on a story the architecture cannot deliver. See §2.4 endings for how this constraint shapes the authored ending model (gated on player-driven decision beats, not on engine-summoned climaxes).

**Fits:**
- Mystery / investigation (Pant Goch is this)
- Cosmic horror inquiry (Lovecraft, True Detective S1, Disco Elysium)
- Character study / slice-of-life
- Heist (with caveats — beats encode the plan, but the climactic execution is hard to time)
- Anything where the player already knows the *shape* of the ending and the journey is the point

**Does not fit:**
- Hero's journey / capital-S Story with a required climax
- Action / adventure with a guaranteed setpiece finale
- Romance with a mandatory happily-ever-after
- Comedy where the punchline must land at a specific moment (Hot Fuzz: if you try to build it on this engine, the engine will fight you)

**Why:** the LLM judges when story beats fire. The engine has no mechanism to summon a specific beat on schedule — no act breaks, no pacing curve, no railroaded climax. Endings exist as multiple authored options, gated on decision beats the player must explicitly trigger. Stories that demand a specific narrative arc cannot be reliably delivered by this architecture.

If you find yourself authoring beats like `the_grand_finale_shootout` and wondering how to guarantee they fire at the right moment, the answer is: you can't. Pick a different engine or a different story.

---

## 1. Player experience

The player arrives in Pant Goch. The screen shows a hand-drawn-style map with ten locations connected by lines; the player's marker is at the Old Vicarage. The vicarage's panel slides in with the opening scene's prose. Below the prose: 2–4 options.

To leave a location, the player clicks an adjacent location on the map. Their marker walks the line; the previous scene closes with a final line of prose from the scene runner (a closing beat, the door, the rain); the new location's scene opens.

Inside a scene, the player picks options to drive the conversation. The scene runner improvises responses, generates new options, and decides when a major story beat has been reached — at which point the engine permanently records that beat (Tom named the vehicle; Dylan confessed; Helen handed over the records). The case file in the sidebar updates with each scene's log entry.

There is no day clock, no save points, no XP, no inventory. The game is one continuous read. Scenes last as long as they need to. The investigation advances when the LLM judges, in conversation, that a beat has been earned.

The game ends when the LLM fires one of the authored endings — when accumulated state and player intent line up with what an ending's `notes_to_llm` describes (the player decides to file the story; the player gives up and goes home; the player is run out of the village; etc.).

---

## 2. Authoring

The total authoring surface for v1 is about **15 YAML files**, mostly prose. The author writes locations, cast, truth, and brief. They do not write storylets, predicates, effects, exits, or options. Those are the LLM's job to generate, constrained by the world bible.

### 2.1 File layout

```
world/
    ├── locations/
    │   ├── the_old_vicarage.yaml
    │   ├── the_slaughtered_lamb.yaml
    │   ├── st_caradocs.yaml
    │   ├── the_post_office.yaml
    │   ├── the_lane.yaml
    │   ├── tan_y_coed.yaml
    │   ├── lower_hafod.yaml
    │   ├── bryn_iago_wood.yaml
    │   ├── the_black_dog_bend.yaml
    │   └── knighton.yaml
    ├── cast.yaml
    ├── truth.yaml
    └── brief.yaml
```

### 2.2 Locations

Each location is one file in `world/locations/`. The author writes a rich prose description of the place plus a list of notable details. The map adjacency is declared explicitly.

```yaml
# world/locations/the_slaughtered_lamb.yaml
id: the_slaughtered_lamb
name: The Slaughtered Lamb
adjacent_to: [the_lane, the_village_green]

description: |
  The village pub. Low ceiling, blackened beams, peat fire smouldering in
  the grate even when it isn't strictly cold enough. Behind the bar: a
  blackboard with prices written in Welsh and English, a row of optics,
  and a faded photograph from a 1987 sheepdog trial.

  Most nights it's six or seven regulars. Mid-afternoon it's usually just
  Old Tom Bevan at the second stool from the door, nursing his bitter.

notes:
  - The fruit machine in the corner is broken and has been since 2003.
  - There's a back room used for funerals and the occasional council meeting.
  - The landlord, Eira Pryce, is Maud's sister-in-law; they don't speak.
  - The pub's name relates to a 19th-century smuggling story, not anything supernatural.
```

**Fields:**
- `id` (unique) — referenced by adjacency lists and `cast.yaml` default_locations.
- `name` — display name in the map UI.
- `adjacent_to` (list of location ids) — symmetric adjacency, validated at load.
- `description` — rich prose. The scene runner pulls this into context whenever a scene plays here.
- `notes` (list of strings) — atmospheric and factual details the LLM can weave in. Treat as a fact bank, not a script.

### 2.3 Cast

One file: `world/cast.yaml`. Each NPC is a richly-described character with themes (what they orbit emotionally), knowledge (what they actually know about the truth), and attitude (how they behave toward the player).

```yaml
# world/cast.yaml
- id: tom
  name: Old Tom Bevan
  default_location: the_slaughtered_lamb

  themes:
    - what he saw on the road that night
    - his guilt about not speaking when it would have mattered

  knowledge: |
    On the night of 14 March 2019, Tom was driving home from a friend's
    farm along the B4368, around quarter past eleven. As he came round
    the Black Dog Bend he had to slow for a Land Rover speeding the
    other way — toward Tan-y-coed. He saw it clearly enough in his
    headlights to know whose it was: Eluned Vaughn's. He carried on home.

    The next morning the village knew Megan Pritchard had died on that
    bend the night before. He went to the police and said he'd seen
    "a vehicle." He has never said more. He told himself the Vaughns
    were good people. He has not slept well since.

  attitude: |
    Initially guarded — he has dealt with journalists before and regretted
    it. He is a decent man and will warm to patience, to being seen as
    a person rather than a source. He will close down completely to
    aggressive pressing, to claims of being police or family, or to lies.
    He drinks bitter; he will let the player buy him one.

- id: maud
  name: Maud Pryce
  default_location: the_post_office

  themes:
    - everyone's business
    - the village's image of itself
    - what is and isn't said out loud

  knowledge: |
    Maud knows almost everything that gets said in the village. She
    knows Dylan Vaughn and Megan Pritchard were sleeping together — she
    saw them once, by accident, in 2018, and never told anyone. She
    suspects Eluned knew. She does not know what happened on the road.

  attitude: |
    Polite and warm on the surface. She will gossip willingly about
    almost anything except the specific things she actually knows are
    serious. She is loyal to the village; she will lie by omission. To
    earn anything real from her, the player must convince her that
    telling is the right thing — usually by demonstrating they already
    know more than she expected.

# ...etc for iwan, wendy, eluned, dylan, gareth, helen, owain
```

**Fields:**
- `id` (unique) — referenced by `default_location` and any state-change emissions (e.g. `move_npc.npc_id`).
- `name` — what the player and the LLM call them.
- `default_location` — where they are at game start. Can change via `move_npc`.
- `themes` (list, 1–2 entries) — what this NPC's content should orbit. The LLM uses these to steer scenes. Author discipline, not engine-enforced.
- `knowledge` — what they actually know. The LLM is briefed on this so dialogue is consistent.
- `attitude` — how they behave toward the player and what shifts them.

Implicit state: every NPC starts at `default_location` and `alive: true`. The engine tracks `npc_locations` and `npc_alive` dicts; the LLM mutates `npc_locations` by emitting a `move_npc` state change. `npc_alive` exists for forward compatibility but no `kill_npc` state change is defined in v1 (see §3.3).

### 2.4 Truth

One file: `world/truth.yaml`. This is the most important file. It defines the crime, the beats that constitute uncovering it, and the authored endings.

```yaml
# world/truth.yaml

crime: |
  Megan Pritchard, 19, died on the night of 14 March 2019 on the Black
  Dog Bend of the B4368. The official record says single-vehicle
  accident, blood-alcohol just over the limit. The official record is
  wrong about both.

  Megan had been in a secret relationship with Dylan Vaughn (her
  childhood friend, then 21) for about a year. The Vaughn family had
  an informal understanding that Dylan would marry Lowri Bevan (Tom's
  grand-niece, who lives in Llanidloes) to consolidate two adjacent
  farms. Eluned Vaughn discovered the affair in February 2019. She
  gave Dylan an ultimatum: end it, or lose the farm.

  The night of 14 March, Megan drove out to Tan-y-coed to confront
  Dylan after he'd been avoiding her. There was a fight in the yard.
  She left in tears. Eluned, drunk and furious, took the Land Rover
  and followed her down the B4368 — meaning to confront her face to
  face, not to kill her.

  On the Black Dog Bend the Land Rover came too close to Megan's car.
  Megan oversteered. The Mini went off the road into the ditch and
  rolled. She died on impact. Eluned did not stop. She drove back to
  the farm. She has not slept since.

  Tom Bevan was on the same road, going the other direction, and saw
  the Land Rover speeding away from the bend. He recognised it.

  DC Helen Mortimer of Knighton suspected Eluned but had nothing usable
  — Tom's vague statement, no forensics on a vehicle never examined,
  the village's solid silence. She closed the case rather than chase
  a prosecution she couldn't win. She has not made peace with it.

  Dylan knew, by April 2019, what his mother had done. He has not
  spoken to her about it. He has not spoken to anyone about it.

  The Black Dog folklore is coincidence. Owain Pugh believes the dog
  was seen that night. He may even believe he saw it himself.

beats:
  - id: tom_named_a_vehicle
    requires_beats: []
    blocked_by_beats: [tom_closes_off]
    description: |
      Tom Bevan tells the player he saw a Land Rover speeding away from
      the Black Dog Bend the night Megan died, and that he recognised
      it as Eluned Vaughn's.
    prerequisites: |
      Tom has been visited at least twice in the pub or at his home.
      The player has not lied about being police or family, has not
      threatened him, and has not pushed too aggressively in any
      previous scene. The conversation has reached a moment where Tom
      feels safe enough — usually after the player has shown they care
      about Megan as a person, not just a story.
    notes_to_llm: |
      Tom won't volunteer this. Earn it through patience and the
      accumulated sense of the player being decent. When this moment
      lands, the LLM should narrate Tom's actual words — quiet, finally,
      a relief he didn't know he wanted — and emit unlock_beat with
      "tom_named_a_vehicle". After this, Tom is a different man in
      subsequent scenes: tired, lighter, sometimes weepy.

  - id: tom_closes_off
    requires_beats: []
    blocked_by_beats: [tom_named_a_vehicle]
    description: |
      Tom has decided he will not speak to the player again about
      anything that matters. He may be civil in passing but he will
      not engage on Megan, the road, or the Vaughns.
    prerequisites: |
      The player has done something Tom cannot forgive: lied about
      being police or family, threatened him, mocked his grief, or
      pushed aggressively after he'd asked them to stop. Usually
      after a single bad scene, sometimes accumulated across two.
    notes_to_llm: |
      Fire this when the player has clearly broken Tom's trust. After
      this, scenes with Tom are short and cold — he drinks, he leaves.
      He will not be moved. This is one of the failure paths; it
      blocks tom_named_a_vehicle permanently.

  - id: dylan_admits_the_relationship
    requires_beats: []
    blocked_by_beats: [dylan_refuses_to_speak_again]
    description: |
      Dylan Vaughn admits to the player that he and Megan were
      together. He does not (yet) admit knowing what his mother did.
    prerequisites: |
      The player has approached Dylan in a private setting (not at
      Tan-y-coed with Eluned around). The player has demonstrated
      they already know or suspect — through Maud's leaks, through
      observed body language, through directly asking.
    notes_to_llm: |
      Dylan is in late twenties, quiet, has not been able to put this
      down for seven years. He will admit the relationship more easily
      than expected if the player is gentle. He will NOT admit knowing
      what his mother did at this beat — that's a later beat.

  - id: dylan_admits_he_knows
    requires_beats: [dylan_admits_the_relationship]
    blocked_by_beats: [dylan_refuses_to_speak_again]
    description: |
      Dylan tells the player he has known since April 2019 what his
      mother did on the road that night.
    notes_to_llm: |
      Hardest beat for Dylan. Requires he has already admitted the
      relationship; the engine will reject this beat otherwise. Even
      after that, he will only break here if the player has been kind
      and the conversation has earned the weight.

  # ...etc for: helen_hands_over_records, maud_admits_she_saw_them,
  # owain_describes_the_dog, etc. Plus negative beats for each major
  # NPC: dylan_refuses_to_speak_again, eluned_throws_you_out,
  # helen_won't_take_your_call, etc.

  # Decision beats — fire only when the player has explicitly chosen
  # an ending action. Endings (below) gate on these so the engine
  # never ends the game on inference.
  - id: decided_to_file
    requires_beats: [tom_named_a_vehicle, dylan_admits_the_relationship]
    blocked_by_beats: []
    description: |
      The player has explicitly chosen to write up the story and file
      it with the magazine. This is the point of no return.
    prerequisites: |
      The player must have the minimum evidence (engine-enforced via
      requires_beats) AND must say it explicitly: "I'm writing this up",
      "I'm filing the story", or pick an option that means the same.
    notes_to_llm: |
      Don't pre-empt the player. If they have the evidence but haven't
      said they're filing, don't fire this. The ending downstream of
      this beat is irrevocable.

  - id: decided_to_walk_away
    requires_beats: []
    blocked_by_beats: [decided_to_file]
    description: |
      The player has explicitly chosen to abandon the investigation
      and go home.

  # ...etc one decision beat per ending.

endings:
  - id: published_story
    requires_beats: [decided_to_file]
    notes_to_llm: |
      Fire end_game with "published_story" once decided_to_file has
      fired. The decision beat carries the evidence prerequisites; this
      ending is unconditional once that beat is in state.
    epilogue: |
      The story runs on Friday in the magazine. The Vaughns are
      questioned within the week. Eluned is charged in early autumn.
      Dylan moves away. You hear, second-hand, that Wendy Pritchard
      sent you a letter — short, almost a card. You did not write back.

  - id: buried_the_story
    requires_beats: [decided_to_walk_away]
    # ...etc

  - id: run_out_of_the_village
    requires_beats: [village_runs_you_out]   # itself a beat that fires
                                              # when accumulated hostility
                                              # tips over a scripted moment
    # ...etc
```

**Fields:**
- `crime` — the full truth, written for the LLM to read once at startup. The model is briefed on what's real. It will never have to "figure out" the mystery; it knows.
- `beats` (list of ~15–25 including negatives and decisions) — the moments that constitute the investigation. Each beat is a unit of progress the LLM unlocks by emitting an `unlock_beat` state-change. Has:
  - `id` (unique across all beats and endings)
  - `description` (prose: what the moment is)
  - `requires_beats` (list of beat ids, default `[]`) — engine rejects `unlock_beat` if any prerequisite isn't in `state.unlocked_beats`. Triggers ModelRetry. Use to enforce ordering (e.g. `dylan_admits_he_knows.requires_beats: [dylan_admits_the_relationship]`).
  - `blocked_by_beats` (list of beat ids, default `[]`) — engine rejects `unlock_beat` if any blocker IS in `state.unlocked_beats`. Triggers ModelRetry. Use to model failure paths: `tom_named_a_vehicle.blocked_by_beats: [tom_closes_off]` makes the positive beat unreachable once Tom has shut down.
  - `prerequisites` (prose, optional) — soft prereqs the LLM judges, beyond what `requires_beats`/`blocked_by_beats` enforce mechanically.
  - `notes_to_llm` (prose) — clear instructions on when to fire and what to narrate.
- **Negative beats.** Author 2–3 per major NPC (e.g. `tom_closes_off`, `dylan_refuses_to_speak_again`, `eluned_throws_you_out`). Use `blocked_by_beats` on the corresponding positive beats. This is what gives "patience" mechanical contrast — without negative beats, there's nothing to fail at.
- **Decision beats.** One per ending. Encode the irrevocable player choice that triggers an ending. `requires_beats` carries the evidence prereqs (engine-enforced); `notes_to_llm` reminds the LLM to wait for explicit player intent. Endings then gate on the decision beat alone — the engine never ends the game on inferred intent.
- `endings` (list of ~5–8) — terminal states the LLM fires by emitting an `end_game` next-discriminator. Has:
  - `id` (unique)
  - `requires_beats` (list of beat ids, default `[]`) — typically just `[<the ending's decision beat>]`. Engine rejects `end_game` if not satisfied.
  - `notes_to_llm` (prose)
  - `epilogue` (prose, streamed when the ending fires)

### 2.5 Brief

One file: `world/brief.yaml`. The opening hook and initial state.

```yaml
# world/brief.yaml
player_starts_at: the_old_vicarage

opening_seed: |
  The player has just arrived in Pant Goch after a three-hour drive
  from London. Mrs Hughes the housekeeper (not a major character — just
  background) lets them into the Old Vicarage and hands over the keys.
  Mentions there's a letter waiting for them on the hall table from
  someone called Iwan Pritchard. The brief from the magazine's editor
  is summarised here:

    Six weeks ago, Iwan Pritchard wrote claiming his daughter's death
    in 2019 was not the drink-driving accident the police recorded.
    He thinks another vehicle was involved and the local force didn't
    want to look. The editor has given the player a six-week budget
    and this room at the Old Vicarage. Story or no, money runs out
    mid-September.

initial_flags: []   # no flags set at start; the game proceeds purely
                    # on unlocked beats
```

The scene runner is invoked with the opening_seed as its opening prose context. The first scene plays out.

### 2.6 Validation invariants

The world loader enforces these at startup. Each is a single Pydantic validator.

1. **Location id uniqueness.**
2. **Symmetric adjacency.** If A's `adjacent_to` contains B, B's must contain A.
3. **NPC id uniqueness.**
4. **Every NPC's `default_location` references a defined location.**
5. **Every beat id is unique across the truth file.**
6. **Every ending id is unique across the truth file.** Beat and ending ids share a namespace (must be globally unique).
7. **`player_starts_at` in brief.yaml references a defined location.**
8. **Every `requires_beats` and `blocked_by_beats` reference resolves to a defined beat id.** No self-reference.
9. **The `requires_beats` graph has no cycles.** A cycle would make a beat unfireable; load fails with the cycle path.
10. **Every ending's `requires_beats` references a defined beat id.** Strongly recommended (warning, not error) that every ending's prereqs include exactly one decision beat.

That's the entire validation surface. There are no predicates to validate, no exit graphs to check, no effects to parse. The world is mostly prose; the validation is mostly "do the few ids resolve and is the beat graph well-formed."

---

## 3. Engine

### 3.1 State

```python
@dataclass
class State:
    player_location: str                          # id of the location the player is at
    npc_locations: dict[str, str]                 # npc_id -> location_id
    npc_alive: dict[str, bool]                    # npc_id -> alive
    unlocked_beats: set[str]                      # beat ids that have fired
    dialogue_history: list[TurnRecord]            # accumulated turns in the current scene
                                                   # each TurnRecord = (prose, applied_state_changes,
                                                   # next_directive, player_pick)
    game_over: GameOver | None = None             # set when end_game fires
```

That's the whole state model. No flags dict, no qualities, no effects vocabulary, no scene/storylet ids tracked. The minimum required to gate behaviour: where the player is, where each NPC is, who's alive, what beats have happened.

### 3.2 The scene loop

Each turn is two LLM calls (see §0.4 two-stage emission). Validation sits between them; ModelRetry catches semantic violations the grammar sampler can't.

```
ON GAME START:
- state = make_initial_state(brief, cast)
- enter_scene(state.player_location, opening_seed=brief.opening_seed)

ENTER SCENE (location_id, opening_seed=None):
- state.dialogue_history = []
- emit SSE: scene_start { location_id, location_name }
- run_turn(opening_seed=opening_seed)

RUN TURN (opening_seed=None, player_pick=None):

  # ---- Phase 1: state changes ----
  - build state-change prompt with:
    - location.description + location.notes
    - all NPCs where npc_locations[id] == location_id (themes, knowledge, attitude)
    - full unlocked_beats, full unfired-beats list
    - all endings (with their requires_beats)
    - full case-file log (all entries to date)
    - state.dialogue_history (this scene so far)
    - opening_seed if first turn of scene
    - player's last pick (text) if any
  - call scene_runner.emit_state_changes() -> StateChangeBundle
    - structured output: list[StateChange] (UnlockBeat | MoveNPC)
    - grammar sampler enforces valid ids
  - validate StateChangeBundle:
    - for each StateChange:
      - UnlockBeat: requires_beats all in unlocked_beats? blocked_by_beats none in unlocked_beats?
      - MoveNPC: to_location is a defined location? npc_id is alive (if we ever add KillNPC)?
    - on validation failure: ModelRetry with the specific error in context, up to N=2
    - on final failure: drop the offending change, emit SSE warning, continue
  - apply validated changes to state in order
  - emit SSE state_change for each applied change

  # ---- Phase 2: prose + next ----
  - build prose prompt (same context as phase 1, plus the just-applied state changes)
  - call scene_runner.emit_prose() -> ProsePart
    - structured output: prose: str, next: ContinueScene | EndScene | EndGame
    - grammar sampler enforces valid ending_id on EndGame
  - validate ProsePart:
    - if next is EndGame: ending.requires_beats all in unlocked_beats?
    - on validation failure: ModelRetry with error, up to N=2
    - on final failure: rewrite next as ContinueScene with a synthetic option ("[engine: ending prereqs not met; continue scene]"); log error
  - stream prose clause-by-clause (paced ~300 wpm via paced_clause_emitter)
  - append (prose, applied_state_changes, options_or_next) to state.dialogue_history
  - dispatch next:
    - ContinueScene: emit options; wait for player pick; recurse to RUN TURN with player_pick
    - EndScene: emit SSE scene_exit; trigger log_writer; return to map view
    - EndGame: set state.game_over, stream ending.epilogue, emit SSE game_over

PLAYER CLICKS MAP (to_location: str):
- only valid if to_location is in current_location.adjacent_to
- if a scene is active:
  - hard-end the scene: call scene_runner.emit_closing_prose(to_location) once,
    stream the result as the scene's final prose, skip phase 1 (no state changes
    on the way out), no options emitted
- emit SSE location_change { from, to }
- state.player_location = to_location
- enter_scene(to_location)
```

### 3.3 LLM output schemas (two-stage)

The scene runner emits two structured outputs per turn (see §3.2), both via pydantic-ai NativeOutput strict mode against a shared system prompt.

**Phase 1 — state changes:**

```python
class StateChangeBundle(BaseModel):
    state_changes: list[StateChange] = []         # 0+ in order

class StateChange(BaseModel):
    # Discriminated union below.

class UnlockBeat(StateChange):
    type: Literal["unlock_beat"] = "unlock_beat"
    beat_id: Literal[<the actual authored beat ids, built per call>]

class MoveNPC(StateChange):
    type: Literal["move_npc"] = "move_npc"
    npc_id: Literal[<authored npc ids>]
    to_location: Literal[<authored location ids>]
```

**Phase 2 — prose + next:**

```python
class ProsePart(BaseModel):
    prose: str                                    # ~80-200 words; streamed to player
    next: ContinueScene | EndScene | EndGame      # exactly one

class ContinueScene(BaseModel):
    type: Literal["continue"] = "continue"
    options: list[Option]                          # 2-4

class EndScene(BaseModel):
    type: Literal["end_scene"] = "end_scene"
    # ProsePart.prose serves as the closing beat

class EndGame(BaseModel):
    type: Literal["end_game"] = "end_game"
    ending_id: Literal[<authored ending ids>]

class Option(BaseModel):
    text: str                                     # what the player sees
```

All `Literal[...]` types are built dynamically per call from the loaded world bible. The llama.cpp grammar sampler enforces them at decode time — the model literally cannot emit an invalid beat id, npc id, location id, or ending id. Application-layer validation (`requires_beats`, `blocked_by_beats`, ending prereqs) is enforced via ModelRetry between or within phases (see §3.2 RUN TURN).

**No `KillNPC`.** This story has no in-game deaths. If a future scenario needs one, add it as a new state-change kind; do not pretend it's authored already.

### 3.4 Player-driven leave

When the player clicks an adjacent location during an active scene, the engine hard-ends the scene without offering the LLM a choice to continue. It calls a dedicated closing-prose method on the scene runner — single phase, structured output is just `closing_prose: str` — passing the destination as context: "the player is leaving for {to_location.name}; write one line of closing prose."

The closing prose streams; SSE `scene_exit` fires; `location_change { from, to }` fires; `enter_scene(to_location)` runs. No state changes are accepted on the way out — the closing-prose schema doesn't include `state_changes`. Player agency on movement is absolute.

### 3.5 What the engine does NOT do

- **No selector.** The scene runner is the only LLM agent that makes scene-level decisions.
- **No effect engine.** State changes are direct dict mutations on `state`; there's no effect parser, no predicate evaluator, no `set/add/max/min` ops. Beats are the only state-change unit.
- **No reactions.** No second-pass post-action firing. Everything that happens in a turn happens in that turn's `state_changes`.
- **No tick.** No day counter, no automatic time progression.
- **No catalog or eligibility filtering.** The LLM looks at the state and the world bible and decides what happens. The engine does not pre-filter possibilities.

---

## 4. The LLM layer

### 4.1 Three agent layers

**Scene runner — Phase 1 (state-change classifier).** Stateless across turns. System prompt baked once at startup (full bible: locations + cast-with-knowledge + beats with `requires_beats`/`blocked_by_beats`/`notes_to_llm` + endings). `emit_state_changes()` → `StateChangeBundle`. Validated; ModelRetry up to N=2 on `requires_beats` / `blocked_by_beats` violations and the ≤2 cap; final failure drops the offending change.

**Scene runner — Phase 2 (narrator).** Scene-bounded conversation. System prompt baked once at startup, separate from Phase 1's (locations + cast-with-voice-samples + crime overview + few-shot style examples — NO beats, NO notes_to_llm, NO endings, NO per-NPC knowledge). `emit_prose(runtime, …)` → `ProsePart`, conditioned on the just-applied state changes (passed in the user message). `runtime.phase2_message_history` accumulates within a scene; cleared in `enter_scene`. Validated for ending prereqs / leave-pick → EndScene / ≥1 leave option; ModelRetry on violation. On retry exhaustion, engine emits generic fallback prose that is NOT persisted to history (voice continuity preserved). Prose streams word-by-word at ~80ms pacing.

The two phases now have DIFFERENT system prompts → no inter-phase KV cache sharing within a turn. Inter-turn cache hits remain on both: Phase 1 system prompt is byte-stable across the session; Phase 2 system prompt is byte-stable across the session AND scenes, with intra-scene message history acting as additional cached prefix. The trade is deliberate: Phase 2 attention focuses on writing, not on beat structure.

A third method, `emit_closing_prose(to_location)` (single phase, schema is just `closing_prose: str`), serves player-driven leave (§3.4).

**Log writer.** Conversation-based, accumulating message_history across scenes. System prompt stable. Each scene-end appends one user message (scene summary: location, NPCs present, beats unlocked, key turns) + one assistant message (the case-file entry). `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` is set in lifespan (load-bearing — without it Gemma 4 drifts in multi-turn). Same flag covers Phase 2's scene-bounded history.

### 4.2 Scene runner prompt structure

**System prompt (stable, baked at startup):**

```
[ROLE]
You are the scene runner for a single-player investigation game set in
a rural Welsh Marches village called Pant Goch. The player is a
freelance journalist investigating the death of Megan Pritchard. You
run scenes turn by turn through structured output.

[THE WORLD]
{full contents of all locations/*.yaml, formatted compactly}

[THE CAST]
{full contents of cast.yaml}

[THE TRUTH]
{full contents of truth.yaml — crime + beats + endings}

[YOUR JOB]
Each turn is two structured emissions against this same system prompt:
  Phase 1 — StateChangeBundle: zero or more state changes
    (UnlockBeat | MoveNPC). The engine validates and applies them.
  Phase 2 — ProsePart: prose (80-200 words) consistent with the
    just-applied changes, plus a `next` directive
    (ContinueScene with 2-4 options | EndScene | EndGame).

The engine will tell you in the user prompt which phase you are in.

[RULES]
1. Never invent characters not in the cast. Never invent locations not
   in the world. The player's understanding of who exists comes from
   you; lie about anyone and the world becomes incoherent.

2. Generate 2-4 options per ContinueScene turn. Options are what the
   player can say or do that has no major consequence — dialogue,
   observation, minor action. Options must NOT promise a consequence
   that requires a beat the engine doesn't have. If you're tempted to
   offer "Punch Tom in the face," don't; there's no beat for it.

3. Unlock beats when their notes_to_llm describes the moment. Re-read
   the prereqs. Don't unlock prematurely; don't withhold past the
   right moment. The beat is the moment, not the conversation around it.
   The engine will reject (and re-prompt you with the error) any
   unlock_beat whose `requires_beats` aren't satisfied or whose
   `blocked_by_beats` already are.

4. End the scene when the conversation has reached a natural stop or
   when a beat has fired and the scene's purpose is done. Aim for
   roughly 4-8 turns per scene; longer is fine if the conversation
   warrants. Don't drag.

5. End the game only when the player has clearly chosen the action
   that triggers an ending — not just when prerequisites are met.
   Concretely: each ending depends on a "decision beat" that fires only
   when the player explicitly commits. End the game only after that
   decision beat is in state. The engine will reject EndGame otherwise.

6. Player-driven leave (the player clicks an adjacent map location) is
   handled by a separate single-phase call to you (`emit_closing_prose`).
   You won't see those as a normal turn.

7. Free options are pure narrative — they have no mechanical effect.
   All state changes flow through unlock_beat / move_npc.
```

**Phase 1 user prompt (state changes):**

```
[YOU ARE AT]
{current_location.name}: {current_location.description}
Notes: {current_location.notes joined}

[PRESENT]
{for each npc_id where npc_locations[id] == current_location:
  - {name}: {themes}. Knowledge: {knowledge}. Attitude: {attitude}.}

[STATE]
Unlocked beats so far: {unlocked_beats}

[ALL UNFIRED BEATS]
{for each beat in truth.beats where beat.id not in unlocked_beats:
  - {beat.id}: {beat.description}.
    requires_beats: {beat.requires_beats}
    blocked_by_beats: {beat.blocked_by_beats}
    Prereqs: {beat.prerequisites}
    Notes: {beat.notes_to_llm}}

[ALL ENDINGS]
{for each ending in truth.endings:
  - {ending.id}: requires {ending.requires_beats}
    Notes: {ending.notes_to_llm}}

[CASE FILE — ALL ENTRIES]
{full log, every entry to date, in order}

[THIS SCENE SO FAR]
{for each turn in dialogue_history:
  prose: ...
  state_changes_applied: [...]
  player picked: "{option text}"}

[OPENING SEED — if this is the scene's first turn]
{opening_seed text, if any}

[GENERATE STATE CHANGES FOR THIS TURN]
Emit zero or more state changes (UnlockBeat, MoveNPC) representing what
this turn's events will accomplish mechanically. Most turns emit zero.
A beat fires only when the moment described in its notes_to_llm has
genuinely landed in the conversation. The engine will reject any
unlock_beat whose requires_beats aren't satisfied or whose
blocked_by_beats already are; you'll be re-prompted with the error.
```

**Phase 2 user prompt (prose + next):** identical to phase 1 plus an additional `[STATE CHANGES JUST APPLIED]` section listing what phase 1 emitted, then:

```
[GENERATE PROSE AND NEXT]
Write the prose for this turn (80-200 words), narrating the events
consistent with the state changes above. Then emit one of:
  - ContinueScene with 2-4 options
  - EndScene (use this when the conversation has reached a natural pause)
  - EndGame with an ending_id (only when that ending's requires_beats
    are satisfied — the engine will reject otherwise)
```

**Note on log size.** The full case file and full unfired-beats list grow over the playthrough. Estimated steady state at end of game: ~15 log entries × ~100 words + ~25 beats × ~80 words ≈ 3.5k words ≈ 5k tokens of dynamic suffix. The 32k context window absorbs this comfortably; the system prefix (world bible) stays cached.

### 4.3 Model + runtime

- **Model:** Gemma 4 E4B (GGUF) via bundled `llama-server`. Same as v1.
- **Launcher flags:** `--jinja --flash-attn on --batch-size 512 --ubatch-size 512`. KV cache fp16.
- **Sampling:** temp=1.0, top_p=0.95, top_k=64. Do not drop temp for "judgment" turns — the model needs creative latitude inside the constraint of the bible.
- **Grammar sampler:** llama.cpp grammar sampling enforces the per-call `Literal[...]` enums for ids (beat, npc, location, ending). The LLM cannot emit a beat that doesn't exist.
- **KV-cache discipline:** each phase's system prompt is baked once at startup and never edited. Per-turn user prompts are appended after the same stable per-phase prefix. The Phase-1/Phase-2 split costs the intra-turn inter-phase hit (system prompts differ); it preserves the intra-session inter-turn hit on both phases and the intra-scene Phase 2 conversation-prefix hit. Cache hit rate is still the lever for latency.
- **Dev gateway:** keep `SMALLVILLE_USE_INFERNO=1` env-var pattern. If set, lifespan probes inferno's `/health`, points pydantic-ai at inferno's `/v1`. Otherwise spawn bundled llama-server on `:8080`.

---

## 5. Frontend

**Visual reference: `design/mockup-stage/`** — a static HTML+React mockup of the final composition. Open `index.html` directly in a browser (via `python -m http.server` from `design/`). Click through it to see the intended UX; read its `styles.css` for design tokens and `app.jsx` for the React patterns. The mockup is the source of truth for layout, motion, and typography. Implementation lives in `web/`.

Single-page web app. Two-pane composition, **58fr / 42fr**:

- **Left pane (the "stage").** Full-height, edge-to-edge. Three modes switched by a small tab strip at the top-left:
  - `location` (default) — a sepia-pushed photograph of the current scene's location. Cross-fades when the scene changes. Film-slate caption at the bottom-left (location name + date).
  - `map` — full Pant Goch survey. Every pin labelled in italic display serif. Current location red, visited locations dim-red, dashed roads, river, contours. Read-only in v1.
  - `notes` — vertical list of case-file entries (one per visited scene). Click an entry → switches to `location` and scrolls the prose panel to that chapter.
  - Walking auto-takes over with a map view + red pencil tracer drawing along the road from current → destination location. Tabs disable during walk; reverts to `location` mode when the walk completes.
- **Right pane (the prose panel).** Vellum paper background. Top bar with case-file label, entry count, restart. Internal scroll — **the page does not scroll, the panel does**. Chapters accumulate inside it separated by a thin rule. Each chapter: italic display chapter heading, drop cap, justified prose streaming word-by-word, choices as numbered italic footnotes at the end.
- **Page-turn scroll.** Within a chapter, no autoscroll on every paragraph. When a new paragraph is about to stream, if the previous paragraph extends past ~68% of the panel height, smooth-scroll so the previous paragraph sits at the top. Same fires for the choices block at ~55%. Predictable chunks; text doesn't move once placed.
- **Walking interstitial.** Map fades into the left pane (~400ms), red tracer animates along the road (~1500ms), destination pin and label settle, then map fades back to the destination's photograph (~400ms). The right pane stays put throughout.
- **Hover popups** (NPCs and locations). Marked phrases in the prose render with a red dotted underline. Hover → a dark floating popup over the left pane with the entity's heading + description. Requires either inline `[text|entity_id]` markup from Phase 2 or a separate annotations field on `ProsePart`; see frontend implementation notes for trade-offs. Gloss bodies derive from `cast.yaml` (NPCs) and `world/locations/*.yaml` (locations).
- **End-of-game screen.** When `end_game` fires, the ending's epilogue streams in a centred panel. No further input.

SSE event surface:

| event | payload | when |
|-------|---------|------|
| `scene_start` | `{ location_id, location_name }` | entering a scene |
| `prose_chunk` | `{ text, idx, is_last }` | each clause of streamed prose |
| `options` | `{ options: [{text}, ...] }` | turn waits on player input |
| `state_change` | `{ kind, payload }` | each applied state change (for animations / debug) |
| `scene_exit` | `{}` | EndScene fired |
| `location_change` | `{ from, to }` | between scenes when location changed |
| `log_entry_chunk` | `{ text, idx, is_last }` | case-file entry streaming |
| `game_over` | `{ ending_id }` | EndGame fired |

Player input over HTTP POST:
- `POST /pick { option_idx }` — option picked in current turn
- `POST /walk { to_location }` — map click

---

## 6. Setting bible — Pant Goch

(The full content of `world/` once authored. This section is a sketch; the real bible lives in the YAML files.)

### 6.1 The village

Pant Goch ("red hollow", Welsh) is a village of ~400 in the Welsh Marches between Knighton and Bishop's Castle. Place-names are bilingual; the language isn't spoken day-to-day. Economy: hill farming, smallholdings, a few Knighton commuters. The pub is the only thing open after 6. The post office shuts at 4. The primary school has 19 pupils.

The village sits in a hollow formed by two streams. Bryn Iago Wood rises to the north; Caer Caradoc, an iron-age hillfort, crowns the south ridge. The Black Dog Bend on the B4368 has been associated since at least the 1840s with a folkloric apparition — a large black dog seen before a death. Most locals shrug at it. They also don't drive that bend at speed.

### 6.2 The map (10 locations)

```
                              [bryn_iago_wood]
                                     |
[knighton] --- [the_lane] --- [tan_y_coed] --- [lower_hafod]
      |              |
[the_black_dog_bend][st_caradocs]
                     |
            [the_slaughtered_lamb]
                     |
            [the_post_office]
                     |
              [the_old_vicarage]
```

| id | name | role |
|----|------|------|
| `the_slaughtered_lamb` | The Slaughtered Lamb | pub; Tom default here |
| `st_caradocs` | St Caradoc's | church + churchyard; ancient yew |
| `the_post_office` | The Post Office | shop + post; Maud default here |
| `the_old_vicarage` | The Old Vicarage | player's lodgings |
| `the_lane` | The Lane | through-road; transition hub |
| `tan_y_coed` | Tan-y-coed | Vaughn farm; Eluned + Dylan default here |
| `lower_hafod` | Lower Hafod | Pritchard smallholding; Iwan, Wendy, Owain |
| `bryn_iago_wood` | Bryn Iago Wood | woods above village |
| `the_black_dog_bend` | The Black Dog Bend | road where Megan died |
| `knighton` | Knighton | market town; Helen + Father Gareth default here |

### 6.3 The cast (9 named)

| id | name | default location | role |
|----|------|------|------|
| `tom` | Old Tom Bevan | the_slaughtered_lamb | saw the Land Rover; carries guilt |
| `maud` | Maud Pryce | the_post_office | postmistress; knew about the affair |
| `iwan` | Iwan Pritchard | lower_hafod | Megan's father; wrote to magazine |
| `wendy` | Wendy Pritchard | lower_hafod | Megan's mother; wants it dropped |
| `eluned` | Eluned Vaughn | tan_y_coed | killed Megan; lives with it |
| `dylan` | Dylan Vaughn | tan_y_coed | Megan's secret lover; knows |
| `gareth` | Father Gareth Hughes | knighton | visiting priest; sympathetic |
| `helen` | DC Helen Mortimer | knighton | closed the case; suspected truth |
| `owain` | Owain Pugh | lower_hafod | folklore-believing smallholder |

(Megan Pritchard exists only as fact and prose; she is dead from before the game starts.)

### 6.4 The truth and the brief

See §2.4 and §2.5 examples — those are nearly the actual content. The crime, ~10–15 beats, ~5–8 endings, and the editor's six-week brief.

### 6.5 Example scene flow (illustrative)

Player walks to the pub on day 1. Tom is at the bar.

Scene runner is called with: pub description + notes; Tom's themes/knowledge/attitude; empty dialogue_history; full unfired-beats list (`tom_named_a_vehicle` is in scope but its prereqs aren't met, `tom_closes_off` blocks it if it ever fires); full case-file log (none yet); the opening_seed for this scene is None (we're not at the opening scene; player walked here).

**Turn 1 — Phase 1** (state changes):
```
StateChangeBundle(state_changes=[])    # nothing fires yet
```

**Turn 1 — Phase 2** (prose + next):
```
prose: "The peat fire's going. Old Tom is at the second stool, like
       Mrs Hughes said. He glances up as the door bangs shut behind
       you, then back down at his bitter. 'Bloody door,' he mutters,
       to no one in particular."
next: ContinueScene(options=[
  "Sit at the bar a few stools down",
  "Take the stool next to him",
  "Order a drink first"
])
```

Player picks "Take the stool next to him."

**Turn 2 — Phase 1:** `state_changes=[]`. **Phase 2:**
```
prose: "You slide onto the stool. Up close he looks older than he did
       from the door — sun-creased, knuckles thick from work. He
       doesn't acknowledge you. There's a moment, and then: 'You'll be
       the one from the magazine, then. Iwan said someone'd be coming.'"
next: ContinueScene(options=[
  "'That's me. I'm sorry to bother you.'",
  "'How did you know?'",
  "'Word travels fast.'"
])
```

(...several turns later, Tom has talked about Megan, expressed his own grief, the player has been patient and not pressed too hard. The scene runner judges Tom is on the edge of saying what he saw.)

**Turn N — Phase 1:**
```
StateChangeBundle(state_changes=[
  UnlockBeat(beat_id="tom_named_a_vehicle")
])
```

Engine validates: `requires_beats=[]` ✓, `blocked_by_beats=[tom_closes_off]` not in unlocked_beats ✓. Applied. (Had the LLM tried to fire `dylan_admits_he_knows` here without `dylan_admits_the_relationship` already in state, validation would have rejected and ModelRetry would re-prompt with: "rejected: dylan_admits_he_knows requires [dylan_admits_the_relationship], not yet unlocked.")

**Turn N — Phase 2:**
```
prose: "He sets the pint down very slowly. Not looking at you. 'Saw
       a Land Rover that night. Going the other way. Past the bend.
       Recognised it.' A long pause. 'Eluned Vaughn's. Knew her vehicle
       since I was a lad and she was a girl.' He looks up. There are
       tears in his eyes that he is not acknowledging. 'I told them I
       saw a vehicle. I didn't say whose.'"
next: ContinueScene(options=[
  "'Why didn't you tell them?'",
  "'It's all right, Tom. It's all right.'",
  "Stay quiet."
])
```

Subsequent scenes with Tom will play differently because his attitude implicitly shifts (the LLM reads `unlocked_beats` from state and knows he's already spoken).

A few turns later the conversation has reached a natural pause. **Phase 1:** `state_changes=[]`. **Phase 2:**
```
prose: "He drains the pint. 'Get me another one, would you. Then I
       think I'd best be off home.'"
next: EndScene
```

Scene ends. Log writer is invoked with the scene summary and writes a case-file entry. Player is back to the map view, marker still at the pub. They click adjacent location to leave.

---

## 7. Out of scope for v1

Deliberately deferred. Each has a credible v1.1+ story.

- **Numeric qualities.** No trust scores, no guilt counters, no coin. State is location + beats + alive.
- **Generic flags.** No `set_flag(name, value)` state-change kind. All state changes flow through authored beats. (Per-NPC disposition counters were considered and rejected — `requires_beats` + `blocked_by_beats` + LLM-judged firing covers the same ground.)
- **NPC schedules.** NPCs are static at their default_location unless `move_npc` fires. No "Tom goes home at night" without authoring it as a beat or scene effect.
- **Inventory.** Out.
- **Time / days / clock.** Out for v1.
- **Authored storylets with predicates and exits.** The whole v1.0 spec direction. Replaced by the bible + structured-output design.
- **Salience-weighted next-scene picking.** Not applicable — there is no "next scene picker"; the LLM constructs scenes from arrival context.
- **Replay / save slots.** Single linear playthrough; close-tab resets.
- **Authoring tools.** Hand-written YAML. LLM-assisted authoring is a v1.1 concern.
- **Pydantic-ai tool calls with mid-turn feedback loops.** v1 uses structured output across a two-stage emission per turn (state changes, then prose + next). The only "feedback" is ModelRetry on application-layer validation failure (§0.4). No tool-execution layer, no LLM-issued tool calls returning results to the model. Revisit only if a future scenario genuinely needs mid-turn information retrieval (e.g., `lookup_npc_knowledge` returning authored facts on demand).

---

## 8. Build plan

Roughly in dependency order.

**M0 — Skeleton.** `pyproject.toml`, FastAPI app, lifespan that spawns bundled llama-server or probes inferno. Empty `world/`. `GET /health` returns 200.

**M1 — World loader.** `models.py` with Pydantic types for Location, NPC, Beat, Ending, Brief. `world.py` globs world files and validates the seven invariants from §2.6. Three sample location files + minimal cast + truth + brief. `pytest` test: world loads, invariants enforced.

**M2 — State + engine core (no LLM).** `state.py` with State dataclass + `make_initial_state(brief, cast)`. Engine harness that applies state_changes directly (without going through the LLM), including the `requires_beats` / `blocked_by_beats` validator. `pytest` tests: apply UnlockBeat / MoveNPC and verify state mutates correctly; assert validation rejects an UnlockBeat whose `requires_beats` aren't satisfied; assert validation rejects an UnlockBeat whose `blocked_by_beats` are satisfied; assert ending-prereq check rejects EndGame when the decision beat hasn't fired.

**M3 — Scene runner agent (two-stage).** `agents.py` with one scene runner exposing three methods: `emit_state_changes()`, `emit_prose()`, `emit_closing_prose()`. All three share a single stable system prompt (the world bible). Dynamic `Literal[...]` types built per call for ids. ModelRetry wired up against `requires_beats` / `blocked_by_beats` and ending prereqs, capped at N=2. `scene.py` with the two-call per-turn loop. Hard-code a starting location; play one scene from the terminal until EndScene fires. Verify state changes apply, structured output validates, ModelRetry fires when ordering is violated.

**M3.5 — Beat-firing validation gate.** Stub world: 3 locations, 3 NPCs, 5 beats including at least one `requires_beats` chain and one `blocked_by_beats` relationship. Play 10 scenes on Gemma 4 E4B (the actual shipping model — do not validate on a larger model). Subjectively grade beat-firing: did the right beat fire at roughly the right moment? Did ModelRetry catch ordering violations? Did the model attempt premature endings?

- ≥80% acceptable beat-firing → proceed to M4.
- <80% → spec out mechanical floors (per-NPC visit counters, dialogue-history-aware prereqs, etc.) before continuing. The whole architecture rests on the LLM being able to judge beat moments competently inside the constraint of `requires_beats`/`blocked_by_beats`; if it can't, the rest of the build is wasted.

This is the single highest-leverage gate in the build plan. Do not skip past it.

**M4 — Frontend (text-only, no map).** `web/` with scene view (streaming prose + options). Click an option → next turn streams. Click "leave" button → scene ends. No map yet; just location name displayed.

**M5 — Clickable map.** Author the full 10-location adjacency. Frontend map view (SVG or canvas). Player marker. Cross-location transitions trigger the player-leaving flow.

**M6 — Log writer.** Second pydantic-ai agent with conversation-based message_history. `OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)` in lifespan. After each EndScene, log writer is called; entry streams to sidebar case-file view.

**M7 — Full world authoring.** Write `cast.yaml`, `truth.yaml` (full crime + ~10–15 beats + ~5–8 endings), all 10 location files. Brief.yaml. Dogfood end-to-end.

**M8 — Playtest.** The author plays through. Tune `notes_to_llm` on beats and endings based on where the LLM fires them too early, too late, or never. Adjust NPC attitudes if behaviour feels off. Add or remove beats.

After M8, v1 is proven. v1.1 work (LLM-assisted authoring? numeric pressure? proper endings UI?) gets scoped from playtest evidence rather than guesses.

---

## Critical files in current repo to read before building

Lift patterns from these (carefully — they're v1 spy-handler and many will need adjustment):

- `src/handler/launcher.py` — bundled llama-server spawn, jinja flag, kernel params.
- `src/handler/main.py` — lifespan with `OpenAIModelProfile` setup, `paced_clause_emitter`, SSE event shape.
- `src/handler/agents.py` — stable-system / dynamic-user prompt split for KV reuse, NativeOutput with `strict=True` + dynamic Literal types for grammar sampling.
- `CLAUDE.md` — invariants and call-site comments explaining why things are the way they are.

Read, understand, rebuild clean. Do not transplant.
