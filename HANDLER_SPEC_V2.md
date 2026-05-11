# HANDLER — Design Spec v2

> Concept artifacts: `_mockups/handler-spymaster.html` (v.0.1, retired), `_mockups/handler-authorize.html` (v.0.2, the verb).
> This document is design-level, not implementation. Tech-stack and file-layout details are sketched in §13–14; the bulk is game design and engine model.

---

## 1. Vision

**Handler** is a single-player spy game built around one verb: *authorize a course of action proposed by Operations*. The player sits at a desk in a room without windows. Operations Desk drafts options each cycle; the player picks one; an agent in the field carries it out and files a dispatch. The campaign closes when a mission resolves, succeeds, or fails.

The game is LLM-native because the Operations Desk's drafting and the agents' dispatches cannot be authored exhaustively — both ends of the loop produce situated prose conditioned on world state. Mechanics, however, are fully deterministic and live in YAML. The LLM never decides outcomes.

**Genre frame: survival and protection.** Tension comes from agents in danger, cover decaying, time running short — not from puzzle-solving. Inspirations: Failbetter Games (Sunless Sea, Fallen London), King of Dragon Pass / Six Ages, The Walking Dead (Telltale), Suzerain, This War of Mine; in fiction: Tinker Tailor Soldier Spy, Slow Horses, The Sandbaggers.

**Core design commitments:**

1. *Choice over composition.* The player picks; they don't write briefs.
2. *Prose is the interface to mechanics.* Option prose telegraphs cost and shape honestly. Player reads, reasons, picks.
3. *Mechanics are deterministic and authored.* Balance lives in YAML, never in the LLM.
4. *The player can always know what to do.* Options are always present. No blank pages.
5. *Decisions accrete and persist.* Sacrificed agents stay sacrificed. The campaign remembers.

---

## 2. The Verb

**Authorize one of 3–4 options per cycle.** That is the entire verb.

Each cycle the Operations Desk presents 3–4 candidate courses of action. Each option is a short paragraph naming an agent, an action, and an honest description of the cost. The player reads, picks one, clicks. The cycle resolves.

There is no other player input. No free-text composition. No multi-step decisions per turn. No agent-by-agent micromanagement.

---

## 3. The Cycle

Every cycle has the same shape. Three LLM calls and one player click:

```
─── CYCLE OPEN ───────────────────────────────────
  i.   STATE OF PLAY MEMO        (LLM: Operations voice)
       Synthesis of where things stand. Looks forward.
  ↓
  ii.  OPTIONS DRAFTED           (LLM: Operations voice)
       3–4 candidate courses of action, tagged with
       (archetype, agent, params), with prose paragraphs.
  ↓
  ── PLAYER PICKS ──             (player click)
  ↓
  iii. EFFECTS APPLY             (engine, deterministic)
       Authored archetype effects mutate state.
       Beats fire on any newly-satisfied predicates.
  ↓
  iv.  DISPATCH                  (LLM: acting agent voice)
       Field report from the named agent, in voice.
       Reflects the resolved state delta.
─── CYCLE CLOSE ──────────────────────────────────
```

Cycles consume exactly one unit from `cycles_remaining`. There are no variable cycle costs per archetype — every authorization is one cycle. Multi-cycle consequences are modeled as authored beats with cycle-delta predicates ("when SIGINT_QUERY fired ≥ 2 cycles ago, surface the result"), not as variable-length actions.

A mission ends when a terminal beat fires (win or loss). Otherwise, the cycle loop continues until `cycles_remaining == 0` (timeout failure).

---

## 4. State Model

All game state is **qualities** — typed, namespaced fields with set/add/max/min effects gated by predicates. There is no other state primitive.

### Quality scopes

State is grouped for readability, but the engine sees a flat dictionary:

| Scope | Examples | Notes |
|---|---|---|
| Mission | `mission.extraction_stage`, `mission.cycles_remaining` | Authored per mission |
| Per-agent | `agent[LYNX].cover`, `agent[KESTREL].cover` | Engine-universal shape |
| Engine internals | `current_cycle`, `fired_beats[*]`, `fired_moments[*]` | Engine-managed |

### Universal qualities

Every mission inherits these without authoring:

- `mission.cycles_remaining: int` — turn budget. Decrements each cycle. At 0 → timeout failure.
- `agent[id].cover: int (0–10)` — exposure meter. **At 0 the agent is lost** (no separate alive flag — cover-zero IS the loss state).

### Mission-authored qualities

Each mission YAML declares its own qualities. For LYNX-Exfil:

- `mission.extraction_stage: int (0–3)` — the win track. Stages: 0 unstarted, 1 papers, 2 location, 3 exfil. At 3 → win terminal.

Other missions might declare different progress qualities (e.g., a recruit mission might use `mission.target_trust: int` and `mission.target_aware: bool`). The engine doesn't care about specific quality names; it cares about the predicates and effects authored over them.

### Visible-to-player state

The player sees three trackable elements at all times:

```
Cycle 4 of 6
LYNX     cover ▓▓▓▓▓▓░░░░  6/10
KESTREL  cover ▓▓▓▓▓▓▓░░░  7/10
Extraction stage  ●─●─○    2/3
```

Plus the inbox of past dispatches (re-readable). That's the entire UI surface for state. No heat meter, no hypothesis bars, no political capital track. Three numbers and an inbox. The pressure is felt through cover decay (driven by passive beats) and the cycle countdown.

---

## 5. Archetypes & Options

### Archetypes

An **archetype** is an authored action template owned by a mission. It has:

- An identifier (e.g., `MOVE_TO_SAFEHOUSE`, `RETRIEVE_PAPERS_VIA_AGENT`).
- A precondition predicate (when is this action valid?).
- An effect template (what state changes when this fires?).
- Optional parameters.

Example for LYNX-Exfil:

```yaml
archetypes:
  ACTIVATE_SLEEPER_FOR_PAPERS:
    when: mission.extraction_stage == 0
    effects:
      - mission.extraction_stage = 1
    params: []

  RETRIEVE_PAPERS_VIA_AGENT:
    when: mission.extraction_stage == 1 AND agent[$agent].cover > 0
    effects:
      - mission.extraction_stage = 2
      - agent[$agent].cover -= 2
    params: [agent]

  RUN_EXFIL:
    when: mission.extraction_stage == 2 AND agent[LYNX].cover > 0
    effects:
      - mission.extraction_stage = 3
      - agent[LYNX].cover -= 3
    params: []

  DIVERSION_BY_AGENT:
    when: agent[$agent].cover > 1
    effects:
      - agent[$agent].cover -= 3
      - agent[LYNX].cover += 1   # protected this cycle
    params: [agent]

  HOLD:
    when: true
    effects:
      - cover_decay_pressure += 1
    params: []
```

Effects can carry simple `when:` branches for conditional outcomes:

```yaml
RUN_EXFIL:
  effects:
    - when: agent[LYNX].cover >= 5
      apply: { mission.extraction_stage = 3, agent[LYNX].cover -= 2 }
    - when: agent[LYNX].cover < 5
      apply: { mission.extraction_stage = 3, agent[LYNX].cover -= 4 }
```

### Options

An **option** is what the player chooses among each cycle. Each option carries:

```
Option:
  archetype: <archetype id>
  agent: <agent id, or null for HOLD>
  params: { ... }
  title: "<short headline>"
  prose: "<the paragraph the player reads>"
```

Options are produced by the drafter from the catalog of currently-valid archetypes. The `archetype`, `agent`, and `params` fields drive the engine's deterministic resolution. The `title` and `prose` are display only — they never influence state.

Generation order per cycle:

1. Engine evaluates `when:` predicates over current state, computes valid (archetype, agent) combinations.
2. Engine checks for forced moments (see §6) — if any are triggered, one option slot is reserved.
3. Drafter LLM is called with: state, valid archetypes, agent dossiers, the SoP memo, and any forced moment template. Output: 3–4 options with tags + prose.
4. Engine validates: tags reference valid archetypes; agents are valid for the params; if a moment was forced, the moment-derived option is in the set.
5. Player sees the options.

---

## 6. Moments (Branch Templates)

A **moment** is an authored option-shape that the engine forces the drafter to include when its trigger predicate fires. Moments express dramatic story beats with permanent consequence — the kind of decision Walking Dead frames as "Carley or Doug" — without authoring literal prose.

### Structure

Moments are stage directions, not example text. The author specifies the dramatic *function*; the LLM writes the situated prose at fire-time.

```yaml
moments:
  - id: kestrel_sacrifice
    when: agent[KESTREL].cover <= 5
       AND mission.extraction_stage <= 2
       AND mission.cycles_remaining <= 2
       AND NOT fired_moments includes "kestrel_sacrifice"
    fires_at_most: once_per_mission

    effects:
      - agent[KESTREL].cover = 0          # lost
      - mission.extraction_stage += 1
      - agent[LYNX].cover_protected = true # one-cycle flag

    intent: |
      KESTREL deliberately sacrifices her remaining cover so LYNX
      has clean lanes for the next stage of extraction. Permanent
      loss. The player is consciously trading one agent for the other.

    must_communicate:
      - The decision is KESTREL's deliberate choice, not accidental
      - The cost is permanent and irreversible
      - LYNX gains a tangible safety advantage

    must_not:
      - Predict whether LYNX successfully escapes
      - Frame KESTREL as already captured (she is choosing to be)
      - Specify tactical details that may not fit current state

    tone: solemn, deliberate, present-tense
```

### How a moment becomes an option

When a moment's `when:` predicate fires, the engine instructs the drafter: *"This cycle, one of your options must satisfy moment template `kestrel_sacrifice`. Adapt prose to current state. The mechanical effect is fixed."*

The drafter writes prose situated in the actual world (which city, which surveillance situation, which cover state) while honoring the constraints. Two example fires producing different prose:

> *State A* (KESTREL near the eastern train station, heat-equivalent high):
> "iv. KESTREL allows counter-intel surveillance to lock onto her position at the eastern safehouse — making herself the visible target while LYNX clears the train terminal. Once exposed, she will not return."

> *State B* (KESTREL in the southern district, brush-pass scenario):
> "iv. KESTREL takes deliberate steps through the southern district that draw counter-intel attention away from LYNX's brush-pass timing. The exposure she invites is one she will not survive."

Both honor: deliberate choice, permanent cost, LYNX gains, no outcome prediction, no captured framing. Different state, different prose, same moment semantically.

### Moments vs archetypes vs beats

Three distinct primitives, three roles:

| Primitive | Who decides it fires | Player choice? |
|---|---|---|
| Archetype | Drafter selects from valid set | Yes — one of several drafted options |
| Moment | Engine forces drafter to include | Yes — but the option is engine-mandated to appear |
| Beat | Engine fires on predicate | No — beats are passive engine events |

Archetypes are the dynamic moment-to-moment menu. Moments are the rare authored decision points. Beats are the silent reactions and terminals.

For MVP: **one moment** authored — `kestrel_sacrifice`.

---

## 7. Beats

A **beat** is a predicate-gated engine event with no player input. Beats serve three roles in the slim engine:

### Terminals

Beats that end the mission:

```yaml
beats:
  - id: extraction_complete
    when: mission.extraction_stage == 3 AND agent[LYNX].cover > 0
    effects:
      - mission.outcome = "win"
    terminal: true

  - id: lynx_lost
    when: agent[LYNX].cover == 0
    effects:
      - mission.outcome = "loss_lynx_captured"
    terminal: true

  - id: timeout
    when: mission.cycles_remaining == 0 AND mission.outcome IS NULL
    effects:
      - mission.outcome = "loss_timeout"
    terminal: true
```

When a terminal beat fires, the engine renders one final dispatch (in the style appropriate to the outcome) then transitions to the outcome card.

### Reaction effects

Beats that respond to state changes without ending the mission:

```yaml
beats:
  - id: agent_lost_when_cover_zero
    when: agent[$any].cover == 0
    effects:
      - agent[$any].lost = true
      - fired_beats append "agent_lost:$any"
    fires_at_most: once_per_agent
```

### Passive pressure

Beats that decay state over time, replacing the explicit heat meter:

```yaml
beats:
  - id: world_attention_tightens
    when: every_cycle AND mission.extraction_stage < 3
    effects:
      - agent[KESTREL].cover -= 1   # only if mission isn't progressing
      - agent[LYNX].cover -= 1
    only_when_no_progress_this_cycle: true
```

Passive pressure beats are how "doing nothing" becomes a real cost. The HOLD archetype doesn't progress the mission; passive beats fire; covers tick down. The player feels the pressure through cover decay rather than a separate heat number.

### What's gone from Smallville's beat system

- No Director picking one beat per tick. Beats just fire when their predicates are satisfied.
- No `scene` vs `one_shot` distinction. All beats are predicate-gated; `fires_at_most` controls re-firing.
- No beat-as-action concept. Beats are engine events, not player choices.
- No `find_due_beats` / `_select_beat` / scheduler. Just predicate evaluation each cycle.

---

## 8. LLM Contracts

Three LLM agents per cycle. Each is a separate call with a strict input/output contract.

### 8.1 SoP Synthesizer (Operations voice)

**Purpose:** synthesize the current situation into a forward-looking memo at cycle open.

**Input:**
- Narrative state block (cycles, agent covers, mission progress — translated to in-fiction descriptors; never raw numerics)
- Recent dispatches (last 2–3, for narrative continuity)
- Mission metadata (mission name, framing, setting, per-agent situation)

**Output:** 1–3 paragraphs of prose in Operations Desk voice. Looks forward. Names the threat, the resources, what hasn't moved.

**Constraints:**
- Voice: third-party intelligence handler, terse, declarative
- No options proposed (that's the drafter's job)
- No outcome prediction
- MUST reference state narratively, not numerically. No "cover at 7", no "six cycles remaining", no "extraction stage two". No game/system vocabulary ("cover level", "cycles remaining", "extraction stage", "operational security points"). The user-prompt CURRENT STATE block is pre-rendered into in-fiction descriptors via `Mission.quality_descriptors` (e.g. "LYNX: cover intact", "the extraction papers have not yet been surfaced", "the window is tightening"); the memo mirrors that register.

### 8.2 Drafter (Operations voice)

**Purpose:** produce 3–4 candidate options for this cycle.

**Input:**
- Full state
- Catalog of currently-valid (archetype, agent) combinations (engine-computed)
- Agent dossiers
- The SoP memo from this cycle
- Any forced moment template (with `intent`, `must_communicate`, `must_not`, `tone`)

**Output:** 3–4 options, each with structured tags + prose paragraph:

```json
{
  "options": [
    {
      "archetype": "RETRIEVE_PAPERS_VIA_AGENT",
      "agent": "KESTREL",
      "params": {},
      "title": "Have KESTREL fetch the papers",
      "prose": "..."
    },
    ...
  ]
}
```

**Constraints:**
- Tags MUST reference valid archetypes from the input catalog
- Agent MUST be valid for the archetype's params
- If a moment is forced, one option MUST satisfy that moment template
- Prose MUST telegraph the option's shape honestly: action, agent involved, cost, risk
- Prose MUST NOT predict the outcome ("LYNX will escape")
- Prose MUST NOT contradict current state
- Option set should span tonal range: at least one safe-but-slow, one fast-but-risky

### 8.3 Dispatch Renderer (Acting agent voice)

**Purpose:** render the field dispatch in the acting agent's voice after effects have applied.

**Input:**
- The authorized option (archetype, agent, params, prose)
- The state delta produced by resolution (which qualities changed and by how much)
- Acting agent's dossier (voice, register, personality)
- Recent dispatch context (last 1–2 dispatches for continuity — prefer prior dispatches from the same acting agent when available, for voice anchoring)

**Output:** 1–4 paragraphs of prose in the acting agent's voice.

**Constraints:**
- Voice: the acting agent's authored register
- MUST honor the resolved state delta (if cover dropped 3, the prose reflects what cost that)
- MUST NOT introduce new facts inconsistent with state
- MAY include first-person justification, color, character
- For HOLD: dispatch is in Operations voice (no acting agent), describing what counter-intel did during the holding cycle

### 8.4 Stateless contract — no message_history

**All three agents are stateless one-shot calls.** No pydantic-ai `message_history` is passed between cycles. Each cycle is a fresh prompt; nothing leaks across calls implicitly.

The decision rests on four concrete properties of the system:

1. **State mutates between calls.** The engine applies archetype effects and beat effects between LLM calls. If a prior dispatch (as an assistant turn) framed KESTREL as "composed," and the engine then drops her cover to 3/10 before the next renderer call, multi-turn forces the model to reconcile its own prior framing against the new state. LLMs weight content in their own assistant turns differently from user-supplied content — the prior framing acts as a sticky conclusion the model owes consistency to. One-shot eliminates the reconciliation: current state is the only authoritative input; recent dispatches are passed as user-side context, which the model treats as facts to integrate rather than positions to defend.

2. **The three jobs are semantically discrete.** SoP synthesizes current state. Drafter enumerates valid options. Renderer narrates a state delta in a specific agent's voice. None is a "response to" the prior call. Multi-turn implies a thread of reasoning that doesn't exist between these tasks. (Smallville's chat agent needed rolling history because it was an actual conversation with the player; none of Handler's agents are.)

3. **Voice multiplexing on the renderer.** The renderer's voice changes per cycle — LYNX, KESTREL, or Operations (for HOLD). A unified message history would be "you said X as LYNX, you said Y as KESTREL, now say Z as LYNX" — incoherent as a conversation. Each render is a discrete prose task in a specific voice.

4. **Bounded inputs.** One-shot input size is bounded by current state plus a small fixed history window. No compaction module, no token-budget spiral, no rolling-history processor to maintain.

**Continuity across cycles** (voice consistency, narrative coherence) is preserved by **explicitly passing context in the prompt**: the last 1–2 dispatches as `recent_dispatches` (with same-agent preference for the renderer, per §8.3), the agent dossier, the current state. We control exactly what the LLM sees.

**Voice consistency caveat.** The strongest empirical argument for multi-turn would be voice anchoring for the renderer when rendering the same agent across cycles — models do anchor on their own prior voice when given assistant-turn history. We accept the one-shot cost here and mitigate with (a) detailed voice register in the dossier, (b) same-agent recent dispatches in user-side context. If playtest shows voice drift that prompt engineering can't fix, a targeted hybrid is available: pass prior same-agent dispatches as faux-assistant turns *to the renderer only*. That is few-shot anchoring, not a real conversation, and SoP + drafter remain one-shot. Do not adopt the hybrid speculatively — only if observed drift demands it.

**Prompt structure for caching.** Each agent's prompt is built as `[stable prefix] + [dynamic suffix]`:
- *Prefix:* agent role, voice register, dossiers, mission framing, valid-archetype catalog template — everything that does not change within a mission.
- *Suffix:* current state, recent dispatches, this-cycle context (the SoP memo for the drafter, the resolved option + delta for the renderer).

This structure is cache-friendly for any LLM with prefix-prompt caching and reduces prompt-processing time on local llama via KV-cache hits on the stable prefix. Concatenating everything into one flat prompt forfeits both. Implementers must preserve this split.

**Practical consequences:**
- No compaction module to write or maintain.
- Input size bounded by current state + fixed history window.
- Reproducible outputs: same state in, same prompt out (modulo sampling temperature).
- Easier debugging: when a dispatch reads wrong, the prompt that produced it is one self-contained artifact.

In pydantic-ai terms: each agent is invoked as `agent.run(prompt)` with no `message_history` argument. Each agent is a stateless prompt template plus a structured output schema.

### 8.5 Engine/LLM responsibility split

Independent of the one-shot decision in §8.4, the architecture commits to a clean split: **the engine decides outcomes; the LLM writes prose.** Outcomes are computed deterministically from option tags by the engine via `apply_archetype_effects`, `apply_moment_effects`, and `apply_beat_effects` (§12). The LLM never mutates state and never decides whether a beat fires, whether a moment triggers, or what an effect resolves to.

Consequences across the design:
- Moments (§6) carry fixed mechanical effects; the LLM writes situated prose around the effect, never the effect itself.
- Beats (§7) fire on engine predicate evaluation, not LLM judgment.
- Renderer dispatches (§8.3) must honor the resolved state delta — a dispatch that contradicts the delta is a prompt bug, not an alternate outcome.

This is a property of the *system*, not of any single agent's prompt structure. It would hold under multi-turn just as it holds under one-shot; §8.4 is justified on its own mechanical grounds, not on this principle.

---

## 9. UI Surface

The frontend displays four regions:

```
┌─ STATE ────────────────────────────────────────┐
│ Cycle 4 of 6                                   │
│ LYNX     cover ▓▓▓▓▓▓░░░░  6/10                │
│ KESTREL  cover ▓▓▓▓▓▓▓░░░  7/10                │
│ Extraction stage  ●─●─○    2/3                 │
└────────────────────────────────────────────────┘

┌─ STATE OF PLAY (cycle open) ───────────────────┐
│ [LLM-rendered Operations memo, this cycle]     │
└────────────────────────────────────────────────┘

┌─ OPTIONS ──────────────────────────────────────┐
│ i.   [option 1 prose]                          │
│ ii.  [option 2 prose]   ← player highlights    │
│ iii. [option 3 prose]                          │
│ iv.  [option 4 prose]                          │
│                              [ Authorize ▸ ]   │
└────────────────────────────────────────────────┘

┌─ INBOX (past dispatches, scrollable) ──────────┐
│ Cycle 3 — KESTREL — "..."                      │
│ Cycle 2 — KESTREL — "..."                      │
│ Cycle 1 — Operations (sleeper activation) — "."│
└────────────────────────────────────────────────┘
```

On terminal: the final dispatch appears, then transitions to an outcome card:

```
┌─ MISSION RESOLVED ─────────────────────────────┐
│  Op. LYNX-Exfil — SUCCESS                      │
│                                                │
│  LYNX safely extracted via the eastern train   │
│  route on cycle 5 of 6. KESTREL was lost in    │
│  the cycle 3 sacrifice.                        │
│                                                │
│  Cycles used:   5 / 6                          │
│  Agents lost:   KESTREL                        │
│  Agents alive:  LYNX                           │
│                                                │
│  [ Restart Mission ]   [ Quit ]                │
└────────────────────────────────────────────────┘
```

Visual aesthetic carries from the v.0.2 mockup: paper textures, classified stamps, Newsreader serif + Courier monospace. Mockup file `_mockups/handler-authorize.html` is the visual reference.

---

## 10. The MVP Mission — Op. LYNX-Exfil

### Setup (player-facing intro)

> LYNX has been deep cover in the city for two years. Forty-eight hours ago she was photographed at a meeting that should not have happened. Counter-intel hasn't moved on her yet but the sweep is tightening. KESTREL is in the same city, two days into a separate reconnaissance op. You have six cycles to get LYNX out.

### Roster

- **LYNX** (file 19-X) — deep-cover, partially blown. Voice register: terse, observational, cautious. Cover starts 6/10.
- **KESTREL** (file 41-K) — active operative, two years in service, improvises. Voice register: elaborate, justifying, present-tense. Cover starts 9/10.

### Qualities

```yaml
qualities:
  cycles_remaining: 6           # universal
  agent[LYNX].cover: 6          # universal
  agent[KESTREL].cover: 9       # universal
  mission.extraction_stage: 0   # mission-authored, 0–3
  mission.outcome: null         # set by terminal beat
```

### Archetypes (~7)

```
ACTIVATE_SLEEPER_FOR_PAPERS    valid: stage 0
RETRIEVE_PAPERS_VIA_AGENT      valid: stage 1, requires agent
LYNX_MOVES_TO_SAFEHOUSE        valid: stage 1 or 2
RUN_EXFIL                      valid: stage 2
DIVERSION_BY_AGENT             valid: any stage, requires agent
HOLD                           valid: always
DEAD_DROP_PAPERS               valid: stage 1 (alternative to RETRIEVE)
```

### Moments (1)

```
kestrel_sacrifice              when: KESTREL.cover ≤ 5 AND
                                     extraction ≤ 2 AND
                                     cycles ≤ 2
```

### Beats

```
extraction_complete            terminal, win
lynx_lost                      terminal, loss
timeout                        terminal, loss
agent_lost_when_cover_zero     reaction, marks lost
world_attention_tightens       passive, decays cover when no progress
```

### Authoring summary

- 1 mission YAML
- ~7 archetypes
- 1 moment template
- 5 beats
- 2 agent dossiers
- ~3 LLM prompt templates (SoP synthesizer, drafter, renderer)

This is a small content surface. Authoring time: 1–2 days for tuned content; rebalancing iterations expected during playtest.

---

## 11. MVP Scope and Validation

### What MVP ships

- One mission: Op. LYNX-Exfil
- Two agents: LYNX, KESTREL
- Six cycles maximum
- One authored moment
- Three LLM calls per cycle (SoP, drafter, renderer)
- Frontend: state panel, SoP memo, options panel, inbox, outcome card
- Backend: engine + scenario + LLM agents wired through llama-server
- No persistence across missions (single-mission MVP — persistence model designed for, not exercised)

### What MVP defers

- Per-agent multipliers (validated in spec, not exercised mechanically)
- Multiple missions / campaign mode
- Mission-to-mission persistence (lost agents staying lost across missions)
- Replay-with-different-truth (each replay should at minimum reseed where genuinely random; in MVP, deterministic seeds may be acceptable)
- Static analysis pipeline (ASP encoder rebuild for Handler scenarios)
- Settings, save/load, accessibility passes

### Validation criteria

The verb works at MVP if:

1. A test player completes the mission without instruction.
2. The player can articulate, after one playthrough, what each option's tradeoffs were.
3. The State of Play memo correctly synthesizes the case at every cycle (no contradictions, no repetitions of dispatch text verbatim).
4. The dispatch reflects the resolved state delta in voice.
5. The option set tonally varies cycle to cycle (drafter doesn't produce 3 equivalent options).
6. The sacrifice moment fires when its predicate is met, surfaces in the option menu, and produces situation-aware prose.
7. The mission resolves cleanly on terminal — final dispatch + outcome card render correctly.
8. The player wants to replay (signal of engagement) OR can articulate why they don't (signal of design issue).

The verb fails if:

- The player says "I had no idea what to pick"
- The player says "the prose didn't match what happened"
- The drafter produces options the engine rejects (tag invalid)
- Same-state replays produce indistinguishable option sets

---

## 12. Engine Architecture

### Pattern

State is **qualities + predicates + effects + slim beats + branch templates**. Three primitives for "things that happen": archetypes (player-driven action templates), moments (authored decision points), beats (engine reactions and terminals). One verb (authorize). One LLM contract per phase (SoP, draft, render).

The engine never decides outcomes via LLM. The LLM never writes to state. State writes happen only through:

1. `apply_archetype_effects(option)` at authorize time.
2. `apply_moment_effects(option)` at authorize time, for option that satisfied a moment.
3. `apply_beat_effects(beat)` when a beat's predicate becomes true.
4. `apply_universal_tick()` to decrement `cycles_remaining`.

### What's kept from Smallville

| Smallville component | Status in Handler |
|---|---|
| Pydantic schema model (models.py) | Kept as pattern; rebuilt for Handler shapes |
| Predicate evaluation (evaluate_predicate) | Kept; same algebra |
| Effect application (apply_literal_effects) | Kept; same primitive |
| ASP encoder pattern | Pattern kept; encoder rewritten for Handler scenarios (deferred past MVP) |
| FastAPI + SSE streaming | Kept |
| llama-server integration via launcher.py | Kept |
| pydantic-ai for agent definitions | Kept |
| Static analysis drift suite (test_encoder_engine_consistency) | Pattern kept; rewritten when encoder is rewritten |

### What's retired

| Smallville component | Replaced by |
|---|---|
| Chat agent + history compaction processor | Three new LLM agents (SoP, drafter, renderer) — none are conversational |
| Extractor agent + `state.player_canon` | None — no free-text input means no player canon |
| Classifier agent | None — outcomes are deterministic from option tags |
| Summary agent | SoP synthesizer (different role, narrower contract) |
| Fact-checker agent | None — facts are not a state primitive |
| Director (LLM beat selection) | None — player chooses from drafter options |
| Beat scheduler / `find_due_beats` / `_select_beat` | Predicate evaluation each cycle; multiple beats can fire |
| `scene` vs `one_shot` distinction | All beats are predicate-gated; `fires_at_most` controls re-firing |
| `TickDecision` audit trail | Replaced with simpler per-cycle audit (option chosen, effects applied, beats fired) |
| Continuity Program scenario.yaml content | Deprecated; LYNX-Exfil is the new authored mission |

### What's rebuilt

- **`spine.py` → engine module.** New tick loop: SoP → draft → authorize → effects + beats → render → cycle++.
- **`agents.py` → three agents.** SoP synthesizer, drafter, dispatch renderer. Each is a pydantic-ai agent with a clear input/output contract.
- **`models.py` → Handler schemas.** Mission, Agent dossier, Archetype, Moment, Beat, Option, State, etc.
- **`scenario.yaml` → mission YAMLs.** First file: `missions/lynx_exfil.yaml`.
- **Frontend.** Drop chat UI. New: state panel, SoP block, options panel, inbox, outcome card. Tailwind + same React + Vite setup.

---

## 13. Tech Stack

Same as Smallville:

- **Backend:** Python 3.13, FastAPI, Pydantic v2, pydantic-ai for LLM agents, SSE streaming.
- **LLM:** local llama-server, Gemma-class model. Reuse the Smallville launcher (`npc_service/src/npc_service/launcher.py`) — `--jinja`, `--flash-attn on`, fp16 KV cache, the documented invariants stay.
- **Frontend:** React + TypeScript + Vite. Reuse the existing build pipeline.
- **Tests:** pytest. Three test layers: unit (engine primitives), integration (live llama, end-to-end mission), drift (mission YAML + engine consistency, deferred until ASP encoder rewritten).
- **Repo layout:** rename `npc_service` to `handler` over time, or keep the name and pivot the contents. Decided at implementation start.

---

## 14. Open Questions for Future Work

These are deliberately deferred past MVP. Capturing here so they aren't lost.

1. **Persistence across missions.** Schema supports `agent[X].lost: bool` from day one but no missions chain yet. Campaign mode is post-MVP.
2. **Per-agent multipliers when N≥2 starts to matter mechanically.** MVP has N=2 but no multiplier authoring; agents differentiate through prose voice + which archetypes the drafter proposes for them. When the player can't tell KESTREL apart from LYNX in dispatches, that's the signal to introduce multipliers.
3. **ASP encoder rebuild.** The static reachability analysis pattern is valuable — catches authoring errors. Rewriting the encoder for Handler scenarios takes work; deferred until mission complexity warrants.
4. **Replay seeding.** If a player replays Op. LYNX-Exfil, does the drafter produce different option prose? (Yes — temperature.) Does the moment fire on different cycles? (Possibly — depends on player choices.) Is the mission's truth ever stochastic? (Not in MVP — single authored truth.) Stochastic mission truth is post-MVP.
5. **Save/load mid-mission.** Player closing the browser mid-mission loses progress in MVP. Add session persistence post-MVP.
6. **Multiple missions, campaign frame.** Section Chief patience, agent persistence, mission selection — all deferred. MVP is one mission, no campaign wrapper.
7. **Casualty mechanic richness.** "Lost" is binary in MVP. Could later be "captured" (rescuable in future mission) vs "killed" vs "burned" (out of cover but alive). Authoring depth.
8. **Reliability rolls / agent unreliability.** The Model 2 framing where KESTREL might not exactly do what you authorized. Powerful but balance-heavy. Post-MVP.
9. **More archetype primitives across mission types.** MVP's archetypes are LYNX-Exfil-specific. A mission template / shared archetype library could emerge after authoring 3–4 missions.

---

## 15. Open Architecture Questions

These are implementation-level questions that don't change the design but need answering before code is written:

1. **State persistence representation.** SQLite, in-memory, JSON file? MVP can be in-memory per-session.
2. **Drafter retry on schema violation.** If the drafter outputs an option with an invalid archetype tag, how many retries before falling back to a hand-authored "default" option set? Probably 1 retry, then fall back.
3. **SoP memo on cycle 1.** Does the very first SoP memo render before any dispatches exist? (Yes — it should set up the mission's opening situation.)
4. **End-screen dispatch voice.** On a loss-by-timeout, who writes the final dispatch? Operations Desk, probably ("the operation has been closed without resolution").
5. **Beat firing order within a cycle.** If two beats become satisfied by the same option's effects, what order do they fire? Authored priority field on each beat, or deterministic by ID order. Doesn't matter much for MVP.

---

## Appendix A — Sample full cycle trace

For verification: a single cycle from cycle 3 of LYNX-Exfil, after the player has authorized cycle 1 (papers activation) and cycle 2 (KESTREL retrieves papers).

**State at cycle open:**
```
cycles_remaining: 4
agent[LYNX].cover: 6
agent[KESTREL].cover: 7
mission.extraction_stage: 2
fired_moments: []
```

**Engine evaluates valid archetypes:**
- `LYNX_MOVES_TO_SAFEHOUSE` (stage 2 ✓)
- `RUN_EXFIL` (stage 2 ✓)
- `DIVERSION_BY_AGENT(KESTREL)` (cover 7 > 1 ✓)
- `DIVERSION_BY_AGENT(LYNX)` (cover 6 > 1 ✓)
- `HOLD` (always ✓)

**Engine checks moments:** `kestrel_sacrifice` predicate: cover ≤ 5? No (it's 7). Moment does not fire.

**SoP synthesizer call:** outputs ~150 words in Operations voice describing current situation.

**Drafter call:** receives state + valid archetypes + dossiers + SoP memo. Outputs 4 options:

```
i. Have LYNX move to the tertiary safehouse alone.
   (LYNX_MOVES_TO_SAFEHOUSE)
ii. Run the eastern train extraction now. Direct.
    (RUN_EXFIL)
iii. KESTREL creates a diversion in district 7 to clear LYNX's lanes.
     (DIVERSION_BY_AGENT, agent=KESTREL)
iv. Hold position. Wait for the sweep to pass.
    (HOLD)
```

**Player picks (iii).**

**Engine applies DIVERSION_BY_AGENT(KESTREL) effects:**
- agent[KESTREL].cover: 7 → 4
- agent[LYNX].cover_protected: false → true (one-cycle flag)

**Engine evaluates beats after effects:**
- `extraction_complete`: stage == 3? No.
- `lynx_lost`: LYNX cover == 0? No.
- `agent_lost_when_cover_zero`: any agent cover == 0? No.
- `world_attention_tightens`: extraction < 3 AND no progress? Stage didn't change. Yes — fires.
  - LYNX.cover: 6 → 5
  - KESTREL.cover: 4 → 3
- `kestrel_sacrifice`: cover ≤ 5 AND stage ≤ 2 AND cycles ≤ 2? cycles is 4 — No.

**Wait** — that doesn't feel right. KESTREL just lost 4 cover in one cycle (3 from diversion + 1 from passive). Two issues:
- Passive pressure shouldn't fire after an action that DID move things forward (diversion isn't main-track progress, but it's a meaningful action)
- Or the passive pressure is too aggressive

This is the kind of balance question that surfaces during playtest. The spec captures the *primitives*; balance numbers are tuning that happens during implementation.

**State at cycle close:**
```
cycles_remaining: 3 (decremented at cycle close)
agent[LYNX].cover: 5
agent[KESTREL].cover: 3
mission.extraction_stage: 2
```

**Renderer call:** receives the option (DIVERSION_BY_AGENT, KESTREL), state delta (KESTREL -3, LYNX +1 then -1, etc), KESTREL's dossier. Outputs ~150 words in KESTREL's voice describing the diversion and its cost.

**Inbox updated. Cycle ends. Cycle 4 begins.**

---

## 16. Codebase Audit

Translates §12's conceptual decisions to file-level, classifying each file as **KEEP** (use as-is), **MODIFY** (rewrite while preserving the file), or **RETIRE** (delete in Phase 0).

---

### 16.1 Backend modules (`npc_service/src/npc_service/*.py`)

| File | Purpose | Decision | Notes |
|---|---|---|---|
| models.py | Pydantic schemas: Quality, Beat, Thread, Scenario, Effect, InboxState. | MODIFY | Rebuild for Handler: Option, Moment, Archetype, Archive, TickLog. Preserve predicate/effect DSL. |
| spine.py | Predicate evaluation, state mutation, beat scheduling. | MODIFY | Keep evaluate_predicate, apply_literal_effects. Retire Director, beat-selector, scene/one_shot. Rebuild tick loop. |
| agents.py | Six LLM agents (CHAT, CLASSIFIER, EXTRACTOR, SUMMARY, FACT_CHECKER, DIRECTOR). | MODIFY | All six chat-specific; contents replaced wholesale with three new agents (SoP, drafter, renderer) in Phase 3. File path preserved. |
| main.py | FastAPI surface, endpoints, session init, streaming. | MODIFY | Keep FastAPI/SSE. Retire `/send`, add `/authorize`, `/sync`. Session structure unchanged. |
| session.py | Per-game state wrapper, event log, inbox state, lock. | KEEP | No change; will ingest new event types. |
| events.py | Event classes and fold function. | MODIFY | Retire chat-specific events. Add `OptionsDrafted`, `OptionAuthorized`, `BeatsEvaluated`, `DispatchRendered`. Keep `fold()`. |
| config.py | Settings: llama, logfire, endpoints. | KEEP | Reuse as-is. |
| launcher.py | llama-server launcher. | KEEP | Unchanged. |
| run_llama.py | Entrypoint to run llama-server standalone. | KEEP | Unchanged. |
| `__main__.py`, `__init__.py` | Package entry points. | KEEP | Minor renaming if `npc_service` → `handler`, no logic change. |

---

### 16.2 Backend tests (`npc_service/tests/*.py`)

| File | Purpose | Decision | Notes |
|---|---|---|---|
| test_spine.py | Unit tests for predicate eval, effect application, tick logic. | MODIFY | Keep tests for `evaluate_predicate`, `apply_literal_effects`. Retire beat-selection, Director, scene. Add new tick tests (SoP → draft → authorize → effects). |
| test_scenario.py | YAML loading and schema validation. | MODIFY | Keep pattern; test new mission YAML (archetypes, moments, beats). |
| test_events.py | Event shape and fold function. | MODIFY | Add tests for new event types. Retire chat-specific tests. Keep `fold()`. |
| test_app.py | FastAPI endpoint unit tests. | RETIRE | Tests old `/send` endpoint. Rewrite for `/authorize` and drafter. |
| test_app_integration.py | Full integration tests with live llama. | RETIRE | Tests chat cycle. Rewrite for Handler cycle (SoP → draft → authorize → effects → render). |
| test_app_cross_thread.py | Multi-thread inbox behavior. | RETIRE | Inbox-specific. No equivalent in Handler. |
| test_session_replay.py | Session replay from event log. | MODIFY | Keep event-replay pattern. Update test data to new event shapes. |
| test_launcher.py | llama-server launcher. | KEEP | Unchanged. |
| test_player_canon.py | Predicate eval and canon snapshot. | RETIRE | Canon-specific to Inbox. Handler has no free-text input. |
| test_history_compaction.py, test_history_compaction_integration.py | History-compaction for token budget. | RETIRE | Chat agent rolling history. Not needed in Handler. |
| test_director.py | Director beat-selection. | RETIRE | Director is gone. |
| test_voice_prompts.py | Voice/dossier prompt rendering. | RETIRE | Rewrite after agents rebuild. |
| test_analyze_scenario.py | Scenario linter integration. | RETIRE | Linter stays; test data changes. Rewrite after ASP encoder rebuild. |
| test_lint_scenario.py | Unit tests for lint rules. | MODIFY | Keep pattern. Update rules for Handler predicates. |
| conftest.py | pytest fixtures: `live_llama`, `consume_sse` helper, autouse stub for chat agent token counter. | MODIFY | Keep `live_llama` + `consume_sse`. Drop the `_stub_llama_token_counter` autouse hook (chat agent retired; no token counter to stub). |
| test_evaluate_all_beats.py | Reachability test for all beats. | RETIRE | Beat model changes. Rewrite when ASP encoder updated. |
| test_encoder_engine_consistency.py | Drift test: ASP and spine consistency. | RETIRE | Tied to current encoder. Rewrite after encoder rebuild (deferred past MVP). |

---

### 16.3 Test fixtures (`npc_service/tests/fixtures/*.yaml`)

| File | Purpose | Decision | Notes |
|---|---|---|---|
| test_scenario.yaml | Small synthetic Inbox scenario. | RETIRE | Replace with tiny Handler YAML (2-3 agents, 3-4 archetypes, 1-2 beats). |
| test_scenario_cross_thread.yaml | Multi-thread Inbox scenario. | RETIRE | Inbox-specific. No replacement. |

---

### 16.4 Backend scripts (`npc_service/scripts/*.py`)

| File | Purpose | Decision | Notes |
|---|---|---|---|
| asp_encoder.py | ASP encoding for reachability analysis. | RETIRE | Pattern valuable; code retired until Handler encoder rewritten. Deferred per §14.3. |
| analyze_scenario.py | Driver for ASP analysis. | RETIRE | Depends on `asp_encoder.py`. Deferred. |
| lint_scenario.py | Scenario linter (spec rules only). | MODIFY | Keep pattern. Update rules for Handler archetypes/moments/beats. |
| lint_rules.md | Linting guidelines. | MODIFY | Update for Handler-specific rules. |
| replay.py | Replay session from JSONL log. | KEEP | Pattern unchanged; event shapes change. |
| bench_kv.py, bench_kv_compare.py, bench_qwen_player.py | llama benchmarks. | KEEP | Unchanged. |
| playtest/`__init__.py`, playtest/`__main__.py` | Playtest harness entry. | MODIFY | Core structure stays. Rewrite player: options picking, not chat. |
| playtest/personas.py, playtest/personas/*.yaml | Synthetic player personas. | MODIFY | Personas reusable. Rewrite decision logic: heuristics for option picking. |
| playtest/llm_playtest.py | LLM-driven synthetic player. | RETIRE | Generates chat. Handler uses heuristic picker. |
| playtest/session.py | Playtest session management. | MODIFY | Calls change (`authorize`, not `send`). |
| playtest/evaluators.py | Eval suite. | MODIFY | Keep pattern. Rewrite checks for Handler outcomes/state/options. |
| playtest/report_io.py | Playtest result reporting. | KEEP | Unchanged. |
| playtest/README.md | Playtest documentation. | MODIFY | Update for option-picking playtest. |

---

### 16.5 Frontend (`web/src/**/*.*`)

#### Main / Entry

| File | Purpose | Decision | Notes |
|---|---|---|---|
| main.tsx | React entry point. | KEEP | Unchanged. |
| App.tsx | Root component layout. | RETIRE | Chat UI. Replace with Handler layout. |
| api.ts | API client (type-safe fetch). | MODIFY | Update endpoints: `/authorize`, `/sync` instead of `/send`. New types for Options/Moments. |
| store.ts | Zustand state store. | RETIRE | Chat-focused. Replace with mission-focused store. |

#### Components

| File | Purpose | Decision | Notes |
|---|---|---|---|
| components/ChatHeader.tsx | Chat header. | RETIRE | No per-NPC view. |
| components/ChatSidebar.tsx | NPC sidebar. | RETIRE | Replace with mission state panel. |
| components/ChatRow.tsx | Thread rows. | RETIRE | No thread model. |
| components/MessageComposer.tsx | Text input. | RETIRE | No free-text input. |
| components/MessageList.tsx | Message list. | RETIRE | Keep pattern for inbox (read-only). |
| components/MessageBubble.tsx | Message rendering. | RETIRE | Keep pattern for options/dispatches. |
| components/MessageGroup.tsx | Message grouping. | RETIRE | No grouping in Handler. |
| components/AppHeader.tsx | Top nav. | MODIFY | Keep structure. Update for mission/cycle display. |
| components/PresenceDot.tsx | NPC presence. | RETIRE | No presence model. |
| components/TypingIndicator.tsx | Typing indicator. | RETIRE | No typing in Handler. |
| components/Avatar.tsx | Avatar rendering. | KEEP | Reuse for agent avatars. |
| components/ErrorBoundary.tsx | Error boundary. | KEEP | Unchanged. |
| components/ThemeToggle.tsx | Theme toggle. | KEEP | Unchanged. |
| components/NewMessagePill.tsx | New message indicator. | RETIRE | No rolling messages. |

#### Hooks

| File | Purpose | Decision | Notes |
|---|---|---|---|
| hooks/useStreamSend.ts | SSE streaming for `/send`. | MODIFY | Repurpose for `/authorize` → `/sync` SSE. Pattern unchanged. |
| hooks/useAutoGrow.ts | Auto-growing textarea. | RETIRE | No textarea. |
| hooks/useScrollAnchor.ts | Scroll anchor. | MODIFY | Repurpose for inbox auto-scroll to new dispatch. |
| hooks/useTheme.ts | Theme toggle. | KEEP | Unchanged. |

#### Debug panels

| File | Purpose | Decision | Notes |
|---|---|---|---|
| debug/DebugDock.tsx | Debug overlay container. | MODIFY | Structure stays. Update tabs for Handler. |
| debug/DebugReopenHandle.tsx | Debug toggle handle. | KEEP | Unchanged. |
| debug/DebugPanels.tsx | Tab switching. | MODIFY | Update tab list for Handler panels. |
| debug/DebugPreview.tsx | Debug preview. | MODIFY | Update for Handler state/options. |
| debug/beats/BeatsPanel.tsx | Beat eligibility. | MODIFY | Update for archetype/moment/beat structure. |
| debug/beats/ScenarioGraph.tsx | Scenario graph. | MODIFY | Update for Handler logic. |
| debug/ticks/TickLogPanel.tsx | Per-tick beat log. | RETIRE | Director-specific. Replace with cycle option log. |
| debug/canon/CanonPanel.tsx | Canon snapshot. | RETIRE | No canon in Handler. |
| debug/pacing/PacingPanel.tsx | Message pacing. | RETIRE | Chat-specific. |
| debug/timeline/TimelinePanel.tsx | Session timeline. | MODIFY | Pattern stays. Update event types. |
| debug/shared/PredicateTree.tsx | Predicate visualization. | MODIFY | Keep pattern. Update for Handler predicates. |
| debug/shared/DebugTooltip.tsx | Debug tooltip. | KEEP | Unchanged. |
| debug/shared/Sparkline.tsx | Sparkline visualization. | KEEP | Unchanged. |
| debug/shared/Empty.tsx | Empty state. | KEEP | Unchanged. |
| debug/shared/labels.tsx | Label rendering. | MODIFY | Update for Handler predicate/quality labels. |
| debug/shared/types.ts | Debug type defs. | MODIFY | Update for Handler event/option/moment types. |
| debug/tabs.ts | Debug tab registry. | MODIFY | Pattern unchanged (one entry per tab). Update imports as panels are rebuilt. |
| debug/fixtures/_base.ts, all_blocked.ts, etc. | Debug mock data. | RETIRE | Replace with Handler mission fixtures. |

#### Utils

| File | Purpose | Decision | Notes |
|---|---|---|---|
| util/avatar.ts | Avatar generation. | KEEP | Reuse for agents. |
| util/email.ts | Email parsing. | RETIRE | Drop; no email surface. |
| util/pacing.ts | Message pacing. | RETIRE | Chat-specific. |

#### Build / Config

| File | Purpose | Decision | Notes |
|---|---|---|---|
| vite.config.ts | Vite config. | KEEP | Unchanged. |
| tsconfig.json | TypeScript config. | KEEP | Unchanged. |
| tailwind.config.ts | Tailwind config. | KEEP | Unchanged. |
| vite-env.d.ts | Vite types. | KEEP | Unchanged. |
| index.css | Tailwind directives + theme CSS variables. | KEEP | Theme palette stays; new components reuse the variables. |
| index.html | HTML template. | MODIFY | Update title to mission name. Keep structure. |

---

### 16.6 Scenario content

| File | Purpose | Decision | Notes |
|---|---|---|---|
| npc_service/scenario.yaml | Continuity Program Inbox scenario. | RETIRE | Inbox-specific. Replace with Op. LYNX-Exfil YAML per §10. |

---

### 16.7 Build and infrastructure

| File | Purpose | Decision | Notes |
|---|---|---|---|
| npc_service/pyproject.toml | Backend deps and build. | KEEP | Reuse. May add YAML schema validators. |
| Makefile | Dev targets. | MODIFY | Keep structure. Update descriptions (lint, analyze deferred). Add mission targets. |
| web/package.json | Frontend deps. | KEEP | Unchanged. |
| web/package-lock.json | Lockfile. | KEEP | Regenerate if needed. |

---

### 16.8 Documentation

| File | Purpose | Decision | Notes |
|---|---|---|---|
| README.md | Repo README. | MODIFY | Update for Handler architecture. Add LYNX-Exfil mission summary. |
| CLAUDE.md | Architecture invariants. | MODIFY | Update tick loop, three agents, effect application, beat eval. Keep session/event-sourcing. |
| docs/agent-patterns.md | pydantic-ai patterns. | KEEP | Reference for new agents. |
| docs/authoring-guide.md | Inbox authoring. | RETIRE | Specific to Inbox. Replace with Handler guide. |
| docs/frankenstein-adaptation.md | Game-jam notes. | RETIRE | Historical; no dependency. |
| docs/LLM-game-research.md | Research bibliography. | KEEP | Background reference. |
| HANDLER_SPEC.md | Early spec. | RETIRE | Superseded by V2. |
| SPEC.md | Inbox spec. | RETIRE | Archived. |
| npc_service/scripts/lint_rules.md | Lint documentation. | MODIFY | Update for Handler rules. |
| npc_service/scripts/playtest/README.md | Playtest docs. | MODIFY | Update for option-picking playtest. |

---

### 16.9 Phase 0 delete list (in dependency order)

Delete these before new work, no broken imports:

#### Frontend UI (retire chat entirely)
- web/src/debug/canon/CanonPanel.tsx
- web/src/debug/pacing/PacingPanel.tsx
- web/src/debug/ticks/TickLogPanel.tsx
- web/src/debug/fixtures/*
- web/src/util/pacing.ts
- web/src/util/email.ts
- web/src/components/MessageComposer.tsx
- web/src/components/MessageList.tsx
- web/src/components/MessageBubble.tsx
- web/src/components/MessageGroup.tsx
- web/src/components/ChatHeader.tsx
- web/src/components/ChatSidebar.tsx
- web/src/components/ChatRow.tsx
- web/src/components/PresenceDot.tsx
- web/src/components/TypingIndicator.tsx
- web/src/components/NewMessagePill.tsx
- web/src/hooks/useAutoGrow.ts
- web/src/store.ts
- web/src/App.tsx

#### Backend tests
- npc_service/tests/test_director.py
- npc_service/tests/test_player_canon.py
- npc_service/tests/test_history_compaction.py
- npc_service/tests/test_history_compaction_integration.py
- npc_service/tests/test_voice_prompts.py
- npc_service/tests/test_app.py
- npc_service/tests/test_app_integration.py
- npc_service/tests/test_app_cross_thread.py
- npc_service/tests/test_evaluate_all_beats.py
- npc_service/tests/test_encoder_engine_consistency.py
- npc_service/tests/test_analyze_scenario.py

#### Scripts (analyzer/encoder deferred; playtest rewritten)
- npc_service/scripts/asp_encoder.py
- npc_service/scripts/analyze_scenario.py
- npc_service/scripts/playtest/llm_playtest.py

#### Content and test fixtures (after LYNX-Exfil authored)
- npc_service/scenario.yaml
- npc_service/tests/fixtures/test_scenario.yaml
- npc_service/tests/fixtures/test_scenario_cross_thread.yaml

#### Documentation
- docs/authoring-guide.md
- docs/frankenstein-adaptation.md
- HANDLER_SPEC.md
- SPEC.md

---

### 16.10 Audit summary

**Totals:**
- **KEEP:** 32 files (session, event-fold, launcher, config, debug utils, build config, theme CSS, research docs)
- **MODIFY:** 49 files (spine, models, agents, events, main, conftest, debug tabs, playtest, frontend layout, lint, docs)
- **RETIRE:** 72 files (chat UI, chat tests, Inbox scenario, ASP analyzer/encoder deferred, Inbox guide)

**Survivors from Smallville:**
- Predicate evaluator + effect applier
- FastAPI + SSE streaming
- llama-server launcher + pydantic-ai pattern
- Event-sourcing + session

**Critical rebuilds (§12):**
1. Three new LLM agents in `agents.py` (SoP, drafter, renderer)
2. New tick loop in `spine.py` (SoP → draft → authorize → effects → render)
3. Handler models in `models.py` (Option, Moment, Archetype, Archive)
4. New frontend UI (state panel, options, inbox, outcome)
5. Mission YAML: Op. LYNX-Exfil

**Deferred (§14):**
- ASP encoder rebuild (after mission complexity warrants)
- Persistence model (schema ready; implementation later)
- Multiple missions / campaign mode
- Stochastic truth / replay seeding

**No surprises:** Clean separation of concerns. Engine primitives intact; chat layer deletes cleanly. Scope matches §12 exactly.

---

End of spec.
