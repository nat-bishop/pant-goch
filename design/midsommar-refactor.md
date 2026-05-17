# Midsommar refactor — design analysis

## What this is

An analysis of swapping the Pant Goch story for a folk-horror story modelled on Midsommar / The Wicker Man / The Third Day. The *engine* stays. The *world bible* is rewritten end-to-end. This document weighs whether the swap is worth doing, what fits and what doesn't, and what engine refinements either the swap requires or simply benefits from.

The engine refinements stand on their own — the "Engine refinements" section is useful to Pant Goch too, regardless of whether the Midsommar refactor ever ships.

Status: planning artefact, not a build plan. No code touched.

---

## The premise

You arrive at a remote commune in northern Sweden for a midsummer festival. You came because you are grieving — an authored loss — and Pelle, a young man who lives at the commune, invited you. You drove out with two anthropology grad students he also invited. The community is warm, communal, formal in its hospitality. The festival lasts nine days.

Over those nine days you learn the commune's customs, eat at their long tables, dance at their maypole, watch their oracle pronounce. You also start to notice things. The elders observe the Ättestupa — the cliff-suicide at age 72 — on the second day, and the community sings while they fall. Your fellow outsiders go missing one by one. The tea you are served at certain rituals has been dosed. By Day 7 you understand that the festival requires nine sacrifices, that four are outsiders, that you are one of them, and that the only way out is to become the May Queen — and even then the festival still needs its ninth.

The truth is fixed and the LLM is briefed on all of it. The player's investigation is not "what is happening" but "how late do I figure it out, and what do I do when I have."

---

## Why this story for this engine

Two arguments. Both are about *engine-model fit*, not just story preference.

### 1. The model's natural prose voice IS folk-horror's horror voice

Pant Goch's prose target is *taciturn, cold, Welsh-Marches reserved*: Old Tom Bevan saying nothing across a pub, the post office mistress lying by omission, the silence in a kitchen at Lower Hafod. Gemma 4 E4B at temp 1.0 does not naturally write this. The model is trained to be warm, explanatory, helpful; the current few-shots and `notes_to_llm` instructions are fighting the model's grain on every turn. The result is acceptable prose that has to be *coaxed* away from over-explaining and toward Welsh restraint.

Midsommar's prose target is *warm, communal, earnestly-explaining, with horrifying content*. Pelle gently telling you about the seasons. Siv translating the runes with the care of a librarian. The whole community singing as Dan falls. The horror does not live in the prose register; it lives in the *gap* between the warm voice and what is being described. This is exactly what Gemma 4 E4B writes naturally. Folk horror weaponises the model's instincts instead of fighting them.

This is the single biggest argument. It is not romantic — it is mechanical. Same engine, same model, same temperature, but the prose lands ~90% right instead of ~70% right because we have stopped asking the model to do something it does not want to do.

### 2. The engine's structural assumptions become *meaningful* instead of being worked around

- **Compact geography.** A commune is one place. Ten map locations cover everything important. Pant Goch had to invent the conceit that "Knighton" is one drive-able location representing all of the outside world.
- **Limited cast.** Nine named NPCs cover the whole community that matters to the player. Pant Goch's "village of 400" is a small lie; a commune of ~40 with nine named members is true.
- **Confinement makes the leave-pick mechanism load-bearing.** Pant Goch's player can drive to London any time; "leaving" is always available, so the leave option is just narrative furniture. A folk-horror commune player *cannot* leave — the forest path is gated by a decision beat — so the leave-pick becomes the climactic choice the engine is already designed to handle.
- **Co-present NPCs are losable.** Pant Goch has no in-game deaths. Folk horror needs companions who disappear one by one — and the engine refinements below make this clean to author.
- **Endings are visceral, not professional.** Pant Goch's endings turn on journalistic ethics; folk-horror endings turn on bodily survival and complicity. Larger dramatic stakes for a wider audience.

---

## Where the fit is *worse* than Pant Goch

Honesty: real engine challenges. None are dealbreakers; each requires authoring or engine discipline.

### Time and ritual pacing

Midsommar happens over nine specific days, each anchored to a specific ritual. The engine has no time mechanism (SPEC §3.5).

Pant Goch handles pacing implicitly: the player draws out beats through accumulated patience. Pacing is *pull*.

Midsommar pacing is partly *push* — the village pushes events at the player on a schedule. The Ättestupa happens on Day 2 whether or not the player "earned" it.

Three options:

1. **Authored beat-chain.** `day_2_ättestupa` has no `requires_beats` but is gated by `notes_to_llm` ("fire when the player has had at least one full scene and a meal has been described"). `day_3_morning` requires `day_2_ättestupa`. The LLM's judgment carries the pacing. *Path of least resistance, fragile to rushing or stalling.*
2. **Mechanical scene floor.** Extend the v1.1 `min_scene_turns` / `min_co_presence` floors to `min_scenes_since_start` and pin each day's beat to a count. *Less elegant but quantitatively reliable.*
3. **Location-as-day.** Each ritual happens at a specific location; entering triggers the day's beat. The village pressures the player toward that location. *Cleanest if the geography supports it; risks on-rails feeling.*

Production would blend (1) and (2): authored chain with a soft scene-count floor as backstop.

### Information flow is the opposite of mystery

In Pant Goch, NPCs *guard* knowledge. In Midsommar, NPCs *teach* willingly. The information flow is open, not guarded.

Beats become different in kind: instead of "Tom finally tells you what he saw," it is "Pelle, in the right moment, explains the 90-year cycle and you understand what is being asked of the elders." Comprehension, not testimony.

The *negative beats* mechanism is weaker here. In Pant Goch, you can lose Tom forever by pressing him too hard. In Midsommar, individual NPCs are patient with you because their job is to keep you here. The negative-beat machinery needs a different shape: *the commune turns on you* (community-level threshold beat) rather than *individuals close off*.

### The player is partly a passenger

Midsommar's player is partly being *conducted through* a festival — the dance happens, the meal is served, you choose to participate or refuse but the events occur. Options in many turns are *participate / observe / refuse / escalate* rather than *say X / ask Y*. The few-shots have to demonstrate this register.

Compensation: when *decision beats* fire ("I am leaving tonight," "Crown me"), they carry more weight per choice than any Pant Goch option pick.

### Group choreography

Commune rituals involve everyone at once. The engine's `[PRESENT]` block already handles this; the prose has to coordinate four+ NPCs without losing the voice. Authoring discipline problem, not engine.

### The drugged-perception register

Midsommar's protagonist is dosed multiple times. The prose register should shift after a `dosed_*` beat fires. The engine constraint: Phase 2 does not see `state.unlocked_beats` directly; it sees the just-applied changes and the in-scene conversation history.

Cheapest path: rely on the scene-bounded conversation history (once a turn establishes the dosed register, subsequent turns continue by inertia) plus few-shots demonstrating the register. If that fails in playtest, add a small `[ONGOING]` block to Phase 2's user prompt naming current persistent states.

---

## The captivation question

The honest answer: **yes, Midsommar is more captivating, for the audience that will actually play this game.**

Pant Goch's captivation is intellectual and moral. The audience that loves Reservoir 13, that watches Kelly Reichardt, that reads Tessa Hadley — they will love it. The audience that finds itself idly checking their phone three scenes in will not.

Midsommar's captivation is visceral and emotional. The dread is immediate and culturally available. The seduction-and-horror duality creates moment-to-moment tension that does not depend on prior literary love. Stakes are bodily, not professional. And the aesthetic — sun-bleached folk horror — is, right now, one of the most alive aesthetics in indie culture.

This game lives in the indie-narrative-game space (Disco Elysium, Hereditary, the Magnus Archives), not the lit-fic-novel space. Folk horror wins on captivation for the likely audience by a large margin.

Subtler point: Pant Goch's *successful* playthrough requires a contemplative player who will sit through ten patient scenes at the pub. The engine cannot force that patience — a rushing player gets `tom_closes_off` in twenty minutes and loses. Midsommar's structure tolerates a wider range of player temperaments. A rushing player gets caught faster, witnesses the Ättestupa earlier, makes the flee-or-comply decision sooner; a contemplative player is more genuinely seduced. Both are legitimate playthroughs of the same architecture. The engine's beat-firing model becomes the *pacing of the player's awakening to what they are in* rather than a binary success/failure gate.

---

## World bible sketch

Not authored content — just shape.

**Brief.** You drove to a commune in northern Sweden for the once-in-ninety-years midsummer festival. Pelle, who you know from a grief group in London, invited you after your sister's death. You came with two anthropology grad students he also invited — Connie and Sten — who are writing a paper. It is the first day. The sun does not set here in summer.

**Cast (nine).**
- *Pelle* — your handler. Gentle, grief-aligned (his parents went over the Ättestupa at the last festival; he says this without sadness).
- *Siv* — elder matriarch. Wears the white robes. Translates the runes. Formal, warm, regal.
- *Dan* — elder patriarch. Will go over the cliff on Day 2. Calm about it.
- *Ulf* — Pelle's brother. Younger, more direct, less gentle.
- *Maja* — nineteen. Has been chosen to mate with one of the outsiders; the community has decided that is you.
- *Ruben* — the Oracle. Deformed by inbreeding, treated as the unmediated voice of the divine.
- *Ingemar* — like Pelle, brings outsiders.
- *Connie* — your fellow outsider. PhD candidate. Increasingly uneasy.
- *Sten* — Connie's partner. Dismissive. Likes the food.

**Locations (ten).** `dormitory_hut`, `dining_meadow`, `dancing_lawn`, `yellow_temple`, `ättestupa_cliff`, `oracle_grove`, `drying_barn`, `herb_garden`, `forest_path` (the only road out), `sacred_pond`.

**Truth.** The festival requires nine sacrifices. Four are outsiders (you and three others). Four are the eldest of the community, all turning 72. One is chosen by the May Queen. The cycle is ninety years. The kindness with which you are treated is not deceit — they genuinely love you and will sing for you when you die.

**Beats — sketch (~18, three categories).**

*Cultural-discovery.* `pelle_explains_the_seasons`; `siv_reads_the_runes`; `meal_at_the_long_tables`; `oracle_pronounces_over_you`.

*Threat-ratchet.* `the_ättestupa` (Day 2, you witness); `connie_disappears` (Day 3); `sten_disappears` (Day 4); `the_dosed_tea`; `you_find_the_drying_barn`.

*Seduction.* `pelle_holds_your_hand`; `the_screaming_circle`; `crowned_may_queen`; `the_meal_after_the_crown`.

*Decision beats* (mutually exclusive).
- `decided_to_flee` — requires `the_ättestupa` AND `connie_disappears`.
- `decided_to_become_may_queen` — requires `the_screaming_circle` AND `oracle_pronounces_over_you`.
- `decided_to_refuse_the_crown` — fires from the dance scene only.
- `decided_to_comply_passively` — silent default that fires on Day 8 if nothing else has.

**Endings (six).** `escaped_into_the_forest`; `caught_at_the_gate`; `crowned_chose_an_outsider`; `crowned_chose_pelle` (the dark Dani ending); `refused_and_burned`; `comply_passively`.

**Few-shots.** Six covering: arrival at the dormitory (Pelle on the bench), first meal at the long table, the Ättestupa as witnessed, the dosed tea (the register shift), the screaming circle, the dance.

---

## Game mechanics

What changes in *how the player plays*. Three buckets: new player-facing systems, small engine extensions, repurposed existing primitives.

### New player-facing systems

1. **Festival day clock.** Visible day-of-9 counter in the UI header, drives ritual pacing: Day 2 Ättestupa, Day 5 dance, Day 7 maypole, Day 9 culmination. *Engine: derived from which day-beats have fired (no new state-change kind needed — days are just authored beats whose firing counts).*

2. **Losable companions.** Connie and Sten start visible in `[PRESENT]` at the dining meadow on Day 1 and disappear over the festival. The player physically watches them stop showing up. Mechanism: chronicler agent (see "Engine refinements" below) emits `npc_deaths` at scene end, engine flips `state.npc_alive`, populator skips dead NPCs on next scene. *No `KillNPC` state-change kind for Phase 1; deaths are a post-scene consequence call.*

3. **Altered-perception register shift.** After the dosed tea, prose dilates. Cheapest: rely on Phase 2's scene-bounded conversation history (the establishing turn carries the register; subsequent turns inherit by inertia). Few-shots demonstrate the dosed voice. Backstop: a small `[ONGOING]` block in Phase 2's user prompt naming current persistent states.

4. **Ritual participation choices.** Group scenes give *participate / observe / refuse / escalate* rather than *say X / ask Y*. Authoring change to options, not engine.

5. **The crown as state-changing.** Once `crowned_may_queen` fires, NPC attitudes pivot. Existing beat machinery — no new mechanic.

6. **The gate as climactic location.** `forest_path` is reachable from Day 1, but walking there before `decided_to_flee` fires triggers a "Pelle is waiting at the gate" scene that gently returns you. After the decision beat, the next gate scene plays as the escape attempt. Existing pattern; authoring carries it.

### What's *removed* from Pant Goch

- **Investigation evidence model.** Decision beats gate on *experience* (`decided_to_flee` requires `the_ättestupa AND connie_disappears` — things seen), not *evidence* (things proven).
- **Journalistic-ethics framing of endings.**
- **NPC trust-as-resource.** Pant Goch's per-NPC failure paths (push Tom too hard → permanent loss) don't have a clean per-NPC equivalent — the commune is patient with you by design. Replaced by a community-wide `village_marks_you` threshold that *intensifies the festival's attention on you* rather than locking individual NPCs.

That last item is the most interesting gameplay shift. Pant Goch's tension is "don't lose any NPCs"; Midsommar's tension is "don't be *seen* by the community as the right one." Different valence, similar machinery.

---

## Engine refinements

These are the architectural improvements that fell out of thinking through Midsommar. They benefit Pant Goch too — they just aren't *required* by Pant Goch.

### The three-timescale agent model

The engine has agents operating at distinct timescales, and getting clear about that resolves several questions.

| Timescale | Agent | Job | What it owns |
|---|---|---|---|
| Per-turn | Phase 1 classifier | Read scene-so-far, decide which beats fired | `unlocked_beats` |
| Per-turn | Phase 2 narrator | Write the next 80-200 words of prose | (prose only; no state) |
| Pre-scene | Populator | Given current world state and the player's destination, decide who's at the player's location | `npc_locations` |
| Post-scene | **Chronicler** | Given the just-completed scene, derive everything that the world now knows | `case_file_entry`, `npc_memory`, `npc_alive`, `last_seen_at` |

The post-scene agent (currently named "log writer") has accumulated responsibilities beyond writing the case file — it already emits `npc_memory_updates`, and the proposed `npc_deaths` is the third. Its real job is **post-scene reckoning**: with the full dialogue history in conversation context, decide what the world looks like now.

**Rename: log writer → chronicler.** "Log writer" describes the original single responsibility (UI case file). "Chronicler" captures the role accurately — the agent that maintains the running record of consequences. The session-bounded `message_history` it carries is genuinely a chronicle. The name generalises across genres (Pant Goch journalist, Midsommar guest, anything).

### Why deaths belong on the chronicler, not Phase 1 or the populator

Three candidates considered:

1. **Phase 1 emits `KillNPC` per-turn.** Requires LLM to pair `UnlockBeat` with `KillNPC` in the same bundle. 4B models forget pairings. Risky.
2. **Populator emits at scene entry.** Cleaner separation than Phase 1, but the populator only has the *prose summary* of the prior scene (the case-file entry the chronicler wrote). For deciding "did Eluned die," raw dialogue history is a strictly better signal than its 100-word recap.
3. **Chronicler emits at scene end.** Has the full dialogue history in conversation. Already runs at scene end. Already emits state-mutating outputs (`npc_memory_updates`). Adding `npc_deaths` is a one-field extension.

(3) is the right answer for two concrete reasons:

- **Signal quality.** Deciding death from raw turns is way more reliable for a 4B model than from a prose recap.
- **Atomicity.** Death is applied the moment the scene ends, before the player returns to the map. No window of incoherence where the dead NPC is still in `npc_alive=True`.

The chronicler's output gains one field:

```python
class ChronicleEntry:
    case_file_entry: str
    npc_memory_updates: dict[str, str]
    npc_deaths: list[str] = []    # ← new: NPCs the scene narrated as dying
```

The engine applies the deaths immediately on receipt. Next scene's populator reads `state.npc_alive` and skips dead NPCs. Phase 2's `[PRESENT]` block filters dead. No `KillNPC` state-change kind, no `effects:` field on Beat, no new Phase 1 emission. One field, one agent, one mechanism.

This handles authored deaths AND improvised deaths uniformly. Authored: `connie_disappears` beat fires (Phase 1), Phase 2 narrates her absence, chronicler reads the turns and emits `npc_deaths: [connie]`. Improvised: player picks "attack Eluned," Phase 2 narrates her dying, chronicler reads the turns and emits `npc_deaths: [eluned]`. Same path, same engine response. The dynamism the LLM is there for is preserved.

### Principle for when to harden state

The engine has three channels of state, and clarity about which to use prevents the architecture sliding back toward v1's qualities-and-effects DSL:

- **Hard structured** (`npc_locations`, `unlocked_beats`, `npc_alive`) — deterministic, engine-validated, programmatically filterable.
- **Soft structured** (`npc_memory`) — LLM-written prose facts surfaced in subsequent prompts.
- **Pure prose** (case file) — LLM-written, LLM-read, no engine reasoning.

**Principle:** structured state should harden the minimum number of bits where (a) errors compound rather than self-correct, OR (b) the engine itself needs to filter on the bit. Everything else stays in the soft / prose channel.

Applied to NPC state:

| State | Channel | Why |
|---|---|---|
| `npc_location` | hard | Engine needs it for `npcs_at` |
| `npc_alive` | hard | Errors compound (resurrection); engine needs it for beat reachability |
| trust / disposition / mood | soft (`npc_memory`) | LLM reads attitude prose + memory entries; no engine filtering needed |
| drunk / injured / disguised | prose | Pure narrative state; if it affects what NPC does, that's `notes_to_llm` and beat firing |
| has-been-told-X | prose (case file) | Conversation context is the natural source |

The principle gives us "where do I add structured state?" without having to ask story-by-story. Pant Goch and Midsommar run on the same minimum set; differences are entirely in authored data.

### Per-NPC opt-outs

For per-NPC `killable: bool = true` (default true; Pant Goch sets every NPC to false in `cast.yaml`):

- Engine refuses to apply `npc_deaths: [tom]` if `tom.killable == false` — chronicler retries with the error.
- Recovers Pant Goch's SPEC §3.5 "this story has no in-game deaths" property as authored data, not as engine architecture.

The architecture stays general; the story narrows it. Same pattern would generalise to other state-changes if added (e.g., per-NPC `movable_by_llm: bool` if we ever revive `MoveNPC`).

### No tool-call loop — and why the current pattern is already a tool catalog

Worth restating because it is load-bearing engine philosophy. The current `StateChange` discriminated union (currently just `UnlockBeat`; could grow) is *functionally* a tool catalog — the model picks one or more variants and the engine applies. The only difference from pydantic-ai's `@agent.tool` calls is: no result roundtrip. The model commits to its bundle and Phase 2 narrates the consequences.

The reasons this is strictly better than tool calls for our use case:

1. **Grammar sampling at decode time.** Literal-narrowed ids are enforced by llama.cpp; the model literally cannot emit invalid ids. Tool calls don't get this for free.
2. **No mid-turn latency.** Structured-output bundles are one call; tool calls imply emit → wait → result → emit, doubling latency and burning KV cache.
3. **ModelRetry is the feedback channel.** Validation errors retry the same emission with the error in context — same information as a tool result, cleaner mechanism.
4. **Two-stage separation stays clean.** Tool calls inside Phase 2 would interleave with prose generation; the entire phase split breaks.

If a future scenario needs *mid-turn information lookup* (e.g., LLM asking the engine "has the player visited Tan-y-coed?" before deciding what beat to fire), that's the one case where real tool calls beat the current pattern. Out of scope for v1.

---

## Open engine questions

What's actually unresolved.

1. **Pacing floor.** Pick one of the three pacing options above. Recommendation: authored beat-chain (option 1) with `min_scenes_since_start` soft floor (option 2) as backstop. Implementation cost: one int field on `Beat`, one validator. Not Midsommar-specific — could benefit Pant Goch too.

2. **`[ONGOING]` channel for Phase 2.** Needed only if the dosed-perception register fails to inherit through scene-bounded conversation history. Cheapest version: a one-line block listing persistent state tags derived from `unlocked_beats`. Defer until playtest evidence.

3. **In-scene death edge case.** Chronicler-emits-deaths-at-scene-end means a death narrated *mid-scene* isn't reflected in state until the scene ends. For the rest of *that* scene, Phase 2 might keep the NPC speaking. This is actually narratively correct most of the time ("Eluned collapses, the scene ends with her motionless on the floor") but breaks if the player wants to do something with the corpse mid-scene. Decision: accept the limitation, author scenes to end naturally on the death. If it bites in playtest, Phase 1 could ALSO emit `npc_deaths` as a peer to `UnlockBeat`, narrowing the gap to per-turn.

4. **Leave-pick semantics for a confined location.** `forest_path` is always reachable; walking there pre-decision triggers a "Pelle at the gate" scene. Existing pattern; just authoring.

5. **Group-scene prose discipline.** Few-shots must demonstrate four+ NPCs in one room without losing voice. Authoring, not engine.

---

## What would actually need to change in code

Surprisingly little. The architecture was already designed for most of this.

**Required:**
- `state.py` — add `npc_alive: dict[str, bool] = field(default_factory=...)` (restore from SPEC §3.1; was dropped). Add `last_seen_at: dict[str, str]` for the case-file convenience field.
- `models.py` — add `killable: bool = True` to `NPC`.
- `schemas.py` — add `npc_deaths: list[str] = []` to `LogEntry` (or rename to `ChronicleEntry`). Validator: each id must reference a `killable=True` NPC.
- `engine.py` — apply `npc_deaths` when the chronicler returns; auto-update `last_seen_at` from `npcs_present` at scene end. Extend `npcs_at()` to filter `npc_alive=False`. Extend `reachable_beat_ids` to mark beats requiring dead NPCs as unreachable (if any beat author chooses to gate on NPC liveness — needs a thinking pass on whether to author this gating at all).
- `prompts.py` — chronicler system prompt mentions the new `npc_deaths` field with examples. Phase 2's `[PRESENT]` block renders `last_seen_at` for dead/missing NPCs ("you last saw Connie at the herb garden").
- **Rename log writer → chronicler** throughout: `agents.py` (`LogWriterRuntime` → `ChroniclerRuntime`, `emit_log_entry` → `emit_chronicle`, `AgentBundle.log_writer` → `AgentBundle.chronicler`), `prompts.py` (`build_log_writer_*` → `build_chronicler_*`), `scene_loop.py` (`play_session(log_writer=...)` → `play_session(chronicler=...)`), `tests/test_log_writer.py` → `tests/test_chronicler.py`, `CLAUDE.md` (agents table + "Prose voice canon" section).

**Optional, depending on playtest:**
- `[ONGOING]` block in Phase 2's user prompt for persistent states.
- `Beat.min_scenes_since_start: int | None` for pacing-floor backstop.
- Phase 1 emission of `npc_deaths` as a peer to `UnlockBeat` for in-scene death resolution (only if scene-end-only proves insufficient).

Everything else — locations, cast, truth, beats, endings, few-shots, map — is authoring, not code.

Total code patch: ~150 lines net plus the rename (~100 lines mechanical). The rename and the chronicler responsibilities expansion should ship in one PR — they are the same architectural decision viewed from two angles.

---

## Migration cost vs. reward

**Migration cost.** ~3 days for the engine refinements (most of which we'd do anyway). ~2 weeks of authoring for the full world bible at Pant Goch's level of polish. Pant Goch authoring would be discarded; engine work is preserved.

**Reward.** A game whose prose grain matches the model's grain, whose architectural choices carry dramatic weight instead of being worked around, whose audience is roughly an order of magnitude larger than Pant Goch's, and whose moment-to-moment captivation does not depend on the player loving a specific literary register.

**Recommendation.** Worth doing — but not blind. Before throwing away Pant Goch, do a small experiment first:

1. Ship the engine refinements (chronicler rename, `npc_alive`, `npc_deaths`, `killable`, `last_seen_at`) against the current Pant Goch bible. Pant Goch sets all NPCs to `killable: false`; nothing changes for that game. The refactor is non-breaking and tested.
2. Author ONE folk-horror scene (the first meal at the long tables, six few-shots, three NPCs, no beats) and run it on Gemma 4 E4B with the refactored engine. Compare prose quality against the equivalent Pant Goch scene.
3. If the prose register lands at the ~90% I am claiming, commit to the bible rewrite. If it does not, Pant Goch is still the safer bet — and the engine refinements from step 1 have already shipped.

The experiment is one evening of authoring after the engine PR. It is the highest-leverage thing to do next.
