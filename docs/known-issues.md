# HANDLER v2 — Known Issues

## What this is

A concise, evidence-backed list of the biggest structural problems in the
engine as of 2026-05-19, ranked by severity. Each issue states the problem,
the evidence (code references and the 19 May Logfire trace), and the
direction of a fix. This is an analysis artefact, not a build plan — no
code is changed by reading it.

The throughline: **state is a subtractive fence around an LLM improviser.**
The grammar sampler guarantees the model can only emit authored, reachable
beats; state can only ever *remove* options or *pick which ending is
allowed*. State never causes an event, and the engine can only remember
what the bible anticipated as a beat. That asymmetry is what keeps a 4B
model coherent — and it is the root of every issue below.

---

## TL;DR

1. **Purgatory softlock.** Endings are pull-based and nothing forces one. A player who never commits and has lost the evidence ladders loops forever — unwinnable and unlosable.
2. **Off-script actions vanish.** You can attempt anything, but the world has no memory outside 34 beats. Kill an NPC and the populator places them back next scene.
3. **State is subtractive-only.** Beats are the single verb. There is no channel for time, reputation, disclosure, or inventory — the kinds of state that make a scene play *differently* rather than gate *whether* a beat can fire.
4. **Permanent failure is invisible.** Negative beats silently amputate whole ladders; the reachability filter hides the closed door from the player.
5. **Structural ceiling.** ~11 broad story shapes. Prose dynamism is excellent; structural dynamism sits near a good small-scale CYOA, below storylet systems.

---

## 1. Purgatory softlock — no failsafe ending

Endings are **pull-based**. An ending fires only when the LLM (a) fires a
terminal beat in Phase 1, (b) the choice author emits `Leave`, and (c)
`resolved_ending_id_at_close` finds a satisfied ending. Nothing forces any
of these. If the player never says "I'm done" in fiction and the evidence
ladders are blocked, the loop `scene → leave → walk → scene` runs forever.

There is no *hard* softlock — `decided_to_walk_away` has no `requires`, so
`buried_the_story` is always technically reachable. But there is a real
*soft* softlock: a game that cannot be won or lost, only continued.

**Evidence:**
- `scene_loop.py:452` — the only assignment to `state.game_over` comes from
  an `EndGame`. There is no "no meaningful progress is possible" guard
  anywhere in `play_session`.
- `engine.py:101` — `resolved_ending_id_at_close` returns `None` freely; the
  loop just continues.
- `brief.yaml` authors a six-week budget deadline ("the money runs out
  mid-September") that the engine has no way to enforce — there is no day
  counter on `State`.

**Fix direction:** a day/time channel that forces a wrap-up ending at the
deadline; and/or bias the choice author toward surfacing walk-away when the
count of reachable positive beats drops near zero.

---

## 2. Off-script actions have no durable consequence

The game will not break if you go off-script — `FreeText` lets the player
type anything, and the grammar sampler still guarantees Phase 1 can only
emit existing, reachable beats and the choice author only valid structures.
But there is no state to *hold* an unauthored action. Kill Tom Bevan and:
Phase 1 cannot represent it (no `tom_is_dead` beat exists, so the grammar
forbids it); the prose writer improvises something; the chronicler may log
it as prose; then the populator places Tom at the pub by default next scene
because in *state* he is alive. The murder is narrated, then forgotten —
and the "dead man at the bar" is an active incoherence risk.

The world's entire memory of player action is: 34 beats + soft prose in
`npc_memory` / `case_file_entries`. Anything outside that is amnesia.

**Evidence:**
- `schemas.py` — the only Phase 1 state-change verb is `UnlockBeat`. There
  is no general fact channel.
- `engine.py:32` — `apply_state_changes` only ever `.add()`s to
  `unlocked_beats`; nothing represents arbitrary world change.
- `scene_loop._apply_placements` + populator defaults — NPCs return to
  `default_location` every scene unless a beat-driven override exists.

**Fix direction:** (1) a prose-writer rule to gracefully absorb off-script
attempts — narrate the attempt and its friction without committing the
fiction to an irreversible state the engine can't track; (2) a small set of
"rupture" negative beats for *thematically meaningful* transgressions
(`committed_violence`, `lied_about_being_police`, `threatened_a_source`)
routed into the existing `village_runs_you_out` → bad-ending machinery.
A general free-form `world_facts` channel is possible but breaks grammar
safety and invites a 4B model to contradict itself — not recommended.

---

## 3. State is subtractive-only — missing the dynamic channels

State reaches the engine through two *hard* gates — grammar narrowing
(`reachable_beat_ids` → which beats Phase 1 may emit) and ending resolution
(`resolved_ending_id_at_close`) — plus the soft prompt layer. Both hard
gates only ever *subtract* from the option space. The state that would make
the game most dynamic is the kind that modulates *how* a scene plays rather
than *whether* a beat can fire — and none of it exists.

Beats model milestones well: discrete, monotonic, grammar-safe, cheap for a
4B model. But they are the *only* shape, so everything is forced into
"a thing became true, forever." Gradients (trust, mood), resources (time,
the six-week budget), and per-target facts (what you've disclosed to whom)
have no representation.

**Evidence:**
- `state.py:58` — `State` has 8 fields; none is a quality score, calendar,
  inventory, or disclosure map.
- `models.py` — `Beat` carries only `requires_beats` / `requires_any_beats`
  / `blocked_by_beats`; the value space per beat is binary (set or unset).
- The 19 May trace confirms the village's collective stance is a single
  binary beat (`village_runs_you_out`) rather than a gradient — the same
  scene cannot play "warmer" or "colder" by accumulated reputation.

**Fix direction:** add companion channels alongside the milestone graph —
calendar/day count first (unlocks content the bible already implies), then
a disclosure ledger and reputation qualities. Keep the beat graph as the
spine; give it gradient/resource state as muscle. See
`story-dynamism-audit.html` §VI for the full proposal and cost estimates.

---

## 4. Permanent failure is invisible to the player

Beats are monotonic (`unlocked_beats` is only ever added to, `engine.py:32`)
and negative beats permanently block whole ladders via `blocked_by_beats`.
Trigger `tom_closes_off` in scene 1 and the entire five-rung Tom ladder
becomes unreachable for the rest of the game — correctly, by design. But the
reachability filter simply stops offering those beats; the player gets no
signal that a door has closed. A meaningful permanent consequence is
authored and then hidden.

**Evidence:**
- `engine.py:66-68` — a beat with any `blocked_by_beats` member in
  `unlocked_beats` is dropped from reachability silently.
- `truth.yaml` — `tom_closes_off` notes: "permanently blocks the entire
  Tom ladder." There is no paired mechanism to surface that loss in prose
  or UI.

**Fix direction:** when a negative beat fires, surface its consequence as
felt prose ("Tom won't meet your eye now") and/or a visible case-file note,
so the loss is experienced rather than silently subtracted. Good design
turns on the player *feeling* the door close.

---

## 5. Structural ceiling — modest branching

The engine wins on prose dynamism by a wide margin: every line is generated
fresh per play. On *structural* dynamism it is modest — roughly 11 broad
story shapes (6 endings × evidence paths), with ~128 atmospheric-beat
variants layered on top. That sits near a good small-scale CYOA, below
storylet systems like Sunless Sea (~600 qualities) or route-graphs like
80 Days (~750 routes).

This is a ceiling, not a bug — and it is downstream of issue #3. More beats
will not raise it; more *kinds* of state will.

**Evidence:**
- 6 endings in `truth.yaml`; decision beats gate on `requires_any_beats` of
  3 evidence sources → ~11 distinct evidence shapes.
- 7 optional atmospheric beats → 2⁷ ≈ 128 binary configurations.
- Full enumeration and peer comparison in `story-dynamism-audit.html`
  §III–IV.

**Fix direction:** the fix is the channels in #3, not more beats. The bible
is already large enough to carry the story.

---

## What's solid (for balance)

- **The grammar fence works.** No off-script input can crash the loop or
  produce an invalid beat; the 19 May Playwright run confirmed the engine
  stays coherent across deliberate misuse.
- **The beat grain is right** for milestones — discrete, legible, cheap for
  a 4B model.
- **The OR-clause de-bottlenecks the story** (shipped 19 May): Tom, Dylan,
  and Helen each carry an independent path to a climax; the game is no
  longer hostage to a single ladder.
- **The pull/leave loop is robust** turn-to-turn; the pacing nudge reliably
  fires `Leave` before a solo scene turns to filler (verified in the trace:
  Leave fired at turn 5 of a solo vicarage scene).

---

## Root cause, in one line

Every issue above is a face of the same fact: **the engine is constrained
improv, not open-world simulation.** The fix is never to loosen the grammar
fence — it is to add the missing *kinds* of state (time, qualities,
disclosure) and let the prose writer gracefully absorb everything the
authored space cannot represent.
