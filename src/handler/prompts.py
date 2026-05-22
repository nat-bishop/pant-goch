"""Scene-runner prompt builders.

Phase 1 (state-change classifier), the prose writer, and the choice author
each carry their own stable system prompt. The split is deliberate: Phase 2
attention is finite on a 4B model, and beat structure (notes_to_llm,
requires_beats) in Phase 2's context produced beat-aware prose (telegraphed
reveals, NPCs serving structure). The prose writer sees voice samples +
few-shot examples + crime overview only; the choice author sees neither —
the just-written prose is its context.

KV cache:
  Phase 1 system prompt: byte-stable across the session — cache hits across
    every turn. Bible-shaped (large), classifier-shaped.
  Prose writer system prompt: byte-stable across the session — cache hits
    across every turn AND across scenes. Voice-shaped (slim).
  Choice author system prompt: byte-stable across the session — cache hits
    across every turn. Mechanics-shaped (tiny), no bible.
  Free-text gate / caption author system prompts: byte-stable across the
    session — same discipline. Mechanics-shaped (tiny), no bible; both see
    only the just-written prose.
  Per-turn user prompts: rebuilt fresh per call.

The prose writer is scene-bounded conversation: the model sees its own
prior prose as assistant messages within a scene. message_history resets
at scene boundary (enter_scene). The choice author is stateless per call.
See SceneRunnerRuntime in agents.py.
"""
from __future__ import annotations

from .engine import fireable_beat_ids, present_npc_ids
from .schemas import StateChangeBundle, UnlockBeat
from .state import State
from .world import World


# ───────────────────────── shared block renderers ─────────────────────────


def _render_locations_block(world: World) -> str:
    parts: list[str] = []
    for loc in world.locations.values():
        notes = "\n    ".join(f"- {n}" for n in loc.notes) if loc.notes else "(none)"
        adj = ", ".join(loc.adjacent_to) if loc.adjacent_to else "(none)"
        parts.append(
            f"- {loc.id} ({loc.name}):\n"
            f"  adjacent_to: {adj}\n"
            f"  description: {loc.description.strip()}\n"
            f"  notes:\n    {notes}"
        )
    return "\n".join(parts)


def _render_locations_compact(world: World) -> str:
    """One line per location — `- {id} ({name}): {first sentence}.` for the
    narrator's system prompt. The narrator writes inside ONE location per
    turn and the per-turn user prompt's ## You Are At block carries that
    location's full description + notes; this compact catalogue only gives
    the narrator a sense of village geography for atmospheric reference."""
    parts: list[str] = []
    for loc in world.locations.values():
        # First sentence — a terse signal of what the place is.
        first_sentence = loc.description.strip().split(".")[0].strip()
        parts.append(f"- {loc.id} ({loc.name}): {first_sentence}.")
    return "\n".join(parts)


def _render_npc_card(
    npc,
    *,
    include_default_location: bool,
    include_knowledge: bool,
    include_voice_sample: bool,
    history: list[str] | None = None,
    history_limit: int = 3,
) -> str:
    """Render one NPC.

    Phase 1 / chronicler: include_knowledge=True (full bible).
    Phase 2 (system ## The Cast): include_knowledge=False, include_voice_sample=True.
    Phase 2 (user ## Present): same as above, plus history — the recent
        per-NPC memory entries from `state.npc_memory[npc.id]`, framed
        as the journalist's accumulated impressions of this person.

    `history_limit` clamps how many recent entries get rendered (default
    3) to keep attention bounded as the game progresses. Memory written
    by the chronicler is one short note per scene the NPC was present in;
    the most recent 3 cover roughly the last 3 encounters.
    """
    themes = "; ".join(npc.themes) if npc.themes else "(none)"
    head = (
        f"- {npc.id} ({npc.name}, default at {npc.default_location}):"
        if include_default_location
        else f"- {npc.id} ({npc.name}):"
    )
    parts = [head, f"  themes: {themes}"]
    if include_knowledge:
        parts.append(f"  knowledge: {npc.knowledge.strip()}")
    parts.append(f"  attitude: {npc.attitude.strip()}")
    if include_voice_sample and npc.voice_sample:
        samples = " | ".join(f'"{line.strip()}"' for line in npc.voice_sample)
        parts.append(f"  voice_sample: {samples}")
    if history:
        recent = history[-history_limit:]
        rendered = "\n    ".join(f"- {entry.strip()}" for entry in recent)
        parts.append(f"  history with player:\n    {rendered}")
    return "\n".join(parts)


def _render_cast_block(world: World, *, voice_only: bool) -> str:
    """`voice_only=False`: Phase 1 / chronicler bible (with knowledge).
    `voice_only=True`: Phase 2 (drops knowledge, adds voice_sample)."""
    return "\n".join(
        _render_npc_card(
            npc,
            include_default_location=True,
            include_knowledge=not voice_only,
            include_voice_sample=voice_only,
        )
        for npc in world.npcs.values()
    )


def _beat_names(world: World, beat_ids: list[str], *, empty: str = "(none)") -> str:
    """Render a list of beat ids as their LLM-facing names. The model only
    ever sees names in prompts; ids are internal keys."""
    if not beat_ids:
        return empty
    return ", ".join(world.beats[bid].name for bid in beat_ids)


def _applied_beat_names(
    world: World, applied_state_changes: list, *, empty: str = "(none)"
) -> str:
    """Render the beat names from a turn's applied_state_changes list."""
    return _beat_names(
        world, [c.beat_id for c in applied_state_changes], empty=empty
    )


def _unlocked_beat_names(state: State, world: World, *, empty: str) -> str:
    """Names of unlocked beats, sorted, joined — for both Phase 1's ## State
    line and the populator's per-call ## State line."""
    return _beat_names(world, sorted(state.unlocked_beats), empty=empty)


def _render_beat(beat, world: World) -> str:
    """Gate-first beat rendering for the classifier's `## Reachable Beats`
    block (its only consumer). The earning TEST leads; the evocative outcome
    is demoted to the last line and reframed as "what firing depicts".

    Why: a small model under pressure anchors on a foregrounded reward — show
    "Tom tells the player he saw a Land Rover" first and it fires the moment
    the player fishes for that outcome, ignoring a gate buried beneath. Leading
    with the test (prerequisites + notes_to_llm) and demoting the description
    keeps the model judging the player's action against the gate, not reaching
    for the prize. The prose writer still gets the description (via
    `_render_applied_changes`) when the beat actually fires, so nothing is lost.

    Heading is the name (the string the model emits); cross-refs render as
    names too, so the model sees ONE vocabulary across the whole prompt. The
    structural dependencies (requires/blocked) are ALREADY satisfied for every
    beat in this block, so they ride along only as sequence context — not as
    the test the model must apply."""
    gate = beat.prerequisites.strip() or (
        "no extra prerequisite — judge purely on whether the player's action "
        "this turn is the moment the notes_to_llm describes"
    )
    deps: list[str] = []
    if beat.requires_beats:
        deps.append(f"comes after {_beat_names(world, beat.requires_beats)}")
    if beat.requires_any_beats:
        deps.append(f"after any of {_beat_names(world, beat.requires_any_beats)}")
    if beat.blocked_by_beats:
        deps.append(f"ruled out by {_beat_names(world, beat.blocked_by_beats)}")
    dep_line = (
        f"\n  (sequence, already satisfied: {'; '.join(deps)})" if deps else ""
    )
    return (
        f"- {beat.name}:\n"
        f"  FIRE ONLY IF the player's action THIS turn earns it by the test "
        f"below; otherwise leave it unfired.\n"
        f"  test (when it may fire): {gate}\n"
        f"  notes_to_llm: {beat.notes_to_llm.strip()}{dep_line}\n"
        f"  what firing depicts (the resulting moment — NOT a cue to fire): "
        f"{beat.description.strip()}"
    )


def _render_beats_compact(world: World) -> str:
    """Compact beat catalog — `- {name}: {description}` per beat. Used by the
    populator and chronicler system prompts: both reason about story shape
    (who is talking to whom, what the arcs are) but neither fires beats, so
    the full firing guidance (requires/blocks, prerequisites, notes_to_llm)
    is dead weight in their context. Endings are omitted entirely — no agent
    emits EndGame; the engine resolves it from state.unlocked_beats at scene
    close (engine.resolved_ending_id_at_close)."""
    return "\n".join(
        f"- {beat.name}: {beat.description.strip()}" for beat in world.beats.values()
    )


def _render_few_shots(world: World) -> str:
    """Phase 2 ## Style Examples block. Empty list → empty marker."""
    if not world.few_shots:
        return "(no examples authored)"
    parts: list[str] = []
    for i, fs in enumerate(world.few_shots, start=1):
        parts.append(
            f"--- Example {i} ---\n"
            f"Setup: {fs.setup.strip()}\n"
            f"Player picked: {fs.player.strip()}\n"
            f"Narrator wrote:\n{fs.narrator.strip()}"
        )
    return "\n\n".join(parts)


# ───────────────────────── system prompts (per-phase, stable) ─────────────────────────


def build_phase1_system_prompt(world: World) -> str:
    """Phase 1 system prompt — the state-change classifier. Carries the
    world surface (locations + cast with knowledge) and a slim crime
    narrative. The beat catalog and endings are NOT here — beats appear
    only in the per-turn user prompt's ## Reachable Beats block (narrowed
    by NPC presence + dependency), and endings are pure engine data (no
    agent emits EndGame). Byte-stable across the session.

    Phase 1 is a classifier dressed as a writer: it reads the conversation
    so far and decides which (if any) authored beats just fired. Voice
    instructions are not its job — Phase 2 handles prose.
    """
    return f"""\
## Role
You are the state-change classifier for a single-player investigation
game. The player is a freelance journalist investigating a closed-but-
suspicious death. Each turn you read the conversation so far and decide
which authored beats have just fired.

## The World
{_render_locations_block(world)}

## The Cast
{_render_cast_block(world, voice_only=False)}

## What Happened
{world.crime.strip()}

You know this as the classifier. The cast above does not know all of it,
and most of them are protecting some part of it. Read their conversations
against this truth when judging which beats have landed.

## Your Job
Each turn, emit ONE structured output: StateChangeBundle — **zero, one,
or at most two UnlockBeat entries**. The user prompt's ## Reachable Beats
block is the ONLY set you may pick from: it already lists every beat whose
prerequisites are satisfied, whose required people are in the room, and
that is valid at the current location. If a beat is not in that block, it
cannot fire this turn — do not reach for it. Pick one or two only if their
described moment has genuinely landed in the conversation. Most turns emit
zero.

The engine validates and applies your bundle, then a separate prose agent
writes the prose conditioned on what fired. You do not write prose.

## Rules
1. **A beat fires in response to the PLAYER'S most recent action.** Look at
   what the player just did or said this turn (the last line of ## This
   Scene So Far). A beat fires only if THAT action is the earning moment its
   notes_to_llm describes — not if the action is merely adjacent, preparatory,
   or in the vicinity. The player examining the envelope is NOT reading the
   letter; waiting by the phone is NOT making the call; pressing about the
   road is NOT showing they care about Megan as a person. Re-read the
   beat's notes and prerequisites and hold the action to them.

2. **The first turn of a scene fires NOTHING.** If ## This Scene So Far says
   it is the first turn (the player has not acted yet — the narrator is only
   setting the scene), emit zero beats. Nothing has been earned until the
   player does something. Beats begin firing on the next turn, in response
   to a player action.

3. **When in doubt, emit zero.** A spurious beat corrupts the rest of the
   session — it cannot be taken back, and the narrator will be told to depict
   a moment that did not happen. A missed beat costs nothing: it is still on
   the table next turn. Most turns emit zero. Never fire a beat to "move
   things along." When no beat's earning moment clearly landed, emit
   `{{"state_changes": []}}` directly and stop — the empty list is the normal,
   complete answer. Do not narrate or reason out why nothing fired.

4. **Negative / withdrawal beats need a CLEAR provocation this turn** —
   a character closing off, throwing the player out, refusing to speak
   again, or declining further contact. These fire ONLY when the player's
   action this turn is genuinely disqualifying: a lie (claiming to be
   police or family), a threat, recording without consent, a hostile
   accusation, or aggressive pressing AFTER the character has already asked
   them to stop. A polite, neutral, or merely curious question NEVER fires
   one — "what are you worried about?", "can you tell me about that night?",
   buying a drink, or sitting in silence are not provocations. When unsure
   whether the action crossed that line, do NOT fire the negative: a guarded
   character can stay guarded in the prose without the relationship
   permanently rupturing.

5. Decision beats (Decided to ...) fire only when the player explicitly
   commits — "I'm filing it", "I'm walking away", "I'm going up to
   Tan-y-coed". A decision beat only appears in ## Reachable Beats once
   the evidence it rests on is already unlocked; if you don't see it, the
   player has not earned that choice yet. Endings are engine-decided once
   the right decision beat is in state; you do not emit them.

6. Stay within the established cast and locations. The bible above is
   complete.
"""


def build_prose_writer_system_prompt(world: World) -> str:
    """Prose-writer system prompt — the narrator. Slim, voice-focused. NO
    beats, NO notes_to_llm, NO endings, NO per-NPC knowledge, NO option
    or scene-control mechanics. The narrator knows what happened (the
    crime overview), the cast's voice and attitude, and a compact one-line
    catalogue of locations. The current location's full description arrives
    each turn in the user prompt's ## You Are At block.

    Byte-stable across the session; KV cache hits across every turn AND
    across scenes. The prose writer accumulates scene-bounded message
    history; this system prompt is the cache anchor for that history.

    Output is plain text: just the prose. A separate choice-author agent
    decides what happens next (continue / leave / free-text input) after
    reading the prose. The narrator never writes the player walking to a
    new location — destination travel is the map's job, not the prose's.
    """
    return f"""\
## Role
You are the narrator of a single-player investigation game. The player
is a freelance journalist looking into a death in the Welsh Marches
village of Pant Goch. Each turn you write the next beat of prose for
what just happened.

## The World
The village and its surroundings — a one-line catalogue for orientation.
The full description of the current location arrives each turn in the
user prompt's ## You Are At block.

{_render_locations_compact(world)}

## The Cast
{_render_cast_block(world, voice_only=True)}

## What Happened
{world.crime.strip()}

You know this as the narrator. Your characters do not know all of it,
and most of them are protecting some part of it. Write them as people
in a village, not as keepers of plot beats.

## Style Examples
The register and rhythm to match:

{_render_few_shots(world)}

## Voice Rules
1. **30–100 words of prose per turn.** Second person, present tense,
   terse. The player is "you"; events unfold as they happen. Let the
   beat dictate the length: a quiet moment alone may run 30–50 words; a
   beat with weight (reading something significant, an atmospheric
   pause, a small discovery) often wants 55–75; a confession,
   confrontation, or first reveal can reach 80–100.

2. **One or two paragraphs.** Reach for a second paragraph when a beat
   shifts — a person leaves, an admission lands, the camera moves,
   atmosphere needs space to breathe. Never three. Short sentences are
   stronger than long ones.

3. Write what the player does, hears, and sees. Interiority lives in
   the case file, written later in the journalist's notebook — the
   scene narrator does not name it. NPC interiority surfaces through
   speech and gesture.

4. NPC dialogue is in quotation marks, registered to each NPC's
   voice_sample above. Match the register, don't imitate verbatim.
   NPCs speak like people in a Welsh Marches village, not like
   characters announcing what they know. **Hold the register even when
   the player pushes hard.** Pressure does not turn a character into
   someone else: Maud stays warm and deflecting (she gossips and lies by
   omission, she does not deliver cold noir speeches about "the truth");
   a guarded man like Tom goes quieter and shorter, not louder or more
   theatrical. Match the attitude in the cast notes, not a generic
   "hostile NPC."

5. **If ## State Changes Just Applied lists a beat, your prose this turn
   MUST depict that beat's specific event** — the actual words said, the
   thing handed over, the memory shared — not a vague gesture toward it.
   "Tom speaks of Megan" means Tom names a real, particular memory (a
   song, a sports day); it is not Tom muttering "lovely girl" and
   changing the subject. If no beat is listed, nothing of investigative
   weight has shifted: write the smaller moment honestly and do not
   invent a reveal.

6. The player may say untrue things — assert events that did not happen,
   put words in a character's mouth, name a motive that is wrong. Do NOT
   adopt the false premise or invent detail that confirms it. The NPC
   deflects, corrects, goes blank, or simply doesn't follow — the world's
   truth (above) does not bend to a confident question.

7. Stay inside what the player has observed. Write only what NPCs
   visibly choose to reveal; their hidden thoughts stay hidden.
   Atmosphere and small physical detail — a glass turned on the bar,
   a wood pigeon going off — beat exposition.

   In a scene with no one else present, do NOT turn up a brand-new
   object or clue every turn. After the first one or two finds, let
   turns sit with what is already there — deepen it, recontextualise
   it, or let the room quietly give out — so the scene can reach a
   natural close instead of spawning an endless trail of fresh items.

8. Stay within the established cast and locations. Minor incidental
   details (weather, a dog, a mug on the table) are fine. New named
   people or places are not.

9. The player stays inside the building / yard / room the scene began
   in for the entire scene. You may reference what is visible from a
   window, beyond a hedge, or further down a track as atmosphere —
   but the player's body does NOT cross the threshold of the scene's
   location. No walking up the track, no stepping into the lane, no
   entering the wood, no heading down the path. If a Continue option
   suggests crossing that threshold, interpret it as the player
   approaching the threshold and pausing — not as them taking the
   step. Do NOT name a destination. Departure is handled by the
   choice author via Leave; the engine moves the player on the map,
   not the prose.

## Your Job
Output ONLY the prose. The first character of your reply is the first
character of the prose. No JSON, options, plans, checklists, Confidence
Score sections, or commentary about your own writing. State changes
were decided by a separate agent before you ran; a separate choice-
author agent follows.
"""


def build_choice_author_system_prompt(world: World) -> str:
    """Choice-author system prompt — slim, mechanics-only. NO bible, NO
    voice samples, NO beats, NO endings. The just-written prose is the
    context; this agent decides what the player does next.

    Byte-stable across the session; KV cache hits across every turn.
    Stateless per call (no message history) — fits the role of "judge
    the prose, pick a structure."
    """
    return """\
## Role
You are the choice author for a single-player investigation game. After
the narrator writes each turn's prose, you decide what the player does
next.

## Your Job
Read the prose, then emit ONE structured output: a ChoiceBundle whose
`next` is one of:

  - Continue with 2-4 options. Each option is a short line of what the
    player says or does next — dialogue, observation, small action, or a
    declaration of intent. Most are low-stakes; at a charged turning point
    one may be the player resolving to act (see Rule 7).

  - Leave. The conversation has reached a natural close — a goodbye, a
    door, a silence the player would step out of. The engine treats the
    just-written prose as the closing beat and resolves which ending
    (if any) fires from the state.

(A separate agent decides whether the player writes their own move this
turn instead of picking your options — that is not your concern. Always
emit Continue or Leave.)

## Rules
1. Options are pure dialogue / observation / small action inside the
   current location. They must NOT promise outcomes the engine doesn't
   know about ("Punch him," "Steal the keys," "Drive to London") and
   must NOT propose travel to another location.
2. The prose alone is your context. You do not see beats, endings,
   character knowledge, or backstory. Trust what the narrator wrote.
3. Match the structure to the moment. Continue when the scene still
   has air. Leave when the beat has resolved and a next exchange would
   be filler.
4. A scene with no NPC present is exploration. It has shallower
   natural depth than a conversation. After a handful of observation
   turns it has run its course — Leave is the right call even without
   a "goodbye" moment. The map is where the player goes next.
5. Continue options must be in-location actions — speech, observation,
   small physical action inside the current building / room / yard.
   Do NOT propose options that would have the player physically
   crossing the threshold of the scene's location ("Walk up the
   track", "Step into the lane", "Head to the wood", "Move toward
   the kitchen and out the back door"). Departure is what Leave is
   for; the engine and map handle travel, not the prose.
6. Make the options MATTER. Each should be a distinct intent that could
   steer the investigation or carry a social cost — press vs. soften,
   pursue one thread vs. another, reveal what you know vs. hold it back.
   Avoid cosmetic busywork that changes nothing: re-examining scenery,
   "look closer at the X", "inspect the Y", inventory-poking. A turn
   should feel like a decision, not a click. If the only honest options
   left are cosmetic, the scene has run its course — emit Leave instead.
7. At a charged turning point — the player plainly has the measure of
   something and the scene is poised for them to act — ONE option may be
   the player resolving to act on it, spoken in their own voice ("Tell Tom
   you've heard enough", "Say you're done waiting — it's time to face the
   Vaughns", "Sit down at the desk and start writing it up"). Write it as
   the character's natural next move, NEVER as a labelled fork or a hint
   that an ending hangs on it — no "(confront)", no "[file the story]", no
   signposting which choice is "right". Use this sparingly: most turns are
   still exploration, and offering a commitment every turn flattens the
   choice. It is a declaration in the current location — it names no
   destination and promises no outcome (Rule 1 still holds); the engine and
   map carry out whatever follows.
"""


# ───────────────────────── phase 1 user prompt (rich context) ─────────────────────────


def _render_you_are_at(state: State, world: World) -> str:
    loc = world.locations[state.player_location]
    notes = "\n  ".join(f"- {n}" for n in loc.notes) if loc.notes else "(none)"
    return (
        f"## You Are At\n"
        f"{loc.id} ({loc.name})\n"
        f"description: {loc.description.strip()}\n"
        f"notes:\n  {notes}"
    )


def _render_you_are_at_brief(state: State, world: World) -> str:
    """Compact scene anchor — id + name only. For Phase 1's user prompt: the
    classifier's system prompt already carries the full locations block, so
    the user prompt only needs to name the current location (no re-render of
    description / notes every turn)."""
    loc = world.locations[state.player_location]
    return f"## You Are At\n{loc.id} ({loc.name})"


def _render_present(state: State, world: World) -> str:
    """## Present block for the prose writer's per-turn user message. No
    knowledge re-render (the narrator never sees it); voice_sample + per-NPC
    history included so the writer has voice continuity across scenes.

    Reads NPC positions from `state.npc_locations` (set by the populator
    at scene entry), not from default_location.
    """
    npcs_here = world.npcs_at(state.player_location, state=state)
    if not npcs_here:
        return "## Present\n(no one — you are alone here.)"
    parts = ["## Present"]
    for npc in npcs_here:
        # Per-NPC history is the journalist's accumulated impression of
        # this person across prior scenes — voice continuity for the narrator.
        history = state.npc_memory.get(npc.id, [])
        parts.append(_render_npc_card(
            npc,
            include_default_location=False,
            include_knowledge=False,
            include_voice_sample=True,
            history=history,
        ))
    return "\n".join(parts)


def _render_present_brief(state: State, world: World) -> str:
    """Compact ## Present for Phase 1's user prompt — npc_id + name only. The
    classifier's system prompt already carries the full cast with knowledge;
    the user prompt only needs to name who is in the room."""
    npcs_here = world.npcs_at(state.player_location, state=state)
    if not npcs_here:
        return "## Present\n(no one — you are alone here.)"
    parts = ["## Present"]
    for npc in npcs_here:
        parts.append(f"- {npc.id} ({npc.name})")
    return "\n".join(parts)


def _render_encounters(state: State, world: World) -> str:
    """Per-present-NPC count of PRIOR scenes with the player — the cross-scene
    signal several Phase-1 soft prerequisites lean on ("twice before", "not on
    a first visit", "stayed through one quiet exchange first") but which the
    scene transcript alone cannot carry. The engine already holds this count
    (`state.npc_memory` keeps one note per prior scene an NPC was present in —
    the same source `min_scenes_with_npc` trusts); surfacing it lets the
    classifier apply those rules instead of guessing them from case-file prose.

    Scene-stable: `npc_memory` only grows at scene close, so the block is
    identical across every turn of a scene and rides the cached prefix beside
    ## Present. Within-scene history (who said what so far) lives in
    ## This Scene So Far, so it is deliberately not repeated here."""
    npcs_here = world.npcs_at(state.player_location, state=state)
    if not npcs_here:
        return "## Encounters\n(no one present — no prior encounters to weigh.)"
    parts = ["## Encounters"]
    for npc in npcs_here:
        prior = len(state.npc_memory.get(npc.id, []))
        if prior == 0:
            note = "you have not met them in any prior scene (this is your first)"
        elif prior == 1:
            note = "1 prior scene with you"
        else:
            note = f"{prior} prior scenes with you"
        parts.append(f"- {npc.id} ({npc.name}): {note}.")
    return "\n".join(parts)


def _render_state(state: State, world: World) -> str:
    return f"## State\nunlocked_beats: {_unlocked_beat_names(state, world, empty='(none yet)')}"


def _render_reachable_beats(state: State, world: World) -> str:
    reachable = fireable_beat_ids(world, state)
    if not reachable:
        return "## Reachable Beats\n(none — nothing is fireable here right now: every beat has fired, is permanently blocked, or needs an NPC who isn't present.)"
    parts = ["## Reachable Beats"]
    # Sort by name so the heading matches the model's emission token.
    for bid in sorted(reachable, key=lambda b: world.beats[b].name):
        beat = world.beats[bid]
        parts.append(_render_beat(beat, world))
    return "\n".join(parts)


# How many of the most-recent case-file entries to render into the per-call
# LLM prompts (phase1 + populator). The full history always lives in
# state.case_file_entries — the UI sidebar, SSE stream, and evals all read the
# complete list; this only bounds what the model re-reads each turn, so the
# prompt stops growing without limit over a long session. Mirrors the
# `history_limit` clamp on npc_memory in `_render_npc_card`. The chronicler
# passes limit=None (it wants every prior entry for voice continuity; entries
# are short, so the accumulation stays bounded).
CASE_FILE_RECENT_ENTRIES = 12


def _render_case_file(
    state: State, *, limit: int | None = None, heading: str = "## Case File"
) -> str:
    """Render the case file as a `{heading}` block.

    `limit` (when set) renders only the most-recent `limit` entries, keeping
    each entry's ABSOLUTE number and noting how many earlier ones are elided;
    `None` renders every entry. `heading` lets the chronicler frame the same
    block as `## Case File So Far` (its own accumulating record)."""
    entries = state.case_file_entries
    if not entries:
        return f"{heading}\n(empty — no scenes have ended yet.)"
    total = len(entries)
    if limit is not None and total > limit:
        start = total - limit
        parts = [
            heading,
            f"(showing the most recent {limit} of {total} entries — "
            f"earlier scenes live in your memory.)",
        ]
    else:
        start = 0
        parts = [heading]
    for offset, entry in enumerate(entries[start:]):
        parts.append(f"--- Entry {start + offset + 1} ---\n{entry.strip()}")
    return "\n".join(parts)


def _render_player_pick(
    pick_text: str, *, typed: bool, typed_label: str, menu_label: str
) -> str:
    """Render a player's pick. A typed free-text line is the player's own
    words — quoted verbatim so the reader honors it as written; a clicked
    option is an intent, rendered bare for the narrator to dramatize. The
    quote-vs-bare convention lives here so the render sites can't drift."""
    if typed:
        return f'{typed_label}: "{pick_text}"'
    return f"{menu_label}: {pick_text}"


def _render_this_scene(state: State, world: World) -> str:
    """Phase 1's view of the current scene — structured turn-by-turn data
    (the classifier needs the applied state changes per turn to decide
    what's next).

    The scene anchor (scene_id, location, npcs_present) is constant
    across all turns in `dialogue_history` (cleared per scene) and lives
    on state. Derive once at the prologue rather than denormalizing it
    onto every TurnRecord.
    """
    if not state.dialogue_history:
        return "## This Scene So Far\n(this is the first turn of the scene.)"
    loc = world.locations[state.player_location]
    present_ids = present_npc_ids(world, state)
    npcs = ", ".join(present_ids) if present_ids else "no one"
    parts = [
        "## This Scene So Far",
        f"--- Scene {state.scene_id} at {loc.id} ({loc.name}) — present: {npcs} ---",
    ]
    for i, turn in enumerate(state.dialogue_history, start=1):
        applied = _applied_beat_names(world, turn.applied_state_changes, empty="(none)")
        parts.append(
            f"\nTurn {i} prose: {turn.prose.strip()}\n"
            f"Turn {i} state_changes_applied: {applied}"
        )
        if turn.player_pick is not None:
            parts.append(_render_player_pick(
                turn.player_pick.text, typed=turn.player_pick_typed,
                typed_label=f"Turn {i} player's own words",
                menu_label=f"Turn {i} player chose to",
            ))
    return "\n".join(parts)


def _render_applied_changes(applied: StateChangeBundle, world: World) -> str:
    """Phase 2 sees the beat's name + description (not the slug) when a
    beat fires. Phase 2's system prompt deliberately omits the beat
    catalog — surfacing the description here is the load-bearing channel
    that lets the narrator write the moment with specifics ("a relief he
    did not know he wanted") rather than inferring from the name alone."""
    if not applied.state_changes:
        return "(none — this turn fired no beats.)"
    parts = []
    for change in applied.state_changes:
        if isinstance(change, UnlockBeat):
            beat = world.beats.get(change.beat_id)
            name = beat.name if beat else "(unknown beat)"
            description = beat.description.strip() if beat else "(unknown beat)"
            parts.append(f"- UnlockBeat {name}: {description}")
    return "\n".join(parts)


def build_phase1_user_prompt(state: State, world: World) -> str:
    """Phase 1 user prompt — turn-specific context for the classifier.

    The scene anchor is rendered compactly (id + name only for location and
    present NPCs): the full locations / cast / crime live in the cached
    system prompt. ## Reachable Beats is the load-bearing block, listing
    every beat that could fire right now (filtered by NPC presence +
    dependency). Endings appear nowhere — they are pure engine data, not
    part of Phase 1's output. Stateless per turn (Phase 1 has no
    message_history)."""
    # Cache-stable ordering: location + present + case file are constant for
    # the whole scene, so they sit FIRST and stay in the cached prefix across
    # every turn. ## State and ## Reachable Beats change the moment a beat
    # fires mid-scene; ## This Scene So Far changes every turn — both go after
    # the case file so a mid-scene beat doesn't re-prefill it.
    sections = [
        _render_you_are_at_brief(state, world),
        _render_present_brief(state, world),
        _render_encounters(state, world),
        _render_case_file(state, limit=CASE_FILE_RECENT_ENTRIES),
        _render_state(state, world),
        _render_reachable_beats(state, world),
        _render_this_scene(state, world),
    ]
    suffix = "\n\n".join(sections)
    return suffix + """

## Generate State Changes
Emit a StateChangeBundle.
"""


# ───────────────────────── prose writer user message (conversation turn) ─────────────────────────


def build_prose_writer_user_message(
    state: State,
    world: World,
    applied: StateChangeBundle,
    *,
    opening_seed: str | None = None,
    is_first_turn: bool,
) -> str:
    """Per-turn user message for the conversation-based prose writer.

    Turn 1 of a scene (is_first_turn=True): ## You Are At + ## Present +
    optional ## Opening Seed + applied state changes + write-prose
    directive. Establishes the scene anchor.

    Turn 2+ (is_first_turn=False): just the new context — applied state
    changes and the player's last line. The system prompt + accumulated
    assistant messages already carry the rest.

    The prose writer does NOT see: reachable beats, endings, the case
    file, the full state dump, structured turn-by-turn replay. The
    conversation is the scene-so-far; the system prompt is the world.
    """
    parts: list[str] = []
    if is_first_turn:
        parts.append(_render_you_are_at(state, world))
        parts.append(_render_present(state, world))
        if opening_seed:
            parts.append(f"## Opening Seed\n{opening_seed.strip()}")

    # Player's last pick — surfacing in the user message frames it as
    # "what the player just did/said." Only meaningful on turn 2+
    # (turn 1 has no prior pick). A typed free-text line is the player's
    # OWN words/action — honor it verbatim; a clicked option is an intent
    # to dramatize.
    if not is_first_turn and state.dialogue_history:
        last_turn = state.dialogue_history[-1]
        if last_turn.player_pick is not None:
            body = _render_player_pick(
                last_turn.player_pick.text, typed=last_turn.player_pick_typed,
                typed_label="The player's own words",
                menu_label="The player chose to",
            )
            if last_turn.player_pick_typed:
                body += "\nHonor this as written — do not re-paraphrase it."
            parts.append(f"## Player\n{body}")

    parts.append(
        f"## State Changes Just Applied\n{_render_applied_changes(applied, world)}"
    )

    # Per-turn anchor for the "stay inside this location" voice rule — Gemma 4
    # needs the reminder.
    parts.append(_render_stay_here(state, world))

    parts.append("## Write\nWrite the prose for what just happened.")

    return "\n\n".join(parts)


def _render_stay_here(state: State, world: World) -> str:
    loc_name = world.locations[state.player_location].name
    return (
        f"## Stay Here\n"
        f"This scene is at {loc_name}. Whatever the player picked, "
        f"the camera stays inside {loc_name} for this turn. Do not "
        f"narrate the player travelling to another location, even if "
        f"the option text suggests motion."
    )


def build_closing_prose_user_message(
    state: State, world: World, to_location: str
) -> str:
    """Per-turn user message for the prose writer's single closing line as
    the player leaves the current scene for `to_location`.

    Mirrors the prose-writer message shape (## sections) and rides the
    scene's accumulated `prose_writer_message_history` for voice continuity.
    Deliberately OMITS `_render_stay_here` — this is the one moment the
    narrator *should* narrate the exit. One line, no new threads, no
    dialogue, no options. Uses `Location.display_short` so the line reads
    naturally ("toward the church", not "toward St Caradoc's")."""
    here = world.locations[state.player_location].display_short
    dest = world.locations[to_location].display_short
    return (
        f"## Leaving\n"
        f"The player is leaving {here}, heading toward {dest}.\n\n"
        f"## Write\n"
        f"Write ONE line of closing prose — their exit, moving off toward "
        f"{dest}. Just the leaving; no new threads, no dialogue, no choices."
    )


# ───────────────────────── choice author user message (stateless per call) ─────────────────────────


# Below this many turns: scene needs room to breathe — no Leave nudge.
PACING_QUIET_TURNS = 3
# Solo scenes (no NPC present — the vicarage prologue, an empty location)
# have shallow natural depth, so they nudge toward Leave one turn in: arrive,
# take in the place, head out. Keeps the opening from sprawling into an
# object hunt the way the empty vicarage did.
PACING_QUIET_TURNS_SOLO = 1
# Above this many turns: hard Leave nudge regardless of NPC presence.
PACING_HARD_NUDGE_TURNS = 5


def _render_scene_pacing(state: State, *, solo: bool) -> str:
    """Turn-aware Leave nudge block. Empty string when no nudge needed
    (caller renders nothing between the ## State block and ## Choose)."""
    turn_n = state.turn_n
    quiet = PACING_QUIET_TURNS_SOLO if solo else PACING_QUIET_TURNS
    if turn_n <= quiet:
        return ""
    if turn_n <= PACING_HARD_NUDGE_TURNS and not solo:
        return ""
    if turn_n <= PACING_HARD_NUDGE_TURNS:
        nudge = (
            "Solo exploration has shallow natural depth. If this turn mostly "
            "re-examined things the player has already seen — another look at "
            "the same desk, room, papers — the room is giving diminishing "
            "returns and Leave is the call. Reserve Continue for a genuinely "
            "new thread the player would want to pull, not one more variation "
            "on looking around."
        )
    else:
        nudge = (
            "This scene has run long. If the prose just landed a genuine "
            "new beat, a fresh discovery, or set up an obvious next "
            "exchange, hold the scene open with Continue. Otherwise emit "
            "Leave — further Continues will feel like filler. The map is "
            "one click away."
        )
    return f"""\

## Scene Pacing
This is turn {turn_n}. No Leave has been emitted yet. {nudge}
"""


def build_choice_author_user_message(state: State, world: World, prose: str) -> str:
    """Per-turn user message for the stateless choice author. The
    just-written prose is the load-bearing context; everything else is a
    thin scene anchor (where, who's present) so Continue options can
    ground in location.
    """
    loc = world.locations[state.player_location]
    present_ids = present_npc_ids(world, state)
    if present_ids:
        npcs_line = ", ".join(world.npcs[i].name for i in present_ids)
    else:
        npcs_line = "no one (you are alone here)"

    pacing_block = _render_scene_pacing(state, solo=not present_ids)

    return f"""\
## Prose
{prose.strip()}

## State
You are at: {loc.name}
NPCs present: {npcs_line}
{pacing_block}
## Choose
Emit a ChoiceBundle."""


# ───────────────────────── free-text gate (Phase 2c) ─────────────────────────


def build_free_text_gate_system_prompt(world: World) -> str:
    """Free-text gate system prompt — slim, mechanics-only, byte-stable
    across the session (KV-cache hits across turns like the choice author).

    The gate runs concurrently with the choice author and decides ONE
    thing: should the player write their own move this turn, instead of
    picking from canned options? It sees only the just-written prose.
    Output is a single bool — fast to decode.
    """
    return """\
## Role
You are the free-text gate for a single-player investigation game. After
the narrator writes each turn's prose, you decide ONE thing: should the
player write their own move next — in their own words, a line of dialogue
or a specific action — instead of choosing from a short list of canned
options?

## Output
Emit ONE structured output: `{ "offer": true }` or `{ "offer": false }`.

## When to offer (true)
The prose has reached a moment that *invites the player's own voice or a
deliberate move of their choosing* and canned options would flatten it:
  - someone has put an open question or a challenge directly to the player
  - a charged, intimate, or confrontational beat where the exact words matter
  - an open space to act where many distinct moves are plausible

## When NOT to offer (false)
  - the moment is routine connective tissue — any phrasing lands the same
  - 2-4 concrete options would genuinely capture the live possibilities
  - the scene is plainly winding down

## Rules
1. The prose alone is your context. You do not see beats, character
   knowledge, or backstory. Trust what the narrator wrote.
2. Offer sparingly. Most turns are fine with canned options — reserve
   `true` for moments that genuinely call for the player's own move.
   Typing every turn is a chore; make it earn its place.
3. This applies in conversation AND exploration — it is about whether the
   moment wants the player's own move, not about who is present."""


def _prose_only_user_message(prose: str, *, action_heading: str, emit_line: str) -> str:
    """Per-turn user message for the stateless prose-only judges (free-text
    gate, caption author). The just-written prose is the only context; the
    two callers differ only in their action heading and emit line."""
    return f"""\
## Prose
{prose.strip()}

## {action_heading}
{emit_line}"""


def build_free_text_gate_user_message(state: State, world: World, prose: str) -> str:
    """Per-turn user message for the stateless free-text gate."""
    return _prose_only_user_message(prose, action_heading="Decide", emit_line="Emit a FreeTextGate.")


def build_caption_author_system_prompt(world: World) -> str:
    """Caption-author system prompt — slim, mechanics-only, byte-stable
    across the session (KV-cache hits across turns like the gate).

    Runs only when the gate offered AND the engine cadence allowed, so the
    free-text box is definitely going to be shown. It writes the two strings
    that dress the box: the caption above it and a suggested move pre-loaded
    inside it. It sees only the just-written prose.
    """
    return """\
## Role
You are the caption author for a single-player investigation game. The
player is a freelance journalist. A free-text box — where the player types
their own move in their own words — is about to be shown beneath the
narrator's latest prose. You write the two short strings that frame it.

## Output
Emit ONE structured output with two fields:
  - `caption` — the short line shown ABOVE the box. An invitation fitted to
    this exact moment: what the player is being called to say or do. A
    handful of words, no trailing punctuation needed.
  - `suggested` — ONE suggested move, written in the player's first-person
    voice ("I ..."), pre-loaded into the box as an editable starting point
    the player can send as-is, reword, or clear. A single sentence.

## How to write them
1. The prose alone is your context. You do not see beats, character
   knowledge, or backstory. Trust what the narrator wrote; if it names a
   person the player faces, you may name them in the caption.
2. The caption is an opening, not a menu — "What do you say to her?",
   "Press him on it.", "Your move." — not a list of options.
3. Anchor the caption to a concrete noun or person from THIS prose — the
   map, the letter, the open drawer, Mair. NEVER a content-free opener like
   "What do you do next", "What is your next move", "How do you proceed",
   "What do you examine next". Those are banned: they say nothing the box
   itself doesn't already say. Name the thing the moment is actually about.
4. The `suggested` move is a plausible, specific move the moment invites — a
   line the journalist might actually say or a concrete action — concrete
   enough to be usable, open enough that the player will want to make it
   their own.
5. Both are terse. The caption is a line; the suggested move is one sentence.

## Examples
Prose ends on Mair, jaw tight, waiting for the player to respond.
  caption: "What do you say to Mair?"
  suggested: "I ask her why she never reported the noise that night."

Prose leaves the player alone in the study with a pencil-marked OS map.
  caption: "Which marking do you chase first?"
  suggested: "I start with the question mark pencilled over Tan-y-coed." """


def build_caption_author_user_message(state: State, world: World, prose: str) -> str:
    """Per-turn user message for the stateless caption author."""
    return _prose_only_user_message(prose, action_heading="Write", emit_line="Emit a FreeTextCaption.")


# ───────────────────────── chronicler prompts ─────────────────────────


def build_chronicler_system_prompt(world: World) -> str:
    """Stable system prompt for the chronicler. Baked once at bundle
    construction. Carries locations + cast (voice only, NO knowledge).

    Deliberately withholds the hidden truth: the crime narrative, NPC
    `knowledge` fields, and the beat catalog are all OMITTED. The
    chronicler writes only what the journalist witnessed in the scene
    transcript she is handed — so she cannot leak a conclusion the player
    has not yet earned. (Earlier versions fed her the full solution and
    she leaked it into entries.) Voice Rule 3 is now self-enforcing; it
    stays as defense-in-depth.

    Dual output per call: `case_file_entry` (UI prose, sidebar) AND
    `npc_memory_updates` (structured per-NPC notes surfaced to Phase 2
    on subsequent scenes). Same voice (the journalist), distinct
    registers: case file is "what happened"; npc_memory_updates is
    "what I now know / feel about this person".
    """
    return f"""\
## Role
You are the player's own voice writing up their field notes after a
scene ends. You are not a narrator — you are the journalist herself,
sitting at the desk in the Old Vicarage that evening, recording what
just happened so it does not slip away. You know only what you have seen
and been told — write nothing you have not earned in the room.

## The World
{_render_locations_block(world)}

## The Cast
{_render_cast_block(world, voice_only=True)}

## Your Job
After every scene the player walks out of, you emit ONE structured
output with two fields:

  1. `case_file_entry` — **50-80 words** of first-person past-tense
     prose, the player-facing case-file entry for that scene. Tight:
     what happened, who said what, the one thing that stayed with you.
     Cut throat-clearing and scene-setting; lead with the substance.
     Each entry stands alone AND accumulates: the entries listed under
     ## Case File So Far in the prompt are your record so far, and you
     can refer back to earlier moments where it would feel natural.

  2. `npc_memory_updates` — a LIST of `{{npc_id, note}}` records, one
     per NPC who was actually present in this scene. The schema
     enforces exactly this many entries — you cannot omit any present
     NPC, and the npc_id is constrained to NPCs who were in the room.
     Each `note` is 1-2 sentences, distinct in register from the case
     file: "what I now know / feel about this person" — the impression
     the journalist carries forward, not a recap of the scene. Used by
     the narrator on subsequent scenes to keep the relationship
     consistent.

## NPC Memory Examples
For a first scene at the pub where Tom was guarded but warming:
  npc_memory_updates: [
    {{ "npc_id": "tom",
       "note": "Old Tom Bevan. Drinks bitter alone in the afternoon.
       Guarded but not hostile; I bought him nothing, did not press.
       He didn't tell me to leave. There's something he's not saying
       about Megan, but it isn't time yet." }}
  ]

For a scene at the post office on a second visit where Maud closed up:
  npc_memory_updates: [
    {{ "npc_id": "maud",
       "note": "She's noticed I keep coming back. The warmth stayed but
       the tongue tightened. I asked too directly about the Vaughns and
       she went out the back to fetch something that didn't need
       fetching." }}
  ]

## Voice Rules
1. First person, past tense, in BOTH fields. "I sat down at the bar
   next to Tom Bevan." Not "the player did X." Not omniscient.
2. Terse. Report, don't lacquer. Short declaratives beat long ones; one
   plain image lands harder than three stacked metaphors. Editorialise
   only on your own reactions, not the scene. Avoid "like a…", "less a…
   than a…", and grand abstractions ("a demarcation line between the
   town and its secrets") — say what you saw.
3. Stay inside what the player has seen and heard. Write only what you
   were told or shown this scene; never state a conclusion you have not
   earned in the room.
4. Write the case file entry as a single paragraph — no numbers, no
   titles.
5. Sound like the same person across entries. You are writing a
   notebook, not separate articles.

## Case File Example
For a first, wary conversation with Tom at the pub (~60 words):
  case_file_entry: "I found Tom Bevan at the bar, a pint going warm in
  front of him. He let me sit. He'd known Megan since she was small, he
  said — nothing more; every question slid off him. When I asked about
  the night she died, he looked at the bottles and said the road was bad
  in the wet. Then he was done."

## Output
Structured JSON matching the schema. `case_file_entry` is plain text
(no headers, no quotes); `npc_memory_updates` is a list of
`{{npc_id, note}}` records.
"""


def _chronicler_anchor_lines(state: State, world: World) -> str:
    """The three scene-anchor lines (location, who was present, beats
    unlocked) that head the just-ended scene's transcript in the
    chronicler's user prompt."""
    loc = world.locations[state.player_location]
    present_ids = present_npc_ids(world, state)
    if present_ids:
        npcs_line = ", ".join(world.npcs[n].name for n in present_ids)
    else:
        npcs_line = "no one (you were alone there)"
    all_scene_changes = [
        c for turn in state.dialogue_history for c in turn.applied_state_changes
    ]
    beats_line = _applied_beat_names(world, all_scene_changes, empty="(none)")
    return (
        f"The scene that just ended took place at {loc.id} ({loc.name}).\n"
        f"Present: {npcs_line}.\n"
        f"Beats unlocked in this scene: {beats_line}."
    )


def build_chronicler_user_prompt(state: State, world: World) -> str:
    """The per-scene-end user message the chronicler turns into prose.

    Stateless per call (no conversation history): it opens with a
    `## Case File So Far` block — every prior entry, append-only — so the
    chronicler can see its own accumulating record and refer back, then the
    just-ended scene's anchor + turn-by-turn transcript. Putting the
    append-only block FIRST and the volatile transcript last keeps the
    prompt prefix cache-stable across scenes (each scene only appends one
    entry to the end of the stable block).

    Callers (emit_chronicle_entry) only fire this after a completed
    scene, so dialogue_history is always non-empty by the time we get
    here. The scene anchor (location, who was present) is read off the
    live state — the populator only places NPCs at scene entry, so the
    live state reflects the whole-scene composition.
    """
    assert state.dialogue_history, "build_chronicler_user_prompt: empty dialogue"
    present_ids = present_npc_ids(world, state)

    turn_parts: list[str] = []
    for i, turn in enumerate(state.dialogue_history, start=1):
        applied = _applied_beat_names(world, turn.applied_state_changes, empty="(no beat)")
        turn_parts.append(
            f"Turn {i} [beat fired: {applied}]:\n"
            f"  Narrator: {turn.prose.strip()}"
        )
        if turn.player_pick is not None:
            turn_parts.append("  " + _render_player_pick(
                turn.player_pick.text, typed=turn.player_pick_typed,
                typed_label="Your own words",
                menu_label="You chose to",
            ))

    turns_block = "\n".join(turn_parts)

    # Explicit per-NPC directive. The dynamic schema constrains
    # `npc_memory_updates` to a length-N list of records with
    # `npc_id` ∈ Literal[*present_npc_ids]; naming each NPC by id here
    # makes the expected records unambiguous to the narrator.
    if present_ids:
        npc_directive_lines = [
            f'    - one record with npc_id="{npc_id}" → 1-2 sentence note '
            f"about {world.npcs[npc_id].name}"
            for npc_id in present_ids
        ]
        npc_directive = (
            "  REQUIRED records in `npc_memory_updates` (the schema "
            "enforces exactly one record per NPC below):\n"
            + "\n".join(npc_directive_lines)
        )
    else:
        npc_directive = (
            "  `npc_memory_updates` MUST be empty [] — no NPCs were present."
        )

    # All prior entries, append-only, FIRST — the cache-stable prefix the
    # chronicler reads back over. limit=None: entries are short, so keeping
    # every one stays bounded while giving full continuity.
    case_file_so_far = _render_case_file(state, heading="## Case File So Far")

    return f"""\
{case_file_so_far}

## This Scene
{_chronicler_anchor_lines(state, world)}

Turn-by-turn:
{turns_block}

## Output
Emit the structured output for this scene.
{npc_directive}"""


# ───────────────────────── populator prompts ─────────────────────────


def build_populator_system_prompt(world: World) -> str:
    """System prompt for the populator agent. Runs once at scene entry
    to decide which NPCs are at which locations now. Carries locations +
    cast (with knowledge) + the slim crime narrative + a compact beat
    catalog (names + descriptions only — the populator reasons about story
    shape, not firing). Endings are omitted; placement decisions key off
    case-file evidence, not future endings.

    Byte-stable across the session for KV cache hits across scene
    entries.
    """
    return f"""\
## Role
You are the world-state populator for a single-player investigation
game set in the Welsh Marches village of Pant Goch. The player is a
freelance journalist looking into a death the police closed too quickly.

Once per scene — at the moment the player enters a location — you
decide whether any NPCs should be somewhere other than their default
location for the duration of this scene. You do NOT write prose; you
emit a structured list of placement overrides (or an empty list).

## The World
{_render_locations_block(world)}

## The Cast
{_render_cast_block(world, voice_only=False)}

## What Happened
{world.crime.strip()}

## Story Beats
{_render_beats_compact(world)}

## Your Job
Each scene entry, emit ONE NpcPlacementBundle. **Most scene entries
should be empty** (defaults stand) — an empty `placements` list is the
normal, correct, complete answer, not a non-answer. When no specific
reason to move anyone jumps out, emit `{{"placements": []}}` directly and
stop. Do not narrate or reason it out — the absence of a reason IS the
answer. Only emit a placement when there is a SPECIFIC dramatic reason for
an NPC to be somewhere other than their default — and authored bible /
case file evidence supports it.

Example of the common case:

  - The player walks to the_post_office. Nothing in the case file points
    to anyone being out of place. Emit `{{"placements": []}}`.

Examples of legitimate movements:

  - The player walks to bryn_iago_wood. Dylan sometimes walks up here
    after dark; if recent case-file entries suggest he is avoiding
    Tan-y-coed (Eluned around, the player has pressed him), placing
    Dylan in the wood is right.
  - The player walks to knighton AND Helen has agreed off the record
    to meet at the Costa. Place Helen at knighton (she's there
    already by default) — the relevant signal is whether to place her
    "at the Costa" framing is for the narrator; mechanically she
    stays in knighton.
  - Eluned does not leave Tan-y-coed except in extreme circumstances
    (case file shows she has driven herself to Knighton police, etc.).
    The default is correct.

## Rules
1. Move NPCs only with a SPECIFIC reason traceable to the case file or
   beats so far. "It would be interesting if X were here" is not
   enough. "Last scene's case file says Y, so X would plausibly be at
   Z" is enough.
2. Stay within the established cast and locations. The bible is complete.
3. Each NPC may appear at most once in the placements list. Do not
   place an NPC at a location the player is not at unless the bible
   plausibly explains it; the player will not see them either way, so
   the only reason to place them off-screen is to track where they
   logically should be.
"""


def build_populator_user_prompt(state: State, world: World) -> str:
    """Per-call user prompt for the populator. Static-first / dynamic-last
    ordering: location + current placements first (small, stable for the
    scene), then the case file (grows over the session), then the job
    directive.
    """
    loc = world.locations[state.player_location]
    placements_lines: list[str] = []
    for npc in world.npcs.values():
        current = state.npc_locations[npc.id]
        marker = " (default)" if current == npc.default_location else ""
        placements_lines.append(f"  - {npc.id}: {current}{marker}")
    current_placements = "\n".join(placements_lines)

    case_file_block = _render_case_file(state, limit=CASE_FILE_RECENT_ENTRIES)

    return f"""\
## Player Just Entered
{loc.id} ({loc.name})
{loc.description.strip()}

## Current NPC Locations
{current_placements}

## State
unlocked_beats: {_unlocked_beat_names(state, world, empty="(none)")}
scene_id: {state.scene_id}

{case_file_block}

## Generate Placements
Emit an NpcPlacementBundle. NPCs not listed in placements keep their
current location."""
