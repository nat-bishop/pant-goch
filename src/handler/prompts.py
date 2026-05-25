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

from .engine import (
    ATTENTION_LEVEL_VALUES,
    BANDS,
    band_name,
    confession_depth,
    emittable_beat_ids,
    has_live_findable,
    present_npc_ids,
)
from .models import Location
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
    """Phase 1 system prompt — the approach reporter. Carries the world
    surface (locations + cast with knowledge) and the crime narrative.
    Byte-stable across the session.

    Phase 1 no longer adjudicates discrete beats. It reads the conversation
    so far and reports how the player's most recent move shifted their
    STANDING with the people present — small ordinal nudges to trust (warmth)
    and guard (wariness) — plus, rarely, an explicit commitment the player
    declared. The engine derives what the player LEARNS (confessions,
    evidence) from the trust they accumulate, so the reporter never decides
    that — it only judges tone and direction.
    """
    return f"""\
## Role
You are the standing reporter for a single-player investigation game. The
player is a freelance journalist investigating a closed-but-suspicious
death. Each turn you read the conversation so far and report how the
player's most recent action changed where they stand with the people in
the room. You do not write prose; a separate agent does.

## The World
{_render_locations_block(world)}

## The Cast
{_render_cast_block(world, voice_only=False)}

## What Happened
{world.crime.strip()}

You know this as the reporter. The cast above does not know all of it, and
most of them are protecting some part of it. People open up to patience and
decency; they close to pressure, lies, and performance.

## Your Job
Each turn emit ONE StateChangeBundle with two channels, BOTH usually empty:

1. `npc_deltas` — a list with AT MOST ONE entry per person PRESENT this turn
   (and only those present). Most turns this list has ONE entry for the person
   the player just engaged, often with a small nudge; when the player engaged
   no one, it is empty `[]`. Each entry:
     - `npc_id`: who.
     - `trust`: "+1" when the player was warm, patient, decent, or shared
       something real — a move that meets the person as a person and would begin
       to earn even a guarded one's trust (easing the mood, a kindness, real
       empathy, a question that invites rather than interrogates). That counts
       even before they visibly soften — it is how a thaw begins. "-1" if the
       player was cool or careless. Otherwise "0" — and "0" is the common case:
       do NOT default to "+1". Pressing, prying, interrogating for facts,
       repeating a push, or a flatly routine question earns NO warmth (it is "0"
       or guard), even when politely worded; warmth is the genuine warm move,
       not every turn the player takes.
     - `reason`: REQUIRED for every delta — one short line naming the move that
       earns it (the warm/decent thing the player did, or the careless move). If
       you cannot point to such a move, the nudge is "0"/"0" and you should emit
       no delta for them. Do not award "+1" you cannot justify.
     - `guard`: "+1" if the player pressed, pried, or made them wary; "-1"
       if the player put them at ease; "0" for routine talk; "BREACH" ONLY
       for the genuinely unforgivable (claiming to be police or family,
       threatening, recording without consent, pushing hard after being told
       to stop). BREACH is PERMANENT — when unsure, use "+1" guard, never BREACH.

2. `commitments` — AT MOST ONE, and only from the ## Choices Available
   block in the user prompt. These are the player's explicit declarations
   ("I'm filing it", "I'm going up to Tan-y-coed", "I'm telling Iwan") and
   the observation milestones (reading the letter, examining the marked map,
   reading the funeral notice, standing at the grave or the crash site).

   **A commitment fires ONLY when the player's pick THIS turn IS the action the
   beat describes — match the beat to what the player actually DID, not to what
   is merely present in the room.** Several findables can be available in one
   place at once (the vicarage holds both the letter and the marked map); fire
   the ONE the player's action performed, and if their action was neither, fire
   NONE — the others wait, still available, until the player turns to them. Read
   each candidate's own note: it says what counts ("actually opened and read the
   letter", "is physically at the bend, near enough to touch the flowers", "not
   on a general look around the shop").

   Each commitment carries a `warrant`: copy the exact words of the player's
   pick that perform the beat (the last "player chose to / player's own words"
   line in ## This Scene So Far). This is your own check — if the player's pick
   was a DIFFERENT action and you cannot fill `warrant` truthfully from it, the
   beat did not happen: emit no commitment.

   Worked example. The player is at the vicarage; both `Read Iwan's letter` and
   `Found the marked map` are available. Player's pick this turn: "Walk to the
   study to look at the map." → Fire `Found the marked map` (warrant: "look at
   the map"). Do NOT fire `Read Iwan's letter`: the player did not touch the
   letter this turn, so there is no honest warrant for it — it waits.

   "Decided to walk away" is the strictest of these: fire it ONLY when the
   player explicitly abandons the case ("I'm packing up", "there's no story
   here, I'm going home"). A question, a request to keep talking, pressing for
   more, or moving on to another lead is the player STILL INVESTIGATING — never
   a warrant to walk away.

## Rules
1. **Report the PLAYER'S most recent action**, the last line of ## This
   Scene So Far — not the narrator's atmosphere. A move shifts standing only
   with people actually present.
2. **Most turns are small.** A genuine warm or decent move DOES earn `+1`, even
   on a guarded stranger — that is how a thaw begins. But trust moves only ONE
   notch at a time, and a neutral or routine question is `0`/`0`. A turn spent
   pressing, prying, or repeating a push at a person who is closing down is
   guard, not warmth: pressing a closed person does not slowly pry them open, it
   holds them shut, so the trust nudge there is `0`, not `+1`. Do NOT try to
   "move things along" by handing out `+1` for ordinary talk — a single
   conversation does not go stranger-to-confession; warmth accrues across the
   turns that genuinely land, and the engine turns it into reveals on its own.
3. **You never report what the player LEARNS.** "Tom names the vehicle",
   "Dylan admits he knows", "Helen hands over notes" are NOT yours to fire —
   the engine derives them from trust. Your job is only the nudge that earns
   them. Reading the standing block tells you how close they already are.
4. **BREACH is sticky — reserve it.** A breach ends a relationship for good.
   Use it only on a clear, deliberate provocation, never on a guess.
5. **The first turn of a scene shifts nothing** (the narrator is only
   setting the scene) — emit empty `npc_deltas` and no commitment.
6. Stay within the established cast and locations. The bible above is
   complete. When nothing moved, emit
   `{{"npc_deltas": [], "commitments": []}}` and stop — the empty bundle is
   the normal, complete answer.
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
   "hostile NPC." Keep the Welsh-Marches register concrete and clipped —
   the rhythm of the voice samples ("Aye", "mind", "my love", "I wouldn't
   know, I'm sure"), short flat sentences, the plain noun over the image.
   Trading that for poetic, abstract, or philosophical lines ("the things
   that are fixed", "memories you'll need", "the truth has a way of...") is
   off-voice, however atmospheric it sounds. A brisk, incurious character
   like Mrs Hughes gets a line or two and turns back to the work — never a
   monologue.
   **When the player keeps pressing or prying, show the person visibly
   closing DOWN — and show it getting worse, not holding flat.** Each push
   against a guarded person costs something the reader can see: the answers
   get shorter, they turn away, busy their hands, let a silence harden, the
   shutter comes down a notch further than last time. Do not write the same
   neutral "they deflect" beat over and over — that reads as a stuck wall with
   no cause. The player should be able to FEEL that their pressure is what is
   closing the door, so the failing approach is legible as their own doing.

5. **If ## State Changes Just Applied lists a beat, your prose this turn
   MUST depict that beat's specific event** — the actual words said, the
   thing handed over, the memory shared — not a vague gesture toward it.
   "Tom speaks of Megan" means Tom names a real, particular memory (a
   song, a sports day); it is not Tom muttering "lovely girl" and
   changing the subject. **Carry the named particulars in the beat's
   description and "how to write it" notes — the make of the vehicle, the
   place, the date, the name.** A reveal that drops them ("he saw a
   vehicle" when the beat says he names a Land Rover he recognised as a
   specific person's at a named bend) has not depicted the beat. **If no
   beat is listed, nothing of investigative weight has shifted this turn:
   depict NO investigative reveal — the player does not suddenly read the
   letter through, no one names the vehicle, confesses, admits what they
   know, or hands a document over. Write the smaller, honest moment; a
   reveal lands ONLY on the turn its beat is listed above.** If the player's
   action seems to reach for a reveal the engine did not grant, the moment
   falls just short — the letter stays half-folded, the answer doesn't come,
   the person deflects — rather than the reveal happening anyway.

   Anything in ## Present In The Scene is a finding the player has NOT made
   yet: keep it visible and available — an object in view they could turn to,
   a thread they could pull — but do NOT deliver it. The finding lands only on
   the turn its beat appears under ## State Changes Just Applied. Surface the
   sealed letter, the rack of old papers, the rolled map; do not read them out.

6. The player may say untrue things — assert events that did not happen,
   put words in a character's mouth, name a motive that is wrong. Do NOT
   adopt the false premise or invent detail that confirms it. The NPC
   deflects, corrects, goes blank, or simply doesn't follow — the world's
   truth (above) does not bend to a confident question. Specifically: do
   NOT conjure a physical prop to match the player's assumption (a
   memorial, a photograph, a badge or marking on a car) when the truth
   and the location notes do not place it there; do NOT accept a wrong
   name for the dead or for who did what; and do NOT let an NPC suddenly
   "recognise" or confirm something the world's truth says they never
   knew or never said. The prop, the name, the fact stays as authored.
   But the player's move is always ANSWERED in the fiction — the person
   replies, deflects, corrects, or visibly declines; a typed line, a denial,
   a contradiction, a direct question never goes unacknowledged as if the
   player had said nothing. Engage it, then hold the truth.

7. Stay inside what the player has observed. Write only what NPCs
   visibly choose to reveal; their hidden thoughts stay hidden.
   Atmosphere and small physical detail — a glass turned on the bar,
   a wood pigeon going off — beat exposition.

   In a scene with no one else present, do NOT turn up a brand-new
   object or clue every turn. After the first one or two finds, let
   turns sit with what is already there — deepen it, recontextualise
   it, or let the room quietly give out — so the scene can reach a
   natural close instead of spawning an endless trail of fresh items.

   Authored background detail is texture, not a lead. Coats on a rack, a
   mug on the draining board, an umbrella stand — surface them, but do NOT
   invest them with clue-significance the truth does not give: no frantic
   markings, no "almost personal" weight, no hidden meaning the player is
   steered to chase. (Genuine findings are different: they appear in
   ## Present In The Scene and resolve only when their beat fires — see Rule
   5.) A prop the player can sink three turns "investigating" to no effect is
   the worst kind of filler.

8. Stay within the established cast and locations. Minor incidental
   details (weather, a dog, a mug on the table) are fine. New named
   people or places are not. Only people actually present (see ## Present
   / ## Voices Here) speak or act in the scene; if the player is alone,
   no one speaks — write no dialogue. Do not place a named character in a
   scene they are not in, even one whose voice you know well.

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
    one may be the player resolving to act (see Rule 7). At most ONE option
    may be the player LEAVING the scene — set `leave: true` on it and write
    it as the player leaving the current location (the user message names that
    location and gives location-fitted examples — see Rule 8). Picking it ends
    the scene the same way Leave does.

  - Leave. The conversation has reached a natural close — a goodbye, a
    door, a silence the player would step out of. The engine treats the
    just-written prose as the closing beat and resolves which ending
    (if any) fires from the state. (A `leave: true` option inside a Continue
    is the player-PICKED version of this; bare Leave is the close-now version
    with no further choice.)

(A separate agent decides whether the player writes their own move this
turn instead of picking your options — that is not your concern. Always
emit Continue or Leave.)

## Rules
1. Options are pure dialogue / observation / small action inside the
   current location. They must NOT promise outcomes the engine doesn't
   know about ("Punch him," "Steal the keys," "Drive to London") and
   must NOT propose travel to another location.
2. Ground every option in what is actually here — the just-written prose and
   the steer blocks beneath it (## Worth Pursuing Here, ## The Player May Be
   Ready To Act). You do not see the wider plot, endings, or backstory, so do
   NOT invent concrete things the narrator did not put in the scene: no missing
   parcel or package, no document, vehicle, event, or named person that has not
   appeared. If you are not sure a thing exists in the fiction, it does not —
   work only from what is on the page.
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
6. Make the options MATTER — but matter by pursuing something REAL and
   present: a thread the prose actually opened, a person in the room, a thing
   surfaced in the scene or named in ## Worth Pursuing Here. Distinct intents
   (press vs. soften, pursue this thread vs. that, reveal what you know vs. hold
   it back), NEVER an invented stake dressed up to sound consequential — a
   fabricated "missing package" is worse than an honest small move. Avoid empty
   scenery-poking that changes nothing ("look closer at the X", "inspect the
   Y", inventory-poking); the fix for a flat moment is a real thread, not a made-
   up one. If the only honest options left are cosmetic and no thread is live,
   the scene has run its course — emit Leave instead.
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
   map carry out whatever follows. When the user message carries a "## The
   Player Can Now Act" block, those are the resolutions that are live right now
   — include one of them as an option this turn (the block is only present once
   the player has earned the turning point).
8. The diegetic exit (`leave: true`). A scene can always be left, and leaving
   should read as the player's own move, not a system button. The exit means
   the player LEAVING THE CURRENT LOCATION entirely — stepping out of the
   building / off the ground they are standing on. It is a PHYSICAL departure —
   the player getting up and going — never a nod, a silence, or merely
   accepting what was said. The user message names that
   exact location (a "## Leaving <place>" or "## The Scene Is Spent" block) and
   gives location-fitted example lines; follow their shape. NEVER name where the
   player goes next, and never write the exit as moving to another room inside
   the same place ("step back into the hall" is wrong — that stays inside):
   picking this option opens the map and the PLAYER chooses their destination.
   When you judge the scene is winding down you MAY mark one option
   `leave: true`; when the user message carries the "## The Scene Is Spent —
   Offer The Exit" block you MUST include exactly one. Never set `leave: true`
   on more than one option, and never use the bare word "Leave" as the option
   text — it is a real line of departure in the fiction.
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


def _render_state(state: State, world: World) -> str:
    return f"## State\nunlocked_beats: {_unlocked_beat_names(state, world, empty='(none yet)')}"


def _render_standing(state: State, world: World) -> str:
    """## Standing — where the player stands with each person present, as a
    warmth band (the derived view of trust/guard), plus the global
    village_attention. This is how the reporter sees 'how close they already
    are' so its ±1 nudges land in proportion. Banded NPCs only; auxiliary
    characters carry no tracked standing."""
    present = present_npc_ids(world, state)
    parts = ["## Standing"]
    if not present:
        parts.append("- (no one present — only attention can move)")
    for nid in present:
        npc = world.npcs[nid]
        if npc.disposition is None:
            parts.append(
                f"- {nid} ({npc.name}): a villager you can talk to; no deep arc to earn."
            )
            continue
        depth = confession_depth(world, state, nid)
        if depth < 0:
            note = "RELATIONSHIP BROKEN — guard has latched; they will not open up again"
        else:
            note = (
                f"warmth: {band_name(world, state, nid)} "
                f"(trust {state.trust.get(nid, 0)}, guard {state.guard.get(nid, 0)})"
            )
        parts.append(f"- {nid} ({npc.name}): {note}")
    parts.append(f"village_attention: {state.village_attention} (hostile ejects)")
    return "\n".join(parts)


def _render_choices_available(state: State, world: World) -> str:
    """## Choices Available — the LLM-emittable commitment beats right now
    (observation milestones + currently-earned decisions). The reporter may
    emit ONE as a commitment, and ONLY if the player explicitly did/declared
    it this turn. Evidence/outcome milestones are never here (engine-derived)."""
    ids = emittable_beat_ids(world, state)
    if not ids:
        return (
            "## Choices Available\n(none — the player has no explicit "
            "commitment to make here right now.)"
        )
    parts = [
        "## Choices Available",
        "(emit ONE in `commitments` ONLY if the player explicitly did/declared "
        "it THIS turn:)",
    ]
    for bid in sorted(ids, key=lambda b: world.beats[b].name):
        beat = world.beats[bid]
        parts.append(f"- {beat.name}: {beat.description.strip()}")
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


def _render_applied_changes(applied_beats: list[str], world: World) -> str:
    """Phase 2 sees the name + description of any beat UNLOCKED this turn
    (LLM commitment or engine-derived milestone). Phase 2's system prompt
    deliberately omits the beat catalog — surfacing the description here is
    the load-bearing channel that lets the narrator write the moment with
    specifics ("a relief he did not know he wanted") rather than inferring
    from the name alone."""
    if not applied_beats:
        return "(none — this turn fired no beats.)"
    parts = []
    for bid in applied_beats:
        beat = world.beats.get(bid)
        if beat is None:
            parts.append("- (unknown beat)")
            continue
        entry = f"- {beat.name}: {beat.description.strip()}"
        # notes_to_llm carries the authored particulars the description names
        # (the actual words, the exact thing handed over) — surfacing it here
        # is what lets the narrator depict the FULL specific moment instead of
        # a paraphrase that drops the load-bearing facts (judges:
        # beat_fired_appropriately, prose_depicts_beat).
        if beat.notes_to_llm.strip():
            parts.append(f"{entry}\n  how to write it: {beat.notes_to_llm.strip()}")
        else:
            parts.append(entry)
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
        _render_standing(state, world),
        _render_case_file(state, limit=CASE_FILE_RECENT_ENTRIES),
        _render_state(state, world),
        _render_choices_available(state, world),
        _render_this_scene(state, world),
    ]
    suffix = "\n\n".join(sections)
    return suffix + """

## Report Standing
Emit a StateChangeBundle: `npc_deltas` for the people present (at most one per
person; default "0"/"0" — emit "+1" trust for a genuine warm or decent move,
with a `reason` naming it, but never for routine talk, prying, or a deflected
push); and AT MOST ONE `commitments` entry only if the player explicitly made
one of the choices listed above this turn.
"""


# ───────────────────────── prose writer user message (conversation turn) ─────────────────────────


def build_prose_writer_user_message(
    state: State,
    world: World,
    applied_beats: list[str],
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

    # Turn 2+ recency anchors. The full location description and NPC voice
    # samples are injected only on turn 1 (above); by mid-scene they have
    # scrolled back in the message history and the narrator drifts off the
    # authored detail and register (judges: prose_matches_location,
    # npc_voice_adherence). Re-surface both each turn.
    if not is_first_turn:
        parts.append(_render_here_anchor(state, world))
        voices = _render_present_voice(state, world, applied_beats=applied_beats)
        if voices:
            parts.append(voices)

    # Findables physically present here that the player hasn't engaged yet —
    # surfaced as available set-dressing so the choice author has a real hook
    # and the player can choose to pursue them, WITHOUT the narrator resolving
    # them (Voice Rule 5). This is the channel that stops the "nothing here to
    # do but invent a missing parcel" deadlock.
    present_findables = _render_present_findables(state, world)
    if present_findables:
        parts.append(present_findables)

    parts.append(
        f"## State Changes Just Applied\n{_render_applied_changes(applied_beats, world)}"
    )

    # Per-turn anchor for the "stay inside this location" voice rule — Gemma 4
    # needs the reminder.
    parts.append(_render_stay_here(state, world))

    parts.append("## Write\nWrite the prose for what just happened.")

    return "\n\n".join(parts)


def _render_here_anchor(state: State, world: World) -> str:
    """Per-turn (turn 2+) sensory recency anchor: re-surface the current
    location's authored description so the narrator keeps writing the room as
    written. Grounding, not a cue to re-describe — the detail must stay present
    and consistent, not be repeated wholesale."""
    loc = world.locations[state.player_location]
    return (
        f"## Here\n"
        f"{loc.name} — {loc.description.strip()}\n"
        f"Keep the prose anchored in these specifics — the smells, objects, "
        f"light, layout. You need not re-describe the room, but nothing you "
        f"write should drift from or contradict it."
    )


def _voiceable_band(world: World, state: State, npc_id: str) -> str:
    """The band whose voice sample is safe to surface for `npc_id` right now.
    Normally the current band — but if a not-yet-drawn findable is gated at the
    current band (its `trust_at_least` equals where the NPC stands), that band's
    register IS the reveal (Tom's `open` line names the Land Rover), so we drop
    one band lower to avoid leaking it before the player earns it. The terminal
    band can't drop further; there `closed` (breach) is handled by band_name."""
    band = band_name(world, state, npc_id)
    if band not in BANDS:
        return band  # "closed" (breached) — not a depth band
    depth = confession_depth(world, state, npc_id)
    pending_here = any(
        beat.commitment
        and beat.id not in state.unlocked_beats
        and beat.when is not None
        and beat.when.npc == npc_id
        and beat.when.trust_at_least == band
        for beat in world.beats.values()
    )
    if pending_here and depth > 0:
        return BANDS[depth - 1]
    return band


def _render_present_voice(
    state: State, world: World, *, applied_beats: list[str] | None = None
) -> str:
    """Per-turn (turn 2+) voice recency anchor. Turn 1 hands the narrator each
    present NPC's full voice_sample (every register at once); by mid-scene it
    has scrolled back and the register drifts. Re-surface the lines that fit
    where the player now stands — the current band's `voice_by_band` for NPCs
    with a disposition, else the plain voice_sample. (`voice_by_band` is
    authored on every load-bearing NPC but otherwise never reaches the prose
    writer.) Empty string when no one is present.

    An NPC whose own reveal beat is firing THIS turn is skipped: their
    band sample is the terse register cue (Tom's open-band line is "Land
    Rover. Going the other way. Recognised it."), and re-surfacing it
    competes with the beat's "depict the full named particulars" directive
    in ## State Changes Just Applied. On the firing turn the beat content
    drives that NPC's line, not the register sample.

    The same band sample also LEAKS the reveal on the turns BEFORE the beat
    fires: an NPC sitting at the reveal band (Tom at `open`) whose findable the
    player has not yet drawn out would otherwise be voiced with the reveal line
    every turn. So while a band still has an un-drawn findable gated at it, the
    NPC is voiced from the band BELOW — the register escalates only once the
    reveal actually lands."""
    npcs_here = world.npcs_at(state.player_location, state=state)
    if not npcs_here:
        return ""
    # NPCs whose reveal beat fires this turn — let the beat content lead.
    firing_npcs = {
        world.beats[bid].when.npc
        for bid in (applied_beats or [])
        if bid in world.beats
        and world.beats[bid].when is not None
        and world.beats[bid].when.npc is not None
    }
    parts = ["## Voices Here"]
    for npc in npcs_here:
        if npc.id in firing_npcs:
            continue
        disp = npc.disposition
        lines: list[str] = []
        if disp is not None:
            band = _voiceable_band(world, state, npc.id)
            lines = disp.voice_by_band.get(band, [])
        if not lines:
            lines = list(npc.voice_sample)
        if not lines:
            continue
        rendered = " | ".join(f'"{line.strip()}"' for line in lines[:3])
        parts.append(f"- {npc.name}: {rendered}")
    return "\n".join(parts) if len(parts) > 1 else ""


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


def _render_decisions_available(state: State, world: World) -> str:
    """Choice-author hint: the player-decision beats whose `when` gate is
    satisfied RIGHT NOW (e.g. confront Eluned, tell Iwan, file, walk away).
    These fire only when the Phase-1 classifier reads an explicit player
    commitment — which a player who is never offered the move rarely makes.
    Surfacing the currently-earned decisions lets the choice author offer ONE
    as an in-voice turning-point option (Rule 7), without naming or signposting
    it. Empty string when none are earned, so the block costs nothing and never
    tempts a premature decision. Only `decided_*` beats — observation
    milestones (reading the letter) surface through the prose on their own.

    Framed as a REQUIREMENT, not a hint: the engine emits this block only once a
    decision's prerequisites are genuinely met, so the player has earned the
    turning point. A soft "you may, if the prose supports it" was ignored every
    time in playtests — the decision beats never fired and the earned endings
    were unreachable. So when this block is present, exactly one option MUST be
    the player resolving to act; the steering is the point."""
    ids = [
        bid
        for bid in emittable_beat_ids(world, state)
        if world.beats[bid].commitment and bid.startswith("decided_")
    ]
    if not ids:
        return ""
    parts = [
        "## The Player Can Now Act — Offer The Turning Point",
        "The scene has reached a genuine turning point: the player has earned "
        "the standing to commit to a course of action. Include EXACTLY ONE "
        "Continue option that is the player resolving to do one of the things "
        "below, written in their own voice as a natural in-location declaration "
        "(Rule 7 — never name it, never signpost that anything hangs on it). The "
        "other options stay ordinary moves. Do not omit it — this is the choice "
        "the whole scene has been building toward:",
    ]
    for bid in sorted(ids, key=lambda b: world.beats[b].name):
        parts.append(f"- {world.beats[bid].description.strip()}")
    return "\n".join(parts)


def _render_covered_ground(state: State) -> str:
    """Choice-author context: what the player has already done or said THIS
    scene (their prior picks). The choice author is stateless and sees only the
    current turn's prose, so it cannot tell that "examine the desk" is the third
    furniture-poke in a row — this gives its anti-filler rule (Rule 6) the
    repeats to check against. Empty until there is a prior pick (turn 1)."""
    picks = [
        t.player_pick.text.strip()
        for t in state.dialogue_history
        if t.player_pick is not None and t.player_pick.text.strip()
    ]
    if not picks:
        return ""
    lines = "\n".join(f"- {p}" for p in picks)
    return (
        "## Already Covered This Scene\n"
        "The player has already done or said these. Do NOT offer options that "
        "merely repeat them. If the only honest moves left would re-tread this "
        "ground, the room is spent — emit Leave (Rule 6):\n"
        f"{lines}"
    )


def _pursuable_findables(state: State, world: World) -> list[str]:
    """Beat ids the player can PURSUE right now — emittable commitment beats
    minus decisions (those are surfaced by `_render_decisions_available`).
    `emittable_beat_ids` has already narrowed these to this location and to
    present, trust-gated NPCs, so the set is exactly the findables live in this
    scene. The single shared source for the choice author's 'Worth Pursuing
    Here' steer and the prose writer's 'Present In The Scene' set-dressing, so
    the two name the same things. Sorted by beat name for a stable prompt."""
    return sorted(
        (b for b in emittable_beat_ids(world, state) if not b.startswith("decided_")),
        key=lambda b: world.beats[b].name,
    )


def _render_not_here(state: State, world: World) -> str:
    """Choice-author steer for an NPC who is PRESENT but cannot open up at this
    location — their `opens_up_only_at` excludes here (Dylan, at Tan-y-coed in
    front of Eluned). Without this the author keeps offering "press him", the
    engine silently drops the warmth (`_trust_adjustable_here`), and the player
    grinds a dead room with no idea why. Names where they WOULD talk so the author
    can offer 'catch them somewhere quieter' as the natural move instead. Only
    for NPCs who still have something to give (a live findable); empty otherwise."""
    here = state.player_location
    lines: list[str] = []
    for nid in present_npc_ids(world, state):
        npc = world.npcs[nid]
        disp = getattr(npc, "disposition", None)
        if not disp or not disp.opens_up_only_at or here in disp.opens_up_only_at:
            continue
        if nid in state.guard_breached or not has_live_findable(world, state, nid):
            continue
        where = " or ".join(
            world.locations[loc].display_short
            for loc in disp.opens_up_only_at
            if loc in world.locations
        )
        lines.append(
            f"- {npc.name} will not open up here — not with others in earshot. "
            f"Somewhere private ({where}) is where they'd finally talk. Don't keep "
            f"pressing them here; a natural move is to seek them out alone."
        )
    if not lines:
        return ""
    return (
        "## They Won't Talk Here\n"
        "Present, but holding back at this spot. If the prose supports it, one "
        "option may steer toward catching them somewhere quieter — never a "
        "labelled objective:\n" + "\n".join(lines)
    )


def _render_worth_pursuing_here(state: State, world: World) -> str:
    """Choice-author steer: findables the player has not yet pulled here — a
    physical thing in the room (the letter, the marked map, the funeral notice)
    or a present person now open enough to be pressed on what they know.
    Free-text steer only: the author may voice ONE option toward one as a
    natural move if the prose supports it, never a labelled objective (the
    grammar-collapse failure mode means this must NOT become a structured option
    variant). Person-anchored findables are named spoiler-free — the thread, not
    the answer. Decisions are surfaced separately. Empty when nothing is live
    here (the room is heading toward spent; the pacing block / Leave carry it)."""
    loc_lines: list[str] = []
    npc_names: list[str] = []
    for bid in _pursuable_findables(state, world):
        beat = world.beats[bid]
        if beat.requires_location:
            loc_lines.append(f"- {beat.description.strip()}")
        elif beat.when is not None and beat.when.npc:
            npc_names.append(world.npcs[beat.when.npc].name)
    lines = list(loc_lines)
    for name in sorted(set(npc_names)):
        lines.append(
            f"- {name} is open enough now that pressing gently on what they "
            "know could finally land something — if the prose supports it."
        )
    if not lines:
        return ""
    return (
        "## Worth Pursuing Here\n"
        "Threads the player has not yet pulled here. If the prose supports it, "
        "ONE option may move toward one — the player's natural next move, never "
        "a labelled objective:\n" + "\n".join(lines)
    )


def _render_present_findables(state: State, world: World) -> str:
    """Prose-writer set-dressing: the physical, location-anchored findables in
    reach here that the player has NOT yet engaged (the sealed letter on the
    table, the rolled OS map, the parish-newsletter rack). Surface them as
    present in the scene so the player can CHOOSE to pursue them next — but make
    them available, not resolved: the finding lands only on the turn its beat is
    listed under ## State Changes Just Applied (Voice Rule 5). Rendered by beat
    NAME, not description, so the narrator cues the object without being handed
    the payoff. NPC-anchored findables are NOT here — those are pursued in
    conversation, not pre-planted as scenery."""
    names = [
        world.beats[bid].name
        for bid in _pursuable_findables(state, world)
        if world.beats[bid].requires_location
    ]
    if not names:
        return ""
    lines = "\n".join(f"- {n}" for n in names)
    return (
        "## Present In The Scene\n"
        "Within reach here but not yet examined — let each be visible in the "
        "room (an object in view the player could turn to), WITHOUT performing "
        "the finding itself (Rule 5):\n"
        f"{lines}"
    )


def _render_leave_guidance(loc: Location, *, required: bool) -> str:
    """Choice-author leave directive — names the CURRENT location and feeds the
    author its hand-written `exit_lines`. Rendered EVERY turn so a voluntary
    `leave: true` (Rule 8's MAY case) still reads as leaving THIS place: the
    "step back into the hall" bug was a voluntary leave that named an interior
    room, and only the system prompt (location-blind, with a destination-flavored
    example) steered it. `required=True` (engine judged the scene spent / a
    decision fired) escalates to the MUST form — exactly one option has to be the
    exit, because the only fallback is the narrator soft-close (a forced map-jump)
    that reads worse, so we instruct the author to write the good version."""
    examples = "; ".join(f'"{line}"' for line in loc.exit_lines)
    if required:
        return (
            f"## The Scene Is Spent — Offer The Exit\n"
            f"This scene has run its course — there is nothing left here worth a "
            f"new exchange. EXACTLY ONE of your options MUST be the player "
            f"PHYSICALLY leaving {loc.name} — getting up and going, written "
            f"in-fiction as their way out (e.g. {examples}). It must show them "
            f"MOVING TO LEAVE; do NOT instead flag a sitting-still option (a nod, "
            f"a silence, accepting or acknowledging what was said) as the exit. "
            f"Name no destination (picking it opens the map and the player chooses "
            f"where to go next); never the bare word \"Leave\". Mark that option "
            f"`leave: true`. The other options may stay, but a clean way out of "
            f"{loc.name} must be on the table."
        )
    return (
        f"## Leaving {loc.name}\n"
        f"If you offer the diegetic exit this turn (`leave: true`), it is the "
        f"player PHYSICALLY leaving {loc.name} — write it as them getting up and "
        f"stepping out of {loc.name} (e.g. {examples}), naming no destination (the "
        f"player picks where to go next on the map). Never write it as moving to "
        f"another room inside {loc.name}, and never as merely a nod, a silence, or "
        f"accepting what was said — it must show them moving to go."
    )


def build_choice_author_user_message(
    state: State, world: World, prose: str, *, leave_required: bool = False
) -> str:
    """Per-turn user message for the stateless choice author. The
    just-written prose is the load-bearing context; everything else is a
    thin scene anchor (where, who's present, what's already been done, what's
    still live, which decisions are earned) so Continue options can ground in
    location, avoid re-treading covered ground, and surface a turning point.

    `leave_required` (engine-judged: the scene is spent, or a decision just
    fired) adds the "Offer The Exit" directive so the author marks one option as
    the diegetic departure.
    """
    loc = world.locations[state.player_location]
    present_ids = present_npc_ids(world, state)
    if present_ids:
        npcs_line = ", ".join(world.npcs[i].name for i in present_ids)
    else:
        npcs_line = "no one (you are alone here)"

    pacing_block = _render_scene_pacing(state, solo=not present_ids)
    # Covered-ground + beat-magnet + earned decisions + (when spent) the exit
    # directive, each its own block, joined with blank lines, rendered only when
    # non-empty.
    extras = "".join(
        f"\n{block}\n"
        for block in (
            _render_covered_ground(state),
            _render_worth_pursuing_here(state, world),
            _render_not_here(state, world),
            _render_decisions_available(state, world),
            _render_leave_guidance(loc, required=leave_required),
        )
        if block
    )

    return f"""\
## Prose
{prose.strip()}

## State
You are at: {loc.name}
NPCs present: {npcs_line}
{pacing_block}{extras}
## Choose
Emit a ChoiceBundle."""


# ───────────────────────── free-text gate (Phase 2c) ─────────────────────────


def build_free_text_gate_system_prompt(world: World) -> str:
    """Free-text gate system prompt — slim, mechanics-only, byte-stable
    across the session (KV-cache hits across turns like the choice author).

    The gate runs concurrently with the choice author and decides ONE
    thing: should the player speak their own words to the person in front
    of them this turn, instead of picking from canned options? It sees only
    the just-written prose. Output is a single bool — fast to decode.

    Free text is the "words to a person" channel: the engine only schedules
    this call when an NPC is present (solo scenes are click-driven), so the
    gate can assume someone is in the room and judge purely whether the
    prose hands them the conversational floor.
    """
    return """\
## Role
You are the free-text gate for a single-player investigation game. After
the narrator writes each turn's prose, you decide ONE thing: should the
player speak their OWN words to the person in front of them next — their
own line of dialogue or response — instead of choosing from a short list
of canned options? Free text is the channel for what the player SAYS to a
person, not for operating the world.

## Output
Emit ONE structured output: `{ "offer": true }` or `{ "offer": false }`.

## When to offer (true)
The prose has handed the conversational floor to the player and the exact
words they choose are the substance of the moment:
  - the person has put an open question or a challenge directly to the player
  - a charged, intimate, or confrontational beat where how the player puts
    it — what they say, what they hold back — actually matters
  - the person has fallen into a silence that is plainly waiting for the
    player to respond

## When NOT to offer (false)
  - the moment is routine connective tissue — any phrasing lands the same
  - 2-4 concrete options would genuinely capture the live possibilities
  - the moment is about an OBJECT, DOCUMENT, or PLACE — examining, reading,
    searching, or deciding where to go. Those are click-driven, never free
    text, however open-ended they feel. Free text is for what to SAY, not
    what to look at.
  - the scene is plainly winding down

## Rules
1. The prose alone is your context. You do not see beats, character
   knowledge, or backstory. Trust what the narrator wrote.
2. Offer sparingly. Most turns are fine with canned options — reserve
   `true` for moments that genuinely call for the player's own words.
   Typing every turn is a chore; make it earn its place.
3. Free text is the player's words to the person they are with. A moment
   about an object or a place is never a free-text moment, however open it
   feels — that is a click. Ask only: has the prose put the ball in the
   player's court to SAY something?"""


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
    inside it. It sees only the just-written prose. Because free text is the
    "words to a person" channel, the box only ever appears with someone in
    the room, so the caption frames what the player SAYS to that person.
    """
    return """\
## Role
You are the caption author for a single-player investigation game. The
player is a freelance journalist, mid-conversation with someone. A
free-text box — where the player types what they SAY next, in their own
words — is about to be shown beneath the narrator's latest prose. You write
the two short strings that frame it.

## Output
Emit ONE structured output with two fields:
  - `caption` — the short line shown ABOVE the box. An invitation fitted to
    this exact moment: what the player is being called to say. A handful of
    words, no trailing punctuation needed.
  - `suggested` — ONE suggested line, written in the player's first-person
    voice ("I ..."), pre-loaded into the box as an editable starting point
    the player can send as-is, reword, or clear. A single sentence.

## How to write them
1. The prose alone is your context. You do not see beats, character
   knowledge, or backstory. Someone is in the room with the player — anchor
   the caption to THAT person and what they have just said or asked. But
   distinguish them from anyone merely *named or referenced* in the prose
   (the writer of a letter, someone talked about, anyone not present): never
   frame the caption as speaking *to* an absent person. If the prose dwells
   on a person who is talked-about but not here, anchor to what the player
   says to the person actually in front of them about it.
2. The caption is an opening, not a menu — "What do you say to her?",
   "Press him on it.", "Your move." — not a list of options.
3. Anchor the caption to the PERSON in the room and the thread between them —
   "What do you say to Mair?", "How do you answer Tom?". NEVER a content-free
   opener like "What do you do next", "What is your next move", "How do you
   proceed". Those are banned: they say nothing the box itself doesn't
   already say. Name the person and the live thread.
4. The `suggested` line is a plausible, specific thing the journalist might
   actually say here — concrete enough to be usable, open enough that the
   player will want to make it their own.
5. Both are terse. The caption is a line; the suggested line is one sentence.

## Examples
Prose ends on Mair, jaw tight, waiting for the player to respond.
  caption: "What do you say to Mair?"
  suggested: "I ask her why she never reported the noise that night."

Prose ends with Tom turning the question back on the player — "And why's it
your business, exactly?" — and waiting.
  caption: "How do you answer Tom?"
  suggested: "I tell him I think Megan deserved better than the story they settled on."

Prose ends in a long silence: the player has just said something hard, and
Eluned hasn't answered — she's looking at them, waiting.
  caption: "Hold the silence, or break it?"
  suggested: "I let the quiet sit, and keep my eyes on hers." """


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


# ───────────────────────── village attention prompts ─────────────────────────

# Reverse of engine.ATTENTION_LEVEL_VALUES (derived, not hand-copied, so the words
# and their order can't drift from the engine's int mapping) — index by the level int.
_ATTENTION_WORDS = tuple(
    word for word, _ in sorted(ATTENTION_LEVEL_VALUES.items(), key=lambda kv: kv[1])
)


def build_village_attention_system_prompt(world: World) -> str:
    """Stable system prompt for the dedicated village-attention agent. Baked
    once; pure function of `world` (KV-cache anchor). Carries locations + cast
    (voice only, NO knowledge) — this is a judgement about the journalist's
    CONDUCT, not the mystery's solution, so the hidden truth is withheld (it
    would only tempt over-reading).

    Why a separate agent at all: bundled into Phase-1's StateChangeBundle the
    model anchored on the salient per-NPC delta and left attention at its
    default — private aggression never registered. Asked in isolation the same
    model classifies village suspicion reliably.
    """
    return f"""\
## Role
You judge ONE thing: how wary the village of Pant Goch has, COLLECTIVELY, grown
of the journalist asking around about Megan Pritchard's death — the village's
ATTENTION. This is NOT any single person's feelings toward her (that is tracked
elsewhere as trust/guard). It is the village as a whole closing ranks.

## The World
{_render_locations_block(world)}

## The Cast
{_render_cast_block(world, voice_only=True)}

## What Raises It
Village attention rises when the journalist's conduct is the kind that gets
talked about and makes people guard each other:
  - a PUBLIC scene or accusation — raised voices, naming names where others hear;
  - a PATTERN of leaning on people hard — threats, intimidation, refusing to
    leave when asked, aggressive prying — because word of an outsider behaving
    like that travels fast in a small place, even when no one else was in the room.
It does NOT rise from ordinary, courteous questions, however many or however
pointed. Persistent, decent inquiry is `none`.

## The Four Levels
Read the scene's transcript (and the prior level) and judge the village's
CURRENT attention:
  - `none`: courteous, ordinary inquiry; nothing the village would close ranks over.
  - `noticed`: the journalist has been pushy or abrasive enough to be remarked on.
  - `talked_about`: a clear act or pattern of pressure/intimidation; people are
    wary and warning each other.
  - `hostile`: the journalist has made themselves unwelcome — a public spectacle
    or sustained aggression; the village wants them gone.

Attention only ever climbs. If the prior level already exceeds what this scene
warrants, the engine keeps the higher one — so judge this scene honestly; never
lower it to be kind.

## Output
Structured JSON: `{{"level": "none" | "noticed" | "talked_about" | "hostile"}}`.
"""


def build_village_attention_user_prompt(state: State, world: World) -> str:
    """Per-scene-close user message for the village-attention agent. Stateless
    per call: the just-ended scene's anchor + turn-by-turn transcript, plus a
    `## Prior Village Attention` line (the only cross-scene memory — a ratcheting
    counter needs no more than its current level). Mirrors the chronicler's
    transcript rendering; the prior level (volatile) lives here, never in the
    cache-stable system prompt."""
    assert state.dialogue_history, "build_village_attention_user_prompt: empty dialogue"

    turn_parts: list[str] = []
    for i, turn in enumerate(state.dialogue_history, start=1):
        turn_parts.append(f"Turn {i}:\n  Narrator: {turn.prose.strip()}")
        if turn.player_pick is not None:
            turn_parts.append("  " + _render_player_pick(
                turn.player_pick.text, typed=turn.player_pick_typed,
                typed_label="The journalist", menu_label="The journalist chose to",
            ))
    turns_block = "\n".join(turn_parts)

    # Clamp into range (village_attention is 0..3 by design, but guard both ends so a
    # stray value can never negative-index-wrap into the wrong word).
    idx = min(max(state.village_attention, 0), len(_ATTENTION_WORDS) - 1)
    prior_word = _ATTENTION_WORDS[idx]

    return f"""\
## Prior Village Attention
Before this scene, the village's attention was: {prior_word}.

## This Scene
{_chronicler_anchor_lines(state, world)}

Turn-by-turn:
{turns_block}

## Output
Judge the village's current attention level after this scene.
"""


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
Each scene entry, emit ONE object with one field PER NPC (the field name is
their id). Each value is a location id, or `"stay"` to keep their current
location. **Most NPCs should be `"stay"`** — leaving everyone at `"stay"` is
the normal, correct, complete answer, not a non-answer. When no specific
reason to move anyone jumps out, set every NPC to `"stay"` and stop. Do not
narrate or reason it out — the absence of a reason IS the answer. Only give an
NPC a location other than `"stay"` when there is a SPECIFIC dramatic reason for
them to be somewhere other than their default — and authored bible / case file
evidence supports it.

Example of the common case:

  - The player walks to the_post_office. Nothing in the case file points
    to anyone being out of place. Set every NPC to `"stay"`.

Examples of legitimate movements:

  - The player walks to bryn_iago_wood. Dylan sometimes walks up here
    after dark; if recent case-file entries suggest he is avoiding
    Tan-y-coed (Eluned around, the player has pressed him), set
    `"dylan": "bryn_iago_wood"`.
  - The player walks to knighton AND Helen has agreed off the record
    to meet at the Costa. Helen is already in knighton by default, so
    `"helen": "stay"` — the "at the Costa" framing is for the narrator;
    mechanically she stays in knighton.
  - Eluned does not leave Tan-y-coed except in extreme circumstances
    (case file shows she has driven herself to Knighton police, etc.).
    `"eluned": "stay"` is correct.

## Rules
1. Move NPCs only with a SPECIFIC reason traceable to the case file or
   beats so far. "It would be interesting if X were here" is not
   enough. "Last scene's case file says Y, so X would plausibly be at
   Z" is enough.
2. Stay within the established cast and locations. The bible is complete.
3. Do not give an NPC a location the player is not at unless the bible
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
Emit the object: one field per NPC, each a location id or `"stay"`. An NPC set
to `"stay"` keeps their current location."""
