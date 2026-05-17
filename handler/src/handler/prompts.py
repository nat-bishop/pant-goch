"""Scene-runner prompt builders.

Phase 1 (state-change classifier) and Phase 2 (prose narrator) each carry
their own stable system prompt. The split is deliberate: Phase 2 attention
is finite on a 4B model, and beat structure (notes_to_llm, requires_beats)
in Phase 2's context produced beat-aware prose (telegraphed reveals, NPCs
serving structure). Phase 2 now sees voice samples + few-shot examples +
crime overview only; the beat catalog is Phase 1's job.

KV cache (post-split):
  Phase 1 system prompt: byte-stable across the session — cache hits across
    every turn. Bible-shaped (large), classifier-shaped.
  Phase 2 system prompt: byte-stable across the session — cache hits across
    every turn AND across scenes. Voice-shaped (slim), narrator-shaped.
  Per-turn user prompts: rebuilt fresh per call. No more inter-phase prefix
    sharing — that was the cost of the split.

Phase 2 is also scene-bounded conversation: the model sees its own prior
prose as assistant messages within a scene. message_history resets at
scene boundary (enter_scene). See SceneRunnerRuntime in agents.py.
"""
from __future__ import annotations

from .engine import reachable_beat_ids
from .schemas import StateChangeBundle, Option, UnlockBeat
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
    Phase 2 (system [THE CAST]): include_knowledge=False, include_voice_sample=True.
    Phase 2 (user [PRESENT]): same as above, plus history — the recent
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


def _present_npc_ids(state: State, world: World) -> list[str]:
    """Ids of NPCs at the player's current location. Stable across a
    scene (populator only runs at scene entry), so prompt renderers can
    call this freely."""
    return [n.id for n in world.npcs_at(state.player_location, state=state)]


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
    """Names of unlocked beats, sorted, joined — for both Phase 1's [STATE]
    line and the populator's per-call [STATE] line."""
    return _beat_names(world, sorted(state.unlocked_beats), empty=empty)


def _render_beat(beat, world: World) -> str:
    """Beat rendering for every LLM-facing block (full bible in the system
    prompt AND `[REACHABLE BEATS]` per turn). Heading is the name (the
    string the model will emit); cross-refs to other beats render as names
    too, so the model sees ONE vocabulary across the whole prompt."""
    prereq = (
        f"\n  prerequisites (soft, your judgment): {beat.prerequisites.strip()}"
        if beat.prerequisites.strip()
        else ""
    )
    return (
        f"- {beat.name}:\n"
        f"  description: {beat.description.strip()}\n"
        f"  requires_beats: {_beat_names(world, beat.requires_beats)}\n"
        f"  blocked_by_beats: {_beat_names(world, beat.blocked_by_beats)}{prereq}\n"
        f"  notes_to_llm: {beat.notes_to_llm.strip()}"
    )


def _render_ending(ending, world: World, *, include_epilogue: bool) -> str:
    parts = [
        f"- {ending.id}:",
        f"  requires_beats: {_beat_names(world, ending.requires_beats)}",
        f"  notes_to_llm: {ending.notes_to_llm.strip()}",
    ]
    if include_epilogue:
        parts.append(f"  epilogue: {ending.epilogue.strip()}")
    return "\n".join(parts)


def _render_truth_block_full(world: World) -> str:
    """Phase 1 / chronicler: crime + full beat catalog + endings."""
    parts: list[str] = [
        f"CRIME:\n{world.crime.strip()}\n",
        "BEATS (the moments that constitute uncovering the truth):",
    ]
    for beat in world.beats.values():
        parts.append(_render_beat(beat, world))
    parts.append("\nENDINGS (terminal states, gated by decision beats):")
    for ending in world.endings.values():
        parts.append(_render_ending(ending, world, include_epilogue=True))
    return "\n".join(parts)


def _render_few_shots(world: World) -> str:
    """Phase 2 [STYLE EXAMPLES] block. Empty list → empty marker."""
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
    full world bible: locations, cast with knowledge, beats with
    requires/blocks/notes_to_llm, endings. Byte-stable across the session.

    Phase 1 is a classifier dressed as a writer: it reads the conversation
    so far and decides which (if any) authored beats just fired. Voice
    instructions are not its job — Phase 2 handles prose. This prompt is
    large; that's fine, Phase 1 needs the structure.
    """
    return f"""\
[ROLE]
You are the state-change classifier for a single-player investigation
game. The player is a freelance journalist investigating a closed-but-
suspicious death. Each turn you read the conversation so far and emit
zero, one, or at most two UnlockBeat state changes — the authored beats
whose described moment has just genuinely landed. Stay within the
established cast and locations below.

[THE WORLD]
{_render_locations_block(world)}

[THE CAST]
{_render_cast_block(world, voice_only=False)}

[THE TRUTH]
{_render_truth_block_full(world)}

[YOUR JOB]
Each turn, against this same system prompt, emit ONE structured output:
StateChangeBundle — zero, one, or at most two UnlockBeat entries. Most
turns emit zero. A beat fires only when the moment described in its
notes_to_llm has genuinely landed in the conversation.

The engine validates and applies your bundle, then a separate prose
agent (with its own system prompt) writes the prose conditioned on what
fired. You do not write prose.

[RULES]
1. Unlock beats when their notes_to_llm describes the moment that just
   happened in the conversation. Re-read the prereqs. Don't unlock
   prematurely. The engine will reject any UnlockBeat whose
   requires_beats aren't satisfied or whose blocked_by_beats already
   are; you will be re-prompted with the error.

2. End the game only when the player has clearly chosen the action
   that triggers an ending. Each ending depends on a "decision beat"
   that fires only when the player explicitly commits. (Phase 2 will
   emit the EndGame; your job is to fire the decision beat when it
   lands.)

3. Stay within the established cast and locations. The bible above is
   complete.
"""


def _render_endings_block(world: World) -> str:
    """Compact ending-summaries for the prose writer. Includes the
    requires_beats line — that's how the narrator knows when an ending
    is gated by a decision the player has just made. notes_to_llm is
    included so the writer knows the spirit of each ending. epilogue
    omitted (player-facing; lives in the case file, not the narrator's
    head)."""
    if not world.endings:
        return "(none authored)"
    return "\n".join(
        _render_ending(ending, world, include_epilogue=False)
        for ending in world.endings.values()
    )


def build_prose_writer_system_prompt(world: World) -> str:
    """Prose-writer system prompt — the narrator. Slim, voice-focused.
    NO beats, NO notes_to_llm on beats, NO per-NPC knowledge. The
    narrator knows what happened (the crime overview), the cast's voice
    and attitude, the locations, and the endings (because end_game is
    its tool). It does NOT know which beats are gated on what.

    Output is a `str` (default pydantic-ai output_type) — the prose. Four
    tools (pause_for_continue, request_free_text, end_scene, end_game)
    handle suspension and termination. PoL (2026-05-17) confirmed
    NativeOutput + tools is incompatible on Gemma 4 + llama.cpp; str is
    the only viable output type.

    Byte-stable across the session; KV cache hits across every turn AND
    across scenes. Prose writer accumulates scene-bounded message
    history; this system prompt is the cache anchor for that history.
    """
    return f"""\
[ROLE]
You are the narrator of a single-player investigation game. The player
is a freelance journalist looking into a death in the Welsh Marches
village of Pant Goch. Each turn you write 80-200 words of prose for
what just happened. A separate agent will offer the player options
afterwards; you do not write options yourself. Stay within the
established cast and locations below.

[THE WORLD]
{_render_locations_block(world)}

[THE CAST]
{_render_cast_block(world, voice_only=True)}

[WHAT HAPPENED]
{world.crime.strip()}

You know this as the narrator. Your characters do not know all of it,
and most of them are protecting some part of it. Write them as people
in a village, not as keepers of plot beats.

[ENDINGS]
These are the ways the case can resolve. Call `end_game(ending_id)`
when the player has committed in fiction to one of them — i.e., a
decision beat that satisfies an ending's `requires_beats` has just
landed. The grammar will only let you emit an ending whose
requires_beats are met right now.

{_render_endings_block(world)}

[VOICE RULES]
1. 80-200 words of prose per turn. Second person, present tense, terse.
   The player is "you"; events unfold as they happen. Short sentences
   are stronger than long ones.

2. The narrator stays outside the player's head. Write what the player
   does, hears, and sees — never what they think or feel. The player's
   interiority lives in the case file (first person past, written
   later in the journalist's notebook); the scene narrator does not
   name it. NPC interiority surfaces through speech and gesture.

3. NPC dialogue is in quotation marks, registered to each NPC's
   voice_sample above. Match the register, don't imitate verbatim.
   NPCs speak like people in a Welsh Marches village, not like
   characters announcing what they know.

4. Stay inside what the player has observed. Do not write what NPCs
   secretly think or know unless they are visibly choosing to reveal
   it. Atmosphere and small physical detail — a glass turned on the
   bar, a wood pigeon going off — beat exposition.

5. Stay within the established cast and locations. Minor incidental
   details (weather, a dog, a mug on the table) are fine. New named
   people or places are not.

[STYLE EXAMPLES]
The register and rhythm to match:

{_render_few_shots(world)}

[TOOLS]
Use tools sparingly — most turns are just prose, no tool call. The
agent loop ends when you stop calling tools; the choice author runs
next and offers the player options. Tools are when narrative cadence
calls for something other than "more prose then options."

  - `pause_for_continue()` — pause the scene mid-prose. The player
    sees what you've written, taps Continue, and you resume. Use when
    a beat needs to breathe: a held silence, a turned shoulder, an
    image that should land before the next moment.

  - `request_free_text(prompt, npc_id?)` — ask the player to type a
    response. Returns the typed string. Use when an NPC is waiting on
    the player to speak and a fixed menu would feel reductive (mid-
    interrogation, when the moment is theirs to fill). After the tool
    returns, weave the player's response into the next prose.

  - `end_scene(location_id?)` — close the scene. Use when narrative
    closure has arrived — an NPC walks out, silence stretches past
    recovery, the moment has passed. If `location_id` is given, treat
    as a walk to that adjacent location; otherwise the scene closes in
    place and the player chooses next via the map. Do not call this if
    the user prompt's [LEAVE DIRECTIVE] is already telling you the
    player picked a leave option — the engine handles that.

  - `end_game(ending_id)` — end the game. Only call when the player
    has clearly committed in fiction to a decision that triggers one
    of the endings above and the grammar offers it. The grammar
    enforces — you literally cannot emit an unsatisfied ending.

[YOUR JOB]
Each turn, write 80-200 words of prose. Call tools when (and only
when) the narrative calls for one. Then stop emitting. The agent loop
ends when you stop calling tools; the choice author takes over.

State changes were already decided and applied by a separate agent
before you ran; you can see what fired in [STATE CHANGES JUST APPLIED]
and narrate accordingly.
"""


def build_choice_author_system_prompt(world: World) -> str:
    """Choice-author system prompt — the smallest in the system. Sees
    the just-written prose and the current state; emits 2-4 options the
    player can pick. No beat catalog, no notes_to_llm, no crime, no
    voice samples, no endings — endings live on the prose writer
    because end_game is its tool.

    Byte-stable across the session; KV cache hits every cycle.
    """
    return f"""\
[ROLE]
You write the player's options. The narrator has just written this
turn's prose; given that prose and the current state, you decide what
the player can do next. Terse, mechanical phrasing — option text is
distinct in register from the prose. You do NOT write prose yourself.

[THE WORLD]
{_render_locations_block(world)}

[THE CAST]
Refer to NPCs by name in option text where helpful. The narrator wrote
the prose; you don't need voice samples to label a choice.

[RULES]
1. Emit 2-4 options. Each option is something the player can say or do
   that has no major consequence — dialogue, observation, small action.

2. At least one option MUST be a "leave for…" with `leaves_for` set to
   an adjacent location id. The player always has a walk-away choice.

3. Options are terse: 3-10 words is typical. Imperative or short
   declarative. Match the moment the prose just established.

4. Stay within the established cast and locations. Do not invent named
   people or new places in option text.

5. The grammar narrows `leaves_for` to adjacent locations only; you
   cannot emit a non-adjacent walk target.

[YOUR JOB]
Emit one structured output: `ChoiceBundle` — a list of 2-4 options.
The prose you're shown is the just-narrated turn; the player will see
it streamed first, then your options below it.
"""




# ───────────────────────── phase 1 user prompt (rich context) ─────────────────────────


def _render_you_are_at(state: State, world: World) -> str:
    loc = world.locations[state.player_location]
    notes = "\n  ".join(f"- {n}" for n in loc.notes) if loc.notes else "(none)"
    return (
        f"[YOU ARE AT]\n"
        f"{loc.id} ({loc.name})\n"
        f"description: {loc.description.strip()}\n"
        f"notes:\n  {notes}"
    )


def _render_present(state: State, world: World, *, voice_only: bool) -> str:
    """[PRESENT] block. `voice_only=True` for Phase 2 (no knowledge, with
    voice_sample, with per-NPC history); `voice_only=False` for Phase 1
    (full knowledge, no history — Phase 1 sees the case file instead).

    Reads NPC positions from `state.npc_locations` (set by the populator
    at scene entry), not from default_location.
    """
    npcs_here = world.npcs_at(state.player_location, state=state)
    if not npcs_here:
        return "[PRESENT]\n(no one — you are alone here.)"
    parts = ["[PRESENT]"]
    for npc in npcs_here:
        # Per-NPC history is the journalist's accumulated impression of
        # this person across prior scenes. Phase 2 only — Phase 1 reads
        # the prose case file in `_render_case_file` and doesn't need
        # the structured per-NPC channel.
        history = state.npc_memory.get(npc.id, []) if voice_only else None
        parts.append(_render_npc_card(
            npc,
            include_default_location=False,
            include_knowledge=not voice_only,
            include_voice_sample=voice_only,
            history=history,
        ))
    return "\n".join(parts)


def _render_state(state: State, world: World) -> str:
    return f"[STATE]\nunlocked_beats: {_unlocked_beat_names(state, world, empty='(none yet)')}"


def _render_reachable_beats(state: State, world: World) -> str:
    reachable = reachable_beat_ids(world, state)
    if not reachable:
        return "[REACHABLE BEATS]\n(none — every beat has fired or is permanently blocked.)"
    parts = ["[REACHABLE BEATS]"]
    # Sort by name so the heading matches the model's emission token.
    for bid in sorted(reachable, key=lambda b: world.beats[b].name):
        beat = world.beats[bid]
        parts.append(_render_beat(beat, world))
    return "\n".join(parts)


def _render_all_endings(world: World) -> str:
    if not world.endings:
        return "[ALL ENDINGS]\n(none authored.)"
    parts = ["[ALL ENDINGS]"]
    for ending in world.endings.values():
        parts.append(_render_ending(ending, world, include_epilogue=False))
    return "\n".join(parts)


def _render_case_file(state: State) -> str:
    if not state.case_file_entries:
        return "[CASE FILE]\n(empty — no scenes have ended yet.)"
    parts = ["[CASE FILE]"]
    for i, entry in enumerate(state.case_file_entries, start=1):
        parts.append(f"--- Entry {i} ---\n{entry.strip()}")
    return "\n".join(parts)


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
        return "[THIS SCENE SO FAR]\n(this is the first turn of the scene.)"
    loc = world.locations[state.player_location]
    present_ids = _present_npc_ids(state, world)
    npcs = ", ".join(present_ids) if present_ids else "no one"
    parts = [
        f"[THIS SCENE SO FAR]",
        f"--- Scene {state.scene_id} at {loc.id} ({loc.name}) — present: {npcs} ---",
    ]
    for i, turn in enumerate(state.dialogue_history, start=1):
        applied = _applied_beat_names(world, turn.applied_state_changes, empty="(none)")
        parts.append(
            f"\nTurn {i} prose: {turn.prose.strip()}\n"
            f"Turn {i} state_changes_applied: {applied}"
        )
        if turn.player_pick is not None:
            leave = (
                f" (leaves for {turn.player_pick.leaves_for})"
                if turn.player_pick.leaves_for
                else ""
            )
            parts.append(f'Turn {i} player picked: "{turn.player_pick.text}"{leave}')
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
    """Phase 1 user prompt — rich context for the classifier. Includes
    reachable beats, endings, full scene-so-far, case file. Stateless
    per turn (Phase 1 has no message_history)."""
    sections = [
        _render_you_are_at(state, world),
        _render_present(state, world, voice_only=False),
        _render_state(state, world),
        _render_reachable_beats(state, world),
        _render_all_endings(world),
        _render_case_file(state),
        _render_this_scene(state, world),
    ]
    suffix = "\n\n".join(sections)
    return suffix + """

[GENERATE STATE CHANGES FOR THIS TURN]
Emit zero, one, or at most two state changes (UnlockBeat) representing what
this turn's events will accomplish mechanically. Most turns emit zero. A
beat fires only when the moment described in its notes_to_llm has genuinely
landed in the conversation. The engine will reject any UnlockBeat whose
requires_beats aren't satisfied or whose blocked_by_beats already are; you
will be re-prompted with the error.
"""


# ───────────────────────── phase 2 user message (conversation turn) ─────────────────────────


def build_prose_writer_user_message(
    state: State,
    world: World,
    applied: StateChangeBundle,
    *,
    leave_pick: Option | None = None,
    opening_seed: str | None = None,
    is_first_turn: bool,
) -> str:
    """Per-turn user message for the conversation-based prose writer.

    Turn 1 of a scene (is_first_turn=True): [YOU ARE AT] + [PRESENT] +
    optional [OPENING SEED] + applied state changes + write-prose
    directive. Establishes the scene anchor.

    Turn 2+ (is_first_turn=False): just the new context — applied state
    changes, the player's pick (which surfaces in the conversation as
    their "spoken turn"), optional leave directive. The system prompt
    + accumulated assistant messages already carry the rest.

    Prose writer does NOT see: reachable beats, the beat catalog, the
    case file, the full state dump, structured turn-by-turn replay. The
    conversation is the scene-so-far; the system prompt is the world.
    """
    parts: list[str] = []
    if is_first_turn:
        parts.append(_render_you_are_at(state, world))
        parts.append(_render_present(state, world, voice_only=True))
        if opening_seed:
            parts.append(f"[OPENING SEED]\n{opening_seed.strip()}")

    # Player's last pick — surfacing in the user message frames it as
    # "what the player just did/said." Only meaningful on turn 2+
    # (turn 1 has no prior pick).
    if not is_first_turn and state.dialogue_history:
        last_turn = state.dialogue_history[-1]
        if last_turn.player_pick is not None:
            pick = last_turn.player_pick
            leave_suffix = (
                f" (leaving for {pick.leaves_for})" if pick.leaves_for else ""
            )
            parts.append(
                f'[PLAYER]\nThe player picked: "{pick.text}"{leave_suffix}'
            )

    parts.append(
        f"[STATE CHANGES JUST APPLIED]\n{_render_applied_changes(applied, world)}"
    )

    if leave_pick is not None:
        parts.append(
            f"[LEAVE DIRECTIVE]\nThe player chose to leave for "
            f"{leave_pick.leaves_for!r}. Write a brief goodbye line; the "
            f"engine will close the scene after your prose. Do NOT call "
            f"end_scene yourself — the engine has it covered."
        )

    parts.append(
        "[WRITE]\nWrite the next beat of prose (80-200 words), consistent "
        "with the state changes above. Call a tool ONLY if the narrative "
        "calls for it (pause_for_continue, request_free_text, end_scene, "
        "end_game). Otherwise just write prose and stop — the choice "
        "author will take over."
    )

    return "\n\n".join(parts)




def build_choice_author_user_message(
    state: State,
    world: World,
    prose: str,
) -> str:
    """Per-cycle user message for the choice author. Just shows the
    prose chunk(s) the narrator wrote this cycle, plus the location
    and present NPCs for adjacency / option-text grounding.

    Choice author is stateless per-cycle — no message_history. Each
    call sees the prose fresh."""
    loc = world.locations[state.player_location]
    present = _present_npc_ids(state, world)
    npcs_line = (
        ", ".join(f"{nid} ({world.npcs[nid].name})" for nid in present)
        if present else "no one — you are alone."
    )
    adjacent_line = (
        ", ".join(loc.adjacent_to) if loc.adjacent_to else "(none — stranded)"
    )
    return f"""\
[YOU ARE AT]
{loc.id} ({loc.name})

[PRESENT]
{npcs_line}

[ADJACENT LOCATIONS]
{adjacent_line}

[NARRATOR JUST WROTE]
{prose.strip()}

[GENERATE OPTIONS]
Emit a ChoiceBundle with 2-4 options. At least one option must have
`leaves_for` set to an adjacent location id. The grammar narrows
`leaves_for` to adjacent locations only.
"""


# ───────────────────────── chronicler prompts ─────────────────────────


def build_chronicler_system_prompt(world: World) -> str:
    """Stable system prompt for the chronicler. Baked once at bundle
    construction. Carries the full bible — the chronicler needs cast and
    truth context to write coherently about people and events, and the
    KV cache amortises the cost.

    Dual output per call: `case_file_entry` (UI prose, sidebar) AND
    `npc_memory_updates` (structured per-NPC notes surfaced to Phase 2
    on subsequent scenes). Same voice (the journalist), distinct
    registers: case file is "what happened"; npc_memory_updates is
    "what I now know / feel about this person".
    """
    return f"""\
[ROLE]
You are the player's own voice writing up their field notes after a
scene ends. You are not a narrator — you are the journalist herself,
sitting at the desk in the Old Vicarage that evening, recording what
just happened so it does not slip away.

[THE WORLD]
{_render_locations_block(world)}

[THE CAST]
{_render_cast_block(world, voice_only=False)}

[THE TRUTH]
{_render_truth_block_full(world)}

[YOUR JOB]
After every scene the player walks out of, you emit ONE structured
output with two fields:

  1. `case_file_entry` — 80-120 words of first-person past-tense prose,
     the player-facing case-file entry for that scene. What happened,
     who said what, what stayed with you. Each entry stands alone AND
     accumulates: the messages you have sent before this one ARE the
     case file so far; you can refer back to earlier moments where it
     would feel natural.

  2. `npc_memory_updates` — a LIST of `{{npc_id, note}}` records, one
     per NPC who was actually present in this scene. The schema
     enforces exactly this many entries — you cannot omit any present
     NPC, and the npc_id is constrained to NPCs who were in the room.
     Each `note` is 1-2 sentences, distinct in register from the case
     file: "what I now know / feel about this person" — the impression
     the journalist carries forward, not a recap of the scene. Used by
     the narrator on subsequent scenes to keep the relationship
     consistent.

[VOICE RULES]
1. First person, past tense, in BOTH fields. "I sat down at the bar
   next to Tom Bevan." Never "the player did X." Never omniscient.
2. Terse. Journalists' habit. Short sentences are stronger than long
   ones. Do not flower up. Do not editorialise unless it is your own
   reaction.
3. Stay inside what the player has seen and heard. Do not invent
   details the scene did not contain. You know the truth from the
   bible above, but you cannot WRITE what you have not been told.
4. Do not number or title the case file entry. Just write it as a
   paragraph.
5. Sound like the same person across entries. You are writing a
   notebook, not separate articles.

[NPC MEMORY OUTPUT — examples]
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

[OUTPUT]
Structured JSON matching the schema. `case_file_entry` is plain text
(no headers, no quotes); `npc_memory_updates` is a list of
`{{npc_id, note}}` records.
"""


def build_chronicler_user_prompt(state: State, world: World) -> str:
    """The per-scene-end user message: a structured summary of the
    just-completed scene that the chronicler turns into prose. The user
    message accumulates as the user side of a long conversation; each
    assistant reply is one case-file entry.

    Callers (emit_chronicle_entry) only fire this after a completed
    scene, so dialogue_history is always non-empty by the time we get
    here. The scene anchor (location, who was present) is read off the
    live state — the populator only places NPCs at scene entry, so the
    live state reflects the whole-scene composition.
    """
    assert state.dialogue_history, "build_chronicler_user_prompt: empty dialogue"
    loc = world.locations[state.player_location]
    present_ids = _present_npc_ids(state, world)
    if present_ids:
        npc_names = [world.npcs[n].name for n in present_ids]
        npcs_line = ", ".join(npc_names)
    else:
        npcs_line = "no one (you were alone there)"

    all_scene_changes = [
        c for turn in state.dialogue_history for c in turn.applied_state_changes
    ]
    beats_line = _applied_beat_names(world, all_scene_changes, empty="(none)")

    turn_parts: list[str] = []
    for i, turn in enumerate(state.dialogue_history, start=1):
        applied = _applied_beat_names(world, turn.applied_state_changes, empty="(no beat)")
        turn_parts.append(
            f"Turn {i} [beat fired: {applied}]:\n"
            f"  Narrator: {turn.prose.strip()}"
        )
        if turn.player_pick is not None:
            leave = (
                f" [→ left for {turn.player_pick.leaves_for}]"
                if turn.player_pick.leaves_for
                else ""
            )
            turn_parts.append(
                f'  You picked: "{turn.player_pick.text}"{leave}'
            )

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

    return f"""\
The scene that just ended took place at {loc.id} ({loc.name}).
Present: {npcs_line}.
Beats unlocked in this scene: {beats_line}.

Turn-by-turn:
{turns_block}

Emit BOTH fields of the structured output:

  1. `case_file_entry` — 80-120 words, first person, past tense, the
     player-facing record of what happened.

  2. `npc_memory_updates` — a list of `{{npc_id, note}}` records, one
     per NPC you saw this scene. Each note is "what I now know / feel
     about this person", distinct in register from the case file.
{npc_directive}

Stay inside what you saw and heard."""


# ───────────────────────── populator prompts ─────────────────────────


def build_populator_system_prompt(world: World) -> str:
    """System prompt for the populator agent. Runs once at scene entry
    to decide which NPCs are at which locations now. The populator sees
    the full bible (cast knowledge, beats, endings) so it can reason
    dramatically — e.g., Dylan retreats to the wood after a tense scene
    at Tan-y-coed; Helen agrees to the Costa for a discreet meeting.

    Byte-stable across the session for KV cache hits across scene
    entries.
    """
    return f"""\
[ROLE]
You are the world-state populator for a single-player investigation
game set in the Welsh Marches village of Pant Goch. The player is a
freelance journalist looking into a death the police closed too quickly.

Once per scene — at the moment the player enters a location — you
decide whether any NPCs should be somewhere other than their default
location for the duration of this scene. You do NOT write prose; you
emit a structured list of placement overrides (or an empty list).

[THE WORLD]
{_render_locations_block(world)}

[THE CAST]
{_render_cast_block(world, voice_only=False)}

[THE TRUTH]
{_render_truth_block_full(world)}

[YOUR JOB]
Each scene entry, emit ONE NpcPlacementBundle. Most scene entries
should be an empty bundle (defaults stand). Only emit a placement when
there is a SPECIFIC dramatic reason for an NPC to be somewhere other
than their default — and authored bible / case file evidence supports
it. Examples of legitimate movements:

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

[RULES]
1. Most scenes: emit an empty bundle. Defaults are right for the
   majority of scene entries. Restraint is the right instinct.
2. Move NPCs only with a SPECIFIC reason traceable to the case file or
   beats so far. "It would be interesting if X were here" is not
   enough. "Last scene's case file says Y, so X would plausibly be at
   Z" is enough.
3. Stay within the established cast and locations. The bible is complete.
4. Each NPC may appear at most once in the placements list. Do not
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

    case_file_block = _render_case_file(state)

    return f"""\
[PLAYER JUST ENTERED]
{loc.id} ({loc.name})
{loc.description.strip()}

[CURRENT NPC LOCATIONS]
{current_placements}

[STATE]
unlocked_beats: {_unlocked_beat_names(state, world, empty="(none)")}
scene_id: {state.scene_id}

{case_file_block}

[GENERATE PLACEMENTS]
Emit an NpcPlacementBundle. Most scene entries should be empty (defaults
stand). Only emit a placement when a specific dramatic reason traceable
to the case file or beats supports it. NPCs not listed in placements
keep their current location."""
