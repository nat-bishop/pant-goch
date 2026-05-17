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

    Phase 1 / log writer: include_knowledge=True (full bible).
    Phase 2 (system [THE CAST]): include_knowledge=False, include_voice_sample=True.
    Phase 2 (user [PRESENT]): same as above, plus history — the recent
        per-NPC memory entries from `state.npc_memory[npc.id]`, framed
        as the journalist's accumulated impressions of this person.

    `history_limit` clamps how many recent entries get rendered (default
    3) to keep attention bounded as the game progresses. Memory written
    by the log writer is one short note per scene the NPC was present in;
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
    """`voice_only=False`: Phase 1 / log writer bible (with knowledge).
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


def _render_beat(beat) -> str:
    requires = ", ".join(beat.requires_beats) if beat.requires_beats else "(none)"
    blocks = ", ".join(beat.blocked_by_beats) if beat.blocked_by_beats else "(none)"
    prereq = (
        f"\n  prerequisites (soft, your judgment): {beat.prerequisites.strip()}"
        if beat.prerequisites.strip()
        else ""
    )
    return (
        f"- {beat.id}:\n"
        f"  description: {beat.description.strip()}\n"
        f"  requires_beats: {requires}\n"
        f"  blocked_by_beats: {blocks}{prereq}\n"
        f"  notes_to_llm: {beat.notes_to_llm.strip()}"
    )


def _render_ending(ending) -> str:
    requires = ", ".join(ending.requires_beats) if ending.requires_beats else "(none)"
    return (
        f"- {ending.id}:\n"
        f"  requires_beats: {requires}\n"
        f"  notes_to_llm: {ending.notes_to_llm.strip()}\n"
        f"  epilogue: {ending.epilogue.strip()}"
    )


def _render_truth_block_full(world: World) -> str:
    """Phase 1 / log writer: crime + full beat catalog + endings."""
    parts: list[str] = [
        f"CRIME:\n{world.crime.strip()}\n",
        "BEATS (the moments that constitute uncovering the truth):",
    ]
    for beat in world.beats.values():
        parts.append(_render_beat(beat))
    parts.append("\nENDINGS (terminal states, gated by decision beats):")
    for ending in world.endings.values():
        parts.append(_render_ending(ending))
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


def build_phase2_system_prompt(world: World) -> str:
    """Phase 2 system prompt — the narrator. Slim, voice-focused. NO
    beats, NO notes_to_llm, NO endings, NO per-NPC knowledge. The
    narrator knows what happened (the crime overview), the cast's voice
    and attitude, and the locations. It does NOT know who-knows-what or
    which beats are gated on what.

    Byte-stable across the session; KV cache hits across every turn AND
    across scenes. Phase 2 also accumulates scene-bounded message
    history; this system prompt is the cache anchor for that history.
    """
    return f"""\
[ROLE]
You are the narrator of a single-player investigation game. The player
is a freelance journalist looking into a death in the Welsh Marches
village of Pant Goch. Each turn you write 80-200 words of prose for
what just happened, followed by 2-4 options the player can pick from.
Stay within the established cast and locations below.

[THE WORLD]
{_render_locations_block(world)}

[THE CAST]
{_render_cast_block(world, voice_only=True)}

[WHAT HAPPENED]
{world.crime.strip()}

You know this as the narrator. Your characters do not know all of it,
and most of them are protecting some part of it. Write them as people
in a village, not as keepers of plot beats.

[VOICE RULES]
1. 80-200 words of prose per turn. Close third, past tense, terse.
   Short sentences are stronger than long ones. Dialogue carries the
   scene where it can; interiority is a tool, not a default.

2. Render dialogue in the cast members' established voice. The
   voice_sample lines above show the register — match it, don't
   imitate verbatim. NPCs speak like people in a Welsh Marches
   village, not like characters announcing what they know.

3. Stay inside what the player has observed. Do not write what
   characters secretly think or know unless they are visibly choosing
   to reveal it. Atmosphere and small physical detail — a glass turned
   on the bar, a wood pigeon going off — beat exposition.

4. Stay within the established cast and locations. Minor incidental
   details (weather, a dog, a mug on the table) are fine. New named
   people or places are not.

[STYLE EXAMPLES]
The register and rhythm to match:

{_render_few_shots(world)}

[OPTIONS RULES]
After the prose, emit a `next` directive:
  - ContinueScene with 2-4 options. Each option is something the
    player can say or do that has no major consequence — dialogue,
    observation, small action. At least one option must be a "leave
    for…" with `leaves_for` set to an adjacent location id; the player
    must always have a walk-away choice.
  - EndScene when the conversation has reached a natural pause. Scenes
    of 4-8 turns are typical; longer is fine if the conversation
    warrants. Don't drag.
  - EndGame with an `ending_id` when the player has clearly committed
    to the action that triggers that ending and the engine state
    supports it. The engine will reject otherwise.

When the user prompt says the player chose to leave for a specific
adjacent location, you MUST emit EndScene. Write a brief goodbye line
in the prose; the `next` is fixed.

[YOUR JOB]
Each turn, emit ONE structured output: ProsePart — prose (80-200 words)
plus the `next` directive. State changes were already decided and
applied by a separate agent before you ran; your job is to narrate.
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


def _render_state(state: State) -> str:
    unlocked = (
        ", ".join(sorted(state.unlocked_beats))
        if state.unlocked_beats
        else "(none yet)"
    )
    if state.npc_scenes_co_present:
        co_pres = ", ".join(
            f"{npc}={n}" for npc, n in sorted(state.npc_scenes_co_present.items())
        )
    else:
        co_pres = "(no NPCs tracked)"
    return (
        f"[STATE]\n"
        f"unlocked_beats: {unlocked}\n"
        f"npc_scenes_co_present (player has been in the same room as): {co_pres}"
    )


def _render_reachable_beats(state: State, world: World) -> str:
    reachable = reachable_beat_ids(world, state)
    if not reachable:
        return "[REACHABLE BEATS]\n(none — every beat has fired or is permanently blocked.)"
    parts = ["[REACHABLE BEATS]"]
    for bid in sorted(reachable):
        beat = world.beats[bid]
        parts.append(_render_beat(beat))
    return "\n".join(parts)


def _render_all_endings(world: World) -> str:
    if not world.endings:
        return "[ALL ENDINGS]\n(none authored.)"
    parts = ["[ALL ENDINGS]"]
    for ending in world.endings.values():
        requires = (
            ", ".join(ending.requires_beats) if ending.requires_beats else "(none)"
        )
        parts.append(
            f"- {ending.id}:\n"
            f"  requires_beats: {requires}\n"
            f"  notes_to_llm: {ending.notes_to_llm.strip()}"
        )
    return "\n".join(parts)


def _render_case_file(case_file_entries: list[str] | None) -> str:
    if not case_file_entries:
        return "[CASE FILE]\n(empty — no scenes have ended yet.)"
    parts = ["[CASE FILE]"]
    for i, entry in enumerate(case_file_entries, start=1):
        parts.append(f"--- Entry {i} ---\n{entry.strip()}")
    return "\n".join(parts)


def _render_this_scene(state: State, world: World) -> str:
    """Phase 1's view of the current scene — structured turn-by-turn data
    (the classifier needs the applied state changes per turn to decide
    what's next)."""
    if not state.dialogue_history:
        return "[THIS SCENE SO FAR]\n(this is the first turn of the scene.)"
    first = state.dialogue_history[0]
    loc = world.locations[first.location_id]
    npcs = ", ".join(first.npcs_present) if first.npcs_present else "no one"
    parts = [
        f"[THIS SCENE SO FAR]",
        f"--- Scene {first.scene_id} at {loc.id} ({loc.name}) — present: {npcs} ---",
    ]
    for i, turn in enumerate(state.dialogue_history, start=1):
        applied = (
            ", ".join(c.beat_id for c in turn.applied_state_changes)
            if turn.applied_state_changes
            else "(none)"
        )
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


def _render_opening_seed(opening_seed: str | None) -> str:
    if not opening_seed:
        return ""
    return f"\n[OPENING SEED]\n{opening_seed.strip()}\n"


def _render_applied_changes(applied: StateChangeBundle, world: World) -> str:
    """Phase 2 sees the beat's description (not just the slug) when a beat
    fires. Phase 2's system prompt deliberately omits the beat catalog —
    surfacing the description here is the load-bearing channel that lets
    the narrator write the moment with specifics ("a relief he did not
    know he wanted") rather than inferring from the slug alone."""
    if not applied.state_changes:
        return "(none — this turn fired no beats.)"
    parts = []
    for change in applied.state_changes:
        if isinstance(change, UnlockBeat):
            beat = world.beats.get(change.beat_id)
            description = beat.description.strip() if beat else "(unknown beat)"
            parts.append(f"- UnlockBeat {change.beat_id}: {description}")
    return "\n".join(parts)


def build_phase1_user_prompt(
    state: State,
    world: World,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
) -> str:
    """Phase 1 user prompt — rich context for the classifier. Includes
    reachable beats, endings, full scene-so-far, case file. Stateless
    per turn (Phase 1 has no message_history)."""
    sections = [
        _render_you_are_at(state, world),
        _render_present(state, world, voice_only=False),
        _render_state(state),
        _render_reachable_beats(state, world),
        _render_all_endings(world),
        _render_case_file(case_file_entries),
        _render_this_scene(state, world),
    ]
    suffix = "\n\n".join(sections)
    seed = _render_opening_seed(opening_seed)
    return suffix + seed + """\

[GENERATE STATE CHANGES FOR THIS TURN]
Emit zero, one, or at most two state changes (UnlockBeat) representing what
this turn's events will accomplish mechanically. Most turns emit zero. A
beat fires only when the moment described in its notes_to_llm has genuinely
landed in the conversation. The engine will reject any UnlockBeat whose
requires_beats aren't satisfied or whose blocked_by_beats already are; you
will be re-prompted with the error.
"""


# ───────────────────────── phase 2 user message (conversation turn) ─────────────────────────


def build_phase2_user_message(
    state: State,
    world: World,
    applied: StateChangeBundle,
    *,
    leave_pick: Option | None = None,
    opening_seed: str | None = None,
    is_first_turn: bool,
) -> str:
    """Per-turn user message for the conversation-based Phase 2 narrator.

    Turn 1 of a scene (is_first_turn=True): [YOU ARE AT] + [PRESENT] +
    optional [OPENING SEED] + applied state changes + write-prose
    directive. Establishes the scene anchor.

    Turn 2+ (is_first_turn=False): just the new context — applied state
    changes, the player's pick (which surfaces in the conversation as
    their "spoken turn"), optional leave directive. The system prompt
    + accumulated assistant messages already carry the rest.

    Phase 2 does NOT see: reachable beats, endings, the case file, the
    full state dump, structured turn-by-turn replay. The conversation
    is the scene-so-far; the system prompt is the world.
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
            f"{leave_pick.leaves_for!r}. Your `next` MUST be EndScene. "
            f"Your prose narrates a brief goodbye line."
        )

    parts.append(
        "[WRITE]\nWrite the next beat of prose (80-200 words), consistent "
        "with the state changes above, then emit ContinueScene / EndScene / "
        "EndGame as `next`."
    )

    return "\n\n".join(parts)


# ───────────────────────── log writer prompts ─────────────────────────


def build_log_writer_system_prompt(world: World) -> str:
    """Stable system prompt for the log writer. Baked once at bundle
    construction. Carries the full bible — the log writer needs cast and
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

  2. `npc_memory_updates` — a dict keyed by NPC id, one short note
     (1-2 sentences) per NPC who was actually present in this scene.
     Distinct from the case file: this is "what I now know / feel about
     this person" — the impression the journalist carries forward, not
     a recap of the scene. Used by the narrator on subsequent scenes to
     keep the relationship consistent. Only emit entries for NPCs who
     were in the room; do not invent updates for NPCs you did not see
     this scene.

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
  npc_memory_updates: {{ "tom": "Old Tom Bevan. Drinks bitter alone in the
  afternoon. Guarded but not hostile; I bought him nothing, did not press.
  He didn't tell me to leave. There's something he's not saying about
  Megan, but it isn't time yet." }}

For a scene at the post office on a second visit where Maud closed up:
  npc_memory_updates: {{ "maud": "She's noticed I keep coming back. The
  warmth stayed but the tongue tightened. I asked too directly about
  the Vaughns and she went out the back to fetch something that didn't
  need fetching." }}

[OUTPUT]
Structured JSON matching the schema. `case_file_entry` is plain text
(no headers, no quotes); `npc_memory_updates` is a dict.
"""


def build_log_writer_user_prompt(state: State, world: World) -> str:
    """The per-scene-end user message: a structured summary of the
    just-completed scene that the log writer turns into prose. The user
    message accumulates as the user side of a long conversation; each
    assistant reply is one case-file entry.

    Callers (emit_log_entry) only fire this after a completed scene, so
    dialogue_history is always non-empty by the time we get here.
    """
    assert state.dialogue_history, "build_log_writer_user_prompt: empty dialogue"
    first = state.dialogue_history[0]
    loc = world.locations[first.location_id]
    if first.npcs_present:
        npc_names = [world.npcs[n].name for n in first.npcs_present]
        npcs_line = ", ".join(npc_names)
    else:
        npcs_line = "no one (you were alone there)"

    beats_unlocked: list[str] = []
    for turn in state.dialogue_history:
        for change in turn.applied_state_changes:
            beats_unlocked.append(change.beat_id)
    beats_line = ", ".join(beats_unlocked) if beats_unlocked else "(none)"

    turn_parts: list[str] = []
    for i, turn in enumerate(state.dialogue_history, start=1):
        applied = (
            ", ".join(c.beat_id for c in turn.applied_state_changes)
            if turn.applied_state_changes
            else "(no beat)"
        )
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

    # Explicit per-NPC directive — Gemma 4 was returning empty
    # npc_memory_updates dicts in early testing even when the system
    # prompt described the requirement; naming each NPC by id in the
    # user prompt makes the expected keys unambiguous.
    if first.npcs_present:
        npc_directive_lines = [
            f'    - key "{npc_id}" → 1-2 sentence note about {world.npcs[npc_id].name}'
            for npc_id in first.npcs_present
        ]
        npc_directive = (
            "  REQUIRED keys in `npc_memory_updates` (one entry per NPC "
            "you saw this scene):\n" + "\n".join(npc_directive_lines)
        )
    else:
        npc_directive = (
            "  `npc_memory_updates` MUST be empty {{}} — no NPCs were present."
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

  2. `npc_memory_updates` — a dict of per-NPC notes ("what I now know /
     feel about this person"), distinct in register from the case file.
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


def build_populator_user_prompt(
    state: State,
    world: World,
    *,
    case_file_entries: list[str] | None = None,
) -> str:
    """Per-call user prompt for the populator. Static-first / dynamic-last
    ordering: location + current placements first (small, stable for the
    scene), then the case file (grows over the session), then the job
    directive.
    """
    loc = world.locations[state.player_location]
    placements_lines: list[str] = []
    for npc in world.npcs.values():
        current = state.npc_locations.get(npc.id, npc.default_location)
        marker = " (default)" if current == npc.default_location else ""
        placements_lines.append(f"  - {npc.id}: {current}{marker}")
    current_placements = "\n".join(placements_lines)

    case_file_block = _render_case_file(case_file_entries)

    return f"""\
[PLAYER JUST ENTERED]
{loc.id} ({loc.name})
{loc.description.strip()}

[CURRENT NPC LOCATIONS]
{current_placements}

[STATE]
unlocked_beats: {", ".join(sorted(state.unlocked_beats)) if state.unlocked_beats else "(none)"}
scene_id: {state.scene_id}

{case_file_block}

[GENERATE PLACEMENTS]
Emit an NpcPlacementBundle. Most scene entries should be empty (defaults
stand). Only emit a placement when a specific dramatic reason traceable
to the case file or beats supports it. NPCs not listed in placements
keep their current location."""
