"""Scene-runner prompt builders.

KV-cache discipline (per plan):
  System prefix (baked at startup, byte-identical forever):
    [ROLE] + [THE WORLD] + [THE CAST] + [THE TRUTH] + [YOUR JOB] + [RULES]

  Dynamic suffix (rebuilt per call, in this order — most-stable → least-stable):
    [YOU ARE AT] / [PRESENT] / [STATE] / [REACHABLE BEATS] /
    [ALL ENDINGS] / [CASE FILE] / [THIS SCENE SO FAR] / [OPENING SEED]
    ↓ DIVERGENCE POINT — phase 1 / phase 2 differ from here
    Phase 1: [GENERATE STATE CHANGES FOR THIS TURN]
    Phase 2: [STATE CHANGES JUST APPLIED] + [GENERATE PROSE AND NEXT]

The system prefix is built ONCE in `build_scene_runner_system_prompt(world)`
at lifespan boot. Per-call, `build_prompt_prefix(state, ...)` renders the
shared dynamic suffix and `build_phase1_tail()` / `build_phase2_tail(...)`
render the divergence-point tails. Phase 2's prompt = phase 1 prompt's
prefix + phase 2 tail, so KV cache hits cover everything before the
divergence.
"""
from __future__ import annotations

from .engine import reachable_beat_ids
from .schemas import StateChangeBundle, Option, UnlockBeat
from .state import State
from .world import World


# ───────────────────────── system prompt (stable prefix) ─────────────────────────


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


def _render_npc_card(npc, *, include_default_location: bool) -> str:
    themes = "; ".join(npc.themes) if npc.themes else "(none)"
    head = (
        f"- {npc.id} ({npc.name}, default at {npc.default_location}):"
        if include_default_location
        else f"- {npc.id} ({npc.name}):"
    )
    return (
        f"{head}\n"
        f"  themes: {themes}\n"
        f"  knowledge: {npc.knowledge.strip()}\n"
        f"  attitude: {npc.attitude.strip()}"
    )


def _render_cast_block(world: World) -> str:
    return "\n".join(
        _render_npc_card(npc, include_default_location=True)
        for npc in world.npcs.values()
    )


def _render_truth_block(world: World) -> str:
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


def build_scene_runner_system_prompt(world: World) -> str:
    """Stable byte-identical system prompt baked at lifespan boot. Carries
    the entire world bible. Never changes within a session — KV cache
    relies on this.
    """
    return f"""\
[ROLE]
You are the scene runner for a single-player investigation game. The
player is a freelance journalist investigating a closed-but-suspicious
death. You run scenes turn by turn through structured output. The world
is hand-authored; you do not invent locations, characters, or facts
beyond what the bible below contains.

[THE WORLD]
{_render_locations_block(world)}

[THE CAST]
{_render_cast_block(world)}

[THE TRUTH]
{_render_truth_block(world)}

[YOUR JOB]
Each turn against this same system prompt, two structured emissions:

  Phase 1 — StateChangeBundle: zero, one, or two state changes
    (UnlockBeat). The engine validates and applies them. Most turns emit
    zero. A beat fires only when the moment described in its
    notes_to_llm has genuinely landed in the conversation.

  Phase 2 — ProsePart: prose (80-200 words) consistent with the
    just-applied changes, plus a `next` directive:
      - ContinueScene with 2-4 options (≥1 must have leaves_for set —
        the player must always have a walk-away choice).
      - EndScene (use when the conversation has reached a natural pause).
      - EndGame with an ending_id (only when that ending's requires_beats
        are satisfied — the engine will reject otherwise).

The user prompt tells you which phase you are in.

[RULES]
1. Never invent characters not in the cast above. Never invent locations
   not in the world. Minor details (weather, a dog, a mug on the bar)
   are fine — full new people or places break the world's coherence.

2. ContinueScene options: 2-4 per turn. Options are what the player
   can say or do that has no major consequence — dialogue, observation,
   minor action. At least one option must have `leaves_for` set to an
   adjacent location id (player exits via leave-options; the map is
   inert during scenes). Do not promise a consequence requiring a beat
   the world doesn't have.

3. Unlock beats when their notes_to_llm describes the moment. Re-read
   the prereqs. Don't unlock prematurely. The engine will reject (and
   re-prompt) any UnlockBeat whose requires_beats aren't satisfied or
   whose blocked_by_beats already are.

4. End the scene when the conversation has reached a natural stop or
   when a beat has fired and the scene's purpose is done. 4-8 turns is
   typical; longer is fine if the conversation warrants. Don't drag.

5. End the game only when the player has clearly chosen the action
   that triggers an ending — not just when prerequisites are met. Each
   ending depends on a "decision beat" that fires only when the player
   explicitly commits. End the game only after that decision beat is
   in state. The engine rejects EndGame otherwise.

6. When the user prompt says the player chose to leave for a specific
   adjacent location, you MUST emit EndScene as your phase-2 next.
   Phase 1 may still fire a beat (the act of leaving CAN be the
   moment), but phase 2's next is fixed.

7. State changes flow through UnlockBeat only. Free options are pure
   narrative — they have no mechanical effect. The engine writes state;
   you write prose.
"""


# ───────────────────────── dynamic suffix (shared phase 1 / phase 2) ─────────────────────────


def _render_you_are_at(state: State, world: World) -> str:
    loc = world.locations[state.player_location]
    notes = "\n  ".join(f"- {n}" for n in loc.notes) if loc.notes else "(none)"
    return (
        f"[YOU ARE AT]\n"
        f"{loc.id} ({loc.name})\n"
        f"description: {loc.description.strip()}\n"
        f"notes:\n  {notes}"
    )


def _render_present(state: State, world: World) -> str:
    npcs_here = world.npcs_at(state.player_location)
    if not npcs_here:
        return "[PRESENT]\n(no one — you are alone here.)"
    parts = ["[PRESENT]"]
    for npc in npcs_here:
        parts.append(_render_npc_card(npc, include_default_location=False))
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
    """Renders dialogue_history for the current scene. With per-scene
    clearing, all turns share the same scene_id — one scene header at the
    top, then bare prose+pick per turn."""
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


def build_prompt_prefix(
    state: State,
    world: World,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
) -> str:
    """The byte-identical-up-to-divergence prefix shared by phase 1 and
    phase 2. Order matters for KV caching (most-stable → least-stable
    sections; opening_seed is last because it appears only on turn 1)."""
    sections = [
        _render_you_are_at(state, world),
        _render_present(state, world),
        _render_state(state),
        _render_reachable_beats(state, world),
        _render_all_endings(world),
        _render_case_file(case_file_entries),
        _render_this_scene(state, world),
    ]
    suffix = "\n\n".join(sections)
    seed = _render_opening_seed(opening_seed)
    return suffix + seed


# ───────────────────────── divergence-point tails ─────────────────────────


def build_phase1_tail() -> str:
    return """\

[GENERATE STATE CHANGES FOR THIS TURN]
Emit zero, one, or at most two state changes (UnlockBeat) representing what
this turn's events will accomplish mechanically. Most turns emit zero. A
beat fires only when the moment described in its notes_to_llm has genuinely
landed in the conversation. The engine will reject any UnlockBeat whose
requires_beats aren't satisfied or whose blocked_by_beats already are; you
will be re-prompted with the error.
"""


def _render_applied_changes(applied: StateChangeBundle) -> str:
    if not applied.state_changes:
        return "(none — this turn fired no beats.)"
    parts = []
    for change in applied.state_changes:
        if isinstance(change, UnlockBeat):
            parts.append(f"- UnlockBeat {change.beat_id}")
    return "\n".join(parts)


def build_phase2_tail(
    applied: StateChangeBundle,
    *,
    leave_pick: Option | None = None,
) -> str:
    leave_directive = ""
    if leave_pick is not None:
        leave_directive = (
            f"\nThe player chose to leave for {leave_pick.leaves_for!r}. "
            "Phase 2's `next` MUST be EndScene. Phase 1 may have fired a "
            "beat-on-exit; your prose narrates the goodbye line.\n"
        )
    return f"""\

[STATE CHANGES JUST APPLIED]
{_render_applied_changes(applied)}
{leave_directive}
[GENERATE PROSE AND NEXT]
Write the prose for this turn (80-200 words), narrating the events
consistent with any state changes above. Then emit one of:
  - ContinueScene with 2-4 options (≥1 with leaves_for set).
  - EndScene (when the conversation has reached a natural pause).
  - EndGame with an ending_id (only when that ending's requires_beats
    are satisfied).
"""


# ───────────────────────── full per-call prompts ─────────────────────────


def build_phase1_user_prompt(
    state: State,
    world: World,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
) -> str:
    return build_prompt_prefix(
        state, world,
        case_file_entries=case_file_entries,
        opening_seed=opening_seed,
    ) + build_phase1_tail()


def build_phase2_user_prompt(
    state: State,
    world: World,
    applied: StateChangeBundle,
    *,
    case_file_entries: list[str] | None = None,
    opening_seed: str | None = None,
    leave_pick: Option | None = None,
) -> str:
    return build_prompt_prefix(
        state, world,
        case_file_entries=case_file_entries,
        opening_seed=opening_seed,
    ) + build_phase2_tail(applied, leave_pick=leave_pick)


# ───────────────────────── log writer prompts ─────────────────────────


def build_log_writer_system_prompt(world: World) -> str:
    """Stable byte-identical system prompt for the log writer. Baked once at
    bundle construction. Carries the full bible — the log writer needs cast
    and truth context to write coherently about people and events, and the
    KV cache amortises the cost.

    Voice: the case file is the PLAYER'S OWN field notes. First-person,
    past tense, terse, journalist's habit — what happened, what was said,
    what felt off. Not omniscient narration; not "the player did X."
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
{_render_cast_block(world)}

[THE TRUTH]
{_render_truth_block(world)}

[YOUR JOB]
After every scene the player walks out of, you write ONE case-file
entry: 80-120 words, first person, past tense. Capture what happened,
who said what, what stayed with you. Notice what was off as much as
what was said. Each entry stands alone as a piece of writing AND as
part of an accumulating notebook — the messages you have sent before
this one ARE the case file so far, and you can refer back to earlier
moments where it would feel natural.

[VOICE RULES]
1. First person, past tense. "I sat down at the bar next to Tom Bevan."
   Never "the player did X." Never omniscient.
2. Terse. Journalists' habit. Short sentences are stronger than long
   ones. Do not flower up. Do not editorialise unless it is your own
   reaction.
3. Stay inside what the player has seen and heard. Do not invent
   details the scene did not contain. You know the truth from the
   bible above, but you cannot WRITE what you have not been told.
4. Do not number or title the entry. Just write it as a paragraph.
5. Sound like the same person across entries. You are writing a
   notebook, not separate articles.

[OUTPUT]
The case-file entry as plain text. No headers, no quotes around it,
no metadata. Just the paragraph.
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

    return f"""\
The scene that just ended took place at {loc.id} ({loc.name}).
Present: {npcs_line}.
Beats unlocked in this scene: {beats_line}.

Turn-by-turn:
{turns_block}

Write the case-file entry now. 80-120 words. First person, past tense.
Stay inside what you saw and heard."""
