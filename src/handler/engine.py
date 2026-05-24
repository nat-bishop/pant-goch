"""Engine helpers — the quality vector, state application, scene lifecycle.

Pure functions (no LLM, no SSE). The quality engine lives here:
  - `confession_depth` / `band_name` / `when_satisfied` — derive the world's
    milestones and ending availability from per-NPC trust/guard.
  - `apply_state_changes` — apply Phase-1 deltas + commitments, then auto-derive
    milestones (`_derive_milestones`).
  - `emittable_beat_ids` — the LLM-emittable commitment beats, narrowed per call
    in `agents._build_state_change_bundle_type` and rendered in `prompts.py`.
  - `resolved_ending_id_at_close` — the ending the engine fires on scene close.
"""
from __future__ import annotations

import random

from .models import When
from .schemas import (
    FreeText,
    QualityDelta,
    StateChangeBundle,
    TRUST_DELTA_VALUES,
    UnlockBeat,
)
from .state import State, TurnRecord
from .world import World


# Fallback free-text box: a fixed, neutral caption with no suggestion,
# surfaced when the gate+cadence said "offer" but the caption author
# failed (retries exhausted). The happy path builds a per-turn FreeText
# from the caption author's output. See `scene_loop._assemble_prose_part`.
FREE_TEXT_FALLBACK = FreeText(prompt="What do you do next?", npc_id=None, suggested=None)


# ───────────────────────── free-text cadence ─────────────────────────
#
# The LLM gate is a stateless per-turn classifier — it sees only the prose
# and so cannot perceive (or regulate) a global offer *rate*. Frequency is
# a property of the distribution over many calls, not of any single call,
# so we own it here: the gate says "this moment *could* take free text",
# and this engine layer decides whether to actually surface it now.

FREE_TEXT_COOLDOWN = 2     # min completed turns since the last typed move
FREE_TEXT_OFFER_P = 0.5    # honor an eligible offer only this often


def _turns_since_typed(state: State) -> int:
    """Completed turns since the player last typed their own move this
    scene. Counts back through `state.dialogue_history` (cleared per
    scene) for the most recent `player_pick_typed`; returns a large
    sentinel when there's been none. The turn currently being assembled
    is not yet in `dialogue_history` (`append_turn_record` runs after
    `_assemble_prose_part`), so this counts strictly prior turns."""
    for i, rec in enumerate(reversed(state.dialogue_history)):
        if rec.player_pick_typed:
            return i
    return len(state.dialogue_history) + 1  # effectively "no recent typed move"


def free_text_cadence_ok(state: State, rng: random.Random) -> bool:
    """Engine-side cadence gate, applied on top of the LLM gate's offer.
    Suppress free text within `FREE_TEXT_COOLDOWN` turns of the last typed
    move, then honor the offer only with probability `FREE_TEXT_OFFER_P`.
    Pure and seedable (`rng`) so tests are deterministic."""
    if _turns_since_typed(state) < FREE_TEXT_COOLDOWN:
        return False
    return rng.random() < FREE_TEXT_OFFER_P


def apply_state_changes(
    bundle: StateChangeBundle, state: State, world: World
) -> list[str]:
    """Apply a Phase-1 bundle to state and return the beat ids newly unlocked
    THIS turn (LLM-emitted commitments + engine-derived milestones), in that
    order — the caller emits one `unlock_beat` SSE per id.

    Two effects, in order:
      1. quality deltas — clamped trust/guard nudges for present NPCs (positive
         trust pinned to `_trust_adjustable_here`); a `BREACH` latches guard.
      2. commitments — the LLM-emitted observation/decision beats.
    Then `_derive_milestones` auto-unlocks any evidence/outcome beat whose
    `when` gate the deltas just satisfied. Validation happens upstream via
    schema validators + per-call Literal narrowing in
    `agents._build_state_change_bundle_type`. Global `village_attention` is NOT
    touched here — it is owned by the dedicated scene-close attention agent
    (see `apply_village_attention`)."""
    for d in bundle.deltas:
        if d.npc_id in state.guard_breached:
            continue  # relationship is over; ignore further nudges
        npc = world.npcs.get(d.npc_id)
        if npc is not None and npc.background:
            continue  # deliberate background figure (Mrs Hughes) — never warms
        t = TRUST_DELTA_VALUES[d.trust]
        if t > 0 and not _trust_adjustable_here(world, state, d.npc_id):
            t = 0  # won't warm here (e.g. Dylan at Tan-y-coed)
        state.trust[d.npc_id] = _clamp(
            state.trust.get(d.npc_id, 0) + t, 0, TRUST_MAX
        )
        if d.guard == "BREACH":
            state.guard_breached.add(d.npc_id)
        else:
            state.guard[d.npc_id] = _clamp(
                state.guard.get(d.npc_id, 0) + TRUST_DELTA_VALUES[d.guard],
                0,
                GUARD_MAX,
            )

    newly: list[str] = []
    for c in bundle.commitments:
        if c.beat_id not in state.unlocked_beats:
            state.unlocked_beats.add(c.beat_id)
            newly.append(c.beat_id)
    newly.extend(_derive_milestones(world, state))
    return newly


# ───────────────────────── quality vector ─────────────────────────
#
# The thin spine that replaces the boolean ladder beats: per-NPC `trust`
# (warmth) and `guard` (alarm) accumulate ±1/turn; the LLM nudges them, the
# engine derives everything else. A confession DEPTH band is a pure function
# of those two numbers, so an evidence milestone can never mis-fire — it falls
# out of the trust the player has actually earned.

# Raw clamps. trust/guard are bounded so a runaway LLM emission saturates
# harmlessly instead of running off the scale (the boolean DAG had no such
# floor — a mis-fired beat was permanent). TRUST_MAX must clear 3×the widest
# band so the terminal `at_peace` band is reachable (Tom → police).
TRUST_MAX = 8
# Guard must be able to OFFSET trust, not saturate below it: effective warmth is
# `trust - guard`, so with GUARD_MAX < TRUST_MAX a player who keeps pressing
# can never hold an NPC cold once trust accrues (the at_peace monopoly — every
# persona warmed Tom regardless of approach). Raised to match TRUST_MAX so
# sustained pressure genuinely suppresses the relationship and aggressive play
# forks away from the warm endings. See STATUS.md (divergence-tuning).
GUARD_MAX = 8

# Confession bands, by depth index. Derived from effective warmth
# (trust − guard) divided by a per-NPC band WIDTH; clamped to the top band.
# `cold` is the stranger default; the evidence milestone trips at `open`;
# `at_peace` is the terminal "decides on their own" band (Tom → police).
BANDS = ("cold", "thawing", "open", "at_peace")
EVIDENCE_BAND = 2  # BANDS.index("open") — depth at which evidence milestones trip

# Default raw-points-per-band when an NPC has no authored `disposition`. The
# per-NPC `warms` rate (slow/steady/fast) overrides this in `confession_depth`.
# Tuned (with TRUST_MAX) so a good single scene of ~3-4 warm turns reaches
# `open` (evidence) for a steady NPC, and a slow NPC earns it over a little
# longer — the band-width knob replaces the old min_scenes_with_npc gate.
DEFAULT_BAND_WIDTH = 1
_BAND_WIDTH_BY_WARMS = {"slow": 2, "steady": 1, "fast": 1}

# Village attention bands (the global heat). Rises ONLY on conspicuous /
# aggressive acts, never on gentle repetition; ratchets up. Ejection (the
# run-out terminal) is `hostile` and is gated on attention ALONE — NOT a
# breach-count, which is the entropy-0 collapse the boolean system was tuned
# to avoid (see truth.yaml scar comments).
ATTENTION_BANDS = {"noticed": 1, "talked_about": 2, "hostile": 3}
ATTENTION_EJECT = ATTENTION_BANDS["hostile"]

# The dedicated scene-close attention agent (agents.emit_village_attention) emits
# one of these four levels; the engine ratchets village_attention UP to its int.
# `none` (0) is the agent's "nothing the village would close ranks over" verdict —
# it has no entry in ATTENTION_BANDS (which only names the >0 gate bands), so it
# lives in this separate level map.
ATTENTION_LEVEL_VALUES = {"none": 0, "noticed": 1, "talked_about": 2, "hostile": 3}


def apply_village_attention(state: State, level: str) -> int:
    """Ratchet `village_attention` up to the agent's verdict level — never down
    (the village does not un-notice you). Returns the resulting value."""
    state.village_attention = max(
        state.village_attention, ATTENTION_LEVEL_VALUES[level]
    )
    return state.village_attention


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _band_width(world: World, npc_id: str) -> int:
    """Raw trust points per confession band for `npc_id`, from the NPC's
    authored `warms` rate (slow warms slowest → widest bands). Falls back to
    `DEFAULT_BAND_WIDTH` for NPCs with no `disposition` block."""
    npc = world.npcs.get(npc_id)
    disp = getattr(npc, "disposition", None) if npc else None
    if disp is None:
        return DEFAULT_BAND_WIDTH
    return _BAND_WIDTH_BY_WARMS.get(disp.warms, DEFAULT_BAND_WIDTH)


def confession_depth(world: World, state: State, npc_id: str) -> int:
    """How far open `npc_id` is RIGHT NOW, as a band depth in [0, 3] — or -1
    when the relationship has latched shut (`guard_breached`). Pure function of
    `trust`/`guard`: effective warmth `trust − guard`, floored at 0, divided by
    the NPC's band width, clamped to the top band. This is the single arithmetic
    expression that the whole Tom/Dylan/Helen ladder collapsed into."""
    if npc_id in state.guard_breached:
        return -1
    effective = max(0, state.trust.get(npc_id, 0) - state.guard.get(npc_id, 0))
    depth = effective // _band_width(world, npc_id)
    return min(depth, len(BANDS) - 1)


def band_name(world: World, state: State, npc_id: str) -> str:
    """The current band's label for prose/prompt rendering. `closed` when the
    relationship has breached (the guard-high register), else the depth band."""
    depth = confession_depth(world, state, npc_id)
    if depth < 0:
        return "closed"
    return BANDS[depth]


def village_ejected(state: State) -> bool:
    """The village has collectively closed ranks — the run-out terminal.
    Gated on `village_attention` ALONE (see ATTENTION_EJECT)."""
    return state.village_attention >= ATTENTION_EJECT


def when_satisfied(world: World, state: State, when: When) -> bool:
    """Evaluate a `When` quality/milestone gate against current state. All set
    conditions must hold (AND). The single replacement for the boolean
    requires_* family — used to derive evidence milestones, gate decision-beat
    availability, and gate ending quality conditions."""
    if when.npc is not None and when.trust_at_least is not None:
        need = BANDS.index(when.trust_at_least)
        if confession_depth(world, state, when.npc) < need:
            return False
    if when.any_evidence_of and not any(
        b in state.unlocked_beats for b in when.any_evidence_of
    ):
        return False
    if when.requires_beats and not set(when.requires_beats).issubset(
        state.unlocked_beats
    ):
        return False
    if when.npcs_thawed_at_least is not None:
        thawed = sum(
            1 for nid in world.npcs if confession_depth(world, state, nid) >= 1
        )
        if thawed < when.npcs_thawed_at_least:
            return False
    if when.village_attention_at_least is not None:
        if state.village_attention < ATTENTION_BANDS[when.village_attention_at_least]:
            return False
    if when.village_attention_below is not None:
        if state.village_attention >= ATTENTION_BANDS[when.village_attention_below]:
            return False
    return True


def _is_decision(beat_id: str) -> bool:
    """The mutually-exclusive player-commitment beats (engine enforces at most
    one per playthrough). Keyed on the `decided_` id convention so the rule is
    one line, not a pairwise blocked_by graph."""
    return beat_id.startswith("decided_")


def _trust_adjustable_here(world: World, state: State, npc_id: str) -> bool:
    """Can positive trust accrue for `npc_id` at the player's CURRENT location?
    False when the NPC has breached, or when their `opens_up_only_at` excludes
    here (Dylan won't warm at Tan-y-coed). Guard/BREACH still apply anywhere —
    this only pins WARMING, the mechanical home of the old requires_location."""
    if npc_id in state.guard_breached:
        return False
    npc = world.npcs.get(npc_id)
    disp = getattr(npc, "disposition", None) if npc else None
    if disp and disp.opens_up_only_at and state.player_location not in disp.opens_up_only_at:
        return False
    return True


def _derive_milestones(world: World, state: State) -> list[str]:
    """Auto-unlock every engine-derived (non-commitment) beat whose `when` gate
    is now met. Returns the newly-unlocked ids (for the SSE unlock_beat emit).
    Idempotent: already-unlocked beats are skipped, so it can run every turn."""
    newly: list[str] = []
    for bid, beat in world.beats.items():
        if beat.commitment or beat.when is None or bid in state.unlocked_beats:
            continue
        if when_satisfied(world, state, beat.when):
            state.unlocked_beats.add(bid)
            newly.append(bid)
    return newly


# ───────────────────────── emittable beats + ending resolution ─────────────────────────


def present_npc_ids(world: World, state: State) -> tuple[str, ...]:
    """NPC ids at the player's current location. Stable across a scene
    (the populator runs only at scene entry), so callers can compute
    on demand without caching."""
    return tuple(n.id for n in world.npcs_at(state.player_location, state=state))


def emittable_beat_ids(world: World, state: State) -> set[str]:
    """The LLM-emitted beats (`commitment: true`) the classifier may fire RIGHT
    NOW — the source of both the Phase-1 commitment `Literal`
    (`agents._build_state_change_bundle_type`) and the `## Choices Available`
    prompt block. Engine-derived milestones are NOT here: they fall out of
    `_derive_milestones` from trust and can never be mis-fired.

    A commitment beat is emittable iff ALL of:
      - it is not already unlocked;
      - it is not a `decided_` beat while another decision is already taken
        (at-most-one-decision, the replacement for pairwise blocked_by);
      - its `requires_location` (if any) contains the player's location;
      - if it is anchored to an NPC (`when.npc` set), that NPC is PRESENT — a
        person-anchored findable (Tom naming the vehicle, Dylan admitting what
        he knows) is pursued face to face, so it is live only in a scene the
        NPC is actually in. `when_satisfied` checks the trust band globally;
        presence is the other half of "anchored to a person" and is enforced
        here so the Phase-1 commitment `Literal` and the choice/prose steers
        narrow to it too;
      - its `when` availability gate (if any) is satisfied.
    """
    here = state.player_location
    present = present_npc_ids(world, state)
    decision_taken = any(_is_decision(b) for b in state.unlocked_beats)
    out: set[str] = set()
    for bid, beat in world.beats.items():
        if not beat.commitment or bid in state.unlocked_beats:
            continue
        if _is_decision(bid) and decision_taken:
            continue
        if beat.requires_location and here not in beat.requires_location:
            continue
        if beat.when is not None and beat.when.npc and beat.when.npc not in present:
            continue
        if beat.when is not None and not when_satisfied(world, state, beat.when):
            continue
        out.add(bid)
    return out


# How many completed turns a scene gets to "breathe" before the mere
# AVAILABILITY of something to pursue (a findable still live here, a present NPC
# who could still open up) stops keeping it alive. Past this window only ACTUAL
# movement holds a scene open: a room with an un-pursued findable the player
# keeps declining closes on the first dry turn instead of milling to
# MAX_TURNS_PER_SCENE. The findable simply goes unfound — you left before doing
# it — which is the intended "you can't find everything in one visit."
SCENE_FINDABLE_GRACE_TURNS = 5


def scene_is_spent(world: World, state: State) -> bool:
    """True when the current scene has nothing productive left AND the last
    completed turn moved nothing — the engine's content-derived soft-close
    signal (scene_loop converts a spent Continue/FreeText into an EndScene).

    Two regimes, split at `SCENE_FINDABLE_GRACE_TURNS`:

    - Within the opening grace window, a scene is kept alive by mere
      possibility — a findable still live here (an observation milestone
      anchored to this location, or a present-NPC findable now in reach), or a
      present tracked NPC who can still open up (warmable here, not breached,
      not yet at the top band). The room gets a few turns to breathe before it
      can close.
    - Past the window, only MOVEMENT keeps it open: the scene is spent the
      moment a completed turn is DRY (unlocked no beat, shifted no disposition).
      A productive conversation never trips this — its deltas keep the last turn
      non-dry — but a stalled room with a findable the player keeps not pursuing
      now closes instead of looping forever (the post-office funeral-notice
      tail, the vicarage map dead-end).

    Returns False before any turn completes. Distinct from the flat
    `MAX_TURNS_PER_SCENE` hang-backstop in scene_loop: this ends a dead room the
    moment it is dead, instead of milling to the cap.
    """
    if not state.dialogue_history:
        return False
    here = state.player_location
    # completed turns this scene == len(dialogue_history) at the call site.
    if len(state.dialogue_history) <= SCENE_FINDABLE_GRACE_TURNS:
        for bid in emittable_beat_ids(world, state):
            beat = world.beats[bid]
            if not beat.requires_location or here in beat.requires_location:
                return False
        for nid in present_npc_ids(world, state):
            npc = world.npcs[nid]
            if npc.disposition is None or nid in state.guard_breached:
                continue
            if not _trust_adjustable_here(world, state, nid):
                continue
            if confession_depth(world, state, nid) < len(BANDS) - 1:
                return False
    last = state.dialogue_history[-1]
    return not last.applied_state_changes and not last.applied_deltas


def resolved_ending_id_at_close(world: World, state: State) -> str | None:
    """Pick the ending that fires when a scene closes: the ending whose
    `requires_beats` are all unlocked AND whose `when` quality gate (if any) is
    satisfied. None if none qualify.

    Ejection takes precedence: a player who provoked the village out is run off
    even if a decision ending also happens to be satisfied. That precedence is
    enforced HERE (any satisfied ending gated on `village_attention_at_least:
    hostile` wins) rather than relying on `run_out_of_the_village` being
    authored first — a YAML reorder or round-trip must not silently flip which
    ending fires. Among non-ejection endings, decisions are mutually exclusive
    (engine rule) so at most one is satisfiable; author order is the defensive
    tie-break.
    """
    satisfied = [
        eid
        for eid, ending in world.endings.items()
        if set(ending.requires_beats).issubset(state.unlocked_beats)
        and (ending.when is None or when_satisfied(world, state, ending.when))
    ]
    if not satisfied:
        return None
    for eid in satisfied:
        when = world.endings[eid].when
        if when is not None and when.village_attention_at_least == "hostile":
            return eid
    return satisfied[0]


# ───────────────────────── scene lifecycle ─────────────────────────


def enter_scene(state: State, world: World, location_id: str) -> None:
    """Begin a new scene at `location_id`. Increments scene_id, clears
    dialogue_history, sets player_location.

    SSE / log-writer / populator triggering happens at the scene_loop
    layer; this function only mutates state.
    """
    state.scene_id += 1
    state.dialogue_history = []
    state.player_location = location_id


def append_turn_record(
    state: State,
    *,
    prose: str,
    applied_state_changes: list[UnlockBeat],
    applied_deltas: list | None = None,
) -> None:
    """Pack a completed turn's data into a TurnRecord and append it to
    dialogue_history. The scene anchor (scene_id, location, npcs_present)
    is constant across `dialogue_history` and lives on `state`, so
    renderers derive it once per scene."""
    state.dialogue_history.append(
        TurnRecord(
            prose=prose,
            applied_state_changes=list(applied_state_changes),
            applied_deltas=list(applied_deltas or []),
        )
    )
