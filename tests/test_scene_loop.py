"""Scene-loop orchestration tests.

Mocks emit_state_changes / stream_prose_text / emit_choices / emit_free_text_gate
/ emit_caption (and the engine cadence gate) to avoid LLM dependency. Verifies
that run_turn sequences phase 1 → apply → prose writer → choice author ‖ gate →
(cadence + caption) → assemble ProsePart → append, that fallbacks fire on
UnexpectedModelBehavior, and that play_scene drives Continue picks and FreeText
typed input until termination.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from handler.agents import (
    build_scene_runner_bundle,
    make_scene_runner_runtime,
)
from handler.engine import FREE_TEXT_FALLBACK, enter_scene
from handler.schemas import (
    ChoiceBundle,
    ChronicleEntry,
    Continue,
    EndGame,
    EndScene,
    FreeText,
    FreeTextCaption,
    Leave,
    NpcPlacementBundle,
    Option,
    ProsePart,
    QualityDelta,
    StateChangeBundle,
    UnlockBeat,
)
from handler.scene_loop import (
    MAX_TURNS_PER_SCENE,
    PickOption,
    TypeText,
    WalkTo,
    _assemble_prose_part,
    play_scene,
    play_session,
    run_turn,
)
from handler.state import State, make_initial_state
from handler.world import World, load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── populator / chronicler no-ops ─────────────────────────


async def _empty_placements(*args, **kwargs):
    """Default populator stub for tests that don't care about NPC
    placement. Returns an empty bundle (defaults stand)."""
    return NpcPlacementBundle()


def _patch_populator(monkeypatch):
    monkeypatch.setattr("handler.scene_loop.emit_placements", _empty_placements)


def _patch_chronicler(monkeypatch, text: str = "entry"):
    async def fake_emit(*args, **kwargs):
        return ChronicleEntry(
            case_file_entry=text,
            npc_memory_updates=[],
        )
    monkeypatch.setattr("handler.scene_loop.emit_chronicle_entry", fake_emit)


def _patch_village_attention(monkeypatch, level: str = "none"):
    """Stub the scene-close village-attention agent (it runs on every Leave /
    WalkTo close, before ending resolution). Default `none` is a no-op ratchet."""
    async def fake_emit(*args, **kwargs):
        return level
    monkeypatch.setattr("handler.scene_loop.emit_village_attention", fake_emit)


def _patch_phase1(monkeypatch, bundle: StateChangeBundle | list | None = None):
    """Default Phase 1 stub: returns the supplied bundle (or empty). A list of
    bundles is CYCLED, one per call — handy for a delta that must realize
    movement every turn without saturating a clamp (oscillate guard ±1)."""
    if isinstance(bundle, list):
        seq = itertools.cycle(bundle)

        async def fake(*args, **kwargs):
            return next(seq)
    else:
        payload = bundle or StateChangeBundle()

        async def fake(*args, **kwargs):
            return payload

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake)


def _patch_prose_writer(monkeypatch, proses):
    """`proses` is an iterable of strings — one per expected call. The
    test should provide enough to cover all turns; if it runs out, the
    iterator raises StopIteration → the test fails loudly.

    The real `stream_prose_text` is an async generator yielding text
    deltas, so the double yields each turn's prose as a single delta."""
    it = iter(proses)

    async def fake(*args, **kwargs):
        yield next(it)

    monkeypatch.setattr("handler.scene_loop.stream_prose_text", fake)


def _patch_choice_author(monkeypatch, choices):
    """`choices` is an iterable of `ChoiceBundle` — one per expected call.
    Mirrors `_patch_prose_writer`."""
    it = iter(choices)

    async def fake(*args, **kwargs):
        return next(it)

    monkeypatch.setattr("handler.scene_loop.emit_choices", fake)


def _patch_gate(monkeypatch, offers):
    """`offers` is a bool (every turn) or an iterable of bools (one per
    expected call). Patches the free-text gate."""
    if isinstance(offers, bool):
        async def fake(*args, **kwargs):
            return offers
    else:
        it = iter(offers)

        async def fake(*args, **kwargs):
            return next(it)

    monkeypatch.setattr("handler.scene_loop.emit_free_text_gate", fake)


def _patch_caption(monkeypatch, captions):
    """`captions` is an iterable of `FreeTextCaption` — one per expected
    call. Mirrors `_patch_choice_author`. Use when the gate+cadence will
    surface a box; the helper also asserts (implicitly, via StopIteration)
    that it's called exactly as many times as captions supplied."""
    it = iter(captions)

    async def fake(*args, **kwargs):
        return next(it)

    monkeypatch.setattr("handler.scene_loop.emit_caption", fake)


def _patch_cadence(monkeypatch, ok: bool):
    """Force the engine-side free-text cadence gate to a fixed result,
    isolating run_turn/play_scene tests from the cooldown bookkeeping."""
    monkeypatch.setattr(
        "handler.scene_loop.free_text_cadence_ok", lambda state: ok
    )


def _patch_closing_prose(monkeypatch, text: str = "You step out into the dark."):
    """Stub the closing-prose LLM call (used on a WalkTo leave) so play_scene
    tests don't hit a real prose-writer run."""
    async def fake(*args, **kwargs):
        return text

    monkeypatch.setattr("handler.scene_loop.emit_closing_prose", fake)


def _walk_first_adjacent(prose_part, state, world):
    """Default `request_player_input`: pick option 0 on a Continue, type a
    stock line on free text, and walk to the first adjacent location on a
    soft-close (EndScene). Tests needing specific actions pass their own."""
    nxt = prose_part.next
    if nxt.type == "continue":
        return PickOption(index=0)
    if nxt.type == "free_text":
        return TypeText(text="I look around.")
    return WalkTo(location_id=world.locations[state.player_location].adjacent_to[0])


@pytest.fixture(autouse=True)
def _default_gate_off(monkeypatch):
    """Default: the free-text gate offers nothing, so run_turn/play_scene
    tests get deterministic canned-option behavior without a real TestModel
    gate call. Tests exercising free text re-patch via `_patch_gate` (the
    later setattr in the test body wins)."""
    async def fake(*args, **kwargs):
        return False

    monkeypatch.setattr("handler.scene_loop.emit_free_text_gate", fake)


@pytest.fixture
def bundle_and_state():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")  # tom is here
    return bundle, runtime, state


# ───────────────────────── _assemble_prose_part ─────────────────────────


def test_assemble_continue_maps_to_continue_scene():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    choices = ChoiceBundle(next=Continue(options=[
        Option(text="a"), Option(text="b"),
    ]))
    pp = _assemble_prose_part(world, state, "prose", choices)
    assert isinstance(pp.next, Continue)
    assert len(pp.next.options) == 2


def test_assemble_leave_with_no_satisfied_ending_maps_to_end_scene():
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    pp = _assemble_prose_part(
        world, state, "prose", ChoiceBundle(next=Leave()),
    )
    assert isinstance(pp.next, EndScene)


def test_assemble_leave_with_satisfied_ending_maps_to_end_game():
    """If the decision beat for an authored ending has fired, Leave
    resolves to EndGame with that ending id."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    ending = world.endings["published_story"]
    state.unlocked_beats = set(ending.requires_beats)
    pp = _assemble_prose_part(
        world, state, "prose", ChoiceBundle(next=Leave()),
    )
    assert isinstance(pp.next, EndGame)
    assert pp.next.ending_id == "published_story"


def test_assemble_free_text_offer_with_caption_builds_per_turn_free_text():
    """Offer + a caption → engine builds a per-turn FreeText carrying the
    caption author's caption and suggested move, dropping canned options."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    choices = ChoiceBundle(next=Continue(options=[
        Option(text="a"), Option(text="b"),
    ]))
    caption = FreeTextCaption(caption="What do you say to Tom?", suggested="I ask Tom about the keys.")
    pp = _assemble_prose_part(
        world, state, "prose", choices, free_text_offer=True, caption=caption,
    )
    assert isinstance(pp.next, FreeText)
    assert pp.next.prompt == "What do you say to Tom?"
    assert pp.next.suggested == "I ask Tom about the keys."
    assert pp.next.npc_id is None


def test_assemble_free_text_offer_without_caption_uses_fallback():
    """Offer stands but the caption author failed (caption=None) → engine
    falls back to the fixed-caption box with no suggestion."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    choices = ChoiceBundle(next=Continue(options=[
        Option(text="a"), Option(text="b"),
    ]))
    pp = _assemble_prose_part(world, state, "prose", choices, free_text_offer=True)
    assert pp.next == FREE_TEXT_FALLBACK
    assert pp.next.prompt == "What do you do next?"
    assert pp.next.suggested is None


def test_assemble_leave_beats_free_text_offer():
    """Precedence: Leave wins over a gate offer — a closing scene never
    gets a free-text box."""
    world = load_world(FIXTURE_VALID)
    state = make_initial_state(world)
    pp = _assemble_prose_part(
        world, state, "prose", ChoiceBundle(next=Leave()), free_text_offer=True,
    )
    assert isinstance(pp.next, EndScene)


# ───────────────────────── run_turn happy path ─────────────────────────


async def test_run_turn_applies_state_changes_and_appends_record(monkeypatch, bundle_and_state):
    bundle, runtime, state = bundle_and_state
    state.trust["tom"] = 1  # one more warm turn reaches `open` → derives the milestone
    _patch_phase1(monkeypatch, StateChangeBundle(
        deltas=[QualityDelta(npc_id="tom", trust="+1")],
    ))
    _patch_prose_writer(monkeypatch, [
        "Tom looks down at his pint, and then up at you.",
    ])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="say nothing"),
            Option(text="ask gently"),
        ])),
    ])

    result = await run_turn(bundle, runtime, state)

    assert isinstance(result.next, Continue)
    # State change applied.
    assert "tom_named_a_vehicle" in state.unlocked_beats
    # TurnRecord appended with prose + applied changes.
    assert len(state.dialogue_history) == 1
    record = state.dialogue_history[0]
    assert record.prose.startswith("Tom looks down")
    assert record.applied_state_changes[0].beat_id == "tom_named_a_vehicle"


async def test_run_turn_phase1_fallback_on_unexpected_behavior(monkeypatch, bundle_and_state):
    """Phase 1 retries exhausted → empty bundle, no state change applied."""
    bundle, runtime, state = bundle_and_state

    async def fake_phase1_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated retry exhaustion")

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1_fails)
    _patch_prose_writer(monkeypatch, ["Tom drinks."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="watch"), Option(text="go"),
        ])),
    ])

    result = await run_turn(bundle, runtime, state)
    assert state.unlocked_beats == set()
    assert isinstance(result.next, Continue)
    # TurnRecord still appended (no state changes).
    assert state.dialogue_history[-1].applied_state_changes == []


async def test_run_turn_prose_writer_fallback(monkeypatch, bundle_and_state):
    """Prose writer retries exhausted → engine-authored generic prose +
    choice author still runs to author options."""
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)

    async def fake_prose_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr("handler.scene_loop.stream_prose_text", fake_prose_fails)
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="wait"), Option(text="look"),
        ])),
    ])

    result = await run_turn(bundle, runtime, state)
    # No engine metatext leaks into prose.
    assert "[engine" not in result.prose.lower()
    assert isinstance(result.next, Continue)


async def test_run_turn_prose_writer_fallback_does_not_pollute_message_history(
    monkeypatch, bundle_and_state
):
    """Failed prose-writer retries → fallback prose to player, but NOT
    appended to runtime.prose_writer_message_history."""
    bundle, runtime, state = bundle_and_state
    runtime.prose_writer_message_history = ["existing-anchor-marker"]  # type: ignore[list-item]

    _patch_phase1(monkeypatch)

    async def fake_prose_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr("handler.scene_loop.stream_prose_text", fake_prose_fails)
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="a"), Option(text="b"),
        ])),
    ])

    await run_turn(bundle, runtime, state)
    assert runtime.prose_writer_message_history == ["existing-anchor-marker"], (
        "prose-writer fallback must not mutate prose_writer_message_history"
    )


async def test_run_turn_choice_author_fallback(monkeypatch, bundle_and_state):
    """Choice author retries exhausted → generic 2-option Continue."""
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The peat fire crackles."])

    async def fake_choices_fails(*args, **kwargs):
        raise UnexpectedModelBehavior("simulated")

    monkeypatch.setattr("handler.scene_loop.emit_choices", fake_choices_fails)

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, Continue)
    assert len(result.next.options) == 2


async def test_run_turn_leave_resolves_to_endgame_when_satisfied(monkeypatch, bundle_and_state):
    """Choice author Leave + satisfied ending in state → engine assembles
    EndGame with the resolved ending id."""
    bundle, runtime, state = bundle_and_state
    # Satisfy the published_story ending's prereqs.
    state.unlocked_beats = set(bundle.world.endings["published_story"].requires_beats)

    _patch_phase1(monkeypatch)
    _patch_village_attention(monkeypatch)  # Leave closes the scene → attention agent runs
    _patch_prose_writer(monkeypatch, ["A pause."])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, EndGame)
    assert result.next.ending_id == "published_story"


async def test_run_turn_leave_resolves_to_endscene_when_no_ending_satisfied(monkeypatch, bundle_and_state):
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)
    _patch_village_attention(monkeypatch)  # Leave closes the scene → attention agent runs
    _patch_prose_writer(monkeypatch, ["A pause."])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, EndScene)


async def test_run_turn_gate_offers_free_text_with_npc_present(monkeypatch):
    """Free text is the "words to a person" channel: with an NPC present and
    the gate + cadence allowing, the caption author runs and its caption +
    suggestion ride the FreeText, overriding Continue."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")  # tom is here

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["Tom waits, watching you."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_gate(monkeypatch, True)
    _patch_cadence(monkeypatch, True)
    _patch_caption(monkeypatch, [
        FreeTextCaption(caption="What do you say to Tom?", suggested="I ask him about Megan."),
    ])

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, FreeText)
    assert result.next.prompt == "What do you say to Tom?"
    assert result.next.suggested == "I ask him about Megan."
    assert result.next.npc_id is None


async def test_run_turn_cadence_suppresses_gate_offer(monkeypatch):
    """Gate offers (NPC present) but the engine cadence declines → no caption
    call, the choice author's Continue stands."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")  # tom is here

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["Tom waits, watching you."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_gate(monkeypatch, True)
    _patch_cadence(monkeypatch, False)

    async def _fail_caption(*args, **kwargs):
        raise AssertionError("caption author must not be called when cadence declines")

    monkeypatch.setattr("handler.scene_loop.emit_caption", _fail_caption)

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, Continue)


async def test_run_turn_caption_failure_falls_back_to_fixed_box(monkeypatch):
    """Gate+cadence say offer but the caption author exhausts retries → the
    box still surfaces, with the fixed fallback caption and no suggestion."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_slaughtered_lamb")  # tom is here

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["Tom waits, watching you."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_gate(monkeypatch, True)
    _patch_cadence(monkeypatch, True)

    async def _boom(*args, **kwargs):
        raise UnexpectedModelBehavior("retries exhausted")

    monkeypatch.setattr("handler.scene_loop.emit_caption", _boom)

    result = await run_turn(bundle, runtime, state)
    assert result.next == FREE_TEXT_FALLBACK


async def test_run_turn_solo_location_never_offers_free_text(monkeypatch):
    """Structural precondition: no NPC present → free text is never offered,
    even if the gate would say yes and cadence allows. Solo scenes are
    click-driven; the choice author's Continue stands."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_old_vicarage")  # solo location — no NPC

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The study is silent."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])
    _patch_gate(monkeypatch, True)
    _patch_cadence(monkeypatch, True)

    async def _fail_caption(*args, **kwargs):
        raise AssertionError("caption author must not run in a solo scene")

    monkeypatch.setattr("handler.scene_loop.emit_caption", _fail_caption)

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, Continue)


async def test_run_turn_solo_location_skips_gate_llm_call(monkeypatch):
    """The gate LLM call is never even scheduled on a solo turn — the
    "words to a person" precondition is enforced before authoring, so no
    cost is paid where there is no one to speak to."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    enter_scene(state, world, "the_old_vicarage")  # solo location — no NPC

    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The study is silent."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    async def _must_not_run(*args, **kwargs):
        raise AssertionError("free-text gate must not be called in a solo scene")

    monkeypatch.setattr("handler.scene_loop.emit_free_text_gate", _must_not_run)

    result = await run_turn(bundle, runtime, state)
    assert isinstance(result.next, Continue)


# ───────────────────────── dossier standings (npc_standing) ─────────────────────────


async def test_run_turn_emits_npc_standing_on_band_change(monkeypatch, bundle_and_state):
    """A turn that moves a tracked, present NPC across a confession band emits
    exactly one npc_standing for them, carrying the new band. Tom (steady, width
    1) starts cold; a +1 trust nudge reaches `thawing`."""
    bundle, runtime, state = bundle_and_state  # entered at the_slaughtered_lamb (tom)
    assert state.trust["tom"] == 0
    _patch_phase1(monkeypatch, StateChangeBundle(
        deltas=[QualityDelta(npc_id="tom", trust="+1")],
    ))
    _patch_prose_writer(monkeypatch, ["Tom sets his glass down."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    events: list[dict] = []

    async def collect(ev: dict) -> None:
        events.append(ev)

    await run_turn(bundle, runtime, state, emit_event=collect)

    standings = [e for e in events if e["event"] == "npc_standing"]
    assert len(standings) == 1
    assert standings[0]["npc_id"] == "tom"
    assert standings[0]["band"] == "thawing"


async def test_run_turn_no_npc_standing_when_band_unchanged(monkeypatch, bundle_and_state):
    """A turn that doesn't move any tracked NPC's band emits no npc_standing —
    the board updates on transitions, not every turn."""
    bundle, runtime, state = bundle_and_state
    _patch_phase1(monkeypatch)  # empty bundle: no deltas
    _patch_prose_writer(monkeypatch, ["Tom says nothing."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    events: list[dict] = []

    async def collect(ev: dict) -> None:
        events.append(ev)

    await run_turn(bundle, runtime, state, emit_event=collect)
    assert [e for e in events if e["event"] == "npc_standing"] == []


# ───────────────────────── play_scene ─────────────────────────


async def test_play_scene_drives_until_walk(monkeypatch):
    """Three turns: turn 1+2 Continue (picks), turn 3 Leave → soft-close →
    the player walks → play_scene returns a finalized EndScene whose
    to_location names the destination, with the closing prose streamed."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2", "t3"])
    _patch_closing_prose(monkeypatch, "You shoulder the door open.")
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="stay"), Option(text="ask")])),
        ChoiceBundle(next=Continue(options=[Option(text="watch"), Option(text="speak")])),
        ChoiceBundle(next=Leave()),
    ])

    result = await play_scene(
        bundle, runtime, state, request_player_input=_walk_first_adjacent
    )
    assert isinstance(result.next, EndScene)
    assert result.next.to_location in world.locations["the_slaughtered_lamb"].adjacent_to
    assert result.prose == "You shoulder the door open."
    # Three TurnRecords (the closing prose is not a turn).
    assert len(state.dialogue_history) == 3
    # Two Continue turns had a pick; the soft-close turn had none (walked).
    assert state.dialogue_history[0].player_pick is not None
    assert state.dialogue_history[1].player_pick is not None
    assert state.dialogue_history[2].player_pick is None


def _gated_scene_fixtures(monkeypatch):
    """Two Continue turns then a Leave soft-close, with all LLM calls stubbed —
    shared setup for the leave-affordance arm tests below. Returns
    (bundle, runtime, state)."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2", "t3"])
    _patch_closing_prose(monkeypatch, "You step back out into the lane.")
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="stay"), Option(text="ask")])),
        ChoiceBundle(next=Continue(options=[Option(text="watch"), Option(text="speak")])),
        ChoiceBundle(next=Leave()),
    ])
    return bundle, runtime, state


async def test_play_scene_plain_continue_has_no_leave_entry_or_midscene_arm(monkeypatch):
    """No authored leave on any Continue turn: the engine appends no exit entry
    (there is no "Step outside") and arms walk_ready ONLY at the narrator
    soft-close — never mid-scene, so the map cannot end a plain Continue turn."""
    bundle, runtime, state = _gated_scene_fixtures(monkeypatch)
    events: list[dict] = []

    async def collect(ev):
        events.append(ev)

    result = await play_scene(
        bundle, runtime, state,
        request_player_input=_walk_first_adjacent,
        emit_event=collect,
    )

    option_events = [e for e in events if e["event"] == "options"]
    # No Continue turn ever carries a leave entry.
    assert not any(o.get("leave") for e in option_events for o in e["options"])
    # walk_ready fires exactly once — at the spent-scene cue, not mid-scene.
    walk_ready_idxs = [i for i, e in enumerate(events) if e["event"] == "walk_ready"]
    spent_idx = next(i for i, e in enumerate(events) if e["event"] == "scene_spent")
    assert walk_ready_idxs == [spent_idx + 1]
    # The soft-close still ends the scene the same way.
    assert isinstance(result.next, EndScene)


async def test_play_scene_authored_leave_arms_map_midscene(monkeypatch):
    """A Continue whose author marked an option `leave: true` is the player's
    in-fiction exit: the engine carries the flag to the client AND arms
    walk_ready that turn (so picking the exit can walk). Together with the
    turn-2 soft-close, walk_ready therefore fires twice."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2"])
    _patch_closing_prose(monkeypatch)
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[
            Option(text="stay"), Option(text="thank her and step out", leave=True),
        ])),
        ChoiceBundle(next=Leave()),
    ])
    events: list[dict] = []

    async def collect(ev):
        events.append(ev)

    await play_scene(
        bundle, runtime, state,
        request_player_input=_walk_first_adjacent,
        emit_event=collect,
    )

    option_events = [e for e in events if e["event"] == "options"]
    # Turn 1 carries the authored leave option through to the client.
    assert any(o.get("leave") for o in option_events[0]["options"])
    # Map armed on the authored-leave turn AND the soft-close → two walk_ready.
    assert sum(e["event"] == "walk_ready" for e in events) == 2


async def test_play_scene_resolves_ending_on_close(monkeypatch):
    """A soft-close into a satisfied ending returns EndGame without awaiting a
    walk — the climax committed in fiction, the player never had to leave."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1"])
    # Drive an ending to satisfied, then Leave on turn 1 → EndGame at close.
    monkeypatch.setattr(
        "handler.scene_loop.resolved_ending_id_at_close",
        lambda world, state: next(iter(world.endings)),
    )
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    def _never_walk(prose_part, s, w):
        raise AssertionError("soft-close into an ending must not await a walk")

    result = await play_scene(
        bundle, runtime, state,
        request_player_input=_never_walk,
    )
    assert isinstance(result.next, EndGame)


async def test_play_scene_dispatches_free_text(monkeypatch):
    """Gate offers free text on turn 1 → request_player_input returns a
    TypeText → typed string becomes the turn's player_pick (flagged typed).
    Turn 2 the gate declines and the choice author emits Leave → soft-close
    → the player walks out."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2"])
    _patch_closing_prose(monkeypatch)
    _patch_choice_author(monkeypatch, [
        # Turn 1 Continue is overridden by the gate offer; turn 2 Leave wins.
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
        ChoiceBundle(next=Leave()),
    ])
    _patch_gate(monkeypatch, [True, False])
    _patch_cadence(monkeypatch, True)
    _patch_caption(monkeypatch, [
        FreeTextCaption(caption="What do you say?", suggested="I tell him about Megan."),
    ])

    def request(prose_part, s, w):
        nxt = prose_part.next
        if nxt.type == "free_text":
            assert nxt.prompt == "What do you say?"
            assert nxt.suggested == "I tell him about Megan."
            assert nxt.npc_id is None
            return TypeText(text="I came because of Megan.")
        # turn 2 soft-close → walk out
        return WalkTo(location_id=w.locations[s.player_location].adjacent_to[0])

    result = await play_scene(
        bundle, runtime, state, request_player_input=request,
    )
    assert isinstance(result.next, EndScene)
    # First TurnRecord got the typed string as the player's pick, flagged typed.
    assert state.dialogue_history[0].player_pick is not None
    assert state.dialogue_history[0].player_pick.text == "I came because of Megan."
    assert state.dialogue_history[0].player_pick_typed is True
    # The soft-close turn has no pick (the player walked).
    assert state.dialogue_history[1].player_pick is None


async def test_play_scene_resets_prose_writer_message_history_at_scene_start(monkeypatch):
    """enter_scene → runtime.prose_writer_message_history must reset to
    [] even if the previous scene left it populated."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    runtime.prose_writer_message_history = ["leftover-from-prior-scene"]  # type: ignore[list-item]
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)

    captured: list[list] = []

    async def fake_prose(bundle, rt, *args, **kwargs):
        # Observe message_history at the time of the first prose-writer
        # call.
        captured.append(list(rt.prose_writer_message_history))
        yield "end"

    monkeypatch.setattr("handler.scene_loop.stream_prose_text", fake_prose)
    _patch_closing_prose(monkeypatch)
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    await play_scene(bundle, runtime, state, request_player_input=_walk_first_adjacent)
    # The first observed prose-writer call MUST see an empty history.
    assert captured == [[]]


async def test_play_scene_caps_turns_and_forces_soft_close(monkeypatch):
    """A scene whose choice author never emits Leave and whose player never
    elects to leave must still end: at MAX_TURNS_PER_SCENE the engine converts
    the turn into a narrator soft-close (EndScene), so play_scene returns a
    finalized EndScene and the player is walked out. This bounds a milling
    scene (the 40-turns-in-one-location pathology) structurally."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    # Every turn moves standing, so the content-derived soft-close (scene_is_spent)
    # never trips — a productive scene is held open. Oscillate Tom's guard ±1 so the
    # realized delta is non-zero EVERY turn (a flat +1 would saturate at TRUST_MAX
    # and then realize 0, correctly going dry — that's the dead-room path, not what
    # this test isolates). This pins the FLAT cap as the sole terminator: a scene
    # that keeps paying off but is never left must still end at MAX_TURNS_PER_SCENE.
    _patch_phase1(monkeypatch, [
        StateChangeBundle(deltas=[QualityDelta(npc_id="tom", guard="+1")]),
        StateChangeBundle(deltas=[QualityDelta(npc_id="tom", guard="-1")]),
    ])
    # Supply more prose/choices than the cap so a failure to cap overruns into
    # StopIteration (a loud failure) rather than silently passing.
    _patch_prose_writer(monkeypatch, [f"t{i}" for i in range(MAX_TURNS_PER_SCENE + 5)])
    _patch_closing_prose(monkeypatch, "You finally step away.")
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="stay"), Option(text="press")]))
        for _ in range(MAX_TURNS_PER_SCENE + 5)
    ])

    # The player never volunteers to leave — pick option 0 on every Continue.
    # The only exit is the engine's cap → forced soft-close → walk.
    def never_leave(prose_part, s, w):
        if prose_part.next.type == "continue":
            return PickOption(index=0)
        return WalkTo(location_id=w.locations[s.player_location].adjacent_to[0])

    result = await play_scene(
        bundle, runtime, state, request_player_input=never_leave,
    )

    assert isinstance(result.next, EndScene)
    assert result.next.to_location in world.locations["the_slaughtered_lamb"].adjacent_to
    # Exactly the cap's worth of turns ran — no more. The capped turn is the one
    # with no pick (the player was walked out by the forced soft-close).
    assert len(state.dialogue_history) == MAX_TURNS_PER_SCENE
    assert all(r.player_pick is not None for r in state.dialogue_history[:-1])
    assert state.dialogue_history[-1].player_pick is None


# ───────────────────────── play_session ─────────────────────────


def _make_session_fixtures():
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    return world, bundle, runtime, state


async def test_play_session_calls_both_callbacks_and_terminates_on_end_game(monkeypatch):
    """Two scenes. Scene 2's third turn fires the decision beat via
    Phase 1, then the choice author emits Leave → engine resolves
    EndGame."""
    world, bundle, runtime, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_closing_prose(monkeypatch)
    # Phase 1 fires nothing until scene 2 turn 2 (the fourth call).
    phase1_outputs = iter([
        StateChangeBundle(),  # scene 1 turn 1
        StateChangeBundle(),  # scene 1 turn 2 (Leave)
        StateChangeBundle(),  # scene 2 turn 1
        StateChangeBundle(   # scene 2 turn 2: fire the decision beat
            commitments=[UnlockBeat(beat_id=b) for b in decision_beats],
        ),
    ])

    async def fake_phase1(*args, **kwargs):
        return next(phase1_outputs)

    monkeypatch.setattr("handler.scene_loop.emit_state_changes", fake_phase1)
    _patch_prose_writer(monkeypatch, ["t1", "t1-end", "t2", "t2-end"])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="ask"), Option(text="go")])),
        ChoiceBundle(next=Leave()),  # scene 1 soft-close (no ending) → player walks
        ChoiceBundle(next=Continue(options=[Option(text="ask"), Option(text="go")])),
        ChoiceBundle(next=Leave()),  # scene 2 EndGame (decision beat now in state)
    ])

    log_entries: list[tuple[State, ChronicleEntry]] = []
    closes: list[ProsePart] = []

    def record_log(s: State, o: ChronicleEntry) -> None:
        log_entries.append((s, o))

    terminal = await play_session(
        bundle, runtime, state, world,
        request_player_input=_walk_first_adjacent,
        on_chronicle_entry=record_log,
        on_scene_close=closes.append,
    )

    # Terminated on EndGame, state reflects it.
    assert isinstance(terminal.next, EndGame)
    assert state.game_over is not None
    assert state.game_over.ending_id == "published_story"
    # on_scene_close + chronicler fired once per scene end (walk-out, then EndGame).
    assert len(closes) == 2
    assert len(log_entries) == 2
    assert len(state.case_file_entries) == 2


async def test_play_session_no_input_when_first_turn_ends_game(monkeypatch):
    """If the session ends in EndGame on the first scene's first turn (the
    decision beat fires, Leave resolves straight to EndGame), the player is
    never asked for input — request_player_input is not called."""
    world, bundle, runtime, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_phase1(monkeypatch, StateChangeBundle(
        commitments=[UnlockBeat(beat_id=b) for b in decision_beats],
    ))
    _patch_prose_writer(monkeypatch, ["opening"])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    input_calls: list[str] = []

    def request(prose_part, s, w):
        input_calls.append(s.player_location)
        return _walk_first_adjacent(prose_part, s, w)

    terminal = await play_session(
        bundle, runtime, state, world,
        request_player_input=request,
    )
    assert isinstance(terminal.next, EndGame)
    assert input_calls == []


async def test_play_session_case_file_entries_accumulate_on_state(monkeypatch):
    world, bundle, runtime, state = _make_session_fixtures()
    decision_beats = world.endings["published_story"].requires_beats

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_chronicler(monkeypatch)
    _patch_phase1(monkeypatch, StateChangeBundle(
        commitments=[UnlockBeat(beat_id=b) for b in decision_beats],
    ))
    _patch_prose_writer(monkeypatch, ["t1"])
    _patch_choice_author(monkeypatch, [ChoiceBundle(next=Leave())])

    await play_session(
        bundle, runtime, state, world,
        request_player_input=_walk_first_adjacent,
    )
    assert len(state.case_file_entries) == 1
    assert state.case_file_entries[0] == "entry"


# ───────────────────────── mid-scene leave (player-authored movement) ─────────────────────────


async def test_play_scene_walk_mid_continue_ends_scene(monkeypatch):
    """The player walks out on turn 1 while a Continue is offered (no pick):
    play_scene runs the closing prose and returns EndScene(to_location). The
    turn keeps no player_pick (they left instead of choosing)."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["Tom eyes you over his pint."])
    _patch_closing_prose(monkeypatch, "You leave him to it.")
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="stay"), Option(text="ask")])),
    ])

    dest = world.locations["the_slaughtered_lamb"].adjacent_to[0]

    def request(prose_part, s, w):
        assert prose_part.next.type == "continue"  # mid-scene, options offered
        return WalkTo(location_id=dest)

    result = await play_scene(bundle, runtime, state, request_player_input=request)
    assert isinstance(result.next, EndScene)
    assert result.next.to_location == dest
    assert result.prose == "You leave him to it."
    assert len(state.dialogue_history) == 1
    assert state.dialogue_history[0].player_pick is None


async def test_play_scene_walk_resolves_endgame_when_ending_satisfied(monkeypatch):
    """A WalkTo while an ending's beats are already satisfied → EndGame,
    reusing the last prose. emit_closing_prose is NOT called (no journey is
    narrated for a game that's ending)."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"
    # Satisfy the ending's prereqs up front.
    state.unlocked_beats = set(world.endings["published_story"].requires_beats)

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch)
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["The room goes quiet."])
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="stay"), Option(text="ask")])),
    ])

    async def _closing_must_not_run(*args, **kwargs):
        raise AssertionError("emit_closing_prose must not run on a walk-to-ending")

    monkeypatch.setattr("handler.scene_loop.emit_closing_prose", _closing_must_not_run)

    dest = world.locations["the_slaughtered_lamb"].adjacent_to[0]
    result = await play_scene(
        bundle, runtime, state,
        request_player_input=lambda pp, s, w: WalkTo(location_id=dest),
    )
    assert isinstance(result.next, EndGame)
    assert result.next.ending_id == "published_story"
    assert result.prose == "The room goes quiet."  # last prose reused, no closing line


async def test_walk_close_ratchets_attention_before_ending_resolution(monkeypatch):
    """Ordering regression: the scene-close village-attention agent runs and
    ratchets village_attention BEFORE resolved_ending_id_at_close, so a scene of
    aggression can trip run_out on the closing turn (not a scene late). We spy on
    resolution to capture the heat value at the moment it's consulted."""
    import handler.scene_loop as sl

    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    runtime = make_scene_runner_runtime()
    state = make_initial_state(world)
    state.player_location = "the_slaughtered_lamb"

    _patch_populator(monkeypatch)
    _patch_village_attention(monkeypatch, level="hostile")  # the scene turned the village
    _patch_phase1(monkeypatch)
    _patch_prose_writer(monkeypatch, ["t1", "t2"])
    _patch_closing_prose(monkeypatch, "Out into the rain.")
    _patch_choice_author(monkeypatch, [
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
        ChoiceBundle(next=Continue(options=[Option(text="a"), Option(text="b")])),
    ])

    seen: dict[str, int] = {}
    real_resolve = sl.resolved_ending_id_at_close

    def spy(world_, state_):
        seen["attention_at_resolution"] = state_.village_attention
        return real_resolve(world_, state_)

    monkeypatch.setattr(sl, "resolved_ending_id_at_close", spy)

    dest = world.locations["the_slaughtered_lamb"].adjacent_to[0]
    actions = iter([PickOption(index=0), WalkTo(location_id=dest)])
    await play_scene(
        bundle, runtime, state,
        request_player_input=lambda pp, s, w: next(actions),
    )

    assert state.village_attention == 3                 # ratcheted to hostile at close
    assert seen["attention_at_resolution"] == 3         # ...and BEFORE resolution ran
