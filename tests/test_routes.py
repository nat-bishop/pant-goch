"""POST route tests — the single `action_future` + legality gating.

The routes resolve one shared future with a tagged PlayerAction. A /walk is
legal at every input-await (so the map is a way out mid-scene); a /pick or
/free_text is legal only in its matching `awaiting` state. Driven by setting
`handler.main.SESSION` directly — no SSE stream, no live llama.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic_ai.models.test import TestModel

import handler.main as main
from handler.agents import (
    build_scene_runner_bundle,
    make_chronicler_runtime,
    make_scene_runner_runtime,
)
from handler.main import FreeTextBody, PickBody, SessionRuntime, WalkBody
from handler.schemas import Continue, EndScene, FreeText, Option
from handler.scene_loop import PickOption, TypeText, WalkTo
from handler.state import make_initial_state
from handler.world import load_world

FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"

_AWAIT_OPTIONS = Continue(options=[Option(text="a"), Option(text="b")])
_AWAIT_FREE_TEXT = FreeText(prompt="What do you do?")
_AWAIT_SOFT_CLOSE = EndScene()


def _session(awaiting_next, *, location: str = "the_old_vicarage") -> SessionRuntime:
    """A SessionRuntime with one pending action_future awaiting the given
    ProsePart.next. the_old_vicarage is adjacent only to the_post_office."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    state = make_initial_state(world)
    state.player_location = location
    rt = SessionRuntime(
        bundle=bundle,
        scene_runner=make_scene_runner_runtime(),
        chronicler=make_chronicler_runtime(),
        state=state,
        world=world,
    )
    rt.action_future = asyncio.get_event_loop().create_future()
    rt.awaiting_next = awaiting_next
    return rt


@pytest.fixture(autouse=True)
def _restore_session():
    prev = main.SESSION
    yield
    main.SESSION = prev


async def test_walk_resolves_mid_scene_while_awaiting_options():
    """/walk is legal even when the scene is awaiting an option pick — the
    map is a way out of any turn."""
    rt = _session(_AWAIT_OPTIONS)
    main.SESSION = rt
    res = await main.walk(WalkBody(to_location="the_post_office"))
    assert res["status"] == "ok"
    assert rt.action_future.result() == WalkTo(location_id="the_post_office")


async def test_walk_resolves_at_soft_close():
    rt = _session(_AWAIT_SOFT_CLOSE)
    main.SESSION = rt
    await main.walk(WalkBody(to_location="the_post_office"))
    assert rt.action_future.result() == WalkTo(location_id="the_post_office")


async def test_walk_rejects_non_adjacent():
    rt = _session(_AWAIT_OPTIONS)  # vicarage is adjacent only to post_office
    main.SESSION = rt
    with pytest.raises(HTTPException) as ei:
        await main.walk(WalkBody(to_location="the_slaughtered_lamb"))
    assert ei.value.status_code == 400
    assert not rt.action_future.done()


async def test_pick_ok_while_awaiting_options():
    rt = _session(_AWAIT_OPTIONS)
    main.SESSION = rt
    await main.pick(PickBody(option_idx=2))
    assert rt.action_future.result() == PickOption(index=2)


async def test_pick_rejected_at_soft_close():
    """A soft-close offers no options — /pick must 409."""
    rt = _session(_AWAIT_SOFT_CLOSE)
    main.SESSION = rt
    with pytest.raises(HTTPException) as ei:
        await main.pick(PickBody(option_idx=0))
    assert ei.value.status_code == 409
    assert not rt.action_future.done()


async def test_free_text_ok_while_awaiting_free_text():
    rt = _session(_AWAIT_FREE_TEXT)
    main.SESSION = rt
    await main.free_text(FreeTextBody(text="  I ask about the keys.  "))
    assert rt.action_future.result() == TypeText(text="I ask about the keys.")


async def test_free_text_rejected_while_awaiting_options():
    rt = _session(_AWAIT_OPTIONS)
    main.SESSION = rt
    with pytest.raises(HTTPException) as ei:
        await main.free_text(FreeTextBody(text="hi"))
    assert ei.value.status_code == 409
    assert not rt.action_future.done()


async def test_routes_409_when_no_future_pending():
    rt = _session(_AWAIT_OPTIONS)
    rt.action_future = None  # nothing awaiting
    main.SESSION = rt
    for call in (
        main.pick(PickBody(option_idx=0)),
        main.walk(WalkBody(to_location="the_post_office")),
        main.free_text(FreeTextBody(text="hi")),
    ):
        with pytest.raises(HTTPException) as ei:
            await call
        assert ei.value.status_code == 409
