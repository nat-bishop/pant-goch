"""Prose-writer tool-loop unit tests.

Verifies the four tools on the prose writer route correctly:
  - pause_for_continue → await_continue callback resolves the suspension
  - request_free_text → await_free_text callback returns the player text
  - end_scene → deps.termination = ("scene", location_id)
  - end_game → deps.termination = ("game", ending_id)

Plus the tool-arg validation paths (npc_id not present, location_id
non-adjacent, ending_id not satisfied) raise ModelRetry.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from handler.agents import ProseWriterDeps, _register_prose_writer_tools
from handler.world import load_world


FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── tool body tests ─────────────────────────


async def _build_test_agent_with_tools():
    """Build a minimal agent that has the four tools registered, so we
    can invoke the tool bodies directly via the toolset."""
    from pydantic_ai import Agent
    agent: Agent[ProseWriterDeps, str] = Agent(
        model="test",  # TestModel placeholder; we won't actually run agent.run here
        deps_type=ProseWriterDeps,
    )
    _register_prose_writer_tools(agent)
    return agent


def _make_deps(**overrides) -> ProseWriterDeps:
    """Construct a deps with no-op async callbacks for testing."""
    async def _ac() -> None:
        return None

    async def _aft(prompt: str, npc_id: str | None) -> str:
        return "player typed this"

    return ProseWriterDeps(
        await_continue=overrides.pop("await_continue", _ac),
        await_free_text=overrides.pop("await_free_text", _aft),
        present_npc_ids=overrides.pop("present_npc_ids", ("tom", "maud")),
        adjacent_ids=overrides.pop("adjacent_ids", ("the_post_office",)),
        satisfied_ending_ids=overrides.pop("satisfied_ending_ids", ("published_story",)),
        **overrides,
    )


async def test_pause_for_continue_awaits_callback():
    """pause_for_continue must call deps.await_continue and return only
    after it resolves."""
    event = asyncio.Event()

    async def slow_continue() -> None:
        await event.wait()

    deps = _make_deps(await_continue=slow_continue)
    agent = await _build_test_agent_with_tools()

    # Find the registered tool and invoke its underlying function.
    tool = _find_tool(agent, "pause_for_continue")
    assert tool is not None

    # Schedule the tool call; it should block until we set the event.
    task = asyncio.create_task(_call_tool(tool, deps))
    await asyncio.sleep(0.01)
    assert not task.done()
    event.set()
    result = await task
    assert result == ""


async def test_request_free_text_returns_player_text():
    deps = _make_deps()
    agent = await _build_test_agent_with_tools()
    tool = _find_tool(agent, "request_free_text")
    result = await _call_tool(tool, deps, prompt="What do you say?", npc_id="tom")
    assert result == "player typed this"


async def test_request_free_text_rejects_unknown_npc():
    from pydantic_ai import ModelRetry
    deps = _make_deps(present_npc_ids=("tom",))
    agent = await _build_test_agent_with_tools()
    tool = _find_tool(agent, "request_free_text")
    with pytest.raises(ModelRetry, match="not present"):
        await _call_tool(tool, deps, prompt="hi", npc_id="ghost")


async def test_end_scene_sets_termination_with_location():
    deps = _make_deps(adjacent_ids=("the_post_office",))
    agent = await _build_test_agent_with_tools()
    tool = _find_tool(agent, "end_scene")
    await _call_tool(tool, deps, location_id="the_post_office")
    assert deps.termination == ("scene", "the_post_office")


async def test_end_scene_sets_termination_with_none():
    deps = _make_deps()
    agent = await _build_test_agent_with_tools()
    tool = _find_tool(agent, "end_scene")
    await _call_tool(tool, deps)  # no location_id
    assert deps.termination == ("scene", None)


async def test_end_scene_rejects_non_adjacent():
    from pydantic_ai import ModelRetry
    deps = _make_deps(adjacent_ids=("the_post_office",))
    agent = await _build_test_agent_with_tools()
    tool = _find_tool(agent, "end_scene")
    with pytest.raises(ModelRetry, match="not adjacent"):
        await _call_tool(tool, deps, location_id="knighton")


async def test_end_game_sets_termination_with_ending():
    deps = _make_deps(satisfied_ending_ids=("published_story",))
    agent = await _build_test_agent_with_tools()
    tool = _find_tool(agent, "end_game")
    await _call_tool(tool, deps, ending_id="published_story")
    assert deps.termination == ("game", "published_story")


async def test_end_game_rejects_unsatisfied_ending():
    from pydantic_ai import ModelRetry
    deps = _make_deps(satisfied_ending_ids=())  # no ending currently satisfied
    agent = await _build_test_agent_with_tools()
    tool = _find_tool(agent, "end_game")
    with pytest.raises(ModelRetry, match="not currently"):
        await _call_tool(tool, deps, ending_id="published_story")


async def test_prose_writer_deps_terminate_helpers():
    """Direct unit test on the helpers — useful for engine-side
    assertions in emit_prose."""
    deps = _make_deps()
    deps.terminate_scene("the_post_office")
    assert deps.termination == ("scene", "the_post_office")

    deps2 = _make_deps()
    deps2.terminate_game("published_story")
    assert deps2.termination == ("game", "published_story")


# ───────────────────────── helpers ─────────────────────────


def _find_tool(agent, name: str):
    """Walk the agent's toolset to find a registered tool by name. The
    toolset structure varies by pydantic-ai version; this helper
    encapsulates the lookup so tests stay tight."""
    # FunctionToolset.tools is a dict[name, Tool]
    toolset = getattr(agent, "_function_toolset", None) or getattr(agent, "_toolset", None)
    if toolset is None:
        # pydantic-ai >= 1.x exposes via _toolset attribute or _builtin_toolset
        for attr in ("_function_toolset", "_builtin_toolset"):
            ts = getattr(agent, attr, None)
            if ts is not None:
                toolset = ts
                break
    if toolset is None:
        raise RuntimeError(f"could not locate toolset on agent (attrs: {dir(agent)})")
    tools = getattr(toolset, "tools", None) or getattr(toolset, "_tools", {})
    return tools.get(name)


async def _call_tool(tool, deps: ProseWriterDeps, **kwargs):
    """Invoke a registered tool's underlying function with a fake
    RunContext that exposes deps. The function-schema layer handles
    the str args directly when called this way."""
    from pydantic_ai import RunContext
    # RunContext signature: model, usage, prompt, messages, run_step, tool_call_id,
    # tool_name, retries, retry, tool_call_token, deps, ...
    # We construct the bare minimum it needs for our tool bodies.
    ctx = _make_run_context(deps)
    # Tool stores the wrapped function in `function`. Call it with ctx + args.
    func = tool.function
    return await func(ctx, **kwargs)


def _make_run_context(deps: ProseWriterDeps):
    """Construct a RunContext-like object exposing `deps`. Tool bodies
    only access ctx.deps, so a SimpleNamespace works."""
    from types import SimpleNamespace
    return SimpleNamespace(deps=deps)
