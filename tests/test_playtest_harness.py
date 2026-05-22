"""Playtest harness tests (Bucket 1 + run_session smoke).

Invariant evaluators are pure functions of a SessionArtifact, tested with a
duck-typed context. The run_session smoke test drives the real engine loop with
TestModel + monkeypatched emit_* (same pattern as test_scene_loop) and a stub
player — no Gemini call, no live llama.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.models.test import TestModel

from handler.agents import build_scene_runner_bundle
from handler.schemas import (
    ChoiceBundle,
    ChronicleEntry,
    Continue,
    Leave,
    NpcPlacementBundle,
    Option,
    StateChangeBundle,
    UnlockBeat,
)
from handler.models import Beat, Brief, Ending, Location, When
from handler.world import World, load_world

from evals.evaluators import (
    EndingEarned,
    EndingReached,
    ExploredBeyondStart,
    NoDuplicateCaseFiles,
    NotTruncated,
    coverage_report,
)
from evals.session import SessionArtifact, run_session

FIXTURE_VALID = Path(__file__).parent / "fixtures" / "world_valid"


# ───────────────────────── artifact builder ─────────────────────────


def _artifact(**overrides) -> SessionArtifact:
    base = dict(
        persona="p",
        session_id="abc12345",
        model_engine="gemma-4-e4b",
        model_player="gemini",
        started_at="2026-05-19T00:00:00",
        duration_s=1.0,
        scene_count=2,
        truncated=False,
        ending_id="published_story",
        unlocked_beats=["tom_named_a_vehicle"],
        case_file_entries=["scene one", "scene two"],
        npc_memory={},
        transcript=[],
    )
    base.update(overrides)
    return SessionArtifact(**base)


def _ctx(art: SessionArtifact) -> SimpleNamespace:
    # Evaluators only read ctx.output; duck-type the EvaluatorContext.
    return SimpleNamespace(output=art)


# ───────────────────────── evaluator unit tests ─────────────────────────


def test_ending_reached_pass_and_fail():
    assert EndingReached().evaluate(_ctx(_artifact())) is True
    assert EndingReached().evaluate(
        _ctx(_artifact(ending_id=None, truncated=False))
    ) is False
    # Truncated runs are exempt (NotTruncated reports those instead).
    assert EndingReached().evaluate(
        _ctx(_artifact(ending_id=None, truncated=True))
    ) is True


def test_not_truncated():
    assert NotTruncated().evaluate(_ctx(_artifact(truncated=False))) is True
    assert NotTruncated().evaluate(_ctx(_artifact(truncated=True))) is False


def test_no_duplicate_case_files():
    assert NoDuplicateCaseFiles().evaluate(_ctx(_artifact(case_file_entries=["a", "b"]))) is True
    assert NoDuplicateCaseFiles().evaluate(
        _ctx(_artifact(case_file_entries=["a", "a"]))
    ) is False


def _evidence_world() -> World:
    """Tiny world: an evidence-bearing ending (gates on a decision beat that
    carries requires_any_beats) plus a trivial walk-away ending (decision beat
    with no evidence). Enough for EndingEarned / ExploredBeyondStart."""
    locs = {
        "start": Location(id="start", name="Start", adjacent_to=["village"], description="x"),
        "village": Location(id="village", name="Village", adjacent_to=["start"], description="y"),
    }
    beats = {
        # Observation milestone (LLM-emitted, location-gated).
        "evidence": Beat(id="evidence", name="Evidence", description="x", notes_to_llm="y",
                         commitment=True, requires_location=["village"]),
        # Evidence-bearing decision: gated on the evidence milestone.
        "decided_file": Beat(id="decided_file", name="Decided file", description="x",
                             notes_to_llm="y", commitment=True,
                             when=When(any_evidence_of=["evidence"])),
        # Walk-away gates on engagement, NOT evidence → vacuous for EndingEarned.
        "decided_quit": Beat(id="decided_quit", name="Decided quit", description="x",
                             notes_to_llm="y", commitment=True,
                             when=When(requires_beats=["engaged"])),
        "engaged": Beat(id="engaged", name="Engaged", description="x", notes_to_llm="y",
                        commitment=True, requires_location=["village"]),
    }
    endings = {
        "published": Ending(id="published", requires_beats=["decided_file"],
                            notes_to_llm="y", epilogue="z"),
        "quit": Ending(id="quit", requires_beats=["decided_quit"],
                       notes_to_llm="y", epilogue="z"),
    }
    return World(
        locations=locs, npcs={}, crime="c", beats=beats, endings=endings,
        brief=Brief(player_starts_at="start", opening_seed="s"),
        npcs_by_location={"start": [], "village": []}, few_shots=[],
        map_coords={}, glosses=[], beat_name_to_id={b.name: b.id for b in beats.values()},
    )


def test_ending_earned_flags_unearned_ending():
    """An evidence-bearing ending reached with NO evidence unlocked fails
    (the prologue-collapse regression); with evidence it passes; a walk-away
    ending passes vacuously."""
    world = _evidence_world()
    earned = EndingEarned(world)
    # decision beat fired but its evidence never unlocked → unearned → fail
    assert earned.evaluate(_ctx(_artifact(
        ending_id="published", unlocked_beats=["decided_file"]))) is False
    # evidence present → earned → pass
    assert earned.evaluate(_ctx(_artifact(
        ending_id="published", unlocked_beats=["decided_file", "evidence"]))) is True
    # walk-away ending (no evidence requirement) → passes vacuously
    assert earned.evaluate(_ctx(_artifact(
        ending_id="quit", unlocked_beats=["decided_quit"]))) is True
    # truncated runs exempt
    assert earned.evaluate(_ctx(_artifact(
        ending_id=None, truncated=True))) is True


def test_explored_beyond_start_flags_prologue_only_session():
    """ANY committed ending without leaving the start location fails —
    including walk-away, which now gates on village engagement. Visiting the
    village passes; truncated sessions are exempt."""
    world = _evidence_world()
    explored = ExploredBeyondStart(world)
    confined = [{"kind": "turn", "location": "start"}]
    walked = [
        {"kind": "turn", "location": "start"},
        {"kind": "walk", "from": "start", "to": "village"},
        {"kind": "turn", "location": "village"},
    ]
    assert explored.evaluate(_ctx(_artifact(
        ending_id="published", transcript=confined))) is False
    assert explored.evaluate(_ctx(_artifact(
        ending_id="published", transcript=walked))) is True
    # Walk-away confined to the prologue is now a regression too (it can only
    # be earned after engaging the village).
    assert explored.evaluate(_ctx(_artifact(
        ending_id="quit", transcript=confined))) is False
    assert explored.evaluate(_ctx(_artifact(
        ending_id="quit", transcript=walked))) is True
    # truncated sessions are exempt
    assert explored.evaluate(_ctx(_artifact(
        ending_id=None, truncated=True, transcript=confined))) is True


# ───────────────────────── coverage_report ─────────────────────────


def test_coverage_report():
    world = load_world(FIXTURE_VALID)
    arts = [
        _artifact(unlocked_beats=["tom_named_a_vehicle"], ending_id="published_story"),
        _artifact(unlocked_beats=[], ending_id=None),
    ]
    cov = coverage_report(arts, world)
    assert cov["sessions"] == 2
    assert cov["beats_total"] == len(world.beats)
    assert cov["endings_total"] == len(world.endings)
    assert "published_story" in cov["endings_reached"]
    # A fired beat must not be listed as never-fired.
    assert "tom_named_a_vehicle" not in cov["beats_never_fired"]


# ───────────────────────── run_session smoke ─────────────────────────


class _StubPlayer:
    async def pick_option(self, prose_part, *, can_leave: bool = False) -> int:
        return 0

    async def pick_free_text(self, req) -> str:
        return "I look around."

    async def pick_walk(self, state, world) -> str:
        return world.locations[state.player_location].adjacent_to[0]


def _patch_engine_for_endgame(monkeypatch, decision_beats):
    async def _placements(*a, **k):
        return NpcPlacementBundle()

    async def _phase1(*a, **k):
        return StateChangeBundle(
            commitments=[UnlockBeat(beat_id=b) for b in decision_beats],
        )

    async def _prose(*a, **k):
        yield "The pub falls quiet."

    async def _choices(*a, **k):
        return ChoiceBundle(next=Leave())

    async def _gate(*a, **k):
        return False

    async def _chronicle(*a, **k):
        return ChronicleEntry(case_file_entry="entry", npc_memory_updates=[])

    monkeypatch.setattr("handler.scene_loop.emit_placements", _placements)
    monkeypatch.setattr("handler.scene_loop.emit_state_changes", _phase1)
    monkeypatch.setattr("handler.scene_loop.stream_prose_text", _prose)
    monkeypatch.setattr("handler.scene_loop.emit_choices", _choices)
    monkeypatch.setattr("handler.scene_loop.emit_free_text_gate", _gate)
    monkeypatch.setattr("handler.scene_loop.emit_chronicle_entry", _chronicle)


async def test_run_session_smoke_produces_artifact(monkeypatch, tmp_path):
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    decision_beats = world.endings["published_story"].requires_beats
    _patch_engine_for_endgame(monkeypatch, decision_beats)

    artifact = await run_session(
        bundle, world,
        persona="diligent_journalist",
        persona_prompt="be thorough",
        max_scenes=30,
        out_dir=tmp_path,
        player=_StubPlayer(),
    )

    assert artifact.ending_id == "published_story"
    assert artifact.truncated is False
    assert artifact.scene_count == 1
    assert artifact.transcript  # at least the scene_close entry
    # Dumped + round-trips.
    path = tmp_path / f"diligent_journalist-{artifact.session_id}.json"
    assert path.exists()
    assert SessionArtifact.load(path).ending_id == "published_story"


async def test_run_session_records_per_turn_beats(monkeypatch, tmp_path):
    """A Continue turn records `applied_beats` (beats that fired on the prose the
    player saw) and `fireable` (the set eligible that turn) — the per-transition
    signal the single-transition judges consume."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    decision_beats = world.endings["published_story"].requires_beats
    _patch_engine_for_endgame(monkeypatch, decision_beats)

    # Override choices: a Continue first (so a real turn entry is recorded),
    # then Leave (so the satisfied ending closes the game on turn 2).
    calls = {"n": 0}

    async def _choices(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return ChoiceBundle(next=Continue(options=[Option(text="A"), Option(text="B")]))
        return ChoiceBundle(next=Leave())

    monkeypatch.setattr("handler.scene_loop.emit_choices", _choices)

    artifact = await run_session(
        bundle, world,
        persona="diligent_journalist", persona_prompt="be thorough",
        max_scenes=30, out_dir=tmp_path, player=_StubPlayer(),
    )

    turns = [e for e in artifact.transcript if e["kind"] == "turn"]
    assert turns, "expected at least one Continue turn entry"
    first = turns[0]
    assert isinstance(first["applied_beats"], list)
    assert isinstance(first["fireable"], list)
    # phase1 fired the decision beats on turn 1, so they show as applied.
    assert set(decision_beats).issubset(set(first["applied_beats"]))


async def test_run_session_truncates_when_budget_hit(monkeypatch, tmp_path):
    """No ending ever fires + low scene budget → truncated, no crash."""
    world = load_world(FIXTURE_VALID)
    bundle = build_scene_runner_bundle(TestModel(), world)
    _patch_engine_for_endgame(monkeypatch, decision_beats=[])  # never satisfies an ending

    artifact = await run_session(
        bundle, world,
        persona="distracted_skimmer",
        persona_prompt="be lazy",
        max_scenes=2,
        out_dir=tmp_path,
        player=_StubPlayer(),
    )

    assert artifact.truncated is True
    assert artifact.ending_id is None
    assert artifact.scene_count == 2


# ───────────────────────── corpus loading + retry wrapper ─────────────────────────


def test_load_run_skips_sidecar_json(tmp_path):
    """load_run (used by run.py for the corpus) must skip the report/judge
    sidecars; a raw glob would feed summary.json / judges.json to
    SessionArtifact.load and crash on the non-artifact payload."""
    import json
    from evals.session import load_run

    _artifact(persona="p", session_id="abc").dump(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"not": "an artifact"}), encoding="utf-8")
    (tmp_path / "judges.json").write_text(json.dumps([{"also": "not"}]), encoding="utf-8")

    arts = load_run(tmp_path)
    assert len(arts) == 1
    assert arts[0].session_id == "abc"


def test_retry_model_rejects_nonpositive_max_attempts():
    """max_attempts < 1 would make request() fall through to an implicit
    return None, which pydantic-ai then crashes on far from the cause — reject
    it at construction instead."""
    import pytest
    from evals.retry_model import RetryModel

    with pytest.raises(ValueError, match="max_attempts"):
        RetryModel(TestModel(), max_attempts=0)
