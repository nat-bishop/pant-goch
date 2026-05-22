"""Transition + session views, and the datasets the judges run over.

A `Transition` is one before→after step (the player's action this turn + the
prose + which beats fired + which were eligible). Single-transition judges score
these — the smallest scope at which engine correctness is decidable, and the unit
a frozen regression case is built from.

A `SessionView` is one whole playthrough rendered as session-as-text (case file +
location path + beats-fired log + ending). Whole-session judges score these.

`judge_run` builds the per-scope item lists from these and wraps each scope in a
pydantic-evals `Dataset` (over an identity task) so the judges run as `Evaluator`s
and report into the same Logfire experiment surface as the deterministic invariants.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .session import SessionArtifact, artifact_paths


class Transition(BaseModel):
    session_id: str
    persona: str
    scene_id: int
    location: str
    prior_case_file: list[str]
    player_action: str
    prose: str
    options: list[str]
    fired_beats: list[str]
    fireable_beats: list[str]
    is_free_text: bool = False
    beat_of_interest: str | None = None


class SessionView(BaseModel):
    session_id: str
    persona: str
    ending_id: str | None
    truncated: bool
    location_path: list[str]
    beats_in_order: list[str]
    case_file: list[str]


def extract_transitions(artifact: SessionArtifact) -> list[Transition]:
    """One Transition per player turn that carries the Phase-A per-turn signal.
    Legacy turns (no `applied_beats` key) are skipped.

    `prior_case_file` approximates per-turn case-file state by the number of
    scenes closed before this turn (entries append one-per-closed-scene in order).
    """
    out: list[Transition] = []
    scenes_closed = 0
    for e in artifact.transcript:
        kind = e.get("kind")
        if kind == "scene_close":
            scenes_closed += 1
            continue
        if kind not in ("turn", "free_text") or "applied_beats" not in e:
            continue
        action = e["typed"] if kind == "free_text" else e.get("pick_text", "")
        out.append(Transition(
            session_id=artifact.session_id,
            persona=artifact.persona,
            scene_id=e["scene_id"],
            location=e["location"],
            prior_case_file=list(artifact.case_file_entries[:scenes_closed]),
            player_action=action,
            prose=e.get("prose", ""),
            options=e.get("options", []),
            fired_beats=e["applied_beats"],
            fireable_beats=e.get("fireable", []),
            is_free_text=(kind == "free_text"),
        ))
    return out


def transitions_from_dir(run_dir: Path) -> list[Transition]:
    out: list[Transition] = []
    for p in artifact_paths(run_dir):
        out.extend(extract_transitions(SessionArtifact.load(p)))
    return out


def session_view(artifact: SessionArtifact) -> SessionView:
    path: list[str] = []
    beats: list[str] = []
    for e in artifact.transcript:
        if e.get("kind") in ("turn", "free_text"):
            if not path or path[-1] != e["location"]:
                path.append(e["location"])
            beats.extend(e.get("applied_beats", []))
        elif e.get("kind") == "walk":
            if not path or path[-1] != e["to"]:
                path.append(e["to"])
    return SessionView(
        session_id=artifact.session_id,
        persona=artifact.persona,
        ending_id=artifact.ending_id,
        truncated=artifact.truncated,
        location_path=path,
        beats_in_order=beats or sorted(artifact.unlocked_beats),
        case_file=list(artifact.case_file_entries),
    )


def sessions_from_dir(run_dir: Path) -> list[SessionView]:
    return [session_view(SessionArtifact.load(p)) for p in artifact_paths(run_dir)]
