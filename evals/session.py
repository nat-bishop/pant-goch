"""One simulated playthrough → one `SessionArtifact`.

`run_session` drives the existing `handler.scene_loop.play_session` with a
player's three callbacks, recording a transcript along the way, and bounds
runaway/never-ending sessions with a scene budget enforced at walk time (the
budget can only trip between scenes, so a legitimate EndGame is never marked
truncated). The resulting artifact is self-describing JSON — it bakes the
engine-derived values the invariant evaluators need, so the evaluators stay
world-free and trivially serializable.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from handler.agents import (
    AgentBundle,
    make_scene_runner_runtime,
)
from handler.schemas import (
    ChronicleEntry,
    Continue,
    EndGame,
    FreeText,
    ProsePart,
)
from handler.engine import emittable_beat_ids
from handler.scene_loop import (
    PickOption,
    PlayerAction,
    RunConfig,
    TypeText,
    WalkTo,
    play_session,
)
from handler.state import State, make_initial_state
from handler.world import World


# Per-session wall-clock ceiling (eval-only safety net; see run_session). This
# is a pure hang-backstop, NOT a session-length gate: session length is bounded
# structurally by the per-scene turn cap (scene_loop.MAX_TURNS_PER_SCENE) × the
# scene budget (max_scenes), so a healthy playthrough finishes well under this
# even under concurrent load (deepest observed ~890s at ~16s/turn). It only trips
# if the engine genuinely wedges — at which point we drop to a truncated artifact
# rather than letting one stalled session block the whole gather. Raised from
# 900s after a corpus where deep, fully-healthy exploratory sessions (50+
# progressing turns) were racing — and losing to — a too-tight 900s wall.
_SESSION_TIMEOUT_S = 2700.0


class SessionBudgetExceeded(Exception):
    """Raised between scenes once the scene budget is hit, to bound a session
    whose player never commits to an ending."""


# Faithfulness: in the real UI a scene is left exactly two ways — the player
# picks an option the author marked `leave: true`, or the narrator soft-closes
# the scene (`Leave` → EndScene → map). There is no always-available exit. The
# harness mirrors that: it honors an authored `leave` pick as a walk and follows
# every soft-close, but never injects its own leave on an ordinary turn. A scene
# that has stopped paying off is bounded instead by the engine's soft-close
# machinery (scene_is_spent / MAX_TURNS_PER_SCENE / the forced authored exit).


class Player(Protocol):
    async def pick_option(self, prose_part: ProsePart) -> int: ...
    async def pick_free_text(self, req: FreeText) -> str: ...
    async def pick_walk(self, state: State, world: World) -> str: ...


@dataclass
class SessionArtifact:
    persona: str
    session_id: str
    model_engine: str
    model_player: str
    started_at: str
    duration_s: float
    scene_count: int
    truncated: bool
    ending_id: str | None
    unlocked_beats: list[str]
    case_file_entries: list[str]
    npc_memory: dict[str, list[str]]
    transcript: list[dict]
    # A/B arm label ("control" / "test"). None for ordinary single-arm
    # playtests. Defaulted so artifacts written before this field still load
    # via `cls(**json)`.
    arm: str | None = None

    @classmethod
    def capture(
        cls,
        state: State,
        *,
        persona: str,
        session_id: str,
        model_engine: str,
        model_player: str,
        started_at: str,
        duration_s: float,
        scene_count: int,
        truncated: bool,
        transcript: list[dict],
        arm: str | None = None,
    ) -> SessionArtifact:
        return cls(
            persona=persona,
            session_id=session_id,
            model_engine=model_engine,
            model_player=model_player,
            started_at=started_at,
            duration_s=round(duration_s, 2),
            scene_count=scene_count,
            truncated=truncated,
            ending_id=state.game_over.ending_id if state.game_over else None,
            unlocked_beats=sorted(state.unlocked_beats),
            case_file_entries=list(state.case_file_entries),
            npc_memory={k: list(v) for k, v in state.npc_memory.items() if v},
            transcript=transcript,
            arm=arm,
        )

    def dump(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.persona}-{self.session_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> SessionArtifact:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# Sidecar files a run dir accumulates that are NOT session artifacts. The report
# writes summary.json; a judge run writes judges.md (and older runs left a
# judges.json list). Globbing artifacts must skip these or `SessionArtifact.load`
# chokes on a non-mapping payload.
_NON_ARTIFACT_JSON = frozenset({"summary.json", "judges.json"})


def artifact_paths(run_dir: Path) -> list[Path]:
    """Session-artifact JSON files in a run dir, sorted, excluding sidecars."""
    return [
        p for p in sorted(Path(run_dir).glob("*.json"))
        if p.name not in _NON_ARTIFACT_JSON
    ]


def load_run(run_dir: Path) -> list[SessionArtifact]:
    """Load every session artifact in a run dir (skipping summary/judges sidecars)."""
    return [SessionArtifact.load(p) for p in artifact_paths(run_dir)]


@dataclass
class _Recorder:
    transcript: list[dict] = field(default_factory=list)
    scene_count: int = 0


async def run_session(
    bundle: AgentBundle,
    world: World,
    *,
    persona: str,
    persona_prompt: str,
    max_scenes: int,
    out_dir: Path,
    model_engine: str = "gemma-4-e4b",
    model_player: str = "gemini-3.1-flash-lite",
    player_api_key: str | None = None,
    player: Player | None = None,
    run_config: RunConfig = RunConfig(),
    arm: str | None = None,
) -> SessionArtifact:
    """Play one full session and return (and dump) its artifact.

    `player` is injectable: real runs leave it None and a `GeminiPlayer` is
    built from `persona_prompt`; tests pass a stub so no cloud call is made.

    `run_config` selects per-arm engine behaviour for A/B diffing (currently
    `RunConfig` is empty — no knob is defined); `arm` tags the artifact. Both
    default to shipped/single-arm behaviour.
    """
    if player is None:
        from .player import GeminiPlayer

        player = GeminiPlayer(
            persona_prompt, model_name=model_player, api_key=player_api_key,
        )

    state = make_initial_state(world)
    runtime = make_scene_runner_runtime()
    rec = _Recorder()

    session_id = uuid.uuid4().hex[:8]
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.monotonic()
    truncated = False

    async def walk_out(s: State, w: World, *, mid_scene: bool) -> WalkTo:
        """Player leaves via the map. The scene budget trips here (only ever on
        a scene transition, never on an EndGame), so a real ending is never
        falsely truncated."""
        if rec.scene_count >= max_scenes:
            raise SessionBudgetExceeded()
        dest = await player.pick_walk(s, w)
        entry = {"kind": "walk", "from": s.player_location, "to": dest}
        if mid_scene:
            entry["mid_scene"] = True
        rec.transcript.append(entry)
        return WalkTo(location_id=dest)

    async def request_player_input(
        prose_part: ProsePart, s: State, w: World
    ) -> PlayerAction:
        # Map the awaited input back onto the Player protocol's three methods.
        # The simulated player acts inside the scene (pick / free-text). It can
        # leave only the two ways a real player can: by picking an option the
        # author marked `leave: true` (routed to a walk below), or by following a
        # narrator soft-close. There is no always-on mid-scene exit.
        #
        # Per-turn eval signal: run_turn appended this turn's record (with its
        # applied beats) BEFORE this callback was awaited, so dialogue_history[-1]
        # is the prose the player now sees. `fireable` is the set that was
        # eligible to fire — together they let the single-transition judges ask
        # "did the beat that fired deserve to, given what was on the table?".
        applied_beats = (
            [c.beat_id for c in state.dialogue_history[-1].applied_state_changes]
            if state.dialogue_history else []
        )
        # Quality deltas the reporter emitted this turn — the "did this turn
        # move standing?" signal the redefined filler metric reads.
        applied_deltas = (
            [d.model_dump() for d in state.dialogue_history[-1].applied_deltas]
            if state.dialogue_history else []
        )
        fireable = sorted(emittable_beat_ids(w, s))
        nxt = prose_part.next
        if isinstance(nxt, Continue):
            options = nxt.options
            idx = min(await player.pick_option(prose_part), len(options) - 1)
            chosen = options[idx]
            rec.transcript.append({
                "kind": "turn",
                "scene_id": state.scene_id,
                "location": state.player_location,
                "prose": prose_part.prose,
                "options": [o.text for o in options],
                "pick_index": idx,
                "pick_text": chosen.text,
                "leave_pick": chosen.leave,
                "applied_beats": applied_beats,
                "applied_deltas": applied_deltas,
                "fireable": fireable,
            })
            if chosen.leave:
                # Player picked the author's diegetic exit — end the scene like a
                # walk (resolves any met ending / surfaces the map), not a Continue.
                return await walk_out(s, w, mid_scene=True)
            return PickOption(index=idx)
        if isinstance(nxt, FreeText):
            typed = await player.pick_free_text(nxt)
            rec.transcript.append({
                "kind": "free_text",
                "scene_id": state.scene_id,
                "location": state.player_location,
                "prose": prose_part.prose,
                "prompt": nxt.prompt,
                "typed": typed,
                "applied_beats": applied_beats,
                "applied_deltas": applied_deltas,
                "fireable": fireable,
            })
            return TypeText(text=typed)
        # Soft-close (EndScene): the scene is spent — the player walks.
        return await walk_out(s, w, mid_scene=False)

    def on_scene_close(terminal: ProsePart) -> None:
        rec.transcript.append({
            "kind": "scene_close",
            "scene_id": state.scene_id,
            "location": state.player_location,
            "prose": terminal.prose,
            "end": "end_game" if isinstance(terminal.next, EndGame) else "end_scene",
        })

    def on_chronicle_entry(_state: State, _entry: ChronicleEntry) -> None:
        rec.scene_count += 1

    try:
        # Wall-clock safety net (eval-only) — a hang-backstop, not a length gate.
        # Session length is bounded by the per-scene turn cap × max_scenes, so a
        # healthy playthrough finishes well under _SESSION_TIMEOUT_S even under
        # load (~16s/turn). This guards the one case the structural bounds can't:
        # if the engine genuinely wedges (a model call that never returns), run.py
        # awaits every session in one gather, so without this one stalled session
        # would block the whole run — and its summary.json — indefinitely. On a
        # true wedge we drop to a truncated artifact instead of hanging the corpus.
        # Not a scoring change — a wedged session has no ending anyway.
        await asyncio.wait_for(
            play_session(
                bundle, runtime, state, world,
                request_player_input=request_player_input,
                on_scene_close=on_scene_close,
                on_chronicle_entry=on_chronicle_entry,
                run_config=run_config,
            ),
            timeout=_SESSION_TIMEOUT_S,
        )
    except SessionBudgetExceeded:
        truncated = True
    except (asyncio.TimeoutError, TimeoutError):
        truncated = True

    artifact = SessionArtifact.capture(
        state,
        persona=persona,
        session_id=session_id,
        model_engine=model_engine,
        model_player=model_player,
        started_at=started_at,
        duration_s=time.monotonic() - t0,
        scene_count=rec.scene_count,
        truncated=truncated,
        transcript=rec.transcript,
        arm=arm,
    )
    artifact.dump(out_dir)
    return artifact
