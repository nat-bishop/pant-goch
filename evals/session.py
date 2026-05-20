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

import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from handler.agents import (
    AgentBundle,
    make_chronicler_runtime,
    make_scene_runner_runtime,
)
from handler.schemas import (
    ChronicleEntry,
    Continue,
    EndGame,
    FreeText,
    ProsePart,
)
from handler.scene_loop import (
    PickOption,
    PlayerAction,
    TypeText,
    WalkTo,
    play_session,
)
from handler.state import State, make_initial_state
from handler.world import World


class SessionBudgetExceeded(Exception):
    """Raised between scenes once the scene budget is hit, to bound a session
    whose player never commits to an ending."""


# Coverage: the simulated player normally leaves ONLY at a soft-close, so the
# engine's mid-scene map walk (a WalkTo that ends a Continue turn — the live
# "Step outside" / map exit in the real UI) would never be exercised by the
# harness. Give it a small, seeded chance to walk out mid-scene once a scene
# has some content behind it, so the eval covers that path too. Seeded per
# session for reproducibility within a run.
MID_SCENE_WALK_P = 0.12
MID_SCENE_WALK_MIN_TURNS = 3


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
        )

    def dump(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.persona}-{self.session_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> SessionArtifact:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


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
) -> SessionArtifact:
    """Play one full session and return (and dump) its artifact.

    `player` is injectable: real runs leave it None and a `GeminiPlayer` is
    built from `persona_prompt`; tests pass a stub so no cloud call is made.
    """
    if player is None:
        from .player import GeminiPlayer

        player = GeminiPlayer(
            persona_prompt, model_name=model_player, api_key=player_api_key,
        )

    state = make_initial_state(world)
    runtime = make_scene_runner_runtime()
    chronicler = make_chronicler_runtime()
    rec = _Recorder()

    session_id = uuid.uuid4().hex[:8]
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.monotonic()
    truncated = False
    walk_rng = random.Random(session_id)

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
        # Map the awaited input back onto the Player protocol's three
        # methods. The simulated player acts inside the scene (pick /
        # free-text) and leaves at a soft-close — plus a small seeded chance of
        # a mid-scene map walk so the eval covers the engine's any-turn WalkTo.
        nxt = prose_part.next
        if isinstance(nxt, Continue):
            if (
                len(state.dialogue_history) >= MID_SCENE_WALK_MIN_TURNS
                and walk_rng.random() < MID_SCENE_WALK_P
            ):
                return await walk_out(s, w, mid_scene=True)
            idx = await player.pick_option(prose_part)
            options = nxt.options
            rec.transcript.append({
                "kind": "turn",
                "scene_id": state.scene_id,
                "location": state.player_location,
                "prose": prose_part.prose,
                "options": [o.text for o in options],
                "pick_index": idx,
                "pick_text": options[idx].text,
            })
            return PickOption(index=idx)
        if isinstance(nxt, FreeText):
            typed = await player.pick_free_text(nxt)
            rec.transcript.append({
                "kind": "free_text",
                "scene_id": state.scene_id,
                "location": state.player_location,
                "prompt": nxt.prompt,
                "typed": typed,
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
        await play_session(
            bundle, runtime, chronicler, state, world,
            request_player_input=request_player_input,
            on_scene_close=on_scene_close,
            on_chronicle_entry=on_chronicle_entry,
        )
    except SessionBudgetExceeded:
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
    )
    artifact.dump(out_dir)
    return artifact
