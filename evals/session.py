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
from handler.engine import fireable_beat_ids
from handler.scene_loop import (
    LEAVE_OPTION_MIN_HISTORY,
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


# Faithfulness: in the real UI the map exit ("Step outside") is live on every
# turn once a scene has a couple of turns behind it, so a player who feels a
# scene has stopped paying off just leaves. The harness mirrors that by offering
# the player a Leave choice on every eligible Continue turn (see
# request_player_input). Previously the player could ONLY leave at a choice-
# author soft-close or on a rare seeded coin flip, so it milled fruitlessly —
# inflating filler_turn_ratio and starving exploration (a single post-office
# scene ran 43 straight filler turns in the 20260520-194311 run). The leave
# affordance unlocks after the same LEAVE_OPTION_MIN_HISTORY the UI uses.


class Player(Protocol):
    async def pick_option(self, prose_part: ProsePart, *, can_leave: bool = False) -> int: ...
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
        # The simulated player acts inside the scene (pick / free-text), and on
        # any Continue turn with a couple of turns behind it may instead elect to
        # Leave via the map — exactly the always-on UI exit — ending the turn
        # with a WalkTo. That is how a real player escapes a scene that has
        # stopped paying off.
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
        fireable = sorted(fireable_beat_ids(w, s))
        nxt = prose_part.next
        if isinstance(nxt, Continue):
            options = nxt.options
            can_leave = len(state.dialogue_history) >= LEAVE_OPTION_MIN_HISTORY
            idx = await player.pick_option(prose_part, can_leave=can_leave)
            if can_leave and idx >= len(options):
                # Player chose the appended Leave entry over any in-scene option.
                return await walk_out(s, w, mid_scene=True)
            idx = min(idx, len(options) - 1)
            rec.transcript.append({
                "kind": "turn",
                "scene_id": state.scene_id,
                "location": state.player_location,
                "prose": prose_part.prose,
                "options": [o.text for o in options],
                "pick_index": idx,
                "pick_text": options[idx].text,
                "applied_beats": applied_beats,
                "fireable": fireable,
            })
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
        await play_session(
            bundle, runtime, state, world,
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
