"""World loader: glob YAML files, parse into Pydantic models, enforce the
10 cross-file invariants from SPEC §2.6.

Loader is intentionally short and load-only — no derived state, no caching,
no normalisation passes. The loaded `World` is frozen and read by the
prompt builder + engine.

Layout (per SPEC §2.1):
    world/
    ├── locations/*.yaml      # one file per location
    ├── cast.yaml             # list of NPC records
    ├── truth.yaml            # crime + beats + endings
    └── brief.yaml            # player_starts_at + opening_seed
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import Beat, Brief, Ending, FewShot, Gloss, Location, NPC, Truth


class WorldLoadError(ValueError):
    """World failed to load. Message names the violated invariant + offender."""


@dataclass(frozen=True)
class World:
    """Loaded world bible. Indexed for O(1) lookups by id. Iteration order
    follows the load order from the YAML files (locations sorted by filename;
    npcs/beats/endings as authored in their files)."""

    locations: dict[str, Location]
    npcs: dict[str, NPC]
    crime: str
    beats: dict[str, Beat]
    endings: dict[str, Ending]
    brief: Brief
    npcs_by_location: dict[str, list[NPC]]  # pre-grouped at load time
    few_shots: list[FewShot]  # rendered into Phase 2 system prompt; may be empty
    # Fractional (x, y) ∈ [0, 1] per location id for the clickable map
    # frontend. Optional file (`world/map.yaml`); when missing or empty,
    # the frontend falls back to the text walk panel.
    map_coords: dict[str, tuple[float, float]]
    # Hover-popup entries for the frontend. Optional file (`world/glosses.yaml`);
    # empty list when absent. Frontend-only data — never surfaced to any agent.
    glosses: list[Gloss]
    # Built at load time from `beats`; the inverse of `Beat.name → Beat.id`.
    # Used by agents.emit_state_changes to translate the LLM-facing
    # `beat_name` back to the canonical `beat_id` after grammar-sampled
    # decode. Name uniqueness is a loader invariant, so this is total.
    beat_name_to_id: dict[str, str]

    def beat_id_for_name(self, name: str) -> str:
        """Translate an LLM-emitted beat name back to the canonical id.

        Grammar sampling guarantees the model only emits names present in
        the per-call `Literal[*reachable_beat_names]`, so a miss here is
        engine drift, not a model fault — let the KeyError surface.
        """
        return self.beat_name_to_id[name]

    def npcs_at(self, location_id: str, state: object | None = None) -> list[NPC]:
        """All NPCs at `location_id` right now.

        When `state` is provided, reads `state.npc_locations[npc.id]` as the
        override (the populator agent writes to this map at scene entry).
        NPCs missing from the override map fall back to their
        `default_location`. When `state` is None, returns the default-only
        view (used by tests and one-off renderers that don't have state).

        Type-checking note: `state` is duck-typed (any object with an
        `npc_locations: Mapping[str, str]` attribute works) to avoid a
        circular import with state.py.
        """
        if state is None:
            return self.npcs_by_location.get(location_id, [])
        overrides = getattr(state, "npc_locations", {}) or {}
        return [
            npc
            for npc in self.npcs.values()
            if overrides.get(npc.id, npc.default_location) == location_id
        ]


# ───────────────────────── YAML helpers ─────────────────────────


def _load_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise WorldLoadError(f"YAML parse error in {path}: {e}") from e


def _parse_model(model: type, data: Any, source: Path) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise WorldLoadError(f"{source.name}: {e}") from e


# ───────────────────────── invariant validators ─────────────────────────


def _check_unique_ids(items: list, label: str, source: str) -> None:
    """Invariants 1, 3, 5, 6 — id uniqueness within a category."""
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise WorldLoadError(
                f"{source}: duplicate {label} id {item.id!r}"
            )
        seen.add(item.id)


def _check_symmetric_adjacency(locations: dict[str, Location]) -> None:
    """Invariant 2 — if A's adjacent_to contains B, B's must contain A."""
    for a in locations.values():
        for b_id in a.adjacent_to:
            if b_id not in locations:
                raise WorldLoadError(
                    f"location {a.id!r}: adjacent_to references undefined "
                    f"location {b_id!r}"
                )
            b = locations[b_id]
            if a.id not in b.adjacent_to:
                raise WorldLoadError(
                    f"asymmetric adjacency: {a.id!r}.adjacent_to includes "
                    f"{b_id!r}, but {b_id!r}.adjacent_to does not include {a.id!r}"
                )


def _check_no_dead_ends(locations: dict[str, Location]) -> None:
    """Invariant 11 — every location has ≥1 adjacent_to. Under
    player-authored movement (the map is the only leave control and the
    LLM soft-close removes all in-scene options), a location with no
    adjacents is an unrecoverable soft-lock: the player can neither stay
    nor go. Hard-fail at load."""
    for loc in locations.values():
        if not loc.adjacent_to:
            raise WorldLoadError(
                f"location {loc.id!r} has no adjacent_to — dead-end "
                f"locations are unreachable-from under player-authored "
                f"movement (the map is the only exit)"
            )


def _check_npc_default_locations(npcs: dict[str, NPC], locations: dict[str, Location]) -> None:
    """Invariant 4 — every NPC's default_location references a defined location."""
    for npc in npcs.values():
        if npc.default_location not in locations:
            raise WorldLoadError(
                f"npc {npc.id!r}: default_location {npc.default_location!r} "
                f"is not a defined location"
            )


def _check_beat_ending_namespace(
    beats: dict[str, Beat], endings: dict[str, Ending]
) -> None:
    """Invariant 6 (cross-cutting) — beat ids and ending ids share a single
    namespace; collisions break the membership semantics of `requires_beats`."""
    overlap = set(beats.keys()) & set(endings.keys())
    if overlap:
        raise WorldLoadError(
            f"beat/ending id collision: {sorted(overlap)} appear as both"
        )


def _build_beat_name_to_id(beats: dict[str, Beat]) -> dict[str, str]:
    """Build the name→id map AND enforce name uniqueness in one pass.
    Beat names are the LLM-facing surface (Phase 1 emits the name, not
    the id); the map is what the engine uses to translate names back to
    canonical ids at the decode boundary. Uniqueness is the invariant
    that makes that translation deterministic."""
    by_name: dict[str, str] = {}
    for beat in beats.values():
        if beat.name in by_name:
            raise WorldLoadError(
                f"duplicate Beat.name {beat.name!r}: used by both "
                f"{by_name[beat.name]!r} and {beat.id!r}"
            )
        by_name[beat.name] = beat.id
    return by_name


def _check_beat_refs(beats: dict[str, Beat]) -> None:
    """Invariant 8 — every requires_beats / requires_any_beats / blocked_by_beats
    reference resolves to a defined beat id; no self-reference."""
    for beat in beats.values():
        for field_name in ("requires_beats", "requires_any_beats", "blocked_by_beats"):
            for ref in getattr(beat, field_name):
                if ref == beat.id:
                    raise WorldLoadError(
                        f"beat {beat.id!r}: {field_name} contains itself"
                    )
                if ref not in beats:
                    raise WorldLoadError(
                        f"beat {beat.id!r}: {field_name} references undefined "
                        f"beat {ref!r}"
                    )


def _check_beat_npc_refs(beats: dict[str, Beat], npcs: dict[str, NPC]) -> None:
    """Invariant 12 — every requires_npcs_present reference resolves to a
    defined NPC id. Gates the presence-aware fireability narrowing in
    engine.fireable_beat_ids."""
    for beat in beats.values():
        for ref in beat.requires_npcs_present:
            if ref not in npcs:
                raise WorldLoadError(
                    f"beat {beat.id!r}: requires_npcs_present references "
                    f"undefined npc {ref!r}"
                )


def _check_beat_location_refs(
    beats: dict[str, Beat], locations: dict[str, Location]
) -> None:
    """Invariant 13 — every requires_location reference resolves to a defined
    location. Gates the place-aware fireability narrowing in
    engine.fireable_beat_ids."""
    for beat in beats.values():
        for ref in beat.requires_location:
            if ref not in locations:
                raise WorldLoadError(
                    f"beat {beat.id!r}: requires_location references "
                    f"undefined location {ref!r}"
                )


def _check_beats_gated(beats: dict[str, Beat]) -> None:
    """Invariant 14 — every beat carries ≥1 hard gate (requires_beats,
    requires_any_beats, requires_npcs_present, or requires_location) OR is
    explicitly marked `always_available`. An ungated, unmarked beat is
    globally fireable from turn 1 regardless of place or progress — the
    failure mode that let `examined_the_crash_site` / `village_runs_you_out`
    fire in the empty prologue. The only legitimate ungated beat is the
    player's always-available "I'm walking away" commitment, which opts in
    via `always_available`."""
    for beat in beats.values():
        has_gate = (
            beat.requires_beats
            or beat.requires_any_beats
            or beat.requires_npcs_present
            or beat.requires_location
        )
        if not has_gate and not beat.always_available:
            raise WorldLoadError(
                f"beat {beat.id!r}: no hard gate (requires_beats / "
                f"requires_any_beats / requires_npcs_present / "
                f"requires_location) and not marked always_available — it "
                f"would be globally fireable from turn 1"
            )


def _check_no_requires_cycles(beats: dict[str, Beat]) -> None:
    """Invariant 9 — the dependency graph (union of requires_beats AND
    requires_any_beats) has no cycles. A cycle anywhere would make a beat
    unfireable. DFS with grey/black colouring; report the cycle path on
    detection."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {bid: WHITE for bid in beats}

    def visit(bid: str, path: list[str]) -> None:
        colour[bid] = GREY
        deps = list(beats[bid].requires_beats) + list(beats[bid].requires_any_beats)
        for dep in deps:
            if colour[dep] == GREY:
                cycle = path[path.index(dep):] + [dep]
                raise WorldLoadError(
                    f"requires cycle: {' -> '.join(cycle)}"
                )
            if colour[dep] == WHITE:
                visit(dep, path + [dep])
        colour[bid] = BLACK

    for bid in beats:
        if colour[bid] == WHITE:
            visit(bid, [bid])


def _check_brief_starts_at(brief: Brief, locations: dict[str, Location]) -> None:
    """Invariant 7 — brief.player_starts_at references a defined location."""
    if brief.player_starts_at not in locations:
        raise WorldLoadError(
            f"brief.yaml: player_starts_at {brief.player_starts_at!r} is "
            f"not a defined location"
        )


def _check_ending_refs(endings: dict[str, Ending], beats: dict[str, Beat]) -> None:
    """Invariant 10 — every ending's requires_beats / requires_any_beats
    references a defined beat id."""
    for ending in endings.values():
        for field_name in ("requires_beats", "requires_any_beats"):
            for ref in getattr(ending, field_name):
                if ref not in beats:
                    raise WorldLoadError(
                        f"ending {ending.id!r}: {field_name} references "
                        f"undefined beat {ref!r}"
                    )


# ───────────────────────── loader ─────────────────────────


def load_world(path: Path) -> World:
    """Load and validate a world bible from `path` (the directory containing
    locations/, cast.yaml, truth.yaml, brief.yaml).

    Raises `WorldLoadError` on the first violation of any of the 10
    invariants (or any per-record shape error). All-or-nothing: a partially
    valid world is not returned.
    """
    locations_dir = path / "locations"
    if not locations_dir.is_dir():
        raise WorldLoadError(f"missing locations directory: {locations_dir}")

    location_files = sorted(locations_dir.glob("*.yaml"))
    if not location_files:
        raise WorldLoadError(f"no location files found in {locations_dir}")
    location_records: list[Location] = [
        _parse_model(Location, _load_yaml(p), p) for p in location_files
    ]
    _check_unique_ids(location_records, "location", str(locations_dir))
    locations: dict[str, Location] = {loc.id: loc for loc in location_records}
    _check_symmetric_adjacency(locations)
    _check_no_dead_ends(locations)

    cast_path = path / "cast.yaml"
    if not cast_path.is_file():
        raise WorldLoadError(f"missing cast.yaml: {cast_path}")
    cast_data = _load_yaml(cast_path)
    if not isinstance(cast_data, list):
        raise WorldLoadError(
            f"cast.yaml: expected a list of NPC records, got {type(cast_data).__name__}"
        )
    npc_records: list[NPC] = [_parse_model(NPC, d, cast_path) for d in cast_data]
    _check_unique_ids(npc_records, "npc", str(cast_path))
    npcs: dict[str, NPC] = {n.id: n for n in npc_records}
    _check_npc_default_locations(npcs, locations)

    truth_path = path / "truth.yaml"
    if not truth_path.is_file():
        raise WorldLoadError(f"missing truth.yaml: {truth_path}")
    truth: Truth = _parse_model(Truth, _load_yaml(truth_path), truth_path)
    _check_unique_ids(truth.beats, "beat", str(truth_path))
    _check_unique_ids(truth.endings, "ending", str(truth_path))
    beats: dict[str, Beat] = {b.id: b for b in truth.beats}
    endings: dict[str, Ending] = {e.id: e for e in truth.endings}
    _check_beat_ending_namespace(beats, endings)
    beat_name_to_id = _build_beat_name_to_id(beats)
    _check_beat_refs(beats)
    _check_beat_npc_refs(beats, npcs)
    _check_beat_location_refs(beats, locations)
    _check_beats_gated(beats)
    _check_no_requires_cycles(beats)
    _check_ending_refs(endings, beats)

    brief_path = path / "brief.yaml"
    if not brief_path.is_file():
        raise WorldLoadError(f"missing brief.yaml: {brief_path}")
    brief: Brief = _parse_model(Brief, _load_yaml(brief_path), brief_path)
    _check_brief_starts_at(brief, locations)

    npcs_by_location: dict[str, list[NPC]] = {loc_id: [] for loc_id in locations}
    for npc in npcs.values():
        npcs_by_location[npc.default_location].append(npc)

    # few_shots.yaml is optional — older fixtures and worlds without authored
    # examples load with an empty list. Phase 2's renderer tolerates [].
    few_shots_path = path / "few_shots.yaml"
    few_shots: list[FewShot] = []
    try:
        few_shots_data = _load_yaml(few_shots_path)
    except FileNotFoundError:
        few_shots_data = None
    if few_shots_data is not None:
        if not isinstance(few_shots_data, list):
            raise WorldLoadError(
                f"few_shots.yaml: expected a list of FewShot records, got "
                f"{type(few_shots_data).__name__}"
            )
        few_shots = [
            _parse_model(FewShot, d, few_shots_path) for d in few_shots_data
        ]

    # glosses.yaml is optional — older fixtures load with []. Loader
    # tolerates missing; frontend renders no hover-popups when empty.
    glosses_path = path / "glosses.yaml"
    glosses: list[Gloss] = []
    try:
        glosses_data = _load_yaml(glosses_path)
    except FileNotFoundError:
        glosses_data = None
    if glosses_data is not None:
        if not isinstance(glosses_data, list):
            raise WorldLoadError(
                f"glosses.yaml: expected a list of Gloss records, got "
                f"{type(glosses_data).__name__}"
            )
        glosses = [_parse_model(Gloss, d, glosses_path) for d in glosses_data]
        _check_unique_ids(glosses, "gloss", str(glosses_path))

    # map.yaml is optional — fixtures without a drawn map load with {} and
    # the frontend falls back to the text walk panel.
    map_path = path / "map.yaml"
    map_coords: dict[str, tuple[float, float]] = {}
    try:
        map_data = _load_yaml(map_path)
    except FileNotFoundError:
        map_data = None
    if map_data is not None:
        if not isinstance(map_data, dict):
            raise WorldLoadError(
                f"map.yaml: expected a mapping of location_id → {{x, y}}, "
                f"got {type(map_data).__name__}"
            )
        for loc_id, entry in map_data.items():
            if loc_id not in locations:
                raise WorldLoadError(
                    f"map.yaml: {loc_id!r} is not a defined location"
                )
            if not isinstance(entry, dict) or "x" not in entry or "y" not in entry:
                raise WorldLoadError(
                    f"map.yaml[{loc_id!r}]: expected {{x: <float>, y: <float>}}"
                )
            x, y = entry["x"], entry["y"]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise WorldLoadError(
                    f"map.yaml[{loc_id!r}]: x={x} y={y} out of [0, 1]"
                )
            map_coords[loc_id] = (float(x), float(y))

    return World(
        locations=locations,
        npcs=npcs,
        crime=truth.crime,
        beats=beats,
        endings=endings,
        brief=brief,
        npcs_by_location=npcs_by_location,
        few_shots=few_shots,
        map_coords=map_coords,
        glosses=glosses,
        beat_name_to_id=beat_name_to_id,
    )
