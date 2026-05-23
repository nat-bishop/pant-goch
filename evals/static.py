"""Static topology / reachability lint — the no-engine, no-LLM eval lane.

Pure graph analysis over the hand-authored world (`map`/`locations` adjacency +
`cast` placement + `truth` beat gates). It answers the question the corpus
playtest could only answer expensively and noisily: *is the content even
shaped so the player can reach it?* — and catches the class of problem that
motivated this whole branch (the pub being a cut-vertex that gates the entire
back half of the map) at authoring time, in milliseconds, instead of via a
4-session playtest.

Checks:
  - Reachability: BFS hops from the start location to every location; flag any
    location unreachable from the start (a hard authoring error).
  - Cut-vertices (chokepoints): a location whose removal disconnects the start
    from some load-bearing location (an NPC's spot or a beat's required
    location). Reported with *what it gates* — this is the pub-chokepoint check.
  - Beat / ending reach: where each beat can first fire and how many hops that
    is from the start; beats sitting behind a cut-vertex or deep in the map are
    flagged (they will under-fire in play, as we saw).

`main()` prints a report and, with `gate=True`, exits non-zero only on a hard
error (a location unreachable from the start) — chokepoints/depth are design
signals, not failures, so they never gate.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional

from handler.config import Settings
from handler.world import World, load_world

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _adjacency(world: World) -> dict[str, set[str]]:
    return {lid: set(loc.adjacent_to) for lid, loc in world.locations.items()}


def _bfs_dist(adj: dict[str, set[str]], start: str, *, blocked: str | None = None) -> dict[str, int]:
    """Hops from `start` to each reachable node, optionally with `blocked`
    removed from the graph. Unreachable nodes are absent from the result."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, ()):  # noqa: SIM118
            if nxt == blocked or nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def beat_fire_locations(world: World, bid: str) -> set[str]:
    """The location(s) where beat `bid` can first fire, or empty for a beat with
    no location anchor (the `decided_*` commitments fire wherever the player
    declares). Location-gated beats use `requires_location`; trust beats fire
    where the NPC opens up (`opens_up_only_at` if authored, else their default
    location)."""
    beat = world.beats[bid]
    if beat.requires_location:
        return set(beat.requires_location)
    when = beat.when
    if when is not None and when.npc is not None:
        npc = world.npcs.get(when.npc)
        if npc is None:
            return set()
        disp = getattr(npc, "disposition", None)
        if disp is not None and disp.opens_up_only_at:
            return set(disp.opens_up_only_at)
        return {npc.default_location}
    return set()


def _load_bearing_locations(world: World) -> dict[str, str]:
    """Locations that must stay reachable, mapped to a one-word reason (for the
    cut-vertex 'what it gates' report): every NPC's spot and every beat's fire
    location."""
    out: dict[str, str] = {}
    for nid, npc in world.npcs.items():
        out.setdefault(npc.default_location, f"npc:{nid}")
        disp = getattr(npc, "disposition", None)
        for loc in (disp.opens_up_only_at if disp else []):
            out.setdefault(loc, f"npc:{nid}")
    for bid in world.beats:
        for loc in beat_fire_locations(world, bid):
            out.setdefault(loc, f"beat:{bid}")
    return out


def analyze(world: World) -> dict:
    adj = _adjacency(world)
    start = world.brief.player_starts_at
    dist = _bfs_dist(adj, start)

    all_locs = set(world.locations)
    unreachable = sorted(all_locs - set(dist))

    key_locs = _load_bearing_locations(world)

    # Cut-vertices: remove each non-start node; any load-bearing location that
    # becomes unreachable is "gated behind" it.
    chokepoints: dict[str, list[str]] = {}
    for v in all_locs:
        if v == start:
            continue
        reachable = set(_bfs_dist(adj, start, blocked=v))
        gated = sorted(
            f"{loc} ({key_locs[loc]})"
            for loc in key_locs
            if loc != v and loc not in reachable and loc in dist  # was reachable before
        )
        if gated:
            chokepoints[v] = gated

    # Per-beat reach.
    beat_reach: list[dict] = []
    for bid in world.beats:
        locs = beat_fire_locations(world, bid)
        if not locs:
            beat_reach.append({"beat": bid, "anchor": None, "hops": None,
                               "note": "no location anchor (decision/commitment)"})
            continue
        reachable_locs = {loc: dist[loc] for loc in locs if loc in dist}
        if not reachable_locs:
            beat_reach.append({"beat": bid, "anchor": sorted(locs), "hops": None,
                               "note": "UNREACHABLE from start"})
            continue
        best_loc = min(reachable_locs, key=reachable_locs.get)
        behind = sorted(cv for cv, gated in chokepoints.items()
                        if any(g.startswith(best_loc + " ") for g in gated))
        beat_reach.append({
            "beat": bid, "anchor": best_loc, "hops": reachable_locs[best_loc],
            "note": f"behind chokepoint {behind}" if behind else "",
        })

    return {
        "start": start,
        "distances": dict(sorted(dist.items(), key=lambda kv: (kv[1], kv[0]))),
        "unreachable_locations": unreachable,
        "chokepoints": chokepoints,
        "beat_reach": sorted(beat_reach, key=lambda b: (b["hops"] is None, b["hops"] or 0, b["beat"])),
    }


def render(f: dict) -> str:
    lines = ["## Static topology lint", f"start: {f['start']}", ""]
    lines.append("### Reachability (hops from start)")
    for loc, d in f["distances"].items():
        lines.append(f"  {d}  {loc}")
    if f["unreachable_locations"]:
        lines.append(f"  ⛔ UNREACHABLE: {', '.join(f['unreachable_locations'])}")
    lines.append("")
    lines.append("### Chokepoints (cut-vertices that gate load-bearing locations)")
    if not f["chokepoints"]:
        lines.append("  (none — no single location gates the content)")
    for cv, gated in sorted(f["chokepoints"].items()):
        lines.append(f"  ⚠️  {cv} gates:")
        for g in gated:
            lines.append(f"        {g}")
    lines.append("")
    lines.append("### Beat reach (hops to where each beat can first fire)")
    for b in f["beat_reach"]:
        anchor = b["anchor"] if b["anchor"] else "—"
        hops = "—" if b["hops"] is None else b["hops"]
        flag = f"  {b['note']}" if b["note"] else ""
        lines.append(f"  {hops:>3}  {b['beat']:34} @ {anchor}{flag}")
    return "\n".join(lines)


def main(
    out: Optional[Path] = None,
    gate: bool = False,
    world_path: Optional[Path] = None,
) -> None:
    settings = Settings()
    world = load_world(world_path or settings.world_path or (_REPO_ROOT / "world"))
    findings = analyze(world)
    print(render(findings))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "static.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
        print(f"\nwrote {out / 'static.json'}")
    if gate and findings["unreachable_locations"]:
        raise SystemExit(
            f"static gate: {len(findings['unreachable_locations'])} location(s) "
            f"unreachable from start: {', '.join(findings['unreachable_locations'])}"
        )
