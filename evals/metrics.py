"""Corpus-level distributional metrics (Tier 1) — pure functions, no LLM.

`corpus_metrics(artifacts, world)` reduces a whole run of `SessionArtifact`s to a
single `CorpusMetrics` (JSON-serializable, stable keys) that answers the question
the per-session invariants cannot: *is the game varied and alive, or collapsing to
one rail?*

This is the layer that would have flashed red on the 20260520-161349 run (all
invariants green, 80% of sessions on one ending, median 2 scenes). The five
north-star fields (`report.NORTH_STAR_KEYS`) sit at the top of the run report;
the rest are drill-down.

Lives separately from the per-session `Evaluator`s in `evaluators.py`: invariants
must *hold* (pass/fail per run); these *trend* (a number you watch build-over-build).
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from pydantic import BaseModel

from handler.engine import confession_depth
from handler.state import make_initial_state
from handler.world import World

from .session import SessionArtifact


def distinct_locations(art: SessionArtifact) -> set[str]:
    """Every location the session touched, read from the transcript (turn /
    free_text / scene_close carry `location`; walk carries `from`/`to`)."""
    locs: set[str] = set()
    for e in art.transcript:
        if "location" in e:
            locs.add(e["location"])
        if e.get("kind") == "walk":
            locs.add(e["from"])
            locs.add(e["to"])
    return locs


def traversed_edges(art: SessionArtifact) -> set[frozenset[str]]:
    """Every map adjacency edge this session actually walked, as undirected
    pairs (read from the transcript's `walk` from/to). The edge analogue of
    `distinct_locations`: node coverage says *where* the player went, edge
    coverage says *which connections* carried them — so a bypass edge that is
    authored but never used shows up here, where node coverage can't see it."""
    edges: set[frozenset[str]] = set()
    for e in art.transcript:
        if e.get("kind") == "walk" and e.get("from") and e.get("to"):
            edges.add(frozenset((e["from"], e["to"])))
    return edges


def _all_edges(world: World) -> set[frozenset[str]]:
    """Undirected adjacency edges of the map (symmetric by the §2.6 invariant,
    so each unordered pair is counted once)."""
    return {
        frozenset((lid, adj))
        for lid, loc in world.locations.items()
        for adj in loc.adjacent_to
    }


def _fmt_edge(edge: frozenset[str]) -> str:
    return " — ".join(sorted(edge))


def _play_turns(art: SessionArtifact) -> list[dict]:
    """The turn / free_text entries (player-facing moments), in order. Walks and
    scene-close lines are excluded — they aren't turns the player acted within."""
    return [e for e in art.transcript if e.get("kind") in ("turn", "free_text")]


def _scene_turn_counts(art: SessionArtifact) -> list[int]:
    """Player turns (turn / free_text) per scene, one count per scene the session
    played — grouped by the `scene_id` each play turn carries. Surfaces milling:
    a scene that keeps offering in-scene options shows up here as a large count.
    With the engine's `MAX_TURNS_PER_SCENE` cap in force, no count should exceed
    it; a `max_turns_per_scene` riding the cap means scenes routinely mill to the
    ceiling rather than ending on their own (a soft-close / leave-pressure gap)."""
    by_scene: Counter[str] = Counter()
    for e in _play_turns(art):
        by_scene[e.get("scene_id", "")] += 1
    return list(by_scene.values())


def _ordered_beats(art: SessionArtifact) -> list[str]:
    """Beats in the order they fired across the session — the per-turn
    `applied_beats` concatenated in transcript order. Empty for legacy artifacts
    written before per-turn capture (Phase A)."""
    out: list[str] = []
    for e in _play_turns(art):
        out.extend(e.get("applied_beats", []))
    return out


def _entropy_bits(counts: list[int]) -> float:
    """Shannon entropy (bits) of a count distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def _ending_entropy_normalized(artifacts: list[SessionArtifact], endings_total: int) -> float:
    """Entropy of the realized-ending distribution, normalized to 0..1 by the
    authored ending count. ~0 = every run lands the same ending (no variety);
    ~1 = endings spread evenly across the catalog. None (no ending) is its own
    bucket so a pile of no-ending runs reads as low variety, not as absent."""
    if endings_total < 2:
        return 0.0
    counts = Counter(art.ending_id or "__none__" for art in artifacts)
    return _entropy_bits(list(counts.values())) / math.log2(endings_total)


def _jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    """JS divergence (bits, 0..1) between two distributions over a shared key set."""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: dict[str, float]) -> float:
        return sum(
            a[k] * math.log2(a[k] / m[k])
            for k in keys
            if a.get(k, 0.0) > 0 and m[k] > 0
        )

    return 0.5 * _kl(p) + 0.5 * _kl(q)


def _persona_outcome_separation(artifacts: list[SessionArtifact]) -> float:
    """Mean pairwise JS divergence between personas' ending distributions. ~0 =
    play style does not change outcomes (the cheap consequence proxy: if the
    aggressive interrogator and the diligent journalist land the same endings,
    early choices don't matter)."""
    by_persona: dict[str, Counter] = {}
    for art in artifacts:
        by_persona.setdefault(art.persona, Counter())[art.ending_id or "__none__"] += 1
    if len(by_persona) < 2:
        return 0.0
    dists: list[dict[str, float]] = []
    for counts in by_persona.values():
        total = sum(counts.values())
        dists.append({k: v / total for k, v in counts.items()})
    pairs = list(combinations(dists, 2))
    return sum(_jensen_shannon(a, b) for a, b in pairs) / len(pairs)


def _mean_pairwise_beat_jaccard_distance(artifacts: list[SessionArtifact]) -> float:
    """Mean `1 - jaccard(beats_a, beats_b)` over session pairs. ~0 = every run
    fires the same beat-set (linear); ~1 = runs diverge in what they uncover."""
    beat_sets = [set(art.unlocked_beats) for art in artifacts]
    pairs = list(combinations(beat_sets, 2))
    if not pairs:
        return 0.0

    def _dist(a: set[str], b: set[str]) -> float:
        union = a | b
        if not union:
            return 0.0
        return 1.0 - len(a & b) / len(union)

    return sum(_dist(a, b) for a, b in pairs) / len(pairs)


def _longest_common_beat_prefix(artifacts: list[SessionArtifact]) -> list[str]:
    """The opening beat-rail shared by a majority (>50%) of sessions: greedily
    extend a prefix while more than half the sessions still agree on the next
    beat. A long prefix means every run opens the same way (linear opening)."""
    lists = [_ordered_beats(art) for art in artifacts]
    lists = [b for b in lists if b]
    if not lists:
        return []
    total = len(lists)
    prefix: list[str] = []
    candidates = lists
    while True:
        i = len(prefix)
        nexts = [b[i] for b in candidates if len(b) > i]
        if not nexts:
            break
        beat, count = Counter(nexts).most_common(1)[0]
        if count <= total / 2:
            break
        prefix.append(beat)
        candidates = [b for b in candidates if len(b) > i and b[i] == beat]
    return prefix


def _filler_turn_ratio(artifacts: list[SessionArtifact]) -> float:
    """Fraction of player turns that moved NO state — neither unlocked a beat
    nor shifted any disposition (`applied_deltas`). The quality-engine agency
    proxy: a turn that nudges trust/guard is meaningful even if it fires no
    milestone, so only a turn that does neither is true filler. Legacy
    artifacts without per-turn capture contribute nothing."""
    filler = 0
    total = 0
    for art in artifacts:
        for e in _play_turns(art):
            if "applied_beats" not in e:
                continue  # legacy artifact: no per-turn signal
            total += 1
            if not e["applied_beats"] and not e.get("applied_deltas"):
                filler += 1
    return filler / total if total else 0.0


def _post_exhaustion_turn_ratio(artifacts: list[SessionArtifact]) -> float:
    """Mean over sessions of the fraction of player turns that fall AFTER the last
    turn to introduce a *new* beat — the trailing run where the player keeps acting
    but the investigation has stopped advancing. This is the grind-after-exhaustion
    that `filler_turn_ratio` blurs: filler counts every beatless turn (including
    legitimate early exploration), whereas this isolates the dead tail (the 40+
    Helen/Owain loop a player ground through after the truth was already in hand).
    ~0 = the session ends promptly once leads run out; high = a long dead tail, or
    (ratio 1.0) a session that never advanced at all. Legacy artifacts without
    per-turn capture contribute nothing."""
    ratios: list[float] = []
    for art in artifacts:
        turns = _play_turns(art)
        if not any("applied_beats" in e for e in turns):
            continue  # legacy artifact: no per-turn signal
        seen: set[str] = set()
        last_productive = -1
        for i, e in enumerate(turns):
            new = [b for b in e.get("applied_beats", []) if b not in seen]
            seen.update(e.get("applied_beats", []))
            if new:
                last_productive = i
        ratios.append((len(turns) - 1 - last_productive) / len(turns))
    return sum(ratios) / len(ratios) if ratios else 0.0


def _beats_per_location(artifacts: list[SessionArtifact]) -> dict[str, int]:
    """Distinct beats that fired at each location across the corpus — surfaces
    locations carrying no investigative weight."""
    by_loc: dict[str, set[str]] = {}
    for art in artifacts:
        for e in _play_turns(art):
            by_loc.setdefault(e["location"], set()).update(e.get("applied_beats", []))
    return {loc: len(beats) for loc, beats in sorted(by_loc.items())}


# ───────────────────────── playability health (W3) ─────────────────────────
#
# The signals that would have flashed red on the noregex-20260524 run, where
# every persona was run out of the village having warmed no one: the mid-game
# (person-anchored findables) and end-game (decisions, real endings) were
# structurally unreachable. `beats_never_fired` lumped them in with everything
# else; these isolate the "is the game actually playable?" question per session.


def _ejection_ending_ids(world: World) -> set[str]:
    """Endings that fire on losing the village — gated on
    `village_attention_at_least: hostile`. The run-out terminal; a corpus that
    only ever reaches these is unplayable, not varied."""
    return {
        eid
        for eid, ending in world.endings.items()
        if ending.when is not None
        and ending.when.village_attention_at_least == "hostile"
    }


def _person_findable_ids(world: World) -> set[str]:
    """Commitment findables anchored to a specific NPC (`when.npc`), excluding
    the player-decision beats — Tom naming the vehicle, Dylan admitting what he
    knows, etc. The mid-game content that only lands once an NPC has thawed."""
    return {
        bid
        for bid, beat in world.beats.items()
        if beat.commitment
        and not bid.startswith("decided_")
        and beat.when is not None
        and beat.when.npc is not None
    }


def _delta_int(s: object) -> int:
    """Parse a wire trust/guard string ("+1"/"-1"/"0") to an int; everything
    else (None, "BREACH") is 0 — BREACH latches separately."""
    try:
        return int(s)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _session_thawed_npcs(art: SessionArtifact, world: World) -> set[str]:
    """The tracked NPCs this session warmed to `thawing`+ (confession depth ≥ 1).
    Reconstructs per-NPC trust/guard by summing the transcript's `applied_deltas`
    — which now record the REALIZED change (post-zero/post-clamp), so the sum
    reproduces final state — then asks the engine for the band. NOTE: artifacts
    written before the effective-delta change logged *emitted* deltas and so
    overcount thaw; this metric is meaningful only on post-change runs."""
    s = make_initial_state(world)
    for e in _play_turns(art):
        for d in e.get("applied_deltas", []):
            nid = d.get("npc_id")
            if nid is None:
                continue
            if d.get("guard") == "BREACH":
                s.guard_breached.add(nid)
            else:
                s.guard[nid] = s.guard.get(nid, 0) + _delta_int(d.get("guard"))
            s.trust[nid] = s.trust.get(nid, 0) + _delta_int(d.get("trust"))
    return {nid for nid in world.npcs if confession_depth(world, s, nid) >= 1}


def _rate(predicate_true: int, total: int) -> float:
    return round(predicate_true / total, 3) if total else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2


class CorpusMetrics(BaseModel):
    """One run's distributional health. Stable, JSON-serializable keys so
    `report.py` can diff two runs field-by-field."""

    sessions: int
    # north-star
    ending_entropy_normalized: float
    beats_fired: int
    filler_turn_ratio: float
    post_exhaustion_turn_ratio: float
    mean_pairwise_beat_jaccard_distance: float
    persona_outcome_separation: float
    # playability health — is the mid/end-game reachable at all?
    ejection_rate: float
    npc_thawed_rate: float
    person_findable_rate: float
    decision_fired_rate: float
    ending_diversity: float
    # coverage
    beats_total: int
    beats_never_fired: list[str]
    endings_total: int
    endings_reached: list[str]
    endings_never_reached: list[str]
    location_coverage: float
    locations_never_visited: list[str]
    edge_coverage: float
    edges_never_traversed: list[str]
    beats_per_location: dict[str, int]
    # depth / linearity
    median_scenes_per_session: float
    median_distinct_locations_per_session: float
    mean_turns_per_scene: float
    max_turns_per_scene: int
    longest_common_beat_prefix: list[str]

    def to_dict(self) -> dict:
        return self.model_dump()


def corpus_metrics(artifacts: list[SessionArtifact], world: World) -> CorpusMetrics:
    """Reduce a run of artifacts to its distributional health."""
    fired_beats: set[str] = set()
    reached_endings: set[str] = set()
    visited: set[str] = set()
    walked: set[frozenset[str]] = set()
    for art in artifacts:
        fired_beats.update(art.unlocked_beats)
        if art.ending_id:
            reached_endings.add(art.ending_id)
        visited |= distinct_locations(art)
        walked |= traversed_edges(art)

    # Flat list of per-scene turn counts across every session — the basis for
    # mean_turns_per_scene / max_turns_per_scene (milling signal).
    scene_turns = [c for art in artifacts for c in _scene_turn_counts(art)]

    all_beats = set(world.beats)
    all_locations = set(world.locations)
    all_edges = _all_edges(world)

    # Playability health: per-session pass rates over the whole corpus.
    n = len(artifacts)
    ejection_ids = _ejection_ending_ids(world)
    person_ids = _person_findable_ids(world)
    ejected = sum(1 for a in artifacts if a.ending_id in ejection_ids)
    thawed = sum(1 for a in artifacts if _session_thawed_npcs(a, world))
    with_person = sum(
        1 for a in artifacts if person_ids & set(a.unlocked_beats)
    )
    with_decision = sum(
        1
        for a in artifacts
        if any(b.startswith("decided_") for b in a.unlocked_beats)
    )

    return CorpusMetrics(
        sessions=len(artifacts),
        ending_entropy_normalized=round(
            _ending_entropy_normalized(artifacts, len(world.endings)), 3
        ),
        beats_fired=len(fired_beats & all_beats),
        filler_turn_ratio=round(_filler_turn_ratio(artifacts), 3),
        post_exhaustion_turn_ratio=round(_post_exhaustion_turn_ratio(artifacts), 3),
        mean_pairwise_beat_jaccard_distance=round(
            _mean_pairwise_beat_jaccard_distance(artifacts), 3
        ),
        persona_outcome_separation=round(_persona_outcome_separation(artifacts), 3),
        ejection_rate=_rate(ejected, n),
        npc_thawed_rate=_rate(thawed, n),
        person_findable_rate=_rate(with_person, n),
        decision_fired_rate=_rate(with_decision, n),
        ending_diversity=_rate(len(reached_endings), n),
        beats_total=len(all_beats),
        beats_never_fired=sorted(all_beats - fired_beats),
        endings_total=len(world.endings),
        endings_reached=sorted(reached_endings),
        endings_never_reached=sorted(set(world.endings) - reached_endings),
        location_coverage=round(len(visited & all_locations) / len(all_locations), 3)
        if all_locations else 0.0,
        locations_never_visited=sorted(all_locations - visited),
        edge_coverage=round(len(walked & all_edges) / len(all_edges), 3)
        if all_edges else 0.0,
        edges_never_traversed=sorted(_fmt_edge(e) for e in (all_edges - walked)),
        beats_per_location=_beats_per_location(artifacts),
        median_scenes_per_session=_median([art.scene_count for art in artifacts]),
        median_distinct_locations_per_session=_median(
            [len(distinct_locations(art)) for art in artifacts]
        ),
        mean_turns_per_scene=round(sum(scene_turns) / len(scene_turns), 2)
        if scene_turns else 0.0,
        max_turns_per_scene=max(scene_turns) if scene_turns else 0,
        longest_common_beat_prefix=_longest_common_beat_prefix(artifacts),
    )


# Keys the legacy `coverage_report` exposed; `evaluators.coverage_report` is now
# a thin projection of `corpus_metrics` over these (test_coverage_report asserts
# subset membership, so the extra CorpusMetrics keys are harmless).
COVERAGE_KEYS = (
    "sessions",
    "beats_total",
    "beats_fired",
    "beats_never_fired",
    "endings_total",
    "endings_reached",
    "endings_never_reached",
)


def coverage_report(artifacts: list[SessionArtifact], world: World) -> dict:
    """Backward-compatible reachability sweep — the original `coverage_report`
    shape, projected from `corpus_metrics`."""
    full = corpus_metrics(artifacts, world).to_dict()
    return {k: full[k] for k in COVERAGE_KEYS}
