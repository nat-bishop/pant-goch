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

from handler.world import World

from .session import SessionArtifact


# The authored relationship ladders, longest dependency chains in truth.yaml.
# Used by `ladder_compression`: a ladder is meant to pace one rung per scene, so
# many rungs in a single scene_id is the "instant unburdening" regression. Kept
# explicit (not derived from requires_beats) so the metric reads at a glance; the
# world is small and fixed.
BEAT_LADDERS: dict[str, list[str]] = {
    "tom": [
        "tom_speaks_of_megan_as_a_person",
        "tom_alludes_to_being_on_the_road",
        "tom_named_a_vehicle",
        "tom_admits_full_silence",
        "tom_decides_to_speak_to_police",
    ],
    "dylan": [
        "dylan_acknowledges_the_player",
        "dylan_admits_the_relationship",
        "dylan_speaks_of_megan",
        "dylan_admits_he_knows",
    ],
    "helen": [
        "helen_will_meet_off_record",
        "helen_confirms_she_suspected",
        "helen_hands_over_notes",
    ],
}


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


def _play_turns(art: SessionArtifact) -> list[dict]:
    """The turn / free_text entries (player-facing moments), in order. Walks and
    scene-close lines are excluded — they aren't turns the player acted within."""
    return [e for e in art.transcript if e.get("kind") in ("turn", "free_text")]


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
    """Fraction of player turns that fired no beat — the agency proxy for the
    'look closer at the box' problem. A turn that advances no investigation
    state is a click, not a decision. (The cosmetic-vs-meaningful distinction
    among these is left to the ChoiceMeaningfulness LLM judge; this is the
    cheap volume signal.) Legacy artifacts without per-turn capture contribute
    nothing."""
    filler = 0
    total = 0
    for art in artifacts:
        for e in _play_turns(art):
            if "applied_beats" not in e:
                continue  # legacy artifact: no per-turn signal
            total += 1
            if not e["applied_beats"]:
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


def _ladder_compression(artifacts: list[SessionArtifact]) -> dict[str, float]:
    """Per ladder, the mean (across sessions that touched it) of the most rungs
    that fired within a single scene. ~1.0 = paced one rung per scene (good);
    approaching the ladder length = the whole arc collapsed into one scene (the
    instant-unburdening bug)."""
    out: dict[str, float] = {}
    for name, rungs in BEAT_LADDERS.items():
        rung_set = set(rungs)
        worst: list[int] = []
        for art in artifacts:
            per_scene: Counter = Counter()
            for e in _play_turns(art):
                for b in e.get("applied_beats", []):
                    if b in rung_set:
                        per_scene[e["scene_id"]] += 1
            if per_scene:
                worst.append(max(per_scene.values()))
        out[name] = round(sum(worst) / len(worst), 3) if worst else 0.0
    return out


def _beats_per_location(artifacts: list[SessionArtifact]) -> dict[str, int]:
    """Distinct beats that fired at each location across the corpus — surfaces
    locations carrying no investigative weight."""
    by_loc: dict[str, set[str]] = {}
    for art in artifacts:
        for e in _play_turns(art):
            by_loc.setdefault(e["location"], set()).update(e.get("applied_beats", []))
    return {loc: len(beats) for loc, beats in sorted(by_loc.items())}


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
    # coverage
    beats_total: int
    beats_never_fired: list[str]
    endings_total: int
    endings_reached: list[str]
    endings_never_reached: list[str]
    location_coverage: float
    locations_never_visited: list[str]
    beats_per_location: dict[str, int]
    # depth / linearity
    median_scenes_per_session: float
    median_distinct_locations_per_session: float
    longest_common_beat_prefix: list[str]
    ladder_compression: dict[str, float]

    def to_dict(self) -> dict:
        return self.model_dump()


def corpus_metrics(artifacts: list[SessionArtifact], world: World) -> CorpusMetrics:
    """Reduce a run of artifacts to its distributional health."""
    fired_beats: set[str] = set()
    reached_endings: set[str] = set()
    visited: set[str] = set()
    for art in artifacts:
        fired_beats.update(art.unlocked_beats)
        if art.ending_id:
            reached_endings.add(art.ending_id)
        visited |= distinct_locations(art)

    all_beats = set(world.beats)
    all_locations = set(world.locations)
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
        beats_total=len(all_beats),
        beats_never_fired=sorted(all_beats - fired_beats),
        endings_total=len(world.endings),
        endings_reached=sorted(reached_endings),
        endings_never_reached=sorted(set(world.endings) - reached_endings),
        location_coverage=round(len(visited & all_locations) / len(all_locations), 3)
        if all_locations else 0.0,
        locations_never_visited=sorted(all_locations - visited),
        beats_per_location=_beats_per_location(artifacts),
        median_scenes_per_session=_median([art.scene_count for art in artifacts]),
        median_distinct_locations_per_session=_median(
            [len(distinct_locations(art)) for art in artifacts]
        ),
        longest_common_beat_prefix=_longest_common_beat_prefix(artifacts),
        ladder_compression=_ladder_compression(artifacts),
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
