"""ASP (Answer Set Programming) encoder + clingo driver for static
scenario reachability analysis.

Replaces the BFS state-space exploration in `analyze_scenario.py` with a
declarative encoding solved by clingo. Published precedent: DendryScope
(Robison & Mateas, AIIDE 2023), which encoded the Dendry quality-based
narrative engine in ASP and ran on Emily Short's *Bee* in ~2s.

The encoder takes a `Scenario` and emits an ASP program string. The driver
runs clingo, enumerates stable models with `--models=0`, and aggregates
reachable atoms (beats, quality values) across models via `#show`
projection.

Identifier convention: beat ids contain dots (`aria.welcome`), which are
not valid in unquoted clingo atoms. We use clingo strings throughout —
`node("aria.welcome").` — so no slugification or collision risk.

Time horizon: `n = 2 * num_beats + 8`. Bounded-trace requirement; complete
for our domain because scene re-fires are bounded by reply-count caps and
one-shots fire at most once.

See the plan at `~/.claude/plans/i-am-going-to-linked-cookie.md` (or the
DendryScope source at `/tmp/asp-research/dendryscope/src/...`) for the
full encoding rationale and the per-rule edge cases this module addresses.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, assert_never

import clingo

from npc_service.models import (
    BeatFired,
    Effect,
    Not_,
    Or_,
    PlayerCanonExists,
    PlayerRepliesToThreadGte,
    Predicate,
    QualityEquals,
    QualityGte,
    QualityLt,
    RecentClassifierValue,
    Scenario,
    iter_predicate_leaves,
)


# ───────────────────────── result dataclasses ─────────────────────────


@dataclass
class SolveResult:
    """Output of `solve()`. Consumed by `analyze_scenario.analyze_scenario`
    to build an `AnalysisReport`.

    `truncated` is True iff clingo hit the wall-clock time limit before
    enumerating all stable models (analogous to the BFS state-space cap).
    """

    reached_beats: set[str]
    quality_bounds: dict[str, tuple[float, float]]
    quality_strings: dict[str, set[str]]
    truncated: bool
    models_enumerated: int
    grounding_atoms: int
    ground_time_s: float
    solve_time_s: float


# ───────────────────────── reachable-value closure ─────────────────────────


def reachable_quality_values(scenario: Scenario) -> dict[str, set[float]]:
    """Compute the per-numeric-quality set of values reachable from boot.

    Iterates a fixpoint: start with `{boot_value}` per quality; repeatedly
    extend with set-targets, classify-map values, and `v + delta` /
    `v - delta` for add/max/min ops; clamp every new value to the quality's
    `[min, max]` if authored. Bounded by `max - min` (or by the literal
    set if unbounded).

    Output is a dict of quality_id -> set[float]. String qualities are
    excluded (they're handled separately via `Quality.allowed_values`).
    """
    qualities_by_id = scenario.quality_by_id

    # Gather all literal write values + add deltas + classify map values per quality.
    set_targets: dict[str, set[float]] = defaultdict(set)
    add_deltas: dict[str, set[float]] = defaultdict(set)

    def _collect_from_effect(effect: Effect) -> None:
        q = effect.quality
        if q not in qualities_by_id or qualities_by_id[q].type == "string":
            return
        if effect.is_classify:
            mapping = effect.classify.map or {}
            for v in mapping.values():
                set_targets[q].add(float(v))
        else:
            v = effect.value
            if v is None or isinstance(v, str):
                return
            fv = float(v)
            if effect.op == "set":
                set_targets[q].add(fv)
            elif effect.op == "add":
                add_deltas[q].add(fv)
            elif effect.op in ("max", "min"):
                # max/min compile to a guarded set with the literal as the target.
                set_targets[q].add(fv)

    for beat in scenario.beats:
        for effect in beat.spec.effects:
            _collect_from_effect(effect)
    for effect in scenario.effects:
        _collect_from_effect(effect)

    out: dict[str, set[float]] = {}
    for q in scenario.qualities:
        if q.type == "string":
            continue
        boot = float(q.default) if q.default is not None else 0.0
        reachable: set[float] = {boot}
        reachable.update(set_targets.get(q.id, set()))

        # Iterate add-deltas to fixpoint, clamping each step.
        deltas = add_deltas.get(q.id, set())
        if deltas:
            lo = q.min if q.min is not None else float("-inf")
            hi = q.max if q.max is not None else float("inf")
            changed = True
            while changed:
                changed = False
                for v in list(reachable):
                    for d in deltas:
                        nv = max(lo, min(hi, v + d))
                        if nv not in reachable:
                            reachable.add(nv)
                            changed = True
        # Always clamp the fixpoint set to authored bounds (in case set-targets
        # landed outside).
        if q.min is not None:
            reachable = {v for v in reachable if v >= q.min}
        if q.max is not None:
            reachable = {v for v in reachable if v <= q.max}
        out[q.id] = reachable
    return out


# ───────────────────────── PredId allocator ─────────────────────────


@dataclass
class PredicateRegistry:
    """Walks every predicate tree in the scenario, assigns each node a
    stable integer PredId, and emits the per-leaf / per-composite ASP
    facts. Caller drives the walk; this object owns the id counter and
    the emitted-fact buffer.

    PredId stability matters for tests — same scenario should produce same
    program. We allocate by walk order (depth-first, left-to-right), so as
    long as the scenario is loaded deterministically from YAML the IDs are
    stable across runs.
    """

    next_id: int = 0
    facts: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    # Cache: id() of predicate object -> assigned PredId. Pydantic models
    # are hashable but expensive to compare; identity is fine because we
    # only register each instance once.
    _by_obj: dict[int, int] = field(default_factory=dict)

    def register(self, pred: Predicate) -> int:
        """Assign a PredId to `pred` and emit its facts. Returns the id.

        Idempotent: same predicate instance returns the same id and
        doesn't re-emit. Sub-predicates inside Not_ / Or_ get registered
        recursively; their relationships to the parent are emitted as
        `pred_not(P, Child).` / `pred_or_arm(P, Child).` facts.
        """
        key = id(pred)
        if key in self._by_obj:
            return self._by_obj[key]
        pid = self.next_id
        self.next_id += 1
        self._by_obj[key] = pid

        if isinstance(pred, QualityGte):
            self.facts.append(
                f'pred_quality_gte({pid}, "{pred.quality}", {_int(pred.value)}).'
            )
        elif isinstance(pred, QualityLt):
            self.facts.append(
                f'pred_quality_lt({pid}, "{pred.quality}", {_int(pred.value)}).'
            )
        elif isinstance(pred, QualityEquals):
            if isinstance(pred.value, str):
                self.facts.append(
                    f'pred_quality_eq_str({pid}, "{pred.quality}", "{pred.value}").'
                )
            else:
                self.facts.append(
                    f'pred_quality_eq_num({pid}, "{pred.quality}", {_int(pred.value)}).'
                )
        elif isinstance(pred, BeatFired):
            self.facts.append(f'pred_beat_fired({pid}, "{pred.beat}").')
        elif isinstance(pred, PlayerCanonExists):
            self.facts.append(f'pred_canon_exists({pid}, "{pred.name}").')
        elif isinstance(pred, PlayerRepliesToThreadGte):
            self.facts.append(
                f'pred_thread_replies_gte({pid}, "{pred.thread}", {pred.value}).'
            )
        elif isinstance(pred, RecentClassifierValue):
            self.facts.append(
                f'recent_predicate({pid}, "{pred.quality}", {pred.window}).'
            )
            for v in pred.values:
                self.facts.append(f'pred_recent_value({pid}, "{v}").')
            if pred.match == "any":
                self.facts.append(f"recent_match_any({pid}).")
            elif pred.match == "all":
                self.facts.append(f"recent_match_all({pid}).")
            else:  # integer N
                self.facts.append(f"recent_match_int({pid}, {pred.match}).")
        elif isinstance(pred, Not_):
            child_pid = self.register(pred.of)
            self.facts.append(f"pred_not({pid}, {child_pid}).")
        elif isinstance(pred, Or_):
            for sub in pred.of:
                child_pid = self.register(sub)
                self.facts.append(f"pred_or_arm({pid}, {child_pid}).")
        else:
            assert_never(pred)
        return pid


# ───────────────────────── helpers ─────────────────────────


def _int(v: float | int) -> int:
    """Coerce a quality value to int. Raises if the value isn't representable
    as int — float-typed qualities aren't supported in this MVP encoder
    (clingo arithmetic is integer-only). Defer the scaling-by-10^N approach
    until a float quality is actually authored."""
    if isinstance(v, bool):  # bool is a subclass of int — guard explicitly
        raise TypeError(f"asp_encoder: boolean value {v!r} not allowed in ASP")
    iv = int(v)
    if float(iv) != float(v):
        raise NotImplementedError(
            f"asp_encoder: non-integer quality value {v!r} not yet supported "
            "(would require value-scaling — see plan §'Float-quality constraint')"
        )
    return iv


def _strs(items: list[str]) -> str:
    """Emit a clingo string-disjunction list: `"a"; "b"; "c"`."""
    return "; ".join(f'"{s}"' for s in items)


# ───────────────────────── encoder ─────────────────────────


def encode_scenario(
    scenario: Scenario,
    *,
    horizon: int | None = None,
) -> str:
    """Emit the full ASP program for `scenario`.

    `horizon` is the time horizon (max steps). Defaults to
    `2 * len(scenario.beats) + 8`, sound for our domain (one_shots fire at
    most once; scenes are reply-count bounded). Authors targeting >200 beats
    should set this explicitly per the plan's grounding-blowup risk.

    No "boot" parameter — boot beats are eligible at T=0 via the standard
    `pre_holds` rule and fire via the same choice rule that picks
    player-tick fires. The Python-side `_boot_tick` over-approximation
    (which fired all boot-eligible beats to a fixpoint, clobbering quality
    state) is gone; ASP's choice rule explores all boot-firing orderings
    correctly under brave-mode.
    """
    if horizon is None:
        horizon = 2 * len(scenario.beats) + 8

    parts: list[str] = []
    parts.append(_emit_skeleton(horizon))
    parts.append(_emit_beats(scenario))
    parts.append(_emit_value_domain(scenario))
    parts.append(_emit_initial_facts(scenario))

    registry = PredicateRegistry()
    parts.append(_emit_predicates(scenario, registry))
    parts.append(_emit_predicate_holds_rules())

    parts.append(_emit_eligibility_and_fires())
    parts.append(_emit_canon(scenario, registry))
    parts.append(_emit_replies())
    parts.append(_emit_effects(scenario, registry))
    parts.append(_emit_state_propagation())
    parts.append(_emit_show())

    return "\n\n".join(p for p in parts if p)


def _emit_skeleton(horizon: int) -> str:
    return f"""% ───── time horizon ─────
#const n = {horizon}.
step(0..n)."""


def _emit_beats(scenario: Scenario) -> str:
    lines = ["% ───── beats ─────"]
    for beat in scenario.beats:
        lines.append(f'node("{beat.id}").')
        if beat.kind == "one_shot":
            lines.append(f'one_shot("{beat.id}").')
        else:
            lines.append(f'scene("{beat.id}").')
        lines.append(f'beat_thread("{beat.id}", "{beat.thread}").')
        if beat.ends_session:
            lines.append(f'ends_session_beat("{beat.id}").')
    return "\n".join(lines)


def _emit_value_domain(scenario: Scenario) -> str:
    """Emit `num_value(Q, V).` per (quality, reachable-value) and
    `str_value(Q, S).` per (string-quality, allowed-value) per the
    reachable-value-closure algorithm."""
    lines = ["% ───── quality value domains ─────"]
    reachable = reachable_quality_values(scenario)
    for q in scenario.qualities:
        if q.type == "string":
            for s in q.allowed_values:
                lines.append(f'str_value("{q.id}", "{s}").')
            # Default value (if not in allowed_values, still must be representable)
            if q.default is not None and q.default not in q.allowed_values:
                lines.append(f'str_value("{q.id}", "{q.default}").')
        else:
            for v in sorted(reachable.get(q.id, set())):
                lines.append(f'num_value("{q.id}", {_int(v)}).')
    return "\n".join(lines)


def _emit_initial_facts(scenario: Scenario) -> str:
    """Emit initial-state quality facts from `scenario.quality_defaults`.

    No boot-fired facts — boot beats are eligible at T=0 via the standard
    `pre_holds` rule and fire via the choice rule like any other tick. No
    `ended(0)` — terminals fired at T=0 propagate via the standard
    `ended(T+1) :- ends_session_beat(B), fires(B, T)` rule.
    """
    lines = ["% ───── initial state (quality defaults) ─────"]
    for qid, val in sorted(scenario.quality_defaults.items()):
        if isinstance(val, str):
            lines.append(f'has_str("{qid}", "{val}", 0).')
        elif val == "" or val is None:
            # Skip empty-string default for an unset string quality —
            # the encoder doesn't distinguish "unset" from "absent."
            continue
        else:
            lines.append(f'has_num("{qid}", {_int(val)}, 0).')
    return "\n".join(lines)


def _emit_predicates(scenario: Scenario, registry: PredicateRegistry) -> str:
    """Walk every predicate tree, assign PredIds, emit pred_*/recent_*/etc.
    facts and per-beat `beat_pre(B, P).` / `beat_exit(B, P).` linkages."""
    lines = ["% ───── per-predicate facts (PredIds + beat linkages) ─────"]
    for beat in scenario.beats:
        for pred in beat.preconditions:
            pid = registry.register(pred)
            lines.append(f'beat_pre("{beat.id}", {pid}).')
        for pred in beat.spec.exit_when:
            pid = registry.register(pred)
            lines.append(f'beat_exit("{beat.id}", {pid}).')
        for effect in beat.spec.effects:
            if effect.when is not None:
                # PredId for the gate; effect-id linkage emitted in _emit_effects.
                registry.register(effect.when)
    for effect in scenario.effects:
        if effect.when is not None:
            registry.register(effect.when)
    # Append the PredicateRegistry-emitted facts after the beat-pre linkages.
    lines.extend(registry.facts)
    return "\n".join(lines)


def _emit_predicate_holds_rules() -> str:
    """The per-leaf and composite `holds(P, T) :- ...` rules. Static —
    independent of the scenario; a scenario contributes facts that bind
    PredIds to leaf shapes."""
    return """% ───── predicate evaluation rules ─────
% Leaves
holds(P, T) :- pred_quality_gte(P, Q, V), step(T), has_num(Q, X, T), X >= V.
holds(P, T) :- pred_quality_lt(P, Q, V), step(T), has_num(Q, X, T), X < V.
holds(P, T) :- pred_quality_eq_num(P, Q, V), step(T), has_num(Q, V, T).
holds(P, T) :- pred_quality_eq_str(P, Q, S), step(T), has_str(Q, S, T).
holds(P, T) :- pred_beat_fired(P, B), step(T), fired(B, T).
holds(P, T) :- pred_canon_exists(P, K), step(T), canon_present(K, T).
holds(P, T) :- pred_thread_replies_gte(P, Th, N), step(T),
               #count{ T1 : reply_on(Th, T1), T1 < T } >= N.

% RecentClassifierValue: walk recent verdict slots per quality.
% A "slot J at time T" is the J-th most recent verdict_at for quality Q
% strictly before T. Bounded by the per-quality window cap.
window_cap(Q, W) :- recent_predicate(_, Q, W0), W = #max{ W1 : recent_predicate(_, Q, W1) }.
recent_slot(Q, J, V, T) :-
    verdict_at(Q, V, T1), step(T), T1 < T,
    J = #count{ T2 : verdict_at(Q, _, T2), T1 < T2, T2 < T },
    window_cap(Q, W), J < W.
recent_count(P, T, N) :- recent_predicate(P, Q, W), step(T),
    N = #count{ J,V : recent_slot(Q, J, V, T), pred_recent_value(P, V), J < W }.
holds(P, T) :- recent_match_any(P), recent_count(P, T, N), N >= 1.
holds(P, T) :- recent_match_all(P), recent_predicate(P, _, W), recent_count(P, T, W).
holds(P, T) :- recent_match_int(P, M), recent_count(P, T, N), N >= M.

% Composition
holds(P, T) :- pred_or_arm(P, Child), holds(Child, T).
holds(P, T) :- pred_not(P, Child), step(T), pred_not(P, _), not holds(Child, T)."""


def _emit_eligibility_and_fires() -> str:
    return """% ───── eligibility, fires, fired propagation ─────
% pre_holds is true when every beat_pre's holds(P, T) is true (vacuous on
% empty preconditions = always-eligible, matches engine).
pre_holds(B, T) :- node(B), step(T), holds(P, T) : beat_pre(B, P).

% exit_when_holds requires at least one beat_exit to exist, then ALL must
% hold. Empty exit_when = scene never exits (engine's defensive behavior).
exit_when_holds(B, T) :- scene(B), step(T), beat_exit(B, _),
                         holds(P, T) : beat_exit(B, P).

eligible(B, T) :- node(B), step(T), not fired(B, T), pre_holds(B, T).

% Choose exactly one fire per step when at least one is eligible (and we
% haven't ended). The eligible(_, T) body atom permits no-fire ticks when
% nothing is eligible.
1 { fires(B, T) : eligible(B, T) } 1 :-
    step(T), eligible(_, T), not ended(T).

% Update fired (one_shot immediate, scene only on exit).
fired(B, T+1) :- one_shot(B), fires(B, T), step(T+1).
fired(B, T+1) :- scene(B), fires(B, T), exit_when_holds(B, T), step(T+1).
fired(B, T+1) :- fired(B, T), step(T+1).
ever_fired(B, T+1) :- fires(B, T), step(T+1).
ever_fired(B, T+1) :- ever_fired(B, T), step(T+1).

% End-of-session propagation.
ended(T+1) :- ends_session_beat(B), fires(B, T), step(T+1).
ended(T+1) :- ended(T), step(T+1)."""


def _emit_canon(scenario: Scenario, registry: PredicateRegistry) -> str:
    """Emit referenced_canon_key facts + the canon_choice rule.

    Mirrors BFS's `not_yet_present` enumeration at `analyze_scenario.py:584`:
    each fire offers the player one new canon-key addition (or none).
    Monotone — canon never lost.
    """
    referenced: set[str] = set()

    def _collect(p: Predicate) -> None:
        for leaf in iter_predicate_leaves(p):
            if isinstance(leaf, PlayerCanonExists):
                referenced.add(leaf.name)

    for beat in scenario.beats:
        for pred in beat.preconditions:
            _collect(pred)
        for pred in beat.spec.exit_when:
            _collect(pred)
        for effect in beat.spec.effects:
            if effect.when is not None:
                _collect(effect.when)
    for effect in scenario.effects:
        if effect.when is not None:
            _collect(effect.when)

    lines = ["% ───── PlayerCanonExists (extractor over-approximation) ─────"]
    for k in sorted(referenced):
        lines.append(f'referenced_canon_key("{k}").')
    lines.append("""0 { canon_choice(K, T) : referenced_canon_key(K), not canon_present(K, T) } 1
    :- fires(_, T).
canon_present(K, T+1) :- canon_choice(K, T), step(T+1).
canon_present(K, T+1) :- canon_present(K, T), step(T+1).""")
    return "\n".join(lines)


def _emit_replies() -> str:
    """Reply-on-thread rule. In the collapsed model, every fire is also a
    reply to that beat — so reply_on(Th, T) is just the fire's thread.
    Boot beats fire via the standard choice rule at T=0 now (no separate
    `boot_fire` predicate needed).
    """
    return """% ───── replies (drives PlayerRepliesToThreadGte) ─────
reply_on(Th, T) :- fires(B, T), beat_thread(B, Th)."""


def _emit_effects(scenario: Scenario, registry: PredicateRegistry) -> str:
    """Emit per-effect facts: literal_op / literal_num_target / literal_str_target
    / classify_num / classify_num_map / classify_str / scenario_literal /
    scenario_classify / thread_match / gate_predicate / ungated.

    Effect IDs are positional (`<beat_id>:<index>` for per-beat,
    `scenario:<index>` for scenario-scope) — stable across runs given a
    deterministic load order.
    """
    lines = ["% ───── effects (literal + classify) ─────"]
    qualities_by_id = scenario.quality_by_id

    def _emit_one(eff_id: str, effect: Effect, owner_beat: str | None) -> None:
        lines.append(f'effect_id("{eff_id}").')
        if owner_beat is not None:
            if effect.is_classify:
                lines.append(f'per_beat_classify("{eff_id}", "{owner_beat}").')
            else:
                lines.append(f'per_beat_literal("{eff_id}", "{owner_beat}").')
        else:
            if effect.is_classify:
                lines.append(f'scenario_classify("{eff_id}").')
            else:
                lines.append(f'scenario_literal("{eff_id}").')

        # Gate
        if effect.when is None:
            lines.append(f'ungated("{eff_id}").')
        else:
            gate_pid = registry.register(effect.when)
            lines.append(f'gate_predicate("{eff_id}", {gate_pid}).')

        # Thread match (only meaningful for scenario-scope; per-beat effects
        # inherit thread from owner beat). We always emit thread_match for
        # uniformity — for per-beat effects, the rule body matches the
        # owner beat's thread.
        if owner_beat is None and effect.thread is not None:
            lines.append(f'scenario_effect_thread("{eff_id}", "{effect.thread}").')

        # Quality target + op
        q = effect.quality
        is_string = qualities_by_id.get(q) and qualities_by_id[q].type == "string"

        if effect.is_classify:
            if is_string:
                lines.append(f'classify_str("{eff_id}", "{q}").')
                for v in effect.classify.allowed_values:
                    lines.append(f'allowed_value("{eff_id}", "{v}").')
            else:
                lines.append(f'classify_num("{eff_id}", "{q}").')
                for v in effect.classify.allowed_values:
                    lines.append(f'allowed_value("{eff_id}", "{v}").')
                    if effect.classify.map and v in effect.classify.map:
                        lines.append(
                            f'classify_num_map("{eff_id}", "{v}", '
                            f'{_int(effect.classify.map[v])}).'
                        )
        else:
            # Literal
            if is_string:
                lines.append(f'literal_str_target("{eff_id}", "{q}", "{effect.value}").')
                # String qualities only support op=set per the model validator.
            else:
                v = _int(effect.value)
                lines.append(f'literal_num_target("{eff_id}", "{q}", {v}).')
                lines.append(f'literal_op("{eff_id}", {effect.op}).')

    for beat in scenario.beats:
        for i, effect in enumerate(beat.spec.effects):
            _emit_one(f"{beat.id}:{i}", effect, beat.id)
    for i, effect in enumerate(scenario.effects):
        _emit_one(f"scenario:{i}", effect, None)

    # Static rules: when_holds, classify_runs, literal_runs, verdict choice,
    # verdict_at, num_set_at/num_add_at/str_write_at.
    lines.append("""
% Gate evaluation: ungated effects always pass; gated effects pass when their
% predicate holds at T.
when_holds(EffId, T) :- ungated(EffId), step(T).
when_holds(EffId, T) :- gate_predicate(EffId, P), holds(P, T).

% Per-beat thread match: trivially the owner beat's thread.
thread_match(EffId, B) :- per_beat_classify(EffId, B).
thread_match(EffId, B) :- per_beat_literal(EffId, B).
% Scenario-scope thread match: matches the fire's beat thread.
thread_match(EffId, B) :- scenario_classify(EffId), node(B),
                          not scenario_effect_thread(EffId, _).
thread_match(EffId, B) :- scenario_literal(EffId), node(B),
                          not scenario_effect_thread(EffId, _).
thread_match(EffId, B) :- scenario_classify(EffId), node(B),
                          beat_thread(B, Th), scenario_effect_thread(EffId, Th).
thread_match(EffId, B) :- scenario_literal(EffId), node(B),
                          beat_thread(B, Th), scenario_effect_thread(EffId, Th).

% A classify effect RUNS (records a verdict in reply history) whenever the
% relevant beat fires. Gate only controls the quality WRITE, not the
% verdict record — matches engine: AxisResult is appended to
% Reply.classifier_results even when when:gate fails the write.
classify_runs(EffId, T) :- per_beat_classify(EffId, B), fires(B, T).
classify_runs(EffId, T) :- scenario_classify(EffId), fires(B, T),
                           thread_match(EffId, B).

% A literal effect runs when the fire happens AND the gate passes.
literal_runs(EffId, T) :- per_beat_literal(EffId, B), fires(B, T),
                          when_holds(EffId, T).
literal_runs(EffId, T) :- scenario_literal(EffId), fires(B, T),
                          thread_match(EffId, B), when_holds(EffId, T).

% Verdict choice — exactly one allowed value per classify-run.
1 { verdict(EffId, V, T) : allowed_value(EffId, V) } 1 :- classify_runs(EffId, T).

% verdict_at is what RecentClassifierValue reads; fires regardless of gate.
verdict_at(Q, V, T) :- classify_runs(EffId, T), classify_num(EffId, Q),
                       verdict(EffId, V, T).
verdict_at(Q, V, T) :- classify_runs(EffId, T), classify_str(EffId, Q),
                       verdict(EffId, V, T).

% Numeric writes (set / add / max / min)
num_set_at(Q, V, T) :- literal_runs(EffId, T), literal_op(EffId, set),
                       literal_num_target(EffId, Q, V).
num_set_at(Q, V, T) :- classify_runs(EffId, T), classify_num(EffId, Q),
                       when_holds(EffId, T), verdict(EffId, Vd, T),
                       classify_num_map(EffId, Vd, V).
num_add_at(Q, D, T) :- literal_runs(EffId, T), literal_op(EffId, add),
                       literal_num_target(EffId, Q, D).
% max/min compile to guarded set against current value.
num_set_at(Q, V, T) :- literal_runs(EffId, T), literal_op(EffId, max),
                       literal_num_target(EffId, Q, V),
                       has_num(Q, V0, T), V0 < V.
num_set_at(Q, V0, T) :- literal_runs(EffId, T), literal_op(EffId, max),
                        literal_num_target(EffId, Q, V),
                        has_num(Q, V0, T), V0 >= V.
num_set_at(Q, V, T) :- literal_runs(EffId, T), literal_op(EffId, min),
                       literal_num_target(EffId, Q, V),
                       has_num(Q, V0, T), V0 > V.
num_set_at(Q, V0, T) :- literal_runs(EffId, T), literal_op(EffId, min),
                        literal_num_target(EffId, Q, V),
                        has_num(Q, V0, T), V0 <= V.

% String writes
str_write_at(Q, S, T) :- literal_runs(EffId, T), literal_str_target(EffId, Q, S).
str_write_at(Q, S, T) :- classify_runs(EffId, T), classify_str(EffId, Q),
                         when_holds(EffId, T), verdict(EffId, S, T)."""
    )
    return "\n".join(lines)


def _emit_state_propagation() -> str:
    return """% ───── quality state propagation (frame axiom) ─────
has_num(Q, V, T+1) :- num_set_at(Q, V, T), num_value(Q, V), step(T+1).
has_num(Q, V, T+1) :- num_add_at(Q, D, T), has_num(Q, V0, T),
                      V = V0 + D, num_value(Q, V), step(T+1).
num_updated(Q, T) :- num_set_at(Q, _, T).
num_updated(Q, T) :- num_add_at(Q, _, T).
has_num(Q, V, T+1) :- has_num(Q, V, T), not num_updated(Q, T), step(T+1).

has_str(Q, S, T+1) :- str_write_at(Q, S, T), step(T+1).
str_updated(Q, T) :- str_write_at(Q, _, T).
has_str(Q, S, T+1) :- has_str(Q, S, T), not str_updated(Q, T), step(T+1)."""


def _emit_show() -> str:
    return """% ───── reachability output ─────
#show reaches_beat/1.
reaches_beat(B) :- ever_fired(B, T), step(T).
#show reaches_num/2.
reaches_num(Q, V) :- has_num(Q, V, T), step(T).
#show reaches_str/2.
reaches_str(Q, S) :- has_str(Q, S, T), step(T)."""


# ───────────────────────── solver ─────────────────────────


def solve(
    program: str,
    *,
    time_limit_s: int = 30,
) -> SolveResult:
    """Run clingo on `program`, aggregate reachable atoms across all stable
    models. Returns a `SolveResult` ready for `AnalysisReport` composition.

    `time_limit_s` is integer seconds; if clingo hits it before exhausting
    the model space, `truncated=True` and reachability claims are partial
    (analogous to the BFS state-space cap). The Python `Control` doesn't
    accept `--time-limit` (CLI-only option), so we use async solve with
    `handle.wait()` and `handle.cancel()` for the time bound.

    `--models=0` enumerates all stable models. Each model is projected via
    `#show` to its reachable atoms; the on_model callback aggregates per
    atom name into the result containers.
    """
    # `--enum-mode=brave` is the killer optimization for reachability:
    # clingo computes the UNION of all stable models — every atom that's
    # true in any stable model — and emits it as the FINAL answer set.
    # That's literally the definition of "reachable" for our domain.
    # ~75x speedup over per-model enumeration on the production scenario.
    #
    # `--models=0` is required by brave-mode: clingo enumerates models
    # internally to compute the brave union, and bailing early would leave
    # consequences uncovered. The on_model callback may fire multiple
    # times — only the LAST invocation has the brave union; earlier ones
    # are intermediate enumerated models. We aggregate by taking the
    # final shown atoms (the per-call aggregation is monotonic-union, so
    # taking max is safe and matches brave semantics either way).
    ctl = clingo.Control([
        "--models=0",
        "--enum-mode=brave",
        "--warn=none",
    ])
    ctl.add("base", [], program)

    t0 = time.monotonic()
    ctl.ground([("base", [])])
    ground_time = time.monotonic() - t0

    reached_beats: set[str] = set()
    quality_bounds: dict[str, tuple[float, float]] = {}
    quality_strings: dict[str, set[str]] = defaultdict(set)
    models_count = [0]

    def on_model(m: clingo.Model) -> None:
        models_count[0] += 1
        for sym in m.symbols(shown=True):
            name = sym.name
            args = sym.arguments
            if name == "reaches_beat" and len(args) == 1:
                reached_beats.add(args[0].string)
            elif name == "reaches_num" and len(args) == 2:
                q = args[0].string
                v = float(args[1].number)
                lo, hi = quality_bounds.get(q, (v, v))
                quality_bounds[q] = (min(lo, v), max(hi, v))
            elif name == "reaches_str" and len(args) == 2:
                quality_strings[args[0].string].add(args[1].string)

    t0 = time.monotonic()
    with ctl.solve(on_model=on_model, async_=True) as handle:
        finished = handle.wait(float(time_limit_s))
        if not finished:
            handle.cancel()
        result = handle.get()
    solve_time = time.monotonic() - t0

    truncated = not result.exhausted

    # Best-effort grounding atom count from clingo statistics.
    try:
        atoms = int(ctl.statistics["problem"]["lp"]["atoms"])
    except (KeyError, TypeError):
        atoms = 0

    return SolveResult(
        reached_beats=reached_beats,
        quality_bounds=quality_bounds,
        quality_strings=dict(quality_strings),
        truncated=truncated,
        models_enumerated=models_count[0],
        grounding_atoms=atoms,
        ground_time_s=round(ground_time, 3),
        solve_time_s=round(solve_time, 3),
    )
