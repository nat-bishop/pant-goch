"""The deterministic spine: predicate evaluation and beat scheduler.

No LLM calls live here — only code-owned state mutation.

Two rules this module enforces:

1. The chat-rendering LLM never touches state. The chat agent returns prose
   only; the spine is the only thing that mutates `InboxState`.
2. Quality writes have ONE code-owned entry point: `apply_effects`. Effects
   come in two flavours (literal `value:` and classifier-driven `classify:`)
   and may be scoped to a beat (`BeatSpec.effects`) or to the scenario
   (`Scenario.effects`). All five surfaces from the pre-refactor era —
   `classify_reply_axes`, `spec.writes`, `spec.writes_by_axis`, the
   verdict-only `axis_name` special case, and the `commitment.*` prefix
   convention — are gone, folded into the unified Effect grammar.

Observability is allowed (and explicitly used here): the spine emits
logfire info/warn lines so per-tick decisions are visible in the dev
console and queryable in the Logfire UI. The "no side effects" rule is
about state mutation, not logging.

Beat selection: `find_due_beats` returns every pending beat whose
preconditions pass, plus a per-beat eligibility report (used by the
per-tick console log and the in-browser debug overlay). When more than
one beat is eligible on the same tick, the LLM Director (in `agents.py`)
picks one — the spine no longer maintains same-tick mutex tracking.
Authoring order is not load-bearing for selection.
"""
from __future__ import annotations

from typing import Literal

import logfire

from .models import (
    AxisResult,
    Beat,
    BeatEligibility,
    BeatEvaluation,
    BeatFired,
    Effect,
    InboxState,
    Not_,
    Or_,
    PendingBeat,
    PlayerCanonEntry,
    PlayerCanonExists,
    PlayerRepliesToThreadGte,
    Predicate,
    PredicateTrace,
    QualityDelta,
    QualityEquals,
    QualityGte,
    QualityLt,
    RecentClassifierValue,
    Scenario,
)


# ───────────────────────── predicate evaluation ─────────────────────────


def evaluate_predicate(pred: Predicate, state: InboxState) -> bool:
    """Evaluate one predicate against current state.

    `BeatFired` only checks `state.fired_beat_ids` — the same-tick mutex
    machinery (extra_fired) is gone. With the Director picking one beat
    per tick, two beats becoming eligible together is a normal case
    resolved by the Director, not a hand-authored mutex chain.
    """
    if isinstance(pred, QualityGte):
        v = state.qualities.get(pred.quality)
        return isinstance(v, (int, float)) and v >= pred.value
    if isinstance(pred, QualityLt):
        v = state.qualities.get(pred.quality)
        return isinstance(v, (int, float)) and v < pred.value
    if isinstance(pred, QualityEquals):
        v = state.qualities.get(pred.quality)
        return v is not None and v == pred.value
    if isinstance(pred, PlayerRepliesToThreadGte):
        n = sum(1 for r in state.replies if r.thread_id == pred.thread)
        return n >= pred.value
    if isinstance(pred, BeatFired):
        return pred.beat in state.fired_beat_ids
    if isinstance(pred, RecentClassifierValue):
        return _evaluate_recent_classifier_value(pred, state)
    if isinstance(pred, Not_):
        return not evaluate_predicate(pred.of, state)
    if isinstance(pred, Or_):
        return any(evaluate_predicate(sub, state) for sub in pred.of)
    if isinstance(pred, PlayerCanonExists):
        return pred.name in state.player_canon
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


def _evaluate_recent_classifier_value(
    pred: RecentClassifierValue, state: InboxState
) -> bool:
    """Walk replies in reverse, collecting up to `pred.window` axis results
    whose quality matches `pred.quality`. Replies without a matching axis
    are skipped (they don't fill a window slot). Resolve `pred.match`
    against the count of collected results whose value ∈ pred.values.
    """
    collected: list[str] = []
    for reply in reversed(state.replies):
        for axis in reply.classifier_results:
            if axis.quality == pred.quality:
                collected.append(axis.value)
                break  # axis-uniqueness invariant: ≤1 per reply per axis
        if len(collected) >= pred.window:
            break

    matches = sum(1 for v in collected if v in pred.values)
    if pred.match == "any":
        return matches >= 1
    if pred.match == "all":
        # Under-filled window can't satisfy "all" — len(collected) < window.
        return matches >= pred.window
    return matches >= pred.match


def evaluate_predicate_with_trace(
    pred: Predicate, state: InboxState
) -> tuple[bool, PredicateTrace]:
    """Evaluate a predicate and return both the bool result AND a trace tree
    that mirrors the predicate's tree shape, with each node carrying its
    own pass/fail.

    Used by `find_due_beats` for blocked beats so the in-browser debug
    overlay can render the predicate tree with each node colored by its
    evaluation result. For an `Or_(A, B, C)` where only B passed but the
    parent failed (because of an outer `Not_`), the frontend can show
    exactly which arm satisfied — flat string descriptions can't.

    Cheaper than running `evaluate_predicate` twice over different shapes:
    leaf evaluation calls back into `evaluate_predicate` (single source of
    truth for leaf semantics); only `Not_` and `Or_` short-circuit are
    overridden here so every child still records.
    """
    if isinstance(pred, Not_):
        sub_pass, sub_trace = evaluate_predicate_with_trace(pred.of, state)
        passed = not sub_pass
        return passed, PredicateTrace(
            passed=passed,
            # Render the negation framing, but expose the inner sub-tree
            # via children so a tree-renderer can collapse "not(...)" into
            # an explicit "NOT" header above the inner.
            label=f"not({sub_trace.label})",
            children=[sub_trace],
        )
    if isinstance(pred, Or_):
        # Walk every child even after one passes — the frontend wants the
        # full picture. `evaluate_predicate` short-circuits via `any()`,
        # which is correct for hot-path eval but wrong here.
        children: list[PredicateTrace] = []
        any_passed = False
        for sub in pred.of:
            sub_pass, sub_trace = evaluate_predicate_with_trace(sub, state)
            children.append(sub_trace)
            if sub_pass:
                any_passed = True
        return any_passed, PredicateTrace(
            passed=any_passed,
            label="or(" + " | ".join(c.label for c in children) + ")",
            children=children,
        )
    # Leaf: defer to evaluate_predicate (single source of truth for leaf
    # semantics — RecentClassifierValue's window walk, BeatFired's set
    # check, etc.). describe_predicate gives the concrete-value label.
    passed = evaluate_predicate(pred, state)
    return passed, PredicateTrace(passed=passed, label=describe_predicate(pred, state))


def describe_predicate(pred: Predicate, state: InboxState) -> str:
    """Render one predicate as a concrete comparison against current state.

    Used by `find_due_beats` to populate `BeatEligibility.blocked_by` and by
    the per-tick console log to answer "why didn't this beat fire?" with
    real values rather than predicate names alone. Output is a one-line
    string designed to read naturally inline in a log message like
    "beat foo blocked: <description>".
    """
    if isinstance(pred, QualityGte):
        return f"{pred.quality}>={pred.value:g} got {_fmt_quality(state.qualities.get(pred.quality))}"
    if isinstance(pred, QualityLt):
        return f"{pred.quality}<{pred.value:g} got {_fmt_quality(state.qualities.get(pred.quality))}"
    if isinstance(pred, QualityEquals):
        return (
            f"{pred.quality}=={pred.value!r} got "
            f"{_fmt_quality(state.qualities.get(pred.quality))}"
        )
    if isinstance(pred, PlayerRepliesToThreadGte):
        n = sum(1 for r in state.replies if r.thread_id == pred.thread)
        return f"replies({pred.thread})>={pred.value} got {n}"
    if isinstance(pred, BeatFired):
        fired = pred.beat in state.fired_beat_ids
        return f"beat_fired({pred.beat})" + ("" if fired else " — not yet")
    if isinstance(pred, RecentClassifierValue):
        collected: list[str] = []
        for reply in reversed(state.replies):
            for axis in reply.classifier_results:
                if axis.quality == pred.quality:
                    collected.append(axis.value)
                    break
            if len(collected) >= pred.window:
                break
        matches = sum(1 for v in collected if v in pred.values)
        match_label = pred.match if isinstance(pred.match, str) else f">={pred.match}"
        values_text = "{" + ", ".join(pred.values) + "}"
        return (
            f"recent({pred.quality}, last {pred.window}) {match_label} "
            f"in {values_text} — got {matches}/{len(collected)}"
        )
    if isinstance(pred, Not_):
        return f"not({describe_predicate(pred.of, state)})"
    if isinstance(pred, Or_):
        return "or(" + " | ".join(describe_predicate(sub, state) for sub in pred.of) + ")"
    if isinstance(pred, PlayerCanonExists):
        entry = state.player_canon.get(pred.name)
        if entry is None:
            return f"player_canon({pred.name}) — not yet"
        return f"player_canon({pred.name})={entry.summary}"
    raise TypeError(f"unhandled predicate: {type(pred).__name__}")


def _fmt_quality(value: float | str | None) -> str:
    if value is None:
        return "(unset)"
    if isinstance(value, str):
        return repr(value)
    return f"{value:g}"


# ───────────────────────── effect application ─────────────────────────


def _clamp_quality(value: float, quality_id: str, scenario: Scenario) -> float:
    q = scenario.quality_by_id.get(quality_id)
    if q is None:
        return value
    if q.min is not None and value < q.min:
        return q.min
    if q.max is not None and value > q.max:
        return q.max
    return value


def _combine_op(
    before: float | str | None,
    incoming: float | str,
    op: str,
    quality_id: str,
    scenario: Scenario,
) -> float | str:
    """Apply an op to combine `before` (current quality value or None for
    unset) with `incoming` (the new value being written). String qualities
    only support `op: set`; numeric qualities support set/add/max/min.

    Numeric writes are clamped via `_clamp_quality` after the op. String
    writes pass through unchanged (no bounds for strings).
    """
    if isinstance(incoming, str):
        # String quality — only `set` is meaningful (op validation upstream
        # guarantees this, but be defensive).
        return incoming
    cur = before if isinstance(before, (int, float)) else 0.0
    if op == "set":
        new = float(incoming)
    elif op == "add":
        new = cur + float(incoming)
    elif op == "max":
        new = max(cur, float(incoming))
    elif op == "min":
        new = min(cur, float(incoming))
    else:
        raise ValueError(f"unknown op {op!r} for quality {quality_id!r}")
    return _clamp_quality(new, quality_id, scenario)


def _apply_one_write(
    state: InboxState,
    scenario: Scenario,
    quality_id: str,
    incoming: float | str,
    op: str,
    *,
    source_label: str,
) -> QualityDelta | None:
    """Apply a single value-write. Returns the resulting QualityDelta if
    the value changed, or None for no-ops. Logs every effective write.
    """
    before = state.qualities.get(quality_id)
    new = _combine_op(before, incoming, op, quality_id, scenario)
    if before == new:
        return None
    state.qualities[quality_id] = new
    logfire.info(
        "quality {q}: {before} -> {after} via {source}",
        q=quality_id,
        before=_fmt_quality(before),
        after=_fmt_quality(new),
        source=source_label,
    )
    return QualityDelta(
        quality=quality_id, before=before, after=new, source=source_label,
    )


def apply_literal_effects(
    state: InboxState,
    scenario: Scenario,
    effects: list[Effect],
    *,
    beat_id: str | None = None,
) -> list[QualityDelta]:
    """Apply every literal-value effect in `effects` (those with
    `effect.value` set). Classify-driven effects are skipped — they fire at
    reply time via `apply_classify_effect`.

    Each effect's `when:` predicate (if set) is evaluated against current
    state; the write is skipped when `when` is false. Source labels read
    `effect:<beat_id>:<quality>` for an auditable trail.
    """
    deltas: list[QualityDelta] = []
    for effect in effects:
        if effect.is_classify:
            continue
        if effect.when is not None and not evaluate_predicate(effect.when, state):
            continue
        source_label = (
            f"effect:{beat_id}:{effect.quality}"
            if beat_id else f"effect:{effect.quality}"
        )
        delta = _apply_one_write(
            state, scenario, effect.quality, effect.value, effect.op,  # type: ignore[arg-type]
            source_label=source_label,
        )
        if delta is not None:
            deltas.append(delta)
    return deltas


def apply_classify_effect(
    state: InboxState,
    scenario: Scenario,
    effect: Effect,
    verdict: str,
    rationale: str = "",
) -> tuple[AxisResult | None, QualityDelta | None]:
    """Apply one classifier verdict to its target quality.

    For numeric qualities, the verdict indexes into `effect.classify.map`
    to produce a float; the caller's `op` (set/add/max/min) combines it
    with the current value. For string qualities, the verdict string is
    the value (only `op: set` is valid; validated at load time).

    Returns `(axis_result, quality_delta)`:
    - `axis_result`: AxisResult on success (caller appends to
      `Reply.classifier_results`), or None when the verdict isn't a
      valid value for this directive.
    - `quality_delta`: the resulting QualityDelta if the value actually
      changed, or None for no-ops (verdict skipped, value unchanged,
      `when:` predicate false).
    """
    if not effect.is_classify:
        raise ValueError(f"apply_classify_effect requires a classify effect; got {effect}")
    directive = effect.classify  # type: ignore[union-attr]
    if verdict not in directive.allowed_values:
        return None, None
    if effect.when is not None and not evaluate_predicate(effect.when, state):
        # Predicate failed at apply time — return the AxisResult so the
        # ledger still records the verdict (predicates downstream may
        # still want to read it), but don't write the quality.
        return AxisResult(quality=effect.quality, value=verdict, rationale=rationale), None

    q = scenario.quality_by_id.get(effect.quality)
    if q is None:
        return None, None

    if q.type == "string":
        incoming: float | str = verdict
    else:
        if directive.map is None:
            return None, None
        incoming = directive.map[verdict]

    delta = _apply_one_write(
        state, scenario, effect.quality, incoming, effect.op,
        source_label=f"classify:{effect.quality}",
    )
    axis_result = AxisResult(
        quality=effect.quality, value=verdict, rationale=rationale,
    )
    return axis_result, delta


# ───────────────────────── player canon mutation ─────────────────────────


# Promotion threshold for extractor candidates. Below this salience, a
# candidate is treated as too speculative to commit to canon. Exposed so
# main.py can categorize candidates (anchor_fail / below_salience /
# promoted) without duplicating the magic number.
CANON_SALIENCE_THRESHOLD = 0.5


def apply_canon_extraction(
    state: InboxState,
    candidates: list,  # list[CanonCandidate] from agents.py — typed loosely to avoid an import cycle
    turn_of_origin: int,
    salience_threshold: float = CANON_SALIENCE_THRESHOLD,
) -> list[str]:
    """Promote extractor-agent candidates above the salience threshold into
    `state.player_canon`. REPLACE semantics on key collision (later mention
    overwrites earlier — supports "actually call me Robert" mid-session).

    The caller is responsible for substring-validating each candidate's
    `literal_text_quote` against the player's reply BEFORE handing it here;
    candidates whose quote isn't a substring of the actual reply must be
    dropped at the orchestration boundary, not silently here.

    Returns the list of canon keys that were created or replaced this call.
    """
    promoted: list[str] = []
    for c in candidates:
        if c.salience < salience_threshold:
            continue
        state.player_canon[c.key] = PlayerCanonEntry(
            summary=c.summary,
            literal_text_quote=c.literal_text_quote,
            salience=c.salience,
            category=c.category,
            turn_of_origin=turn_of_origin,
        )
        promoted.append(c.key)
    return promoted


# ───────────────────────── beat scheduler ─────────────────────────


def find_due_beats(
    state: InboxState, scenario: Scenario
) -> tuple[list[Beat], list[BeatEligibility]]:
    """Return (eligible beats, eligibility report).

    The report classifies every "interesting" pending beat as eligible or
    blocked. Beats already in `fired_beat_ids` are excluded (uninteresting
    after they've fired). Beats blocked on an unsatisfied positive-form
    `BeatFired` prereq are also excluded — they're "structurally not ready"
    (downstream of unfired work) and listing them every tick is noise the
    author doesn't need.

    For blocked beats that DO show up, `blocked_by` carries a rendered
    description of the first failing predicate (with concrete values, e.g.
    `engagement>=4.0 got 2.5`). That's what the per-tick console log
    surfaces and the debug overlay displays.

    Beats stay in `pending_beats` until they're added to `fired_beat_ids`.
    One-shot beats land in `fired_beat_ids` immediately after rendering, so
    they're pruned on the next tick. Scene beats (`kind="scene"`) enter
    `fired_beat_ids` only on the firing tick that satisfies every entry of
    `spec.exit_when`; until then they re-appear in the due list each tick.
    """
    eligible: list[Beat] = []
    report: list[BeatEligibility] = []
    pending_by_id = {pb.beat_id: pb for pb in state.pending_beats}

    for beat in scenario.beats:
        pb = pending_by_id.get(beat.id)
        if pb is None:
            continue
        if beat.id in state.fired_beat_ids:
            continue
        first_failed = next(
            (p for p in beat.preconditions if not evaluate_predicate(p, state)),
            None,
        )
        is_scene = beat.kind == "scene"
        if first_failed is None:
            eligible.append(beat)
            report.append(BeatEligibility(
                beat_id=beat.id, status="eligible", is_scene=is_scene,
            ))
        elif _has_unsatisfied_structural_prereq(beat.preconditions, state):
            # Downstream of unfired work — omit from report (noise).
            continue
        else:
            # Build a structured trace alongside the one-line description so
            # the debug overlay can render an interactive tree (with each
            # node colored by its own pass/fail). `blocked_by` stays as the
            # at-a-glance summary for the per-tick console log + the panel's
            # collapsed view; `predicate_tree` + `predicate_trace` drive
            # expand-on-click in the panel.
            _, trace = evaluate_predicate_with_trace(first_failed, state)
            report.append(BeatEligibility(
                beat_id=beat.id, status="blocked",
                blocked_by=describe_predicate(first_failed, state),
                is_scene=is_scene,
                predicate_tree=first_failed,
                predicate_trace=trace,
            ))

    # Prune completed beats from pending; keep scene beats around so
    # they re-fire on subsequent ticks until exit_when catches them.
    still_pending: list[PendingBeat] = []
    for pb in state.pending_beats:
        if pb.beat_id in state.fired_beat_ids:
            continue
        if pb.beat_id not in scenario.beat_by_id:
            logfire.warn("pending beat not in scenario", beat_id=pb.beat_id)
            continue
        still_pending.append(pb)
    state.pending_beats = still_pending
    return eligible, report


def evaluate_all_beats(state: InboxState, scenario: Scenario) -> list[BeatEvaluation]:
    """Trace every beat's preconditions + exit_when against current state.

    Differs from `find_due_beats` in three ways:
      1. No pruning — every beat in `scenario.beats` produces an entry
         (already-fired one_shots, structurally-blocked downstream beats,
         retired scenes).
      2. Returns a parallel trace per precondition (and per exit_when entry
         on scenes), not just the first failing one. The Beats inspector
         renders the full AND-list with each atom colored by its own
         pass/fail.
      3. No `pending_beats` requirement — operates on the static
         `scenario.beats` set, so it works even after a beat has been
         pruned from `pending_beats` post-fire.

    Reuses `evaluate_predicate_with_trace` for every atom — single source
    of truth for predicate semantics.
    """
    out: list[BeatEvaluation] = []
    for beat in scenario.beats:
        is_fired = beat.id in state.fired_beat_ids
        entry_traces = [
            evaluate_predicate_with_trace(p, state)[1] for p in beat.preconditions
        ]
        exit_traces = [
            evaluate_predicate_with_trace(p, state)[1] for p in beat.spec.exit_when
        ]
        if is_fired:
            status: Literal["fired", "eligible", "blocked"] = "fired"
        elif all(t.passed for t in entry_traces):
            status = "eligible"
        else:
            status = "blocked"
        out.append(BeatEvaluation(
            beat_id=beat.id,
            status=status,
            is_scene=beat.kind == "scene",
            entry_traces=entry_traces,
            exit_traces=exit_traces,
        ))
    return out


def scene_exit_satisfied(beat: Beat, state: InboxState) -> bool:
    """Return True iff every predicate in `beat.spec.exit_when` evaluates to
    true against the current state. Returns False for one-shot beats — the
    caller should only invoke this on `kind="scene"` beats, but defensive
    return is cheaper than adding a precondition.

    List-AND semantics, matching `preconditions`. Empty `exit_when` is
    forbidden by the load-time validator on scene beats, so an empty list
    only appears on one-shots (where this returns False trivially).
    """
    if beat.kind != "scene":
        return False
    if not beat.spec.exit_when:
        return False
    return all(evaluate_predicate(p, state) for p in beat.spec.exit_when)


def _has_unsatisfied_structural_prereq(
    preds: list[Predicate], state: InboxState
) -> bool:
    """A "structural prereq" is a positive-form `BeatFired` predicate. If
    any such prereq isn't yet satisfied, the beat is downstream of unfired
    work. We omit such beats from the eligibility report entirely — every
    tick the author can already see the upstream beat hasn't fired, so
    repeating "downstream blocked: beat_fired(X) — not yet" N ticks running
    is just noise. Negative-form `Not_(BeatFired …)` is excluded — that's
    the mutual-exclusion idiom (always satisfied until the other beat
    actually fires), so it never causes the noise this skip is preventing.
    """
    for p in preds:
        if isinstance(p, BeatFired) and p.beat not in state.fired_beat_ids:
            return True
    return False




def thread_id_from_beat_id(beat_id: str | None) -> str | None:
    """Beat ids are namespaced as `<thread>.<local>` — extract the thread.
    Single source of truth for this convention; do NOT re-implement elsewhere.
    """
    if not beat_id:
        return None
    return beat_id.split(".", 1)[0] if "." in beat_id else beat_id
