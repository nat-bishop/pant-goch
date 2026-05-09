"""Engine-vs-encoder consistency tests.

Drift detection for the ASP encoder. The encoder is a second implementation
of `evaluate_predicate` + `derive_*_event` semantics (see `scripts/asp_encoder.py`);
when spine semantics shift, the encoder must shift in lockstep or it
silently desyncs from engine behavior.

Each test:
1. Hand-authors a small Scenario covering one semantic axis.
2. Runs `asp_encoder.encode_scenario` + solve → `encoder_reached`.
3. Runs `drive_engine` deterministically across multiple picker policies
   to enumerate engine-reachable beats → `engine_fired_union`.
4. Asserts `engine_fired_union ⊆ encoder_reached`.

The encoder is allowed to over-approximate (claim reachable when the
engine wouldn't pick that path); it must never under-approximate (claim
unreachable when the engine can in fact fire it).

Bypasses pytest's conftest by importing test functions directly — see
`test_analyze_scenario.py` for the same workaround.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import analyze_scenario  # noqa: E402
import asp_encoder  # noqa: E402
from npc_service.events import (  # noqa: E402
    BeatFired as BeatFiredEvent,
    CanonPromoted,
    SessionEnded,
    fold,
)
from npc_service.models import (  # noqa: E402
    BeatFired,
    Beat,
    BeatSpec,
    CanonCategory,
    Effect,
    ClassifyDirective,
    InboxState,
    NPCDossier,
    Not_,
    Or_,
    PlayerCanonEntry,
    PlayerCanonExists,
    PlayerRepliesToThreadGte,
    QualityEquals,
    QualityGte,
    QualityLt,
    Quality,
    RecentClassifierValue,
    Reply,
    Scenario,
    Thread,
    iter_predicate_leaves,
)
from npc_service.spine import (  # noqa: E402
    derive_classify_effect_event,
    derive_literal_effect_event,
    find_due_beats,
    scene_exit_satisfied,
)


# ───────────────────────── engine driver ─────────────────────────


def _referenced_canon_keys(scenario: Scenario) -> set[str]:
    """All canon keys referenced by any PlayerCanonExists in the scenario.
    Mirrors `ScenarioProfile.from_scenario` walk; needed by canon_picker.
    """
    out: set[str] = set()

    def collect(p) -> None:
        for leaf in iter_predicate_leaves(p):
            if isinstance(leaf, PlayerCanonExists):
                out.add(leaf.name)

    for beat in scenario.beats:
        for pre in beat.preconditions:
            collect(pre)
        for ex in beat.spec.exit_when:
            collect(ex)
        for eff in beat.spec.effects:
            if eff.when is not None:
                collect(eff.when)
    for eff in scenario.effects:
        if eff.when is not None:
            collect(eff.when)
    return out


def drive_engine(
    scenario: Scenario,
    *,
    beat_picker: Callable[[list[Beat]], Beat] = lambda e: e[0],
    verdict_picker: Callable[[Effect], str] = lambda eff: eff.classify.allowed_values[0],  # type: ignore[union-attr]
    canon_picker: Callable[[set[str]], str | None] = lambda refs: None,
    max_ticks: int = 60,
) -> set[str]:
    """Drive the engine deterministically, return set of fired beat IDs.

    Mirrors `main._tick` (minus LLM calls). Each tick:
    1. find_due_beats → eligible list
    2. beat_picker → chosen beat
    3. apply per-beat literal effects (derive + fold)
    4. apply scenario-level literal effects (with thread filter)
    5. record a synthetic reply (so AxisResults attach + reply count grows)
    6. apply per-beat + scenario classify effects (with verdict_picker)
    7. promote canon (if canon_picker returns a key)
    8. emit BeatFired event (completed iff one_shot OR scene_exit_satisfied)
    9. emit SessionEnded if beat is terminal

    The collapsed model matches the encoder: each tick = one beat-fire +
    one implicit player reply to that beat.
    """
    state = InboxState.from_scenario(scenario)
    fired: set[str] = set()
    referenced_canon = _referenced_canon_keys(scenario)

    for tick in range(max_ticks):
        if state.ended:
            break
        eligible, _ = find_due_beats(state, scenario)
        if not eligible:
            break
        beat = beat_picker(eligible)

        # Per-beat literal effects.
        for eff in beat.spec.effects:
            if eff.is_classify:
                continue
            ev = derive_literal_effect_event(state, scenario, eff, beat_id=beat.id)
            if ev is not None:
                fold(state, ev)

        # Scenario-level literal effects with optional thread filter.
        for eff in scenario.effects:
            if eff.is_classify:
                continue
            if eff.thread is not None and eff.thread != beat.thread:
                continue
            ev = derive_literal_effect_event(state, scenario, eff, beat_id=None)
            if ev is not None:
                fold(state, ev)

        # Synthetic player reply — gives reply_on(thread) count + classifier
        # results an anchor. id is unique per tick. body empty (predicates
        # don't read it).
        synthetic = Reply(
            id=f"synth_{tick}",
            thread_id=beat.thread,
            beat_responded_to=beat.id,
            recipient="",
            body="",
            classifier_results=[],
        )
        state.replies.append(synthetic)

        # Per-beat classify effects.
        for eff in beat.spec.effects:
            if not eff.is_classify:
                continue
            verdict = verdict_picker(eff)
            ev = derive_classify_effect_event(
                state, scenario, eff, verdict,
                reply_id=synthetic.id,
            )
            if ev is not None:
                fold(state, ev)

        # Scenario-level classify effects with optional thread filter.
        for eff in scenario.effects:
            if not eff.is_classify:
                continue
            if eff.thread is not None and eff.thread != beat.thread:
                continue
            verdict = verdict_picker(eff)
            ev = derive_classify_effect_event(
                state, scenario, eff, verdict,
                reply_id=synthetic.id,
            )
            if ev is not None:
                fold(state, ev)

        # Canon promotion (one new key per tick, picker decides which).
        new_key = canon_picker(referenced_canon - set(state.player_canon))
        if new_key:
            entry = PlayerCanonEntry(
                summary=f"synth canon {new_key}",
                literal_text_quote="",
                salience=1.0,
                category=CanonCategory.PERSON,
                turn_of_origin=tick,
            )
            fold(state, CanonPromoted(key=new_key, entry=entry))

        # Beat fired event — one_shots always completed; scenes only if
        # scene_exit holds against post-effect state.
        completed = (
            beat.kind == "one_shot"
            or scene_exit_satisfied(beat, state)
        )
        fold(state, BeatFiredEvent(
            beat_id=beat.id, beat_kind=beat.kind, completed=completed,
        ))
        # ever_fired is a private analyzer field, but find_due_beats reads
        # ever_fired_beat_ids for scene re-fire tracking — populate it too.
        state.ever_fired_beat_ids.add(beat.id)
        fired.add(beat.id)

        if beat.ends_session:
            fold(state, SessionEnded(beat_id=beat.id))

    return fired


def encoder_reached(scenario: Scenario) -> set[str]:
    """Run the encoder + solve once, return set of reachable beat IDs."""
    program = asp_encoder.encode_scenario(scenario)
    solved = asp_encoder.solve(program, time_limit_s=20)
    return set(solved.reached_beats)


def assert_engine_subset_encoder(
    scenario: Scenario,
    *,
    picker_policies: list[dict] = None,  # type: ignore[assignment]
) -> tuple[set[str], set[str]]:
    """For each picker policy, drive the engine and union the fired sets.
    Assert engine_fired_union ⊆ encoder_reached. Return both sets for
    diagnostic printing on failure.
    """
    if picker_policies is None:
        picker_policies = [
            {"beat_picker": lambda e: e[0]},   # first eligible
            {"beat_picker": lambda e: e[-1]},  # last eligible
        ]

    union: set[str] = set()
    for policy in picker_policies:
        union |= drive_engine(scenario, **policy)

    encoder_set = encoder_reached(scenario)
    missing = union - encoder_set
    assert not missing, (
        f"encoder under-approximates: engine fired {sorted(missing)} "
        f"but encoder claims unreachable.\n"
        f"engine_fired_union={sorted(union)}\n"
        f"encoder_reached={sorted(encoder_set)}"
    )
    return union, encoder_set


# ───────────────────────── fixture helpers ─────────────────────────


def _npc(npc_id: str = "a") -> NPCDossier:
    return NPCDossier(id=npc_id, display_name=npc_id.upper(),
                      email_address=f"{npc_id}@x.com")


def _thread(thread_id: str = "a", npc_id: str = "a") -> Thread:
    return Thread(id=thread_id, npc=npc_id)


def _make_scenario(beats: list[Beat], qualities: list[Quality] = None,
                   scenario_effects: list[Effect] = None) -> Scenario:
    """Minimal scenario wrapper. One NPC, one thread by default."""
    if qualities is None:
        qualities = [Quality(id="trust", type="int", min=0, max=10, default=0,
                             description="t")]
    return Scenario(
        title="t", premise="t", email_domain="x.test",
        qualities=qualities,
        npcs=[_npc()],
        threads=[_thread()],
        beats=beats,
        effects=scenario_effects or [],
    )


def _beat(bid: str, *, kind="one_shot", preconds=None, effects=None,
          exit_when=None, ends_session=False, prose="hi") -> Beat:
    spec_kwargs = {"effects": effects or [], "exit_when": exit_when or []}
    if kind == "one_shot":
        spec_kwargs["pre_authored_prose"] = prose
    else:  # scene
        spec_kwargs["intent"] = prose
    return Beat(
        id=bid, thread="a", npc="a", kind=kind,
        preconditions=preconds or [],
        spec=BeatSpec(**spec_kwargs),
        ends_session=ends_session,
    )


# ───────────────────────── tests ─────────────────────────


def test_multi_anchor_boot():
    """3 anchored beats with empty preconditions. Engine fires one per
    tick across 3 different picker policies. Encoder reaches all 3."""
    scenario = _make_scenario([
        _beat("a.b1"),
        _beat("a.b2"),
        _beat("a.b3"),
    ])
    union, _ = assert_engine_subset_encoder(scenario, picker_policies=[
        {"beat_picker": lambda e: e[0]},
        {"beat_picker": lambda e: e[1] if len(e) > 1 else e[0]},
        {"beat_picker": lambda e: e[-1]},
    ])
    assert union == {"a.b1", "a.b2", "a.b3"}


def test_chained_boot():
    """welcome (anchor) → bumper (literal trust+=5) → gated (quality_gte 5)."""
    scenario = _make_scenario([
        _beat("a.welcome"),
        _beat("a.bumper", preconds=[BeatFired(beat="a.welcome")],
              effects=[Effect(quality="trust", op="add", value=5)],
              prose="b"),
        _beat("a.gated", preconds=[
            BeatFired(beat="a.bumper"),
            QualityGte(quality="trust", value=5),
        ]),
    ])
    union, _ = assert_engine_subset_encoder(scenario)
    assert "a.gated" in union, "engine must reach gated via the bumper chain"


def test_quality_clobbered_downstream():
    """Regression test for the boot fix. Anchor with literal `trust = 5` +
    a beat gated on `quality_lt trust 1`. Pre-Fix-A the encoder reported
    the gated beat as unreachable because boot pre-fire clobbered trust.
    Post-Fix-A it must be reachable (choice rule fires gated first).
    """
    scenario = _make_scenario([
        _beat("a.welcome"),
        _beat("a.clobber", preconds=[BeatFired(beat="a.welcome")],
              effects=[Effect(quality="trust", op="set", value=5)]),
        _beat("a.gated_lt", preconds=[
            QualityLt(quality="trust", value=1),
        ]),
    ])
    encoder_set = encoder_reached(scenario)
    assert "a.gated_lt" in encoder_set, (
        f"encoder missed a.gated_lt — boot over-approximation may have "
        f"regressed. Got {sorted(encoder_set)}"
    )


def test_scene_re_fire_until_exit_when():
    """Scene gated on `bargain_replies >= 3`, fires repeatedly until
    `exit_when` satisfied. Engine fires it 3 times, encoder reaches it."""
    scenario = _make_scenario(
        beats=[
            _beat("a.welcome"),
            _beat("a.scene", kind="scene",
                  preconds=[BeatFired(beat="a.welcome")],
                  effects=[Effect(quality="bargain_replies", op="add", value=1)],
                  exit_when=[QualityGte(quality="bargain_replies", value=3)],
                  prose="scene"),
        ],
        qualities=[
            Quality(id="bargain_replies", type="int", min=0, max=10, default=0,
                    description="r"),
        ],
    )
    union, _ = assert_engine_subset_encoder(scenario)
    assert "a.scene" in union


def test_classify_branching():
    """Classify effect with allowed_values=[a, b] sets `bargain_state`. Two
    downstream beats gated on `quality_equals` for each verdict — both
    must be reachable in the encoder; engine reaches each across pickers.
    """
    scenario = _make_scenario(
        beats=[
            _beat("a.opener", effects=[Effect(
                quality="bargain_state",
                classify=ClassifyDirective(
                    question="?",
                    allowed_values=["accepted", "refused"],
                ),
            )]),
            _beat("a.via_accepted", preconds=[
                BeatFired(beat="a.opener"),
                QualityEquals(quality="bargain_state", value="accepted"),
            ]),
            _beat("a.via_refused", preconds=[
                BeatFired(beat="a.opener"),
                QualityEquals(quality="bargain_state", value="refused"),
            ]),
        ],
        qualities=[
            Quality(id="bargain_state", type="string", default="untouched",
                    description="s",
                    allowed_values=["untouched", "accepted", "refused"]),
        ],
    )
    union, _ = assert_engine_subset_encoder(scenario, picker_policies=[
        {"beat_picker": lambda e: e[0],
         "verdict_picker": lambda eff: "accepted"},
        {"beat_picker": lambda e: e[0],
         "verdict_picker": lambda eff: "refused"},
    ])
    assert "a.via_accepted" in union
    assert "a.via_refused" in union


def test_or_precondition():
    """Beat gated on `Or(BeatFired X, BeatFired Y)`. Engine fires X path
    → reaches; Y path → reaches. Encoder reaches via either."""
    scenario = _make_scenario([
        _beat("a.x"),
        _beat("a.y"),
        _beat("a.gated_or", preconds=[
            Or_(of=[BeatFired(beat="a.x"), BeatFired(beat="a.y")]),
        ]),
    ])
    union, _ = assert_engine_subset_encoder(scenario)
    assert "a.gated_or" in union


def test_not_mutex():
    """Two beats each gated on `Not(BeatFired other)`. Engine picks one
    path or the other across pickers; both reachable in encoder."""
    scenario = _make_scenario([
        _beat("a.welcome"),
        _beat("a.path_left", preconds=[
            BeatFired(beat="a.welcome"),
            Not_(of=BeatFired(beat="a.path_right")),
        ]),
        _beat("a.path_right", preconds=[
            BeatFired(beat="a.welcome"),
            Not_(of=BeatFired(beat="a.path_left")),
        ]),
    ])
    union, _ = assert_engine_subset_encoder(scenario, picker_policies=[
        {"beat_picker": lambda e: next(b for b in e if "left" in b.id) if any("left" in b.id for b in e) else e[0]},
        {"beat_picker": lambda e: next(b for b in e if "right" in b.id) if any("right" in b.id for b in e) else e[0]},
    ])
    assert "a.path_left" in union
    assert "a.path_right" in union


def test_canon_dependence():
    """Beat gated on `PlayerCanonExists name`. Engine driver promotes the
    canon key after the first tick; downstream beat then reaches.
    """
    scenario = _make_scenario([
        _beat("a.welcome"),
        _beat("a.canon_gated", preconds=[
            PlayerCanonExists(name="name"),
        ]),
    ])
    union = drive_engine(
        scenario,
        canon_picker=lambda refs: "name" if "name" in refs else None,
    )
    encoder_set = encoder_reached(scenario)
    assert "a.canon_gated" in union, "engine driver should promote canon and reach"
    assert "a.canon_gated" in encoder_set, "encoder over-approximates canon → reachable"


def test_recent_classifier_value_match_all():
    """RecentClassifierValue with window=3 + match='all'. A scene fires 3
    times, each producing a "yes" verdict on `axis`; downstream beat is
    gated on "the last 3 verdicts on axis were all yes". Engine reaches it
    when verdict_picker always returns "yes"; encoder reaches it via the
    stable model that picks "yes" at every fire.

    Closes the documented gap that no drift test exercises window>1 +
    non-trivial match. If spine's `_evaluate_recent_classifier_value`
    walk diverges from the encoder's `recent_slot` rule, this test fails.
    """
    scenario = _make_scenario(
        beats=[
            _beat("a.welcome"),
            _beat(
                "a.scene_loop",
                kind="scene",
                preconds=[BeatFired(beat="a.welcome")],
                effects=[
                    Effect(quality="reply_count", op="add", value=1),
                    Effect(
                        quality="axis",
                        classify=ClassifyDirective(
                            question="?", allowed_values=["yes", "no"],
                        ),
                    ),
                ],
                exit_when=[QualityGte(quality="reply_count", value=3)],
                prose="loop",
            ),
            _beat("a.gated_recent", preconds=[
                BeatFired(beat="a.scene_loop"),
                RecentClassifierValue(
                    quality="axis", values=["yes"], window=3, match="all",
                ),
            ]),
        ],
        qualities=[
            Quality(id="reply_count", type="int", min=0, max=10, default=0,
                    description="r"),
            Quality(id="axis", type="string", default="untouched", description="a",
                    allowed_values=["untouched", "yes", "no"]),
        ],
    )
    union, encoder_set = assert_engine_subset_encoder(
        scenario,
        picker_policies=[
            {"beat_picker": lambda e: e[0],
             "verdict_picker": lambda eff: "yes"},
        ],
    )
    assert "a.gated_recent" in union, (
        "engine should fire gated_recent after 3 'yes' verdicts; "
        f"got fired={sorted(union)}"
    )
    assert "a.gated_recent" in encoder_set


def test_effect_when_gate():
    """Effect.when gates a literal write. A bumper beat tries to add to
    `primer`, but the effect is gated on `QualityGte(trigger, 5)`. Order
    matters: if the bumper fires before the trigger-setter, primer stays
    0 and a downstream beat gated on `QualityGte(primer, 5)` is unreachable
    on that path; if the trigger-setter fires first, the bumper's gate
    passes and primer reaches 5.

    Both orderings exist in the brave-mode union; encoder must reach the
    target. Engine driven through both orderings via picker policies.
    """
    scenario = _make_scenario(
        beats=[
            _beat("a.welcome"),
            _beat(
                "a.trigger_setter",
                preconds=[BeatFired(beat="a.welcome")],
                effects=[Effect(quality="trigger", op="add", value=5)],
                prose="set trigger",
            ),
            _beat(
                "a.bumper",
                preconds=[BeatFired(beat="a.welcome")],
                effects=[Effect(
                    quality="primer", op="add", value=5,
                    when=QualityGte(quality="trigger", value=5),
                )],
                prose="bump primer if gate",
            ),
            _beat("a.target", preconds=[QualityGte(quality="primer", value=5)]),
        ],
        qualities=[
            Quality(id="trigger", type="int", min=0, max=10, default=0,
                    description="t"),
            Quality(id="primer", type="int", min=0, max=10, default=0,
                    description="p"),
        ],
    )
    # Two pickers: trigger-first (gate passes; target reachable) and
    # bumper-first (gate fails; target unreachable on this path).
    union, encoder_set = assert_engine_subset_encoder(
        scenario,
        picker_policies=[
            {"beat_picker": lambda e: next(
                (b for b in e if "trigger" in b.id), e[0])},
            {"beat_picker": lambda e: next(
                (b for b in e if "bumper" in b.id), e[0])},
        ],
    )
    assert "a.target" in union, (
        "trigger-first ordering must let bumper's gate pass and reach target"
    )
    assert "a.target" in encoder_set


def test_literal_string_quality_write():
    """Literal-value write to a string quality, with a downstream beat
    gated on `QualityEquals(quality, "value")`. Tests the literal-to-string
    code path in both engine `_combine_op` (string branch) and encoder
    `literal_str_target` rule. Production uses string qualities heavily
    (`bargain_state`, `recognition_state`, `aria_acted_in_player_name`)
    but no existing drift test exercises a literal-value write to one.
    """
    scenario = _make_scenario(
        beats=[
            _beat("a.welcome"),
            _beat(
                "a.set_state",
                preconds=[BeatFired(beat="a.welcome")],
                effects=[Effect(quality="phase", op="set", value="armed")],
                prose="phase armed",
            ),
            _beat("a.gated_phase", preconds=[
                QualityEquals(quality="phase", value="armed"),
            ]),
        ],
        qualities=[
            Quality(id="phase", type="string", default="idle", description="p",
                    allowed_values=["idle", "armed"]),
        ],
    )
    union, encoder_set = assert_engine_subset_encoder(scenario)
    assert "a.gated_phase" in union, (
        "engine must reach gated_phase after literal-string set; "
        f"fired={sorted(union)}"
    )
    assert "a.gated_phase" in encoder_set
