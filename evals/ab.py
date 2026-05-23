"""A/B experiment orchestrator — the multi-arm case of the corpus run.

Driven by `evals play --experiment <name>`. Runs each arm's corpus through the
reusable `run.run_corpus` (one `RunConfig` per arm, artifacts tagged by arm),
then:

  1. diffs the deterministic corpus metrics (test vs the control baseline),
     reusing `report._fmt_delta` / `_HIGHER_IS_BETTER`;
  2. (with --judge) judges matched blinded transcript pairs head-to-head and
     clusters the rationales into common themes (`judges.pairwise`);
  3. always writes blinded `pairs.jsonl` (+ a `pairs.key.json` sidecar) for the
     optional human rater (`scripts/ab_rater.html`); with --human, merges any
     `human_ratings.jsonl` back in as a human win-rate + agreement-with-LLM.

Output lands in its own trend root (`evals/artifacts/ab/<ts>/`) as `ab_summary.json`
+ `ab_report.md`, with per-arm corpora under `<ts>/<arm>/`. Deterministic metric
deltas are the trustworthy signal; pairwise + themes are sampled trend (the
CONTROL_LOOP.md doctrine: assert what's true, measure the rest).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

# Tag eval spans env=eval before importing handler.main (idempotent with evals.cli).
os.environ.setdefault("HANDLER_ENV", "eval")

import yaml

from handler.config import Settings
from handler.main import build_model
from handler.scene_loop import RunConfig
from handler.world import load_world
from handler.agents import build_scene_runner_bundle

from . import report as run_report
from . import run as run_lane
from .judges.model import build_judge_model, warm_judge
from .judges.pairwise import cluster_themes, run_pairwise, tally
from .judges.session_judges import _session_context
from .metrics import corpus_metrics
from .retry_model import RetryModel
from .session import SessionArtifact, load_run
from .transitions import SessionView, session_view

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


@dataclass
class Arm:
    name: str
    run_config: dict


@dataclass
class Experiment:
    name: str
    description: str
    personas: str | None
    runs_per_persona: int
    max_scenes: int
    seed: int
    judge_model: str | None
    arms: list[Arm]


def load_experiment(experiment: str) -> Experiment:
    """Resolve `experiment` (a bare name → evals/experiments/<name>.yaml, or a
    direct path) and validate its shape: exactly two arms named `control`
    (the diff baseline) and `test`, each with a `run_config` whose keys are real
    RunConfig fields; `shared` carries the fairness-locked corpus params.

    Names and keys are validated HERE — at load, before any engine setup — so a
    typo fails fast with a clear message instead of a KeyError/TypeError after a
    minutes-long corpus has already run."""
    valid_keys = {f.name for f in fields(RunConfig)}
    path = Path(experiment)
    if not path.exists():
        path = _EXPERIMENTS_DIR / f"{experiment}.yaml"
    if not path.exists():
        raise SystemExit(
            f"experiment not found: {experiment!r} "
            f"(looked for a path and {_EXPERIMENTS_DIR / (experiment + '.yaml')})"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    shared = data.get("shared", {})
    arms = [Arm(name=a["name"], run_config=a.get("run_config", {})) for a in data["arms"]]
    if {a.name for a in arms} != {"control", "test"}:
        raise SystemExit(
            f"experiment {path.name} must define exactly 2 arms named "
            f"'control' and 'test', got {[a.name for a in arms]}"
        )
    for arm in arms:
        bad = set(arm.run_config) - valid_keys
        if bad:
            raise SystemExit(
                f"experiment {path.name} arm '{arm.name}': unknown run_config "
                f"keys {sorted(bad)}; valid keys are {sorted(valid_keys)}"
            )
    return Experiment(
        name=data["name"],
        description=data.get("description", ""),
        personas=shared.get("personas"),
        runs_per_persona=int(shared.get("runs_per_persona", 4)),
        max_scenes=int(shared.get("max_scenes", 60)),
        seed=int(shared.get("seed", 0)),
        judge_model=(data.get("judge") or {}).get("model"),
        arms=arms,
    )


def _build_pairs(
    control: list[SessionArtifact], test: list[SessionArtifact]
) -> list[tuple[str, str, SessionView, SessionView]]:
    """Matched pairs by persona. Runs within a persona are exchangeable samples
    (temp=1.0), so we sort each arm's sessions for a deterministic order and zip
    them per persona — a balanced control-vs-test pair for each cell, to the
    shorter side when a session dropped."""
    def by_persona(arts: list[SessionArtifact]) -> dict[str, list[SessionArtifact]]:
        out: dict[str, list[SessionArtifact]] = {}
        for a in sorted(arts, key=lambda a: a.session_id):
            out.setdefault(a.persona, []).append(a)
        return out

    cmap, tmap = by_persona(control), by_persona(test)
    pairs: list[tuple[str, str, SessionView, SessionView]] = []
    for persona in sorted(set(cmap) & set(tmap)):
        for i, (c, t) in enumerate(zip(cmap[persona], tmap[persona])):
            pairs.append((f"{persona}-{i}", persona, session_view(c), session_view(t)))
    return pairs


def _render_report(summary: dict) -> str:
    """The A/B markdown report: north-star diff (test vs control baseline),
    coverage side-by-side, pairwise win-rate, themes, optional human block."""
    arms = summary["arms"]
    mc = summary["metrics"]["control"]
    mt = summary["metrics"]["test"]
    L: list[str] = []
    L.append(f"# A/B report — {summary['experiment']}  ({summary['generated_at']})")
    L.append(f"_{summary['description']}_")
    L.append("")
    L.append(f"- control: `{arms['control']}`  ({mc['sessions']} sessions)")
    L.append(f"- test:    `{arms['test']}`  ({mt['sessions']} sessions)")
    L.append("")
    L.append("## North star (test vs control)")
    for key in run_report.NORTH_STAR_KEYS:
        cur, base = mt.get(key), mc.get(key)
        delta = run_report._fmt_delta(key, cur, base)
        L.append(f"- **{key}**: control={base}  test={cur}{delta}")
    L.append("")
    L.append("## Coverage & depth (control | test)")
    for key in ("endings_reached", "beats_fired", "beats_never_fired",
                "location_coverage", "median_scenes_per_session",
                "median_distinct_locations_per_session"):
        L.append(f"- {key}: {mc.get(key)} | {mt.get(key)}")
    L.append("")
    pw = summary.get("pairwise")
    if pw:
        L.append("## Pairwise preference (subjective trend, not a gate)")
        L.append(f"- pairs judged: {pw['n_pairs']}  "
                 f"(control {pw['wins']['control']} / test {pw['wins']['test']} / ties {pw['ties']})")
        L.append(f"- test win-rate (decided pairs): {pw['win_rate_test']}")
        L.append("")
    themes = summary.get("themes")
    if themes:
        L.append("## Common themes")
        for t in themes:
            L.append(f"- **{t['label']}** — favors {t['favors_arm']} (n={t['frequency']}): {t['summary']}")
        L.append("")
    human = summary.get("human")
    if human:
        L.append("## Human ratings")
        L.append(f"- pairs rated: {human['n']}  test win-rate: {human['win_rate_test']}")
        L.append(f"- agreement with LLM judge: {human['agreement_with_llm']}")
    return "\n".join(L)


def _merge_human(out_root: Path, llm_outcomes_by_id: dict[str, str | None]) -> dict | None:
    """Merge a human_ratings.jsonl (if present) against the blinded pairs.key.json:
    human test win-rate + agreement with the LLM judge (same arm picked)."""
    ratings_path = out_root / "human_ratings.jsonl"
    key_path = out_root / "pairs.key.json"
    if not ratings_path.exists() or not key_path.exists():
        return None
    key = json.loads(key_path.read_text(encoding="utf-8"))
    wins = {"control": 0, "test": 0}
    agree = with_llm = total = 0
    for line in ratings_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pid, choice = r["pair_id"], r["choice"]
        if choice == "tie" or pid not in key:
            continue
        arm = key[pid].get(choice)
        if arm not in wins:
            continue
        wins[arm] += 1
        total += 1
        # Agreement is computed only over pairs the LLM judge actually decided
        # (a non-None verdict); if --judge wasn't run, llm_outcomes_by_id is
        # empty and agreement is None — never a misleading 0.0.
        if llm_outcomes_by_id.get(pid) is not None:
            with_llm += 1
            if llm_outcomes_by_id[pid] == arm:
                agree += 1
    decided = wins["control"] + wins["test"]
    return {
        "n": total,
        "win_rate_test": round(wins["test"] / decided, 3) if decided else None,
        "agreement_with_llm": round(agree / with_llm, 3) if with_llm else None,
    }


async def main(
    *,
    experiment: str,
    runs_per_persona: int | None = None,
    parallel: int | None = None,
    max_scenes: int | None = None,
    judge: bool = False,
    human: bool = False,
    judge_model: str | None = None,
    out: Path | None = None,
) -> None:
    exp = load_experiment(experiment)
    settings = Settings()
    if not settings.gemini_api_key:
        raise SystemExit(
            "no Gemini API key — set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env "
            "or the environment."
        )
    runs = runs_per_persona or exp.runs_per_persona
    parallel = parallel or settings.playtest_parallel
    max_scenes = max_scenes if max_scenes is not None else exp.max_scenes
    out_root = out or (_REPO_ROOT / "evals" / "artifacts" / "ab" / time.strftime("%Y%m%d-%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)

    api_url = run_lane.resolve_api_url(settings)
    world = load_world(_REPO_ROOT / "world")
    model = RetryModel(build_model(api_url))
    bundle = build_scene_runner_bundle(
        model, world, decision_max_tokens=settings.decision_agent_max_tokens
    )
    persona_map = run_lane.load_personas(exp.personas.split(",") if exp.personas else None)

    print(f"A/B '{exp.name}': {len(exp.arms)} arms x {len(persona_map)} personas x "
          f"{runs} runs @ parallel={parallel}; engine={api_url}")
    print(f"artifacts -> {out_root}")

    # ── run each arm's corpus (sequentially — keeps engine KV-slot accounting
    #    identical to a single playtest, no two arms contending for slots) ──
    metrics: dict[str, dict] = {}
    invariants: dict[str, dict] = {}
    artifacts: dict[str, list[SessionArtifact]] = {}
    expected = len(persona_map) * runs
    below_floor: list[str] = []
    for arm in exp.arms:
        arm_dir = out_root / arm.name
        print(f"\n=== arm '{arm.name}' — run_config={arm.run_config} ===")
        invariant_report = await run_lane.run_corpus(
            bundle, world,
            personas=persona_map,
            runs_per_persona=runs,
            parallel=parallel,
            max_scenes=max_scenes,
            out_dir=arm_dir,
            player_model=settings.playtest_player_model,
            player_api_key=settings.gemini_api_key,
            run_config=RunConfig(**arm.run_config),
            arm=arm.name,
        )
        arts = load_run(arm_dir)
        survivors = len(arts)
        if survivors < 0.8 * expected:
            print(f"⚠️  arm '{arm.name}': only {survivors}/{expected} sessions survived "
                  f"(<80%) — its metrics are over survivors only.")
            below_floor.append(arm.name)
        artifacts[arm.name] = arts
        metrics[arm.name] = corpus_metrics(arts, world).to_dict()
        invariants[arm.name] = run_report.invariant_pass_rates(invariant_report)

    # ── matched blinded pairs (used by both the LLM judge and the human rater) ──
    pairs = _build_pairs(artifacts["control"], artifacts["test"])
    _write_pairs(out_root, pairs, seed=exp.seed)
    print(f"\nbuilt {len(pairs)} matched pairs -> {out_root / 'pairs.jsonl'}")

    summary: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": exp.name,
        "description": exp.description,
        "arms": {a.name: a.run_config for a in exp.arms},
        "metrics": metrics,
        "invariants": invariants,
    }

    llm_by_id: dict[str, str | None] = {}
    if judge and pairs:
        jmodel = build_judge_model(settings, override=judge_model or exp.judge_model)
        warm_judge(settings, override=judge_model or exp.judge_model)
        print(f"pairwise judging {len(pairs)} pairs ...", flush=True)
        outcomes = await run_pairwise(jmodel, pairs, concurrency=parallel, seed=exp.seed)
        llm_by_id = {o.pair_id: o.winner_arm for o in outcomes}
        summary["pairwise"] = tally(outcomes)
        # Themes are a best-effort summary — never let a judge timeout/error here
        # discard the whole run (both corpora + the pairwise verdicts already in
        # `summary`). Fall back to an empty themes list and keep going.
        try:
            themes = await cluster_themes(jmodel, outcomes)
            summary["themes"] = [t.model_dump() for t in themes.themes]
        except Exception as e:  # noqa: BLE001 — best-effort; the report still lands
            print(f"⚠️  theme clustering failed ({e}); writing report without themes.")
            summary["themes"] = []

    if human:
        merged = _merge_human(out_root, llm_by_id)
        if merged:
            summary["human"] = merged

    (out_root / "ab_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md = _render_report(summary)
    (out_root / "ab_report.md").write_text(report_md, encoding="utf-8")
    print("\n" + report_md)
    print(f"\nwrote {out_root / 'ab_summary.json'}\n      {out_root / 'ab_report.md'}")

    # Match run.main's integrity gate: write the report (forensics) but exit
    # non-zero if any arm fell below the survivor floor, so a decimated A/B
    # isn't mistaken for a clean result.
    if below_floor:
        raise SystemExit(
            f"corpus too small to trust: arm(s) {below_floor} under the 80% "
            f"survivor floor. Report above is over survivors only."
        )


def _write_pairs(out_root: Path, pairs, *, seed: int) -> None:
    """Blinded `pairs.jsonl` + un-blind sidecar `pairs.key.json` for the human
    rater. Independent blinding from the LLM judge's — agreement is computed on
    the chosen ARM, so the two need not share an A/B assignment."""
    import random

    rng = random.Random(seed + 1)  # +1 so it's not identical to the judge's blinding
    lines, key = [], {}
    for pid, persona, control, test in pairs:
        control_is_a = rng.random() < 0.5
        a, b = (control, test) if control_is_a else (test, control)
        key[pid] = {"A": "control" if control_is_a else "test",
                    "B": "test" if control_is_a else "control"}
        lines.append(json.dumps({
            "pair_id": pid,
            "persona": persona,
            "A": _session_context(a),
            "B": _session_context(b),
        }))
    (out_root / "pairs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "pairs.key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
