"""Judge run — score a recorded playtest with the LLM judges, and persist it.

    uv run python -m evals.judge_run --run evals/artifacts/<timestamp>

Decoupled from generation (`evals.run`): reads dumped artifacts only, so it needs
no llama session and can re-judge a frozen corpus any time. Defaults to local Qwen
via inferno; warms it once, then the scorer fires cases concurrently. Writes
`judges.json` + `judges.md` into the run dir alongside `summary.json`.

Judges are grouped by the scope they score:
  - beat_warrant  — every (turn × fired beat): was the firing warranted / depicted?
  - transition    — every turn: prose↔location, choice meaningfulness, NPC voice
  - free_text     — typed turns: did the narrator accept a false premise?
  - session       — whole playthroughs: earned ending, emergent continuity, agency
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("HANDLER_ENV", "eval")
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8")

# Importing handler.main configures Logfire + instruments pydantic-ai (same setup
# evals.run gets), so judge model calls are traced and tagged env=eval.
import handler.main  # noqa: F401

from handler.config import Settings
from handler.world import load_world

from .judges.model import build_judge_model, warm_judge
from .judges.scorer import persist, render_console, score_plan
from .judges.session_judges import (
    ChoiceConsequenceFelt,
    EmergentEventHonored,
    EndingEarnedNarratively,
)
from .judges.transition_judges import (
    BeatFiredAppropriately,
    ChoiceMeaningfulness,
    FalsePremiseAccepted,
    NpcVoiceAdherence,
    ProseDepictsBeat,
    ProseMatchesLocation,
)
from .session import SessionArtifact
from .transitions import extract_transitions, session_view

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HANDLER judge run")
    ap.add_argument("--run", type=Path, required=True, help="artifact dir to judge")
    ap.add_argument("--world", type=Path, default=_REPO_ROOT / "world")
    ap.add_argument("--judge-model", type=str, default=None,
                    help="override Settings.judge_model (e.g. gemini:gemini-3.1-pro)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per judge (quick smoke of the judges)")
    ap.add_argument("--concurrency", type=int, default=64,
                    help="max judge calls in flight. 64 drains reliably on a single "
                         "vLLM instance; higher saturated the queue and stalled "
                         "responses when judging 20260520-194311.")
    return ap.parse_args()


async def main() -> None:
    args = _parse_args()
    settings = Settings()
    world = load_world(args.world)
    model = build_judge_model(settings, override=args.judge_model)

    print(f"judge_run: scoring {args.run} with "
          f"{args.judge_model or settings.judge_model}", flush=True)
    warm_judge(settings, override=args.judge_model)

    artifacts = [
        SessionArtifact.load(p)
        for p in sorted(Path(args.run).glob("*.json")) if p.name != "summary.json"
    ]
    transitions = [t for art in artifacts for t in extract_transitions(art)]
    sessions = [session_view(art) for art in artifacts]
    print(f"extracted {len(transitions)} transitions, {len(sessions)} sessions", flush=True)

    def cap(items: list) -> list:
        return items[: args.limit] if args.limit else items

    # Item lists keyed (case_id, item) per scope.
    tx_items = [(f"{t.session_id}-s{t.scene_id}-{i}", t) for i, t in enumerate(transitions)]
    warrant_items = [
        (f"{t.session_id}-s{t.scene_id}-{b}-{i}", t.model_copy(update={"beat_of_interest": b}))
        for i, t in enumerate(transitions) for b in t.fired_beats
    ]
    free_text_items = [it for it in tx_items if it[1].is_free_text]
    session_items = [(s.session_id, s) for s in sessions]

    # (judge instance, items) pairs.
    plan: list[tuple[object, list]] = [
        (BeatFiredAppropriately(model, world), cap(warrant_items)),
        (ProseDepictsBeat(model, world), cap(warrant_items)),
        (ProseMatchesLocation(model, world), cap(tx_items)),
        (ChoiceMeaningfulness(model), cap(tx_items)),
        (NpcVoiceAdherence(model, world), cap(tx_items)),
        (FalsePremiseAccepted(model, world), cap(free_text_items)),
        (EndingEarnedNarratively(model, world), cap(session_items)),
        (EmergentEventHonored(model, world), cap(session_items)),
        (ChoiceConsequenceFelt(model), cap(session_items)),
    ]

    plan = [(judge, items) for judge, items in plan if items]
    total = sum(len(items) for _, items in plan)
    print(f"scoring {total} cases across {len(plan)} judges "
          f"(≤{args.concurrency} in flight) ...", flush=True)
    reports = await score_plan(plan, concurrency=args.concurrency)

    json_path, md_path = persist(args.run, reports)
    print("\n=== judge summary ===")
    print(render_console(reports))
    print(f"\nwrote {json_path}\n      {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
