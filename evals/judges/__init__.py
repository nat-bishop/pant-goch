"""LLM-as-judge layer (Tier 2).

Each judge scores exactly ONE thing and returns one structured verdict. Two
scopes:
  - `transition_judges` — single-transition (before→after one turn): engine
    correctness, isolating the agent at fault. The regression-test surface.
  - `session_judges` — whole-session arc properties (earned ending, consequence,
    emergent-event continuity).

Judges run on a configurable model, defaulting to local Qwen via inferno
(`judges.model.build_judge_model`). They read recorded artifacts, so a judge run
is fully decoupled from playthrough generation (`judge_run.py`).
"""
