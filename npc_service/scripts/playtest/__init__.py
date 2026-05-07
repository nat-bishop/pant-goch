"""LLM playtest harness for smallville scenarios.

Drives full sessions against the production engine + local llama with
synthetic player personas. Evaluated via `pydantic-evals` — programmatic
evaluators (TerminalReached, BeatCoverage, CanonHarvested) are the actual
quality bar; LLMJudge is included for directional smoke only.

See README.md in this directory for usage and the LLMJudge caveat.
"""
