"""Structured judge verdicts — exactly one judgment + its reason.

`reason` comes first so the model commits to a rationale before the verdict
(a small but real quality lever), and gives a readable trace in the report.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PassFail(BaseModel):
    """A binary judgment. Maps to a pydantic-evals assertion (pass-rate)."""

    reason: str
    passed: bool


class Scored(BaseModel):
    """A 1–5 quality judgment. Maps to a pydantic-evals score (averaged)."""

    reason: str
    score: int = Field(ge=1, le=5)
