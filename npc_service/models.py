from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


HostageStatus = Literal["held", "released", "harmed"]
EndReason = Literal[
    "budget_exhausted",
    "captor_surrendered",
    "all_released",
    "hostage_harmed",
    "auto_breach",
]
CaptorAction = Literal["release_hostage", "harm_hostage", "surrender"]


# ───────────────────────── scenario data (authored, immutable) ─────────────────────────


class SeedExchange(BaseModel):
    user: str
    assistant: str


class Meters(BaseModel):
    stress: int = Field(ge=0, le=100)
    trust: int = Field(ge=0, le=100)
    desperation: int = Field(ge=0, le=100)


class Captor(BaseModel):
    name: str
    persona: str
    deflection_style: str
    initial_demands: list[str]
    known_facts: list[str] = Field(min_length=1)
    seed_exchanges: list[SeedExchange] = Field(default_factory=list)
    starting_meters: Meters


class HostageDef(BaseModel):
    """Authored hostage definition (immutable). Live state lives in GameState."""

    name: str
    description: str


class TimelineEvent(BaseModel):
    id: str
    after_actions: int = Field(ge=1)
    kind: Literal["check_in", "medical", "ultimatum", "other"]
    notification_text: str
    injection_text: str


class Scenario(BaseModel):
    title: str
    setup: str
    max_actions: int = Field(gt=0)
    captor: Captor
    hostages: list[HostageDef]
    timeline: list[TimelineEvent]

    @field_validator("hostages")
    @classmethod
    def exactly_two_hostages(cls, v: list[HostageDef]) -> list[HostageDef]:
        if len(v) != 2:
            raise ValueError(f"scenario must have exactly 2 hostages, got {len(v)}")
        names = {h.name for h in v}
        if len(names) != len(v):
            raise ValueError("hostage names must be unique")
        return v

    @field_validator("timeline")
    @classmethod
    def timeline_3_to_4_sorted(cls, v: list[TimelineEvent]) -> list[TimelineEvent]:
        if not (3 <= len(v) <= 4):
            raise ValueError(f"timeline must have 3 or 4 events, got {len(v)}")
        ids = [e.id for e in v]
        if len(set(ids)) != len(ids):
            raise ValueError("timeline event ids must be unique")
        sorted_v = sorted(v, key=lambda e: e.after_actions)
        if [e.id for e in sorted_v] != ids:
            raise ValueError("timeline events must be sorted by after_actions")
        return v

    @model_validator(mode="after")
    def timeline_within_budget(self) -> "Scenario":
        for evt in self.timeline:
            if evt.after_actions > self.max_actions:
                raise ValueError(
                    f"timeline event {evt.id} after_actions={evt.after_actions} "
                    f"is past scenario max_actions {self.max_actions}"
                )
        return self


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate scenario JSON. Fails loudly on bad data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Scenario.model_validate(data)


# ───────────────────────── HTTP request / response DTOs ─────────────────────────


class ActionRequest(BaseModel):
    line: str


class CaptorResponse(BaseModel):
    """Structured output the captor agent must produce each turn.

    Defaults on every field so partial validation succeeds during streaming.
    Range constraints fire on the final fully-validated output.
    """

    response_text: str = ""
    stress_delta: int = Field(default=0, ge=-20, le=20)
    trust_delta: int = Field(default=0, ge=-20, le=20)
    desperation_delta: int = Field(default=0, ge=-20, le=20)
    action: Literal["release_hostage", "harm_hostage", "surrender"] | None = None
    action_target: str | None = None


class DialogueOption(BaseModel):
    """One LLM-generated player dialogue option.

    The player sees only `tag` (a short emotion/intent label like "Empathize");
    `hidden_line` is the full sentence sent to the captor when chosen.
    """

    tag: str = Field(min_length=1, max_length=24)
    hidden_line: str = Field(min_length=1, max_length=400)


class OptionsResponse(BaseModel):
    """Structured output the options agent must produce each call."""

    options: list[DialogueOption] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def _tags_distinct(self) -> "OptionsResponse":
        tags = [o.tag.strip().lower() for o in self.options]
        if len(set(tags)) != len(tags):
            raise ValueError("option tags must be distinct within a set")
        return self


# ───────────────────────── live game state (mutable, not a DTO) ─────────────────────────


@dataclass
class HostageState:
    name: str
    description: str
    status: HostageStatus = "held"


@dataclass
class GameState:
    max_actions: int
    stress: int
    trust: int
    desperation: int
    hostages: list[HostageState]
    action_count: int = 0
    fired_timeline_event_ids: set[str] = field(default_factory=set)
    ended: bool = False
    captor_outcome: str | None = None  # "surrendered" | "breached" | "harmed"
    end_reason: EndReason | None = None

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "GameState":
        return cls(
            max_actions=scenario.max_actions,
            stress=scenario.captor.starting_meters.stress,
            trust=scenario.captor.starting_meters.trust,
            desperation=scenario.captor.starting_meters.desperation,
            hostages=[HostageState(name=h.name, description=h.description) for h in scenario.hostages],
        )

    def hostage_by_name(self, name: str | None) -> HostageState | None:
        if not name:
            return None
        for h in self.hostages:
            if h.name == name:
                return h
        return None

    def actions_remaining(self) -> int:
        return max(0, self.max_actions - self.action_count)

    def public_view(self) -> dict:
        """Client-facing state. Never includes meters."""
        return {
            "hostages": [
                {"name": h.name, "description": h.description, "status": h.status}
                for h in self.hostages
            ],
            "actions_remaining": self.actions_remaining(),
            "max_actions": self.max_actions,
            "ended": self.ended,
            "captor_outcome": self.captor_outcome,
            "end_reason": self.end_reason,
        }
