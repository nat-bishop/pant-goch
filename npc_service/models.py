from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


HostageStatus = Literal["held", "released", "harmed"]
EndReason = Literal[
    "timer_expired",
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
    at_seconds: int = Field(ge=0)
    kind: Literal["check_in", "medical", "ultimatum", "other"]
    notification_text: str
    injection_text: str


class Scenario(BaseModel):
    title: str
    setup: str
    duration_seconds: int = Field(gt=0)
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
        sorted_v = sorted(v, key=lambda e: e.at_seconds)
        if [e.id for e in sorted_v] != ids:
            raise ValueError("timeline events must be sorted by at_seconds")
        return v

    @model_validator(mode="after")
    def timeline_within_duration(self) -> "Scenario":
        for evt in self.timeline:
            if evt.at_seconds >= self.duration_seconds:
                raise ValueError(
                    f"timeline event {evt.id} at_seconds={evt.at_seconds} "
                    f"is past scenario duration {self.duration_seconds}"
                )
        return self


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate scenario JSON. Fails loudly on bad data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Scenario.model_validate(data)


# ───────────────────────── HTTP request / response DTOs ─────────────────────────


class ActionRequest(BaseModel):
    type: Literal["offer", "refuse", "empathize", "threaten"]
    detail: str = ""
    line: str


class PlayerOption(BaseModel):
    """One generated dialogue option the player can pick.

    `line` is shown verbatim to the player and sent verbatim to the captor.
    `action_type` + `detail` are the hidden tag the server uses to drive
    meter deltas, mirroring the four hardcoded action types of the prior UI.
    """

    line: str
    action_type: Literal["offer", "refuse", "empathize", "threaten"]
    detail: str = ""


class OptionsResponse(BaseModel):
    options: list[PlayerOption] = Field(min_length=3, max_length=3)


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


# ───────────────────────── live game state (mutable, not a DTO) ─────────────────────────


@dataclass
class HostageState:
    name: str
    description: str
    status: HostageStatus = "held"


@dataclass
class GameState:
    scenario_start: datetime
    duration_seconds: int
    stress: int
    trust: int
    desperation: int
    hostages: list[HostageState]
    ended: bool = False
    captor_outcome: str | None = None  # "surrendered" | "breached" | "harmed"
    end_reason: EndReason | None = None
    last_event_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "GameState":
        return cls(
            scenario_start=datetime.now(timezone.utc),
            duration_seconds=scenario.duration_seconds,
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

    def time_remaining_seconds(self) -> int:
        elapsed = (datetime.now(timezone.utc) - self.scenario_start).total_seconds()
        return max(0, int(self.duration_seconds - elapsed))

    def public_view(self) -> dict:
        """Client-facing state. Never includes meters."""
        return {
            "hostages": [
                {"name": h.name, "description": h.description, "status": h.status}
                for h in self.hostages
            ],
            "time_remaining_seconds": self.time_remaining_seconds(),
            "ended": self.ended,
            "captor_outcome": self.captor_outcome,
            "end_reason": self.end_reason,
        }
