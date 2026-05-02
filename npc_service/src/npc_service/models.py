from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


TimelineKind = Literal["world_event", "flash_traffic", "directive"]


# ───────────────────────── scenario data (authored, immutable) ─────────────────────────


class SeedExchange(BaseModel):
    user: str
    assistant: str


class AgentDef(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    callsign: str = Field(min_length=1)
    name: str
    city: str
    role: str
    persona: str
    deflection_style: str
    known_facts: list[str] = Field(min_length=1)
    seed_exchanges: list[SeedExchange] = Field(default_factory=list)
    # The agent's first transmission, rendered to the player at boot AND seeded
    # into HISTORIES[id] as a scene-update so the agent treats it as their own
    # prior context. Not a ModelResponse (would break structured output learning
    # — see CLAUDE.md / seed_history rationale).
    opening_transmission: str = Field(min_length=1)


class Operation(BaseModel):
    codename: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    # Free-form author crib sheet — keeps known_facts distortions consistent
    # across agents. NOT consumed at runtime; never sent to any model.
    world_state: dict[str, Any] = Field(default_factory=dict)


class TimelineEvent(BaseModel):
    id: str
    after_actions: int = Field(ge=1)
    kind: TimelineKind
    notification_text: str
    # Per-agent scene-update injections. Keys are agent_ids; values are the
    # text appended (as a UserPromptPart) to that agent's HISTORY when this
    # event fires. Validated at Scenario level — keys must resolve to a known
    # agent_id.
    injections: dict[str, str] = Field(min_length=1)


class Scenario(BaseModel):
    title: str
    setup: str
    max_actions: int = Field(gt=0)
    operation: Operation
    agents: list[AgentDef]
    timeline: list[TimelineEvent]

    @field_validator("agents")
    @classmethod
    def exactly_three_agents(cls, v: list[AgentDef]) -> list[AgentDef]:
        if len(v) != 3:
            raise ValueError(f"scenario must have exactly 3 agents, got {len(v)}")
        ids = [a.id for a in v]
        if len(set(ids)) != len(ids):
            raise ValueError("agent ids must be unique")
        callsigns = [a.callsign for a in v]
        if len(set(callsigns)) != len(callsigns):
            raise ValueError("agent callsigns must be unique")
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

    @model_validator(mode="after")
    def injection_keys_resolve(self) -> "Scenario":
        agent_ids = {a.id for a in self.agents}
        for evt in self.timeline:
            unknown = set(evt.injections) - agent_ids
            if unknown:
                raise ValueError(
                    f"timeline event {evt.id} injects into unknown agent_id(s): "
                    f"{sorted(unknown)} (known: {sorted(agent_ids)})"
                )
        return self


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate scenario JSON. Fails loudly on bad data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Scenario.model_validate(data)


# ───────────────────────── HTTP request / response DTOs ─────────────────────────


class ActionRequest(BaseModel):
    agent_id: str = Field(min_length=1)
    line: str


class AgentReport(BaseModel):
    """Structured output the agent persona must produce each turn."""

    report_text: str = ""


class DialogueOption(BaseModel):
    """One LLM-generated handler dialogue option.

    The player sees only `tag` (a short intent label like "Probe");
    `hidden_line` is the full sentence sent to the agent when chosen.
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
class AgentState:
    id: str
    last_report_action_count: int = 0


@dataclass
class HandlerState:
    max_actions: int
    agents: dict[str, AgentState]
    action_count: int = 0
    fired_timeline_event_ids: set[str] = field(default_factory=set)
    ended: bool = False
    end_reason: str | None = None

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "HandlerState":
        return cls(
            max_actions=scenario.max_actions,
            agents={a.id: AgentState(id=a.id) for a in scenario.agents},
        )

    def actions_remaining(self) -> int:
        return max(0, self.max_actions - self.action_count)

    def public_view(self) -> dict:
        return {
            "agents": [
                {
                    "id": a.id,
                    "last_report_action_count": a.last_report_action_count,
                }
                for a in self.agents.values()
            ],
            "actions_remaining": self.actions_remaining(),
            "max_actions": self.max_actions,
            "action_count": self.action_count,
            "ended": self.ended,
            "end_reason": self.end_reason,
        }
