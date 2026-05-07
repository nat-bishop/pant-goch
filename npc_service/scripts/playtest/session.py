"""Session record — the output of one playtest run.

Holds enough information for evaluators to decide pass/fail and for the
report to dump everything to JSON. Mirrors the engine's TickDecision shape
where it overlaps so the future ScenarioGraph overlay can consume the same
JSON.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from npc_service.models import Email, InboxState, TickDecision


@dataclass
class SessionRecord:
    """Output of one playtest session."""

    persona_name: str
    transcript: list[Email]
    tick_decisions: list[TickDecision]
    final_state: InboxState
    error: str | None = None  # None on clean session; populated on chat-agent failure or persona timeout

    def reached_terminal(self) -> str | None:
        """The id of the terminal beat that ended the session, or None."""
        if not self.final_state.ended:
            return None
        # `end_reason` carries the terminal id when the engine ends a session.
        return self.final_state.end_reason

    def fired_beats(self) -> list[str]:
        return sorted(self.final_state.fired_beat_ids)

    def canon_keys(self) -> list[str]:
        return sorted(self.final_state.player_canon.keys())

    def to_json(self) -> dict:
        """Lossy serialization for archival. Engine objects are dataclasses
        so `asdict` mostly works; Pydantic models (TickDecision) round-trip
        via model_dump."""
        return {
            "persona_name": self.persona_name,
            "transcript": [asdict(e) for e in self.transcript],
            "tick_decisions": [d.model_dump() for d in self.tick_decisions],
            "final_state": {
                "qualities": dict(self.final_state.qualities),
                "fired_beat_ids": sorted(self.final_state.fired_beat_ids),
                "player_canon": {
                    k: {
                        "summary": v.summary,
                        "salience": v.salience,
                        "category": v.category.value,
                        "turn_of_origin": v.turn_of_origin,
                    }
                    for k, v in self.final_state.player_canon.items()
                },
                "ended": self.final_state.ended,
                "end_reason": self.final_state.end_reason,
            },
            "error": self.error,
        }
