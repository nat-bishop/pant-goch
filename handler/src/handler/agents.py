"""LLM agents. Three stateless one-shot pydantic-ai agents:

- `SOP_AGENT` — State-of-Play memo each cycle, Operations voice (str output).
- `DRAFTER_AGENT` — 3-4 candidate options per cycle, structured `OptionSet`
  output, Operations voice.
- `RENDERER_AGENT` — field dispatch in the acting agent's voice (str output).

All three are stateless — no pydantic-ai `message_history` is passed between
cycles. Each cycle is a fresh prompt; continuity is preserved by explicitly
passing recent dispatches as user-side context. See HANDLER_SPEC_V2 §8.4.

Authored in Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentBundle:
    """Phase 2 fills with the three pydantic-ai agent instances."""

    pass
