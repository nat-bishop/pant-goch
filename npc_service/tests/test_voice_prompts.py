"""Unit tests for the voice/prompt scaffolding added on top of the dossier:
voice_examples, length_budget_words, lexical_budget + lexical_history_snapshot,
subtextual_motif rendering."""
from __future__ import annotations

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from npc_service.agents import (
    LEXICAL_HISTORY_WINDOW,
    _count_token,
    _format_dossier,
    lexical_history_snapshot,
    resolve_length_budget,
)
from npc_service.models import (
    BeatSpec,
    LexicalBudgetEntry,
    NPCDossier,
)


# ───────────────────────── _count_token ─────────────────────────


def test_count_token_word_boundary():
    """Whole-word match — 'partner' should not match inside 'partnership'."""
    assert _count_token("we are partners in this", "partner") == 1
    # Suffix-greedy: 'partner' DOES match 'partnership' because of \w*
    assert _count_token("our partnership endures", "partner") == 1
    # But the prefix is anchored — 'artner' alone matches nothing
    assert _count_token("our partnership endures", "artner") == 0


def test_count_token_suffix_greedy():
    """'calibration' catches 'calibrations' but NOT 'calibrating' — the
    suffix-greedy match only extends past the literal stem, not into a
    different inflection. That's a deliberate floor: stemming would need
    a NLP lib we don't pull in for one validator."""
    body = "the calibration was logged. additional calibrations followed. calibrating now."
    assert _count_token(body, "calibration") == 2


def test_count_token_zero():
    assert _count_token("nothing relevant here", "partnership") == 0


# ───────────────────────── lexical_history_snapshot ─────────────────────────


def _aria_with_budgets() -> NPCDossier:
    return NPCDossier(
        id="aria",
        display_name="Aria",
        email_address="aria@x",
        lexical_budget=[
            LexicalBudgetEntry(token="partnership", max_per_email=1, max_per_window=3),
            LexicalBudgetEntry(token="calibration", max_per_email=2, max_per_window=5),
        ],
    )


def _resp(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _req(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def test_snapshot_no_budgets_returns_empty():
    """An NPC without lexical_budget should never trigger the walk."""
    npc = NPCDossier(id="x", display_name="X", email_address="x@x")
    assert lexical_history_snapshot([_resp("partnership partnership")], npc) == {}


def test_snapshot_counts_npc_responses_only():
    """Player turns (ModelRequest) should NOT contribute to the count —
    the budget polices the NPC's vocabulary, not the player's."""
    npc = _aria_with_budgets()
    history = [
        _resp("our partnership begins"),
        _req("partnership partnership partnership"),  # player saying it — ignored
        _resp("calibration noted"),
    ]
    counts = lexical_history_snapshot(history, npc)
    assert counts == {"partnership": 1, "calibration": 1}


def test_snapshot_respects_window_size():
    """Only the last LEXICAL_HISTORY_WINDOW NPC messages count."""
    npc = _aria_with_budgets()
    # Make WINDOW + 2 NPC messages each containing 'partnership' once.
    history = [_resp("partnership") for _ in range(LEXICAL_HISTORY_WINDOW + 2)]
    counts = lexical_history_snapshot(history, npc)
    # Only the last WINDOW count.
    assert counts["partnership"] == LEXICAL_HISTORY_WINDOW


def test_snapshot_walks_oldest_to_newest_within_window():
    """Even with a player turn interleaved, the window only consumes NPC msgs."""
    npc = _aria_with_budgets()
    history = [
        _resp("calibration calibration calibration"),  # outside the window
        _resp("partnership"),  # outside the window
        _req("..."),
        _resp("calibration"),
        _resp("calibration"),
        _resp("partnership"),
        _req("..."),
        _resp("calibration"),
        _resp("partnership"),
    ]
    # Last 5 NPC messages: calibration, calibration, partnership, calibration, partnership
    counts = lexical_history_snapshot(history, npc)
    assert counts == {"partnership": 2, "calibration": 3}


def test_snapshot_zero_when_token_unused():
    npc = _aria_with_budgets()
    history = [_resp("nothing relevant here") for _ in range(3)]
    counts = lexical_history_snapshot(history, npc)
    # Both keyed at zero — useful for tests, harmless in the prompt block.
    assert counts == {"partnership": 0, "calibration": 0}


# ───────────────────────── resolve_length_budget ─────────────────────────


def test_resolve_uses_dossier_default_when_no_directive():
    npc = NPCDossier(id="x", display_name="X", email_address="x@x", length_budget_words=70)
    assert resolve_length_budget(npc, None) == 70


def test_resolve_dossier_default_when_directive_no_override():
    npc = NPCDossier(id="x", display_name="X", email_address="x@x", length_budget_words=70)
    spec = BeatSpec(intent="x")  # length_budget_words=None
    assert resolve_length_budget(npc, spec) == 70


def test_resolve_beat_override_wins():
    npc = NPCDossier(id="x", display_name="X", email_address="x@x", length_budget_words=70)
    spec = BeatSpec(intent="x", length_budget_words=200)
    assert resolve_length_budget(npc, spec) == 200


# ───────────────────────── _format_dossier ─────────────────────────


def test_format_dossier_includes_length_budget():
    npc = NPCDossier(id="x", display_name="X", email_address="x@x", length_budget_words=80)
    out = _format_dossier(npc, length_budget_words=80)
    assert "LENGTH BUDGET: 80 words" in out


def test_format_dossier_voice_examples_block_when_authored():
    npc = NPCDossier(
        id="x", display_name="X", email_address="x@x",
        voice_examples=[
            "First example body.",
            "Second example body.",
        ],
    )
    out = _format_dossier(npc, length_budget_words=80)
    assert "VOICE EXAMPLES" in out
    assert "--- example 1 ---" in out
    assert "First example body." in out
    assert "--- example 2 ---" in out
    assert "Second example body." in out


def test_format_dossier_no_voice_examples_block_when_empty():
    npc = NPCDossier(id="x", display_name="X", email_address="x@x")
    out = _format_dossier(npc, length_budget_words=80)
    assert "VOICE EXAMPLES" not in out


def test_format_dossier_lexical_history_hint_only_when_hot():
    """Hint surfaces only for tokens at-or-over max_per_window — quiet
    tokens stay off the prompt to keep the dossier block tight."""
    npc = _aria_with_budgets()
    # partnership at-cap (3); calibration under (2 < 5).
    out = _format_dossier(
        npc,
        length_budget_words=80,
        lexical_history={"partnership": 3, "calibration": 2},
    )
    assert "LEXICAL HISTORY" in out
    assert "'partnership'" in out
    assert "'calibration'" not in out  # under cap → omitted


def test_format_dossier_no_lexical_block_when_all_quiet():
    npc = _aria_with_budgets()
    out = _format_dossier(
        npc,
        length_budget_words=80,
        lexical_history={"partnership": 0, "calibration": 1},
    )
    assert "LEXICAL HISTORY" not in out
