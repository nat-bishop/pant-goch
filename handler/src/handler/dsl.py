"""String-form shorthand for authoring predicates and effects.

Author-facing surface — every predicate and every effect in a mission YAML
may be written as a short string instead of a tagged-union dict. The loader
runs this module's parsers once, before pydantic, and produces the same
canonical dicts the engine has always consumed. There is no runtime cost
beyond YAML load.

PREDICATE GRAMMAR (precedence low → high):

    expr        := or_expr
    or_expr     := and_expr ( 'or' and_expr )*
    and_expr    := unary_expr ( 'and' unary_expr )*
    unary_expr  := 'not' unary_expr | atom
    atom        := '(' expr ')' | comparison | flag
    comparison  := qref OP value | qref 'in' number '..' number
    flag        := 'event_fired' ':' ident
    OP          := '>=' | '<=' | '==' | '=' | '>' | '<'
    value       := number | quoted-string | 'true' | 'false'
    qref        := quality reference, e.g. agent[LYNX].cover, mission.stage

Examples:
    "agent[LYNX].cover > 0"
    "mission.extraction_stage == 1"
    "agent[KESTREL].cover in 1..5"
    "(extraction_stage >= 2 and agent[LYNX].cover > 0)"
    "not event_fired:kestrel_sacrifice"
    "mission.northern_lanes_open == true"

EFFECT GRAMMAR:

    effect      := qref OP value [ 'if' expr ]
    OP          := '=' | '+=' | '-=' | 'max=' | 'min='

Examples:
    "agent[LYNX].cover -= 2"
    "mission.extraction_stage = 2"
    "agent[$agent].cover_protected = true"
    "agent[LYNX].cover -= 4 if agent[LYNX].cover < 5"

The output dicts are exactly what the existing tagged-union YAML form would
parse to — the rest of the engine (predicate evaluation, effect application,
substitution) needs no changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class DSLParseError(ValueError):
    """A predicate or effect string failed to parse. Surfaces with the original
    source text so the author can see what tripped the parser."""

    def __init__(self, msg: str, source: str, pos: int = -1) -> None:
        if pos >= 0:
            indicator = " " * pos + "^"
            super().__init__(f"{msg}\n  {source}\n  {indicator}")
        else:
            super().__init__(f"{msg}\n  {source}")
        self.source = source
        self.pos = pos


# ───────────────────────── tokenizer ─────────────────────────


@dataclass
class _Token:
    kind: str
    value: str
    pos: int


# Order matters: longer / more-specific patterns first. KEYWORD must come
# before QREF so `and`/`or`/`not`/`true`/`false`/`event_fired`
# don't get swallowed by the qref pattern.
_TOKEN_SPEC: list[tuple[str, str]] = [
    ("WS", r"\s+"),
    ("NUMBER", r"-?\d+\.\d+|-?\d+"),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("RANGE", r"\.\."),
    # Effect ops first so '+=' / '-=' / 'max=' / 'min=' don't tokenize as '+'/'-'/etc.
    ("EOP", r"\+=|-=|max=|min="),
    # Predicate / assignment ops.
    ("OP", r">=|<=|==|=|>|<"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COLON", r":"),
    (
        "KEYWORD",
        r"\b(?:and|or|not|in|if|true|false|event_fired)\b",
    ),
    # Quality reference: identifier with optional bracketed qualifiers / dotted
    # path. `$` allowed as the leading sigil for placeholders ($agent, $pays).
    ("QREF", r"\$?[A-Za-z_][A-Za-z0-9_]*(?:\[[A-Za-z0-9_*$]+\]|\.[A-Za-z_][A-Za-z0-9_]*)*"),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC)
)


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise DSLParseError(f"unexpected character {source[pos]!r}", source, pos)
        kind = m.lastgroup or ""
        if kind != "WS":
            tokens.append(_Token(kind=kind, value=m.group(), pos=pos))
        pos = m.end()
    return tokens


# ───────────────────────── parser ─────────────────────────


class _Parser:
    def __init__(self, tokens: list[_Token], source: str) -> None:
        self.tokens = tokens
        self.source = source
        self.i = 0

    # --- token primitives ---

    def peek(self) -> _Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def advance(self) -> _Token:
        if self.i >= len(self.tokens):
            raise DSLParseError("unexpected end of input", self.source, len(self.source))
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> _Token:
        tok = self.peek()
        if tok is None:
            raise DSLParseError(f"expected {kind}, got end of input", self.source, len(self.source))
        if tok.kind != kind or (value is not None and tok.value != value):
            wanted = f"{kind}={value!r}" if value else kind
            raise DSLParseError(f"expected {wanted}, got {tok.kind}={tok.value!r}", self.source, tok.pos)
        return self.advance()

    def consume(self, kind: str, value: str | None = None) -> _Token | None:
        tok = self.peek()
        if tok is None:
            return None
        if tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        return self.advance()

    def at_end(self) -> bool:
        return self.i >= len(self.tokens)

    # --- top-level entry points ---

    def parse_predicate_full(self) -> dict:
        result = self._parse_or()
        if not self.at_end():
            tok = self.peek()
            assert tok is not None
            raise DSLParseError(
                f"unexpected trailing tokens after predicate (next: {tok.value!r})",
                self.source,
                tok.pos,
            )
        return result

    def parse_effect_full(self) -> dict:
        result = self._parse_effect()
        if not self.at_end():
            tok = self.peek()
            assert tok is not None
            raise DSLParseError(
                f"unexpected trailing tokens after effect (next: {tok.value!r})",
                self.source,
                tok.pos,
            )
        return result

    # --- predicate productions ---

    def _parse_or(self) -> dict:
        left = self._parse_and()
        operands = [left]
        while self.consume("KEYWORD", "or"):
            operands.append(self._parse_and())
        if len(operands) == 1:
            return operands[0]
        return {"kind": "or", "of": operands}

    def _parse_and(self) -> dict:
        left = self._parse_unary()
        operands = [left]
        while self.consume("KEYWORD", "and"):
            operands.append(self._parse_unary())
        if len(operands) == 1:
            return operands[0]
        return {"kind": "and", "of": operands}

    def _parse_unary(self) -> dict:
        if self.consume("KEYWORD", "not"):
            return {"kind": "not", "of": self._parse_unary()}
        return self._parse_atom()

    def _parse_atom(self) -> dict:
        if self.consume("LPAREN"):
            inner = self._parse_or()
            self.expect("RPAREN")
            return inner
        # `event_fired:foo`
        if (kw := self.consume("KEYWORD", "event_fired")) is not None:
            self.expect("COLON")
            ident = self._parse_identifier(after=kw)
            return {"kind": "event_fired", "event": ident}
        # comparison: qref OP value  OR  qref in low..high
        return self._parse_comparison()

    def _parse_identifier(self, after: _Token) -> str:
        """Read a single identifier token (used for event names after the
        colon). Accepts QREF or KEYWORD/identifier shapes since some valid
        ids might overlap (e.g. event named 'turning_point')."""
        tok = self.peek()
        if tok is None:
            raise DSLParseError("expected identifier", self.source, after.pos + len(after.value))
        if tok.kind not in ("QREF", "KEYWORD"):
            raise DSLParseError(
                f"expected identifier, got {tok.kind}={tok.value!r}",
                self.source,
                tok.pos,
            )
        return self.advance().value

    def _parse_comparison(self) -> dict:
        qref = self.expect("QREF").value
        # Range: `qref in low..high` → And_(Gte(low), Lte(high))
        if self.consume("KEYWORD", "in"):
            low = self._parse_number_value()
            self.expect("RANGE")
            high = self._parse_number_value()
            return {
                "kind": "and",
                "of": [
                    {"kind": "quality_gte", "quality": qref, "value": low},
                    {"kind": "quality_lte", "quality": qref, "value": high},
                ],
            }
        op = self.expect("OP").value
        value = self._parse_value()
        kind_map = {
            ">=": "quality_gte",
            "<=": "quality_lte",
            ">":  "quality_gt",
            "<":  "quality_lt",
            "==": "quality_equals",
            "=":  "quality_equals",
        }
        kind = kind_map[op]
        return {"kind": kind, "quality": qref, "value": value}

    def _parse_value(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise DSLParseError("expected value", self.source, len(self.source))
        if tok.kind == "NUMBER":
            return self._parse_number_value()
        if tok.kind == "STRING":
            self.advance()
            # Strip the surrounding double-quotes; honour basic backslash escapes.
            raw = tok.value[1:-1]
            return raw.encode().decode("unicode_escape")
        if tok.kind == "KEYWORD" and tok.value in ("true", "false"):
            self.advance()
            return tok.value == "true"
        # Bare identifier as a string value (handy for enum-ish QualityEquals).
        if tok.kind == "QREF":
            self.advance()
            return tok.value
        raise DSLParseError(
            f"expected value, got {tok.kind}={tok.value!r}",
            self.source,
            tok.pos,
        )

    def _parse_number_value(self) -> int | float:
        tok = self.expect("NUMBER")
        if "." in tok.value:
            return float(tok.value)
        return int(tok.value)

    # --- effect production ---

    def _parse_effect(self) -> dict:
        qref = self.expect("QREF").value
        op_tok = self.peek()
        if op_tok is None:
            raise DSLParseError("expected effect operator", self.source, len(self.source))
        if op_tok.kind == "OP" and op_tok.value == "=":
            self.advance()
            op = "set"
        elif op_tok.kind == "EOP":
            self.advance()
            if op_tok.value == "+=":
                op = "add"
            elif op_tok.value == "-=":
                op = "add"  # negative add — value is negated below
            elif op_tok.value == "max=":
                op = "max"
            elif op_tok.value == "min=":
                op = "min"
            else:  # pragma: no cover — guarded by tokenizer pattern
                raise DSLParseError(f"unknown effect op {op_tok.value!r}", self.source, op_tok.pos)
        else:
            raise DSLParseError(
                f"expected effect operator (=, +=, -=, max=, min=), got {op_tok.value!r}",
                self.source,
                op_tok.pos,
            )

        value = self._parse_value()
        if op_tok.kind == "EOP" and op_tok.value == "-=" and isinstance(value, (int, float)):
            value = -value

        out: dict = {"quality": qref, "value": value, "op": op}

        # Optional `if <predicate>` guard.
        if self.consume("KEYWORD", "if"):
            out["when"] = self._parse_or()
        return out


# ───────────────────────── public API ─────────────────────────


def parse_predicate_string(source: str) -> dict:
    """Parse a string-form predicate to a tagged-union dict. Pydantic
    validates the result against the existing `Predicate` model — there is
    no DSL-specific runtime path."""
    tokens = _tokenize(source)
    if not tokens:
        raise DSLParseError("empty predicate string", source, 0)
    return _Parser(tokens, source).parse_predicate_full()


def parse_effect_string(source: str) -> dict:
    """Parse a string-form effect to a dict suitable for `Effect` validation."""
    tokens = _tokenize(source)
    if not tokens:
        raise DSLParseError("empty effect string", source, 0)
    return _Parser(tokens, source).parse_effect_full()


# ───────────────────────── YAML pre-pass ─────────────────────────
#
# Walks a loaded mission dict and converts string-form predicates / effects
# into their tagged-union equivalents. Called by `mission_loader.py` before
# `Mission.model_validate`.

# Field names whose value is (or contains) a Predicate.
_PREDICATE_FIELDS = {"when"}
# Field names whose value is a list of Effect.
_EFFECT_LIST_FIELDS = {"effects"}


def normalise_mission_dsl(raw: dict) -> dict:
    """Recursively walk `raw` and rewrite string-form predicates / effects
    into their tagged-union dict form. Returns a new dict; `raw` is not
    mutated. Non-string values are passed through unchanged so an author can
    mix shorthand and the explicit form freely."""
    return _normalise(raw)


def _normalise(node: Any, parent_key: str | None = None) -> Any:
    if isinstance(node, str) and parent_key in _PREDICATE_FIELDS:
        return parse_predicate_string(node)
    if isinstance(node, dict):
        return {key: _normalise(value, parent_key=key) for key, value in node.items()}
    if isinstance(node, list):
        if parent_key in _EFFECT_LIST_FIELDS:
            return [_normalise_effect_item(item) for item in node]
        return [_normalise(item, parent_key=parent_key) for item in node]
    return node


def _normalise_effect_item(item: Any) -> Any:
    if isinstance(item, str):
        return parse_effect_string(item)
    if isinstance(item, dict):
        # Inside an effect dict, the `when:` field may itself be a shorthand
        # predicate string — recurse so it gets parsed.
        return {key: _normalise(value, parent_key=key) for key, value in item.items()}
    return item
