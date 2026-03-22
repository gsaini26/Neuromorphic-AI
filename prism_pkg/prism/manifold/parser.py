"""
prism/manifold/parser.py
------------------------
Lexer + parser for the ν-Flow 1.5 language.

Produces a Manifold AST node from a .vf source string.

Usage
-----
    parser = VFlowParser()
    manifold = parser.parse(open("word_processor.vf").read())
    print(manifold.summary())
"""

from __future__ import annotations
import re
import logging
from typing import Iterator

from prism.manifold.ast_nodes import Manifold, Cell, Flow, Stack, PulseBlock

logger = logging.getLogger(__name__)


# ── Token definitions ────────────────────────────────────────────────

_TOKEN_SPEC: list[tuple[str, str]] = [
    ("UNIT_VAL",  r"[-+]?\d+(\.\d+)?(mV|ms|V|Hz)"),
    ("NUMBER",    r"[-+]?\d+(\.\d+)?"),
    ("BOOL",      r"\b(true|false)\b"),
    ("KEYWORD",   r"\b(manifold|cell|flow|stack|on|emit|if)\b"),
    ("RULE_KW",   r"\b(Hebbian|STDP|WTA|Oja)\b"),
    ("FLOW_TYPE", r"\b(Excitatory|Inhibitory|Plastic)\b"),
    ("PROPERTY",  r"\b(v_threshold|v_decay|v_min_exp|du|role|type|"
                  r"init_weight|source|target|rule|plasticity|"
                  r"depth|persistence)\b"),
    ("STRING",    r'"[^"]*"'),
    ("ID",        r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LBRACE",    r"\{"),
    ("RBRACE",    r"\}"),
    ("LPAREN",    r"\("),
    ("RPAREN",    r"\)"),
    ("SEMICOLON", r";"),
    ("COLON",     r":"),
    ("DOT",       r"\."),
    ("PLUS_EQ",   r"\+="),
    ("STAR",      r"\*"),
    ("ASSIGN",    r"="),
    ("SKIP",      r"[ \t\n\r]+"),
    ("COMMENT",   r"//[^\n]*"),
    ("MISMATCH",  r"."),
]

_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC),
    re.MULTILINE,
)


# ── Lexer ────────────────────────────────────────────────────────────

class LexError(SyntaxError):
    pass

class ParseError(SyntaxError):
    pass


def tokenize(source: str) -> list[tuple[str, str]]:
    """
    Convert ν-Flow source text to a flat list of (kind, value) pairs.
    Whitespace and comments are stripped.
    """
    tokens: list[tuple[str, str]] = []
    for mo in _REGEX.finditer(source):
        kind  = mo.lastgroup
        value = mo.group()
        if kind in ("SKIP", "COMMENT"):
            continue
        if kind == "MISMATCH":
            raise LexError(f"Unexpected character: {value!r}")
        tokens.append((kind, value))
    return tokens


# ── Recursive-descent parser ─────────────────────────────────────────

class VFlowParser:
    """
    Recursive-descent parser for ν-Flow 1.5.

    Grammar (simplified BNF)
    ------------------------
    program      := manifold_block*
    manifold_block := 'manifold' ID '{' manifold_body '}'
    manifold_body  := (cell_block | flow_block | stack_block | pulse_block)*
    cell_block   := 'cell' ID '{' property* '}'
    flow_block   := 'flow' ID '{' property* '}'
    stack_block  := 'stack' ID '{' property* '}'
    pulse_block  := 'on' ID '(' ID ')' '{' raw_body '}'
    property     := PROPERTY ':' value ';'
    value        := UNIT_VAL | NUMBER | BOOL | STRING | RULE_KW | FLOW_TYPE | ID
    """

    def __init__(self):
        self._tokens: list[tuple[str, str]] = []
        self._pos: int = 0

    # ── Public entry point ───────────────────────────────────────────

    def parse(self, source: str) -> Manifold:
        """
        Parse a ν-Flow source string and return the top-level Manifold node.

        Raises ParseError on syntax problems.
        """
        self._tokens = tokenize(source)
        self._pos    = 0

        manifolds = []
        while not self._at_end():
            kind, val = self._peek()
            if kind == "KEYWORD" and val == "manifold":
                manifolds.append(self._parse_manifold())
            # Top-level stack / cell / flow outside a manifold wrapper
            elif kind == "KEYWORD" and val == "stack":
                # Wrap in an anonymous manifold
                s = self._parse_stack()
                manifolds.append(Manifold(name=s.name, stacks=[s]))
            else:
                self._advance()   # skip unrecognised top-level tokens

        if not manifolds:
            raise ParseError("No manifold or stack block found in source.")
        if len(manifolds) == 1:
            return manifolds[0]

        # Multiple top-level manifolds → return first, log the rest
        logger.warning(
            "Multiple top-level blocks found; returning first (%s). "
            "Use prism.parse_all() for multi-manifold files.",
            manifolds[0].name,
        )
        return manifolds[0]

    def parse_all(self, source: str) -> list[Manifold]:
        """Like parse() but returns every manifold in the file."""
        self._tokens = tokenize(source)
        self._pos    = 0
        results = []
        while not self._at_end():
            kind, val = self._peek()
            if kind == "KEYWORD" and val == "manifold":
                results.append(self._parse_manifold())
            else:
                self._advance()
        return results

    # ── Block parsers ────────────────────────────────────────────────

    def _parse_manifold(self) -> Manifold:
        self._expect("KEYWORD", "manifold")
        _, name = self._expect("ID")
        self._expect("LBRACE")

        cells, flows, stacks, pulses = [], [], [], []

        while not self._check("RBRACE"):
            if self._at_end():
                raise ParseError(f"Unclosed manifold '{name}'")
            kind, val = self._peek()
            if kind == "KEYWORD":
                if val == "cell":
                    cells.append(self._parse_cell())
                elif val == "flow":
                    flows.append(self._parse_flow())
                elif val == "stack":
                    stacks.append(self._parse_stack())
                elif val == "on":
                    pulses.append(self._parse_pulse())
                else:
                    self._advance()
            else:
                self._advance()

        self._expect("RBRACE")
        return Manifold(name=name, cells=cells, flows=flows,
                        stacks=stacks, pulse_blocks=pulses)

    def _parse_cell(self) -> Cell:
        self._expect("KEYWORD", "cell")
        _, name = self._expect("ID")
        props   = self._parse_props_block()
        c = Cell(name=name)
        for k, v in props.items():
            if   k == "v_threshold": c.v_threshold = v
            elif k == "v_decay":     c.v_decay      = v
            elif k == "v_min_exp":   c.v_min_exp    = int(float(v))
            elif k == "du":          c.du           = int(float(v))
            elif k == "role":        c.role         = v.strip('"')
            elif k == "plasticity":  c.plasticity   = (str(v).lower() == "true")
            else:                    c.extra[k]     = v
        return c

    def _parse_flow(self) -> Flow:
        self._expect("KEYWORD", "flow")
        _, name = self._expect("ID")
        props   = self._parse_props_block()
        f = Flow(name=name)
        for k, v in props.items():
            if   k == "type":        f.flow_type   = v
            elif k == "init_weight": f.init_weight = v
            elif k == "source":      f.source      = v
            elif k == "target":      f.target      = v
            elif k == "rule":        f.rule        = v
            else:                    f.extra[k]    = v
        return f

    def _parse_stack(self) -> Stack:
        self._expect("KEYWORD", "stack")
        _, name = self._expect("ID")
        props   = self._parse_props_block()
        s = Stack(name=name)
        for k, v in props.items():
            if   k == "depth":       s.depth       = int(float(v))
            elif k == "persistence": s.persistence = v
            elif k == "rule":        s.rule        = v
            elif k == "plasticity":  s.plasticity  = (str(v).lower() == "true")
            else:                    s.extra[k]    = v
        return s

    def _parse_pulse(self) -> PulseBlock:
        self._expect("KEYWORD", "on")
        self._expect("ID")         # "Pulse" keyword-as-identifier
        self._expect("LPAREN")
        _, signal = self._expect("ID")
        self._expect("RPAREN")
        body = self._collect_braced_body()
        return PulseBlock(signal=signal, body_text=body)

    # ── Property block helper ────────────────────────────────────────

    def _parse_props_block(self) -> dict[str, str]:
        """Parse { key: value; ... } and return as dict of strings."""
        self._expect("LBRACE")
        props: dict[str, str] = {}
        while not self._check("RBRACE"):
            if self._at_end():
                raise ParseError("Unclosed property block")
            kind, key = self._peek()
            if kind not in ("PROPERTY", "ID"):
                self._advance()
                continue
            self._advance()
            self._expect("COLON")
            val = self._parse_value()
            # Optional semicolon
            if self._check("SEMICOLON"):
                self._advance()
            props[key] = val
        self._expect("RBRACE")
        return props

    def _parse_value(self) -> str:
        """Consume one value token and return its string representation."""
        kind, val = self._peek()
        if kind in ("UNIT_VAL", "NUMBER", "BOOL", "STRING",
                    "RULE_KW", "FLOW_TYPE", "ID", "PROPERTY"):
            self._advance()
            return val
        raise ParseError(f"Expected a value, got ({kind}, {val!r})")

    def _collect_braced_body(self) -> str:
        """
        Consume everything between { and the matching } as raw text.
        Used for PulseBlock bodies that we don't fully parse yet.
        """
        self._expect("LBRACE")
        depth   = 1
        parts: list[str] = []
        while depth > 0:
            if self._at_end():
                raise ParseError("Unclosed pulse block body")
            kind, val = self._advance()
            if kind == "LBRACE":
                depth += 1
            elif kind == "RBRACE":
                depth -= 1
                if depth == 0:
                    break
            parts.append(val)
        return " ".join(parts)

    # ── Token stream helpers ─────────────────────────────────────────

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._pos]

    def _advance(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _at_end(self) -> bool:
        return self._pos >= len(self._tokens)

    def _check(self, kind: str, value: str | None = None) -> bool:
        if self._at_end():
            return False
        k, v = self._peek()
        return k == kind and (value is None or v == value)

    def _expect(self, kind: str, value: str | None = None) -> tuple[str, str]:
        if self._at_end():
            raise ParseError(
                f"Unexpected end of input, expected ({kind}, {value!r})"
            )
        k, v = self._advance()
        if k != kind or (value is not None and v != value):
            raise ParseError(
                f"Expected ({kind}, {value!r}), got ({k}, {v!r})"
            )
        return k, v
