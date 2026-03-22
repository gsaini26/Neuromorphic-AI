"""
prism/manifold/ast_nodes.py
---------------------------
Dataclasses representing the Abstract Syntax Tree (AST) produced
by the ν-Flow parser.  Each class mirrors one ν-Flow keyword block.

Hierarchy
---------
  Manifold
  ├── Cell        (neuron definition)
  ├── Flow        (synapse / connection)
  ├── Stack       (context resonator)
  └── PulseBlock  (on Pulse { ... } handler)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Cell:
    """
    Represents a `cell { }` block — a single LIF neuron definition.

    Attributes
    ----------
    name         : identifier, e.g. "LanguageNeuron_A"
    v_threshold  : firing threshold (mV string or int)
    v_decay      : dv parameter (int 0..4095)
    v_min_exp    : saturation floor exponent (int, default 9 → −512)
    du           : current decay (int 0..4095, default 4095 = no decay)
    role         : semantic label, e.g. "meaning" or "context"
    plasticity   : whether STDP is enabled on this cell
    extra        : any unrecognised key-value pairs for forward compat
    """
    name:        str
    v_threshold: Any   = "50mV"
    v_decay:     Any   = 500
    v_min_exp:   int   = 9
    du:          int   = 4095
    role:        str   = ""
    plasticity:  bool  = False
    extra:       dict  = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Cell({self.name!r}, vth={self.v_threshold}, "
            f"dv={self.v_decay}, role={self.role!r})"
        )


@dataclass
class Flow:
    """
    Represents a `flow { }` block — a synapse or connection.

    Attributes
    ----------
    name         : identifier, e.g. "InhibitoryLink"
    flow_type    : "Excitatory" | "Inhibitory"
    init_weight  : initial synaptic weight (mV string or int)
    source       : name of the source Cell (may be empty for fan-in)
    target       : name of the target Cell
    rule         : learning rule, e.g. "Hebbian", "WTA", "STDP"
    extra        : forward-compat overflow
    """
    name:        str
    flow_type:   str  = "Excitatory"
    init_weight: Any  = "10mV"
    source:      str  = ""
    target:      str  = ""
    rule:        str  = ""
    extra:       dict = field(default_factory=dict)

    @property
    def is_inhibitory(self) -> bool:
        return self.flow_type.lower() == "inhibitory"

    def __repr__(self) -> str:
        arrow = "⊣" if self.is_inhibitory else "→"
        return f"Flow({self.name!r}, {self.source!r} {arrow} {self.target!r}, w={self.init_weight})"


@dataclass
class Stack:
    """
    Represents a `stack { }` block — a context resonator / memory buffer.

    Attributes
    ----------
    name        : identifier, e.g. "ContextResonator"
    depth       : number of time-steps retained (int)
    persistence : how long a memory voltage stays active (ms string or int)
    rule        : plasticity rule, e.g. "Hebbian"
    plasticity  : whether on-chip learning is enabled
    extra       : forward-compat overflow
    """
    name:        str
    depth:       int  = 4096
    persistence: Any  = "500ms"
    rule:        str  = "Hebbian"
    plasticity:  bool = True
    extra:       dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Stack({self.name!r}, depth={self.depth}, rule={self.rule!r})"


@dataclass
class PulseBlock:
    """
    Represents an `on Pulse(signal) { }` block — the execution logic.

    Attributes
    ----------
    signal    : name of the input variable, e.g. "input_token"
    body_text : raw source text of the block body (for now stored verbatim)
    """
    signal:    str = "input_signal"
    body_text: str = ""

    def __repr__(self) -> str:
        lines = len(self.body_text.strip().splitlines())
        return f"PulseBlock(signal={self.signal!r}, {lines} line(s))"


@dataclass
class Manifold:
    """
    Top-level ν-Flow manifold AST node.

    Contains collections of Cells, Flows, Stacks, and PulseBlocks.

    Attributes
    ----------
    name         : manifold identifier, e.g. "WordProcessor"
    cells        : list of Cell nodes
    flows        : list of Flow nodes
    stacks       : list of Stack nodes
    pulse_blocks : list of PulseBlock nodes
    """
    name:         str
    cells:        list[Cell]       = field(default_factory=list)
    flows:        list[Flow]       = field(default_factory=list)
    stacks:       list[Stack]      = field(default_factory=list)
    pulse_blocks: list[PulseBlock] = field(default_factory=list)

    # ── Convenience accessors ────────────────────────────────────────

    def get_cell(self, name: str) -> Cell | None:
        return next((c for c in self.cells if c.name == name), None)

    def get_flow(self, name: str) -> Flow | None:
        return next((f for f in self.flows if f.name == name), None)

    @property
    def inhibitory_flows(self) -> list[Flow]:
        return [f for f in self.flows if f.is_inhibitory]

    @property
    def excitatory_flows(self) -> list[Flow]:
        return [f for f in self.flows if not f.is_inhibitory]

    def summary(self) -> str:
        return (
            f"Manifold '{self.name}': "
            f"{len(self.cells)} cell(s), "
            f"{len(self.flows)} flow(s) "
            f"({len(self.inhibitory_flows)} inhibitory), "
            f"{len(self.stacks)} stack(s)"
        )

    def __repr__(self) -> str:
        return self.summary()
