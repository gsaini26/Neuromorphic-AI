"""
Prism — The ν-Flow Hardware Abstraction Engine
===============================================
Translates ν-Flow 1.5 "biological intent" (voltage thresholds,
decay constants, inhibitory weights) into hardware-specific
register configurations for Intel Loihi 2 and BrainChip Akida.

Quick start
-----------
>>> from prism import PrismEngine
>>> engine = PrismEngine(target="loihi2")
>>> cfg = engine.map_to_hardware({"v_threshold": "50mV", "v_decay": 500})
>>> print(cfg)
{'vth': 200, 'dv': 500, 'du': 4095, 'v_min_exp': 9, 'v_bits': 32}
"""

from prism.engine import PrismEngine
from prism.manifold.parser import VFlowParser
from prism.manifold.ast_nodes import Manifold, Cell, Flow, Stack
from prism.backends.loihi2 import Loihi2Backend
from prism.backends.akida import AkidaBackend

__version__ = "1.5.0"
__author__  = "ν-Flow Project"
__all__ = [
    "PrismEngine",
    "VFlowParser",
    "Manifold", "Cell", "Flow", "Stack",
    "Loihi2Backend", "AkidaBackend",
]
