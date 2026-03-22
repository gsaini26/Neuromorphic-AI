"""
prism/engine.py
---------------
PrismEngine: the universal translation layer for ν-Flow 1.5.

Responsibilities
----------------
1. Parse physical units from ν-Flow source (mV, ms, V).
2. Apply the three v-Flow 1.5 Silicon-Safe design rules:
      Rule I   — Saturation Constraint  (v_min_exp floor)
      Rule II  — Multi-Scale Weight Map (8-bit safe zone)
      Rule III — Asymmetric Decay       (viscosity tuning)
3. Dispatch compiled config to the correct hardware backend.

Usage
-----
    engine = PrismEngine(target="loihi2")
    cfg = engine.map_to_hardware({
        "v_threshold": "50mV",
        "v_decay":     500,
        "v_min_exp":   9,
    })
    process = engine.compile(cfg)   # returns Lava LIF process
"""

import re
import math
import logging
from typing import Any

from prism.backends.loihi2 import Loihi2Backend
from prism.backends.akida   import AkidaBackend
from prism.utils.validators import validate_params, SiliconSafeError

logger = logging.getLogger(__name__)


# ── Hardware capability registry ─────────────────────────────────────
_HW_REGISTRY: dict[str, dict] = {
    "loihi2": {
        "v_bits":    32,
        "max_int":   2**31 - 1,
        "dv_bits":   12,          # decay mantissa width
        "weight_bits": 8,
        "floor_supported": True,
        "backend_cls": Loihi2Backend,
        "description": "Intel Loihi 2 — fixed_pt + v_bits=32",
    },
    "akida": {
        "v_bits":    8,
        "max_int":   255,
        "dv_bits":   4,
        "weight_bits": 4,
        "floor_supported": False,  # Akida clips at uint boundary
        "backend_cls": AkidaBackend,
        "description": "BrainChip Akida AKD1000 — digital 8-bit SNN",
    },
    "analog": {
        "v_bits":    None,        # continuous
        "max_int":   None,
        "dv_bits":   None,
        "weight_bits": None,
        "floor_supported": True,
        "backend_cls": None,      # PrismAnalog — future
        "description": "Analog / Memristor — voltage-native (stub)",
    },
}


class PrismEngine:
    """
    Universal ν-Flow → hardware translation engine.

    Parameters
    ----------
    target : str
        One of "loihi2", "akida", or "analog".
    safe_mode : bool
        If True (default), apply all v-Flow 1.5 Silicon-Safe constraints.
        Set False only for research/exploration.
    verbose : bool
        Emit INFO-level log messages for each translation step.
    """

    def __init__(
        self,
        target: str   = "loihi2",
        safe_mode: bool = True,
        verbose: bool   = False,
    ):
        if target not in _HW_REGISTRY:
            raise ValueError(
                f"Unknown target '{target}'. "
                f"Choose from: {list(_HW_REGISTRY.keys())}"
            )
        self.target    = target
        self.safe_mode = safe_mode
        self.hw        = _HW_REGISTRY[target]
        self._backend  = None   # lazy-init on first compile()

        if verbose:
            logging.basicConfig(level=logging.INFO)
        logger.info("PrismEngine init — target=%s safe_mode=%s", target, safe_mode)

    # ── Public API ────────────────────────────────────────────────────

    def map_to_hardware(self, vflow_params: dict[str, Any]) -> dict[str, Any]:
        """
        Translate a ν-Flow parameter dict to hardware-specific register values.

        Parameters
        ----------
        vflow_params : dict
            Keys accepted:
              v_threshold  — firing threshold, e.g. "50mV" or 50
              v_decay      — dv value 0..4095 (Loihi scale) or 0.0..1.0 fraction
              v_min_exp    — floor exponent for saturation constraint (default 9)
              w_exc        — excitatory weight (int or "10mV")
              w_inh        — inhibitory weight (int or "-100mV")
              du           — current decay, default 4095 (no decay)
              plasticity   — bool, enable on-chip STDP

        Returns
        -------
        dict  — hardware register config ready for the backend
        """
        # ── Rule I: Saturation Constraint ────────────────────────────
        v_min_exp = int(vflow_params.get("v_min_exp", 9))
        floor_val = -(2 ** v_min_exp)   # e.g. -512 for exp=9

        # ── Rule II: Multi-Scale Weight Map ──────────────────────────
        vth_raw  = self._parse_unit(vflow_params.get("v_threshold", "50mV"))
        w_exc    = self._parse_unit(vflow_params.get("w_exc", "10mV"))
        w_inh    = self._parse_unit(vflow_params.get("w_inh", "-100mV"))

        # Scale threshold to hardware integer range
        vth_hw   = self._scale_voltage(vth_raw)
        w_exc_hw = self._scale_weight(w_exc)
        w_inh_hw = self._scale_weight(w_inh)

        # ── Rule III: Asymmetric Decay ────────────────────────────────
        dv_raw   = vflow_params.get("v_decay", 500)
        dv_hw    = self._scale_decay(dv_raw)
        du_hw    = int(vflow_params.get("du", 4095))

        if self.safe_mode:
            validate_params(
                vth=vth_hw, dv=dv_hw, w_exc=w_exc_hw, w_inh=w_inh_hw,
                floor=floor_val, hw=self.hw,
            )

        mapped: dict[str, Any] = {
            # Core LIF params
            "vth":        vth_hw,
            "dv":         dv_hw,
            "du":         du_hw,
            # Weight map
            "w_exc":      w_exc_hw,
            "w_inh":      w_inh_hw,
            # Silicon-Safe floor
            "v_min_exp":  v_min_exp,
            "floor_val":  floor_val,
            # Hardware metadata
            "v_bits":     self.hw["v_bits"],
            "target":     self.target,
            # Optional: plasticity
            "plasticity": bool(vflow_params.get("plasticity", False)),
        }

        logger.info(
            "Mapped ν-Flow → %s: vth=%d dv=%d floor=%d",
            self.target, vth_hw, dv_hw, floor_val,
        )
        return mapped

    def compile(self, mapped_params: dict[str, Any]) -> Any:
        """
        Instantiate the hardware process / layer from mapped parameters.

        Returns a Lava LIF process (Loihi 2) or an Akida layer object.
        Raises NotImplementedError for the 'analog' stub target.
        """
        backend = self._get_backend()
        return backend.compile(mapped_params)

    def decay_info(self, dv: int) -> dict[str, float]:
        """
        Human-readable breakdown of a dv value.

        Returns
        -------
        dict with keys: dv, decay_factor, decay_pct, recovery_steps_from_512
        """
        k       = (4096 - dv) / 4096
        pct     = (1 - k) * 100
        # Steps to drain from -512 to 0 with no excitatory input
        steps   = math.ceil(math.log(1 / 512) / math.log(k)) if 0 < k < 1.0 else (0 if k <= 0 else float('inf'))
        return {
            "dv":                        dv,
            "decay_factor":              round(k, 6),
            "decay_pct_per_step":        round(pct, 2),
            "recovery_steps_from_512":   steps,
        }

    def weight_safe_range(self) -> dict[str, int]:
        """Return the safe weight range for the current hardware target."""
        bits = self.hw["weight_bits"]
        if bits is None:
            return {"min": None, "max": None}
        return {"min": -(2 ** (bits - 1)), "max": (2 ** (bits - 1)) - 1}

    def describe(self) -> str:
        """One-line hardware description string."""
        return self.hw["description"]

    # ── Private helpers ───────────────────────────────────────────────

    def _parse_unit(self, val: Any) -> float:
        """
        Convert a ν-Flow unit string or number to a float.

        Examples
        --------
        "50mV"  → 0.05
        "1.0V"  → 1.0
        "500ms" → 0.5    (normalised to seconds, 0..1 range)
        500     → 500.0  (dimensionless integer, returned as-is)
        """
        s = str(val).strip()
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if not nums:
            raise ValueError(f"Cannot parse unit value: '{val}'")
        num = float(nums[0])
        if "mV" in s:
            return num          # keep in mV; scale later
        if "V" in s and "mV" not in s:
            return num * 1000   # convert V → mV
        if "ms" in s:
            return num          # keep in ms
        return num              # dimensionless

    def _scale_voltage(self, mv: float) -> int:
        """Map a millivolt value to a hardware threshold integer."""
        if self.target == "loihi2":
            # Loihi 2 fixed_pt: threshold is a plain integer in the same
            # units as the weight accumulator.  1 mV → 1 unit (identity).
            return max(1, int(mv))
        elif self.target == "akida":
            # Akida 8-bit: map 0..1000 mV → 0..255
            return max(1, min(255, int(mv / 1000 * 255)))
        return int(mv)

    def _scale_weight(self, mw: float) -> int:
        """Map a millivolt weight to a hardware integer."""
        if self.target == "loihi2":
            return int(mw)      # identity for fixed_pt
        elif self.target == "akida":
            bits = self.hw["weight_bits"]
            hi   = (2 ** (bits - 1)) - 1   # +7 for 4-bit signed
            lo   = -(2 ** (bits - 1))       # -8 for 4-bit signed
            # Scale: assume ±200 mV maps to full ±range
            scaled = int(mw / 200 * hi)
            return max(lo, min(hi, scaled))
        return int(mw)

    def _scale_decay(self, dv: Any) -> int:
        """
        Normalise a decay value to the Loihi 2 fixed_pt dv integer (0..4095).

        Accepted inputs:
          int 0..4095  — used directly (Loihi fixed_pt convention)
          float 0..1   — fraction of maximum decay; converted to dv int
        """
        dv_f = float(dv)
        if 0.0 <= dv_f <= 1.0 and "." in str(dv):
            # Fraction: 0.0 = no decay (dv=0), 1.0 = max decay (dv=4095)
            dv_int = int(dv_f * 4095)
        else:
            dv_int = int(dv_f)

        dv_int = max(0, min(4095, dv_int))

        if self.target == "akida":
            # Akida 4-bit decay: map 0..4095 → 0..15
            return int(dv_int / 4095 * 15)
        return dv_int

    def _get_backend(self):
        if self._backend is None:
            cls = self.hw["backend_cls"]
            if cls is None:
                raise NotImplementedError(
                    f"Backend for target '{self.target}' is not yet implemented."
                )
            self._backend = cls()
        return self._backend
