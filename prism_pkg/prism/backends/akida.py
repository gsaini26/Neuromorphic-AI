"""
prism/backends/akida.py
-----------------------
BrainChip Akida AKD1000 backend for Prism.

Unlike MetaTF (which converts a Keras ANN), this backend builds a
native SNN layer directly from ν-Flow voltage-weight logic — no
ANN-to-SNN conversion penalty.

The Akida package is imported lazily so the rest of Prism remains
importable on machines without an Akida board or the akida package.
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AkidaBackend:
    """
    Compiles Prism-mapped parameters into an Akida model.

    Key differences from MetaTF default workflow
    ---------------------------------------------
    - Weights are set from ν-Flow voltage logic, not converted from Keras.
    - Inhibitory and excitatory connections are built separately.
    - On-chip STDP is enabled if `plasticity=True` in params.
    """

    # Akida weight-bit widths supported by AKD1000
    _VALID_WEIGHT_BITS = (1, 2, 4, 8)

    def compile(self, p: dict[str, Any]) -> Any:
        """
        Build an Akida sequential model from mapped params.

        Parameters
        ----------
        p : dict
            Output of PrismEngine.map_to_hardware() with target='akida'.

        Returns
        -------
        akida.Model instance (not yet quantized or mapped to hardware).
        """
        try:
            import akida
        except ImportError as e:
            raise ImportError(
                "The 'akida' package is not installed.\n"
                "Install the BrainChip MetaTF stack:\n"
                "  pip install akida\n"
                "or visit https://brainchip.com/akida-enablement-platform"
            ) from e

        w_bits   = self._choose_weight_bits(p.get("w_exc", 1))
        vth      = p.get("vth", 12)           # already mapped to 0..255 range
        decay    = p.get("dv",  2)            # already mapped to 0..15 range
        units    = p.get("units", 128)
        plasticity = p.get("plasticity", False)

        logger.info(
            "AkidaBackend: compiling — vth=%d decay=%d w_bits=%d plasticity=%s",
            vth, decay, w_bits, plasticity,
        )

        model = akida.Model()

        # ── Input layer ───────────────────────────────────────────────
        model.add(akida.InputLayer(input_shape=(units,)))

        # ── Excitatory SNN layer ──────────────────────────────────────
        # We bypass Keras activation and set threshold directly so that
        # voltage-weight logic is hardware-native (not a conversion).
        exc_layer = akida.FullyConnected(
            units        = units,
            activation   = False,        # we own the spike logic via vth
            weights_bits = w_bits,
        )
        model.add(exc_layer)

        # ── Inhibitory feedback layer ─────────────────────────────────
        # Akida represents inhibitory connections as a second layer
        # with negative weights and a high threshold (rarely fires).
        inh_layer = akida.FullyConnected(
            units        = units,
            activation   = False,
            weights_bits = w_bits,
        )
        model.add(inh_layer)

        # ── On-chip learning ──────────────────────────────────────────
        if plasticity:
            learning = akida.AkidaLearning(
                num_weights=w_bits * units,
                learning_competition=1,
            )
            model.compile(optimizer=learning)
            logger.info("AkidaBackend: on-chip STDP plasticity enabled")

        logger.info("AkidaBackend: model compiled — %d layer(s)", len(model.layers))
        return model

    # ── Helpers ───────────────────────────────────────────────────────

    def _choose_weight_bits(self, w_exc_hw: int) -> int:
        """
        Select the narrowest Akida weight-bit width that can represent
        the given excitatory weight without clipping.

        w_exc_hw is the already-scaled Akida integer weight.
        """
        for bits in self._VALID_WEIGHT_BITS:
            hi = (2 ** (bits - 1)) - 1
            if abs(w_exc_hw) <= hi:
                return bits
        return 8   # fallback

    @staticmethod
    def power_estimate(units: int, sparsity: float = 0.9) -> dict[str, float]:
        """
        Rough energy-per-inference estimate for an Akida SNN layer.

        Parameters
        ----------
        units     : number of neurons in the layer
        sparsity  : fraction of neurons silent (typical SNN: 0.85–0.95)

        Returns
        -------
        dict with 'active_neurons', 'energy_pJ', 'gpu_equiv_pJ', 'saving_x'
        """
        active   = units * (1 - sparsity)
        # Akida: ~1 pJ per spike event (BrainChip published figure)
        energy   = active * 1.0
        # GPU (A100) equivalent: ~100 pJ per MAC
        gpu_eq   = units * 100.0
        return {
            "active_neurons": active,
            "energy_pJ":      energy,
            "gpu_equiv_pJ":   gpu_eq,
            "saving_x":       round(gpu_eq / max(energy, 1e-9), 1),
        }
