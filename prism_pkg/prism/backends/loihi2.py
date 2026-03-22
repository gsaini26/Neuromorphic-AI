"""
prism/backends/loihi2.py
------------------------
Loihi 2 hardware backend for Prism.

Translates a mapped parameter dict into Lava processes (LIF neurons,
Dense synapses, RingBuffer sources) and wires them together.

Lava is imported lazily so the rest of the package remains importable
on machines without Lava installed — useful for testing and CI.
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Loihi2Backend:
    """
    Compiles Prism-mapped parameters into a runnable Lava process graph.

    The compiled graph contains:
      source    — RingBuffer (10-step all-ones spike train)
      dense_exc — Dense synapse with excitatory weight
      neuron_a  — LIF (the "meaning" neuron, leaky, with floor)
      dense_b   — Dense synapse feeding neuron_b
      neuron_b  — LIF (the "context anchor", zero-leak)
      dense_inh — Dense inhibitory synapse (negative weight)
    """

    def compile(self, p: dict[str, Any]) -> dict[str, Any]:
        """
        Build and wire a Lava process graph from mapped params.

        Parameters
        ----------
        p : dict
            Output of PrismEngine.map_to_hardware().

        Returns
        -------
        dict with keys:
          'neuron_a', 'neuron_b'  — Lava LIF processes
          'source'                — Lava RingBuffer
          'run_cfg'               — Loihi2SimCfg instance
          'wiring'                — human-readable wiring description
        """
        try:
            import numpy as np
            from lava.proc.io.source      import RingBuffer
            from lava.proc.dense.process  import Dense
            from lava.proc.lif.process    import LIF
            from lava.magma.core.run_configs import Loihi2SimCfg
        except ImportError as e:
            raise ImportError(
                "Lava is not installed.  Install it with:\n"
                "  pip install lava-nc\n"
                "or visit https://lava-nc.org"
            ) from e

        vth_a    = p.get("vth",      50)
        vth_b    = vth_a * 2          # Context anchor fires at 2× A threshold
        dv_a     = p.get("dv",       500)
        du       = p.get("du",       4095)
        v_min    = p.get("v_min_exp", 9)
        v_bits   = p.get("v_bits",   32)
        w_exc    = p.get("w_exc",    10)
        w_inh    = p.get("w_inh",   -100)
        steps    = p.get("steps",    25)

        logger.info(
            "Loihi2Backend: compiling — vth_a=%d vth_b=%d dv_a=%d "
            "w_exc=%d w_inh=%d v_bits=%d v_min_exp=%d",
            vth_a, vth_b, dv_a, w_exc, w_inh, v_bits, v_min,
        )

        # ── Source ───────────────────────────────────────────────────
        data   = np.ones((1, steps), dtype=int)
        source = RingBuffer(data=data)

        # ── Synapses ─────────────────────────────────────────────────
        dense_exc = Dense(weights=np.array([[w_exc]],  dtype=int))
        dense_b   = Dense(weights=np.array([[w_exc]],  dtype=int))
        dense_inh = Dense(weights=np.array([[w_inh]],  dtype=int))

        # ── Neurons ───────────────────────────────────────────────────
        # Neuron A — leaky meaning neuron with saturation floor
        neuron_a = LIF(
            shape     = (1,),
            vth       = vth_a,
            du        = du,
            dv        = dv_a,
            v_min_exp = v_min,
            v_bits    = v_bits,
        )

        # Neuron B — zero-leak context anchor (no floor needed)
        neuron_b = LIF(
            shape  = (1,),
            vth    = vth_b,
            du     = du,
            dv     = 0,       # zero decay — frozen context anchor
            v_bits = v_bits,
        )

        # ── Wiring ────────────────────────────────────────────────────
        #   source ──► dense_exc ──► neuron_a
        #   source ──► dense_b   ──► neuron_b
        #   neuron_b ──► dense_inh ──► neuron_a  (inhibitory feedback)
        source.s_out.connect(dense_exc.s_in)
        dense_exc.a_out.connect(neuron_a.a_in)

        source.s_out.connect(dense_b.s_in)
        dense_b.a_out.connect(neuron_b.a_in)

        neuron_b.s_out.connect(dense_inh.s_in)
        dense_inh.a_out.connect(neuron_a.a_in)

        run_cfg = Loihi2SimCfg()   # fixed_pt — respects dv and v_min_exp

        logger.info("Loihi2Backend: process graph wired successfully")

        return {
            "neuron_a": neuron_a,
            "neuron_b": neuron_b,
            "source":   source,
            "run_cfg":  run_cfg,
            "wiring":   (
                "source → dense_exc → neuron_a\n"
                "source → dense_b   → neuron_b\n"
                "neuron_b ⊣ dense_inh → neuron_a"
            ),
        }

    # ── Utility: run the compiled graph ──────────────────────────────

    @staticmethod
    def run(graph: dict, steps: int = 25) -> list[dict]:
        """
        Execute a compiled Lava graph step-by-step.

        Parameters
        ----------
        graph : dict
            The dict returned by compile().
        steps : int
            Number of simulation steps.

        Returns
        -------
        list of dicts, one per step:
          {'step': int, 'va': float, 'vb': float,
           'a_fired': bool, 'b_fired': bool}
        """
        from lava.magma.core.run_conditions import RunSteps

        neuron_a = graph["neuron_a"]
        neuron_b = graph["neuron_b"]
        run_cfg  = graph["run_cfg"]
        results  = []

        import numpy as np

        def _get_v(n) -> float:
            raw = n.v.get()
            if isinstance(raw, np.ndarray): return float(raw.flat[0])
            if isinstance(raw, np.generic):  return float(raw)
            return float(raw)

        prev_a = prev_b = 0.0
        for step in range(1, steps + 1):
            neuron_a.run(condition=RunSteps(num_steps=1), run_cfg=run_cfg)
            va = _get_v(neuron_a)
            vb = _get_v(neuron_b)
            a_fired = (va == 0.0 and step > 1 and prev_a > 0)
            b_fired = (vb == 0.0 and step > 1 and prev_b > 0)
            results.append({
                "step": step, "va": va, "vb": vb,
                "a_fired": a_fired, "b_fired": b_fired,
            })
            prev_a, prev_b = va, vb

        neuron_a.stop()
        return results
