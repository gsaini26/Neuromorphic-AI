"""
examples/run_word_processor.py
-------------------------------
End-to-end example: parse a ν-Flow file, translate with Prism,
and run a 25-step inhibitory simulation on Loihi 2.

Requires:
    pip install lava-nc

Usage:
    python examples/run_word_processor.py
    python examples/run_word_processor.py --target akida
    python examples/run_word_processor.py --steps 40
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from prism import PrismEngine, VFlowParser

# ── Inline ν-Flow source (normally loaded from a .vf file) ───────────
VFLOW_SRC = """
manifold WordProcessor {

  cell LanguageNeuron_A {
    v_threshold: 50mV;
    v_decay:     500;
    v_min_exp:   9;
    role:        "meaning";
  }

  cell ContextAnchor_B {
    v_threshold: 100mV;
    v_decay:     0;
    role:        "context";
  }

  flow AssociativeLink {
    type:        Excitatory;
    init_weight: +10mV;
    target:      LanguageNeuron_A;
  }

  flow InhibitoryLink {
    type:        Inhibitory;
    init_weight: -100mV;
    source:      ContextAnchor_B;
    target:      LanguageNeuron_A;
  }

  stack ContextResonator {
    depth:       4096;
    persistence: 500ms;
    rule:        Hebbian;
  }
}
"""


def main():
    ap = argparse.ArgumentParser(description="ν-Flow Prism example")
    ap.add_argument("--target", default="loihi2",
                    choices=["loihi2", "akida"],
                    help="Hardware compilation target")
    ap.add_argument("--steps",  type=int, default=25,
                    help="Number of simulation steps")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # ── Step 1: Parse ν-Flow source ───────────────────────────────────
    print("─" * 60)
    print("  Prism ν-Flow Compiler v1.5")
    print("─" * 60)

    parser   = VFlowParser()
    manifold = parser.parse(VFLOW_SRC)
    print(f"  Parsed:  {manifold.summary()}")

    # ── Step 2: Extract params from AST ──────────────────────────────
    cell_a  = manifold.get_cell("LanguageNeuron_A")
    cell_b  = manifold.get_cell("ContextAnchor_B")
    exc     = manifold.get_flow("AssociativeLink")
    inh     = manifold.get_flow("InhibitoryLink")

    vflow_params = {
        "v_threshold": cell_a.v_threshold,
        "v_decay":     cell_a.v_decay,
        "v_min_exp":   cell_a.v_min_exp,
        "w_exc":       exc.init_weight,
        "w_inh":       inh.init_weight,
        "steps":       args.steps,
    }

    # ── Step 3: Translate via Prism ───────────────────────────────────
    engine  = PrismEngine(target=args.target, verbose=args.verbose)
    cfg     = engine.map_to_hardware(vflow_params)

    print(f"  Target:  {engine.describe()}")
    print(f"  Mapped:  vth={cfg['vth']}  dv={cfg['dv']}  "
          f"w_exc={cfg['w_exc']}  w_inh={cfg['w_inh']}")
    decay = engine.decay_info(cfg['dv'])
    print(f"  Decay:   {decay['decay_pct_per_step']}%/step  "
          f"  recovery ~{decay['recovery_steps_from_512']} steps from floor")
    print("─" * 60)

    # ── Step 4: Compile & run ────────────────────────────────────────
    try:
        graph   = engine.compile(cfg)
    except (ImportError, NotImplementedError) as e:
        print(f"\n  ⚠  Hardware backend not available: {e}")
        print("  Running software simulation instead…\n")
        _soft_simulate(cfg, args.steps)
        return

    from prism.backends.loihi2 import Loihi2Backend
    results = Loihi2Backend.run(graph, steps=args.steps)

    print(f"{'Step':>5}  {'V_A':>8}  {'V_B':>8}  {'Event'}")
    print("-" * 60)
    for r in results:
        evt = ""
        if r["a_fired"]: evt = "⚡ A FIRES"
        elif r["va"] < 0: evt = "↓ inhibited"
        if r["b_fired"]: evt += "  ⚡ B FIRES"
        print(f"{r['step']:>5}  {r['va']:>8.1f}  {r['vb']:>8.1f}  {evt}")

    fires_a = sum(1 for r in results if r["a_fired"])
    fires_b = sum(1 for r in results if r["b_fired"])
    print("─" * 60)
    print(f"  Complete — A fired {fires_a}×  B fired {fires_b}×")


def _soft_simulate(cfg, steps):
    """Pure-Python fallback simulation (no Lava required)."""
    W_EXC = cfg["w_exc"]
    W_INH = cfg["w_inh"]
    VTH_A = cfg["vth"]
    VTH_B = VTH_A * 2
    DV_A  = cfg["dv"]
    FLOOR = cfg["floor_val"]
    kA    = (4096 - DV_A) / 4096

    va = vb = 0.0
    fa = fb = 0
    print(f"{'Step':>5}  {'V_A':>8}  {'V_B':>8}  {'Event'}")
    print("-" * 60)
    for step in range(1, steps + 1):
        va *= kA
        va += W_EXC; vb += W_EXC
        b_fired = vb >= VTH_B
        if b_fired: vb = 0; va += W_INH; fb += 1
        va = max(va, FLOOR)
        a_fired = va >= VTH_A
        if a_fired: va = 0; fa += 1
        evt = ""
        if a_fired: evt = "⚡ A FIRES"
        elif va < 0: evt = "↓ inhibited"
        if b_fired: evt += "  ⚡ B FIRES"
        print(f"{step:>5}  {va:>8.1f}  {vb:>8.1f}  {evt}")
    print("─" * 60)
    print(f"  Complete — A fired {fa}×  B fired {fb}×")


if __name__ == "__main__":
    main()
