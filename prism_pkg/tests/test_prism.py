"""
tests/test_prism.py
-------------------
Comprehensive test suite for the Prism engine.

All tests run without Lava or Akida installed — hardware backend
calls are skipped/mocked where necessary.

Run with:
    python -m pytest tests/ -v
or standalone:
    python tests/test_prism.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import unittest

from prism.engine              import PrismEngine
from prism.manifold.parser     import VFlowParser, LexError, ParseError
from prism.manifold.ast_nodes  import Manifold, Cell, Flow, Stack
from prism.utils.validators    import (
    validate_params, SiliconSafeError, _suggest_dv
)


# ═══════════════════════════════════════════════════════════════════════
#  ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestPrismEngineLoihi2(unittest.TestCase):

    def setUp(self):
        self.engine = PrismEngine(target="loihi2")

    def test_describe(self):
        self.assertIn("Loihi 2", self.engine.describe())

    def test_map_basic(self):
        cfg = self.engine.map_to_hardware({
            "v_threshold": "50mV",
            "v_decay":     500,
            "v_min_exp":   9,
            "w_exc":       "10mV",
            "w_inh":       "-100mV",
        })
        self.assertEqual(cfg["vth"],       50)
        self.assertEqual(cfg["dv"],       500)
        self.assertEqual(cfg["v_min_exp"],  9)
        self.assertEqual(cfg["w_exc"],     10)
        self.assertEqual(cfg["w_inh"],   -100)
        self.assertEqual(cfg["floor_val"], -512)
        self.assertEqual(cfg["v_bits"],    32)

    def test_map_voltage_unit_conversion(self):
        cfg = self.engine.map_to_hardware({"v_threshold": "1.0V"})
        # 1.0 V → 1000 mV → vth = 1000 (identity mapping for Loihi2)
        self.assertEqual(cfg["vth"], 1000)

    def test_map_millivolt(self):
        cfg = self.engine.map_to_hardware({"v_threshold": "200mV"})
        self.assertEqual(cfg["vth"], 200)

    def test_decay_info_zero_decay(self):
        info = self.engine.decay_info(0)
        self.assertAlmostEqual(info["decay_factor"], 1.0, places=4)
        self.assertAlmostEqual(info["decay_pct_per_step"], 0.0, places=1)

    def test_decay_info_full_decay(self):
        info = self.engine.decay_info(4095)
        self.assertAlmostEqual(info["decay_factor"], 1/4096, places=6)
        self.assertGreater(info["decay_pct_per_step"], 99.9)

    def test_decay_info_vflow15_value(self):
        """dv=500 should give ≈ 12.2% decay per step."""
        info = self.engine.decay_info(500)
        self.assertAlmostEqual(info["decay_pct_per_step"], 12.2, delta=0.1)

    def test_weight_safe_range_loihi2(self):
        r = self.engine.weight_safe_range()
        self.assertEqual(r["min"], -128)
        self.assertEqual(r["max"],  127)

    def test_safe_mode_off_skips_validation(self):
        """With safe_mode=False, extreme params should not raise."""
        engine = PrismEngine(target="loihi2", safe_mode=False)
        # Would fail validation in safe_mode=True (ratio > 20)
        cfg = engine.map_to_hardware({
            "v_threshold": "50mV",
            "v_decay": 0,
            "w_exc":   "1mV",
            "w_inh":   "-9999mV",
        })
        self.assertIsNotNone(cfg)

    def test_unknown_target_raises(self):
        with self.assertRaises(ValueError):
            PrismEngine(target="nonexistent_chip")


class TestPrismEngineAkida(unittest.TestCase):

    def setUp(self):
        self.engine = PrismEngine(target="akida")

    def test_describe(self):
        self.assertIn("Akida", self.engine.describe())

    def test_map_scales_to_8bit(self):
        cfg = self.engine.map_to_hardware({
            "v_threshold": "800mV",
            "v_decay":     500,
            "w_exc":       "10mV",
            "w_inh":       "-100mV",
            # Akida maps -100mV → -3 (4-bit). Floor must be within 20× of that.
            # -3 × 5 = -15 → use v_min_exp=4 → floor = -16
            "v_min_exp":   4,
        })
        # Akida 8-bit: 800/1000 * 255 = 204
        self.assertLessEqual(cfg["vth"], 255)
        self.assertGreaterEqual(cfg["vth"], 0)
        # Akida decay mapped to 0..15
        self.assertLessEqual(cfg["dv"], 15)

    def test_weight_safe_range_akida(self):
        r = self.engine.weight_safe_range()
        self.assertEqual(r["min"], -8)
        self.assertEqual(r["max"],  7)


# ═══════════════════════════════════════════════════════════════════════
#  PARSER TESTS
# ═══════════════════════════════════════════════════════════════════════

WORD_PROCESSOR_SRC = """
// v-Flow 1.5 test source
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

  on Pulse(input_token) {
    LanguageNeuron_A.v_state += input_token * AssociativeLink.weight;
    emit Pulse(next_manifold);
  }
}
"""

class TestVFlowParser(unittest.TestCase):

    def setUp(self):
        self.parser = VFlowParser()

    def test_parse_returns_manifold(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        self.assertIsInstance(m, Manifold)

    def test_manifold_name(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        self.assertEqual(m.name, "WordProcessor")

    def test_two_cells_parsed(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        self.assertEqual(len(m.cells), 2)

    def test_cell_names(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        names = {c.name for c in m.cells}
        self.assertIn("LanguageNeuron_A", names)
        self.assertIn("ContextAnchor_B", names)

    def test_cell_threshold(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        a = m.get_cell("LanguageNeuron_A")
        self.assertEqual(str(a.v_threshold), "50mV")

    def test_cell_role(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        a = m.get_cell("LanguageNeuron_A")
        self.assertEqual(a.role, "meaning")

    def test_two_flows_parsed(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        self.assertEqual(len(m.flows), 2)

    def test_inhibitory_flow(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        inh = m.get_flow("InhibitoryLink")
        self.assertIsNotNone(inh)
        self.assertTrue(inh.is_inhibitory)

    def test_excitatory_flow(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        exc = m.get_flow("AssociativeLink")
        self.assertIsNotNone(exc)
        self.assertFalse(exc.is_inhibitory)

    def test_stack_parsed(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        self.assertEqual(len(m.stacks), 1)
        self.assertEqual(m.stacks[0].name, "ContextResonator")
        self.assertEqual(m.stacks[0].depth, 4096)
        self.assertEqual(m.stacks[0].rule, "Hebbian")

    def test_pulse_block_parsed(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        self.assertEqual(len(m.pulse_blocks), 1)
        self.assertEqual(m.pulse_blocks[0].signal, "input_token")

    def test_manifold_summary(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        s = m.summary()
        self.assertIn("WordProcessor", s)
        self.assertIn("2 cell", s)
        self.assertIn("2 flow", s)

    def test_inhibitory_flows_property(self):
        m = self.parser.parse(WORD_PROCESSOR_SRC)
        self.assertEqual(len(m.inhibitory_flows), 1)
        self.assertEqual(len(m.excitatory_flows), 1)

    def test_empty_source_raises(self):
        with self.assertRaises(ParseError):
            self.parser.parse("")

    def test_bad_char_raises_lex_error(self):
        with self.assertRaises(LexError):
            self.parser.parse("manifold Foo { v_threshold: @bad; }")

    def test_parse_all_returns_list(self):
        src = WORD_PROCESSOR_SRC + "\n" + """
manifold SecondManifold {
  cell NeuronX { v_threshold: 100mV; }
}
"""
        results = self.parser.parse_all(src)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].name, "SecondManifold")

    def test_standalone_stack(self):
        src = """
stack ContextResonator {
  depth: 512;
  persistence: 100ms;
  rule: Hebbian;
}
"""
        m = self.parser.parse(src)
        self.assertEqual(m.stacks[0].depth, 512)


# ═══════════════════════════════════════════════════════════════════════
#  VALIDATOR TESTS
# ═══════════════════════════════════════════════════════════════════════

_HW_LOIHI = {"v_bits": 32, "weight_bits": 8}
_HW_AKIDA  = {"v_bits":  8, "weight_bits": 4}

class TestValidators(unittest.TestCase):

    def _ok(self, **kwargs):
        """Call validate_params with defaults overridable by kwargs."""
        defaults = dict(vth=50, dv=500, w_exc=10, w_inh=-100,
                        floor=-512, hw=_HW_LOIHI)
        defaults.update(kwargs)
        validate_params(**defaults)   # should not raise

    def test_valid_params_pass(self):
        self._ok()

    def test_vth_too_large_raises(self):
        with self.assertRaises(SiliconSafeError):
            self._ok(vth=99_999)

    def test_w_inh_too_large_ratio_raises(self):
        with self.assertRaises(SiliconSafeError):
            self._ok(w_exc=1, w_inh=-1000)   # ratio = 1000:1

    def test_zero_decay_infinite_memory_raises(self):
        """dv=0 with w_inh deeper than w_exc should raise."""
        with self.assertRaises(SiliconSafeError):
            self._ok(dv=0, w_exc=10, w_inh=-100)

    def test_floor_too_deep_raises(self):
        with self.assertRaises(SiliconSafeError):
            self._ok(w_inh=-10, floor=-10_000)   # floor >> 2*|w_inh|

    def test_akida_8bit_range(self):
        """Akida 8-bit: vth=200 should pass."""
        self._ok(vth=200, w_exc=3, w_inh=-50, floor=-100, hw=_HW_AKIDA)

    def test_akida_overflow_raises(self):
        with self.assertRaises(SiliconSafeError):
            self._ok(vth=1000, hw=_HW_AKIDA)   # 1000 > 255

    def test_suggest_dv_returns_int(self):
        dv = _suggest_dv(crush=100, exc=10, max_steps=10)
        self.assertIsInstance(dv, int)
        self.assertGreater(dv, 0)
        self.assertLessEqual(dv, 4095)

    def test_suggest_dv_sufficient_recovery(self):
        """Verify the suggested dv actually recovers within max_steps."""
        crush, exc, steps = 200, 10, 15
        dv = _suggest_dv(crush, exc, steps)
        k  = (4096 - dv) / 4096
        if k < 1:
            recovery = math.ceil(math.log(exc / crush) / math.log(k))
            self.assertLessEqual(recovery, steps)


# ═══════════════════════════════════════════════════════════════════════
#  AST NODE TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestASTNodes(unittest.TestCase):

    def test_cell_repr(self):
        c = Cell("NeuronA", v_threshold="50mV", role="meaning")
        self.assertIn("NeuronA", repr(c))

    def test_flow_inhibitory_property(self):
        f = Flow("Inh", flow_type="Inhibitory")
        self.assertTrue(f.is_inhibitory)
        f2 = Flow("Exc", flow_type="Excitatory")
        self.assertFalse(f2.is_inhibitory)

    def test_manifold_get_cell(self):
        m = Manifold("Test", cells=[Cell("A"), Cell("B")])
        self.assertIsNotNone(m.get_cell("A"))
        self.assertIsNone(m.get_cell("C"))

    def test_manifold_inhibitory_flows(self):
        m = Manifold("Test", flows=[
            Flow("Exc", flow_type="Excitatory"),
            Flow("Inh", flow_type="Inhibitory"),
        ])
        self.assertEqual(len(m.inhibitory_flows), 1)
        self.assertEqual(len(m.excitatory_flows), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
