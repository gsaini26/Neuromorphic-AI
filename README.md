# Neuromorphic-AI

> Investigation of Neuromorphic Chips and Abstraction Layers.

---

## ν-Flow · Prism IDE

A full neuromorphic model design environment built on the **v-Flow 1.5 Silicon-Safe** architecture.

---

### Interface Overview

#### Left Panel — Manifold Explorer + Prism Mapper

The tree shows the full manifold hierarchy:

- `WordProcessor`
- `ContextResonator`
- `InhibitoryGate`
- Individual **cells** and **flows**

Click any node to load its ν-Flow source. The mini **Prism Mapper** at the bottom shows live voltage → hardware translation with visual bars, updating instantly when you switch hardware targets.

---

#### Center — ν-Flow Editor + Console

Full syntax-highlighted ν-Flow source with all v1.5 keywords:

```
manifold   cell   flow   stack   on Pulse   emit
```

The console logs every Prism compiler stage in order:

```
lexer → quantization pass → floor mapping → decay check → Lava process graph
```

---

#### Right Panel — Live Circuit + Spike Trace + Params

The circuit canvas renders the full neuron graph:

```
source → dense_exc → Neuron A ⊣ Neuron B
```

- Live **glow effects** when neurons fire
- **Spike trace** charts `V_A` and `V_B` in real time
- Four **parameter cards** show live voltage, spike count, and an energy estimate in `µJ` — demonstrating the neuromorphic advantage over GPU

---

### Hardware Targets

Switch between targets and the Prism Mapper updates to the correct backend config:

| Target | Backend | Notes |
|---|---|---|
| **Loihi 2** | `Loihi2SimCfg()` | `fixed_pt` + `v_bits=32` |
| **Akida** | `akida.Model()` | Digital 8-bit SNN |
| **Analog** | `PrismAnalog()` | Voltage-native / memristor |

---

### Running a Simulation

Click **▶ Compile + Run** or press `Ctrl+Enter` to watch the full v-Flow 1.5 simulation step by step.
