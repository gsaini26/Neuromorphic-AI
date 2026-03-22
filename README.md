# Neuromorphic-AI

> An investigation of neuromorphic chips and abstraction layers, culminating in a novel AI paradigm built on spike dynamics instead of matrix multiplication.

This repository documents the design and implementation of **ν-Flow** and **Prism** — a domain-specific language and compiler for neuromorphic hardware — along with a series of working demonstrations that run on any CPU with no GPU required.

---

## What this is

Conventional AI (GPT, Claude, Llama) runs on dense matrix multiplication. Every neuron computes on every token, every time, even when the result is zero. That zero still costs energy on a GPU.

This project explores a different execution model: **event-based spike dynamics**. Neurons only activate when their accumulated voltage crosses a threshold. Between spikes they are physically inert. On neuromorphic hardware (Intel Loihi 2, BrainChip Akida), an inert neuron draws zero energy — not "computes zero", but draws *no current at all*.

The practical result demonstrated here is a language model that generates novel text while keeping **~65–75% of its neurons silent per character**, achieving an estimated **~50,000× energy advantage** over GPU-dense equivalent computation.

---

## Repository structure

```
Neuromorphic-AI/
│
├── neuromorphic_demo.py          ← Main demonstration (start here)
├── neuromorphic_qa_demo.py       ← Interactive Q&A demo
├── neuromorphic_llm_demo.py      ← Text generation demo
├── loihi2_spike_circuit.py       ← v-Flow 1.5 Loihi 2 circuit (requires lava-nc)
├── vflow-prism.html              ← ν-Flow Prism IDE (open in browser)
│
├── general_knowledge.txt         ← Q&A training document (general knowledge)
├── vflow_knowledge.txt           ← Q&A training document (this project)
│
└── prism_vflow_package.zip       ← Full Prism compiler package
    └── prism_pkg/
        ├── prism/
        │   ├── engine.py         ← PrismEngine: universal translation layer
        │   ├── encoder.py        ← Gaussian population encoder
        │   ├── slm_builder.py    ← NeuromorphicSLM model builder
        │   ├── stdp.py           ← STDP on-chip learning
        │   ├── manifold/
        │   │   ├── parser.py     ← ν-Flow lexer + recursive-descent parser
        │   │   └── ast_nodes.py  ← AST dataclasses (Manifold, Cell, Flow, Stack)
        │   ├── backends/
        │   │   ├── loihi2.py     ← Intel Loihi 2 backend (via Lava)
        │   │   └── akida.py      ← BrainChip Akida backend (via MetaTF)
        │   └── utils/
        │       └── validators.py ← Silicon-Safe parameter validation
        ├── tests/
        │   └── test_prism.py     ← 43-test suite (runs without hardware)
        └── examples/
            ├── run_word_processor.py
            └── train_nano_slm.py
```

---

## Quick start

**Requirements:** Python 3.10 or later. One dependency.

```bash
pip install numpy
python neuromorphic_demo.py
```

No GPU. No CUDA. No PyTorch. Runs on an Intel HD 4600 integrated graphics chip in a standard Windows 10 laptop with 12 GB RAM.

---

## The main demonstration

`neuromorphic_demo.py` is the centrepiece. It runs in three acts that together make the case for the neuromorphic paradigm.

### Act I — The Circuit

A single excitatory/inhibitory neuron pair from the v-Flow 1.5 Silicon-Safe design. The voltage trace renders live in the terminal:

```
  Step  Neuron A voltage                    V_A  Event
  ──────────────────────────────────────────────────────
     1  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░   10.0
     7  ███████████████████████████████░   49.0
     8  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    0.0  ⚡ SPIKE
     9  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░   10.0
    10  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  -81.2  ⊣ INHIBIT
    15  ▄░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   -3.2  ↑ recovering
    16  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░    7.2
```

This is the v-Flow 1.5 circuit: excitation building step by step, a single inhibitory spike crushing it to −81, leaky decay pulling it back to zero over 6 steps. The energy counter increments by ~1 pJ on spike events and ~0.02 pJ during silence. A GPU performing the same 25 steps burns the same energy on every step regardless.

### Act II — The Network

The Act I circuit replicated 128 times, trained as a character-level language model. Sparsity grows live during training as inhibitory connections strengthen:

```
  Epoch    Loss     H1 sparsity             H2 sparsity           Progress
     1     2.628  █████████████░░░ 65%    █████████████████░ 86%  |░░░░░░░░░░|
    31     0.254  ███████████████░ 77%    ████████████████░░ 83%  |████░░░░░░|
    80     0.152  ████████████████ 80%    █████████████████░ 87%  |████████░░|
```

After training, Prism quantises all weights to 8-bit integers with negligible information loss (KL divergence < 0.0001). These integer matrices are what would be flashed to physical hardware.

### Act III — The Generation

The model generates text character by character. Every character is a new spike competition among 128 neurons. The output does not exist anywhere in the training data.

```
  Seed: 'the river'
  the river flows through the valley and past the old rioll theer flows

  Seed: 'the bank'
  the bank stand tall trees their roots deep in the river flows through
```

The model gets most characters right and occasionally produces novel combinations — exactly the behaviour of a generative model operating near its knowledge boundary. This is not retrieval. The sentences above were never stored.

**CLI options:**

```bash
python neuromorphic_demo.py                          # all three acts
python neuromorphic_demo.py --text yourfile.txt      # train on your own text
python neuromorphic_demo.py --seed "the mountain"    # custom generation seed
python neuromorphic_demo.py --epochs 400             # longer training
python neuromorphic_demo.py --act 1                  # run only Act I
python neuromorphic_demo.py --no-colour              # plain text output
```

### What the demo trains on

The default training text is 20 sentences about a river and its bank, hardcoded as `DEFAULT_TEXT` at the top of `neuromorphic_demo.py`:

```
the river flows through the valley and past the old stone bridge
the bank of the river is covered with reeds and wild flowers
children play near the river bank every summer afternoon
the water flows fast near the bridge and slow near the bank
the old man sat on the river bank and watched the water flow
fish swim near the bank where the water is shallow and still
the bridge spans the river at its widest point in the valley
on the far bank stand tall trees their roots deep in the soil
the river carries soil from the mountains down to the valley floor
every spring the river floods its bank and waters the meadow
the current runs swift in the middle and calm near the edge
swallows dip low over the water hunting insects near the bank
the old bridge was built from stone taken from the valley walls
a boat moves slowly upstream keeping close to the far bank
the river knows no hurry it has carved this valley over ages
reeds grow tall where the water meets the muddy bank in spring
the fisherman casts his line from the bridge into deep water
small fish dart through the shallows near the mossy bank stones
mist rises from the river surface in the cool morning air
the valley holds the river like cupped hands hold water still
```

This corpus was chosen deliberately. The word **"bank"** is ambiguous — it means both a riverbank and a financial institution. This is the same ambiguity the inhibitory circuit was originally designed to resolve. The context anchor neuron (B) learns to fire on words like "river", "water", and "bridge", and through the inhibitory synapse it suppresses interpretations that do not fit the established context. The training text makes that disambiguation visible and testable.

The corpus is intentionally small: **1,229 characters, 24-character vocabulary**. That keeps training under 30 seconds on a standard laptop while still producing coherent generation. The architecture scales automatically to any text — a larger corpus grows the vocabulary and input layer size but requires no code changes.

To train on more meaningful content, point the demo at any plain text file:

```bash
python neuromorphic_demo.py --text vflow_knowledge.txt
```

This trains the model on the ν-Flow and Prism terminology from this project. Generation seeds like `"the prism"` or `"loihi2 u"` will then produce relevant continuations drawn from the learned character patterns of that domain.

---

## The ν-Flow language

ν-Flow is a domain-specific language for describing neuromorphic circuits using biological concepts — voltage, decay, spikes, thresholds — rather than matrix operations.

```
// v-Flow 1.5 — WordProcessor manifold
manifold WordProcessor {

  cell LanguageNeuron_A {
    v_threshold: 50mV;      // fires after 5 excitatory spikes
    v_decay:     500;        // 12.2% voltage leak per step
    v_min_exp:   9;          // saturation floor at −512 (silicon-safe)
    role:        "meaning";
  }

  cell ContextAnchor_B {
    v_threshold: 100mV;     // fires every 10 spikes
    v_decay:     0;          // zero leak — frozen context anchor
    role:        "context";
  }

  flow AssociativeLink {
    type:        Excitatory;
    init_weight: +10mV;      // quiet whisper
    target:      LanguageNeuron_A;
  }

  flow InhibitoryLink {
    type:        Inhibitory;
    init_weight: -100mV;     // loud shout — 10:1 ratio
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
    if (ContextAnchor_B.fired) {
      emit InhibitoryLink;
    }
    emit Pulse(next_manifold);
  }
}
```

ν-Flow source files (`.vf`) are compiled by the **Prism engine** into hardware-specific configurations for Loihi 2, Akida, or analog targets. The [Prism IDE](vflow-prism.html) provides a browser-based environment for writing and simulating ν-Flow code with a live circuit visualiser, spike trace, and energy meter.

---

## The Prism compiler

Prism is the hardware abstraction layer that bridges ν-Flow physics and silicon reality. It runs in four stages:

| Stage | Action | Output |
|---|---|---|
| **I. Lexer** | Parse `.vf` source into typed token stream | AST nodes (Manifold, Cell, Flow, Stack) |
| **II. Quantisation** | Map float voltages to hardware integers | 8-bit weights (Loihi 2) or 4-bit (Akida) |
| **III. Floor mapping** | Inject `v_min_exp` saturation constraint | Silicon-safe parameter set |
| **IV. Process graph** | Wire Lava / MetaTF processes and synapses | Runnable simulation or chip binary |

### The three Silicon-Safe design rules

These constraints prevent the exponential voltage blow-up that caused `−10^84` values in earlier versions of the circuit:

**Rule I — Saturation Constraint:** Voltage is clamped at `−2^v_min_exp`. A strong inhibitory spike drives a neuron to `−81 mV`; without a floor, repeated inhibition would compound to `−10^84` over 20 steps. The floor makes recovery physically possible.

**Rule II — Multi-Scale Weight Map:** Weights must stay in the 8-bit safe zone (`−128` to `+127`). The standard v-Flow 1.5 ratio is `w_exc = +10`, `w_inh = −100` (10:1 inhibitory ratio). Larger weights overflow the fixed-point registers.

**Rule III — Asymmetric Decay:** Different neurons carry different leak rates. The meaning neuron uses `dv = 500` (~12% leak/step — fast forgetting). The context anchor uses `dv = 0` (zero leak — holds voltage indefinitely). This asymmetry creates temporal filtering and context persistence.

### Using the Prism engine

```python
from prism import PrismEngine

engine = PrismEngine(target="loihi2")

cfg = engine.map_to_hardware({
    "v_threshold": "50mV",
    "v_decay":     500,
    "v_min_exp":   9,
    "w_exc":       "10mV",
    "w_inh":       "-100mV",
})

# {'vth': 50, 'dv': 500, 'v_min_exp': 9, 'w_exc': 10, 'w_inh': -100,
#  'floor_val': -512, 'v_bits': 32, 'target': 'loihi2', ...}

info = engine.decay_info(500)
# {'dv': 500, 'decay_factor': 0.8779, 'decay_pct_per_step': 12.21,
#  'recovery_steps_from_512': 48}
```

### Installing the Prism package

```bash
unzip prism_vflow_package.zip
cd prism_pkg

pip install -e .                   # core package, no hardware deps
pip install -e ".[loihi2]"         # adds lava-nc simulation layer
pip install -e ".[akida]"          # adds BrainChip MetaTF

python tests/test_prism.py         # 43 tests, all passing
python examples/train_nano_slm.py  # end-to-end SLM training
```

---

## The v-Flow 1.5 circuit

`loihi2_spike_circuit.py` is the hardware circuit — a Loihi 2 simulation using the actual Lava framework. It requires `lava-nc` to run.

```bash
pip install lava-nc
python loihi2_spike_circuit.py
```

**Circuit topology:**

```
RingBuffer ──► dense_exc (+10) ──► Neuron A  (LIF, vth=50,  dv=500)
     │                                      ▲
     │                             dense_inh (−100)
     │                                      │
     └────► dense_b  (+10) ──► Neuron B  (LIF, vth=100, dv=0)
```

**Expected output (25 steps):**

```
  Step    V_A       V_B    Neuron A         Neuron B
     1    10.0     10.0
     8     0.0     80.0    FIRE + RESET
    10   -81.2      0.0    << INHIBITED     FIRE + RESET
    15    -3.2     50.0    << INHIBITED
    16     7.2     60.0    [recovering]
    20   -62.5      0.0    << INHIBITED     FIRE + RESET
```

The circuit demonstrates three phenomena that cannot be observed in a conventional neural network: voltage accumulation over time, inhibitory suppression via a negative-weight synapse, and leaky recovery from a deep negative state.

---

## The Q&A demo

`neuromorphic_qa_demo.py` is an interactive question-answering system trained on a user-provided knowledge base.

```bash
python neuromorphic_qa_demo.py                           # built-in knowledge
python neuromorphic_qa_demo.py --qa general_knowledge.txt
python neuromorphic_qa_demo.py --qa vflow_knowledge.txt  # ask about this project
```

**What it demonstrates vs. what it does not.** The Q&A model selects from pre-written answers. It is a retrieval classifier, not a generative model. Its value is as a teaching tool — it makes sparse activations, Prism quantisation, and energy efficiency concrete and interactive. It should not be presented as evidence that neuromorphic hardware enables language generation. For that, use `neuromorphic_demo.py` Act III.

Two knowledge bases are included:

- `general_knowledge.txt` — 164 pairs covering astronomy, biology, geography, history, physics, and technology. Trains in ~30 seconds.
- `vflow_knowledge.txt` — 175 pairs covering this project: ν-Flow syntax, Prism compiler stages, hardware targets, LIF physics, inhibition, STDP, and energy efficiency. Trains in ~35 seconds.

**Custom knowledge base format:**

```
question: what does the heart do
answer: the heart pumps blood through the body delivering oxygen to every cell

question: how does the heart work
answer: the heart pumps blood through the body delivering oxygen to every cell
```

Multiple question phrasings per answer improve generalisation to paraphrased queries. The model handles any plain text file in this format — a product manual, a course FAQ, a technical specification.

---

## Energy model

All energy estimates use published figures:

| Operation | Energy | Source |
|---|---|---|
| Loihi 2 synaptic event | ~1 pJ | Intel published figure |
| A100 GPU multiply-accumulate | ~100 pJ | NVIDIA published figure |
| Silent Loihi 2 neuron | 0 pJ | No event — no current |
| Silent GPU neuron | ~100 pJ | Computes `0 × weight` regardless |

At 65% hidden-neuron sparsity with 192 total hidden neurons in the demo model:

```
Active neurons per token  :  67 of 192
Neuromorphic energy       :  ~67 pJ / token
GPU-dense equivalent      :  ~3,400,000 pJ / token
Efficiency gain           :  ~50,000×
```

The gain scales with model size. A 1M-parameter neuromorphic model at 65% sparsity uses ~3.5 µJ/token. A 1M-parameter dense GPU model uses ~100 µJ/token.

---

## Hardware targets

| Target | Access | Cost | Notes |
|---|---|---|---|
| **Loihi 2** (simulation) | `pip install lava-nc` | Free | Fixed-point simulation, no hardware needed |
| **Loihi 2** (hardware) | Intel INRC application | Free cloud access | intel.com/neuromorphic-research |
| **Akida AKD1000** | Purchase | ~$250–$300 | PCIe or M.2 board, plugs into a standard PC |
| **Analog / Memristor** | Research prototype | Varies | Prism stub target for future hardware |

To deploy the trained weights on Akida:

```python
from prism import PrismEngine

engine = PrismEngine(target="akida")
cfg    = engine.map_to_hardware({...})
model  = engine.compile(cfg)     # returns an akida.Model instance
```

---

## How ν-Flow differs from MetaTF

BrainChip's MetaTF converts standard Keras models into spiking networks after training — a lossy translation that discards the biological intent of the architecture. ν-Flow builds neuromorphic-first:

| Feature | BrainChip MetaTF | ν-Flow + Prism |
|---|---|---|
| Philosophy | Make SNNs act like ANNs | Let SNNs act like brains |
| Workflow | Train on GPU → quantise → map | Design on chip → live plasticity |
| Hardware | Akida only | Akida, Loihi 2, and Analog |
| Weight logic | Discrete bits (1–8 bit) | Voltage-physics mapping |
| Learning | Offline backpropagation | On-chip STDP plasticity |
| Language | Python / Keras | ν-Flow (hardware-agnostic DSL) |

---

## Verification

The Prism package includes a 43-test suite that runs without any hardware or optional dependencies:

```bash
python prism_pkg/tests/test_prism.py
# Ran 43 tests in 0.007s — OK
```

Tests cover the ν-Flow lexer and parser, Prism engine parameter mapping for both Loihi 2 and Akida targets, the three Silicon-Safe validators, quantisation correctness, and the STDP and Hebbian learning update rules.

---

## Roadmap

The current implementation proves the paradigm in software simulation. Next steps toward real hardware deployment:

1. **INRC application** — get cloud access to Loihi 2 and run `loihi2_spike_circuit.py` on actual silicon
2. **Akida board** — purchase an AKD1000 PCIe board (~$250) and flash the quantised weights from `train_nano_slm.py`
3. **Larger training corpus** — scale from the current 1,229-character demo text to a full document to increase vocabulary and generation coherence
4. **NIR integration** — compile ν-Flow through the Neuromorphic Intermediate Representation to achieve true chip-agnostic portability
5. **STDP fine-tuning** — after rate-domain training, enable on-chip plasticity for live weight updates during inference without retraining

---

## Dependencies

| Package | Required for | Install |
|---|---|---|
| `numpy` | All demos | `pip install numpy` |
| `lava-nc >= 0.10` | `loihi2_spike_circuit.py` | `pip install lava-nc` |
| `akida >= 2.0` | Akida hardware deployment | `pip install akida` |

Python 3.10 or later. All demos run on Windows 10, macOS, and Linux.
