# Neuromorphic-AI
Investigation of Neuromorphic Chips And Abstraction Layers.
V-Flow · Prism IDE — a full neuromorphic model design environment. 
Left panel — Manifold Explorer + Prism Mapper
The tree shows manifold hierarchy (WordProcessor, ContextResonator, InhibitoryGate, individual cells and flows). Click any node to load its ν-Flow source. The mini Prism Mapper at the bottom shows the live voltage→hardware translation with visual bars, and updates instantly when you switch hardware targets.
Center — ν-Flow Editor + Console
Full syntax-highlighted ν-Flow source with all the v1.5 keywords: manifold, cell, flow, stack, on Pulse, emit. The console logs every Prism compiler stage: lexer → quantization pass → floor mapping → decay check → Lava process graph.
Right panel — Live Circuit + Spike Trace + Params
The circuit canvas renders the neuron graph (source → dense_exc → Neuron A ⊣ Neuron B) with live glow effects when neurons fire. The spike trace charts V_A and V_B in real time. The four parameter cards show live voltage, spike count, and an energy estimate in µJ — showing the neuromorphic advantage vs GPU.
Hardware targets — switch between Loihi 2, Akida, and Analog (memristor) and the Prism Mapper output updates to show the correct backend config for each chip.
Hit ▶ Compile + Run (or Ctrl+Enter) to watch the full v-Flow 1.5 simulation run step by step. 
