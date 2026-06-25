# SpikeGCPN: Graph Convolutional Policy Network for Spiking Neural Network Topology Search

## Proof of Concept — Image Classification (N-MNIST)

## Core Idea

Evolutionary/genetic algorithms for SNN topology search are sample-inefficient — they treat evaluation as a black box and have no model of *why* certain topologies work. SpikeGCPN replaces random mutation and crossover with a learned policy that builds SNN topologies node-by-node and edge-by-edge, guided by a reward signal from a neuromorphic simulator. The policy learns structural priors over topology space, enabling it to search more intelligently than evolution while respecting hard architectural constraints via environment dynamics.

The approach adapts GCPN (You et al., NeurIPS 2018) — originally designed for goal-directed molecular graph generation — to the domain of spiking neural network architecture search. This PoC targets N-MNIST classification as the evaluation task.

---

## Representation

SNN topologies map naturally onto GCPN's graph representation:

| Molecular Domain | SNN Domain |
|---|---|
| Atoms | Neurons |
| Bonds | Synaptic connections |
| Bond types (single/double) | Excitatory / inhibitory |
| Valency constraints | DAG + reachability constraints |
| logP / QED score | Classification accuracy on N-MNIST |

**Node features**: neuron type (input / hidden / output), threshold voltage, time constant τ  
**Edge features**: excitatory/inhibitory sign, synaptic weight range, transmission delay

---

## Structural Constraints

The output graph must satisfy the following hard constraints, enforced in the environment's transition dynamics:

1. **DAG**: The graph must be a directed acyclic graph. No cycles are permitted. Every proposed edge addition is validated before being applied.

2. **Fixed source nodes**: The number of source nodes (in-degree zero) equals the number of pixels in the input image. These nodes are pre-instantiated at episode start and are never added or removed by the policy.

3. **Fixed sink nodes**: The number of sink nodes (out-degree zero) equals the number of output classes. These nodes are also pre-instantiated at episode start. For N-MNIST, this is 10 nodes.

The policy's job is therefore to add hidden neurons and directed edges between any existing nodes, subject to the DAG constraint. An episode is valid if every sink node is reachable from at least one source node.

---

## Architecture

SpikeGCPN follows the GCPN architecture:

1. **GCN encoder**: A 3-layer graph convolutional network runs over the current partial topology G_t. Produces 64-dim node embeddings via message passing over each directed edge type separately, then aggregated with SUM.

2. **Policy head**: Four sequential MLPs predict each action component:
   - `a_first` — which existing neuron to connect from
   - `a_second` — which neuron to connect to (existing or new hidden neuron)
   - `a_edge` — connection type (excitatory / inhibitory)
   - `a_stop` — whether to terminate generation

3. **Value function**: MLP over the aggregated graph embedding, used to estimate expected future reward and reduce variance in policy gradient updates.

---

## Training Objective

Policy gradient with PPO:
```
L_CLIP(θ) = E_t [ min( r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]
```
where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` and `Â_t` is the advantage estimate.

---

## Reward Signal

The reward signal bridges the policy to the simulator. SuperNeuroMAT (ORNL) serves as the black-box evaluation oracle — it takes a weighted directed graph, simulates LIF dynamics on N-MNIST, and returns classification accuracy. Gradients do **not** flow through the simulator.

**Final rewards** (on completed topology):
- `r_accuracy` — N-MNIST validation accuracy from SuperNeuroMAT
- `r_efficiency` — penalty for neuron count, synapse count, or total spike energy

**Intermediate rewards** (at each step):

- `r_validity` — small positive reward for maintaining a valid, connected DAG with all sinks reachable

**Advantage estimate:**
```
Â_t = r_t + γ V_ω(s_{t+1}) - V_ω(s_t)
```

---

## Simulator Interface

A lightweight wrapper script takes a weighted, directed acyclic graph and:
1. Imprints it onto SuperNeuroMAT
2. Runs N-MNIST spike encoding and forward pass
3. Returns accuracy and optionally spike count / energy metrics

This wrapper is called once per completed episode to compute `r_accuracy`.

---

## Advantage Over Evolutionary Search

| Aspect | Evolutionary Search | SpikeGCPN |
|---|---|---|
| Exploration | Random mutation/crossover | Policy-guided, structure-aware |
| Sample efficiency | Poor — no model of topology quality | Better — GCN learns structural priors |
| Constraint handling | Post-hoc filtering | Built into environment dynamics |
| Transfer | Restarts from scratch | Policy generalizes across topology families |
| Parallelism | Naturally parallel | Supported via PPO's batch rollouts |

---

## Open Issues

### 1. Reward Sparsity (Critical)
Full accuracy from SuperNeuroMAT is only available at episode termination. Partially built topologies give no accuracy signal, making policy gradient updates noisy and slow to converge. Options:
- **Surrogate model**: Train a GNN to predict accuracy from graph structure using completed evaluations. Use as intermediate proxy reward.
- **Shaped structural rewards**: Cheap graph-theoretic proxies (reachability, depth, fan-out balance) computed without simulation.
- **Curriculum**: Start with small topologies (5–10 hidden neurons) where SuperNeuroMAT is fast, scale up as policy matures.

### 2. Evaluation Cost
Each SuperNeuroMAT call requires a full spiking simulation over N-MNIST. Need to estimate upfront how many evaluations the training budget allows and design reward shaping accordingly.

### 3. Action Space Scale
With source nodes equal to the full pixel count, the initial graph is large and `a_first`/`a_second` have very large action spaces. Consider:

- Grouping pixels into receptive field patches as source nodes
- Masking action logits to only locally reachable nodes early in training

### 4. Temporal Dynamics in N-MNIST
N-MNIST is event-based — spike timing matters, not just firing rate. The reward signal implicitly captures this, but the policy has no explicit representation of temporal computation. Encoding time constants and delays as rich node/edge features would help the GCN reason about temporal processing capacity.

---

## Suggested First Steps

1. Implement the SuperNeuroMAT wrapper and benchmark evaluation speed
2. Define node/edge feature schema (types, thresholds, time constants, delays)
3. Implement DAG validity checks and reachability checks in the environment
4. Train with shaped structural rewards first, introduce accuracy reward via curriculum
5. Benchmark against an evolutionary search baseline on N-MNIST at matched evaluation budgets

---

## References

- You et al., "Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation", NeurIPS 2018
- SuperNeuroMAT: https://github.com/ORNL/superneuromat
- N-MNIST Dataset: Orchard et al., 2015
