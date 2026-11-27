# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpikeGCPN replaces evolutionary topology search for spiking neural networks with a two-level search:

- **Outer loop:** a learned policy builds an SNN with macro actions (`env/snn_env.py`), in roughly 20–60 decisions rather than one per synapse:
  - **NEW** creates a hidden neuron. Its receptive field is a (2r+1)² window of patch cells around a chosen centre, covering one or both polarities, with one shared weight. Its threshold and leak come from a chosen preset.
  - **CONNECT** adds one synapse from a hidden neuron to another hidden neuron or a sink. The policy is trained by either:
  - GRPO-style critic-free policy gradient (`train/grpo.py`), or
  - a GFlowNet with trajectory balance (`train/gflownet.py`).

  `--algo random` is the random-topology baseline.
- **Inner loop:** each sampled topology is scored by validation accuracy, with the scoring set by `inner.mode`:
  - `reservoir` (`train/readout.py`, `configs/reservoir.yaml`, the current main line): the network keeps its fixed weights and may be recurrent (`env.allow_cycles`). A logistic-regression readout is fit on every hidden and sink neuron's spike counts in W time windows. A ridge readout is also available, but it is sensitive to λ and barely faster.
  - `bptt` (`train/inner.py`, `configs/default.yaml`): surrogate-gradient BPTT trains synapse magnitudes, thresholds and leaks of a DAG, and the readout is sink spike counts. This mode underperformed: about 0.35 validation accuracy for 300 random synapses, versus 0.73 with a reservoir readout.

An optional GNN-ensemble surrogate (`model/surrogate.py`) decides which topologies are worth an inner-loop evaluation. SuperNeuroMAT (ORNL, a git submodule) is the reference simulator. `idea.md` is the original design doc and still describes GCPN+PPO with fixed weights, both since replaced.

## Setup and commands

Use the project venv. The base conda env on this machine is an x86_64 build running under Rosetta, and its NumPy-2 / numba / torch builds are broken there (torch↔numpy interop fails).

```bash
git submodule update --init
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q                                  # all tests (~2 s)
.venv/bin/python -m pytest -q tests/test_core.py -k parents    # single test
.venv/bin/python scripts/train.py --config configs/smoke.yaml --algo gflownet   # ~5 s end-to-end
.venv/bin/python scripts/train.py --algo grpo|gflownet|random [--no-surrogate] [--no-inner] [--seed N]
.venv/bin/python scripts/compare.py launch --tag NAME --seeds 0 1 [--config configs/reservoir.yaml] [--variants random:nosur grpo:nosur ...]
.venv/bin/python scripts/compare.py report --tag NAME
```

N-MNIST must already be extracted at `data/NMNIST/{Train,Test}/{0..9}/*.bin`; the dataset is gitignored. Each run writes `metrics.csv`, `summary.yaml`, `best_graph.pt` (with trained parameters) and `ckpt.pt` to `experiments/<run>/`. Keep `threads: 1`: the per-step tensors are small, and multithreading makes training about 3× slower.

## Architecture

Pipeline:

`policy samples trajectory (rollout.py)` → `RewardOracle (oracle.py)` decides cache / simulator / surrogate per graph → `GRPOTrainer` or `TBTrainer` update

Invariants that span several files:

- **Node layout is fixed and append-only.** Sources occupy `[0, S)`, sinks `[S, S+K)`, and hidden neurons `[S+K, …)` in creation order.
  - Sources are polarity-major over a G×G patch grid, so `S = 2·G·G` (`GraphSpec` enforces this).
  - Nodes and edges are only ever appended, so the state before action `t` is a prefix of the final graph. `Trajectory` stores the hidden and edge counts at each step, and `prefix_graph` rebuilds any state from them.
  - `simulator/wrapper.py`, `validity.py`, the encoder and `PolicyHead`'s cell pooling all rely on this layout.
- **Masks are exact, and illegal actions raise.**
  - `snn_env.masks` covers stop, continue, NEW and CONNECT. `GraphSpec.rf_edge_table()` masks receptive fields that don't fit the remaining edge budget.
  - `validity.connect_first_mask` and `second_mask` rule out duplicates, self-loops, edges into sources or out of sinks, and cycles (unless `allow_cycles` is set).
  - Because masks are exact, `log π(a|s)` is the log-probability of the real transition, which the trajectory-balance loss requires. Don't reintroduce no-op "invalid" steps.
- **Policy evaluation is batched.** `PolicyHead.forward_batch` evaluates every decision stage for B graph states encoded as one disjoint union (`policy.collate`). Candidate sets of different sizes are padded with masked logits.
  - `rollout.sample_trajectories` advances all episodes of a group together.
  - `rollout.rescore_batch` scores every prefix state of several trajectories in one call, using the masks stored at sampling time (`Trajectory.masks` and `second_masks`).
  - A test checks that re-scored log-probs match the sampling ones, so change both paths together.
  - Batching made each iteration about 15× faster. Before it, the policy took 90% of wall-clock time.
- **GFlowNet backward policy.** P_B is uniform over (parent, action) pairs, counted by `validity.num_parents(edge_index, spec, H)`. A synapse with a hidden tail can only come from CONNECT; synapses with a source tail come only from the NEW that created their neuron. So the parents of a state are:
  - undo any one hidden-tail synapse, or
  - undo the most recent neuron, if it has no out-synapses and no hidden-tail in-synapses. Count this once for every (cell, radius, polarity) choice that clips to the same receptive field (`GraphSpec.rf_multiplicity`).

  `tests/test_core.py` checks this against a brute force through `env.step`. If you change action semantics, rerun that test.
- **Torch simulator ↔ SuperNeuroMAT equivalence.** `simulator/torch_lif.py` reproduces SuperNeuroMAT's update rule (linear leak toward reset, strict `>` threshold, hard reset, refractory counter, delay-1 synapses). In float64 its sink spike counts are bit-identical to SuperNeuroMAT's, which `tests/test_inner.py` checks. Two consequences:
  - Rewards use the torch forward pass.
  - The final best graph is re-verified in SuperNeuroMAT (`summary.yaml`: `best_val_acc_superneuromat`).

  Any change to neuron dynamics must keep that test passing. Sources are not simulated, because a source's spike train equals its input train; `LIFBatch` refuses source configurations where that isn't true.
- **Reservoir mode.**
  - `LIFBatch(fixed=True)` uses the graph's weights, thresholds and leaks exactly as stored, with no softplus and no clamping. Its windowed spike counts are therefore bit-identical to `wrapper.window_spike_counts` in SuperNeuroMAT, including on recurrent graphs with integer weights where threshold ties are common. `tests/test_reservoir.py` checks this.
  - Recurrent graphs have no longest path, so every reservoir is simulated for `num_bins + inner.extra_ticks` steps.
  - `summary.yaml` reports the input-readout baselines. A reservoir only adds value if it beats `input_readout_matched`, a readout on the same number of random input channels as the largest allowed reservoir.
  - The readout runs off-chip.
- **Inner training is batched but independent.** `LIFBatch` trains a group of topologies as one disconnected network: messages pass along an edge list, and no edges cross graphs. Adam is elementwise and the summed loss separates per graph, so each graph ends up with the same parameters it would get if trained alone. A test checks this. Together with the fixed `inner.seed` and `simulator.data_seed`, this makes the reward a deterministic function of the topology, which the oracle's cache relies on. Do not add global gradient clipping or batch-coupled terms to the inner loss.
- **Inner-loop parameterization.**
  - Weights are `sign · softplus(raw)`, so signs never flip (Dale's law).
  - Thresholds and leaks of sinks and hidden neurons are `softplus` of raw parameters. Presets with `leak=inf` start at `leak_cap` and stay effectively memoryless.
  - `export()` writes the trained parameters back into float64 graphs.
- **Feature schemas.** Node `x` has 9 columns: `[threshold, leak, reset, refractory, node_type, pos_x, pos_y, polarity, sink_id]`. `node_type` is SOURCE=0, SINK=1, HIDDEN=2. Edge `edge_attr` is `[weight, delay]`, with the sign of `weight` marking excitatory vs inhibitory. The simulator reads only `x[:, :4]`.
- **The encoder** (`model/encoder.py`):
  - passes messages in both directions, with separate weights for each (sign, direction) pair;
  - gives sources Fourier features of their pixel position and sinks a per-class embedding.

  Without these, all 2312 sources had identical embeddings and `a_first` was uniform over pixels.
- **Action fields.**
  - `a_edge` is a signed weight bin (`weight_levels` × ±). With BPTT, set `weight_levels: [1.0]`, so the policy picks only the sign.
  - `a_node` picks a `hidden_presets` entry, which sets the neuron's initial (threshold, leak).
  - `min_edges` masks the stop action until that many edges exist. In reservoir mode it is 0, so the 300-synapse budget is a cap, not a requirement.

  All of these live in `GraphSpec` and the `env:` config section.
- **Inputs are pooled into patches.** With `patch_size: 4`, each 4×4 pixel block per polarity becomes one source (162 sources). The source fires in a time bin if any pixel in its block did. `num_source` is derived from `patch_size`, and source position features come from the patch centres.
- **SuperNeuroMAT leak semantics.** `leak` is a linear decay toward the reset state applied every step. `leak=0` gives a perfect integrator. `leak=inf` makes the neuron memoryless: its state is wiped every step.
- **Rewards.** `rewards.shaped_reward = accuracy − efficiency_coeff·(hidden + edges)`. GRPO normalizes rewards within each group. The GFlowNet uses `log R = beta·reward`, and results are sensitive to `beta`.
  - The Train split provides two disjoint per-class subsets: `inner.train_per_class` samples for inner training and `simulator.val_per_class` for the reward.
  - The Test split is used only to report the best graph in `summary.yaml`.
- **The oracle and surrogate.**
  - Results are cached by `rollout.graph_key`.
  - Until `warmup` graphs have been simulated, every graph goes to the simulator.
  - After warm-up, only the `sim_per_batch` graphs with the highest UCB (ensemble mean + κ·std) are simulated; the rest are scored by the surrogate mean.
  - `sim_calls` counts topology evaluations (inner training runs), and it is the budget compared across algorithms.
- **Simulator speed.** `wrapper.evaluate_samples` skips sources with no outgoing edge and writes `snn.input_spikes` directly. A test asserts this gives the same accuracy as naive simulation of the full graph. Readout is the spike count at each sink over `num_bins + longest_path` steps. Ties earn fractional credit.
