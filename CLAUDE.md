# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SpikeGCPN replaces evolutionary topology search for spiking neural networks with a two-level search:

- **Outer loop:** a learned policy builds an SNN with macro actions (`env/snn_env.py`), in roughly 20–60 decisions rather than one per synapse:
  - **NEW** creates a hidden neuron. Its receptive field is a (2r+1)² window of patch cells around a chosen centre, covering one or both polarities, with one shared weight. Its threshold and leak come from a chosen preset.
  - **CONNECT** adds one synapse from a hidden neuron to another hidden neuron or a sink.

  The policy is trained with GRPO-style critic-free policy gradient (`train/grpo.py`). `--algo random` is the random-topology baseline. A trajectory-balance GFlowNet was tried and removed: with log P_F in the hundreds per trajectory it never learned (no better than random).
- **Inner loop:** each sampled topology is scored by validation accuracy, with the scoring set by `inner.mode`:
  - `reservoir` (`train/readout.py`): the network keeps its fixed weights and may be recurrent (`env.allow_cycles`). A logistic-regression readout is fit on spike counts in W time windows. A ridge readout is also available, but it is sensitive to λ and barely faster. `inner.readout_neurons` picks what the readout sees:
    - `all` (`configs/reservoir.yaml`): every hidden and sink neuron. Hidden neurons with receptive fields can relay the input straight to the readout, and on N-MNIST the search converged to exactly that.
    - `outputs` (`configs/latent.yaml`, the current main line): only the sinks, which then act as `num_sink` output neurons independent of the 10 classes. Sinks have no input of their own, so everything the readout gets has flowed through the searched graph. `rf_radii` includes -1, which makes NEW create a latent neuron with no receptive field, driven only by CONNECT. Ablations: `env.allow_cycles=false` (depth without recurrence) and `env.hidden_to_hidden=false` (receptive-field neurons → outputs only).
  - `control` (`control/evaluator.py`, `configs/cartpole.yaml`): CartPole with the velocities hidden. The masked observation (x, θ) is population-coded onto the sources, the reservoir runs stepwise with state carried across env steps (`LIFBatch.step`), a logistic readout is behaviour-cloned from a full-state expert on precomputed DART-noise rollouts, and the reward is the closed-loop return / 500 over fixed-seed episodes. A memoryless readout on the code scores ~42 and one step of input memory scores ~500, so the reward measures whether the reservoir supplies memory. `control/cartpole.py` is a vectorised copy of gymnasium's dynamics (tested step for step).
  - `bptt` (`train/inner.py`, `configs/default.yaml`): surrogate-gradient BPTT trains synapse magnitudes, thresholds and leaks of a DAG, and the readout is sink spike counts. This mode underperformed: about 0.35 validation accuracy for 300 random synapses, versus 0.73 with a reservoir readout.

An optional GNN-ensemble surrogate (`model/surrogate.py`) decides which topologies are worth an inner-loop evaluation. SuperNeuroMAT (ORNL, a git submodule) is the reference simulator. `idea.md` is the original design doc and still describes GCPN+PPO with fixed weights, both since replaced.

## Setup and commands

Use the project venv. The base conda env on this machine is an x86_64 build running under Rosetta, and its NumPy-2 / numba / torch builds are broken there (torch↔numpy interop fails).

```bash
git submodule update --init
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q                                  # all tests (~3 s; testpaths excludes the submodule's tests)
.venv/bin/python -m pytest -q tests/test_core.py -k encode     # single test
.venv/bin/python scripts/train.py --config configs/smoke.yaml   # ~5 s end-to-end
.venv/bin/python scripts/train.py --algo grpo|random [--no-surrogate] [--no-inner] [--seed N]
.venv/bin/python scripts/compare.py launch --tag NAME --seeds 0 1 [--config configs/reservoir.yaml] [--variants random:nosur grpo:nosur ...]
.venv/bin/python scripts/compare.py launch --tag NAME --seeds 0 1 --config configs/latent.yaml --variants grpo:nosur --suffix dag --set env.allow_cycles=false   # variant under the same tag
.venv/bin/python scripts/compare.py report --tag NAME
.venv/bin/python scripts/topology_evolution.py --tag NAME      # how sampled topologies changed over training → experiments/NAME/topology/
.venv/bin/python scripts/probe_reservoirs.py --tag NAME        # hand-designed reservoir ablations + input baselines (readout_neurons: all only)
.venv/bin/python scripts/draw_topologies.py --tag NAME         # the sampled graphs drawn over the patch grid through training, + digit tuning
```

N-MNIST must already be extracted at `data/NMNIST/{Train,Test}/{0..9}/*.bin`, and SHD at `data/SHD/shd_{train,test}.h5` (from https://zenkelab.org/datasets/); `data/` is gitignored. `simulator.dataset` (nmnist | shd) selects the data, and `data/datasets.load_splits(cfg)` is the one loader shared by training and the analysis scripts. Each run writes `metrics.csv`, `summary.yaml`, `best_graph.pt` (with trained parameters), `ckpt.pt` and `samples.pt` (every sampled trajectory's actions and accuracy per iteration) to `experiments/<run>/`. Keep `threads: 1`: the per-step tensors are small, and multithreading makes training about 3× slower.

## Architecture

Pipeline:

`policy samples trajectory (rollout.py)` → `RewardOracle (oracle.py)` decides cache / simulator / surrogate per graph → `GRPOTrainer` update

Invariants that span several files:

- **Node layout is fixed and append-only.** Sources occupy `[0, S)`, sinks `[S, S+K)`, and hidden neurons `[S+K, …)` in creation order.
  - `env.input_layout` sets how sources are arranged. `grid2d` (N-MNIST): polarity-major over a G×G patch grid, so `S = 2·G·G`. `line1d`: `line_groups` × C channels on a line, group-major: SHD's 140 pooled frequency channels (1 group), or CartPole's x and θ population codes (2 groups). Receptive fields are windows of 2r+1 adjacent channels; `GraphSpec.num_cells`, `num_polarities` and `num_pol_choices` abstract over the layouts.
  - Nodes and edges are only ever appended, so the state before action `t` is a prefix of the final graph. `Trajectory` stores the hidden and edge counts at each step, and `prefix_graph` rebuilds any state from them.
  - `simulator/wrapper.py`, `validity.py`, the encoder and `PolicyHead`'s cell pooling all rely on this layout.
- **Masks are exact, and illegal actions raise.**
  - `snn_env.masks` covers stop, continue, NEW and CONNECT. `GraphSpec.rf_edge_table()` masks receptive fields that don't fit the remaining edge budget.
  - `validity.connect_first_mask` and `second_mask` rule out duplicates, self-loops, edges into sources or out of sinks, and cycles (unless `allow_cycles` is set).
  - Because masks are exact, `log π(a|s)` is the log-probability of the real transition, which GRPO's importance ratios rely on. Don't reintroduce no-op "invalid" steps.
- **Policy evaluation is batched.** `PolicyHead.forward_batch` evaluates every decision stage for B graph states encoded as one disjoint union (`policy.collate`). Candidate sets of different sizes are padded with masked logits.
  - `rollout.sample_trajectories` advances all episodes of a group together.
  - `rollout.rescore_batch` scores every prefix state of several trajectories in one call, using the masks stored at sampling time (`Trajectory.masks` and `second_masks`).
  - A test checks that re-scored log-probs match the sampling ones, so change both paths together.
  - Batching made each iteration about 15× faster. Before it, the policy took 90% of wall-clock time.
  - `GraphEncoder.encode_batch` computes the embeddings of sources with no synapses once and gathers them (they get no messages, so they're the same in every state); a test checks it equals `forward`. CONNECT tails and heads are scored only over non-source nodes.
  - `LIFBatch.step(state, x)` advances one tick with per-graph inputs `[B, G, S]` (each graph drives its own environment). Stepping reproduces `forward` exactly (tested), and both share `_tick`, so change the dynamics in one place.
  - `grpo.step_frac < 1` re-scores a random subset of each trajectory's steps per minibatch (unbiased, since all steps share the trajectory's advantage); `configs/reservoir.yaml` uses 0.5.
- **Torch simulator ↔ SuperNeuroMAT equivalence.** `simulator/torch_lif.py` reproduces SuperNeuroMAT's update rule (linear leak toward reset, strict `>` threshold, hard reset, refractory counter, delay-1 synapses). In float64 its sink spike counts are bit-identical to SuperNeuroMAT's, which `tests/test_inner.py` checks. Two consequences:
  - Rewards use the torch forward pass.
  - The final best graph is re-verified in SuperNeuroMAT (`summary.yaml`: `best_val_acc_superneuromat`).

  Any change to neuron dynamics must keep that test passing. Sources are not simulated, because a source's spike train equals its input train; `LIFBatch` refuses source configurations where that isn't true.
- **Reservoir mode.**
  - `LIFBatch(fixed=True)` uses the graph's weights, thresholds and leaks exactly as stored, with no softplus and no clamping. Its windowed spike counts are therefore bit-identical to `wrapper.window_spike_counts` in SuperNeuroMAT, including on recurrent graphs with integer weights where threshold ties are common. `tests/test_reservoir.py` checks this.
  - Recurrent graphs have no longest path, so every reservoir is simulated for `num_bins + inner.extra_ticks` steps.
  - `summary.yaml` reports input-readout baselines with as many channels as the readout has neurons: `input_readout_matched` uses random channels and is weak; `input_readout_top` uses the channels with the highest ANOVA F-statistic on the training split, and is the fair one to beat. With `readout_neurons: all`, a readout on the most informative cells beat the searched reservoirs on N-MNIST Test (`scripts/probe_reservoirs.py`).
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
- **`num_sink` vs classes.** In BPTT mode sinks are the class neurons, so `num_sink` must be 10. In reservoir mode they are just read-out neurons; class counts come from `data.nmnist.NUM_CLASSES`, so `ReservoirEvaluator` and the baselines take `num_classes` separately.
- **Action fields.**
  - `a_edge` is a signed weight bin (`weight_levels` × ±). With BPTT, set `weight_levels: [1.0]`, so the policy picks only the sign.
  - `a_node` picks a `hidden_presets` entry, which sets the neuron's initial (threshold, leak).
  - `min_edges` masks the stop action until that many edges exist. In reservoir mode it is 0, so the 300-synapse budget is a cap, not a requirement.

  All of these live in `GraphSpec` and the `env:` config section.
- **Inputs are pooled into patches.** With `patch_size: 4`, each 4×4 pixel block per polarity becomes one source (162 sources). The source fires in a time bin if any pixel in its block did. `num_source` is derived from `patch_size`, and source position features come from the patch centres.
- **SuperNeuroMAT leak semantics.** `leak` is a linear decay toward the reset state applied every step. `leak=0` gives a perfect integrator. `leak=inf` makes the neuron memoryless: its state is wiped every step.
- **Rewards.** `rewards.shaped_reward = accuracy − efficiency_coeff·(hidden + edges)`. GRPO normalizes rewards within each group.
  - The Train split provides two disjoint per-class subsets: `inner.train_per_class` samples for inner training and `simulator.val_per_class` for the reward.
  - The Test split is used only to report the best graph in `summary.yaml`.
- **The oracle and surrogate.**
  - Results are cached by `rollout.graph_key`.
  - Until `warmup` graphs have been simulated, every graph goes to the simulator.
  - After warm-up, only the `sim_per_batch` graphs with the highest UCB (ensemble mean + κ·std) are simulated; the rest are scored by the surrogate mean.
  - `sim_calls` counts topology evaluations (inner training runs), and it is the budget compared across algorithms.
- **Simulator speed.** `wrapper.evaluate_samples` skips sources with no outgoing edge and writes `snn.input_spikes` directly. A test asserts this gives the same accuracy as naive simulation of the full graph. Readout is the spike count at each sink over `num_bins + longest_path` steps. Ties earn fractional credit.
