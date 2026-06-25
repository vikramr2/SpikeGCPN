"""
SuperNeuroMAT wrapper: imprint a PyG Data graph onto an SNN and evaluate
it on the N-MNIST test (or validation) split.

Graph conventions (must be satisfied before calling these functions):
  - data.x        : float tensor [num_nodes, 4]
                    columns: [threshold, leak, reset_state, refractory_period]
  - data.edge_index : long tensor [2, num_edges], COO directed format
  - data.edge_attr  : float tensor [num_edges, 2]
                    columns: [weight, delay]
                    weight > 0 → excitatory, weight < 0 → inhibitory
  - Source nodes  : indices [0, num_source)          — fixed, pre-instantiated
  - Sink nodes    : indices [num_nodes-num_sink, num_nodes) — fixed, 10 nodes
  - Graph must be a DAG (enforced by the environment; checked here too)

Classification head: total spike count at each sink neuron over the full
simulation window; argmax gives the predicted class.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from superneuromat import SNN

from spikegcpn.data.nmnist import NUM_SOURCE_NODES, NUM_CLASSES, iter_dataset
from spikegcpn.env.validity import is_dag, all_sinks_reachable

# Default node feature values used when data.x is None.
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_LEAK = float("inf")
_DEFAULT_RESET = 0.0
_DEFAULT_REFRAC = 0


def _validate(data: Data, num_source: int, num_sink: int) -> None:
    """Raise ValueError if the graph violates any structural constraint."""
    n = data.num_nodes
    if n < num_source + num_sink:
        raise ValueError(
            f"Graph has {n} nodes but needs at least "
            f"{num_source + num_sink} (source + sink)."
        )
    if not is_dag(data.edge_index, n):
        raise ValueError("Graph is not a DAG.")
    if data.edge_index.numel() > 0 and not all_sinks_reachable(
        data.edge_index, n, num_source, num_sink
    ):
        raise ValueError("Not all sink nodes are reachable from source nodes.")


def imprint(
    data: Data,
    num_source: int = NUM_SOURCE_NODES,
    num_sink: int = NUM_CLASSES,
) -> tuple[SNN, list, list]:
    """
    Build a fresh SuperNeuroMAT SNN from a PyG Data object.

    Returns
    -------
    snn         : SNN instance with neurons and synapses but no input spikes yet
    src_neurons : list of Neuron objects for the num_source source nodes
    sink_neurons: list of Neuron objects for the num_sink sink nodes
    """
    _validate(data, num_source, num_sink)

    snn = SNN()
    n = data.num_nodes
    x = data.x  # may be None

    # --- Create neurons in node-index order -----------------------------------
    neurons: list = []
    for i in range(n):
        if x is not None:
            threshold = float(x[i, 0])
            leak = float(x[i, 1])
            reset_state = float(x[i, 2])
            refractory_period = int(x[i, 3])
        else:
            threshold = _DEFAULT_THRESHOLD
            leak = _DEFAULT_LEAK
            reset_state = _DEFAULT_RESET
            refractory_period = _DEFAULT_REFRAC

        neuron = snn.create_neuron(
            threshold=threshold,
            leak=leak,
            reset_state=reset_state,
            refractory_period=refractory_period,
        )
        neurons.append(neuron)

    # --- Create synapses ------------------------------------------------------
    if data.edge_index.numel() > 0:
        src_idx = data.edge_index[0].tolist()
        dst_idx = data.edge_index[1].tolist()
        edge_attr = data.edge_attr

        for k, (s, d) in enumerate(zip(src_idx, dst_idx)):
            if edge_attr is not None:
                weight = float(edge_attr[k, 0])
                delay = max(1, int(round(float(edge_attr[k, 1]))))
            else:
                weight = 1.0
                delay = 1
            snn.create_synapse(neurons[s], neurons[d], weight=weight, delay=delay)

    src_neurons = neurons[:num_source]
    sink_neurons = neurons[n - num_sink:]
    return snn, src_neurons, sink_neurons


def evaluate(
    data: Data,
    nmnist_root: str,
    num_source: int = NUM_SOURCE_NODES,
    num_sink: int = NUM_CLASSES,
    split: str = "Test",
    num_bins: int = 100,
    max_samples: int | None = None,
) -> float:
    """
    Evaluate a topology graph on the N-MNIST test split.

    The SNN structure is built once from `data`; for each sample the network
    state is reset, input spike trains are fed to the source neurons, the
    simulation runs for `num_bins` steps, and the predicted class is the
    sink neuron with the highest total spike count.

    Parameters
    ----------
    data          : PyG Data object describing the SNN topology
    nmnist_root   : path to the NMNIST directory (contains Train/ Test/)
    num_source    : number of source nodes in the graph (default: 2312)
    num_sink      : number of sink/output nodes (default: 10)
    split         : "Test" or "Train"
    num_bins      : temporal resolution for spike encoding
    max_samples   : cap evaluation at this many samples (None = all)

    Returns
    -------
    accuracy : float in [0, 1]
    """
    snn, src_neurons, sink_neurons = imprint(data, num_source, num_sink)

    # Pre-compute sink column indices in snn.ispikes (shape: time × all_neurons).
    # We use neuron.idx rather than our node index because delay-chain synapses
    # may insert internal hidden neurons that shift the column layout.
    sink_ids = [n.idx for n in sink_neurons]

    correct = 0
    total = 0

    for sample in iter_dataset(nmnist_root, split=split, num_bins=num_bins, max_samples=max_samples):
        # Reset all neuron states, spike history, and queued input spikes.
        snn.reset()

        # Feed spike train to source neurons.
        # spike_train[t, node_idx] == True means that source node fired in bin t.
        spike_train = sample.spike_train  # (num_bins, num_source)
        for t in range(num_bins):
            active = np.where(spike_train[t])[0]
            for node_idx in active:
                snn.add_spike(
                    time=t,
                    neuron_id=src_neurons[node_idx],
                    value=1.0,
                    exist="dontadd",  # ignore duplicate spikes in same bin
                )

        snn.simulate(time_steps=num_bins)

        # ispikes shape: (num_bins, total_neurons_including_delay_chains)
        spike_counts = np.array(
            [int(snn.ispikes[:, sid].sum()) for sid in sink_ids],
            dtype=np.int64,
        )
        predicted = int(np.argmax(spike_counts))

        if predicted == sample.label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0
