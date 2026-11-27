"""
SuperNeuroMAT wrapper: imprint a topology graph onto an SNN and measure
N-MNIST accuracy.

Graph conventions (see env/snn_env.py):
  - x[:, :4]   : [threshold, leak, reset_state, refractory_period]
  - edge_attr  : [weight, delay]; weight sign = excitatory / inhibitory
  - sources    : [0, S); sinks: [S, S+K); hidden: [S+K, n)

Speed-ups that do not change the result:
  - Source nodes with no outgoing edge cannot influence any sink, so they are
    not instantiated at all (typically ~2300 of 2312 sources are isolated).
  - Input spikes are written straight into `snn.input_spikes` instead of one
    `add_spike` call each (`add_spike` does a linear duplicate scan per call).

Readout: total spike count per sink over num_bins + longest-path steps (so
spikes from the last bins can still reach the sinks). Ties between the top
sinks give fractional credit 1/#tied, i.e. expected accuracy under random
tie-breaking — the old argmax always picked sink 0, so an empty graph scored
exactly the class-0 frequency.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch_geometric.data import Data

from superneuromat import SNN

from spikegcpn.data.nmnist import NUM_CLASSES, NUM_SOURCE_NODES, NMNISTSample, load_samples
from spikegcpn.env.validity import is_dag, longest_path_length


def imprint(
    data: Data,
    num_source: int = NUM_SOURCE_NODES,
    num_sink: int = NUM_CLASSES,
) -> tuple[SNN, np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Build a SuperNeuroMAT SNN from a topology graph (DAG or recurrent).

    Returns
    -------
    snn              : the network (no input spikes queued)
    kept_sources     : graph indices of the sources that were instantiated
    kept_source_ids  : SNN neuron ids for those sources (aligned with kept_sources)
    sink_ids         : SNN neuron ids of the K sinks, in class order
    nonsource_ids    : SNN neuron ids of every non-source node, in graph order
    """
    n = data.num_nodes
    if n < num_source + num_sink:
        raise ValueError(f"graph has {n} nodes, needs ≥ {num_source + num_sink}")

    ei = data.edge_index
    used_src = torch.zeros(num_source, dtype=torch.bool)
    if ei.numel() > 0:
        tails = ei[0][ei[0] < num_source]
        used_src[tails] = True
    kept_sources = used_src.nonzero(as_tuple=True)[0].numpy()
    keep = np.concatenate([kept_sources, np.arange(num_source, n)])

    snn = SNN()
    x = data.x
    neuron_of: dict[int, int] = {}
    for i in keep.tolist():
        neuron = snn.create_neuron(
            threshold=float(x[i, 0]),
            leak=float(x[i, 1]),
            reset_state=float(x[i, 2]),
            refractory_period=int(x[i, 3]),
        )
        neuron_of[i] = neuron.idx

    if ei.numel() > 0:
        for (s, d), (w, delay) in zip(ei.t().tolist(), data.edge_attr.tolist()):
            snn.create_synapse(neuron_of[s], neuron_of[d], weight=float(w), delay=max(1, int(round(delay))))

    kept_ids = np.array([neuron_of[i] for i in kept_sources.tolist()], dtype=np.int64)
    sink_ids = [neuron_of[i] for i in range(num_source, num_source + num_sink)]
    nonsource_ids = [neuron_of[i] for i in range(num_source, n)]
    return snn, kept_sources, kept_ids, sink_ids, nonsource_ids


def _simulate_each(data, samples, num_source, num_sink, extra_ticks):
    """Yield (sample index, ispikes array, ids) after simulating each sample."""
    snn, kept_sources, kept_ids, sink_ids, ns_ids = imprint(data, num_source, num_sink)
    if extra_ticks is None:
        if not is_dag(data.edge_index, data.num_nodes):
            raise ValueError("recurrent graph: pass a fixed extra_ticks")
        extra_ticks = longest_path_length(data.edge_index, data.num_nodes)
    for n, sample in enumerate(samples):
        snn.reset()
        train = sample.spike_train[:, kept_sources]          # (T, kept)
        T = train.shape[0]
        spikes = {}
        for t in range(T):
            cols = np.flatnonzero(train[t])
            if cols.size:
                spikes[t] = {"nids": kept_ids[cols].tolist(), "values": [1.0] * cols.size}
        snn.input_spikes = spikes
        snn.simulate(time_steps=T + extra_ticks, use="cpu")
        yield n, snn.ispikes, sink_ids, ns_ids


def sink_spike_counts(
    data: Data,
    samples: Iterable[NMNISTSample],
    num_source: int = NUM_SOURCE_NODES,
    num_sink: int = NUM_CLASSES,
    extra_ticks: int | None = None,
) -> np.ndarray:
    """
    [num_samples, num_sink] total sink spike counts over num_bins + extra_ticks
    steps (extra_ticks=None: the longest path, DAGs only).
    """
    samples = list(samples)
    counts = np.zeros((len(samples), num_sink), dtype=np.int64)
    for n, isp, sink_ids, _ in _simulate_each(data, samples, num_source, num_sink, extra_ticks):
        counts[n] = isp[:, sink_ids].sum(axis=0)
    return counts


def window_spike_counts(
    data: Data,
    samples: Iterable[NMNISTSample],
    num_source: int,
    num_sink: int,
    extra_ticks: int,
    windows: int,
) -> np.ndarray:
    """
    Reservoir features: [num_samples, num_nonsource, windows] spike counts of every
    sink + hidden neuron in `windows` equal time windows over num_bins + extra_ticks.
    """
    samples = list(samples)
    M = data.num_nodes - num_source
    out = np.zeros((len(samples), M, windows), dtype=np.int64)
    for n, isp, _, ns_ids in _simulate_each(data, samples, num_source, num_sink, extra_ticks):
        sp = isp[:, ns_ids]                                   # [ticks, M]
        ticks = sp.shape[0]
        widx = np.arange(ticks) * windows // ticks
        for w in range(windows):
            out[n, :, w] = sp[widx == w].sum(axis=0)
    return out


def tie_credit_accuracy(counts: np.ndarray, labels: np.ndarray) -> float:
    """Mean over samples of 1/#tied if the label is among the top-count sinks, else 0."""
    if len(labels) == 0:
        return 0.0
    winners = counts == counts.max(axis=1, keepdims=True)
    credit = winners[np.arange(len(labels)), labels] / winners.sum(axis=1)
    return float(credit.mean())


def evaluate_samples(
    data: Data,
    samples: Iterable[NMNISTSample],
    num_source: int = NUM_SOURCE_NODES,
    num_sink: int = NUM_CLASSES,
) -> float:
    """Accuracy of the topology on the given samples (fractional credit on ties)."""
    samples = list(samples)
    counts = sink_spike_counts(data, samples, num_source, num_sink)
    return tie_credit_accuracy(counts, np.array([s.label for s in samples], dtype=np.int64))


def evaluate(
    data: Data,
    nmnist_root: str,
    num_source: int = NUM_SOURCE_NODES,
    num_sink: int = NUM_CLASSES,
    split: str = "Test",
    num_bins: int = 100,
    per_class: int | None = None,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
    patch: int = 1,
) -> float:
    """Accuracy on a (cached) N-MNIST subset."""
    samples = load_samples(nmnist_root, split, num_bins, per_class, max_samples, shuffle, seed, patch)
    return evaluate_samples(data, samples, num_source, num_sink)
