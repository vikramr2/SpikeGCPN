"""Recurrent topologies (allow_cycles) and reservoir evaluation."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from spikegcpn.data.nmnist import NMNISTSample
from spikegcpn.env.snn_env import GraphSpec
from spikegcpn.env.validity import is_dag
from spikegcpn.simulator.torch_lif import LIFBatch
from spikegcpn.simulator.wrapper import sink_spike_counts, window_spike_counts
from spikegcpn.train.readout import ReadoutConfig, ReservoirEvaluator, input_readout_baselines

from test_core import random_state

CYC = GraphSpec(num_source=18, num_sink=3, max_hidden=6, max_edges=30, allow_cycles=True,
                rf_radii=[0, 1], weight_levels=[0.5, 1.0, 2.0])
EXTRA = 6


def _cyclic_graphs(n: int, steps: int = 30):
    return [random_state(CYC, random.Random(seed), steps=steps).graph for seed in range(n)]


def test_cycles_actually_occur():
    assert any(not is_dag(g.edge_index, g.num_nodes) for g in _cyclic_graphs(20))


def test_fixed_reservoir_matches_superneuromat():
    graphs = _cyclic_graphs(8)
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.random((6, 25, CYC.num_source)) < 0.3)
    samples = [NMNISTSample(spike_train=X[i].numpy(), label=0) for i in range(len(X))]
    net = LIFBatch(graphs, CYC.num_source, CYC.num_sink, extra_ticks=EXTRA, fixed=True)
    with torch.no_grad():
        counts, win = net(X, windows=4)
    total = 0
    for gi, (g, sl) in enumerate(zip(graphs, net.ns_slices)):
        np.testing.assert_array_equal(
            sink_spike_counts(g, samples, CYC.num_source, CYC.num_sink, EXTRA), counts[:, gi].numpy().astype(int))
        snm = window_spike_counts(g, samples, CYC.num_source, CYC.num_sink, EXTRA, 4)
        np.testing.assert_array_equal(snm, win[:, sl].numpy().astype(int))
        total += int(snm.sum())
    assert total > 0


def _evaluator():
    rng = np.random.default_rng(1)
    tr = (torch.from_numpy(rng.random((60, 20, CYC.num_source)) < 0.3), torch.from_numpy(rng.integers(0, 3, 60)))
    va = (torch.from_numpy(rng.random((30, 20, CYC.num_source)) < 0.3), torch.from_numpy(rng.integers(0, 3, 30)))
    return ReservoirEvaluator(CYC.num_source, CYC.num_sink, tr, va, ReadoutConfig(windows=3, max_iter=50), EXTRA), tr, va


def test_reservoir_eval_deterministic_and_batch_independent():
    graphs = _cyclic_graphs(4)
    ev, _, _ = _evaluator()
    accs_b, _ = ev.train_and_eval(graphs)
    assert accs_b == ev.train_and_eval(graphs)[0]
    for g, a in zip(graphs, accs_b):
        assert ev.train_and_eval([g])[0][0] == pytest.approx(a, abs=1e-12)


def test_input_baselines_run():
    ev, tr, va = _evaluator()
    out = input_readout_baselines(tr, va, 3, ReadoutConfig(windows=3, max_iter=50), matched_neurons=5)
    assert 0 <= out["input_readout_all"] <= 1 and out["matched_channels"] == 5
